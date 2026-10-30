from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.preprocessing.features import (
    OneEuroFilter,
    align_labels_to_target_fps,
    build_feature_matrix,
    correct_aspect_ratio,
    downsample_sequence,
    mirror_hand_landmarks,
    normalize_hand_landmarks,
    palm_size,
)
from utils.schema import (
    FEATURE_DIM,
    FEATURE_NAMES,
    SequenceRecord,
    load_sequence,
    load_sequence_header,
    save_sequence,
)


class SequenceHeaderTests(unittest.TestCase):
    def _record(self, labels: np.ndarray | None) -> SequenceRecord:
        frames = 24
        return SequenceRecord(
            features=np.random.default_rng(0)
            .standard_normal((FEATURE_DIM, frames))
            .astype(np.float32),
            labels=labels,
            valid_mask=np.ones(frames, dtype=np.float32),
            video_id="header",
            fps=15.0,
            subject_id="S9",
            source="data_demo",
            scene="sit",
            polarity="positive",
            metadata={"detection_rate": 0.75, "quality_passed": True},
        )

    def test_header_agrees_with_the_full_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "header.npz"
            save_sequence(path, self._record(np.arange(24, dtype=np.int64) % 3))

            full = load_sequence(path)
            header = load_sequence_header(path)

            self.assertEqual(header.video_id, full.video_id)
            self.assertEqual(header.num_frames, full.num_frames)
            self.assertEqual(header.fps, full.fps)
            self.assertEqual(header.subject_id, full.subject_id)
            self.assertEqual(header.source, full.source)
            self.assertEqual(header.scene, full.scene)
            self.assertEqual(header.polarity, full.polarity)
            self.assertEqual(header.metadata, full.metadata)
            np.testing.assert_array_equal(header.labels, full.labels)

    def test_header_handles_unlabeled_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "unlabeled.npz"
            save_sequence(path, self._record(None))

            header = load_sequence_header(path)

            self.assertIsNone(header.labels)
            self.assertEqual(header.num_frames, 24)

    def test_header_rejects_a_foreign_feature_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "stale.npz"
            save_sequence(path, self._record(None))
            payload = dict(np.load(path, allow_pickle=True))
            payload["feature_names"] = np.asarray(["bogus"])
            np.savez_compressed(path, **payload)

            with self.assertRaises(ValueError):
                load_sequence_header(path)


class FeatureSchemaTests(unittest.TestCase):
    def test_feature_dimension_is_128(self) -> None:
        self.assertEqual(FEATURE_DIM, 128)
        self.assertEqual(len(FEATURE_NAMES), 128)

    def test_normalize_hand_landmarks_centers_wrist(self) -> None:
        landmarks = np.zeros((21, 3), dtype=np.float32)
        landmarks[:, 0] = np.linspace(0.2, 0.8, 21)
        landmarks[9] = np.array([0.2, 0.4, 0.0], dtype=np.float32)
        normalized = normalize_hand_landmarks(landmarks).reshape(21, 3)
        np.testing.assert_allclose(normalized[0], 0.0, atol=1e-6)
        self.assertGreater(palm_size(landmarks), 0.0)

    def make_palm(self) -> np.ndarray:
        """A flat right hand in pixel-like units: MCPs fanned out above the wrist."""
        landmarks = np.zeros((21, 3), dtype=np.float32)
        landmarks[5] = [-0.3, -0.9, 0.0]
        landmarks[9] = [0.0, -1.0, 0.0]
        landmarks[13] = [0.25, -0.95, 0.0]
        landmarks[17] = [0.5, -0.8, 0.0]
        return landmarks

    def test_palm_size_averages_the_four_metacarpals(self) -> None:
        landmarks = self.make_palm()
        expected = np.mean(
            [np.linalg.norm(landmarks[index]) for index in (5, 9, 13, 17)]
        )
        self.assertAlmostEqual(palm_size(landmarks), float(expected), places=6)

    def test_aspect_correction_makes_palm_size_rotation_invariant(self) -> None:
        width, height = 1920.0, 1080.0
        hand = self.make_palm() * 100.0 + np.array([900.0, 500.0, 0.0])
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        center = hand[0]
        rotated = (hand - center) @ rotation.T + center

        def to_mediapipe(points: np.ndarray) -> np.ndarray:
            normalized = points.copy()
            normalized[:, 0] /= width
            normalized[:, 1] /= height
            normalized[:, 2] /= width
            return normalized.astype(np.float32)

        upright = to_mediapipe(hand)
        turned = to_mediapipe(rotated)
        self.assertNotAlmostEqual(palm_size(upright), palm_size(turned), places=3)
        aspect = width / height
        self.assertAlmostEqual(
            palm_size(correct_aspect_ratio(upright, aspect)),
            palm_size(correct_aspect_ratio(turned, aspect)),
            places=5,
        )

    def test_mirroring_negates_wrist_centred_x(self) -> None:
        hand = self.make_palm() * 0.1 + np.array([0.3, 0.6, 0.0], dtype=np.float32)
        original = normalize_hand_landmarks(hand).reshape(21, 3)
        mirrored = normalize_hand_landmarks(mirror_hand_landmarks(hand)).reshape(21, 3)
        np.testing.assert_allclose(mirrored[:, 0], -original[:, 0], atol=1e-5)
        np.testing.assert_allclose(mirrored[:, 1:], original[:, 1:], atol=1e-5)

    def test_one_euro_filter_is_stable_and_finite(self) -> None:
        smoother = OneEuroFilter(3, min_cutoff=1.0, beta=0.01)
        values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        first = smoother(values, 1.0 / 15.0)
        second = smoother(values + 0.5, 1.0 / 15.0)
        self.assertEqual(first.shape, (3,))
        self.assertTrue(np.isfinite(second).all())

    def test_downsample_and_label_alignment(self) -> None:
        values = np.arange(30, dtype=np.float32).reshape(30, 1)
        down = downsample_sequence(values, source_fps=30.0, target_fps=15.0)
        self.assertEqual(down.shape[0], 15)
        labels = np.array([0] * 10 + [1] * 20, dtype=np.int64)
        aligned = align_labels_to_target_fps(labels, 30.0, 15.0)
        self.assertEqual(aligned.shape[0], 15)
        self.assertEqual(int(aligned[0]), 0)
        self.assertEqual(int(aligned[-1]), 1)

    def test_build_and_roundtrip_sequence(self) -> None:
        frames = 20
        coords = np.zeros((frames, 63), dtype=np.float32)
        coords[:, 0] = np.linspace(0.0, 1.0, frames)
        features = build_feature_matrix(
            coords,
            np.ones(frames, dtype=np.float32),
            np.ones(frames, dtype=np.float32),
            fps=15.0,
        )
        self.assertEqual(features.shape, (128, frames))
        record = SequenceRecord(
            features=features,
            labels=np.zeros(frames, dtype=np.int64),
            valid_mask=np.ones(frames, dtype=np.float32),
            video_id="demo",
            fps=15.0,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "demo.npz"
            save_sequence(path, record)
            loaded = load_sequence(path)
        self.assertEqual(loaded.video_id, "demo")
        self.assertEqual(loaded.features.shape, (128, frames))
        np.testing.assert_array_equal(loaded.labels, record.labels)


if __name__ == "__main__":
    unittest.main()
