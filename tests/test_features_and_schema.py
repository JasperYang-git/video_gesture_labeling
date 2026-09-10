from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.preprocessing.features import (
    OneEuroFilter,
    align_labels_to_target_fps,
    build_feature_matrix,
    downsample_sequence,
    normalize_hand_landmarks,
    palm_size,
)
from utils.schema import (
    FEATURE_DIM,
    FEATURE_NAMES,
    SequenceRecord,
    load_sequence,
    save_sequence,
)


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
