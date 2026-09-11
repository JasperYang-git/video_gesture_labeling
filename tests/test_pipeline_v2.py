from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.data_loader import GestureWindowDataset
from utils.inference import validate_record_compatibility
from utils.preprocessing.features import (
    aggregate_hand_tracks,
    align_labels_to_target_fps,
    build_feature_matrix,
)
from utils.preprocessing.pipeline import (
    PreprocessConfig,
    compute_quality_audit,
    process_raw_videos,
    source_fingerprint,
)
from utils.schema import BACKGROUND_ID, COORD_DIM, SequenceRecord, save_sequence
from utils.split import split_grouped_files


def make_record(video_id: str, labels: np.ndarray, subject_id: str = "") -> SequenceRecord:
    frames = len(labels)
    features = build_feature_matrix(
        np.zeros((frames, COORD_DIM), dtype=np.float32),
        np.ones(frames, dtype=np.float32),
        np.ones(frames, dtype=np.float32),
        15.0,
    )
    return SequenceRecord(
        features=features,
        labels=labels.astype(np.int64),
        valid_mask=np.ones(frames, dtype=np.float32),
        video_id=video_id,
        fps=15.0,
        subject_id=subject_id,
        metadata={
            "preprocess_fingerprint": "source-specific",
            "preprocess_config_fingerprint": "recipe-v2",
        },
    )


class PipelineV2Tests(unittest.TestCase):
    def test_aggregate_then_normalize_uses_bucket_palm(self) -> None:
        coords = np.zeros((2, COORD_DIM), dtype=np.float32)
        coords[:, 0] = 2.0
        normalized, quality, mask = aggregate_hand_tracks(
            coords,
            np.asarray([1.0, 3.0], dtype=np.float32),
            np.asarray([0.8, 1.0], dtype=np.float32),
            np.ones(2, dtype=np.float32),
            source_fps=30.0,
            target_fps=15.0,
        )
        self.assertAlmostEqual(float(normalized[0, 0]), 1.0)
        self.assertAlmostEqual(float(quality[0]), 0.9, places=6)
        self.assertEqual(float(mask[0]), 1.0)

    def test_quality_audit_uses_action_frames_and_classes(self) -> None:
        labels = np.asarray([BACKGROUND_ID, 0, 0, 1, 1, BACKGROUND_ID])
        valid = np.asarray([0, 1, 0, 1, 1, 0], dtype=np.float32)
        audit = compute_quality_audit(
            np.zeros((6, COORD_DIM), dtype=np.float32),
            np.ones(6, dtype=np.float32),
            valid,
            labels,
            fps=2.0,
        )
        self.assertAlmostEqual(audit["detection_rate"], 0.75)
        self.assertAlmostEqual(audit["per_class_detection_rate"]["0"], 0.5)
        self.assertEqual(audit["min_class_id"], "0")
        self.assertEqual(audit["longest_missing_frames"], 1)

    def test_non_integer_fps_features_and_labels_stay_aligned(self) -> None:
        frames = 101
        coordinates = np.zeros((frames, COORD_DIM), dtype=np.float32)
        normalized, _, _ = aggregate_hand_tracks(
            coordinates,
            np.ones(frames, dtype=np.float32),
            np.ones(frames, dtype=np.float32),
            np.ones(frames, dtype=np.float32),
            source_fps=29.97,
            target_fps=15.0,
        )
        labels = align_labels_to_target_fps(
            np.zeros(frames, dtype=np.int64),
            source_fps=29.97,
            target_fps=15.0,
        )
        self.assertEqual(len(normalized), len(labels))

    def test_fingerprint_changes_when_annotation_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "video.mp4"
            annotation = root / "gestures.annotation~"
            video.write_bytes(b"fake-video")
            annotation.write_text("0;1;0;1;\n", encoding="utf-8")
            first, _, _ = source_fingerprint(PreprocessConfig(), video, annotation)
            annotation.write_text("0;1;1;1;\n", encoding="utf-8")
            second, _, _ = source_fingerprint(PreprocessConfig(), video, annotation)
            self.assertNotEqual(first, second)

    def test_parallel_preprocessing_reuses_valid_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "processed"
            config = PreprocessConfig(cache_dir=str(root / "cache"), num_workers=2)
            items = []
            for index in range(3):
                video_id = f"video_{index}"
                video = root / f"{video_id}.mp4"
                annotation = root / f"{video_id}.annotation"
                video.write_bytes(f"video-{index}".encode())
                annotation.write_text("0;1;0;1;\n", encoding="utf-8")
                fingerprint, _, _ = source_fingerprint(config, video, annotation)
                record = make_record(
                    video_id,
                    np.zeros(20, dtype=np.int64),
                )
                record.metadata["preprocess_fingerprint"] = fingerprint
                save_sequence(output_dir / f"{video_id}.npz", record)
                items.append(
                    {
                        "video_id": video_id,
                        "video_path": video,
                        "annotation_path": annotation,
                    }
                )
            records = process_raw_videos(items, config, output_dir)
            self.assertEqual([record.video_id for record in records], [
                "video_0",
                "video_1",
                "video_2",
            ])

    def test_subject_split_has_no_cross_split_leakage(self) -> None:
        files = [Path(f"a{i}.npz") for i in range(2)] + [
            Path("b0.npz"),
            Path("c0.npz"),
        ]
        groups = {"a0": "A", "a1": "A", "b0": "B", "c0": "C"}
        splits = split_grouped_files(files, groups, 0.5, 0.25, 0.25, seed=3)
        locations = {
            path.stem: split_name
            for split_name, paths in splits.items()
            for path in paths
        }
        self.assertEqual(locations["a0"], locations["a1"])

    def test_dynamic_windows_change_by_epoch_without_pure_background(self) -> None:
        labels = np.full(150, BACKGROUND_ID, dtype=np.int64)
        labels[40:50] = 2
        labels[100:115] = 3
        dataset = GestureWindowDataset(
            [make_record("video", labels)],
            window_size=60,
            stride=12,
            keep_background_prob=0.0,
            dynamic_action_windows=True,
            action_windows_per_segment=3,
            seed=9,
        )
        starts0 = [window.start for window in dataset.windows]
        dataset.set_epoch(1)
        starts1 = [window.start for window in dataset.windows]
        self.assertNotEqual(starts0, starts1)
        self.assertTrue(all(not window.is_background for window in dataset.windows))

    def test_checkpoint_compatibility_uses_recipe_fingerprint(self) -> None:
        record = make_record("video", np.zeros(20, dtype=np.int64))
        validate_record_compatibility(
            record, {"preprocess_fingerprints": ["recipe-v2"]}
        )
        with self.assertRaisesRegex(ValueError, "not compatible"):
            validate_record_compatibility(
                record, {"preprocess_fingerprints": ["other-recipe"]}
            )


if __name__ == "__main__":
    unittest.main()
