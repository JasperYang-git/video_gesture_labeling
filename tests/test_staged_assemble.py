from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from utils.preprocessing.assemble import AssembleConfig, assemble_entry
from utils.preprocessing.tracks import (
    TrackConfig,
    load_track,
    save_track,
    track_cache_path,
)
from utils.schema import load_sequence


class StagedAssembleTests(unittest.TestCase):
    def make_entry(self, root: Path, polarity: str, annotation: str):
        video = root / "video.mp4"
        annotation_path = root / "gestures.annotation~"
        video.write_bytes(b"video")
        annotation_path.write_text(annotation, encoding="utf-8")
        return SimpleNamespace(
            video_id="data_lm__sample",
            source="data_lm",
            subject_id="LMVibra006",
            gender="M",
            field3="L",
            scene="ring-strong" if polarity == "negative" else "stand",
            scene_raw="ring-strong" if polarity == "negative" else "stand",
            polarity=polarity,
            scene_category="confounder" if polarity == "negative" else "posture",
            hard_negative_eligible=polarity == "negative",
            taxonomy_version=1,
            video_path=video,
            annotation_path=annotation_path,
            status="ready",
        )

    def save_fake_track(self, entry, config: TrackConfig) -> None:
        frames = 30
        landmarks = np.zeros((frames, 21, 3), dtype=np.float32)
        landmarks[:, 9, 1] = 0.2
        path = track_cache_path(config, entry.source, entry.video_id)
        save_track(
            path,
            landmarks,
            np.ones(frames, dtype=np.float32),
            np.ones(frames, dtype=np.float32),
            30.0,
            {
                "track_fingerprint": "track",
                "track_config_fingerprint": config.fingerprint(),
            },
        )

    def test_assemble_from_cached_track_without_mediapipe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            entry = self.make_entry(root, "positive", "0;0.5;0;1;\n")
            track_config = TrackConfig(cache_dir=str(root / "tracks"))
            self.save_fake_track(entry, track_config)
            assemble_config = AssembleConfig(output_dir=str(root / "processed"))
            result = assemble_entry(entry, track_config, assemble_config)
            self.assertEqual(result["status"], "completed")
            record = load_sequence(result["sequence_path"])
            self.assertEqual(record.subject_id, "LMVibra006")
            self.assertEqual(record.source, "data_lm")
            self.assertEqual(record.features.shape, (128, 15))

    def save_hand_track(
        self,
        entry,
        config: TrackConfig,
        landmarks: np.ndarray,
        hand_side: np.ndarray,
    ) -> None:
        frames = len(landmarks)
        save_track(
            track_cache_path(config, entry.source, entry.video_id),
            landmarks,
            np.ones(frames, dtype=np.float32),
            np.ones(frames, dtype=np.float32),
            30.0,
            {
                "track_fingerprint": f"track-{entry.video_id}",
                "track_config_fingerprint": config.fingerprint(),
                "frame_width": 1920,
                "frame_height": 1080,
            },
            hand_side=hand_side,
        )

    def test_mirrored_left_hand_matches_the_right_hand(self) -> None:
        rng = np.random.default_rng(0)
        frames = 30
        right = (0.4 + 0.1 * rng.random((frames, 21, 3))).astype(np.float32)
        left = right.copy()
        left[:, :, 0] = 1.0 - left[:, :, 0]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            track_config = TrackConfig(
                physical_hand="auto", cache_dir=str(root / "tracks")
            )
            assemble_config = AssembleConfig(output_dir=str(root / "processed"))
            records = {}
            for name, landmarks, side in (
                ("right", right, 1),
                ("left", left, -1),
            ):
                folder = root / name
                folder.mkdir()
                entry = self.make_entry(folder, "positive", "0;0.5;0;1;\n")
                entry.video_id = f"data_lm__{name}"
                self.save_hand_track(
                    entry,
                    track_config,
                    landmarks,
                    np.full(frames, side, dtype=np.int8),
                )
                result = assemble_entry(entry, track_config, assemble_config)
                self.assertEqual(result["status"], "completed")
                records[name] = load_sequence(result["sequence_path"])

            np.testing.assert_allclose(
                records["left"].features, records["right"].features, atol=1e-4
            )
            self.assertEqual(records["left"].metadata["left_hand_ratio"], 1.0)
            self.assertEqual(records["right"].metadata["left_hand_ratio"], 0.0)
            self.assertAlmostEqual(
                float(records["right"].metadata["aspect_ratio"]), 1920 / 1080
            )
            self.assertTrue(bool(records["right"].metadata["aspect_corrected"]))

    def test_legacy_track_without_side_or_frame_size_still_assembles(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            entry = self.make_entry(root, "positive", "0;0.5;0;1;\n")
            track_config = TrackConfig(cache_dir=str(root / "tracks"))
            self.save_fake_track(entry, track_config)
            track = load_track(track_cache_path(track_config, entry.source, entry.video_id))
            np.testing.assert_array_equal(track["hand_side"], np.ones(30, dtype=np.int8))
            result = assemble_entry(
                entry,
                track_config,
                AssembleConfig(output_dir=str(root / "processed")),
            )
            self.assertEqual(result["status"], "completed")
            record = load_sequence(result["sequence_path"])
            self.assertEqual(record.hand_side, "physical_right")
            self.assertFalse(bool(record.metadata["aspect_corrected"]))
            self.assertIn("core_detection_rate", record.metadata)

    def test_negative_scene_rejects_action_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            entry = self.make_entry(root, "negative", "0;0.5;0;1;\n")
            track_config = TrackConfig(cache_dir=str(root / "tracks"))
            self.save_fake_track(entry, track_config)
            result = assemble_entry(
                entry,
                track_config,
                AssembleConfig(output_dir=str(root / "processed")),
            )
            self.assertEqual(result["status"], "annotation_conflict")


if __name__ == "__main__":
    unittest.main()
