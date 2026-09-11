from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from utils.preprocessing.assemble import AssembleConfig, assemble_entry
from utils.preprocessing.tracks import (
    TrackConfig,
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
