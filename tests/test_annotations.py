from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.preprocessing.annotations import (
    infer_hand_side_from_folder,
    labels_from_intervals,
    parse_nova_annotation,
)
from utils.preprocessing.pipeline import discover_raw_videos
from utils.schema import BACKGROUND_ID


class AnnotationTests(unittest.TestCase):
    def test_parse_nova_annotation_and_label_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "gestures.annotation~"
            path.write_text(
                "0.0000;0.2000;10;1;\n"
                "1.0000;1.4000;-1;1;\n"
                "2.0000;2.5000;12;1;\n",
                encoding="utf-8",
            )
            intervals = parse_nova_annotation(path)
            self.assertEqual(len(intervals), 3)
            self.assertEqual(intervals[0].label, 10)
            self.assertEqual(intervals[1].label, BACKGROUND_ID)
            labels = labels_from_intervals(num_frames=30, fps=10.0, intervals=intervals)
            self.assertTrue(np.all(labels[0:2] == 10))
            self.assertTrue(np.all(labels[10:14] == BACKGROUND_ID))
            self.assertTrue(np.all(labels[20:25] == 12))
            self.assertTrue(np.all(labels[25:] == BACKGROUND_ID))

    def test_invalid_interval_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "gestures.annotation~"
            path.write_text("1.0000;0.2000;1;1;\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_nova_annotation(path)

    def test_hand_side_from_folder_name(self) -> None:
        self.assertEqual(infer_hand_side_from_folder("12_M_R_session"), "R")
        self.assertEqual(infer_hand_side_from_folder("12_F_L_session"), "L")
        self.assertIsNone(infer_hand_side_from_folder("session"))

    def test_discovers_labeled_videos_below_type_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            valid = root / "sit" / "DG2024062702_F_R_sit"
            (valid / "NOVA project").mkdir(parents=True)
            (valid / "recording.mp4").touch()
            (valid / "NOVA project" / "gestures.annotation~").write_text(
                "0;1;0;1;\n",
                encoding="utf-8",
            )

            missing_video = root / "walk" / "DG2024062702_F_R_walk"
            (missing_video / "NOVA project").mkdir(parents=True)
            (missing_video / "NOVA project" / "gestures.annotation~").touch()

            missing_annotation = root / "tap" / "DG2024062702_F_R_tap"
            missing_annotation.mkdir(parents=True)
            (missing_annotation / "recording.mp4").touch()

            wrong_side = root / "sit" / "DG2024062702_F_L_sit"
            (wrong_side / "NOVA project").mkdir(parents=True)
            (wrong_side / "recording.mp4").touch()
            (wrong_side / "NOVA project" / "gestures.annotation~").touch()

            items = discover_raw_videos(root, hand_side="R")

            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["video_id"], "sit__DG2024062702_F_R_sit")
            self.assertEqual(items[0]["video_path"], valid / "recording.mp4")
            self.assertEqual(
                items[0]["annotation_path"],
                valid / "NOVA project" / "gestures.annotation~",
            )


if __name__ == "__main__":
    unittest.main()
