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


if __name__ == "__main__":
    unittest.main()
