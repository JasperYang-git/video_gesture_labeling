from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

from utils.preprocessing.tracks import (
    LEFT_SIDE,
    NO_HAND,
    RIGHT_SIDE,
    TrackConfig,
    extract_inventory,
    mediapipe_label_for_physical_hand,
    select_auto_hand,
    select_hand,
    select_strict_hand,
)


def fake_hand(label: str, score: float, x: float):
    landmarks = SimpleNamespace(
        landmark=[
            SimpleNamespace(x=x, y=0.5, z=0.0)
            for _ in range(21)
        ]
    )
    handedness = SimpleNamespace(
        classification=[SimpleNamespace(label=label, score=score)]
    )
    return landmarks, handedness


class StrictRightTrackTests(unittest.TestCase):
    def test_non_mirrored_physical_right_maps_to_left_label(self) -> None:
        self.assertEqual(
            mediapipe_label_for_physical_hand("right", input_mirrored=False),
            "Left",
        )
        self.assertEqual(
            mediapipe_label_for_physical_hand("right", input_mirrored=True),
            "Right",
        )

    def test_never_falls_back_to_opposite_hand(self) -> None:
        left_points, left_handedness = fake_hand("Left", 0.8, 0.2)
        right_points, right_handedness = fake_hand("Right", 0.99, 0.8)
        result = SimpleNamespace(
            multi_hand_landmarks=[left_points, right_points],
            multi_handedness=[left_handedness, right_handedness],
        )
        selected = select_strict_hand(
            result,
            TrackConfig(
                physical_hand="right",
                input_mirrored=False,
                handedness_threshold=0.7,
            ),
        )
        self.assertIsNotNone(selected)
        points, score = selected
        self.assertAlmostEqual(float(points[0, 0]), 0.2)
        self.assertAlmostEqual(score, 0.8)

        only_opposite = SimpleNamespace(
            multi_hand_landmarks=[right_points],
            multi_handedness=[right_handedness],
        )
        self.assertIsNone(
            select_strict_hand(
                only_opposite,
                TrackConfig(physical_hand="right", input_mirrored=False),
            )
        )

    def test_auto_mode_selects_a_lone_left_hand(self) -> None:
        # Non-mirrored input: MediaPipe calls the physical left hand "Right".
        left_points, left_handedness = fake_hand("Right", 0.9, 0.7)
        result = SimpleNamespace(
            multi_hand_landmarks=[left_points],
            multi_handedness=[left_handedness],
        )
        config = TrackConfig(physical_hand="auto", input_mirrored=False)
        selected = select_hand(result, config)
        self.assertIsNotNone(selected)
        points, score, side = selected
        self.assertEqual(side, LEFT_SIDE)
        self.assertAlmostEqual(float(points[0, 0]), 0.7)
        self.assertIsNone(select_strict_hand(result, TrackConfig(physical_hand="right")))

    def test_auto_mode_sticks_to_the_previous_side(self) -> None:
        right_points, right_handedness = fake_hand("Left", 0.75, 0.2)
        left_points, left_handedness = fake_hand("Right", 0.95, 0.8)
        result = SimpleNamespace(
            multi_hand_landmarks=[right_points, left_points],
            multi_handedness=[right_handedness, left_handedness],
        )
        config = TrackConfig(physical_hand="auto", input_mirrored=False)
        self.assertEqual(select_auto_hand(result, config, NO_HAND)[2], LEFT_SIDE)
        self.assertEqual(select_auto_hand(result, config, RIGHT_SIDE)[2], RIGHT_SIDE)
        self.assertEqual(select_auto_hand(result, config, LEFT_SIDE)[2], LEFT_SIDE)

    def test_strict_mode_reports_its_own_side(self) -> None:
        right_points, right_handedness = fake_hand("Left", 0.9, 0.2)
        result = SimpleNamespace(
            multi_hand_landmarks=[right_points],
            multi_handedness=[right_handedness],
        )
        selected = select_hand(result, TrackConfig(physical_hand="right"), LEFT_SIDE)
        self.assertEqual(selected[2], RIGHT_SIDE)

    def test_opposite_fallback_cannot_be_enabled(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbidden"):
            TrackConfig(allow_opposite_hand_fallback=True)

    def test_extraction_failures_are_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            entries = [
                SimpleNamespace(
                    video_id=f"video_{index}",
                    source="data_lm",
                    video_path=str(Path(temp_dir) / f"missing_{index}.mp4"),
                    status="ok",
                )
                for index in range(2)
            ]
            results = extract_inventory(
                entries,
                TrackConfig(
                    num_workers=1,
                    cache_dir=str(Path(temp_dir) / "tracks"),
                ),
            )
            self.assertEqual(len(results), 2)
            self.assertTrue(all(item["status"] == "failed" for item in results))


if __name__ == "__main__":
    unittest.main()
