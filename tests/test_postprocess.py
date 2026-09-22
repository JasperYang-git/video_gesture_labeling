from __future__ import annotations

import unittest

import numpy as np

from utils.postprocess import (
    PostprocessConfig,
    apply_postprocess,
    class_probabilities,
    filter_by_confidence,
    filter_by_duration,
    masked_argmax,
    resolve_class_ids,
)
from utils.schema import BACKGROUND_ID, NUM_CLASSES, class_name_to_id

DOUBLE_CLASSES = (
    "double-knock",
    "double-tap",
    "double-slide-up",
    "double-slide-down",
    "double-clench",
)


def logits_for(prediction: np.ndarray, margin: float = 5.0) -> np.ndarray:
    """Logits whose argmax reproduces the given prediction."""
    logits = np.zeros((NUM_CLASSES, len(prediction)), dtype=np.float32)
    logits[prediction, np.arange(len(prediction))] = margin
    return logits


class ConfigTests(unittest.TestCase):
    def test_defaults_are_a_no_op(self) -> None:
        config = PostprocessConfig.from_dict(None)
        self.assertFalse(config.enabled)
        prediction = np.full(30, BACKGROUND_ID, dtype=np.int64)
        prediction[5:20] = 3
        result = apply_postprocess(prediction, logits_for(prediction), 15.0, config)
        np.testing.assert_array_equal(result.prediction, prediction)
        self.assertFalse(result.changed)
        self.assertEqual(result.stages, [])

    def test_class_selectors_accept_names_and_ids(self) -> None:
        by_name = resolve_class_ids(["double-tap", "knock"])
        by_id = resolve_class_ids([class_name_to_id("double-tap"), 0])
        self.assertEqual(by_name, by_id)
        # Background is always allowed, otherwise nothing could be predicted as idle.
        self.assertIn(BACKGROUND_ID, by_name)

    def test_empty_selection_stays_empty(self) -> None:
        self.assertEqual(resolve_class_ids([]), ())

    def test_invalid_selectors_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_class_ids(["not-a-gesture"])
        with self.assertRaises(ValueError):
            resolve_class_ids([NUM_CLASSES + 3])


class MaskedArgmaxTests(unittest.TestCase):
    def test_masking_picks_the_best_allowed_class_instead_of_blanking(self) -> None:
        knock = class_name_to_id("knock")
        double_knock = class_name_to_id("double-knock")
        logits = np.zeros((NUM_CLASSES, 4), dtype=np.float32)
        logits[knock] = 3.0
        logits[double_knock] = 2.0  # runner-up

        unrestricted = masked_argmax(logits, [])
        np.testing.assert_array_equal(unrestricted, np.full(4, knock))

        restricted = masked_argmax(logits, resolve_class_ids(DOUBLE_CLASSES))
        # The segment survives as its double counterpart rather than becoming a hole.
        np.testing.assert_array_equal(restricted, np.full(4, double_knock))

    def test_background_wins_when_it_outranks_every_allowed_class(self) -> None:
        logits = np.zeros((NUM_CLASSES, 3), dtype=np.float32)
        logits[BACKGROUND_ID] = 9.0
        logits[class_name_to_id("knock")] = 5.0
        restricted = masked_argmax(logits, resolve_class_ids(DOUBLE_CLASSES))
        np.testing.assert_array_equal(restricted, np.full(3, BACKGROUND_ID))


class ConfidenceFilterTests(unittest.TestCase):
    def test_low_confidence_segments_go_back_to_background(self) -> None:
        prediction = np.full(40, BACKGROUND_ID, dtype=np.int64)
        prediction[5:15] = 3
        prediction[20:30] = 9
        probabilities = np.zeros((NUM_CLASSES, 40), dtype=np.float32)
        probabilities[3, 5:15] = 0.95
        probabilities[9, 20:30] = 0.42

        filtered = filter_by_confidence(prediction, probabilities, 0.7)
        self.assertTrue(np.all(filtered[5:15] == 3))
        self.assertTrue(np.all(filtered[20:30] == BACKGROUND_ID))

    def test_threshold_of_zero_changes_nothing(self) -> None:
        prediction = np.full(20, BACKGROUND_ID, dtype=np.int64)
        prediction[2:8] = 1
        probabilities = np.zeros((NUM_CLASSES, 20), dtype=np.float32)
        probabilities[1, 2:8] = 0.05
        np.testing.assert_array_equal(
            filter_by_confidence(prediction, probabilities, 0.0), prediction
        )


class DurationFilterTests(unittest.TestCase):
    def test_short_segments_are_dropped(self) -> None:
        prediction = np.full(60, BACKGROUND_ID, dtype=np.int64)
        prediction[10:22] = 3  # 12 frames, a plausible gesture
        prediction[40:42] = 9  # 2 frames, jitter

        filtered = filter_by_duration(prediction, min_frames=5)
        self.assertTrue(np.all(filtered[10:22] == 3))
        self.assertTrue(np.all(filtered[40:42] == BACKGROUND_ID))

    def test_a_blip_inside_one_gesture_is_absorbed_not_blanked(self) -> None:
        # Blanking here would split one gesture into two and make over-segmentation
        # worse, which is the opposite of what this filter is for.
        prediction = np.full(60, BACKGROUND_ID, dtype=np.int64)
        prediction[10:40] = 12
        prediction[24:26] = 3  # two-frame blip in the middle

        filtered = filter_by_duration(prediction, min_frames=5)
        self.assertTrue(np.all(filtered[10:40] == 12))
        self.assertEqual(int((np.diff(filtered) != 0).sum()), 2)

    def test_background_segments_are_never_touched(self) -> None:
        prediction = np.full(30, BACKGROUND_ID, dtype=np.int64)
        prediction[0:14] = 3
        prediction[16:30] = 3  # a two-frame background gap between two gestures
        filtered = filter_by_duration(prediction, min_frames=5)
        np.testing.assert_array_equal(filtered, prediction)


class PipelineTests(unittest.TestCase):
    def test_stages_run_in_order_and_report_what_changed(self) -> None:
        prediction = np.full(90, BACKGROUND_ID, dtype=np.int64)
        prediction[10:30] = class_name_to_id("knock")
        prediction[50:52] = class_name_to_id("double-tap")
        logits = logits_for(prediction)
        # Make double-knock the runner-up so the mask can rescue the knock segment.
        logits[class_name_to_id("double-knock"), 10:30] = 4.0

        config = PostprocessConfig.from_dict(
            {
                "allowed_classes": list(DOUBLE_CLASSES),
                "min_confidence": 0.0,
                "min_duration_sec": 0.5,  # 7.5 -> 8 frames at 15fps
            }
        )
        result = apply_postprocess(prediction, logits, 15.0, config)

        self.assertEqual(
            [stage["stage"] for stage in result.stages],
            ["allowed_classes", "min_duration"],
        )
        self.assertTrue(result.changed)
        # knock became double-knock, the two-frame double-tap was dropped.
        self.assertTrue(
            np.all(result.prediction[10:30] == class_name_to_id("double-knock"))
        )
        self.assertTrue(np.all(result.prediction[50:52] == BACKGROUND_ID))
        self.assertEqual(result.stages[-1]["min_frames"], 8)

    def test_confidence_stage_uses_softmax_of_the_logits(self) -> None:
        prediction = np.full(40, BACKGROUND_ID, dtype=np.int64)
        prediction[10:30] = 3
        logits = np.zeros((NUM_CLASSES, 40), dtype=np.float32)
        logits[3, 10:30] = 0.2  # barely above the other classes

        probabilities = class_probabilities(logits)
        self.assertLess(float(probabilities[3, 15]), 0.7)

        config = PostprocessConfig.from_dict({"min_confidence": 0.7})
        result = apply_postprocess(prediction, logits, 15.0, config)
        self.assertTrue(np.all(result.prediction == BACKGROUND_ID))
        self.assertEqual(result.stages[0]["action_segments_after"], 0)

    def test_input_prediction_is_never_mutated(self) -> None:
        prediction = np.full(40, BACKGROUND_ID, dtype=np.int64)
        prediction[10:12] = 3
        original = prediction.copy()
        config = PostprocessConfig.from_dict({"min_duration_sec": 1.0})
        apply_postprocess(prediction, logits_for(prediction), 15.0, config)
        np.testing.assert_array_equal(prediction, original)


if __name__ == "__main__":
    unittest.main()
