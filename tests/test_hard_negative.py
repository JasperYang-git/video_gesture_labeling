from __future__ import annotations

import unittest

import numpy as np
from torch.utils.data import DataLoader

from evaluate_script import grouped_metrics, negative_video_metrics
from utils.data_loader import GestureWindowDataset, collate_windows
from utils.schema import BACKGROUND_ID, FEATURE_DIM, SequenceRecord


def make_record(
    video_id: str,
    labels: np.ndarray,
    **metadata: object,
) -> SequenceRecord:
    return SequenceRecord(
        features=np.zeros((FEATURE_DIM, len(labels)), dtype=np.float32),
        labels=labels,
        valid_mask=np.ones(len(labels), dtype=np.float32),
        video_id=video_id,
        fps=10.0,
        metadata=metadata,
    )


class HardNegativeDatasetTests(unittest.TestCase):
    def test_hard_negatives_use_action_ratio_independent_of_background_prob(self) -> None:
        action = make_record("action", np.ones(100, dtype=np.int64), polarity="positive")
        hard_negative = make_record(
            "negative",
            np.full(100, BACKGROUND_ID, dtype=np.int64),
            polarity="negative",
            hard_negative_eligible=True,
        )
        ordinary_background = make_record(
            "positive_background",
            np.full(100, BACKGROUND_ID, dtype=np.int64),
            polarity="positive",
            hard_negative_eligible=True,
        )

        dataset = GestureWindowDataset(
            [action, hard_negative, ordinary_background],
            window_size=10,
            stride=10,
            keep_background_prob=0.0,
            hard_negative_ratio=0.2,
            seed=4,
        )

        roles = [window.window_role for window in dataset.windows]
        self.assertEqual(roles.count("action"), 10)
        self.assertEqual(roles.count("hard_negative"), 2)
        self.assertNotIn("positive_background", {item.video_id for item in dataset.windows})

    def test_hard_negative_per_video_cap_and_batch_role(self) -> None:
        action = make_record("action", np.ones(60, dtype=np.int64))
        hard_negative = make_record(
            "negative",
            np.full(60, BACKGROUND_ID, dtype=np.int64),
            polarity="negative",
            hard_negative_eligible="true",
        )
        dataset = GestureWindowDataset(
            [action, hard_negative],
            window_size=10,
            stride=10,
            keep_background_prob=0.0,
            hard_negative_ratio=1.0,
            hard_negative_max_windows_per_video=2,
            seed=1,
        )
        self.assertEqual(
            sum(item.window_role == "hard_negative" for item in dataset.windows),
            2,
        )

        batch = next(
            iter(
                DataLoader(
                    dataset,
                    batch_size=len(dataset),
                    shuffle=False,
                    collate_fn=collate_windows,
                )
            )
        )
        self.assertEqual(batch["window_role"], [item.window_role for item in dataset.windows])

    def test_negative_with_action_is_not_a_hard_negative_video(self) -> None:
        labels = np.full(40, BACKGROUND_ID, dtype=np.int64)
        labels[10:20] = 2
        record = make_record(
            "not_all_background",
            labels,
            polarity="negative",
            hard_negative_eligible=True,
        )
        dataset = GestureWindowDataset(
            [record],
            window_size=10,
            stride=10,
            keep_background_prob=0.0,
        )
        self.assertTrue(dataset.windows)
        self.assertTrue(all(item.window_role == "action" for item in dataset.windows))


class NegativeEvaluationTests(unittest.TestCase):
    def test_negative_video_metrics_and_grouping(self) -> None:
        prediction = np.array([14, 1, 1, 14, 2, 14], dtype=np.int64)
        metrics = negative_video_metrics(prediction, fps=1.0)
        self.assertAlmostEqual(metrics["false_positive_frame_rate"], 0.5)
        self.assertEqual(metrics["predicted_action_segments"], 2)
        self.assertAlmostEqual(metrics["false_actions_per_minute"], 20.0)

        per_video = [
            {
                "video_id": "negative",
                "source": "field",
                "scene": "desk",
                "polarity": "negative",
                "frame_accuracy": 0.5,
                **metrics,
            },
            {
                "video_id": "positive",
                "source": "field",
                "scene": "room",
                "polarity": "positive",
                "frame_accuracy": 1.0,
            },
        ]
        grouped = grouped_metrics(per_video)
        self.assertEqual(grouped["source"]["field"]["video_count"], 2)
        self.assertAlmostEqual(grouped["source"]["field"]["frame_accuracy"], 0.75)
        self.assertEqual(grouped["polarity"]["negative"]["predicted_action_segments"], 2.0)


if __name__ == "__main__":
    unittest.main()
