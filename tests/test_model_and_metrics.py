from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from model import MS_TCN2, build_model
from utils.inference import overlap_window_starts, predict_sequence
from utils.metrics import (
    ActionSegment,
    edit_score,
    evaluate_sequence,
    f1_at_iou,
    frame_accuracy,
    labels_to_segments,
)
from utils.mock_data import generate_mock_sequence
from utils.schema import FEATURE_DIM, NUM_CLASSES
from utils.trainer import multi_stage_loss, tmse_loss


class ModelAndMetricsTests(unittest.TestCase):
    def test_model_output_shapes(self) -> None:
        model = MS_TCN2(
            num_layers_PG=3,
            num_layers_R=2,
            num_R=2,
            num_f_maps=16,
            dim=FEATURE_DIM,
            num_classes=NUM_CLASSES,
        )
        inputs = torch.randn(2, FEATURE_DIM, 40)
        outputs = model(inputs)
        self.assertEqual(tuple(outputs.shape), (3, 2, NUM_CLASSES, 40))

    def test_factory_accepts_parameter_aliases(self) -> None:
        model = build_model(
            "MS_TCN2",
            {
                "num_layers_pg": 2,
                "num_layers_r": 1,
                "num_refinement_stages": 1,
                "num_feature_maps": 8,
                "input_dim": FEATURE_DIM,
                "num_classes": NUM_CLASSES,
                "dropout": 0.1,
            },
        )
        outputs = model(torch.randn(1, FEATURE_DIM, 12))
        self.assertEqual(tuple(outputs.shape), (2, 1, NUM_CLASSES, 12))

    def test_multi_stage_loss_is_finite(self) -> None:
        outputs = torch.randn(2, 2, NUM_CLASSES, 16)
        labels = torch.randint(0, NUM_CLASSES, (2, 16))
        labels[0, -2:] = -100
        valid_mask = torch.ones(2, 16)
        valid_mask[0, -2:] = 0
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        loss = multi_stage_loss(outputs, labels, valid_mask, criterion, 0.1, 16.0)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(tmse_loss(outputs[0], valid_mask)), 0.0)

    def test_metrics_on_perfect_and_shifted_segments(self) -> None:
        target = np.array([14, 14, 1, 1, 1, 14, 2, 2, 14], dtype=np.int64)
        perfect = target.copy()
        self.assertEqual(frame_accuracy(perfect, target), 1.0)
        self.assertEqual(edit_score(perfect, target), 100.0)
        self.assertEqual(f1_at_iou(perfect, target, 0.5), 1.0)
        shifted = np.array([14, 1, 1, 1, 14, 14, 2, 2, 14], dtype=np.int64)
        metrics = evaluate_sequence(shifted, target)
        self.assertLess(metrics["frame_accuracy"], 1.0)
        self.assertGreater(metrics["edit"], 0.0)
        segments = labels_to_segments(target)
        self.assertEqual(segments[1], ActionSegment(1, 2, 5))

    def test_overlap_windows_cover_the_sequence(self) -> None:
        starts = overlap_window_starts(100, window_size=60, stride=12)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1], 40)
        self.assertIn(12, starts)

    def test_overlap_inference_returns_full_timeline(self) -> None:
        model = MS_TCN2(
            num_layers_PG=2,
            num_layers_R=1,
            num_R=1,
            num_f_maps=8,
            dim=FEATURE_DIM,
            num_classes=NUM_CLASSES,
        )
        model.eval()
        record = generate_mock_sequence("infer", 90, 15.0, seed=5)
        prediction, logits = predict_sequence(
            model,
            record,
            torch.device("cpu"),
            window_size=30,
            stride=15,
        )
        self.assertEqual(prediction.shape, (90,))
        self.assertEqual(logits.shape, (NUM_CLASSES, 90))
        self.assertTrue(np.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
