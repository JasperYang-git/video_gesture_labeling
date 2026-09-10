from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.schema import BACKGROUND_ID, CLASS_NAMES, class_id_to_name


@dataclass(frozen=True)
class ActionSegment:
    label: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start

    def to_dict(self, fps: float | None = None) -> dict[str, float | int | str]:
        payload: dict[str, float | int | str] = {
            "label": int(self.label),
            "name": class_id_to_name(int(self.label)),
            "start_frame": int(self.start),
            "end_frame": int(self.end),
        }
        if fps:
            payload["start_sec"] = float(self.start / fps)
            payload["end_sec"] = float(self.end / fps)
        return payload


def labels_to_segments(labels: np.ndarray) -> list[ActionSegment]:
    if labels.ndim != 1:
        raise ValueError("labels must be 1-D")
    if len(labels) == 0:
        return []
    segments: list[ActionSegment] = []
    start = 0
    current = int(labels[0])
    for index in range(1, len(labels)):
        label = int(labels[index])
        if label != current:
            segments.append(ActionSegment(current, start, index))
            start = index
            current = label
    segments.append(ActionSegment(current, start, len(labels)))
    return segments


def segment_iou(left: ActionSegment, right: ActionSegment) -> float:
    start = max(left.start, right.start)
    end = min(left.end, right.end)
    intersection = max(0, end - start)
    union = left.length + right.length - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def levenshtein(left: list[int], right: list[int]) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            cost = 0 if left_item == right_item else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def edit_score(prediction: np.ndarray, target: np.ndarray) -> float:
    pred_labels = [segment.label for segment in labels_to_segments(prediction)]
    target_labels = [segment.label for segment in labels_to_segments(target)]
    if not pred_labels and not target_labels:
        return 100.0
    distance = levenshtein(pred_labels, target_labels)
    return (1.0 - distance / max(len(pred_labels), len(target_labels))) * 100.0


def frame_accuracy(prediction: np.ndarray, target: np.ndarray) -> float:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    if len(target) == 0:
        return 0.0
    return float(np.mean(prediction == target))


def segmental_f1(
    prediction: np.ndarray,
    target: np.ndarray,
    iou_threshold: float,
    ignore_background: bool = True,
) -> float:
    return 100.0 * f1_at_iou(
        prediction,
        target,
        iou_threshold,
        ignore_background=ignore_background,
    )


def f1_at_iou(
    prediction: np.ndarray,
    target: np.ndarray,
    iou_threshold: float,
    ignore_background: bool = True,
) -> float:
    pred_segments = [
        segment
        for segment in labels_to_segments(prediction)
        if not ignore_background or segment.label != BACKGROUND_ID
    ]
    target_segments = [
        segment
        for segment in labels_to_segments(target)
        if not ignore_background or segment.label != BACKGROUND_ID
    ]
    if not pred_segments and not target_segments:
        return 1.0
    if not pred_segments or not target_segments:
        return 0.0

    matched = [False] * len(target_segments)
    true_positive = 0
    for pred in pred_segments:
        best_index = -1
        best_iou = 0.0
        for index, target_segment in enumerate(target_segments):
            if matched[index] or pred.label != target_segment.label:
                continue
            iou = segment_iou(pred, target_segment)
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_index = index
        if best_index >= 0:
            matched[best_index] = True
            true_positive += 1
    precision = true_positive / len(pred_segments)
    recall = true_positive / len(target_segments)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def evaluate_sequence(
    prediction: np.ndarray,
    target: np.ndarray,
    iou_thresholds: list[float] | tuple[float, ...] = (0.1, 0.25, 0.5),
) -> dict[str, float]:
    metrics = {
        "frame_accuracy": frame_accuracy(prediction, target),
        "edit": edit_score(prediction, target),
    }
    for threshold in iou_thresholds:
        metrics[f"f1@{threshold}"] = f1_at_iou(prediction, target, threshold)
    return metrics


def aggregate_metrics(per_video: list[dict[str, float]]) -> dict[str, float]:
    if not per_video:
        return {}
    keys = per_video[0].keys()
    return {
        key: float(np.mean([item[key] for item in per_video]))
        for key in keys
    }


def confusion_matrix(
    prediction: np.ndarray,
    target: np.ndarray,
    num_classes: int = len(CLASS_NAMES),
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(target.tolist(), prediction.tolist()):
        if 0 <= true_label < num_classes and 0 <= pred_label < num_classes:
            matrix[true_label, pred_label] += 1
    return matrix
