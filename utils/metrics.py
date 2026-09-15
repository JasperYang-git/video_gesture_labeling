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


def f1_at_iou_counts(
    prediction: np.ndarray,
    target: np.ndarray,
    iou_threshold: float,
    ignore_background: bool = True,
) -> dict[int, dict[str, int]]:
    """Per-class segment matching counts (tp/fp/fn) at one IoU threshold."""
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
    counts: dict[int, dict[str, int]] = {}

    def bucket(label: int) -> dict[str, int]:
        return counts.setdefault(int(label), {"tp": 0, "fp": 0, "fn": 0})

    matched = [False] * len(target_segments)
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
            bucket(pred.label)["tp"] += 1
        else:
            bucket(pred.label)["fp"] += 1
    for index, target_segment in enumerate(target_segments):
        if not matched[index]:
            bucket(target_segment.label)["fn"] += 1
    return counts


def f1_from_counts(true_positive: int, false_positive: int, false_negative: int) -> float:
    if true_positive + false_positive + false_negative == 0:
        return 1.0
    if true_positive == 0:
        return 0.0
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    return 2.0 * precision * recall / (precision + recall)


def f1_at_iou(
    prediction: np.ndarray,
    target: np.ndarray,
    iou_threshold: float,
    ignore_background: bool = True,
) -> float:
    counts = f1_at_iou_counts(
        prediction,
        target,
        iou_threshold,
        ignore_background=ignore_background,
    )
    true_positive = sum(item["tp"] for item in counts.values())
    false_positive = sum(item["fp"] for item in counts.values())
    false_negative = sum(item["fn"] for item in counts.values())
    return f1_from_counts(true_positive, false_positive, false_negative)


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


def frame_metrics_from_confusion(
    matrix: np.ndarray,
) -> dict[str, dict[str, float | int | str | None]]:
    """Per-class frame precision/recall/F1/support from an accumulated confusion matrix.

    Classes with neither ground-truth nor predicted frames report ``None`` instead of a
    vacuous 1.0, so they do not pollute rankings of the worst-performing classes.
    """
    report: dict[str, dict[str, float | int | str | None]] = {}
    for class_id in range(matrix.shape[0]):
        true_positive = int(matrix[class_id, class_id])
        support = int(matrix[class_id, :].sum())
        predicted = int(matrix[:, class_id].sum())
        false_negative = support - true_positive
        false_positive = predicted - true_positive
        row = matrix[class_id, :].copy()
        row[class_id] = 0
        top_confusion = int(row.argmax()) if row.sum() > 0 else -1
        absent = support == 0 and predicted == 0
        report[class_id_to_name(class_id)] = {
            "class_id": class_id,
            "support_frames": support,
            "predicted_frames": predicted,
            "frame_precision": None if absent else (true_positive / predicted if predicted else 0.0),
            "frame_recall": None if absent else (true_positive / support if support else 0.0),
            "frame_f1": (
                None
                if absent
                else f1_from_counts(true_positive, false_positive, false_negative)
            ),
            "top_confusion": (
                class_id_to_name(top_confusion) if top_confusion >= 0 else ""
            ),
            "top_confusion_frames": int(row[top_confusion]) if top_confusion >= 0 else 0,
        }
    return report


def segment_metrics_from_counts(
    counts: dict[int, dict[str, int]],
) -> dict[str, dict[str, float | int]]:
    """Per-class segment precision/recall/F1 from accumulated tp/fp/fn counts."""
    report: dict[str, dict[str, float | int]] = {}
    for class_id, item in sorted(counts.items()):
        true_positive = int(item["tp"])
        false_positive = int(item["fp"])
        false_negative = int(item["fn"])
        predicted = true_positive + false_positive
        support = true_positive + false_negative
        report[class_id_to_name(class_id)] = {
            "class_id": class_id,
            "support_segments": support,
            "predicted_segments": predicted,
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
            "segment_precision": (true_positive / predicted) if predicted else 0.0,
            "segment_recall": (true_positive / support) if support else 0.0,
            "segment_f1": f1_from_counts(true_positive, false_positive, false_negative),
        }
    return report
