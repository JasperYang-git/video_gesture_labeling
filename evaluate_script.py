from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from utils.config import load_config
from utils.logger import create_run_directory, setup_logger
from utils.metrics import (
    aggregate_metrics,
    confusion_matrix,
    evaluate_sequence,
    f1_at_iou_counts,
    f1_from_counts,
    frame_metrics_from_confusion,
    labels_to_segments,
    segment_metrics_from_counts,
)
from utils.schema import (
    BACKGROUND_ID,
    CLASS_NAMES,
    NUM_CLASSES,
    SequenceRecord,
    class_id_to_name,
    load_sequence,
)
from utils.split import load_manifest
from utils.trainer import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate full-video gesture predictions")
    parser.add_argument(
        "--config",
        default="config/config_eval.yaml",
        help="Path to evaluation YAML",
    )
    parser.add_argument(
        "--predictions-dir",
        default=None,
        help="Directory containing per-video prediction.npy files",
    )
    return parser.parse_args()


def find_latest_inference_dir(root: str | Path) -> Path:
    root_path = Path(root)
    candidates = sorted(
        path
        for path in root_path.glob("*/*")
        if path.is_dir() and (path / "predictions.json").is_file()
    )
    if not candidates:
        raise FileNotFoundError(f"No inference runs with predictions.json under {root_path}")
    return candidates[-1]


def load_prediction(predictions_dir: Path, video_id: str) -> np.ndarray:
    direct = predictions_dir / video_id / "prediction.npy"
    if direct.is_file():
        return np.load(direct)
    nested = list(predictions_dir.glob(f"*/{video_id}/prediction.npy"))
    if nested:
        return np.load(sorted(nested)[-1])
    raise FileNotFoundError(f"No prediction.npy found for {video_id} under {predictions_dir}")


def negative_video_metrics(
    prediction: np.ndarray,
    fps: float,
) -> dict[str, float | int | dict[str, int]]:
    action_segments = [
        segment
        for segment in labels_to_segments(prediction)
        if segment.label != BACKGROUND_ID
    ]
    duration_minutes = len(prediction) / fps / 60.0 if fps > 0 else 0.0
    by_class: dict[str, int] = {}
    for segment in action_segments:
        name = class_id_to_name(segment.label)
        by_class[name] = by_class.get(name, 0) + 1
    return {
        "false_positive_frame_rate": (
            float(np.mean(prediction != BACKGROUND_ID)) if len(prediction) else 0.0
        ),
        "predicted_action_segments": len(action_segments),
        "false_actions_per_minute": (
            len(action_segments) / duration_minutes if duration_minutes > 0 else 0.0
        ),
        "false_actions_by_class": dict(sorted(by_class.items())),
    }


def _metadata_group(record: SequenceRecord, key: str) -> str:
    value = record.metadata.get(key, "unknown")
    if isinstance(value, np.ndarray):
        value = value.item() if value.shape == () else ",".join(map(str, value.tolist()))
    text = str(value).strip()
    return text or "unknown"


def grouped_metrics(
    per_video: list[dict[str, float | int | str]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    grouped: dict[str, dict[str, dict[str, float | int]]] = {}
    for field in ("source", "scene", "polarity"):
        buckets: dict[str, list[dict[str, float | int | str]]] = {}
        for metrics in per_video:
            buckets.setdefault(str(metrics[field]), []).append(metrics)
        grouped[field] = {}
        for value, items in sorted(buckets.items()):
            numeric_keys = sorted(
                {
                    key
                    for item in items
                    for key, metric in item.items()
                    if isinstance(metric, (int, float)) and not isinstance(metric, bool)
                }
            )
            summary: dict[str, float | int] = {"video_count": len(items)}
            for key in numeric_keys:
                values = [
                    float(item[key])
                    for item in items
                    if isinstance(item.get(key), (int, float))
                    and not isinstance(item.get(key), bool)
                ]
                if values:
                    summary[key] = float(np.mean(values))
            grouped[field][value] = summary
    return grouped


def class_coverage(
    matrix: np.ndarray,
    videos_with_class: dict[int, set[str]],
    gt_segment_counts: dict[int, int],
) -> dict[str, dict[str, int]]:
    return {
        class_id_to_name(class_id): {
            "class_id": class_id,
            "gt_frames": int(matrix[class_id, :].sum()),
            "gt_segments": int(gt_segment_counts.get(class_id, 0)),
            "videos": len(videos_with_class.get(class_id, ())),
        }
        for class_id in range(NUM_CLASSES)
    }


def per_class_report(
    matrix: np.ndarray,
    segment_counts: dict[float, dict[int, dict[str, int]]],
    coverage: dict[str, dict[str, int]],
) -> dict[str, dict[str, float | int | str | None]]:
    frame_report = frame_metrics_from_confusion(matrix)
    segment_reports = {
        threshold: segment_metrics_from_counts(counts)
        for threshold, counts in segment_counts.items()
    }
    report: dict[str, dict[str, float | int | str | None]] = {}
    for name, frame_stats in frame_report.items():
        entry: dict[str, float | int | str | None] = dict(frame_stats)
        entry["gt_segments"] = coverage[name]["gt_segments"]
        entry["videos"] = coverage[name]["videos"]
        for threshold, segment_report in segment_reports.items():
            stats = segment_report.get(name)
            if stats is None:
                continue
            entry[f"segment_f1@{threshold}"] = stats["segment_f1"]
            entry[f"segment_precision@{threshold}"] = stats["segment_precision"]
            entry[f"segment_recall@{threshold}"] = stats["segment_recall"]
            entry[f"predicted_segments@{threshold}"] = stats["predicted_segments"]
        report[name] = entry
    return report


def micro_segment_metrics(
    segment_counts: dict[float, dict[int, dict[str, int]]],
) -> dict[str, float]:
    summary: dict[str, float] = {}
    for threshold, counts in segment_counts.items():
        true_positive = sum(item["tp"] for item in counts.values())
        false_positive = sum(item["fp"] for item in counts.values())
        false_negative = sum(item["fn"] for item in counts.values())
        summary[f"micro_f1@{threshold}"] = f1_from_counts(
            true_positive, false_positive, false_negative
        )
        per_class_f1 = [
            f1_from_counts(item["tp"], item["fp"], item["fn"])
            for item in counts.values()
        ]
        summary[f"macro_class_f1@{threshold}"] = (
            float(np.mean(per_class_f1)) if per_class_f1 else 0.0
        )
    return summary


def write_confusion_csv(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true\\pred", *CLASS_NAMES])
        for class_id, name in enumerate(CLASS_NAMES):
            writer.writerow([name, *matrix[class_id, :].tolist()])


def write_per_class_csv(
    path: Path,
    report: dict[str, dict[str, float | int | str | None]],
) -> None:
    if not report:
        return
    columns: list[str] = []
    for entry in report.values():
        for key in entry:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["class_name", *columns])
        for name, entry in report.items():
            writer.writerow(
                [
                    name,
                    *(
                        f"{entry[key]:.4f}"
                        if isinstance(entry.get(key), float)
                        else entry.get(key, "")
                        for key in columns
                    ),
                ]
            )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    run_dir = create_run_directory(config["output"]["root_dir"])
    logger = setup_logger("evaluation", run_dir / "evaluation.log")
    shutil.copy2(args.config, run_dir / "config_eval.yaml")

    data_config = config["data"]
    splits = load_manifest(data_config["manifest_path"])
    split_name = str(data_config.get("split", "test"))
    files = splits[split_name]
    predictions_dir = Path(
        args.predictions_dir
        or config["evaluation"].get("predictions_dir")
        or find_latest_inference_dir("outputs/inference_results")
    )
    iou_thresholds = list(config["evaluation"].get("iou_thresholds", [0.1, 0.25, 0.5]))
    logger.info("Evaluating split=%s | videos=%d | predictions=%s", split_name, len(files), predictions_dir)

    per_video = []
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    segment_counts: dict[float, dict[int, dict[str, int]]] = {
        threshold: {} for threshold in iou_thresholds
    }
    videos_with_class: dict[int, set[str]] = {}
    gt_segment_counts: dict[int, int] = {}
    for path in files:
        record = load_sequence(path)
        if record.labels is None:
            raise ValueError(f"{record.video_id} has no labels")
        prediction = load_prediction(predictions_dir, record.video_id)
        if prediction.shape != record.labels.shape:
            raise ValueError(
                f"{record.video_id}: prediction {prediction.shape} != labels {record.labels.shape}"
            )
        matrix += confusion_matrix(prediction, record.labels)
        for segment in labels_to_segments(record.labels):
            if segment.label == BACKGROUND_ID:
                continue
            gt_segment_counts[segment.label] = gt_segment_counts.get(segment.label, 0) + 1
            videos_with_class.setdefault(segment.label, set()).add(record.video_id)
        for threshold in iou_thresholds:
            for class_id, counts in f1_at_iou_counts(
                prediction, record.labels, threshold
            ).items():
                bucket = segment_counts[threshold].setdefault(
                    class_id, {"tp": 0, "fp": 0, "fn": 0}
                )
                for key, value in counts.items():
                    bucket[key] += value
        metrics = evaluate_sequence(prediction, record.labels, iou_thresholds)
        metrics["video_id"] = record.video_id
        metrics["source"] = _metadata_group(record, "source")
        metrics["scene"] = _metadata_group(record, "scene")
        metrics["polarity"] = _metadata_group(record, "polarity")
        if str(metrics["polarity"]).lower() == "negative":
            metrics.update(negative_video_metrics(prediction, record.fps))
        per_video.append(metrics)
        logger.info(
            "%s | acc=%.3f | edit=%.2f | %s",
            record.video_id,
            metrics["frame_accuracy"],
            metrics["edit"],
            " | ".join(
                f"f1@{threshold}={metrics[f'f1@{threshold}']:.3f}"
                for threshold in iou_thresholds
            ),
        )

    numeric = [
        {
            key: value
            for key, value in item.items()
            if key in {"frame_accuracy", "edit"}
            or key.startswith("f1@")
        }
        for item in per_video
    ]
    summary = aggregate_metrics(numeric)
    summary.update(micro_segment_metrics(segment_counts))
    coverage = class_coverage(matrix, videos_with_class, gt_segment_counts)
    class_report = per_class_report(matrix, segment_counts, coverage)
    payload = {
        "summary": summary,
        "class_coverage": coverage,
        "per_class": class_report,
        "per_video": per_video,
        "grouped": grouped_metrics(per_video),
    }
    output_path = run_dir / "metrics.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_confusion_csv(run_dir / "confusion_matrix.csv", matrix)
    write_per_class_csv(run_dir / "per_class_metrics.csv", class_report)

    primary_threshold = max(iou_thresholds) if iou_thresholds else None
    logger.info("Per-class report (sorted by frame F1, worst first):")
    scored = [
        (name, entry)
        for name, entry in class_report.items()
        if entry["frame_f1"] is not None and entry["support_frames"]
    ]
    for name, entry in sorted(scored, key=lambda item: float(item[1]["frame_f1"])):
        extra = ""
        if primary_threshold is not None and f"segment_f1@{primary_threshold}" in entry:
            extra = f" | seg_f1@{primary_threshold}={entry[f'segment_f1@{primary_threshold}']:.3f}"
        logger.info(
            "  %-18s frames=%-7d segs=%-4d P=%.3f R=%.3f F1=%.3f%s | top_confusion=%s(%d)",
            name,
            entry["support_frames"],
            entry["gt_segments"],
            entry["frame_precision"],
            entry["frame_recall"],
            entry["frame_f1"],
            extra,
            entry["top_confusion"] or "-",
            entry["top_confusion_frames"],
        )
    missing = [name for name, item in coverage.items() if item["gt_frames"] == 0]
    if missing:
        logger.warning("Classes absent from this split: %s", ", ".join(missing))
    logger.info("Summary: %s", summary)
    logger.info("Wrote %s", output_path)
    logger.info("Wrote %s", run_dir / "confusion_matrix.csv")
    logger.info("Wrote %s", run_dir / "per_class_metrics.csv")


if __name__ == "__main__":
    main()
