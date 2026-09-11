from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from utils.config import load_config
from utils.logger import create_run_directory, setup_logger
from utils.metrics import aggregate_metrics, evaluate_sequence, labels_to_segments
from utils.schema import BACKGROUND_ID, SequenceRecord, load_sequence
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
) -> dict[str, float | int]:
    action_segments = [
        segment
        for segment in labels_to_segments(prediction)
        if segment.label != BACKGROUND_ID
    ]
    duration_minutes = len(prediction) / fps / 60.0 if fps > 0 else 0.0
    return {
        "false_positive_frame_rate": (
            float(np.mean(prediction != BACKGROUND_ID)) if len(prediction) else 0.0
        ),
        "predicted_action_segments": len(action_segments),
        "false_actions_per_minute": (
            len(action_segments) / duration_minutes if duration_minutes > 0 else 0.0
        ),
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
    for path in files:
        record = load_sequence(path)
        if record.labels is None:
            raise ValueError(f"{record.video_id} has no labels")
        prediction = load_prediction(predictions_dir, record.video_id)
        if prediction.shape != record.labels.shape:
            raise ValueError(
                f"{record.video_id}: prediction {prediction.shape} != labels {record.labels.shape}"
            )
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
    payload = {
        "summary": summary,
        "per_video": per_video,
        "grouped": grouped_metrics(per_video),
    }
    output_path = run_dir / "metrics.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("Summary: %s", summary)
    logger.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
