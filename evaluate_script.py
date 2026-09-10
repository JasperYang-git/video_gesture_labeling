from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from utils.config import load_config
from utils.logger import create_run_directory, setup_logger
from utils.metrics import aggregate_metrics, evaluate_sequence
from utils.schema import load_sequence
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
        {key: value for key, value in item.items() if key != "video_id"}
        for item in per_video
    ]
    summary = aggregate_metrics(numeric)
    payload = {"summary": summary, "per_video": per_video}
    output_path = run_dir / "metrics.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("Summary: %s", summary)
    logger.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
