from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from utils.config import load_config
from utils.inference import (
    load_checkpoint_model,
    predict_sequence,
    validate_record_compatibility,
    write_prediction_outputs,
)
from utils.logger import create_run_directory, setup_logger
from utils.schema import load_sequence
from utils.split import load_manifest
from utils.trainer import resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-video gesture inference")
    parser.add_argument(
        "--config",
        default="config/config_infer.yaml",
        help="Path to inference YAML",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional checkpoint override",
    )
    return parser.parse_args()


def resolve_input_files(data_config: dict) -> list[Path]:
    explicit = [Path(item) for item in data_config.get("input_paths") or []]
    if explicit:
        files: list[Path] = []
        for path in explicit:
            if path.is_file() and path.suffix == ".npz":
                files.append(path)
            elif path.is_dir():
                files.extend(sorted(path.glob("*.npz")))
        if not files:
            raise FileNotFoundError("No .npz sequences found in data.input_paths")
        return files
    splits = load_manifest(data_config["manifest_path"])
    split_name = str(data_config.get("split", "test"))
    if split_name not in splits:
        raise ValueError(f"Unknown split '{split_name}'")
    return splits[split_name]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    run_dir = create_run_directory(config["output"]["root_dir"])
    logger = setup_logger("inference", run_dir / "inference.log")
    shutil.copy2(args.config, run_dir / "config_infer.yaml")

    device = resolve_device(str(config["device"]))
    model_path = args.model_path or config["inference"]["model_path"]
    model, checkpoint = load_checkpoint_model(model_path, device)
    data_config = config["data"]
    window_size = int(data_config.get("window_size", checkpoint.get("window_size", 60)))
    stride = int(data_config.get("stride", checkpoint.get("stride", 12)))
    files = resolve_input_files(data_config)
    logger.info("Run directory: %s", run_dir)
    logger.info("Device: %s; model: %s", device, model_path)
    logger.info("Videos: %d | window_size=%d | stride=%d", len(files), window_size, stride)

    index_payload = []
    for path in files:
        record = load_sequence(path)
        validate_record_compatibility(record, checkpoint)
        prediction, logits = predict_sequence(
            model,
            record,
            device,
            window_size,
            stride,
        )
        written = write_prediction_outputs(
            run_dir,
            record,
            prediction,
            logits,
            save_frame_csv=bool(config["inference"].get("save_frame_csv", True)),
            save_frame_npy=bool(config["inference"].get("save_frame_npy", True)),
            save_segments=bool(config["inference"].get("save_segments", True)),
        )
        if config["inference"].get("log_every_video", True):
            logger.info(
                "Predicted %s | frames=%d | outputs=%s",
                record.video_id,
                record.num_frames,
                {key: str(value) for key, value in written.items()},
            )
        index_payload.append(
            {
                "video_id": record.video_id,
                "source": str(path),
                "outputs": {key: str(value) for key, value in written.items()},
            }
        )

    index_path = run_dir / "predictions.json"
    index_path.write_text(json.dumps(index_payload, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote prediction index: %s", index_path)


if __name__ == "__main__":
    main()
