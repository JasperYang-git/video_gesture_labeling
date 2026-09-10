from __future__ import annotations

import argparse
from pathlib import Path

from utils.config import load_config
from utils.mock_data import generate_mock_and_manifest
from utils.preprocessing.pipeline import (
    PreprocessConfig,
    discover_raw_videos,
    process_raw_videos,
)
from utils.schema import dump_mapping
from utils.split import split_video_files, write_manifest
from utils.trainer import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare gesture sequences from mock data or raw videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config/config_prepare.yaml",
        help="Path to the data-preparation YAML",
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Generate mock sequences even if mock.enabled is false",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    data_config = config["data"]
    dump_mapping(data_config.get("mapping_path", "data/mapping.txt"))

    use_mock = bool(args.use_mock or config.get("mock", {}).get("enabled", False))
    processed_dir = Path(data_config["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    if use_mock:
        manifest_path = generate_mock_and_manifest(config)
        print(f"Wrote mock dataset and manifest: {manifest_path}")
        return

    preprocess_config = PreprocessConfig(**config["preprocessing"])
    items = discover_raw_videos(data_config["raw_root"], preprocess_config.hand_side)
    if not items:
        raise FileNotFoundError(
            f"No raw videos found under {data_config['raw_root']}. "
            "Use --use-mock or enable mock.enabled to generate fake data."
        )
    records = process_raw_videos(
        items,
        preprocess_config,
        processed_dir,
        overwrite=bool(data_config.get("overwrite", False)),
    )
    kept = [
        processed_dir / f"{record.video_id}.npz"
        for record in records
        if bool(record.metadata.get("quality_passed", True))
    ]
    if len(kept) < 3:
        raise ValueError("Need at least three quality-passed videos to create splits")
    split_files = split_video_files(
        kept,
        float(data_config["train_ratio"]),
        float(data_config["validation_ratio"]),
        float(data_config["test_ratio"]),
        int(config["seed"]),
    )
    manifest_path = write_manifest(
        data_config["manifest_path"],
        split_files,
        seed=int(config["seed"]),
        ratios={
            "train": float(data_config["train_ratio"]),
            "validation": float(data_config["validation_ratio"]),
            "test": float(data_config["test_ratio"]),
        },
        processed_dir=processed_dir,
    )
    print(f"Processed {len(kept)} videos. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
