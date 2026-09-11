from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from utils.config import load_config
from utils.mock_data import generate_mock_and_manifest
from utils.preprocessing.pipeline import (
    PreprocessConfig,
    discover_raw_videos,
    process_raw_videos,
)
from utils.schema import dump_mapping
from utils.schema import SequenceRecord, load_sequence, save_sequence
from utils.split import (
    load_manifest,
    split_grouped_files,
    split_video_files,
    write_manifest,
)
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
    parser.add_argument(
        "--raw-root",
        help=(
            "Training-data root. Overrides data.raw_root and processes real data "
            "even if mock.enabled is true"
        ),
    )
    return parser.parse_args()


def _load_identity_mapping(path: str | Path) -> dict[str, tuple[str, str]]:
    mapping_path = Path(path)
    if not mapping_path.is_file():
        return {}
    mapping: dict[str, tuple[str, str]] = {}
    with mapping_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            video_id = str(row.get("video_id", "")).strip()
            if video_id:
                mapping[video_id] = (
                    str(row.get("subject_id", "")).strip(),
                    str(row.get("session_id", "")).strip(),
                )
    return mapping


def _write_missing_subjects(path: str | Path, records: list[SequenceRecord]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["video_id", "subject_id", "session_id"])
        for record in records:
            writer.writerow([record.video_id, "", ""])
    return output


def _write_quality_report(path: str | Path, records: list[SequenceRecord]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "video_id",
        "quality_passed",
        "detection_rate",
        "min_class_id",
        "min_class_detection_rate",
        "longest_missing_seconds",
        "palm_outlier_rate",
        "trajectory_jump_rate",
        "per_class_detection_rate",
    ]
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            metadata = record.metadata
            writer.writerow(
                {
                    "video_id": record.video_id,
                    **{
                        field: metadata.get(field, "")
                        for field in fields
                        if field not in {"video_id", "per_class_detection_rate"}
                    },
                    "per_class_detection_rate": json.dumps(
                        metadata.get("per_class_detection_rate", {}),
                        sort_keys=True,
                    ),
                }
            )
    return output


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    data_config = config["data"]
    dump_mapping(data_config.get("mapping_path", "data/mapping.txt"))

    raw_root = args.raw_root or data_config["raw_root"]
    use_mock = bool(
        args.use_mock
        or (
            config.get("mock", {}).get("enabled", False)
            and args.raw_root is None
        )
    )
    processed_dir = Path(data_config["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    if use_mock:
        manifest_path = generate_mock_and_manifest(config)
        mock_records = [
            load_sequence(path)
            for files in load_manifest(manifest_path).values()
            for path in files
        ]
        _write_quality_report(processed_dir / "quality_audit_report.csv", mock_records)
        print(f"Wrote mock dataset and manifest: {manifest_path}")
        return

    preprocess_config = PreprocessConfig(**config["preprocessing"])
    items = discover_raw_videos(raw_root, preprocess_config.hand_side)
    if not items:
        raise FileNotFoundError(
            f"No sample directories containing both an MP4 and "
            f"'NOVA project/gestures.annotation~' were found under {raw_root} "
            f"for hand side {preprocess_config.hand_side}. "
            "Use --use-mock or enable mock.enabled to generate fake data."
        )
    print(f"Discovered {len(items)} labeled videos under {raw_root}")
    records = process_raw_videos(
        items,
        preprocess_config,
        processed_dir,
        overwrite=bool(data_config.get("overwrite", False)),
    )
    _write_quality_report(processed_dir / "quality_audit_report.csv", records)
    identity_mapping = _load_identity_mapping(
        data_config.get("subject_mapping_path", "data/subject_mapping.csv")
    )
    for record in records:
        subject_id, session_id = identity_mapping.get(record.video_id, ("", ""))
        record.subject_id = subject_id
        record.session_id = session_id
        save_sequence(processed_dir / f"{record.video_id}.npz", record)
    kept = [
        processed_dir / f"{record.video_id}.npz"
        for record in records
        if bool(record.metadata.get("quality_passed", False))
    ]
    if len(kept) < 3:
        raise ValueError("Need at least three quality-passed videos to create splits")
    kept_records = [
        record for record in records if bool(record.metadata.get("quality_passed", False))
    ]
    strategy = str(config.get("split", {}).get("strategy", "subject"))
    if strategy == "subject":
        missing = [record for record in kept_records if not record.subject_id]
        if missing:
            report = _write_missing_subjects(
                data_config.get(
                    "missing_subject_report",
                    "data/processed/missing_subject_mapping.csv",
                ),
                missing,
            )
            raise ValueError(
                "Subject-level split requires subject_id for every video. "
                f"Fill the generated mapping template: {report}"
            )
        split_files = split_grouped_files(
            kept,
            {record.video_id: record.subject_id for record in kept_records},
            float(data_config["train_ratio"]),
            float(data_config["validation_ratio"]),
            float(data_config["test_ratio"]),
            int(config["seed"]),
        )
    elif strategy == "video":
        split_files = split_video_files(
            kept,
            float(data_config["train_ratio"]),
            float(data_config["validation_ratio"]),
            float(data_config["test_ratio"]),
            int(config["seed"]),
        )
    else:
        raise ValueError("split.strategy must be 'subject' or 'video'")
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
        records=kept_records,
        grouping=strategy,
    )
    print(f"Processed {len(kept)} videos. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
