"""Experiment manifests built cheaply from audited sequence metadata."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from utils.preprocessing.audit import load_audit_rows


@dataclass(frozen=True)
class ManifestConfig:
    name: str = "all5"
    output_dir: str = "data/manifests"
    processed_dir: str = "data/processed"
    audit_path: str = "data/audit/quality_audit.csv"
    subject_splits_path: str = "data/manifests/subject_splits.json"
    seed: int = 42
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    sources: tuple[str, ...] = ()
    train_sources: tuple[str, ...] = ()
    test_sources: tuple[str, ...] = ()
    scenes: tuple[str, ...] = ()
    polarities: tuple[str, ...] = ("positive", "negative")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ManifestConfig":
        converted = dict(values)
        for key in ("sources", "train_sources", "test_sources", "scenes", "polarities"):
            converted[key] = tuple(converted.get(key) or ())
        return cls(**converted)


def _validate_ratios(config: ManifestConfig) -> None:
    ratios = (
        config.train_ratio,
        config.validation_ratio,
        config.test_ratio,
    )
    if any(value <= 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Manifest split ratios must be positive and sum to 1")


def build_subject_splits(
    subject_ids: list[str],
    config: ManifestConfig,
    overwrite: bool = False,
) -> dict[str, str]:
    _validate_ratios(config)
    path = Path(config.subject_splits_path).expanduser()
    if path.is_file() and not overwrite:
        payload = json.loads(path.read_text(encoding="utf-8"))
        mapping = {
            subject: split_name
            for split_name, subjects in payload["splits"].items()
            for subject in subjects
        }
        missing = sorted(set(subject_ids).difference(mapping))
        if missing:
            raise ValueError(
                f"Subject split registry misses {len(missing)} subjects. "
                "Regenerate it explicitly to preserve experiment comparability."
            )
        return mapping

    subjects = sorted(set(subject_ids))
    if len(subjects) < 3:
        raise ValueError("At least three subjects are required")
    random.Random(config.seed).shuffle(subjects)
    train_count = max(1, round(len(subjects) * config.train_ratio))
    validation_count = max(1, round(len(subjects) * config.validation_ratio))
    if train_count + validation_count >= len(subjects):
        train_count = len(subjects) - 2
        validation_count = 1
    split_subjects = {
        "train": subjects[:train_count],
        "validation": subjects[
            train_count : train_count + validation_count
        ],
        "test": subjects[train_count + validation_count :],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "seed": config.seed,
                "ratios": {
                    "train": config.train_ratio,
                    "validation": config.validation_ratio,
                    "test": config.test_ratio,
                },
                "splits": split_subjects,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        subject: split_name
        for split_name, split_values in split_subjects.items()
        for subject in split_values
    }


def build_experiment_manifest(
    config: ManifestConfig,
    overwrite_subject_splits: bool = False,
) -> Path:
    rows = [
        row
        for row in load_audit_rows(config.audit_path)
        if row["quality_passed"] and row.get("subject_id")
    ]
    if not rows:
        raise ValueError("Audit contains no quality-passed records with subject_id")
    subject_mapping = build_subject_splits(
        [str(row["subject_id"]) for row in rows],
        config,
        overwrite_subject_splits,
    )
    source_filter = set(config.sources)
    scene_filter = set(config.scenes)
    polarity_filter = set(config.polarities)
    train_sources = set(config.train_sources)
    test_sources = set(config.test_sources)

    selected: list[dict[str, Any]] = []
    split_paths = {"train": [], "validation": [], "test": []}
    processed_root = Path(config.processed_dir).expanduser()
    for row in rows:
        source = str(row["source"])
        if source_filter and source not in source_filter:
            continue
        if scene_filter and row.get("scene") not in scene_filter:
            continue
        if polarity_filter and row.get("polarity") not in polarity_filter:
            continue
        split_name = subject_mapping[str(row["subject_id"])]
        if split_name in {"train", "validation"} and train_sources:
            if source not in train_sources:
                continue
        if split_name == "test" and test_sources:
            if source not in test_sources:
                continue
        sequence_path = Path(str(row["sequence_path"]))
        try:
            relative_path = sequence_path.relative_to(processed_root)
        except ValueError:
            relative_path = Path(source) / sequence_path.name
        split_paths[split_name].append(relative_path.as_posix())
        selected.append(
            {
                "video_id": row["video_id"],
                "path": relative_path.as_posix(),
                "split": split_name,
                "source": source,
                "subject_id": row["subject_id"],
                "scene": row.get("scene", "unknown"),
                "scene_raw": row.get("scene_raw", ""),
                "polarity": row.get("polarity", "unknown"),
                "scene_category": row.get("scene_category", "unknown"),
                "hard_negative_eligible": bool(
                    row.get("hard_negative_eligible", False)
                ),
                "quality_passed": True,
                "preprocess_fingerprint": row.get(
                    "preprocess_fingerprint", ""
                ),
            }
        )
    if any(not values for values in split_paths.values()):
        counts = {key: len(value) for key, value in split_paths.items()}
        raise ValueError(f"Experiment filter produced an empty split: {counts}")

    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{config.name}.json"
    output.write_text(
        json.dumps(
            {
                "seed": config.seed,
                "processed_dir": str(processed_root),
                "split_ratios": {
                    "train": config.train_ratio,
                    "validation": config.validation_ratio,
                    "test": config.test_ratio,
                },
                "grouping": "subject",
                "experiment": asdict(config),
                "splits": split_paths,
                "records": selected,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output
