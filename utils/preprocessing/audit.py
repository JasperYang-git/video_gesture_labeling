"""Independent quality gates over already assembled sequences."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from utils.preprocessing.assemble import AssembleConfig, sequence_output_path
from utils.schema import load_sequence


@dataclass(frozen=True)
class AuditConfig:
    output_dir: str = "data/audit"
    quality_threshold: float = 0.6
    min_class_detection_threshold: float = 0.0
    exclude_unknown_scenes: bool = True


def audit_inventory(
    entries: Iterable[Any],
    assemble_config: AssembleConfig,
    config: AuditConfig,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        path = sequence_output_path(
            assemble_config, entry.source, entry.video_id
        )
        if not path.is_file():
            continue
        try:
            record = load_sequence(path)
        except Exception as exc:
            rows.append(
                {
                    "video_id": entry.video_id,
                    "source": entry.source,
                    "subject_id": entry.subject_id,
                    "scene": entry.scene,
                    "polarity": entry.polarity,
                    "sequence_path": str(path),
                    "quality_passed": False,
                    "scene_valid": entry.polarity != "unknown",
                    "status": "load_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        metadata = record.metadata
        detection_rate = float(metadata.get("detection_rate", 0.0))
        min_class_rate = float(
            metadata.get("min_class_detection_rate", detection_rate)
        )
        scene_valid = entry.polarity != "unknown"
        passed = (
            detection_rate >= config.quality_threshold
            and min_class_rate >= config.min_class_detection_threshold
            and (scene_valid or not config.exclude_unknown_scenes)
        )
        rows.append(
            {
                "video_id": record.video_id,
                "source": entry.source,
                "subject_id": record.subject_id,
                "gender": metadata.get("gender", ""),
                "field3": metadata.get("field3", ""),
                "scene": metadata.get("scene", "unknown"),
                "scene_raw": metadata.get("scene_raw", ""),
                "polarity": metadata.get("polarity", "unknown"),
                "scene_category": metadata.get("scene_category", "unknown"),
                "hard_negative_eligible": bool(
                    metadata.get("hard_negative_eligible", False)
                ),
                "preprocess_fingerprint": metadata.get(
                    "preprocess_fingerprint", ""
                ),
                "track_fingerprint": metadata.get("track_fingerprint", ""),
                "sequence_path": str(path),
                "detection_rate": detection_rate,
                "min_class_id": metadata.get("min_class_id", ""),
                "min_class_detection_rate": min_class_rate,
                "longest_missing_seconds": metadata.get(
                    "longest_missing_seconds", ""
                ),
                "palm_outlier_rate": metadata.get("palm_outlier_rate", ""),
                "trajectory_jump_rate": metadata.get(
                    "trajectory_jump_rate", ""
                ),
                "scene_valid": scene_valid,
                "quality_passed": passed,
                "status": "passed" if passed else "rejected",
                "error": "",
            }
        )

    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "quality_audit.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    pass_path = output_dir / "pass_list.json"
    pass_path.write_text(
        json.dumps(
            {
                "quality_threshold": config.quality_threshold,
                "min_class_detection_threshold": (
                    config.min_class_detection_threshold
                ),
                "videos": [
                    row["video_id"]
                    for row in rows
                    if row.get("quality_passed")
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return csv_path, pass_path, rows


def load_audit_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).expanduser().open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["quality_passed"] = (
            str(row.get("quality_passed", "")).lower() == "true"
        )
        row["hard_negative_eligible"] = (
            str(row.get("hard_negative_eligible", "")).lower() == "true"
        )
    return rows
