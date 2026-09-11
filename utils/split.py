from __future__ import annotations

import json
import math
import random
import hashlib
from pathlib import Path
from typing import Iterable, Sequence

from utils.schema import SequenceRecord, load_sequence


SPLIT_NAMES = ("train", "validation", "test")


def discover_sequence_files(processed_dir: str | Path) -> list[Path]:
    root = Path(processed_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Processed directory does not exist: {root}")
    files = sorted(path for path in root.glob("*.npz") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No sequence .npz files found in {root}")
    return files


def split_video_files(
    files: Sequence[Path],
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[Path]]:
    _validate_ratios(train_ratio, validation_ratio, test_ratio)
    if len(files) < 3:
        raise ValueError("At least three videos are required for a train/val/test split")

    shuffled = list(files)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    counts = _split_counts(len(shuffled), train_ratio, validation_ratio, test_ratio)
    cursor = 0
    assigned: dict[str, list[Path]] = {}
    for name, count in zip(SPLIT_NAMES, counts):
        assigned[name] = sorted(shuffled[cursor : cursor + count])
        cursor += count
    return assigned


def split_grouped_files(
    files: Sequence[Path],
    group_ids: dict[str, str],
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[Path]]:
    """Split whole groups while keeping every subject in exactly one split."""
    _validate_ratios(train_ratio, validation_ratio, test_ratio)
    grouped: dict[str, list[Path]] = {}
    for path in files:
        group_id = group_ids.get(path.stem, "").strip()
        if not group_id:
            raise ValueError(f"Missing subject_id for video '{path.stem}'")
        grouped.setdefault(group_id, []).append(path)
    if len(grouped) < 3:
        raise ValueError("At least three subjects are required for subject-level splits")

    rng = random.Random(seed)
    groups = list(grouped.items())
    rng.shuffle(groups)
    groups.sort(key=lambda item: len(item[1]), reverse=True)
    target = {
        "train": len(files) * train_ratio,
        "validation": len(files) * validation_ratio,
        "test": len(files) * test_ratio,
    }
    assigned: dict[str, list[Path]] = {name: [] for name in SPLIT_NAMES}
    for index, (_, group_files) in enumerate(groups):
        if index < len(SPLIT_NAMES):
            destination = SPLIT_NAMES[index]
        else:
            destination = min(
                SPLIT_NAMES,
                key=lambda name: len(assigned[name]) / max(target[name], 1e-9),
            )
        assigned[destination].extend(group_files)
    return {name: sorted(paths) for name, paths in assigned.items()}


def _validate_ratios(
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> None:
    ratios = {
        "train_ratio": train_ratio,
        "validation_ratio": validation_ratio,
        "test_ratio": test_ratio,
    }
    invalid = [name for name, value in ratios.items() if not 0.0 < value < 1.0]
    if invalid:
        raise ValueError(f"{', '.join(invalid)} must be between 0 and 1")
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("split ratios must sum to 1")


def _split_counts(
    total: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    raw = [
        max(1, int(round(total * train_ratio))),
        max(1, int(round(total * validation_ratio))),
        max(1, int(round(total * test_ratio))),
    ]
    while sum(raw) > total:
        largest = max(range(3), key=lambda index: raw[index])
        if raw[largest] > 1:
            raw[largest] -= 1
        else:
            break
    while sum(raw) < total:
        raw[0] += 1
    return raw[0], raw[1], raw[2]


def write_manifest(
    path: str | Path,
    split_files: dict[str, Iterable[Path]],
    seed: int,
    ratios: dict[str, float],
    processed_dir: str | Path,
    records: Sequence[SequenceRecord] | None = None,
    grouping: str = "video",
) -> Path:
    root = Path(processed_dir)
    record_by_id = {record.video_id: record for record in records or []}
    record_entries = []
    for split_name in SPLIT_NAMES:
        for item in split_files[split_name]:
            record = record_by_id.get(Path(item).stem)
            record_entries.append(
                {
                    "video_id": Path(item).stem,
                    "path": str(Path(item).name),
                    "split": split_name,
                    "subject_id": record.subject_id if record else "",
                    "session_id": record.session_id if record else "",
                    "quality_passed": bool(record.metadata.get("quality_passed", True))
                    if record
                    else True,
                    "preprocess_fingerprint": str(
                        record.metadata.get("preprocess_fingerprint", "")
                    )
                    if record
                    else "",
                }
            )
    payload = {
        "seed": seed,
        "processed_dir": str(root),
        "split_ratios": ratios,
        "grouping": grouping,
        "splits": {
            name: [str(Path(item).name) for item in split_files[name]]
            for name in SPLIT_NAMES
        },
        "records": record_entries,
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def manifest_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manifest(path: str | Path) -> dict[str, list[Path]]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    processed_dir = Path(payload.get("processed_dir", manifest_path.parent))
    splits: dict[str, list[Path]] = {}
    for name in SPLIT_NAMES:
        files = [processed_dir / item for item in payload["splits"][name]]
        missing = [str(item) for item in files if not item.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Manifest {manifest_path} references missing files: {missing}"
            )
        splits[name] = files
    _assert_no_leakage(splits)
    return splits


def _assert_no_leakage(splits: dict[str, list[Path]]) -> None:
    seen: dict[str, str] = {}
    for split_name, files in splits.items():
        for path in files:
            video_id = path.stem
            if video_id in seen:
                raise ValueError(
                    f"Video '{video_id}' appears in both {seen[video_id]} and {split_name}"
                )
            seen[video_id] = split_name


def load_split_records(files: Sequence[Path]):
    return [load_sequence(path) for path in files]
