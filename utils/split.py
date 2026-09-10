from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterable, Sequence

from utils.schema import load_sequence


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
) -> Path:
    root = Path(processed_dir)
    payload = {
        "seed": seed,
        "processed_dir": str(root),
        "split_ratios": ratios,
        "splits": {
            name: [str(Path(item).name) for item in split_files[name]]
            for name in SPLIT_NAMES
        },
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


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
