from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils.schema import BACKGROUND_ID, NUM_CLASSES


@dataclass(frozen=True)
class AnnotationInterval:
    start_sec: float
    end_sec: float
    label: int
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.end_sec < self.start_sec:
            raise ValueError(
                f"Annotation end {self.end_sec} is earlier than start {self.start_sec}"
            )
        if self.label < 0 or self.label >= NUM_CLASSES:
            raise ValueError(f"Annotation label out of range: {self.label}")


def _normalize_label(raw_label: int) -> int:
    if raw_label == -1:
        return BACKGROUND_ID
    if raw_label < 0 or raw_label >= NUM_CLASSES:
        raise ValueError(f"Unsupported annotation label: {raw_label}")
    return raw_label


def parse_nova_annotation(path: str | Path) -> list[AnnotationInterval]:
    intervals: list[AnnotationInterval] = []
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(";") if part.strip() != ""]
        if len(parts) < 3:
            raise ValueError(f"{path}:{line_number} expected at least 3 fields")
        try:
            start_sec = float(parts[0])
            end_sec = float(parts[1])
            label = _normalize_label(int(parts[2]))
            confidence = float(parts[3]) if len(parts) > 3 else 1.0
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number} has invalid fields") from exc
        intervals.append(
            AnnotationInterval(
                start_sec=start_sec,
                end_sec=end_sec,
                label=label,
                confidence=confidence,
            )
        )
    return intervals


def labels_from_intervals(
    num_frames: int,
    fps: float,
    intervals: list[AnnotationInterval],
    background_id: int = BACKGROUND_ID,
) -> np.ndarray:
    if num_frames < 0:
        raise ValueError("num_frames must be non-negative")
    if fps <= 0:
        raise ValueError("fps must be positive")
    labels = np.full(num_frames, background_id, dtype=np.int64)
    for interval in intervals:
        start_index = int(np.floor(interval.start_sec * fps))
        end_index = int(np.ceil(interval.end_sec * fps))
        start_index = max(0, start_index)
        end_index = min(num_frames, max(start_index, end_index))
        if start_index < end_index:
            labels[start_index:end_index] = interval.label
    return labels


def infer_hand_side_from_folder(folder_name: str) -> str | None:
    parts = folder_name.split("_")
    if len(parts) < 3:
        return None
    side = parts[2].upper()
    if side in {"L", "R"}:
        return side
    return None
