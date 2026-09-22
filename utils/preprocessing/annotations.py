from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils.metrics import labels_to_segments
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


def intervals_from_prediction(
    prediction: np.ndarray,
    fps: float,
    logits: np.ndarray | None = None,
    background_id: int = BACKGROUND_ID,
) -> list[AnnotationInterval]:
    """Turn a frame-level prediction into NOVA intervals, dropping background.

    ``logits`` is the fused ``[num_classes, num_frames]`` array from
    :func:`utils.inference.predict_sequence`; when given, each interval carries the
    mean softmax probability of its predicted class so low-confidence guesses can be
    filtered before review.
    """
    from utils.postprocess import class_probabilities

    if fps <= 0:
        raise ValueError("fps must be positive")
    probabilities = None if logits is None else class_probabilities(logits)

    intervals: list[AnnotationInterval] = []
    for segment in labels_to_segments(prediction):
        if segment.label == background_id:
            continue
        confidence = 1.0
        if probabilities is not None:
            confidence = float(
                probabilities[segment.label, segment.start : segment.end].mean()
            )
        intervals.append(
            AnnotationInterval(
                start_sec=segment.start / fps,
                end_sec=segment.end / fps,
                label=segment.label,
                confidence=confidence,
            )
        )
    return intervals


TIMESTAMP_QUANTUM = 1e-4


def _timestamp_pair(start_sec: float, end_sec: float) -> tuple[float, float]:
    """Quantize an interval to 0.1 ms so it re-imports onto the same frames.

    :func:`labels_from_intervals` floors the start and ceils the end, so a timestamp
    sitting exactly on a frame boundary can spill into the neighbouring frame once
    float rounding is involved (``2.24 * 25`` is ``56.00000000000001``). Pulling both
    ends one quantum inwards keeps each timestamp strictly inside its own frame; 0.1 ms
    is two orders of magnitude smaller than a frame even at 60 fps.
    """
    start = math.ceil((start_sec + TIMESTAMP_QUANTUM) * 1e4) / 1e4
    end = math.floor((end_sec - TIMESTAMP_QUANTUM) * 1e4) / 1e4
    if end < start:
        start = end = math.floor(end_sec * 1e4) / 1e4
    return start, end


def write_nova_annotation(
    path: str | Path,
    intervals: list[AnnotationInterval],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for interval in intervals:
        start, end = _timestamp_pair(interval.start_sec, interval.end_sec)
        lines.append(
            f"{start:.4f};{end:.4f};{interval.label};{interval.confidence:.4f};"
        )
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return output


def infer_hand_side_from_folder(folder_name: str) -> str | None:
    parts = folder_name.split("_")
    if len(parts) < 3:
        return None
    side = parts[2].upper()
    if side in {"L", "R"}:
        return side
    return None
