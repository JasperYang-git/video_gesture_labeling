"""Optional inference-side filters applied on top of a raw frame-level prediction.

Everything here runs after the model and touches neither extraction nor assembly, so
none of it can cause train/test skew. All filters default to no-ops; they only kick in
when explicitly configured. The raw prediction is always kept alongside the filtered
one so that a filter making things worse stays visible instead of silently winning.

Evaluation deliberately does not use any of this: single-versus-double confusion is the
failure mode most worth seeing, and a class prior would hide exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from utils.metrics import labels_to_segments
from utils.schema import BACKGROUND_ID, NUM_CLASSES, class_name_to_id


def class_probabilities(logits: np.ndarray) -> np.ndarray:
    """Softmax over the class axis of a ``[num_classes, num_frames]`` array."""
    shifted = logits - logits.max(axis=0, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=0, keepdims=True)


def resolve_class_ids(values: Iterable[Any]) -> tuple[int, ...]:
    """Accept class names or ids, always including background."""
    resolved: set[int] = set()
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"Invalid class selector: {value!r}")
        if isinstance(value, (int, np.integer)):
            class_id = int(value)
            if class_id < 0 or class_id >= NUM_CLASSES:
                raise ValueError(f"Class id out of range: {class_id}")
        else:
            class_id = class_name_to_id(str(value).strip())
        resolved.add(class_id)
    if resolved:
        # Without background the masked argmax could never predict "nothing happening".
        resolved.add(BACKGROUND_ID)
    return tuple(sorted(resolved))


@dataclass(frozen=True)
class PostprocessConfig:
    allowed_classes: tuple[int, ...] = ()
    min_confidence: float = 0.0
    min_duration_sec: float = 0.0

    @classmethod
    def from_dict(cls, mapping: dict[str, Any] | None) -> "PostprocessConfig":
        values = dict(mapping or {})
        return cls(
            allowed_classes=resolve_class_ids(values.get("allowed_classes") or []),
            min_confidence=float(values.get("min_confidence", 0.0) or 0.0),
            min_duration_sec=float(values.get("min_duration_sec", 0.0) or 0.0),
        )

    @property
    def enabled(self) -> bool:
        return bool(
            self.allowed_classes or self.min_confidence > 0.0 or self.min_duration_sec > 0.0
        )


@dataclass
class PostprocessResult:
    prediction: np.ndarray
    stages: list[dict[str, Any]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return any(stage["changed_frames"] for stage in self.stages)

    def summary(self) -> dict[str, Any]:
        return {
            "changed": self.changed,
            "stages": self.stages,
        }


def masked_argmax(logits: np.ndarray, allowed_classes: Iterable[int]) -> np.ndarray:
    """Argmax restricted to a class subset.

    Masking beats dropping predictions afterwards: the model still gets to pick its
    best allowed class (a knock becomes a double-knock rather than a background hole),
    which keeps segments intact instead of fragmenting them.
    """
    allowed = sorted(set(int(item) for item in allowed_classes))
    if not allowed:
        return logits.argmax(axis=0).astype(np.int64)
    mask = np.full(logits.shape[0], -np.inf, dtype=np.float64)
    mask[allowed] = 0.0
    return (logits.astype(np.float64) + mask[:, np.newaxis]).argmax(axis=0).astype(np.int64)


def filter_by_confidence(
    prediction: np.ndarray,
    probabilities: np.ndarray,
    min_confidence: float,
    background_id: int = BACKGROUND_ID,
) -> np.ndarray:
    """Send action segments whose mean class probability is too low back to background."""
    output = prediction.copy()
    if min_confidence <= 0.0:
        return output
    for segment in labels_to_segments(prediction):
        if segment.label == background_id:
            continue
        confidence = float(
            probabilities[segment.label, segment.start : segment.end].mean()
        )
        if confidence < min_confidence:
            output[segment.start : segment.end] = background_id
    return output


def filter_by_duration(
    prediction: np.ndarray,
    min_frames: int,
    background_id: int = BACKGROUND_ID,
) -> np.ndarray:
    """Remove action segments too short to be a real gesture.

    A segment flanked by the same class on both sides is absorbed into it rather than
    blanked, because blanking a two-frame blip in the middle of one long gesture would
    split that gesture into two and make the over-segmentation worse, not better.
    """
    output = prediction.copy()
    if min_frames <= 1:
        return output
    segments = labels_to_segments(prediction)
    for index, segment in enumerate(segments):
        if segment.label == background_id or segment.length >= min_frames:
            continue
        previous = segments[index - 1].label if index > 0 else None
        following = segments[index + 1].label if index + 1 < len(segments) else None
        if previous is not None and previous == following and previous != background_id:
            output[segment.start : segment.end] = previous
        else:
            output[segment.start : segment.end] = background_id
    return output


def _action_segment_count(prediction: np.ndarray, background_id: int = BACKGROUND_ID) -> int:
    return sum(
        1 for segment in labels_to_segments(prediction) if segment.label != background_id
    )


def apply_postprocess(
    prediction: np.ndarray,
    logits: np.ndarray,
    fps: float,
    config: PostprocessConfig,
) -> PostprocessResult:
    """Run the configured filters in order, recording what each one changed."""
    result = PostprocessResult(prediction=prediction.copy())
    if not config.enabled:
        return result

    probabilities = class_probabilities(logits)
    current = prediction

    def record(name: str, updated: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
        result.stages.append(
            {
                "stage": name,
                **detail,
                "changed_frames": int(np.sum(updated != current)),
                "action_segments_before": _action_segment_count(current),
                "action_segments_after": _action_segment_count(updated),
            }
        )
        return updated

    if config.allowed_classes:
        current = record(
            "allowed_classes",
            masked_argmax(logits, config.allowed_classes),
            {"allowed_classes": list(config.allowed_classes)},
        )
    if config.min_confidence > 0.0:
        current = record(
            "min_confidence",
            filter_by_confidence(current, probabilities, config.min_confidence),
            {"min_confidence": config.min_confidence},
        )
    if config.min_duration_sec > 0.0:
        min_frames = max(1, int(round(config.min_duration_sec * fps)))
        current = record(
            "min_duration",
            filter_by_duration(current, min_frames),
            {"min_duration_sec": config.min_duration_sec, "min_frames": min_frames},
        )

    result.prediction = current
    return result
