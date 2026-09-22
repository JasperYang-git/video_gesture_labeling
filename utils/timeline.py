"""Timeline strip plots for comparing a prediction against ground truth at a glance."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.schema import BACKGROUND_ID, CLASS_NAMES, NUM_CLASSES

BACKGROUND_COLOR = (0.92, 0.92, 0.92)
ERROR_COLOR = (0.85, 0.12, 0.12)
MATCH_COLOR = (0.96, 0.96, 0.96)
# Frames where MediaPipe lost the hand: the model only sees a frozen trajectory there,
# so they must stand out rather than blend into the grey confidence ramp.
LOST_TRACKING_COLOR = (0.95, 0.65, 0.15)


def class_colors() -> dict[int, tuple[float, float, float]]:
    """Fixed class-to-color map so strips from different videos stay comparable."""
    from matplotlib import colormaps

    palette = colormaps["tab20"].colors
    colors: dict[int, tuple[float, float, float]] = {}
    for class_id in range(NUM_CLASSES):
        if class_id == BACKGROUND_ID:
            colors[class_id] = BACKGROUND_COLOR
        else:
            colors[class_id] = tuple(palette[class_id % len(palette)])
    return colors


def _label_strip(
    labels: np.ndarray,
    colors: dict[int, tuple[float, float, float]],
) -> np.ndarray:
    strip = np.zeros((len(labels), 3), dtype=np.float32)
    for class_id, color in colors.items():
        strip[labels == class_id] = color
    return strip


def _tracking_strip(
    valid_mask: np.ndarray,
    tracking_quality: np.ndarray | None,
) -> np.ndarray:
    """Grey ramp for MediaPipe confidence, with lost frames painted in a warning color."""
    valid = np.asarray(valid_mask) > 0.5
    if tracking_quality is None:
        shade = np.where(valid, 0.35, 1.0).astype(np.float32)
    else:
        quality = np.clip(np.asarray(tracking_quality, dtype=np.float32), 0.0, 1.0)
        # High confidence renders dark, low confidence light, so a washed-out band
        # reads as "MediaPipe was unsure here" without needing the legend.
        shade = 0.85 - 0.6 * quality
    strip = np.repeat(shade[:, np.newaxis], 3, axis=1).astype(np.float32)
    strip[~valid] = LOST_TRACKING_COLOR
    return strip


def _longest_gap_seconds(valid_mask: np.ndarray, fps: float) -> float:
    longest = current = 0
    for is_valid in np.asarray(valid_mask) > 0.5:
        current = 0 if is_valid else current + 1
        longest = max(longest, current)
    return longest / fps


def render_timeline(
    output_path: str | Path,
    prediction: np.ndarray,
    fps: float,
    labels: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    tracking_quality: np.ndarray | None = None,
    filtered: np.ndarray | None = None,
    title: str = "",
) -> Path:
    """Render a prediction (and optionally ground truth and tracking quality) as bands.

    The tracking row sits directly under the prediction so that "the model got this
    stretch wrong" and "MediaPipe had no hand here" line up vertically, which is the
    whole point: it separates an extraction failure from a classification failure.

    ``filtered`` is the post-processed prediction. It gets its own row right below the
    raw one so a filter that removed real gestures is as obvious as one that cleaned up
    noise; accuracy is then reported for both.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if fps <= 0:
        raise ValueError("fps must be positive")
    if labels is not None and labels.shape != prediction.shape:
        raise ValueError(
            f"labels {labels.shape} does not match prediction {prediction.shape}"
        )
    if valid_mask is not None and len(valid_mask) != len(prediction):
        raise ValueError(
            f"valid_mask {len(valid_mask)} does not match prediction {len(prediction)}"
        )
    if tracking_quality is not None and len(tracking_quality) != len(prediction):
        raise ValueError(
            f"tracking_quality {len(tracking_quality)} does not match "
            f"prediction {len(prediction)}"
        )
    if filtered is not None and filtered.shape != prediction.shape:
        raise ValueError(
            f"filtered {filtered.shape} does not match prediction {prediction.shape}"
        )

    colors = class_colors()
    rows = [("prediction", _label_strip(prediction, colors))]
    if filtered is not None:
        rows.append(("filtered", _label_strip(filtered, colors)))
    detection_rate: float | None = None
    longest_gap: float | None = None
    if valid_mask is not None:
        rows.append(("tracking", _tracking_strip(valid_mask, tracking_quality)))
        detection_rate = (
            float(np.mean(np.asarray(valid_mask) > 0.5)) if len(valid_mask) else 0.0
        )
        longest_gap = _longest_gap_seconds(valid_mask, fps)
    accuracy: float | None = None
    filtered_accuracy: float | None = None
    delivered = filtered if filtered is not None else prediction
    if labels is not None:
        rows.append(("ground truth", _label_strip(labels, colors)))
        # Errors are scored against what actually ships, so a filter that helps or
        # hurts shows up directly in this row.
        mismatch = delivered != labels
        error_strip = np.tile(np.asarray(MATCH_COLOR, dtype=np.float32), (len(prediction), 1))
        error_strip[mismatch] = ERROR_COLOR
        rows.append(("errors", error_strip))
        accuracy = float(np.mean(prediction == labels)) if len(prediction) else 0.0
        if filtered is not None:
            filtered_accuracy = float(np.mean(~mismatch)) if len(prediction) else 0.0

    duration = len(prediction) / fps
    width = float(np.clip(duration / 6.0, 8.0, 40.0))
    figure, axes = plt.subplots(
        len(rows),
        1,
        figsize=(width, 0.9 * len(rows) + 1.6),
        sharex=True,
        squeeze=False,
    )
    for axis, (name, strip) in zip(axes[:, 0], rows):
        axis.imshow(
            strip[np.newaxis, :, :],
            aspect="auto",
            extent=(0.0, duration, 0.0, 1.0),
            interpolation="nearest",
        )
        axis.set_yticks([])
        axis.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
    axes[-1, 0].set_xlabel("time (s)")

    present = sorted(
        set(np.unique(prediction).tolist())
        | (set(np.unique(labels).tolist()) if labels is not None else set())
        | (set(np.unique(filtered).tolist()) if filtered is not None else set())
    )
    handles = [
        Patch(facecolor=colors[class_id], edgecolor="0.6", label=CLASS_NAMES[class_id])
        for class_id in present
        if 0 <= class_id < NUM_CLASSES
    ]
    if valid_mask is not None and not np.all(np.asarray(valid_mask) > 0.5):
        handles.append(
            Patch(facecolor=LOST_TRACKING_COLOR, edgecolor="0.6", label="hand not tracked")
        )
    if handles:
        figure.legend(
            handles=handles,
            loc="lower center",
            ncol=min(len(handles), 8),
            fontsize=8,
            frameon=False,
        )

    heading = title or "timeline"
    if accuracy is not None:
        heading = f"{heading} | frame accuracy {accuracy:.3f}"
        if filtered_accuracy is not None:
            heading = f"{heading} -> {filtered_accuracy:.3f} filtered"
    if detection_rate is not None:
        heading = f"{heading} | detection {detection_rate:.2f}"
        if longest_gap:
            heading = f"{heading} | longest gap {longest_gap:.1f}s"
    figure.suptitle(heading, fontsize=11)
    figure.tight_layout(rect=(0.0, 0.10, 1.0, 0.97))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=120)
    plt.close(figure)
    return output
