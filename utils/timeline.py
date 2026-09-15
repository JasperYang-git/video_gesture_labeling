"""Timeline strip plots for comparing a prediction against ground truth at a glance."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.schema import BACKGROUND_ID, CLASS_NAMES, NUM_CLASSES

BACKGROUND_COLOR = (0.92, 0.92, 0.92)
ERROR_COLOR = (0.85, 0.12, 0.12)
MATCH_COLOR = (0.96, 0.96, 0.96)


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


def render_timeline(
    output_path: str | Path,
    prediction: np.ndarray,
    fps: float,
    labels: np.ndarray | None = None,
    title: str = "",
) -> Path:
    """Render a prediction (and optionally ground truth) as horizontal color bands."""
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

    colors = class_colors()
    rows = [("prediction", _label_strip(prediction, colors))]
    accuracy: float | None = None
    if labels is not None:
        rows.append(("ground truth", _label_strip(labels, colors)))
        mismatch = prediction != labels
        error_strip = np.tile(np.asarray(MATCH_COLOR, dtype=np.float32), (len(prediction), 1))
        error_strip[mismatch] = ERROR_COLOR
        rows.append(("errors", error_strip))
        accuracy = float(np.mean(~mismatch)) if len(prediction) else 0.0

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
    )
    handles = [
        Patch(facecolor=colors[class_id], edgecolor="0.6", label=CLASS_NAMES[class_id])
        for class_id in present
        if 0 <= class_id < NUM_CLASSES
    ]
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
    figure.suptitle(heading, fontsize=11)
    figure.tight_layout(rect=(0.0, 0.10, 1.0, 0.97))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=120)
    plt.close(figure)
    return output
