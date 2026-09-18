"""Per-source class and scene breakdown over assembled sequences.

Inventory summaries describe what was labelled on disk, at source FPS, including
samples the pipeline later discards. These statistics instead read the assembled
``.npz`` sequences, so the counts reflect the 15 FPS timeline and the class
mapping that training actually sees.

Segment *lengths* are reported alongside the counts because they decide what a
training window can hold. ``_dynamic_action_windows_for_record`` draws window
starts from ``[segment_end - window_size, segment_start]``, so a segment longer
than the window falls back to a single truncated window, and a segment merely
close to the window length leaves too few distinct starts for
``action_windows_per_segment`` to produce the requested augmentation.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from utils.metrics import labels_to_segments
from utils.schema import (
    BACKGROUND_ID,
    CLASS_NAMES,
    class_id_to_name,
    load_sequence_header,
)


DEFAULT_WINDOW_SIZE = 60
GAP_QUANTILES = (0.05, 0.25, 0.5)


@dataclass(frozen=True)
class SourceGroup:
    label: str
    sources: tuple[str, ...]
    paths: tuple[Path, ...]


@dataclass(frozen=True)
class ClassStats:
    class_id: int
    name: str
    segments: int
    share: float
    frames_total: int
    mean_frames: float
    p50_frames: int
    p90_frames: int
    p95_frames: int
    p99_frames: int
    max_frames: int
    p50_sec: float
    p95_sec: float
    over_window: int
    over_window_share: float
    starts_p50: int


@dataclass(frozen=True)
class SceneStats:
    name: str
    polarity: str
    videos: int


@dataclass(frozen=True)
class GroupStats:
    label: str
    sources: list[str]
    videos: int
    unlabeled_videos: int
    subjects: int
    segments: int
    polarities: dict[str, int]
    scenes: list[SceneStats]
    classes: list[ClassStats]
    missing_classes: list[str]
    window_size: int = DEFAULT_WINDOW_SIZE
    gap_percentiles: dict[str, int] = field(default_factory=dict)


def action_segment_spans(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """``(class_id, start, end)`` of every maximal non-background run, in order.

    End is exclusive, so the length is ``end - start``. Runs of two different
    actions that touch stay separate, matching the training-time segmentation.
    """
    return [
        (int(segment.label), int(segment.start), int(segment.end))
        for segment in labels_to_segments(np.asarray(labels))
        if int(segment.label) != BACKGROUND_ID
    ]


def action_segment_class_ids(labels: np.ndarray) -> list[int]:
    """Class id of every maximal non-background run, in order."""
    return [class_id for class_id, _, _ in action_segment_spans(labels)]


def nearest_rank(values: Sequence[float], quantile: float) -> float:
    """Nearest-rank percentile of already-sorted ``values``.

    Plain indexing rather than ``np.percentile`` keeps frame counts integral and
    keeps the result independent of the numpy version's default interpolation.
    """
    if not values:
        return 0.0
    index = min(max(int(np.ceil(quantile * len(values))) - 1, 0), len(values) - 1)
    return values[index]


def discover_sources(processed_dir: str | Path) -> dict[str, list[Path]]:
    root = Path(processed_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Processed directory does not exist: {root}")
    found: dict[str, list[Path]] = {}
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        files = sorted(directory.glob("*.npz"))
        if files:
            found[directory.name] = files
    if not found:
        # Mock and legacy layouts keep sequences directly under the root.
        loose = sorted(root.glob("*.npz"))
        if loose:
            found[root.name] = loose
    return found


def build_groups(
    processed_dir: str | Path,
    sources: Sequence[str] | None = None,
) -> list[SourceGroup]:
    """Without ``sources`` every source is its own group; with ``sources`` the
    listed ones are merged into a single group."""
    found = discover_sources(processed_dir)
    if not found:
        raise FileNotFoundError(
            f"No assembled sequences under {processed_dir}; "
            "run `python prepare_data.py assemble` first"
        )
    if not sources:
        return [
            SourceGroup(name, (name,), tuple(paths))
            for name, paths in found.items()
        ]
    selected = tuple(dict.fromkeys(sources))
    missing = [name for name in selected if name not in found]
    if missing:
        raise FileNotFoundError(
            f"No assembled sequences for {missing}; "
            f"available under {processed_dir}: {sorted(found)}"
        )
    paths = tuple(path for name in selected for path in found[name])
    return [SourceGroup("+".join(selected), selected, paths)]


def _class_stats(
    class_id: int,
    lengths: list[int],
    durations: list[float],
    total_segments: int,
    window_size: int,
) -> ClassStats:
    lengths = sorted(lengths)
    durations = sorted(durations)
    count = len(lengths)
    p50_frames = int(nearest_rank(lengths, 0.5))
    over_window = sum(1 for length in lengths if length > window_size)
    return ClassStats(
        class_id=class_id,
        name=class_id_to_name(class_id),
        segments=count,
        share=count / total_segments if total_segments else 0.0,
        frames_total=sum(lengths),
        mean_frames=sum(lengths) / count if count else 0.0,
        p50_frames=p50_frames,
        p90_frames=int(nearest_rank(lengths, 0.9)),
        p95_frames=int(nearest_rank(lengths, 0.95)),
        p99_frames=int(nearest_rank(lengths, 0.99)),
        max_frames=lengths[-1] if lengths else 0,
        p50_sec=float(nearest_rank(durations, 0.5)),
        p95_sec=float(nearest_rank(durations, 0.95)),
        over_window=over_window,
        over_window_share=over_window / count if count else 0.0,
        starts_p50=max(window_size - p50_frames + 1, 0),
    )


def collect_group_stats(
    group: SourceGroup,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> GroupStats:
    """Count segments, lengths and scenes; reads headers so features stay untouched."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    lengths_by_class: dict[int, list[int]] = defaultdict(list)
    durations_by_class: dict[int, list[float]] = defaultdict(list)
    gaps: list[int] = []
    scene_counts: Counter[tuple[str, str]] = Counter()
    polarities: Counter[str] = Counter()
    subjects: set[str] = set()
    videos = 0
    unlabeled = 0

    for path in group.paths:
        header = load_sequence_header(path)
        videos += 1
        polarity = header.polarity or "unknown"
        scene_counts[(header.scene or "unknown", polarity)] += 1
        polarities[polarity] += 1
        if header.subject_id:
            subjects.add(str(header.subject_id))
        if header.labels is None:
            unlabeled += 1
            continue
        spans = action_segment_spans(header.labels)
        for class_id, start, end in spans:
            length = end - start
            lengths_by_class[class_id].append(length)
            durations_by_class[class_id].append(
                length / header.fps if header.fps > 0 else 0.0
            )
        for (_, _, end), (_, next_start, _) in zip(spans, spans[1:]):
            gaps.append(next_start - end)

    total = sum(len(item) for item in lengths_by_class.values())
    classes = sorted(
        (
            _class_stats(
                class_id,
                lengths,
                durations_by_class[class_id],
                total,
                window_size,
            )
            for class_id, lengths in lengths_by_class.items()
        ),
        key=lambda item: (-item.segments, item.class_id),
    )
    missing = [
        class_id_to_name(class_id)
        for class_id in range(len(CLASS_NAMES))
        if class_id != BACKGROUND_ID and class_id not in lengths_by_class
    ]
    scenes = sorted(
        (
            SceneStats(name=name, polarity=polarity, videos=count)
            for (name, polarity), count in scene_counts.items()
        ),
        key=lambda item: (item.polarity, -item.videos, item.name),
    )
    gaps.sort()
    return GroupStats(
        label=group.label,
        sources=list(group.sources),
        videos=videos,
        unlabeled_videos=unlabeled,
        subjects=len(subjects),
        segments=total,
        polarities=dict(sorted(polarities.items())),
        scenes=scenes,
        classes=classes,
        missing_classes=missing,
        window_size=window_size,
        gap_percentiles={
            f"p{int(quantile * 100):02d}": int(nearest_rank(gaps, quantile))
            for quantile in GAP_QUANTILES
        }
        if gaps
        else {},
    )


def render_group_stats(stats: GroupStats) -> str:
    lines = [f"===== {stats.label} ====="]
    if len(stats.sources) > 1:
        lines.append(f"  merged: {', '.join(stats.sources)}")
    lines.append(
        f"  videos {stats.videos:,} | subjects {stats.subjects} | "
        f"action segments {stats.segments:,} | window {stats.window_size}"
    )
    if stats.unlabeled_videos:
        lines.append(f"  WARNING: {stats.unlabeled_videos} videos carry no labels")
    lines.append("")
    lines.append(
        f"  {'class':<22}{'segments':>10}{'share':>8}"
        f"{'p50f':>8}{'p95f':>8}{'maxf':>8}{'p95s':>8}{'>win':>8}{'starts50':>10}"
    )
    for item in stats.classes:
        lines.append(
            f"  {item.name:<22}{item.segments:>10,}{item.share:>8.1%}"
            f"{item.p50_frames:>8}{item.p95_frames:>8}{item.max_frames:>8}"
            f"{item.p95_sec:>7.2f}s{item.over_window_share:>8.1%}"
            f"{item.starts_p50:>10}"
        )
    if stats.missing_classes:
        lines.append("")
        lines.append(
            f"  absent ({len(stats.missing_classes)}): "
            f"{', '.join(stats.missing_classes)}"
        )
    if stats.gap_percentiles:
        lines.append("")
        lines.append(
            "  gap to next action (frames): "
            + ", ".join(
                f"{name} {value:,}" for name, value in stats.gap_percentiles.items()
            )
        )
    lines.append("")
    lines.append(
        "  polarity: "
        + ", ".join(f"{name} {count:,}" for name, count in stats.polarities.items())
    )
    lines.append("")
    lines.append(f"  {f'scene ({len(stats.scenes)})':<38}{'videos':>10}")
    for scene in stats.scenes:
        lines.append(f"  {f'{scene.name} ({scene.polarity})':<38}{scene.videos:>10,}")
    return "\n".join(lines)


def write_stats_outputs(
    output_dir: str | Path,
    groups: Sequence[GroupStats],
) -> tuple[Path, Path]:
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "dataset_stats.json"
    json_path.write_text(
        json.dumps(
            {group.label: asdict(group) for group in groups},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    csv_path = root / "class_stats.csv"
    fieldnames = [
        "group",
        "class_id",
        "name",
        "segments",
        "share",
        "frames_total",
        "mean_frames",
        "p50_frames",
        "p90_frames",
        "p95_frames",
        "p99_frames",
        "max_frames",
        "p50_sec",
        "p95_sec",
        "over_window",
        "over_window_share",
        "starts_p50",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for group in groups:
            for item in group.classes:
                writer.writerow({"group": group.label, **asdict(item)})
    return json_path, csv_path
