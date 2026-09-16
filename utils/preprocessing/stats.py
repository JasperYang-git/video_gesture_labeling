"""Per-source class and scene breakdown over assembled sequences.

Inventory summaries describe what was labelled on disk, at source FPS, including
samples the pipeline later discards. These statistics instead read the assembled
``.npz`` sequences, so the counts reflect the 15 FPS timeline and the class
mapping that training actually sees.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
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


def action_segment_class_ids(labels: np.ndarray) -> list[int]:
    """Class id of every maximal non-background run, in order."""
    return [
        int(segment.label)
        for segment in labels_to_segments(np.asarray(labels))
        if int(segment.label) != BACKGROUND_ID
    ]


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


def collect_group_stats(group: SourceGroup) -> GroupStats:
    """Count segments and scenes; reads headers so features stay untouched."""
    segment_counts: Counter[int] = Counter()
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
        segment_counts.update(action_segment_class_ids(header.labels))

    total = sum(segment_counts.values())
    classes = sorted(
        (
            ClassStats(
                class_id=class_id,
                name=class_id_to_name(class_id),
                segments=count,
                share=count / total if total else 0.0,
            )
            for class_id, count in segment_counts.items()
        ),
        key=lambda item: (-item.segments, item.class_id),
    )
    missing = [
        class_id_to_name(class_id)
        for class_id in range(len(CLASS_NAMES))
        if class_id != BACKGROUND_ID and class_id not in segment_counts
    ]
    scenes = sorted(
        (
            SceneStats(name=name, polarity=polarity, videos=count)
            for (name, polarity), count in scene_counts.items()
        ),
        key=lambda item: (item.polarity, -item.videos, item.name),
    )
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
    )


def render_group_stats(stats: GroupStats) -> str:
    lines = [f"===== {stats.label} ====="]
    if len(stats.sources) > 1:
        lines.append(f"  merged: {', '.join(stats.sources)}")
    lines.append(
        f"  videos {stats.videos:,} | subjects {stats.subjects} | "
        f"action segments {stats.segments:,}"
    )
    if stats.unlabeled_videos:
        lines.append(f"  WARNING: {stats.unlabeled_videos} videos carry no labels")
    lines.append("")
    lines.append(f"  {'class':<22}{'segments':>10}{'share':>9}")
    for item in stats.classes:
        lines.append(
            f"  {item.name:<22}{item.segments:>10,}{item.share:>8.1%}"
        )
    if stats.missing_classes:
        lines.append("")
        lines.append(
            f"  absent ({len(stats.missing_classes)}): "
            f"{', '.join(stats.missing_classes)}"
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
    fieldnames = ["group", "class_id", "name", "segments", "share"]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for group in groups:
            for item in group.classes:
                writer.writerow({"group": group.label, **asdict(item)})
    return json_path, csv_path
