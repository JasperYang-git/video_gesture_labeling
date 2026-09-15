from __future__ import annotations

import csv
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar

import yaml


DEFAULT_TAXONOMY_PATH = Path(__file__).resolve().parents[2] / "config" / "scene_taxonomy.yaml"
ANNOTATION_DIRECTORY_NAME = "NOVA project"
ANNOTATION_FILE_NAME = "gestures.annotation~"
ANNOTATION_RELATIVE_PATH = Path(ANNOTATION_DIRECTORY_NAME) / ANNOTATION_FILE_NAME
# Written next to a machine-generated annotation so later runs can tell their own
# output apart from a hand annotation instead of scoring predictions against them.
PREDICTION_MARKER_NAME = ".gestures.predicted.json"
DEFAULT_SCAN_WORKERS = 16

_Item = TypeVar("_Item")
_Result = TypeVar("_Result")


@dataclass(frozen=True)
class InventoryEntry:
    source: str
    video_id: str
    directory_path: str
    video_path: str | None
    annotation_path: str | None
    subject_id: str | None
    gender: str | None
    field3: str | None
    scene_raw: str | None
    scene: str | None
    polarity: str | None
    status: str
    scene_category: str = "unknown"
    hard_negative_eligible: bool = False
    taxonomy_version: int = 1
    video_size: int | None = None
    video_mtime_ns: int | None = None
    # Only read back from inventories written before hashing moved into
    # assemble, which computes it itself; scanning never populates it.
    annotation_sha256: str | None = None
    error: str | None = None


def normalize_scene(value: str) -> str:
    """Apply only the normalization allowed by the inventory contract."""
    return value.strip().lower().replace("_", "-")


def load_taxonomy(path: str | Path = DEFAULT_TAXONOMY_PATH) -> dict[str, Any]:
    taxonomy_path = Path(path).expanduser()
    payload = yaml.safe_load(taxonomy_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Taxonomy must be a mapping: {taxonomy_path}")
    exact = payload.get("exact")
    aliases = payload.get("aliases")
    if not isinstance(exact, dict) or not isinstance(aliases, dict):
        raise ValueError("Taxonomy requires mapping-valued 'exact' and 'aliases'")
    return payload


def resolve_scene(
    scene_raw: str,
    taxonomy: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    """Resolve an exact scene or explicit alias; never infer from prefixes."""
    normalized = normalize_scene(scene_raw)
    exact = taxonomy.get("exact", {})
    aliases = taxonomy.get("aliases", {})

    exact_by_normalized = {normalize_scene(str(key)): value for key, value in exact.items()}
    alias_by_normalized = {
        normalize_scene(str(key)): normalize_scene(str(value))
        for key, value in aliases.items()
    }
    target = alias_by_normalized.get(normalized, normalized)
    definition = exact_by_normalized.get(target)
    if definition is None:
        return None, None
    if isinstance(definition, str):
        return target, definition
    if not isinstance(definition, Mapping):
        raise ValueError(f"Invalid taxonomy definition for scene {target!r}")
    canonical = normalize_scene(str(definition.get("canonical", target)))
    polarity = definition.get("polarity")
    return canonical, str(polarity) if polarity is not None else None


def resolve_scene_details(
    scene_raw: str,
    taxonomy: Mapping[str, Any],
) -> tuple[str | None, str | None, str, bool]:
    scene, polarity = resolve_scene(scene_raw, taxonomy)
    if scene is None:
        return None, None, "unknown", False
    exact = {
        normalize_scene(str(key)): value
        for key, value in taxonomy.get("exact", {}).items()
    }
    aliases = {
        normalize_scene(str(key)): normalize_scene(str(value))
        for key, value in taxonomy.get("aliases", {}).items()
    }
    definition = exact[aliases.get(normalize_scene(scene_raw), normalize_scene(scene_raw))]
    if isinstance(definition, Mapping):
        category = str(definition.get("category", "unknown"))
        hard_negative = bool(definition.get("hard_negative", False))
    else:
        category = "posture" if polarity == "positive" else "confounder"
        hard_negative = polarity == "negative"
    return scene, polarity, category, hard_negative


def _parse_directory_name(
    basename: str,
) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    parts = basename.split("_")
    if len(parts) < 4 or any(not part for part in parts[:4]):
        return None, None, None, None, "invalid_name"
    # Fields after scene are acquisition metadata and deliberately ignored.
    return parts[0], parts[1], parts[2], parts[3], None


def _map_threaded(
    function: Callable[[_Item], _Result],
    items: Iterable[_Item],
    max_workers: int,
) -> list[_Result]:
    """Fan out latency-bound filesystem probes while preserving input order."""
    item_list = list(items)
    if max_workers <= 1 or len(item_list) <= 1:
        return [function(item) for item in item_list]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(item_list))) as executor:
        return list(executor.map(function, item_list))


def _io_error_text(exc: OSError) -> str:
    reason = exc.strerror or type(exc).__name__
    return f"io_error:{reason.replace(';', ',')}"


@dataclass(frozen=True)
class _VideoFile:
    path: Path
    size: int | None
    mtime_ns: int | None


@dataclass(frozen=True)
class _DirectoryListing:
    subdirectories: tuple[Path, ...] = ()
    linked_subdirectories: tuple[Path, ...] = ()
    videos: tuple[_VideoFile, ...] = ()
    error: str | None = None


def _video_file(child: os.DirEntry) -> _VideoFile:
    try:
        stat_result = child.stat()
    except OSError:
        return _VideoFile(Path(child.path), None, None)
    return _VideoFile(
        Path(child.path),
        int(stat_result.st_size),
        int(stat_result.st_mtime_ns),
    )


def _list_directory(directory: Path) -> _DirectoryListing:
    """Read one directory once, keeping everything later decisions need."""
    subdirectories: list[Path] = []
    linked_subdirectories: list[Path] = []
    videos: list[_VideoFile] = []
    try:
        with os.scandir(directory) as scan:
            for child in scan:
                try:
                    if child.is_dir():
                        target = (
                            linked_subdirectories
                            if child.is_symlink()
                            else subdirectories
                        )
                        target.append(Path(child.path))
                    elif child.is_file() and child.name.lower().endswith(".mp4"):
                        videos.append(_video_file(child))
                except OSError:
                    continue
    except OSError as exc:
        return _DirectoryListing(error=_io_error_text(exc))
    return _DirectoryListing(
        subdirectories=tuple(subdirectories),
        linked_subdirectories=tuple(linked_subdirectories),
        videos=tuple(sorted(videos, key=lambda video: video.path.name)),
    )


def _walk_source(
    source_root: Path,
    max_workers: int,
) -> dict[Path, _DirectoryListing]:
    """List every directory below source_root exactly once, level by level.

    Symlinked directories are listed but not descended into, so a link cycle
    cannot make the scan run forever.
    """
    listings: dict[Path, _DirectoryListing] = {}
    pending: list[tuple[Path, bool]] = [(source_root, True)]
    while pending:
        results = _map_threaded(
            _list_directory,
            [directory for directory, _ in pending],
            max_workers,
        )
        children: list[tuple[Path, bool]] = []
        for (directory, descend), listing in zip(pending, results):
            listings[directory] = listing
            if not descend:
                continue
            children.extend((child, True) for child in listing.subdirectories)
            children.extend(
                (child, False) for child in listing.linked_subdirectories
            )
        pending = []
        queued: set[Path] = set()
        for child, descend in children:
            if child in listings or child in queued:
                continue
            queued.add(child)
            pending.append((child, descend))
    return listings


def _annotation_path(directory: Path) -> Path | None:
    candidate = directory / ANNOTATION_RELATIVE_PATH
    try:
        return candidate if candidate.is_file() else None
    except OSError:
        return None


def _source_level_entry(
    source_name: str,
    source_root: Path,
    error: str,
) -> InventoryEntry:
    return InventoryEntry(
        source=source_name,
        video_id=source_name,
        directory_path=str(source_root),
        video_path=None,
        annotation_path=None,
        subject_id=None,
        gender=None,
        field3=None,
        scene_raw=None,
        scene=None,
        polarity=None,
        status="error",
        error=error,
    )


def scan_inventory(
    raw_root: str | Path,
    sources: Iterable[str],
    taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH,
    max_workers: int = DEFAULT_SCAN_WORKERS,
) -> list[InventoryEntry]:
    """Scan named source directories and return deterministic inventory entries.

    Unreadable inodes become ``io_error`` records instead of aborting the scan,
    so one bad file on a flaky mount cannot discard hours of work.
    """
    root = Path(raw_root).expanduser()
    taxonomy = load_taxonomy(taxonomy_path)
    entries: list[InventoryEntry] = []

    for source in sources:
        source_name = str(source)
        source_root = root / source_name
        try:
            source_exists = source_root.is_dir()
        except OSError as exc:
            entries.append(
                _source_level_entry(source_name, source_root, _io_error_text(exc))
            )
            continue
        if not source_exists:
            entries.append(
                _source_level_entry(source_name, source_root, "missing_source")
            )
            continue

        listings = _walk_source(source_root, max_workers)
        root_error = listings[source_root].error
        if root_error:
            entries.append(
                _source_level_entry(source_name, source_root, root_error)
            )
            continue

        considered = sorted(
            (
                directory
                for directory in listings
                if directory.name != ANNOTATION_DIRECTORY_NAME
            ),
            key=lambda item: item.relative_to(source_root).as_posix(),
        )
        annotations = _map_threaded(_annotation_path, considered, max_workers)
        for directory, annotation in zip(considered, annotations):
            listing = listings[directory]
            looks_named = len(directory.name.split("_")) >= 4
            # A directory that could not be listed is always reported: its
            # contents are unknown, so it may have hidden whole samples.
            if not (
                listing.videos
                or annotation is not None
                or looks_named
                or listing.error
            ):
                continue

            relative = directory.relative_to(source_root)
            video_id = "__".join((source_name, *relative.parts))
            subject_id, gender, field3, scene_raw, name_error = _parse_directory_name(
                directory.name
            )
            errors = []
            if listing.error:
                errors.append(listing.error)
            if name_error:
                errors.append(name_error)
            if not listing.videos:
                errors.append("missing_mp4")
            if annotation is None:
                errors.append("missing_nova_annotation")

            scene = polarity = None
            scene_category = "unknown"
            hard_negative_eligible = False
            if scene_raw is not None:
                (
                    scene,
                    polarity,
                    scene_category,
                    hard_negative_eligible,
                ) = resolve_scene_details(scene_raw, taxonomy)

            video = listing.videos[0] if listing.videos else None
            entries.append(
                InventoryEntry(
                    source=source_name,
                    video_id=video_id,
                    directory_path=str(directory),
                    video_path=str(video.path) if video else None,
                    annotation_path=str(annotation) if annotation else None,
                    subject_id=subject_id,
                    gender=gender,
                    field3=field3,
                    scene_raw=scene_raw,
                    scene=scene,
                    polarity=polarity,
                    status="ok" if not errors else "error",
                    scene_category=scene_category,
                    hard_negative_eligible=hard_negative_eligible,
                    taxonomy_version=int(taxonomy.get("version", 1)),
                    video_size=video.size if video else None,
                    video_mtime_ns=video.mtime_ns if video else None,
                    error=";".join(errors) if errors else None,
                )
            )
    return entries


PREDICTION_VIDEO_SUFFIX = "_pred.mp4"


def _is_scannable_video(video: _VideoFile) -> bool:
    """Skip our own muxed output so a second run does not annotate its own results."""
    return not video.path.name.endswith(PREDICTION_VIDEO_SUFFIX)


def scan_video_root(
    video_root: str | Path,
    source_name: str = "inference",
    max_workers: int = DEFAULT_SCAN_WORKERS,
) -> list[InventoryEntry]:
    """Scan recordings for inference: one entry per .mp4, independent of naming.

    Two layouts are supported side by side. A directory holding a single video is
    treated as the sample folder itself, which is the training layout
    (``<subject>_<gender>_<field3>_<scene>/clip.mp4`` next to ``NOVA project/``) and the
    only case where directory-name metadata is parsed. Anything else, in particular a
    flat dump of files such as ``DJI_20260613115350_LN043-bike1-5.mp4``, gets a sample
    folder named after the video file, so outputs never collide and no naming
    convention is assumed.

    Unlike :func:`scan_inventory` this never marks an entry as ``error``: a missing
    annotation is the normal case here. Existing annotations are still reported so
    callers can show ground truth next to the prediction.
    """
    root = Path(video_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Video root is not a directory: {root}")

    listings = _walk_source(root, max_workers)
    videos_by_directory = {
        directory: tuple(video for video in listing.videos if _is_scannable_video(video))
        for directory, listing in listings.items()
        if directory.name != ANNOTATION_DIRECTORY_NAME
    }

    sample_directories: list[tuple[Path, _VideoFile, bool]] = []
    for directory, videos in videos_by_directory.items():
        # A lone video inside a nested folder means that folder is the sample; the scan
        # root itself never is, otherwise outputs would land next to every recording.
        folder_is_sample = len(videos) == 1 and directory != root
        for video in videos:
            sample_dir = directory if folder_is_sample else directory / video.path.stem
            sample_directories.append((sample_dir, video, folder_is_sample))
    sample_directories.sort(key=lambda item: item[0].relative_to(root).as_posix())

    annotations = _map_threaded(
        _annotation_path, [item[0] for item in sample_directories], max_workers
    )

    entries: list[InventoryEntry] = []
    for (sample_dir, video, folder_is_sample), annotation in zip(
        sample_directories, annotations
    ):
        name_error: str | None = None
        subject_id = gender = field3 = scene_raw = None
        if folder_is_sample:
            subject_id, gender, field3, scene_raw, name_error = _parse_directory_name(
                sample_dir.name
            )
        entries.append(
            InventoryEntry(
                source=source_name,
                video_id="__".join((source_name, *sample_dir.relative_to(root).parts)),
                directory_path=str(sample_dir),
                video_path=str(video.path),
                annotation_path=str(annotation) if annotation else None,
                subject_id=subject_id or "unknown",
                gender=gender or "unknown",
                field3=field3 or "unknown",
                scene_raw=scene_raw or "unknown",
                scene="unknown",
                polarity="unknown",
                status="ok",
                video_size=video.size,
                video_mtime_ns=video.mtime_ns,
                error=name_error,
            )
        )
    return entries


def _annotation_labels(path: str) -> list[str] | None:
    from utils.preprocessing.annotations import parse_nova_annotation

    try:
        return [str(interval.label) for interval in parse_nova_annotation(path)]
    except (OSError, ValueError):
        return None


def summarize_inventory(
    entries: Iterable[InventoryEntry],
    max_workers: int = DEFAULT_SCAN_WORKERS,
) -> dict[str, Any]:
    entries_list = list(entries)
    label_intervals: Counter[str] = Counter()
    annotation_parse_errors = 0
    annotation_paths = [
        entry.annotation_path for entry in entries_list if entry.annotation_path
    ]
    for labels in _map_threaded(_annotation_labels, annotation_paths, max_workers):
        if labels is None:
            annotation_parse_errors += 1
            continue
        label_intervals.update(labels)
    return {
        "total": len(entries_list),
        "by_status": dict(sorted(Counter(entry.status for entry in entries_list).items())),
        "by_source": dict(sorted(Counter(entry.source for entry in entries_list).items())),
        "by_subject": dict(
            sorted(Counter(entry.subject_id or "unknown" for entry in entries_list).items())
        ),
        "by_gender": dict(
            sorted(Counter(entry.gender or "unknown" for entry in entries_list).items())
        ),
        "by_scene": dict(
            sorted(Counter(entry.scene or "unknown" for entry in entries_list).items())
        ),
        "by_polarity": dict(
            sorted(Counter(entry.polarity or "unknown" for entry in entries_list).items())
        ),
        "errors": dict(
            sorted(
                Counter(
                    error
                    for entry in entries_list
                    for error in (entry.error or "").split(";")
                    if error
                ).items()
            )
        ),
        "annotation_label_intervals": dict(sorted(label_intervals.items())),
        "annotation_parse_errors": annotation_parse_errors,
    }


def write_inventory_jsonl(path: str | Path, entries: Iterable[InventoryEntry]) -> None:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for entry in entries:
            stream.write(json.dumps(asdict(entry), ensure_ascii=False, sort_keys=True) + "\n")


def read_inventory_jsonl(path: str | Path) -> list[InventoryEntry]:
    input_path = Path(path).expanduser()
    with input_path.open("r", encoding="utf-8") as stream:
        return [
            InventoryEntry(**json.loads(line))
            for raw_line in stream
            if (line := raw_line.strip())
        ]


def write_inventory_csv(path: str | Path, entries: Iterable[InventoryEntry]) -> None:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    field_names = [field.name for field in fields(InventoryEntry)]
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=field_names)
        writer.writeheader()
        for entry in entries:
            writer.writerow(asdict(entry))


def read_inventory_csv(path: str | Path) -> list[InventoryEntry]:
    input_path = Path(path).expanduser()
    nullable = {
        "video_path",
        "annotation_path",
        "subject_id",
        "gender",
        "field3",
        "scene_raw",
        "scene",
        "polarity",
        "error",
        "video_size",
        "video_mtime_ns",
        "annotation_sha256",
    }
    boolean_fields = {"hard_negative_eligible"}
    integer_fields = {"taxonomy_version", "video_size", "video_mtime_ns"}
    with input_path.open("r", encoding="utf-8", newline="") as stream:
        return [
            InventoryEntry(
                **{
                    key: (
                        None
                        if key in nullable and value == ""
                        else value.strip().lower() == "true"
                        if key in boolean_fields
                        else int(value)
                        if key in integer_fields
                        else value
                    )
                    for key, value in row.items()
                }
            )
            for row in csv.DictReader(stream)
        ]


def write_summary_json(
    path: str | Path,
    entries: Iterable[InventoryEntry],
    max_workers: int = DEFAULT_SCAN_WORKERS,
) -> None:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            summarize_inventory(entries, max_workers),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


# Clear compatibility names for callers that prefer build/save/load terminology.
build_inventory = scan_inventory
save_inventory_jsonl = write_inventory_jsonl
load_inventory_jsonl = read_inventory_jsonl
save_inventory_csv = write_inventory_csv
load_inventory_csv = read_inventory_csv


__all__: Sequence[str] = (
    "InventoryEntry",
    "build_inventory",
    "load_inventory_csv",
    "load_inventory_jsonl",
    "load_taxonomy",
    "normalize_scene",
    "read_inventory_csv",
    "read_inventory_jsonl",
    "resolve_scene",
    "resolve_scene_details",
    "save_inventory_csv",
    "save_inventory_jsonl",
    "scan_inventory",
    "summarize_inventory",
    "write_inventory_csv",
    "write_inventory_jsonl",
    "write_summary_json",
)
