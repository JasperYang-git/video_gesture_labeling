from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


DEFAULT_TAXONOMY_PATH = Path(__file__).resolve().parents[2] / "config" / "scene_taxonomy.yaml"
ANNOTATION_RELATIVE_PATH = Path("NOVA project") / "gestures.annotation~"


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


def _candidate_directories(source_root: Path) -> list[Path]:
    candidates: set[Path] = set()
    for directory in (source_root, *source_root.rglob("*")):
        if not directory.is_dir() or directory.name == "NOVA project":
            continue
        has_video = any(
            child.is_file() and child.suffix.lower() == ".mp4"
            for child in directory.iterdir()
        )
        has_annotation = (directory / ANNOTATION_RELATIVE_PATH).is_file()
        looks_named = len(directory.name.split("_")) >= 4
        if has_video or has_annotation or looks_named:
            candidates.add(directory)
    return sorted(candidates, key=lambda item: item.relative_to(source_root).as_posix())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_inventory(
    raw_root: str | Path,
    sources: Iterable[str],
    taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH,
) -> list[InventoryEntry]:
    """Scan named source directories and return deterministic inventory entries."""
    root = Path(raw_root).expanduser()
    taxonomy = load_taxonomy(taxonomy_path)
    entries: list[InventoryEntry] = []

    for source in sources:
        source_name = str(source)
        source_root = root / source_name
        if not source_root.is_dir():
            entries.append(
                InventoryEntry(
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
                    error="missing_source",
                )
            )
            continue

        for directory in _candidate_directories(source_root):
            relative = directory.relative_to(source_root)
            video_id = "__".join((source_name, *relative.parts))
            subject_id, gender, field3, scene_raw, name_error = _parse_directory_name(
                directory.name
            )
            videos = sorted(
                (
                    child
                    for child in directory.iterdir()
                    if child.is_file() and child.suffix.lower() == ".mp4"
                ),
                key=lambda child: child.name,
            )
            annotation = directory / ANNOTATION_RELATIVE_PATH
            errors = []
            if name_error:
                errors.append(name_error)
            if not videos:
                errors.append("missing_mp4")
            if not annotation.is_file():
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

            entries.append(
                InventoryEntry(
                    source=source_name,
                    video_id=video_id,
                    directory_path=str(directory),
                    video_path=str(videos[0]) if videos else None,
                    annotation_path=str(annotation) if annotation.is_file() else None,
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
                    video_size=videos[0].stat().st_size if videos else None,
                    video_mtime_ns=videos[0].stat().st_mtime_ns if videos else None,
                    annotation_sha256=_sha256(annotation)
                    if annotation.is_file()
                    else None,
                    error=";".join(errors) if errors else None,
                )
            )
    return entries


def summarize_inventory(entries: Iterable[InventoryEntry]) -> dict[str, Any]:
    entries_list = list(entries)
    label_intervals: Counter[str] = Counter()
    annotation_parse_errors = 0
    from utils.preprocessing.annotations import parse_nova_annotation

    for entry in entries_list:
        if not entry.annotation_path:
            continue
        try:
            intervals = parse_nova_annotation(entry.annotation_path)
        except (OSError, ValueError):
            annotation_parse_errors += 1
            continue
        label_intervals.update(str(interval.label) for interval in intervals)
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


def write_summary_json(path: str | Path, entries: Iterable[InventoryEntry]) -> None:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summarize_inventory(entries), ensure_ascii=False, indent=2, sort_keys=True)
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
