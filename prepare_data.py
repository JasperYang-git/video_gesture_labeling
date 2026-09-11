"""Stage-based data preparation for large gesture-video collections."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from utils.config import load_config
from utils.mock_data import generate_mock_and_manifest
from utils.preprocessing.assemble import (
    AssembleConfig,
    assemble_inventory,
)
from utils.preprocessing.audit import AuditConfig, audit_inventory
from utils.preprocessing.inventory import (
    InventoryEntry,
    read_inventory_jsonl,
    scan_inventory,
    write_inventory_csv,
    write_inventory_jsonl,
    write_summary_json,
)
from utils.preprocessing.manifest import (
    ManifestConfig,
    build_experiment_manifest,
)
from utils.preprocessing.status import status_summary, write_status_jsonl
from utils.preprocessing.tracks import (
    TrackConfig,
    extract_inventory,
    write_handedness_previews,
)
from utils.trainer import seed_everything


STAGES = ("inventory", "preview", "extract", "assemble", "audit", "manifest", "all", "mock")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("stage", nargs="?", choices=STAGES, default="all")
    parser.add_argument("--config", default="config/config_prepare.yaml")
    parser.add_argument("--raw-root", help="Override inventory.raw_root")
    parser.add_argument(
        "--sources",
        nargs="+",
        help="Only process these source directories",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Pilot limit per source, applied after deterministic sorting",
    )
    parser.add_argument("--workers", type=int, help="Override extract.num_workers")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore matching track/sequence cache entries",
    )
    parser.add_argument(
        "--overwrite-subject-splits",
        action="store_true",
        help="Explicitly regenerate the global subject split registry",
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Compatibility alias for the mock stage",
    )
    return parser.parse_args()


def _inventory_settings(config: dict[str, Any], args: argparse.Namespace):
    section = config["inventory"]
    raw_root = args.raw_root or section["raw_root"]
    sources = args.sources or list(section["sources"])
    return section, raw_root, sources


def _run_inventory(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> list[InventoryEntry]:
    section, raw_root, sources = _inventory_settings(config, args)
    entries = scan_inventory(raw_root, sources, section["taxonomy_path"])
    write_inventory_jsonl(section["output_jsonl"], entries)
    write_inventory_csv(section["output_csv"], entries)
    write_summary_json(section["summary_path"], entries)
    print(
        f"Inventory: {len(entries)} records; "
        f"status={dict(Counter(item.status for item in entries))}; "
        f"polarity={dict(Counter(item.polarity or 'unknown' for item in entries))}"
    )
    return entries


def _load_inventory(config: dict[str, Any]) -> list[InventoryEntry]:
    path = Path(config["inventory"]["output_jsonl"]).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Inventory not found: {path}. Run `python prepare_data.py inventory` first."
        )
    return read_inventory_jsonl(path)


def _select_entries(
    entries: list[InventoryEntry],
    args: argparse.Namespace,
    include_unknown: bool,
) -> list[InventoryEntry]:
    source_filter = set(args.sources or [])
    selected = [
        entry
        for entry in entries
        if (not source_filter or entry.source in source_filter)
        and (include_unknown or entry.polarity in {"positive", "negative"})
    ]
    if args.limit is None:
        return selected
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    by_source: dict[str, list[InventoryEntry]] = defaultdict(list)
    for entry in selected:
        by_source[entry.source].append(entry)
    return [
        entry
        for source in sorted(by_source)
        for entry in sorted(by_source[source], key=lambda item: item.video_id)[
            : args.limit
        ]
    ]


def _track_config(
    config: dict[str, Any], args: argparse.Namespace
) -> TrackConfig:
    values = dict(config["extract"])
    for runtime_key in ("status_path", "preview_dir", "preview_count"):
        values.pop(runtime_key, None)
    track_config = TrackConfig(**values)
    if args.workers is not None:
        track_config = replace(track_config, num_workers=args.workers)
    return track_config


def _run_preview(
    entries: list[InventoryEntry],
    config: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    section = config["extract"]
    track_config = _track_config(config, args)
    candidates = _select_entries(entries, args, include_unknown=False)
    if args.limit is None:
        candidates = candidates[:1]
    written = []
    for entry in candidates:
        if entry.status != "ok":
            continue
        written.extend(
            write_handedness_previews(
                entry,
                track_config,
                section["preview_dir"],
                int(section.get("preview_count", 6)),
            )
        )
    print(f"Wrote {len(written)} physical-right preview images")


def _run_extract(
    entries: list[InventoryEntry],
    config: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    selected = _select_entries(
        entries,
        args,
        bool(config["inventory"].get("include_unknown_for_extraction", False)),
    )
    results = extract_inventory(
        selected,
        _track_config(config, args),
        resume=not args.no_resume,
    )
    path = write_status_jsonl(config["extract"]["status_path"], results)
    print(f"Extract status: {status_summary(results)}; details={path}")
    return results


def _run_assemble(
    entries: list[InventoryEntry],
    config: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    selected = _select_entries(
        entries,
        args,
        bool(config["inventory"].get("include_unknown_for_extraction", False)),
    )
    results = assemble_inventory(
        selected,
        _track_config(config, args),
        AssembleConfig(
            **{
                key: value
                for key, value in config["assemble"].items()
                if key != "status_path"
            }
        ),
        resume=not args.no_resume,
    )
    path = write_status_jsonl(config["assemble"]["status_path"], results)
    print(f"Assemble status: {status_summary(results)}; details={path}")
    return results


def _run_audit(
    entries: list[InventoryEntry],
    config: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    selected = _select_entries(entries, args, include_unknown=True)
    assemble_config = AssembleConfig(
        **{
            key: value
            for key, value in config["assemble"].items()
            if key != "status_path"
        }
    )
    csv_path, pass_path, rows = audit_inventory(
        selected,
        assemble_config,
        AuditConfig(**config["audit"]),
    )
    print(
        f"Audit: {len(rows)} sequences, "
        f"{sum(bool(row.get('quality_passed')) for row in rows)} passed; "
        f"report={csv_path}; pass_list={pass_path}"
    )


def _run_manifest(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    values = dict(config["manifest"])
    values.pop("enabled_in_all", None)
    if args.sources:
        values["sources"] = args.sources
    output = build_experiment_manifest(
        ManifestConfig.from_dict(values),
        overwrite_subject_splits=args.overwrite_subject_splits,
    )
    print(f"Wrote experiment manifest: {output}")


def _run_mock(config: dict[str, Any]) -> None:
    if "data" not in config:
        raise ValueError(
            "Mock compatibility requires a data section in config_prepare.yaml"
        )
    print(f"Wrote mock manifest: {generate_mock_and_manifest(config)}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))
    if args.use_mock or args.stage == "mock":
        _run_mock(config)
        return

    if args.stage == "inventory":
        _run_inventory(config, args)
        return
    entries = (
        _run_inventory(config, args)
        if args.stage == "all"
        else _load_inventory(config)
    )
    if args.stage == "preview":
        _run_preview(entries, config, args)
    elif args.stage == "extract":
        _run_extract(entries, config, args)
    elif args.stage == "assemble":
        _run_assemble(entries, config, args)
    elif args.stage == "audit":
        _run_audit(entries, config, args)
    elif args.stage == "manifest":
        _run_manifest(config, args)
    elif args.stage == "all":
        _run_extract(entries, config, args)
        _run_assemble(entries, config, args)
        _run_audit(entries, config, args)
        if bool(config["manifest"].get("enabled_in_all", False)):
            _run_manifest(config, args)
    else:
        raise ValueError(f"Unsupported stage: {args.stage}")


if __name__ == "__main__":
    main()
