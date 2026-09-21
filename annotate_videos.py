"""Batch-annotate raw recordings: mp4 in, predicted gesture annotations out."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from utils.config import load_config
from utils.inference import (
    load_checkpoint_model,
    predict_sequence,
    validate_record_compatibility,
)
from utils.logger import create_run_directory, setup_logger
from utils.preprocessing.annotations import (
    intervals_from_prediction,
    write_nova_annotation,
)
from utils.preprocessing.assemble import (
    AssembleConfig,
    assemble_inventory,
    sequence_output_path,
)
from utils.preprocessing.inventory import (
    ANNOTATION_DIRECTORY_NAME,
    ANNOTATION_FILE_NAME,
    DEFAULT_SCAN_WORKERS,
    PREDICTION_MARKER_NAME,
    PREDICTION_VIDEO_SUFFIX,
    InventoryEntry,
    scan_video_root,
)
from utils.preprocessing.status import status_summary
from utils.preprocessing.tracks import (
    TrackConfig,
    extract_inventory,
    load_track,
    lost_tracking_positions,
    track_cache_path,
    write_handedness_previews,
)
from utils.schema import SCORE_INDEX, SequenceRecord, load_sequence
from utils.subtitles import mux_subtitles, write_srt
from utils.timeline import render_timeline
from utils.trainer import resolve_device, seed_everything

PREDICTED_ANNOTATION_FILE_NAME = "gestures.pred.annotation~"
PREVIEW_DIRECTORY_NAME = "handedness_preview"
QUALITY_KEYS = (
    "detection_rate",
    "longest_missing_seconds",
    "palm_outlier_rate",
    "trajectory_jump_rate",
    "trajectory_jump_reliable",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config/config_annotate.yaml")
    parser.add_argument("--video-root", default=None, help="Override video_root")
    parser.add_argument("--model-path", default=None, help="Checkpoint override")
    parser.add_argument("--workers", type=int, default=None, help="Extraction workers")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N videos")
    parser.add_argument(
        "--mux",
        action="store_true",
        help="Also write <name>_pred.mp4 with the prediction as a soft subtitle track",
    )
    parser.add_argument(
        "--overwrite-annotation",
        action="store_true",
        help="Overwrite an existing gestures.annotation~ instead of writing a .pred. copy",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore cached tracks and sequences",
    )
    return parser.parse_args()


def build_track_config(config: dict[str, Any], workers: int | None) -> TrackConfig:
    track_config = TrackConfig(**dict(config["extract"]))
    if workers is not None:
        track_config = replace(track_config, num_workers=workers)
    return track_config


def is_self_written(entry: InventoryEntry) -> bool:
    """True when the annotation on disk was produced by a previous run of this script."""
    if not entry.annotation_path:
        return False
    return (Path(entry.annotation_path).parent / PREDICTION_MARKER_NAME).is_file()


def forget_self_annotations(entries: list[InventoryEntry]) -> list[InventoryEntry]:
    """Hide our own past predictions so they are never treated as ground truth.

    Without this a second run would assemble the previous prediction as labels and the
    timeline would proudly report a frame accuracy of 1.000 against itself.
    """
    return [
        replace(entry, annotation_path=None) if is_self_written(entry) else entry
        for entry in entries
    ]


def annotation_output_path(
    entry: InventoryEntry,
    overwrite: bool,
) -> tuple[Path, bool]:
    """Return the annotation path and whether it shadows an existing hand annotation."""
    directory = Path(entry.directory_path) / ANNOTATION_DIRECTORY_NAME
    target = directory / ANNOTATION_FILE_NAME
    already_ours = (directory / PREDICTION_MARKER_NAME).is_file()
    if target.is_file() and not overwrite and not already_ours:
        return directory / PREDICTED_ANNOTATION_FILE_NAME, True
    return target, False


def quality_metrics(record: SequenceRecord, threshold: float) -> dict[str, Any]:
    """Pull the audit numbers assemble already stored in the sequence metadata."""
    metrics: dict[str, Any] = {
        key: record.metadata.get(key) for key in QUALITY_KEYS if key in record.metadata
    }
    detection_rate = metrics.get("detection_rate")
    if detection_rate is None:
        # Fall back to the mask itself so a sequence written before the audit existed
        # still reports something usable.
        detection_rate = (
            float(np.mean(record.valid_mask > 0.5)) if record.num_frames else 0.0
        )
        metrics["detection_rate"] = detection_rate
    metrics["detection_rate"] = float(detection_rate)
    metrics["quality_passed"] = float(detection_rate) >= threshold
    return metrics


def write_quality_report(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Worst tracking first, because that is the list worth acting on."""
    columns = [
        "video_id",
        "detection_rate",
        "quality_passed",
        "longest_missing_seconds",
        "duration_sec",
        "predicted_actions",
        "frame_accuracy",
        "palm_outlier_rate",
        "trajectory_jump_rate",
        "trajectory_jump_reliable",
        "video_path",
    ]
    ordered = sorted(rows, key=lambda row: float(row.get("detection_rate") or 0.0))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for row in ordered:
            writer.writerow(
                [
                    f"{row[key]:.4f}"
                    if isinstance(row.get(key), float)
                    else ("" if row.get(key) is None else row.get(key))
                    for key in columns
                ]
            )
    return path


def preview_frame_positions(
    entry: InventoryEntry,
    track_config: TrackConfig,
) -> list[int]:
    """Source-fps frames worth eyeballing, read back from the cached track."""
    cache_path = track_cache_path(track_config, entry.source, entry.video_id)
    if not cache_path.is_file():
        return []
    return lost_tracking_positions(load_track(cache_path)["valid_mask"])


def write_prediction_marker(annotation_path: Path, entry: InventoryEntry, model_path: str) -> None:
    payload = {
        "video_id": entry.video_id,
        "video_path": entry.video_path,
        "model_path": str(model_path),
        "written_at": datetime.now().isoformat(timespec="seconds"),
    }
    (annotation_path.parent / PREDICTION_MARKER_NAME).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))
    outputs = dict(config.get("outputs", {}))
    run_dir = create_run_directory(outputs.get("root_dir", "outputs/annotate_results"))
    logger = setup_logger("annotate", run_dir / "annotate.log")
    shutil.copy2(args.config, run_dir / "config_annotate.yaml")

    video_root = args.video_root or config["video_root"]
    entries = scan_video_root(video_root, max_workers=DEFAULT_SCAN_WORKERS)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        entries = entries[: args.limit]
    if not entries:
        raise FileNotFoundError(f"No .mp4 files found under {video_root}")
    entries = forget_self_annotations(entries)
    labeled = sum(1 for entry in entries if entry.annotation_path)
    logger.info(
        "Run directory: %s | videos=%d (%d with ground truth) | root=%s",
        run_dir,
        len(entries),
        labeled,
        video_root,
    )

    track_config = build_track_config(config, args.workers)
    assemble_config = AssembleConfig(**dict(config["assemble"]))
    resume = not args.no_resume

    extract_results = extract_inventory(entries, track_config, resume=resume)
    logger.info("Track extraction: %s", status_summary(extract_results))
    assemble_results = assemble_inventory(
        entries, track_config, assemble_config, resume=resume
    )
    logger.info("Sequence assembly: %s", status_summary(assemble_results))
    assembled = {
        result["video_id"]: result
        for result in assemble_results
        if result["status"] in {"completed", "cached"}
    }

    device = resolve_device(str(config.get("device", "auto")))
    model_path = args.model_path or config["inference"]["model_path"]
    model, checkpoint = load_checkpoint_model(model_path, device)
    window_size = int(config["inference"].get("window_size", checkpoint.get("window_size", 60)))
    stride = int(config["inference"].get("stride", checkpoint.get("stride", 12)))
    logger.info("Device: %s | model: %s | window=%d stride=%d", device, model_path, window_size, stride)

    overwrite = args.overwrite_annotation or bool(outputs.get("overwrite_annotation", False))
    want_mux = args.mux or bool(outputs.get("mux", False))
    quality_threshold = float(outputs.get("quality_threshold", 0.6))
    preview_threshold = float(outputs.get("preview_threshold", 0.8))
    index_payload: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []

    for entry in entries:
        result = assembled.get(entry.video_id)
        if result is None:
            logger.warning("Skipping %s: no assembled sequence", entry.video_id)
            continue
        sequence_path = Path(
            result.get("sequence_path")
            or sequence_output_path(assemble_config, entry.source, entry.video_id)
        )
        record = load_sequence(sequence_path)
        validate_record_compatibility(record, checkpoint)
        prediction, logits = predict_sequence(
            model, record, device, window_size, stride
        )

        directory = Path(entry.directory_path)
        written: dict[str, str] = {}
        intervals = intervals_from_prediction(prediction, record.fps, logits)
        quality = quality_metrics(record, quality_threshold)

        if bool(outputs.get("write_annotation", True)):
            annotation_path, shadowed = annotation_output_path(entry, overwrite)
            write_nova_annotation(annotation_path, intervals)
            written["annotation"] = str(annotation_path)
            if shadowed:
                logger.warning(
                    "%s already has a hand annotation; wrote prediction to %s",
                    entry.video_id,
                    annotation_path.name,
                )
            else:
                write_prediction_marker(annotation_path, entry, model_path)
        if bool(outputs.get("write_timeline", True)):
            timeline_path = directory / "gestures_timeline.png"
            render_timeline(
                timeline_path,
                prediction,
                record.fps,
                labels=record.labels,
                valid_mask=record.valid_mask,
                tracking_quality=record.features[SCORE_INDEX],
                title=entry.video_id,
            )
            written["timeline"] = str(timeline_path)
        if bool(outputs.get("write_srt", True)) or want_mux:
            srt_path = directory / "gestures.srt"
            write_srt(srt_path, intervals)
            written["srt"] = str(srt_path)
            if want_mux and entry.video_path:
                video_path = Path(entry.video_path)
                muxed = directory / f"{video_path.stem}{PREDICTION_VIDEO_SUFFIX}"
                mux_subtitles(video_path, srt_path, muxed)
                written["muxed_video"] = str(muxed)
        if quality["detection_rate"] < preview_threshold:
            positions = preview_frame_positions(entry, track_config)
            if positions:
                preview_dir = directory / PREVIEW_DIRECTORY_NAME
                try:
                    previews = write_handedness_previews(
                        entry, track_config, preview_dir, positions=positions
                    )
                except Exception as exc:
                    # Previews need to decode the video again and re-run MediaPipe.
                    # They are a diagnostic extra, so a failure here must not cost the
                    # annotation that was already produced.
                    logger.warning(
                        "%s | could not write handedness previews: %s: %s",
                        entry.video_id,
                        type(exc).__name__,
                        exc,
                    )
                else:
                    written["handedness_preview"] = str(preview_dir)
                    logger.info(
                        "%s | detection_rate=%.3f below %.2f; wrote %d preview frames",
                        entry.video_id,
                        quality["detection_rate"],
                        preview_threshold,
                        len(previews),
                    )

        artifacts_dir = run_dir / entry.video_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        np.save(artifacts_dir / "prediction.npy", prediction)
        np.save(artifacts_dir / "logits.npy", logits)

        accuracy = (
            float(np.mean(prediction == record.labels))
            if record.labels is not None
            else None
        )
        logger.info(
            "%s | frames=%d | actions=%d | detection=%.3f%s",
            entry.video_id,
            record.num_frames,
            len(intervals),
            quality["detection_rate"],
            f" | frame_accuracy={accuracy:.3f}" if accuracy is not None else "",
        )
        index_payload.append(
            {
                "video_id": entry.video_id,
                "video_path": entry.video_path,
                "sequence_path": str(sequence_path),
                "num_frames": record.num_frames,
                "fps": record.fps,
                "predicted_actions": len(intervals),
                "has_ground_truth": record.labels is not None,
                "frame_accuracy": accuracy,
                "quality": quality,
                "outputs": written,
            }
        )
        quality_rows.append(
            {
                "video_id": entry.video_id,
                "video_path": entry.video_path,
                "duration_sec": record.num_frames / record.fps if record.fps else 0.0,
                "predicted_actions": len(intervals),
                "frame_accuracy": accuracy,
                **quality,
            }
        )

    index_path = run_dir / "annotations.json"
    index_path.write_text(json.dumps(index_payload, indent=2) + "\n", encoding="utf-8")
    logger.info("Annotated %d/%d videos; index: %s", len(index_payload), len(entries), index_path)

    if quality_rows and bool(outputs.get("write_quality_report", True)):
        report_path = write_quality_report(run_dir / "quality_report.csv", quality_rows)
        logger.info("Wrote %s", report_path)

    low_quality = sorted(
        (row for row in quality_rows if not row["quality_passed"]),
        key=lambda row: row["detection_rate"],
    )
    if low_quality:
        logger.warning(
            "Low tracking quality (detection_rate < %.2f): %d/%d videos. "
            "Predictions there say more about MediaPipe than about the model.",
            quality_threshold,
            len(low_quality),
            len(quality_rows),
        )
        for row in low_quality:
            logger.warning(
                "  %-56s detection=%.3f longest_gap=%.1fs",
                row["video_id"],
                row["detection_rate"],
                float(row.get("longest_missing_seconds") or 0.0),
            )


if __name__ == "__main__":
    main()
