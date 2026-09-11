"""Cheap conversion from cached raw tracks to model-ready sequence NPZ files."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from utils.preprocessing.annotations import (
    labels_from_intervals,
    parse_nova_annotation,
)
from utils.preprocessing.features import (
    OneEuroFilter,
    aggregate_hand_tracks,
    align_labels_to_target_fps,
    build_feature_matrix,
    center_hand_landmarks,
)
from utils.preprocessing.pipeline import compute_quality_audit
from utils.preprocessing.tracks import TrackConfig, load_track, track_cache_path
from utils.schema import BACKGROUND_ID, COORD_DIM, SequenceRecord, save_sequence


@dataclass(frozen=True)
class AssembleConfig:
    output_dir: str = "data/processed"
    target_fps: float = 15.0
    one_euro_min_cutoff: float = 1.0
    one_euro_beta: float = 0.007
    one_euro_d_cutoff: float = 1.0

    def fingerprint(self) -> str:
        values = asdict(self)
        values.pop("output_dir", None)
        payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _annotation_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_output_path(config: AssembleConfig, source: str, video_id: str) -> Path:
    return (
        Path(config.output_dir).expanduser()
        / source
        / f"{video_id.replace('/', '__')}.npz"
    )


def _prepare_source_tracks(
    landmarks: np.ndarray,
    valid_mask: np.ndarray,
    source_fps: float,
    config: AssembleConfig,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.zeros((len(landmarks), COORD_DIM), dtype=np.float32)
    palms = np.ones(len(landmarks), dtype=np.float32)
    coordinate_filter = OneEuroFilter(
        COORD_DIM,
        min_cutoff=config.one_euro_min_cutoff,
        beta=config.one_euro_beta,
        d_cutoff=config.one_euro_d_cutoff,
    )
    palm_filter = OneEuroFilter(
        1,
        min_cutoff=config.one_euro_min_cutoff,
        beta=config.one_euro_beta,
        d_cutoff=config.one_euro_d_cutoff,
    )
    last_coordinates = np.zeros(COORD_DIM, dtype=np.float32)
    last_palm = 1.0
    dt = 1.0 / source_fps
    for index, points in enumerate(landmarks):
        if valid_mask[index] <= 0.5 or not np.isfinite(points).all():
            coordinates[index] = last_coordinates
            palms[index] = last_palm
            continue
        centered, palm = center_hand_landmarks(points)
        last_coordinates = coordinate_filter(centered, dt)
        last_palm = float(
            palm_filter(np.asarray([palm], dtype=np.float32), dt)[0]
        )
        coordinates[index] = last_coordinates
        palms[index] = last_palm
    return coordinates, palms


def assemble_entry(
    entry: Any,
    track_config: TrackConfig,
    config: AssembleConfig,
    resume: bool = True,
) -> dict[str, Any]:
    track_path = track_cache_path(track_config, entry.source, entry.video_id)
    if not track_path.is_file():
        return {
            "video_id": entry.video_id,
            "source": entry.source,
            "status": "missing_track",
            "error": str(track_path),
        }
    track = load_track(track_path)
    annotation_sha = _annotation_sha256(entry.annotation_path)
    combined_fingerprint = hashlib.sha256(
        (
            str(track["metadata"]["track_fingerprint"])
            + config.fingerprint()
            + annotation_sha
            + str(getattr(entry, "taxonomy_version", 1))
        ).encode()
    ).hexdigest()[:16]
    output = sequence_output_path(config, entry.source, entry.video_id)
    if resume and output.is_file():
        from utils.schema import load_sequence

        try:
            existing = load_sequence(output)
        except (ValueError, KeyError):
            existing = None
        if (
            existing is not None
            and existing.metadata.get("assemble_fingerprint")
            == combined_fingerprint
        ):
            return {
                "video_id": entry.video_id,
                "source": entry.source,
                "status": "cached",
                "sequence_path": str(output),
            }

    source_fps = float(track["source_fps"])
    intervals = parse_nova_annotation(entry.annotation_path)
    source_labels = labels_from_intervals(
        len(track["landmarks"]), source_fps, intervals
    )
    if entry.polarity == "negative" and np.any(source_labels != BACKGROUND_ID):
        return {
            "video_id": entry.video_id,
            "source": entry.source,
            "status": "annotation_conflict",
            "error": "Negative scene contains non-background annotation labels",
        }

    coordinates, palms = _prepare_source_tracks(
        track["landmarks"],
        track["valid_mask"],
        source_fps,
        config,
    )
    normalized, quality, mask = aggregate_hand_tracks(
        coordinates,
        palms,
        track["tracking_quality"],
        track["valid_mask"],
        source_fps,
        config.target_fps,
    )
    labels = align_labels_to_target_fps(
        source_labels, source_fps, config.target_fps
    )
    features = build_feature_matrix(normalized, quality, mask, config.target_fps)
    if features.shape[1] != len(labels):
        raise ValueError(
            f"{entry.video_id}: feature/label length mismatch "
            f"{features.shape[1]} != {len(labels)}"
        )
    audit = compute_quality_audit(
        coordinates,
        palms,
        track["valid_mask"],
        source_labels,
        source_fps,
    )
    record = SequenceRecord(
        features=features,
        labels=labels,
        valid_mask=mask,
        video_id=entry.video_id,
        fps=config.target_fps,
        source_path=str(entry.video_path),
        hand_side="physical_right",
        subject_id=entry.subject_id,
        session_id="",
        source=entry.source,
        gender=entry.gender,
        field3=entry.field3,
        scene=entry.scene,
        polarity=entry.polarity,
        metadata={
            "source": entry.source,
            "gender": entry.gender,
            "field3": entry.field3,
            "scene": entry.scene,
            "scene_raw": entry.scene_raw,
            "polarity": entry.polarity,
            "scene_category": entry.scene_category,
            "hard_negative_eligible": bool(entry.hard_negative_eligible),
            "taxonomy_version": entry.taxonomy_version,
            "track_path": str(track_path),
            "track_fingerprint": track["metadata"]["track_fingerprint"],
            "track_config_fingerprint": track["metadata"][
                "track_config_fingerprint"
            ],
            "assemble_fingerprint": combined_fingerprint,
            "preprocess_config_fingerprint": config.fingerprint(),
            "preprocess_fingerprint": combined_fingerprint,
            "annotation_sha256": annotation_sha,
            **audit,
        },
    )
    save_sequence(output, record)
    return {
        "video_id": entry.video_id,
        "source": entry.source,
        "status": "completed",
        "sequence_path": str(output),
    }


def assemble_inventory(
    entries: Iterable[Any],
    track_config: TrackConfig,
    config: AssembleConfig,
    resume: bool = True,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    ready = [entry for entry in entries if entry.status in {"ok", "ready"}]
    for index, entry in enumerate(ready, start=1):
        try:
            result = assemble_entry(entry, track_config, config, resume)
        except Exception as exc:
            result = {
                "video_id": entry.video_id,
                "source": entry.source,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(result)
        if index % 100 == 0 or index == len(ready):
            print(f"Sequence assembly progress: {index}/{len(ready)}")
    return results
