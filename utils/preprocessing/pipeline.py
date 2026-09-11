from __future__ import annotations

import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from utils.preprocessing.annotations import (
    infer_hand_side_from_folder,
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
from utils.schema import BACKGROUND_ID, COORD_DIM, NUM_CLASSES, SequenceRecord, save_sequence


@dataclass(frozen=True)
class PreprocessConfig:
    target_fps: float = 15.0
    hand_side: str = "R"
    min_detection_confidence: float = 0.7
    quality_threshold: float = 0.6
    one_euro_min_cutoff: float = 1.0
    one_euro_beta: float = 0.007
    one_euro_d_cutoff: float = 1.0
    cache_dir: str = "data/cache"
    num_workers: int = 1

    def fingerprint(self) -> str:
        values = asdict(self)
        # Runtime scheduling and cache location do not change extracted features.
        values.pop("num_workers", None)
        values.pop("cache_dir", None)
        payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(
    config: PreprocessConfig,
    video_path: str | Path,
    annotation_path: str | Path | None,
) -> tuple[str, str, dict[str, int]]:
    """Fingerprint the recipe, video identity and annotation contents."""
    video = Path(video_path)
    stat = video.stat()
    video_signature = {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
    annotation_sha256 = _file_sha256(annotation_path) if annotation_path else ""
    payload = {
        "config": config.fingerprint(),
        "video": video_signature,
        "annotation_sha256": annotation_sha256,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16], annotation_sha256, video_signature


def discover_raw_videos(
    raw_root: str | Path,
    hand_side: str | None = "R",
) -> list[dict[str, Path | str]]:
    root = Path(raw_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Raw data root does not exist: {root}")

    discovered: list[dict[str, Path | str]] = []
    video_folders = sorted(
        {
            video_path.parent
            for pattern in ("*.mp4", "*.MP4")
            for video_path in root.rglob(pattern)
            if video_path.is_file()
        }
    )
    for folder in video_folders:
        folder_side = infer_hand_side_from_folder(folder.name)
        if hand_side and folder_side not in {None, hand_side.upper()}:
            continue
        videos = sorted(folder.glob("*.mp4")) + sorted(folder.glob("*.MP4"))
        annotation = _find_annotation(folder)
        if annotation is None:
            continue
        relative_folder = folder.relative_to(root)
        video_id = "__".join(relative_folder.parts)
        discovered.append(
            {
                "video_id": video_id,
                "video_path": videos[0],
                "annotation_path": annotation,
                "hand_side": folder_side or (hand_side or "R"),
            }
        )
    return discovered


def _find_annotation(folder: Path) -> Path | None:
    path = folder / "NOVA project" / "gestures.annotation~"
    return path if path.is_file() else None


def _cache_path(cache_dir: Path, video_id: str, fingerprint: str) -> Path:
    return cache_dir / fingerprint / f"{video_id}.npz"


def _optional_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required to process raw videos"
        ) from exc
    return cv2


def _optional_mediapipe():
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise ImportError(
            "mediapipe is required to process raw videos"
        ) from exc
    return mp


def extract_hand_tracks(
    video_path: str | Path,
    config: PreprocessConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    cv2 = _optional_cv2()
    mp = _optional_mediapipe()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 1e-3:
        source_fps = 30.0

    desired_side = "Left" if config.hand_side.upper() == "L" else "Right"
    coordinates: list[np.ndarray] = []
    palm_sizes: list[float] = []
    tracking_quality: list[float] = []
    valid_mask: list[float] = []
    last_valid = np.zeros(COORD_DIM, dtype=np.float32)
    last_palm = 1.0
    smoother = OneEuroFilter(
        COORD_DIM,
        min_cutoff=config.one_euro_min_cutoff,
        beta=config.one_euro_beta,
        d_cutoff=config.one_euro_d_cutoff,
    )
    palm_smoother = OneEuroFilter(
        1,
        min_cutoff=config.one_euro_min_cutoff,
        beta=config.one_euro_beta,
        d_cutoff=config.one_euro_d_cutoff,
    )
    dt = 1.0 / source_fps

    with mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=config.min_detection_confidence,
        min_tracking_confidence=config.min_detection_confidence,
    ) as hands:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            selected = _select_hand(result, desired_side, config.min_detection_confidence)
            if selected is None:
                coordinates.append(last_valid.copy())
                palm_sizes.append(last_palm)
                tracking_quality.append(0.0)
                valid_mask.append(0.0)
                continue
            landmarks, quality = selected
            centered, palm = center_hand_landmarks(landmarks)
            smoothed = smoother(centered, dt)
            smoothed_palm = float(palm_smoother(np.asarray([palm], dtype=np.float32), dt)[0])
            last_valid = smoothed
            last_palm = smoothed_palm
            coordinates.append(smoothed)
            palm_sizes.append(smoothed_palm)
            tracking_quality.append(quality)
            valid_mask.append(1.0)
    capture.release()

    if not coordinates:
        raise ValueError(f"No frames decoded from {video_path}")
    return (
        np.stack(coordinates, axis=0),
        np.asarray(palm_sizes, dtype=np.float32),
        np.asarray(tracking_quality, dtype=np.float32),
        np.asarray(valid_mask, dtype=np.float32),
        source_fps,
    )


def _select_hand(
    result: Any,
    desired_side: str,
    min_confidence: float,
) -> tuple[np.ndarray, float] | None:
    if result.multi_hand_landmarks is None or result.multi_handedness is None:
        return None
    best: tuple[np.ndarray, float] | None = None
    best_score = -1.0
    for landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
        label = handedness.classification[0].label
        score = float(handedness.classification[0].score)
        if score < min_confidence:
            continue
        if label != desired_side:
            continue
        points = np.asarray(
            [[pt.x, pt.y, pt.z] for pt in landmarks.landmark],
            dtype=np.float32,
        )
        if score > best_score:
            best = (points, score)
            best_score = score
    return best


def compute_quality_audit(
    centered_coordinates: np.ndarray,
    palm_sizes: np.ndarray,
    valid_mask: np.ndarray,
    labels: np.ndarray | None,
    fps: float,
) -> dict[str, Any]:
    """Compute source-FPS quality metrics, with action-only detection rates."""
    mask = np.asarray(valid_mask, dtype=np.float32) > 0.5
    coords = np.asarray(centered_coordinates, dtype=np.float32)
    palms = np.asarray(palm_sizes, dtype=np.float32)
    action_mask = np.ones(len(mask), dtype=bool)
    if labels is not None:
        action_mask = labels != BACKGROUND_ID
    detection_rate = float(mask[action_mask].mean()) if action_mask.any() else float(mask.mean())

    per_class: dict[str, float] = {}
    if labels is not None:
        for class_id in range(NUM_CLASSES):
            if class_id == BACKGROUND_ID:
                continue
            class_mask = labels == class_id
            if class_mask.any():
                per_class[str(class_id)] = float(mask[class_mask].mean())
    min_class_id = min(per_class, key=per_class.get) if per_class else ""
    min_class_rate = per_class[min_class_id] if min_class_id else detection_rate

    longest_missing = 0
    current_missing = 0
    for is_valid in mask:
        current_missing = 0 if is_valid else current_missing + 1
        longest_missing = max(longest_missing, current_missing)

    palm_valid = np.isfinite(palms) & (palms > 1e-3)
    if palm_valid.any():
        median_palm = float(np.median(palms[palm_valid]))
        palm_outlier = np.abs(palms - median_palm) > max(3.0 * median_palm, 1e-3)
        palm_outlier_rate = float(palm_outlier.mean())
    else:
        palm_outlier_rate = 1.0

    if len(coords) > 1:
        jumps = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        median_jump = float(np.median(jumps))
        mad = float(np.median(np.abs(jumps - median_jump)))
        threshold = median_jump + 6.0 * max(mad, 1e-6)
        trajectory_jump_rate = float((jumps > threshold).mean())
    else:
        trajectory_jump_rate = 0.0
    return {
        "detection_rate": detection_rate,
        "per_class_detection_rate": per_class,
        "min_class_id": min_class_id,
        "min_class_detection_rate": float(min_class_rate),
        "longest_missing_frames": int(longest_missing),
        "longest_missing_seconds": float(longest_missing / fps),
        "palm_outlier_rate": palm_outlier_rate,
        "trajectory_jump_rate": trajectory_jump_rate,
    }


def process_raw_video(
    video_id: str,
    video_path: str | Path,
    annotation_path: str | Path | None,
    config: PreprocessConfig,
    output_dir: str | Path,
    overwrite: bool = False,
) -> SequenceRecord:
    fingerprint, annotation_sha256, video_signature = source_fingerprint(
        config, video_path, annotation_path
    )
    cache_path = _cache_path(Path(config.cache_dir), video_id, fingerprint)
    output_path = Path(output_dir) / f"{video_id}.npz"
    if output_path.is_file() and not overwrite:
        from utils.schema import load_sequence

        try:
            record = load_sequence(output_path)
        except (KeyError, ValueError):
            record = None
        if record is not None and record.metadata.get("preprocess_fingerprint") == fingerprint:
            return record
    if cache_path.is_file() and not overwrite:
        from utils.schema import load_sequence

        try:
            record = load_sequence(cache_path)
        except (KeyError, ValueError):
            record = None
        if record is not None:
            save_sequence(output_path, record)
            return record

    coordinates, palm_sizes, tracking_quality, valid_mask, source_fps = extract_hand_tracks(
        video_path, config
    )
    labels = None
    source_labels = None
    if annotation_path is not None:
        intervals = parse_nova_annotation(annotation_path)
        source_labels = labels_from_intervals(len(coordinates), source_fps, intervals)
        labels = align_labels_to_target_fps(source_labels, source_fps, config.target_fps)

    coords_ds, quality_ds, mask_ds = aggregate_hand_tracks(
        coordinates,
        palm_sizes,
        tracking_quality,
        valid_mask,
        source_fps,
        config.target_fps,
    )
    features = build_feature_matrix(coords_ds, quality_ds, mask_ds, config.target_fps)
    if labels is not None and labels.shape[0] != features.shape[1]:
        raise ValueError(
            f"{video_id}: label length {labels.shape[0]} != feature length {features.shape[1]}"
        )

    audit = compute_quality_audit(
        coordinates, palm_sizes, valid_mask, source_labels, source_fps
    )
    record = SequenceRecord(
        features=features,
        labels=labels,
        valid_mask=mask_ds,
        video_id=video_id,
        fps=config.target_fps,
        source_path=str(video_path),
        hand_side=config.hand_side,
        metadata={
            "source_fps": source_fps,
            **audit,
            "preprocess_fingerprint": fingerprint,
            "preprocess_config_fingerprint": config.fingerprint(),
            "annotation_sha256": annotation_sha256,
            "video_signature": video_signature,
            "quality_passed": audit["detection_rate"] >= config.quality_threshold,
        },
    )
    save_sequence(cache_path, record)
    save_sequence(output_path, record)
    return record


def process_raw_videos(
    items: Iterable[dict[str, Path | str]],
    config: PreprocessConfig,
    output_dir: str | Path,
    overwrite: bool = False,
) -> list[SequenceRecord]:
    item_list = list(items)
    if config.num_workers <= 1 or len(item_list) <= 1:
        return [
            process_raw_video(
                video_id=str(item["video_id"]),
                video_path=item["video_path"],
                annotation_path=item.get("annotation_path"),
                config=config,
                output_dir=output_dir,
                overwrite=overwrite,
            )
            for item in item_list
        ]

    worker_count = min(int(config.num_workers), len(item_list))
    output_paths: list[Path | None] = [None] * len(item_list)
    failures: list[str] = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
    ) as executor:
        futures = {
            executor.submit(
                _process_raw_video_worker,
                index,
                item,
                config,
                str(output_dir),
                overwrite,
            ): (index, str(item["video_id"]))
            for index, item in enumerate(item_list)
        }
        for future in as_completed(futures):
            index, video_id = futures[future]
            try:
                _, output_path = future.result()
            except Exception as exc:
                failures.append(f"{video_id}: {type(exc).__name__}: {exc}")
            else:
                output_paths[index] = Path(output_path)
                print(f"Completed {video_id} ({sum(path is not None for path in output_paths)}/{len(item_list)})")
    if failures:
        details = "\n".join(f"  - {failure}" for failure in failures)
        raise RuntimeError(
            f"{len(failures)} of {len(item_list)} videos failed during preprocessing:\n{details}"
        )
    from utils.schema import load_sequence

    return [load_sequence(path) for path in output_paths if path is not None]


def _process_raw_video_worker(
    index: int,
    item: dict[str, Path | str],
    config: PreprocessConfig,
    output_dir: str,
    overwrite: bool,
) -> tuple[int, str]:
    """Spawn-safe worker that returns only a path, not large feature arrays."""
    video_id = str(item["video_id"])
    process_raw_video(
        video_id=video_id,
        video_path=item["video_path"],
        annotation_path=item.get("annotation_path"),
        config=config,
        output_dir=output_dir,
        overwrite=overwrite,
    )
    return index, str(Path(output_dir) / f"{video_id}.npz")
