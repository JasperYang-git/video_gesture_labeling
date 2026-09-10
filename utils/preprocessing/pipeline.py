from __future__ import annotations

import hashlib
import json
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
    align_labels_to_target_fps,
    build_feature_matrix,
    downsample_sequence,
    landmarks_to_vector,
    normalize_hand_landmarks,
)
from utils.schema import COORD_DIM, SequenceRecord, save_sequence


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

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def discover_raw_videos(
    raw_root: str | Path,
    hand_side: str | None = "R",
) -> list[dict[str, Path | str]]:
    root = Path(raw_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Raw data root does not exist: {root}")

    discovered: list[dict[str, Path | str]] = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        folder_side = infer_hand_side_from_folder(folder.name)
        if hand_side and folder_side not in {None, hand_side.upper()}:
            continue
        videos = sorted(folder.glob("*.mp4")) + sorted(folder.glob("*.MP4"))
        if not videos:
            continue
        annotation = _find_annotation(folder)
        if annotation is None:
            continue
        discovered.append(
            {
                "video_id": folder.name,
                "video_path": videos[0],
                "annotation_path": annotation,
                "hand_side": folder_side or (hand_side or "R"),
            }
        )
    return discovered


def _find_annotation(folder: Path) -> Path | None:
    candidates = [
        folder / "NOVA project" / "gestures.annotation~",
        folder / "gestures.annotation~",
        *sorted(folder.rglob("gestures.annotation~")),
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
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
    scores: list[float] = []
    valid_mask: list[float] = []
    last_valid = np.zeros(COORD_DIM, dtype=np.float32)
    smoother = OneEuroFilter(
        COORD_DIM,
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
                scores.append(0.0)
                valid_mask.append(0.0)
                continue
            landmarks, score = selected
            normalized = normalize_hand_landmarks(landmarks_to_vector(landmarks).reshape(21, 3))
            smoothed = smoother(normalized, dt)
            last_valid = smoothed
            coordinates.append(smoothed)
            scores.append(score)
            valid_mask.append(1.0)
    capture.release()

    if not coordinates:
        raise ValueError(f"No frames decoded from {video_path}")
    return (
        np.stack(coordinates, axis=0),
        np.asarray(scores, dtype=np.float32),
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


def process_raw_video(
    video_id: str,
    video_path: str | Path,
    annotation_path: str | Path | None,
    config: PreprocessConfig,
    output_dir: str | Path,
    overwrite: bool = False,
) -> SequenceRecord:
    cache_path = _cache_path(Path(config.cache_dir), video_id, config.fingerprint())
    output_path = Path(output_dir) / f"{video_id}.npz"
    if output_path.is_file() and not overwrite:
        from utils.schema import load_sequence

        return load_sequence(output_path)
    if cache_path.is_file() and not overwrite:
        from utils.schema import load_sequence

        record = load_sequence(cache_path)
        save_sequence(output_path, record)
        return record

    coordinates, scores, valid_mask, source_fps = extract_hand_tracks(video_path, config)
    labels = None
    if annotation_path is not None:
        intervals = parse_nova_annotation(annotation_path)
        source_labels = labels_from_intervals(len(coordinates), source_fps, intervals)
        labels = align_labels_to_target_fps(source_labels, source_fps, config.target_fps)

    coords_ds = downsample_sequence(coordinates, source_fps, config.target_fps)
    scores_ds = downsample_sequence(scores[:, None], source_fps, config.target_fps)[:, 0]
    mask_ds = downsample_sequence(valid_mask[:, None], source_fps, config.target_fps)[:, 0]
    mask_ds = (mask_ds >= 0.5).astype(np.float32)
    features = build_feature_matrix(coords_ds, scores_ds, mask_ds, config.target_fps)
    if labels is not None and labels.shape[0] != features.shape[1]:
        raise ValueError(
            f"{video_id}: label length {labels.shape[0]} != feature length {features.shape[1]}"
        )

    action_mask = np.ones(features.shape[1], dtype=bool)
    if labels is not None:
        from utils.schema import BACKGROUND_ID

        action_mask = labels != BACKGROUND_ID
    detection_rate = float(mask_ds[action_mask].mean()) if action_mask.any() else float(mask_ds.mean())
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
            "detection_rate": detection_rate,
            "preprocess_fingerprint": config.fingerprint(),
            "quality_passed": detection_rate >= config.quality_threshold,
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
    records: list[SequenceRecord] = []
    for item in items:
        records.append(
            process_raw_video(
                video_id=str(item["video_id"]),
                video_path=item["video_path"],
                annotation_path=item.get("annotation_path"),
                config=config,
                output_dir=output_dir,
                overwrite=overwrite,
            )
        )
    return records
