"""MediaPipe-only extraction of a strict physical-hand landmark track."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class TrackConfig:
    physical_hand: str = "right"
    input_mirrored: bool = False
    allow_opposite_hand_fallback: bool = False
    handedness_threshold: float = 0.7
    max_num_hands: int = 2
    model_complexity: int = 1
    num_workers: int = 4
    cache_dir: str = "data/cache/tracks"

    def __post_init__(self) -> None:
        if self.physical_hand.lower() not in {"left", "right"}:
            raise ValueError("physical_hand must be left or right")
        if self.allow_opposite_hand_fallback:
            raise ValueError(
                "Opposite-hand fallback is forbidden for the strict-hand feature schema"
            )
        if self.max_num_hands < 1:
            raise ValueError("max_num_hands must be positive")

    def fingerprint(self) -> str:
        values = asdict(self)
        values.pop("num_workers", None)
        values.pop("cache_dir", None)
        payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def mediapipe_label_for_physical_hand(
    physical_hand: str,
    input_mirrored: bool,
) -> str:
    """Map physical hand to MediaPipe's selfie-oriented handedness label."""
    hand = physical_hand.strip().lower()
    if hand not in {"left", "right"}:
        raise ValueError("physical_hand must be left or right")
    if input_mirrored:
        return hand.title()
    return "Left" if hand == "right" else "Right"


def select_strict_hand(
    result: Any,
    config: TrackConfig,
) -> tuple[np.ndarray, float] | None:
    if result.multi_hand_landmarks is None or result.multi_handedness is None:
        return None
    expected_label = mediapipe_label_for_physical_hand(
        config.physical_hand, config.input_mirrored
    )
    selected: tuple[np.ndarray, float] | None = None
    best_score = -1.0
    for landmarks, handedness in zip(
        result.multi_hand_landmarks, result.multi_handedness
    ):
        classification = handedness.classification[0]
        label = str(classification.label)
        score = float(classification.score)
        if label != expected_label or score < config.handedness_threshold:
            continue
        if score > best_score:
            selected = (
                np.asarray(
                    [[point.x, point.y, point.z] for point in landmarks.landmark],
                    dtype=np.float32,
                ),
                score,
            )
            best_score = score
    return selected


def _optional_dependencies():
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise ImportError(
            "opencv-python and mediapipe are required for track extraction"
        ) from exc
    return cv2, mp


def video_signature(path: str | Path) -> dict[str, int]:
    stat = Path(path).stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def track_fingerprint(config: TrackConfig, video_path: str | Path) -> str:
    payload = {
        "config": config.fingerprint(),
        "video": video_signature(video_path),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


def track_cache_path(config: TrackConfig, source: str, video_id: str) -> Path:
    safe_name = video_id.replace("/", "__")
    return (
        Path(config.cache_dir).expanduser()
        / config.fingerprint()
        / source
        / f"{safe_name}.npz"
    )


def extract_track(
    video_path: str | Path,
    config: TrackConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    cv2, mp = _optional_dependencies()
    capture = cv2.VideoCapture(str(Path(video_path).expanduser()))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 1e-3:
        source_fps = 30.0

    landmarks: list[np.ndarray] = []
    quality: list[float] = []
    valid_mask: list[float] = []
    with mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=config.max_num_hands,
        model_complexity=config.model_complexity,
        min_detection_confidence=config.handedness_threshold,
        min_tracking_confidence=config.handedness_threshold,
    ) as hands:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame.flags.writeable = False
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            selected = select_strict_hand(hands.process(rgb), config)
            if selected is None:
                landmarks.append(np.full((21, 3), np.nan, dtype=np.float32))
                quality.append(0.0)
                valid_mask.append(0.0)
            else:
                points, score = selected
                landmarks.append(points)
                quality.append(score)
                valid_mask.append(1.0)
    capture.release()
    if not landmarks:
        raise ValueError(f"No frames decoded from {video_path}")
    return (
        np.stack(landmarks).astype(np.float32),
        np.asarray(quality, dtype=np.float32),
        np.asarray(valid_mask, dtype=np.float32),
        source_fps,
    )


def save_track(
    path: str | Path,
    landmarks: np.ndarray,
    quality: np.ndarray,
    valid_mask: np.ndarray,
    source_fps: float,
    metadata: dict[str, Any],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.npz")
    np.savez_compressed(
        temporary,
        landmarks=np.asarray(landmarks, dtype=np.float32),
        tracking_quality=np.asarray(quality, dtype=np.float32),
        valid_mask=np.asarray(valid_mask, dtype=np.float32),
        source_fps=np.asarray(source_fps, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    temporary.replace(output)
    return output


def load_track(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            "landmarks": np.asarray(archive["landmarks"], dtype=np.float32),
            "tracking_quality": np.asarray(
                archive["tracking_quality"], dtype=np.float32
            ),
            "valid_mask": np.asarray(archive["valid_mask"], dtype=np.float32),
            "source_fps": float(archive["source_fps"]),
            "metadata": json.loads(str(archive["metadata_json"].item())),
        }


def extract_inventory_entry(entry: Any, config: TrackConfig, resume: bool = True) -> dict[str, Any]:
    output = track_cache_path(config, entry.source, entry.video_id)
    fingerprint = track_fingerprint(config, entry.video_path)
    if resume and output.is_file():
        try:
            existing = load_track(output)
        except (ValueError, KeyError, json.JSONDecodeError):
            existing = None
        if existing and existing["metadata"].get("track_fingerprint") == fingerprint:
            return {
                "video_id": entry.video_id,
                "source": entry.source,
                "status": "cached",
                "track_path": str(output),
            }
    landmarks, quality, valid_mask, source_fps = extract_track(
        entry.video_path, config
    )
    save_track(
        output,
        landmarks,
        quality,
        valid_mask,
        source_fps,
        {
            "video_id": entry.video_id,
            "source": entry.source,
            "video_path": str(entry.video_path),
            "physical_hand": config.physical_hand,
            "input_mirrored": config.input_mirrored,
            "mediapipe_label": mediapipe_label_for_physical_hand(
                config.physical_hand, config.input_mirrored
            ),
            "track_config_fingerprint": config.fingerprint(),
            "track_fingerprint": fingerprint,
            "video_signature": video_signature(entry.video_path),
        },
    )
    return {
        "video_id": entry.video_id,
        "source": entry.source,
        "status": "completed",
        "track_path": str(output),
    }


def _extract_worker(index: int, entry: Any, config: TrackConfig, resume: bool):
    return index, extract_inventory_entry(entry, config, resume)


def extract_inventory(
    entries: Iterable[Any],
    config: TrackConfig,
    resume: bool = True,
) -> list[dict[str, Any]]:
    eligible = [entry for entry in entries if entry.status in {"ok", "ready"}]
    if not eligible:
        return []
    results: list[dict[str, Any] | None] = [None] * len(eligible)
    workers = min(max(1, config.num_workers), len(eligible))
    if workers == 1:
        for index, entry in enumerate(eligible):
            try:
                results[index] = extract_inventory_entry(entry, config, resume)
            except Exception as exc:
                results[index] = {
                    "video_id": entry.video_id,
                    "source": entry.source,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        return [item for item in results if item is not None]

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = {
            executor.submit(_extract_worker, index, entry, config, resume): index
            for index, entry in enumerate(eligible)
        }
        for future in as_completed(futures):
            index = futures[future]
            entry = eligible[index]
            try:
                _, results[index] = future.result()
            except Exception as exc:
                results[index] = {
                    "video_id": entry.video_id,
                    "source": entry.source,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            done = sum(item is not None for item in results)
            print(f"Track extraction progress: {done}/{len(eligible)}")
    return [item for item in results if item is not None]


def write_handedness_previews(
    entry: Any,
    config: TrackConfig,
    output_dir: str | Path,
    count: int = 6,
) -> list[Path]:
    cv2, mp = _optional_dependencies()
    capture = cv2.VideoCapture(str(entry.video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    positions = np.linspace(0, max(frame_count - 1, 0), max(1, count), dtype=int)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=config.max_num_hands,
        model_complexity=config.model_complexity,
        min_detection_confidence=config.handedness_threshold,
    ) as hands:
        for ordinal, position in enumerate(positions):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
            ok, frame = capture.read()
            if not ok:
                continue
            selected = select_strict_hand(
                hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), config
            )
            if selected is not None:
                points, score = selected
                height, width = frame.shape[:2]
                for x, y, _ in points:
                    cv2.circle(
                        frame,
                        (int(x * width), int(y * height)),
                        3,
                        (0, 255, 0),
                        -1,
                    )
                cv2.putText(
                    frame,
                    f"physical {config.physical_hand} score={score:.3f}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    f"physical {config.physical_hand}: NOT FOUND",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
            path = output_root / f"{entry.video_id}__{ordinal:02d}.jpg"
            cv2.imwrite(str(path), frame)
            written.append(path)
    capture.release()
    return written
