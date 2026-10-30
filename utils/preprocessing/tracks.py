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
        if self.physical_hand.lower() not in {"left", "right", "auto"}:
            raise ValueError("physical_hand must be left, right or auto")
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


RIGHT_SIDE = 1
LEFT_SIDE = -1
NO_HAND = 0


def _confident_hands(
    result: Any,
    config: TrackConfig,
) -> list[tuple[np.ndarray, float, int]]:
    """Every hand above the handedness threshold as (points, score, physical side)."""
    if result.multi_hand_landmarks is None or result.multi_handedness is None:
        return []
    right_label = mediapipe_label_for_physical_hand("right", config.input_mirrored)
    hands: list[tuple[np.ndarray, float, int]] = []
    for landmarks, handedness in zip(
        result.multi_hand_landmarks, result.multi_handedness
    ):
        classification = handedness.classification[0]
        score = float(classification.score)
        if score < config.handedness_threshold:
            continue
        side = RIGHT_SIDE if str(classification.label) == right_label else LEFT_SIDE
        points = np.asarray(
            [[point.x, point.y, point.z] for point in landmarks.landmark],
            dtype=np.float32,
        )
        hands.append((points, score, side))
    return hands


def select_strict_hand(
    result: Any,
    config: TrackConfig,
) -> tuple[np.ndarray, float] | None:
    expected_side = RIGHT_SIDE if config.physical_hand.lower() == "right" else LEFT_SIDE
    candidates = [
        (points, score)
        for points, score, side in _confident_hands(result, config)
        if side == expected_side
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[1])


def select_auto_hand(
    result: Any,
    config: TrackConfig,
    previous_side: int = NO_HAND,
) -> tuple[np.ndarray, float, int] | None:
    """Pick either hand, sticking to the previous frame's side while it stays visible."""
    hands = _confident_hands(result, config)
    if not hands:
        return None
    same_side = [hand for hand in hands if hand[2] == previous_side]
    return max(same_side or hands, key=lambda item: item[1])


def select_hand(
    result: Any,
    config: TrackConfig,
    previous_side: int = NO_HAND,
) -> tuple[np.ndarray, float, int] | None:
    if config.physical_hand.lower() == "auto":
        return select_auto_hand(result, config, previous_side)
    selected = select_strict_hand(result, config)
    if selected is None:
        return None
    side = RIGHT_SIDE if config.physical_hand.lower() == "right" else LEFT_SIDE
    return selected[0], selected[1], side


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


def probe_frame_size(video_path: str | Path) -> tuple[int, int] | None:
    """Decoded (width, height) of the first frame, or None when it cannot be read.

    The decoded frame is used instead of CAP_PROP_FRAME_WIDTH because phone videos
    carry a rotation tag that OpenCV applies when decoding but not to the properties.
    """
    try:
        import cv2
    except ImportError:
        return None
    capture = cv2.VideoCapture(str(Path(video_path).expanduser()))
    try:
        if not capture.isOpened():
            return None
        ok, frame = capture.read()
        if not ok or frame is None:
            return None
        height, width = frame.shape[:2]
        return int(width), int(height)
    finally:
        capture.release()


def extract_track(
    video_path: str | Path,
    config: TrackConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, tuple[int, int] | None]:
    """Return landmarks, quality, valid mask, fps, per-frame hand side and frame size."""
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
    hand_side: list[int] = []
    frame_size: tuple[int, int] | None = None
    previous_side = NO_HAND
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
            if frame_size is None:
                height, width = frame.shape[:2]
                frame_size = (int(width), int(height))
            frame.flags.writeable = False
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            selected = select_hand(hands.process(rgb), config, previous_side)
            if selected is None:
                landmarks.append(np.full((21, 3), np.nan, dtype=np.float32))
                quality.append(0.0)
                valid_mask.append(0.0)
                hand_side.append(NO_HAND)
            else:
                points, score, side = selected
                landmarks.append(points)
                quality.append(score)
                valid_mask.append(1.0)
                hand_side.append(side)
                previous_side = side
    capture.release()
    if not landmarks:
        raise ValueError(f"No frames decoded from {video_path}")
    return (
        np.stack(landmarks).astype(np.float32),
        np.asarray(quality, dtype=np.float32),
        np.asarray(valid_mask, dtype=np.float32),
        source_fps,
        np.asarray(hand_side, dtype=np.int8),
        frame_size,
    )


def save_track(
    path: str | Path,
    landmarks: np.ndarray,
    quality: np.ndarray,
    valid_mask: np.ndarray,
    source_fps: float,
    metadata: dict[str, Any],
    hand_side: np.ndarray | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.npz")
    arrays: dict[str, np.ndarray] = {
        "landmarks": np.asarray(landmarks, dtype=np.float32),
        "tracking_quality": np.asarray(quality, dtype=np.float32),
        "valid_mask": np.asarray(valid_mask, dtype=np.float32),
        "source_fps": np.asarray(source_fps, dtype=np.float32),
        "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False)),
    }
    if hand_side is not None:
        arrays["hand_side"] = np.asarray(hand_side, dtype=np.int8)
    np.savez_compressed(temporary, **arrays)
    temporary.replace(output)
    return output


def _default_hand_side(valid_mask: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    """Hand side for tracks cached before it was recorded: always the strict hand."""
    side = LEFT_SIDE if str(metadata.get("physical_hand", "right")).lower() == "left" else RIGHT_SIDE
    return np.where(valid_mask > 0.5, side, NO_HAND).astype(np.int8)


def load_track(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        valid_mask = np.asarray(archive["valid_mask"], dtype=np.float32)
        metadata = json.loads(str(archive["metadata_json"].item()))
        hand_side = (
            np.asarray(archive["hand_side"], dtype=np.int8)
            if "hand_side" in archive.files
            else _default_hand_side(valid_mask, metadata)
        )
        return {
            "landmarks": np.asarray(archive["landmarks"], dtype=np.float32),
            "tracking_quality": np.asarray(
                archive["tracking_quality"], dtype=np.float32
            ),
            "valid_mask": valid_mask,
            "hand_side": hand_side,
            "source_fps": float(archive["source_fps"]),
            "metadata": metadata,
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
    landmarks, quality, valid_mask, source_fps, hand_side, frame_size = extract_track(
        entry.video_path, config
    )
    metadata: dict[str, Any] = {
        "video_id": entry.video_id,
        "source": entry.source,
        "video_path": str(entry.video_path),
        "physical_hand": config.physical_hand,
        "input_mirrored": config.input_mirrored,
        "mediapipe_label": (
            "any"
            if config.physical_hand.lower() == "auto"
            else mediapipe_label_for_physical_hand(
                config.physical_hand, config.input_mirrored
            )
        ),
        "track_config_fingerprint": config.fingerprint(),
        "track_fingerprint": fingerprint,
        "video_signature": video_signature(entry.video_path),
    }
    if frame_size is not None:
        metadata["frame_width"], metadata["frame_height"] = frame_size
    save_track(
        output,
        landmarks,
        quality,
        valid_mask,
        source_fps,
        metadata,
        hand_side=hand_side,
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


def lost_tracking_positions(
    valid_mask: np.ndarray,
    lost_count: int = 4,
    valid_count: int = 2,
) -> list[int]:
    """Frame positions that explain a low detection rate: inside the worst gap, plus controls.

    Sampling uniformly across a video mostly returns frames that already work. The
    interesting frames are the ones inside the longest stretch where MediaPipe found
    nothing, which is where you can see whether the hand is occluded, out of frame, or
    simply being read as the wrong hand. A couple of tracked frames come along as a
    baseline for comparison.
    """
    valid = np.asarray(valid_mask) > 0.5
    if not len(valid):
        return []

    best_start = best_length = current_start = 0
    current_length = 0
    for index, is_valid in enumerate(valid):
        if is_valid:
            current_length = 0
            continue
        if current_length == 0:
            current_start = index
        current_length += 1
        if current_length > best_length:
            best_start, best_length = current_start, current_length

    positions: list[int] = []
    if best_length > 0 and lost_count > 0:
        positions.extend(
            np.linspace(
                best_start, best_start + best_length - 1, lost_count, dtype=int
            ).tolist()
        )
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) and valid_count > 0:
        picks = np.linspace(0, len(valid_indices) - 1, valid_count, dtype=int)
        positions.extend(int(valid_indices[pick]) for pick in picks)
    return sorted(set(positions))


def write_handedness_previews(
    entry: Any,
    config: TrackConfig,
    output_dir: str | Path,
    count: int = 6,
    positions: list[int] | None = None,
) -> list[Path]:
    cv2, mp = _optional_dependencies()
    capture = cv2.VideoCapture(str(entry.video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if positions is None:
        positions = np.linspace(
            0, max(frame_count - 1, 0), max(1, count), dtype=int
        ).tolist()
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
            selected = select_hand(
                hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), config
            )
            if selected is not None:
                points, score, side = selected
                side_name = "right" if side == RIGHT_SIDE else "left"
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
                    f"physical {side_name} score={score:.3f}",
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
            cv2.putText(
                frame,
                f"frame {int(position)}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            path = output_root / f"{entry.video_id}__{ordinal:02d}_f{int(position):06d}.jpg"
            cv2.imwrite(str(path), frame)
            written.append(path)
    capture.release()
    return written
