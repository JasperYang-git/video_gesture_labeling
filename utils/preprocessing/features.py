from __future__ import annotations

import numpy as np

from utils.schema import (
    BACKGROUND_ID,
    COORD_DIM,
    FEATURE_DIM,
    SCORE_INDEX,
    VALID_MASK_INDEX,
)


WRIST_INDEX = 0
MIDDLE_MCP_INDEX = 9


class OneEuroFilter:
    """Low-latency low-pass filter for jittery landmark tracks."""

    def __init__(
        self,
        channels: int,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev = np.zeros(channels, dtype=np.float32)
        self._dx_prev = np.zeros(channels, dtype=np.float32)
        self._initialized = False

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> np.ndarray | float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._initialized = False
        self._x_prev.fill(0.0)
        self._dx_prev.fill(0.0)

    def __call__(self, values: np.ndarray, dt: float) -> np.ndarray:
        sample = np.asarray(values, dtype=np.float32)
        if sample.shape != self._x_prev.shape:
            raise ValueError(
                f"Expected shape {self._x_prev.shape}, got {sample.shape}"
            )
        if dt <= 0:
            raise ValueError("dt must be positive")
        if not self._initialized:
            self._x_prev = sample.copy()
            self._dx_prev.fill(0.0)
            self._initialized = True
            return sample.copy()

        dx = (sample - self._x_prev) / dt
        dx_hat = (
            self._alpha(self.d_cutoff, dt) * dx
            + (1.0 - self._alpha(self.d_cutoff, dt)) * self._dx_prev
        )
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        x_hat = self._alpha(cutoff, dt) * sample + (1.0 - self._alpha(cutoff, dt)) * self._x_prev
        self._x_prev = x_hat.astype(np.float32, copy=False)
        self._dx_prev = np.asarray(dx_hat, dtype=np.float32)
        return self._x_prev.copy()


def landmarks_to_vector(landmarks: np.ndarray) -> np.ndarray:
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (21, 3):
        raise ValueError(f"Expected landmarks with shape (21, 3), got {points.shape}")
    return points.reshape(COORD_DIM)


def center_hand_landmarks(landmarks: np.ndarray) -> tuple[np.ndarray, float]:
    """Return wrist-centred coordinates and the unnormalised palm scale."""
    points = np.asarray(landmarks, dtype=np.float32).reshape(21, 3).copy()
    scale = palm_size(points)
    points -= points[WRIST_INDEX]
    return points.reshape(COORD_DIM), scale


def palm_size(landmarks: np.ndarray) -> float:
    points = np.asarray(landmarks, dtype=np.float32).reshape(21, 3)
    wrist = points[WRIST_INDEX]
    middle_mcp = points[MIDDLE_MCP_INDEX]
    scale = float(np.linalg.norm(middle_mcp - wrist))
    return max(scale, 1e-3)


def normalize_hand_landmarks(landmarks: np.ndarray) -> np.ndarray:
    centred, scale = center_hand_landmarks(landmarks)
    return centred / scale


def _bucket_bounds(
    num_frames: int,
    source_fps: float,
    target_fps: float,
    offset_frames: int = 0,
) -> list[tuple[int, int]]:
    if num_frames <= 0:
        return []
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("fps must be positive")
    if target_fps > source_fps + 1e-6:
        raise ValueError("target_fps cannot exceed source_fps")
    if offset_frames < 0:
        raise ValueError("offset_frames must be non-negative")
    ratio = source_fps / target_fps
    bounds: list[tuple[int, int]] = []
    index = 0
    while True:
        start = offset_frames + int(np.floor(index * ratio))
        if start >= num_frames:
            break
        end = offset_frames + int(np.floor((index + 1) * ratio))
        end = min(num_frames, max(end, start + 1))
        bounds.append((start, end))
        index += 1
    return bounds


def _robust_mean(values: np.ndarray) -> np.ndarray:
    """Trim one sample from each tail when a bucket is large enough."""
    if len(values) < 5:
        return values.mean(axis=0)
    ordered = np.sort(values, axis=0)
    return ordered[1:-1].mean(axis=0)


def downsample_sequence(
    values: np.ndarray,
    source_fps: float,
    target_fps: float,
) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError("values must have shape [T, C]")
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("fps must be positive")
    if target_fps > source_fps + 1e-6:
        raise ValueError("target_fps cannot exceed source_fps")
    bounds = _bucket_bounds(len(values), source_fps, target_fps)
    output = np.zeros((len(bounds), values.shape[1]), dtype=np.float32)
    for index, (start, end) in enumerate(bounds):
        output[index] = values[start:end].mean(axis=0)
    return output


def align_labels_to_target_fps(
    labels: np.ndarray,
    source_fps: float,
    target_fps: float,
    background_id: int = BACKGROUND_ID,
) -> np.ndarray:
    if labels.ndim != 1:
        raise ValueError("labels must be 1-D")
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("fps must be positive")
    bounds = _bucket_bounds(len(labels), source_fps, target_fps)
    output = np.full(len(bounds), background_id, dtype=np.int64)
    for index, (start, end) in enumerate(bounds):
        window = labels[start:end]
        values, counts = np.unique(window, return_counts=True)
        output[index] = int(values[int(np.argmax(counts))])
    return output


def aggregate_hand_tracks(
    centered_coordinates: np.ndarray,
    palm_sizes: np.ndarray,
    tracking_quality: np.ndarray,
    valid_mask: np.ndarray,
    source_fps: float,
    target_fps: float,
    offset_frames: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate source-FPS tracks, then apply robust palm normalisation."""
    coords = np.asarray(centered_coordinates, dtype=np.float32)
    palms = np.asarray(palm_sizes, dtype=np.float32)
    quality = np.asarray(tracking_quality, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != COORD_DIM:
        raise ValueError(f"centered_coordinates must have shape [T, {COORD_DIM}]")
    frames = len(coords)
    if palms.shape != (frames,) or quality.shape != (frames,) or mask.shape != (frames,):
        raise ValueError("track arrays must share the same time dimension")

    bounds = _bucket_bounds(frames, source_fps, target_fps, offset_frames)
    normalized = np.zeros((len(bounds), COORD_DIM), dtype=np.float32)
    quality_out = np.zeros(len(bounds), dtype=np.float32)
    mask_out = np.zeros(len(bounds), dtype=np.float32)
    for index, (start, end) in enumerate(bounds):
        bucket_coords = coords[start:end]
        bucket_palms = palms[start:end]
        aggregated_coords = _robust_mean(bucket_coords)
        valid_palms = bucket_palms[np.isfinite(bucket_palms) & (bucket_palms > 1e-3)]
        scale = float(np.median(valid_palms)) if len(valid_palms) else 1.0
        normalized[index] = aggregated_coords / max(scale, 1e-3)
        quality_out[index] = float(quality[start:end].mean())
        mask_out[index] = float(mask[start:end].mean() > 0.5)
    return normalized, quality_out, mask_out


def build_feature_matrix(
    coordinates: np.ndarray,
    tracking_quality: np.ndarray,
    valid_mask: np.ndarray,
    fps: float,
) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != COORD_DIM:
        raise ValueError(f"coordinates must have shape [T, {COORD_DIM}]")
    frames = coords.shape[0]
    if tracking_quality.shape != (frames,) or valid_mask.shape != (frames,):
        raise ValueError("tracking_quality and valid_mask must match the coordinate length")
    if fps <= 0:
        raise ValueError("fps must be positive")

    velocity = np.zeros_like(coords)
    if frames > 1:
        dt = 1.0 / fps
        velocity[1:] = (coords[1:] - coords[:-1]) / dt
    features = np.zeros((FEATURE_DIM, frames), dtype=np.float32)
    features[SCORE_INDEX] = np.asarray(tracking_quality, dtype=np.float32)
    features[VALID_MASK_INDEX] = np.asarray(valid_mask, dtype=np.float32)
    features[2 : 2 + COORD_DIM] = coords.T
    features[2 + COORD_DIM :] = velocity.T
    return features
