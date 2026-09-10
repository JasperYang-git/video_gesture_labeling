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


def palm_size(landmarks: np.ndarray) -> float:
    points = np.asarray(landmarks, dtype=np.float32).reshape(21, 3)
    wrist = points[WRIST_INDEX]
    middle_mcp = points[MIDDLE_MCP_INDEX]
    scale = float(np.linalg.norm(middle_mcp - wrist))
    return max(scale, 1e-3)


def normalize_hand_landmarks(landmarks: np.ndarray) -> np.ndarray:
    points = np.asarray(landmarks, dtype=np.float32).reshape(21, 3).copy()
    scale = palm_size(points)
    points -= points[WRIST_INDEX]
    points /= scale
    return points.reshape(COORD_DIM)


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
    ratio = source_fps / target_fps
    num_out = max(1, int(np.floor(len(values) / ratio)))
    output = np.zeros((num_out, values.shape[1]), dtype=np.float32)
    for index in range(num_out):
        start = int(np.floor(index * ratio))
        end = int(np.floor((index + 1) * ratio))
        end = max(end, start + 1)
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
    ratio = source_fps / target_fps
    num_out = max(1, int(np.floor(len(labels) / ratio)))
    output = np.full(num_out, background_id, dtype=np.int64)
    for index in range(num_out):
        start = int(np.floor(index * ratio))
        end = int(np.floor((index + 1) * ratio))
        end = max(end, start + 1)
        window = labels[start:end]
        values, counts = np.unique(window, return_counts=True)
        output[index] = int(values[int(np.argmax(counts))])
    return output


def build_feature_matrix(
    coordinates: np.ndarray,
    scores: np.ndarray,
    valid_mask: np.ndarray,
    fps: float,
) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != COORD_DIM:
        raise ValueError(f"coordinates must have shape [T, {COORD_DIM}]")
    frames = coords.shape[0]
    if scores.shape != (frames,) or valid_mask.shape != (frames,):
        raise ValueError("scores and valid_mask must match the coordinate length")
    if fps <= 0:
        raise ValueError("fps must be positive")

    velocity = np.zeros_like(coords)
    if frames > 1:
        dt = 1.0 / fps
        velocity[1:] = (coords[1:] - coords[:-1]) / dt
    features = np.zeros((FEATURE_DIM, frames), dtype=np.float32)
    features[SCORE_INDEX] = np.asarray(scores, dtype=np.float32)
    features[VALID_MASK_INDEX] = np.asarray(valid_mask, dtype=np.float32)
    features[2 : 2 + COORD_DIM] = coords.T
    features[2 + COORD_DIM :] = velocity.T
    return features
