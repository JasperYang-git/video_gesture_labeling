from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.schema import (
    BACKGROUND_ID,
    FEATURE_DIM,
    IGNORE_INDEX,
    SequenceRecord,
    load_sequence,
)
from utils.split import load_manifest


@dataclass(frozen=True)
class WindowSample:
    video_id: str
    start: int
    end: int
    is_background: bool
    window_role: str = "action"


class GestureWindowDataset(Dataset):
    """Generate training windows on demand from full-video sequences."""

    def __init__(
        self,
        records: Sequence[SequenceRecord],
        window_size: int,
        stride: int,
        keep_background_prob: float | None = None,
        keep_background_probability: float | None = None,
        seed: int = 42,
        require_labels: bool = True,
        dynamic_action_windows: bool = False,
        action_windows_per_segment: int = 3,
        hard_negative_ratio: float = 0.15,
        hard_negative_max_windows_per_video: int = 20,
    ) -> None:
        if window_size <= 0 or stride <= 0:
            raise ValueError("window_size and stride must be positive")
        if keep_background_prob is None:
            keep_background_prob = (
                0.0 if keep_background_probability is None else keep_background_probability
            )
        if not 0.0 <= keep_background_prob <= 1.0:
            raise ValueError("keep_background_prob must be in [0, 1]")
        if hard_negative_ratio < 0.0:
            raise ValueError("hard_negative_ratio must be non-negative")
        if hard_negative_max_windows_per_video < 0:
            raise ValueError(
                "hard_negative_max_windows_per_video must be non-negative"
            )

        self.window_size = window_size
        self.stride = stride
        self.keep_background_prob = keep_background_prob
        self.seed = seed
        self.dynamic_action_windows = dynamic_action_windows
        self.action_windows_per_segment = max(1, int(action_windows_per_segment))
        self.hard_negative_ratio = float(hard_negative_ratio)
        self.hard_negative_max_windows_per_video = int(
            hard_negative_max_windows_per_video
        )
        self.records = list(records)
        self.record_by_id = {record.video_id: record for record in self.records}
        self.windows: list[WindowSample] = []
        for record in self.records:
            if require_labels and record.labels is None:
                raise ValueError(f"{record.video_id}: labels are required for training")
        self.set_epoch(0)
        if not self.windows:
            raise ValueError("No training windows were generated")

    def set_epoch(self, epoch: int) -> None:
        rng = np.random.default_rng(self.seed + int(epoch))
        windows: list[WindowSample] = []
        hard_negative_candidates: list[WindowSample] = []
        for record in self.records:
            if self.hard_negative_ratio > 0.0 and _is_hard_negative_record(record):
                candidates = _windows_for_record(
                    record,
                    self.window_size,
                    self.stride,
                    keep_background_prob=1.0,
                    rng=rng,
                    background_role="hard_negative",
                )
                if len(candidates) > self.hard_negative_max_windows_per_video:
                    selected = rng.choice(
                        len(candidates),
                        size=self.hard_negative_max_windows_per_video,
                        replace=False,
                    )
                    candidates = [candidates[index] for index in sorted(selected)]
                hard_negative_candidates.extend(candidates)
                continue
            if self.dynamic_action_windows and record.labels is not None:
                windows.extend(
                    _dynamic_action_windows_for_record(
                        record,
                        self.window_size,
                        self.action_windows_per_segment,
                        self.keep_background_prob,
                        self.stride,
                        rng,
                    )
                )
            else:
                windows.extend(
                    _windows_for_record(
                        record,
                        self.window_size,
                        self.stride,
                        self.keep_background_prob,
                        rng,
                    )
                )
        action_window_count = sum(
            window.window_role == "action" for window in windows
        )
        hard_negative_count = _hard_negative_target_count(
            action_window_count,
            len(hard_negative_candidates),
            self.hard_negative_ratio,
        )
        if hard_negative_count < len(hard_negative_candidates):
            selected = rng.choice(
                len(hard_negative_candidates),
                size=hard_negative_count,
                replace=False,
            )
            hard_negative_candidates = [
                hard_negative_candidates[index] for index in sorted(selected)
            ]
        windows.extend(hard_negative_candidates)
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        sample = self.windows[index]
        record = self.record_by_id[sample.video_id]
        features, labels, valid_mask = _slice_window(
            record,
            sample.start,
            self.window_size,
        )
        return {
            "features": torch.from_numpy(features),
            "labels": torch.from_numpy(labels),
            "valid_mask": torch.from_numpy(valid_mask),
            "video_id": record.video_id,
            "start": sample.start,
            "window_role": sample.window_role,
        }

    @property
    def frame_labels(self) -> list[int]:
        labels: list[int] = []
        for sample in self.windows:
            record = self.record_by_id[sample.video_id]
            if record.labels is None:
                continue
            window_labels = _padded_labels(record, sample.start, self.window_size)
            labels.extend(int(item) for item in window_labels if item != IGNORE_INDEX)
        return labels


def _windows_for_record(
    record: SequenceRecord,
    window_size: int,
    stride: int,
    keep_background_prob: float,
    rng: np.random.Generator,
    background_role: str = "background",
) -> list[WindowSample]:
    frames = record.num_frames
    if frames <= 0:
        return []
    starts = list(range(0, max(frames - window_size, 0) + 1, stride))
    if not starts or starts[-1] != max(frames - window_size, 0):
        starts.append(max(frames - window_size, 0))

    windows: list[WindowSample] = []
    for start in starts:
        end = min(start + window_size, frames)
        if record.labels is None:
            is_background = False
        else:
            window_labels = record.labels[start:end]
            is_background = bool(np.all(window_labels == BACKGROUND_ID))
        if is_background and rng.random() > keep_background_prob:
            continue
        windows.append(
            WindowSample(
                video_id=record.video_id,
                start=start,
                end=end,
                is_background=is_background,
                window_role=background_role if is_background else "action",
            )
        )
    return windows


def _is_hard_negative_record(record: SequenceRecord) -> bool:
    if record.labels is None or not bool(np.all(record.labels == BACKGROUND_ID)):
        return False
    polarity = str(record.metadata.get("polarity", "")).strip().lower()
    eligible = record.metadata.get("hard_negative_eligible", False)
    if isinstance(eligible, str):
        eligible = eligible.strip().lower() in {"1", "true", "yes", "y"}
    return polarity == "negative" and bool(eligible)


def _hard_negative_target_count(
    action_window_count: int,
    candidate_count: int,
    ratio: float,
) -> int:
    if candidate_count <= 0 or ratio <= 0.0:
        return 0
    if action_window_count <= 0:
        return min(1, candidate_count)
    return min(candidate_count, max(1, math.ceil(action_window_count * ratio)))


def _action_segments(labels: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, label in enumerate(labels.tolist()):
        if label != BACKGROUND_ID and start is None:
            start = index
        if start is not None and (
            label == BACKGROUND_ID
            or (index > start and labels[index - 1] != label)
        ):
            segments.append((start, index))
            start = index if label != BACKGROUND_ID else None
    if start is not None:
        segments.append((start, len(labels)))
    return segments


def _dynamic_action_windows_for_record(
    record: SequenceRecord,
    window_size: int,
    samples_per_segment: int,
    keep_background_prob: float,
    stride: int,
    rng: np.random.Generator,
) -> list[WindowSample]:
    assert record.labels is not None
    max_start = max(record.num_frames - window_size, 0)
    selected: set[int] = set()
    for segment_start, segment_end in _action_segments(record.labels):
        low = max(0, segment_end - window_size)
        high = min(segment_start, max_start)
        if high < low:
            center = min(max((segment_start + segment_end - window_size) // 2, 0), max_start)
            selected.add(center)
            continue
        anchors = np.linspace(low, high, samples_per_segment)
        jitter = max(1, stride // 2)
        for anchor in anchors:
            start = int(round(anchor)) + int(rng.integers(-jitter, jitter + 1))
            selected.add(min(max(start, low), high))

    windows = [
        WindowSample(
            video_id=record.video_id,
            start=start,
            end=min(start + window_size, record.num_frames),
            is_background=False,
            window_role="action",
        )
        for start in sorted(selected)
    ]
    if keep_background_prob > 0:
        existing = {window.start for window in windows}
        for candidate in _windows_for_record(
            record, window_size, stride, keep_background_prob, rng
        ):
            if candidate.is_background and candidate.start not in existing:
                windows.append(candidate)
    return sorted(windows, key=lambda item: item.start)


def _slice_window(
    record: SequenceRecord,
    start: int,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    end = min(start + window_size, record.num_frames)
    actual = end - max(start, 0)
    features = np.zeros((FEATURE_DIM, window_size), dtype=np.float32)
    valid_mask = np.zeros(window_size, dtype=np.float32)
    labels = np.full(window_size, IGNORE_INDEX, dtype=np.int64)
    if actual > 0:
        source_start = max(start, 0)
        features[:, :actual] = record.features[:, source_start:end]
        valid_mask[:actual] = record.valid_mask[source_start:end]
        if record.labels is not None:
            labels[:actual] = record.labels[source_start:end]
        if actual < window_size:
            features[:, actual:] = features[:, actual - 1 : actual]
            valid_mask[actual:] = 0.0
    return features, labels, valid_mask


def _padded_labels(record: SequenceRecord, start: int, window_size: int) -> np.ndarray:
    _, labels, _ = _slice_window(record, start, window_size)
    return labels


def collate_windows(
    batch: list[dict[str, torch.Tensor | str | int]],
) -> dict[str, torch.Tensor | list[str] | list[int]]:
    return {
        "features": torch.stack([item["features"] for item in batch], dim=0),
        "labels": torch.stack([item["labels"] for item in batch], dim=0),
        "valid_mask": torch.stack([item["valid_mask"] for item in batch], dim=0),
        "video_id": [str(item["video_id"]) for item in batch],
        "start": [int(item["start"]) for item in batch],
        "window_role": [str(item["window_role"]) for item in batch],
    }


def build_train_loaders(config: dict) -> tuple[DataLoader, DataLoader, list[Path], list[Path]]:
    data_config = config["data"]
    splits = load_manifest(data_config["manifest_path"])
    train_files = splits["train"]
    validation_files = splits["validation"]
    if not train_files:
        raise FileNotFoundError("Training split is empty")
    if not validation_files:
        raise FileNotFoundError("Validation split is empty")

    train_records = [load_sequence(path) for path in train_files]
    validation_records = [load_sequence(path) for path in validation_files]
    dataset_kwargs = {
        "window_size": int(data_config["window_size"]),
        "stride": int(data_config["stride"]),
        "seed": int(config["seed"]),
    }
    train_dataset = GestureWindowDataset(
        train_records,
        keep_background_prob=float(data_config.get("keep_background_prob", 0.0)),
        dynamic_action_windows=bool(data_config.get("dynamic_action_windows", True)),
        action_windows_per_segment=int(
            data_config.get("action_windows_per_segment", 3)
        ),
        hard_negative_ratio=float(data_config.get("hard_negative_ratio", 0.15)),
        hard_negative_max_windows_per_video=int(
            data_config.get("hard_negative_max_windows_per_video", 20)
        ),
        **dataset_kwargs,
    )
    validation_dataset = GestureWindowDataset(
        validation_records,
        keep_background_prob=1.0,
        hard_negative_ratio=0.0,
        **dataset_kwargs,
    )
    loader_kwargs = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(data_config.get("num_workers", 0)),
        "collate_fn": collate_windows,
    }
    generator = torch.Generator().manual_seed(int(config["seed"]))
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, validation_loader, train_files, validation_files
