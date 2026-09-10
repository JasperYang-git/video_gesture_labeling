from __future__ import annotations

from dataclasses import dataclass
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
    ) -> None:
        if window_size <= 0 or stride <= 0:
            raise ValueError("window_size and stride must be positive")
        if keep_background_prob is None:
            keep_background_prob = (
                0.0 if keep_background_probability is None else keep_background_probability
            )
        if not 0.0 <= keep_background_prob <= 1.0:
            raise ValueError("keep_background_prob must be in [0, 1]")

        self.window_size = window_size
        self.stride = stride
        self.records = list(records)
        self.record_by_id = {record.video_id: record for record in self.records}
        rng = np.random.default_rng(seed)
        self.windows: list[WindowSample] = []
        for record in self.records:
            if require_labels and record.labels is None:
                raise ValueError(f"{record.video_id}: labels are required for training")
            self.windows.extend(
                _windows_for_record(
                    record,
                    window_size,
                    stride,
                    keep_background_prob,
                    rng,
                )
            )
        if not self.windows:
            raise ValueError("No training windows were generated")

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
            )
        )
    return windows


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
        **dataset_kwargs,
    )
    validation_dataset = GestureWindowDataset(
        validation_records,
        keep_background_prob=1.0,
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
