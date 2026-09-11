from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from model import build_model
from utils.metrics import ActionSegment, labels_to_segments
from utils.schema import (
    FEATURE_DIM,
    FEATURE_NAMES,
    SCHEMA_VERSION,
    SequenceRecord,
    class_id_to_name,
)


def load_checkpoint_model(
    model_path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    required = {
        "model_state_dict",
        "model_name",
        "model_config",
        "feature_dim",
        "num_classes",
        "schema_version",
        "preprocess_fingerprints",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing keys: {sorted(missing)}")
    if int(checkpoint["feature_dim"]) != FEATURE_DIM:
        raise ValueError(
            f"Checkpoint feature_dim {checkpoint['feature_dim']} != schema {FEATURE_DIM}"
        )
    if str(checkpoint["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(
            f"Checkpoint schema {checkpoint['schema_version']} != current {SCHEMA_VERSION}"
        )
    model = build_model(str(checkpoint["model_name"]), checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def validate_record_compatibility(
    record: SequenceRecord,
    checkpoint: dict[str, Any],
) -> None:
    fingerprint = str(
        record.metadata.get(
            "preprocess_config_fingerprint",
            record.metadata.get("preprocess_fingerprint", ""),
        )
    )
    allowed = {str(item) for item in checkpoint.get("preprocess_fingerprints", [])}
    if not fingerprint:
        raise ValueError(f"{record.video_id}: sequence has no preprocess fingerprint")
    if fingerprint not in allowed:
        raise ValueError(
            f"{record.video_id}: preprocessing fingerprint '{fingerprint}' is not "
            f"compatible with checkpoint fingerprints {sorted(allowed)}"
        )


def overlap_window_starts(num_frames: int, window_size: int, stride: int) -> list[int]:
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    if num_frames <= 0:
        return []
    starts = list(range(0, max(num_frames - window_size, 0) + 1, stride))
    last = max(num_frames - window_size, 0)
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def hamming_weights(length: int) -> np.ndarray:
    if length <= 1:
        return np.ones(length, dtype=np.float32)
    n = np.arange(length, dtype=np.float32)
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * n / (length - 1))


@torch.inference_mode()
def predict_sequence(
    model: nn.Module,
    record: SequenceRecord,
    device: torch.device,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    frames = record.num_frames
    num_classes = int(getattr(model, "num_classes", model.PG.conv_out.out_channels))
    accum = np.zeros((num_classes, frames), dtype=np.float64)
    weights = np.zeros(frames, dtype=np.float64)
    window_weights = hamming_weights(window_size)

    for start in overlap_window_starts(frames, window_size, stride):
        end = min(start + window_size, frames)
        actual = end - start
        window = np.zeros((FEATURE_DIM, window_size), dtype=np.float32)
        window[:, :actual] = record.features[:, start:end]
        if actual < window_size:
            window[:, actual:] = window[:, actual - 1 : actual]
        inputs = torch.from_numpy(window).unsqueeze(0).to(device)
        logits = model(inputs)[-1][0, :, :actual].detach().cpu().numpy()
        accum[:, start:end] += logits * window_weights[:actual]
        weights[start:end] += window_weights[:actual]

    weights = np.maximum(weights, 1e-8)
    fused = (accum / weights).astype(np.float32)
    prediction = fused.argmax(axis=0).astype(np.int64)
    return prediction, fused


def write_prediction_outputs(
    output_dir: str | Path,
    record: SequenceRecord,
    prediction: np.ndarray,
    logits: np.ndarray,
    save_frame_csv: bool = True,
    save_frame_npy: bool = True,
    save_segments: bool = True,
) -> dict[str, Path]:
    video_dir = Path(output_dir) / record.video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    if save_frame_npy:
        npy_path = video_dir / "prediction.npy"
        np.save(npy_path, prediction)
        written["prediction_npy"] = npy_path
        np.save(video_dir / "logits.npy", logits)
        written["logits_npy"] = video_dir / "logits.npy"
    if save_frame_csv:
        csv_path = video_dir / "frames.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["frame", "time_sec", "label", "name"])
            for index, label in enumerate(prediction.tolist()):
                writer.writerow(
                    [
                        index,
                        f"{index / record.fps:.4f}",
                        int(label),
                        class_id_to_name(int(label)),
                    ]
                )
        written["frames_csv"] = csv_path
    if save_segments:
        segments = labels_to_segments(prediction)
        json_path = video_dir / "segments.json"
        payload = {
            "video_id": record.video_id,
            "schema_version": SCHEMA_VERSION,
            "fps": record.fps,
            "num_frames": record.num_frames,
            "feature_names": FEATURE_NAMES,
            "segments": [segment.to_dict(record.fps) for segment in segments],
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        written["segments_json"] = json_path
    return written


def segments_from_prediction(prediction: np.ndarray) -> list[ActionSegment]:
    return labels_to_segments(prediction)
