from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.preprocessing.features import build_feature_matrix
from utils.schema import (
    BACKGROUND_ID,
    CLASS_NAMES,
    COORD_DIM,
    NUM_CLASSES,
    SequenceRecord,
    load_sequence,
    save_sequence,
)
from utils.split import write_manifest


GESTURE_IDS = tuple(index for index in range(NUM_CLASSES) if index != BACKGROUND_ID)


def _class_motion(class_id: int, time: np.ndarray, phase: float) -> np.ndarray:
    coords = np.zeros((len(time), COORD_DIM), dtype=np.float32)
    group = (class_id % 7) * 9
    freq = 0.8 + 0.35 * class_id
    amp = 0.12 + 0.015 * class_id
    envelope = np.sin(np.pi * np.linspace(0.0, 1.0, len(time), dtype=np.float32))
    wave = amp * np.sin(2.0 * np.pi * freq * time + phase) * envelope
    pulse = amp * np.sin(2.0 * np.pi * (freq * 2.0) * time + phase) * (envelope ** 2)
    for offset in range(9):
        channel = (group + offset) % COORD_DIM
        coords[:, channel] = wave if offset % 2 == 0 else pulse
        if class_id >= 9:
            coords[:, channel] *= 1.0 + 0.35 * np.sin(4.0 * np.pi * time + offset)
    return coords


def _sample_segments(
    num_frames: int,
    fps: float,
    rng: np.random.Generator,
) -> list[tuple[int, int, int]]:
    remaining = num_frames
    cursor = 0
    segments: list[tuple[int, int, int]] = []
    while remaining > 0:
        if rng.random() < 0.35 or remaining < int(0.6 * fps):
            length = min(remaining, int(rng.integers(int(0.4 * fps), int(1.2 * fps) + 1)))
            label = BACKGROUND_ID
        else:
            length = min(remaining, int(rng.integers(int(0.6 * fps), int(1.8 * fps) + 1)))
            label = int(GESTURE_IDS[int(rng.integers(0, len(GESTURE_IDS)))])
        segments.append((cursor, cursor + length, label))
        cursor += length
        remaining -= length
    return segments


def generate_mock_sequence(
    video_id: str,
    num_frames: int,
    fps: float,
    seed: int,
    miss_rate: float = 0.08,
    noise_std: float = 0.04,
    subject_id: str = "",
    session_id: str = "",
) -> SequenceRecord:
    rng = np.random.default_rng(seed)
    time = np.arange(num_frames, dtype=np.float32) / fps
    coordinates = rng.normal(0.0, 0.02, (num_frames, COORD_DIM)).astype(np.float32)
    labels = np.full(num_frames, BACKGROUND_ID, dtype=np.int64)
    scores = np.full(num_frames, 0.95, dtype=np.float32)
    valid_mask = np.ones(num_frames, dtype=np.float32)

    for start, end, label in _sample_segments(num_frames, fps, rng):
        labels[start:end] = label
        if label == BACKGROUND_ID:
            continue
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        coordinates[start:end] += _class_motion(label, time[start:end] - time[start], phase)

    coordinates += rng.normal(0.0, noise_std, coordinates.shape).astype(np.float32)
    miss_flags = rng.random(num_frames) < miss_rate
    last_valid = coordinates[0].copy()
    for index in range(num_frames):
        if miss_flags[index]:
            coordinates[index] = last_valid
            scores[index] = 0.15
            valid_mask[index] = 0.0
        else:
            last_valid = coordinates[index]
    features = build_feature_matrix(coordinates, scores, valid_mask, fps)
    action_mask = labels != BACKGROUND_ID
    detection_rate = (
        float(valid_mask[action_mask].mean())
        if action_mask.any()
        else float(valid_mask.mean())
    )
    per_class_detection_rate = {
        str(class_id): float(valid_mask[labels == class_id].mean())
        for class_id in GESTURE_IDS
        if np.any(labels == class_id)
    }
    return SequenceRecord(
        features=features,
        labels=labels,
        valid_mask=valid_mask,
        video_id=video_id,
        fps=fps,
        source_path=f"mock://{video_id}",
        hand_side="R",
        subject_id=subject_id,
        session_id=session_id,
        source="mock",
        gender="unknown",
        field3="mock",
        scene="mock-positive",
        polarity="positive",
        metadata={
            "source": "mock",
            "scene": "mock-positive",
            "polarity": "positive",
            "scene_category": "mock",
            "hard_negative_eligible": False,
            "class_names": list(CLASS_NAMES),
            "preprocess_fingerprint": "mock_v3",
            "preprocess_config_fingerprint": "mock_v3",
            "quality_passed": True,
            "detection_rate": detection_rate,
            "per_class_detection_rate": per_class_detection_rate,
            "min_class_id": min(
                per_class_detection_rate,
                key=per_class_detection_rate.get,
                default="",
            ),
            "min_class_detection_rate": min(
                per_class_detection_rate.values(),
                default=detection_rate,
            ),
        },
    )


def generate_mock_dataset(
    output_dir: str | Path,
    videos_per_split: dict[str, int],
    min_frames: int,
    max_frames: int,
    fps: float,
    seed: int,
    miss_rate: float = 0.08,
    noise_std: float = 0.04,
    overwrite: bool = False,
) -> dict[str, list[Path]]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    split_files: dict[str, list[Path]] = {name: [] for name in ("train", "validation", "test")}
    file_index = 0
    for split_name, count in videos_per_split.items():
        if split_name not in split_files:
            raise ValueError(f"Unknown split name: {split_name}")
        for local_index in range(int(count)):
            rng = np.random.default_rng(seed + file_index)
            num_frames = int(rng.integers(min_frames, max_frames + 1))
            video_id = f"mock_{split_name}_{local_index:02d}"
            subject_id = f"mock_subject_{split_name}_{local_index // 2:02d}"
            session_id = f"{subject_id}_session_{local_index % 2:02d}"
            output_path = root / f"{video_id}.npz"
            if output_path.exists() and not overwrite:
                try:
                    existing = load_sequence(output_path)
                except (ValueError, KeyError):
                    output_path.unlink()
                else:
                    if (
                        existing.metadata.get("preprocess_config_fingerprint")
                        == "mock_v3"
                    ):
                        split_files[split_name].append(output_path)
                        file_index += 1
                        continue
            record = generate_mock_sequence(
                video_id=video_id,
                num_frames=num_frames,
                fps=fps,
                seed=seed + file_index,
                miss_rate=miss_rate,
                noise_std=noise_std,
                subject_id=subject_id,
                session_id=session_id,
            )
            save_sequence(output_path, record)
            split_files[split_name].append(output_path)
            file_index += 1
    return split_files


def generate_mock_and_manifest(config: dict) -> Path:
    data_config = config["data"]
    mock_config = config["mock"]
    split_files = generate_mock_dataset(
        output_dir=mock_config.get("output_dir", data_config["processed_dir"]),
        videos_per_split=mock_config["videos_per_split"],
        min_frames=int(mock_config["min_frames"]),
        max_frames=int(mock_config["max_frames"]),
        fps=float(mock_config["fps"]),
        seed=int(config["seed"]),
        miss_rate=float(mock_config.get("miss_rate", 0.08)),
        noise_std=float(mock_config.get("noise_std", 0.04)),
        overwrite=bool(data_config.get("overwrite", False)),
    )
    records = [
        load_sequence(path)
        for split_name in ("train", "validation", "test")
        for path in split_files[split_name]
    ]
    return write_manifest(
        data_config["manifest_path"],
        split_files,
        seed=int(config["seed"]),
        ratios={
            "train": float(data_config.get("train_ratio", 0.7)),
            "validation": float(data_config.get("validation_ratio", 0.15)),
            "test": float(data_config.get("test_ratio", 0.15)),
        },
        processed_dir=mock_config.get("output_dir", data_config["processed_dir"]),
        records=records,
        grouping="subject",
    )
