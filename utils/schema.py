from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


CLASS_NAMES: tuple[str, ...] = (
    "knock",
    "tap",
    "slide-up",
    "slide-down",
    "clench",
    "up-down",
    "wrist",
    "nodded",
    "snap",
    "double-knock",
    "double-tap",
    "double-slide-up",
    "double-slide-down",
    "double-clench",
    "background",
)
BACKGROUND_ID = 14
NUM_CLASSES = len(CLASS_NAMES)
COORD_DIM = 63
VELOCITY_DIM = 63
FEATURE_DIM = 2 + COORD_DIM + VELOCITY_DIM
SCHEMA_VERSION = "gesture_sequence_v2"
IGNORE_INDEX = -100
SCORE_INDEX = 0
VALID_MASK_INDEX = 1
COORD_SLICE = slice(2, 2 + COORD_DIM)
VELOCITY_SLICE = slice(2 + COORD_DIM, FEATURE_DIM)


def feature_names() -> list[str]:
    names = ["tracking_quality", "v_mask"]
    names.extend(f"c_{index}" for index in range(COORD_DIM))
    names.extend(f"v_{index}" for index in range(VELOCITY_DIM))
    return names


FEATURE_NAMES = feature_names()


def class_id_to_name(class_id: int) -> str:
    if class_id < 0 or class_id >= NUM_CLASSES:
        raise ValueError(f"Unknown class id: {class_id}")
    return CLASS_NAMES[class_id]


def class_name_to_id(name: str) -> int:
    try:
        return CLASS_NAMES.index(name)
    except ValueError as exc:
        raise ValueError(f"Unknown class name: {name}") from exc


def load_class_mapping(path: str | Path | None = None) -> dict[int, str]:
    if path is None:
        return {index: name for index, name in enumerate(CLASS_NAMES)}
    mapping: dict[int, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        class_id_text, name = stripped.split(maxsplit=1)
        mapping[int(class_id_text)] = name
    if mapping != {index: name for index, name in enumerate(CLASS_NAMES)}:
        raise ValueError(f"Class mapping in {path} does not match the frozen 15-class schema")
    return mapping


@dataclass
class SequenceRecord:
    features: np.ndarray
    labels: np.ndarray | None
    valid_mask: np.ndarray
    video_id: str
    fps: float
    source_path: str = ""
    hand_side: str = "R"
    subject_id: str = ""
    session_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.features.ndim != 2:
            raise ValueError(f"{self.video_id}: features must be [C, T]")
        channels, frames = self.features.shape
        if channels != FEATURE_DIM:
            raise ValueError(
                f"{self.video_id}: expected {FEATURE_DIM} feature channels, got {channels}"
            )
        if self.valid_mask.shape != (frames,):
            raise ValueError(f"{self.video_id}: valid_mask must have shape [{frames}]")
        if self.labels is not None and self.labels.shape != (frames,):
            raise ValueError(f"{self.video_id}: labels must have shape [{frames}]")
        if self.features.dtype != np.float32:
            self.features = self.features.astype(np.float32, copy=False)
        if self.valid_mask.dtype != np.float32:
            self.valid_mask = self.valid_mask.astype(np.float32, copy=False)
        if self.labels is not None and self.labels.dtype != np.int64:
            self.labels = self.labels.astype(np.int64, copy=False)

    @property
    def num_frames(self) -> int:
        return int(self.features.shape[1])

    def to_npz_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "features": self.features,
            "valid_mask": self.valid_mask,
            "video_id": np.asarray(self.video_id),
            "fps": np.asarray(self.fps, dtype=np.float32),
            "source_path": np.asarray(self.source_path),
            "hand_side": np.asarray(self.hand_side),
            "subject_id": np.asarray(self.subject_id),
            "session_id": np.asarray(self.session_id),
            "schema_version": np.asarray(SCHEMA_VERSION),
            "feature_names": np.asarray(FEATURE_NAMES),
            "class_names": np.asarray(CLASS_NAMES),
        }
        if self.labels is not None:
            payload["labels"] = self.labels
        for key, value in self.metadata.items():
            payload[f"meta_{key}"] = np.asarray(value)
        return payload


def save_sequence(path: str | Path, record: SequenceRecord) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.npz")
    np.savez_compressed(temporary, **record.to_npz_payload())
    temporary.replace(output_path)
    return output_path


def load_sequence(path: str | Path) -> SequenceRecord:
    archive = np.load(path, allow_pickle=True)
    schema_version = str(archive["schema_version"])
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: unsupported schema_version '{schema_version}', expected {SCHEMA_VERSION}"
        )
    stored_names = [str(name) for name in archive["feature_names"].tolist()]
    if stored_names != FEATURE_NAMES:
        raise ValueError(f"{path}: feature schema mismatch")
    labels = archive["labels"] if "labels" in archive.files else None
    metadata = {
        key.removeprefix("meta_"): archive[key].item()
        if archive[key].shape == ()
        else archive[key]
        for key in archive.files
        if key.startswith("meta_")
    }
    return SequenceRecord(
        features=np.asarray(archive["features"], dtype=np.float32),
        labels=None if labels is None else np.asarray(labels, dtype=np.int64),
        valid_mask=np.asarray(archive["valid_mask"], dtype=np.float32),
        video_id=str(archive["video_id"]),
        fps=float(archive["fps"]),
        source_path=str(archive["source_path"]) if "source_path" in archive.files else "",
        hand_side=str(archive["hand_side"]) if "hand_side" in archive.files else "R",
        subject_id=str(archive["subject_id"]) if "subject_id" in archive.files else "",
        session_id=str(archive["session_id"]) if "session_id" in archive.files else "",
        metadata=metadata,
    )


def sequence_summary(record: SequenceRecord) -> dict[str, Any]:
    summary = {
        "video_id": record.video_id,
        "num_frames": record.num_frames,
        "fps": record.fps,
        "feature_dim": int(record.features.shape[0]),
        "hand_side": record.hand_side,
        "subject_id": record.subject_id,
        "session_id": record.session_id,
        "has_labels": record.labels is not None,
        "valid_ratio": float(record.valid_mask.mean()) if record.num_frames else 0.0,
    }
    if record.labels is not None:
        unique, counts = np.unique(record.labels, return_counts=True)
        summary["label_histogram"] = {
            class_id_to_name(int(class_id)): int(count)
            for class_id, count in zip(unique, counts)
        }
    return summary


def dump_mapping(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{index} {name}" for index, name in enumerate(CLASS_NAMES)]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def as_plain_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: as_plain_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: as_plain_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_plain_dict(item) for item in value]
    return value
