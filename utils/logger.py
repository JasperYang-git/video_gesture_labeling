from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

from utils.schema import CLASS_NAMES, class_id_to_name


def create_run_directory(root_dir: str | Path) -> Path:
    now = datetime.now()
    run_dir = Path(root_dir) / now.strftime("%m%d") / now.strftime("%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def setup_logger(name: str, log_path: str | Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def log_class_distribution(
    logger: logging.Logger,
    split_name: str,
    labels: list[int],
) -> None:
    total = len(labels)
    if total == 0:
        logger.info("%s distribution: empty", split_name)
        return
    counts = Counter(labels)
    parts = [
        f"{class_id_to_name(class_id)}={counts.get(class_id, 0)}"
        for class_id in range(len(CLASS_NAMES))
        if counts.get(class_id, 0)
    ]
    logger.info(
        "%s distribution: total=%d | %s",
        split_name,
        total,
        ", ".join(parts),
    )
