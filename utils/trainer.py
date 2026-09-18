from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.schema import IGNORE_INDEX, NUM_CLASSES
from utils.inference import DEFAULT_PREDICT_BATCH_SIZE, predict_sequence
from utils.metrics import aggregate_metrics, evaluate_sequence
from utils.schema import SequenceRecord


DEFAULT_CLASS_WEIGHT_MAX_RATIO = 50.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def make_class_weights(
    labels: list[int],
    device: torch.device,
    num_classes: int = NUM_CLASSES,
    max_ratio: float = DEFAULT_CLASS_WEIGHT_MAX_RATIO,
) -> torch.Tensor:
    """Inverse-frequency weights whose largest/smallest ratio is bounded.

    Plain inverse frequency is unusable while the taxonomy is only partially
    collected: a class holding a thousand of twenty million frames would
    outweigh the common ones by four orders of magnitude and its handful of
    samples would dominate every gradient. Classes missing from ``labels`` are
    pinned to the same ceiling rather than the 1/0 blow-up that clamping their
    count to one used to produce.
    """
    if max_ratio < 1.0:
        raise ValueError("max_ratio must be at least 1")
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    present = counts > 0
    if not present.any():
        return torch.ones(len(counts), dtype=torch.float32, device=device)
    weights = np.empty_like(counts)
    weights[present] = counts[present].sum() / (
        int(present.sum()) * counts[present]
    )
    ceiling = float(weights[present].min()) * max_ratio
    weights[present] = np.minimum(weights[present], ceiling)
    weights[~present] = ceiling
    return torch.tensor(weights, dtype=torch.float32, device=device)


DEFAULT_WARMUP_START_FACTOR = 0.1
DEFAULT_MIN_LR_RATIO = 0.01


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    training_config: dict[str, Any],
    num_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Per-epoch learning-rate schedule, or ``None`` when none is configured.

    Cosine annealing needs the full epoch budget up front, so ``T_max`` is tied
    to ``num_epochs``; shortening a run without shortening the schedule would
    stop it mid-decay and lose the low-rate epochs that make the final
    checkpoints comparable.
    """
    section = training_config.get("scheduler")
    if not section:
        return None
    name = str(section.get("name", "cosine")).strip().lower()
    if name != "cosine":
        raise ValueError(f"Unknown scheduler '{name}'; supported: cosine")

    warmup_epochs = int(section.get("warmup_epochs", 0))
    if warmup_epochs < 0:
        raise ValueError("scheduler.warmup_epochs must be non-negative")
    if warmup_epochs >= num_epochs:
        raise ValueError(
            f"scheduler.warmup_epochs ({warmup_epochs}) must be below "
            f"num_epochs ({num_epochs})"
        )
    min_lr_ratio = float(section.get("min_lr_ratio", DEFAULT_MIN_LR_RATIO))
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("scheduler.min_lr_ratio must be in [0, 1]")
    start_factor = float(
        section.get("warmup_start_factor", DEFAULT_WARMUP_START_FACTOR)
    )
    if not 0.0 < start_factor <= 1.0:
        raise ValueError("scheduler.warmup_start_factor must be in (0, 1]")

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs - warmup_epochs,
        eta_min=float(optimizer.defaults["lr"]) * min_lr_ratio,
    )
    if warmup_epochs == 0:
        return cosine
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,
        total_iters=warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def tmse_loss(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    clamp_value: float = 16.0,
) -> torch.Tensor:
    if logits.size(-1) < 2:
        return logits.new_zeros(())
    log_probs = torch.log_softmax(logits, dim=1)
    delta = (log_probs[:, :, 1:] - log_probs[:, :, :-1]) ** 2
    pair_mask = valid_mask[:, 1:] * valid_mask[:, :-1]
    pair_mask = pair_mask.unsqueeze(1)
    masked = torch.clamp(delta, max=clamp_value) * pair_mask
    denom = pair_mask.sum() * logits.size(1)
    if denom <= 0:
        return logits.new_zeros(())
    return masked.sum() / denom


def multi_stage_loss(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    criterion: nn.Module,
    tmse_weight: float,
    tmse_clamp: float,
) -> torch.Tensor:
    total = outputs.new_zeros(())
    for stage_logits in outputs:
        ce = criterion(stage_logits, labels)
        smooth = tmse_loss(stage_logits, valid_mask, tmse_clamp)
        total = total + ce + tmse_weight * smooth
    return total


def _frame_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[int, int]:
    predictions = logits.argmax(dim=1)
    valid = labels != IGNORE_INDEX
    if valid.sum() == 0:
        return 0, 0
    correct = int(((predictions == labels) & valid).sum().item())
    return correct, int(valid.sum().item())


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float,
    tmse_weight: float,
    tmse_clamp: float,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(features)
        loss = multi_stage_loss(
            outputs,
            labels,
            valid_mask,
            criterion,
            tmse_weight,
            tmse_clamp,
        )
        loss.backward()
        if gradient_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        batch_correct, batch_total = _frame_accuracy(outputs[-1], labels)
        total_loss += float(loss.item()) * max(batch_total, 1)
        correct += batch_correct
        total += batch_total
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    tmse_weight: float,
    tmse_clamp: float,
) -> tuple[float, float, dict[int, float]]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct: Counter[int] = Counter()
    class_total: Counter[int] = Counter()
    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        outputs = model(features)
        loss = multi_stage_loss(
            outputs,
            labels,
            valid_mask,
            criterion,
            tmse_weight,
            tmse_clamp,
        )
        predictions = outputs[-1].argmax(dim=1)
        valid = labels != IGNORE_INDEX
        batch_correct = int(((predictions == labels) & valid).sum().item())
        batch_total = int(valid.sum().item())
        total_loss += float(loss.item()) * max(batch_total, 1)
        correct += batch_correct
        total += batch_total
        for class_id in labels[valid].unique().tolist():
            class_mask = valid & (labels == class_id)
            class_total[int(class_id)] += int(class_mask.sum().item())
            class_correct[int(class_id)] += int(
                ((predictions == labels) & class_mask).sum().item()
            )
    class_accuracy = {
        class_id: class_correct[class_id] / class_total[class_id]
        for class_id in class_total
        if class_total[class_id] > 0
    }
    return total_loss / max(total, 1), correct / max(total, 1), class_accuracy


@torch.no_grad()
def evaluate_full_videos(
    model: nn.Module,
    records: list[SequenceRecord],
    device: torch.device,
    window_size: int,
    stride: int,
    iou_thresholds: tuple[float, ...] = (0.1, 0.25, 0.5),
    batch_size: int = DEFAULT_PREDICT_BATCH_SIZE,
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    model.eval()
    per_video: list[dict[str, float | str]] = []
    numeric: list[dict[str, float]] = []
    for record in records:
        if record.labels is None:
            raise ValueError(f"{record.video_id}: labels are required for validation")
        prediction, _ = predict_sequence(
            model,
            record,
            device,
            window_size=window_size,
            stride=stride,
            batch_size=batch_size,
        )
        metrics = evaluate_sequence(prediction, record.labels, iou_thresholds)
        numeric.append(metrics)
        per_video.append({"video_id": record.video_id, **metrics})
    return aggregate_metrics(numeric), per_video


def save_checkpoint(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path
