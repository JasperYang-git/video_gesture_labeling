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
) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


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


def save_checkpoint(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path
