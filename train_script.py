from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from torch import nn

from model import build_model
from utils.config import load_config
from utils.data_loader import build_train_loaders
from utils.logger import create_run_directory, log_class_distribution, setup_logger
from utils.schema import CLASS_NAMES, FEATURE_DIM, FEATURE_NAMES, NUM_CLASSES, SCHEMA_VERSION
from utils.trainer import (
    evaluate,
    make_class_weights,
    resolve_device,
    save_checkpoint,
    seed_everything,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MS-TCN2 gesture segmenter")
    parser.add_argument(
        "--config",
        default="config/config_train.yaml",
        help="Path to training YAML",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["seed"]))

    run_dir = create_run_directory(config["output"]["root_dir"])
    logger = setup_logger("training", run_dir / "training.log")
    shutil.copy2(args.config, run_dir / "config_train.yaml")
    device = resolve_device(str(config["device"]))
    logger.info("Run directory: %s", run_dir)
    logger.info("Device: %s", device)

    train_loader, validation_loader, train_files, val_files = build_train_loaders(config)
    logger.info("Training videos (%d):", len(train_files))
    for path in train_files:
        logger.info("Training video: %s", path.name)
    logger.info("Validation videos (%d):", len(val_files))
    for path in val_files:
        logger.info("Validation video: %s", path.name)
    log_class_distribution(logger, "Training frames", train_loader.dataset.frame_labels)
    log_class_distribution(
        logger,
        "Validation frames",
        validation_loader.dataset.frame_labels,
    )

    model_section = config["model"]
    model_name = str(model_section["name"])
    model_config = dict(model_section.get("params", {}))
    model_config.setdefault("dim", FEATURE_DIM)
    model_config.setdefault("num_classes", NUM_CLASSES)
    if int(model_config["dim"]) != FEATURE_DIM:
        raise ValueError(
            f"model.params.dim {model_config['dim']} must match schema {FEATURE_DIM}"
        )
    if int(model_config["num_classes"]) != NUM_CLASSES:
        raise ValueError(
            f"model.params.num_classes {model_config['num_classes']} must be {NUM_CLASSES}"
        )
    model = build_model(model_name, model_config).to(device)

    training_config = config["training"]
    class_weights = (
        make_class_weights(train_loader.dataset.frame_labels, device)
        if training_config.get("class_weighted_loss", False)
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )

    history: list[dict[str, float]] = []
    best_accuracy = -1.0
    best_path = run_dir / "best_model.pth"
    last_path = run_dir / "last_model.pth"
    tmse_weight = float(training_config.get("tmse_weight", 0.1))
    tmse_clamp = float(training_config.get("tmse_clamp", 16.0))

    for epoch in range(1, int(training_config["num_epochs"]) + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            float(training_config["gradient_clip_norm"]),
            tmse_weight,
            tmse_clamp,
        )
        validation_loss, validation_accuracy, validation_class_accuracy = evaluate(
            model,
            validation_loader,
            criterion,
            device,
            tmse_weight,
            tmse_clamp,
        )
        logger.info(
            "Epoch %03d | train_loss=%.6f | train_acc=%.2f%% | "
            "val_loss=%.6f | val_acc=%.2f%%",
            epoch,
            train_loss,
            100.0 * train_accuracy,
            validation_loss,
            100.0 * validation_accuracy,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "model_config": model_config,
            "feature_dim": FEATURE_DIM,
            "feature_names": FEATURE_NAMES,
            "num_classes": NUM_CLASSES,
            "class_names": list(CLASS_NAMES),
            "schema_version": SCHEMA_VERSION,
            "window_size": int(config["data"]["window_size"]),
            "stride": int(config["data"]["stride"]),
            "tmse_weight": tmse_weight,
            "best_validation_accuracy": max(best_accuracy, validation_accuracy),
            "epoch": epoch,
        }
        save_checkpoint(last_path, checkpoint)
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            save_checkpoint(best_path, checkpoint)
            logger.info("Saved new best checkpoint: %s", best_path)

    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n",
        encoding="utf-8",
    )
    latest_path = Path(config["output"]["latest_model_path"])
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, latest_path)
    logger.info(
        "Training complete. Best validation accuracy=%.2f%%; best_model=%s",
        100.0 * best_accuracy,
        best_path,
    )


if __name__ == "__main__":
    main()
