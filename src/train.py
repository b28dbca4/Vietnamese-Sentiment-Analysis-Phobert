"""Training loop for Transformer fine-tuning."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
except ImportError as exc:  # pragma: no cover - exercised only when dependencies are missing.
    AutoModelForSequenceClassification = None
    get_linear_schedule_with_warmup = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None

from src import config
from src.model import save_model


def _require_transformers() -> None:
    """Raise a clear error if Hugging Face Transformers is unavailable."""
    if AutoModelForSequenceClassification is None or get_linear_schedule_with_warmup is None:
        raise ImportError(
            "The transformers package is required for Transformer training. "
            "Install project dependencies with 'pip install -r requirements.txt'."
        ) from _TRANSFORMERS_IMPORT_ERROR


def _batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move a batch dictionary to the target device."""
    return {key: value.to(device) for key, value in batch.items()}


def _metric_summary(labels: list[int], predictions: list[int], prefix: str) -> dict[str, float]:
    """Compute accuracy and macro F1 with a consistent prefix."""
    return {
        f"{prefix}_accuracy": accuracy_score(labels, predictions),
        f"{prefix}_f1": f1_score(labels, predictions, average="macro", zero_division=0),
    }


def train_one_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    max_grad_norm: float = config.MAX_GRAD_NORM,
) -> dict[str, float]:
    """Execute one full training epoch.

    Args:
        model: Transformer sequence classifier.
        dataloader: Training DataLoader.
        optimizer: AdamW optimizer.
        scheduler: Linear warmup and decay scheduler.
        device: Training device.
        max_grad_norm: Gradient clipping threshold.

    Returns:
        Training loss, accuracy, and macro F1 for the epoch.
    """
    model.train()
    total_loss = 0.0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    progress = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress:
        batch = _batch_to_device(batch, device)
        labels = batch["label"]

        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()

        # Clipping protects fine-tuning from rare batches with very large gradients.
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        predictions = torch.argmax(outputs.logits.detach(), dim=1)
        all_labels.extend(labels.detach().cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

        running_loss = total_loss / max(1, len(all_labels))
        running_acc = accuracy_score(all_labels, all_predictions)
        progress.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.4f}")

    metrics = _metric_summary(all_labels, all_predictions, "train")
    metrics["train_loss"] = total_loss / max(1, len(all_labels))
    return metrics


def validate(model, dataloader: DataLoader, device: torch.device) -> dict[str, float]:
    """Evaluate a model on a validation set without updating weights.

    Args:
        model: Transformer sequence classifier.
        dataloader: Validation DataLoader.
        device: Evaluation device.

    Returns:
        Validation loss, accuracy, and macro F1.
    """
    model.eval()
    total_loss = 0.0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            batch = _batch_to_device(batch, device)
            labels = batch["label"]
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=labels,
            )

            batch_size = labels.size(0)
            total_loss += outputs.loss.item() * batch_size
            predictions = torch.argmax(outputs.logits, dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    metrics = _metric_summary(all_labels, all_predictions, "val")
    metrics["val_loss"] = total_loss / max(1, len(all_labels))
    return metrics


def _build_optimizer(model, weight_decay: float, lr: float) -> AdamW:
    """Build AdamW with no weight decay for bias and normalization weights."""
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias", "layer_norm.weight", "layer_norm.bias"]
    grouped_parameters = [
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if not any(pattern in name for pattern in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if any(pattern in name for pattern in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(grouped_parameters, lr=lr, eps=1e-8)


def _save_training_log(result: dict[str, Any], model_key: str) -> None:
    """Persist training history as JSON."""
    os.makedirs(config.LOG_DIR, exist_ok=True)
    path = os.path.join(config.LOG_DIR, f"{model_key}_history.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)


def train_model(
    model_key: str,
    model,
    tokenizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = config.EPOCHS,
    lr: float = config.LEARNING_RATE,
    weight_decay: float = config.WEIGHT_DECAY,
    warmup_ratio: float = config.WARMUP_RATIO,
    max_grad_norm: float = config.MAX_GRAD_NORM,
    patience: int = config.EARLY_STOPPING_PATIENCE,
) -> dict[str, Any]:
    """Train one Transformer model with checkpointing and early stopping.

    Args:
        model_key: Short model key such as ``distilbert``.
        model: Transformer sequence classifier.
        tokenizer: Tokenizer associated with the model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        device: Training device.
        epochs: Maximum number of epochs.
        lr: Learning rate.
        weight_decay: AdamW weight decay.
        warmup_ratio: Fraction of steps used for learning-rate warmup.
        max_grad_norm: Gradient clipping threshold.
        patience: Early stopping patience in epochs.

    Returns:
        Training history and best checkpoint metadata.
    """
    _require_transformers()
    model.to(device)
    optimizer = _build_optimizer(model, weight_decay=weight_decay, lr=lr)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model_identifier = config.TRANSFORMER_MODELS.get(model_key, model_key)
    save_dir = os.path.join(config.MODELS_DIR, model_identifier)
    history: list[dict[str, float | int]] = []
    best_val_loss = float("inf")
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    total_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            max_grad_norm=max_grad_norm,
        )
        val_metrics = validate(model, val_loader, device)
        epoch_metrics = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_time_seconds": time.time() - epoch_start,
        }
        history.append(epoch_metrics)

        print(
            "Epoch "
            f"{epoch}: train_loss={train_metrics['train_loss']:.4f}, "
            f"val_loss={val_metrics['val_loss']:.4f}, "
            f"val_f1={val_metrics['val_f1']:.4f}"
        )

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = float(val_metrics["val_loss"])
            best_val_f1 = float(val_metrics["val_f1"])
            best_epoch = epoch
            patience_counter = 0
            save_model(model, tokenizer, save_dir)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if os.path.isdir(save_dir):
        best_model = AutoModelForSequenceClassification.from_pretrained(save_dir)
        model.load_state_dict(best_model.state_dict())
        model.to(device)

    result = {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_f1": best_val_f1,
        "total_train_time_seconds": time.time() - total_start,
        "model_key": model_key,
        "model_identifier": model_identifier,
        "hyperparameters": {
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": config.BATCH_SIZE,
            "max_length": config.MAX_LENGTH,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "max_grad_norm": max_grad_norm,
            "patience": patience,
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
        },
    }
    _save_training_log(result, model_key)
    return result
