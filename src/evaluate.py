"""Evaluation and error analysis utilities for IMDb sentiment models."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src import config


def _json_ready(value: Any) -> Any:
    """Convert metric objects into JSON-serializable Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def predict_all(
    model,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on an entire DataLoader.

    Args:
        model: Trained sequence classifier.
        dataloader: DataLoader to evaluate.
        device: Compute device.

    Returns:
        Arrays for true labels, predicted labels, and class probabilities.
    """
    model.eval()
    model.to(device)
    all_labels: list[int] = []
    all_predictions: list[int] = []
    all_probabilities: list[list[float]] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_probabilities.extend(probabilities.cpu().tolist())

    return (
        np.asarray(all_labels),
        np.asarray(all_predictions),
        np.asarray(all_probabilities),
    )


def compute_metrics(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    pred_probs: np.ndarray,
) -> dict[str, Any]:
    """Compute comprehensive classification metrics.

    Args:
        true_labels: Ground-truth integer labels.
        pred_labels: Predicted integer labels.
        pred_probs: Predicted probabilities with shape ``[N, 2]``.

    Returns:
        Dictionary of scalar metrics, confusion matrix, and report details.
    """
    target_names = [config.ID2LABEL[index] for index in sorted(config.ID2LABEL)]
    report = classification_report(
        true_labels,
        pred_labels,
        labels=[0, 1],
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    try:
        auc_score = roc_auc_score(true_labels, pred_probs[:, config.LABEL2ID["positive"]])
    except ValueError:
        auc_score = None

    precision_macro = precision_score(true_labels, pred_labels, average="macro", zero_division=0)
    recall_macro = recall_score(true_labels, pred_labels, average="macro", zero_division=0)
    f1_macro = f1_score(true_labels, pred_labels, average="macro", zero_division=0)

    metrics = {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_score(true_labels, pred_labels, average="weighted", zero_division=0),
        "recall_weighted": recall_score(true_labels, pred_labels, average="weighted", zero_division=0),
        "f1_weighted": f1_score(true_labels, pred_labels, average="weighted", zero_division=0),
        "precision": precision_macro,
        "recall": recall_macro,
        "f1": f1_macro,
        "roc_auc": auc_score,
        "auc": auc_score,
        "confusion_matrix": confusion_matrix(true_labels, pred_labels, labels=[0, 1]),
        "classification_report": report,
    }
    return _json_ready(metrics)


def evaluate_model(
    model,
    test_loader: DataLoader,
    device: torch.device,
    model_name: str,
) -> dict[str, Any]:
    """Evaluate a model on the test set and save metrics.

    Args:
        model: Trained sequence classifier.
        test_loader: Test DataLoader.
        device: Evaluation device.
        model_name: Name used for the saved JSON file.

    Returns:
        Full test metrics plus predictions and probabilities.
    """
    true_labels, pred_labels, pred_probs = predict_all(model, test_loader, device)
    metrics = compute_metrics(true_labels, pred_labels, pred_probs)
    metrics.update(
        {
            "model_name": model_name,
            "true_labels": true_labels.tolist(),
            "predictions": pred_labels.tolist(),
            "probabilities": pred_probs.tolist(),
        }
    )

    os.makedirs(config.METRICS_DIR, exist_ok=True)
    path = os.path.join(config.METRICS_DIR, f"{model_name}_results.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(_json_ready(metrics), file, indent=2)
    return _json_ready(metrics)


def error_analysis(
    texts: list[str],
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    pred_probs: np.ndarray,
) -> pd.DataFrame:
    """Create a DataFrame of misclassified examples.

    Args:
        texts: Original review texts aligned with predictions.
        true_labels: Ground-truth labels.
        pred_labels: Predicted labels.
        pred_probs: Class probability matrix.

    Returns:
        Misclassified examples sorted by confidence descending.
    """
    rows: list[dict[str, Any]] = []
    for text, true_label, pred_label, probabilities in zip(texts, true_labels, pred_labels, pred_probs):
        if int(true_label) == int(pred_label):
            continue
        confidence = float(probabilities[int(pred_label)])
        rows.append(
            {
                "text": text,
                "true_label": int(true_label),
                "pred_label": int(pred_label),
                "true_label_name": config.ID2LABEL[int(true_label)],
                "pred_label_name": config.ID2LABEL[int(pred_label)],
                "confidence": confidence,
                "error_type": "False Positive" if int(pred_label) == 1 else "False Negative",
                "word_count": len(str(text).split()),
                "was_truncated": len(str(text).split()) > config.MAX_LENGTH,
            }
        )

    return pd.DataFrame(rows).sort_values("confidence", ascending=False).reset_index(drop=True)


def truncation_analysis(
    texts: list[str],
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    tokenizer,
    max_length: int,
) -> dict[str, Any]:
    """Analyze performance for examples inside and beyond the token limit.

    Args:
        texts: Review texts.
        true_labels: Ground-truth labels.
        pred_labels: Predicted labels.
        tokenizer: Tokenizer used by the evaluated model.
        max_length: Token length threshold.

    Returns:
        Accuracy and F1 comparison for within-limit and truncated reviews.
    """
    token_lengths = np.asarray(
        [
            len(tokenizer(str(text), add_special_tokens=True, truncation=False)["input_ids"])
            for text in tqdm(texts, desc="Token lengths", leave=False)
        ]
    )
    within_mask = token_lengths <= max_length
    truncated_mask = ~within_mask

    def _safe_accuracy(mask: np.ndarray) -> float | None:
        if int(mask.sum()) == 0:
            return None
        return float(accuracy_score(true_labels[mask], pred_labels[mask]))

    def _safe_f1(mask: np.ndarray) -> float | None:
        if int(mask.sum()) == 0:
            return None
        return float(f1_score(true_labels[mask], pred_labels[mask], average="macro", zero_division=0))

    within_accuracy = _safe_accuracy(within_mask)
    truncated_accuracy = _safe_accuracy(truncated_mask)
    accuracy_gap = (
        None
        if within_accuracy is None or truncated_accuracy is None
        else within_accuracy - truncated_accuracy
    )

    return {
        "within_limit_count": int(within_mask.sum()),
        "truncated_count": int(truncated_mask.sum()),
        "within_limit_accuracy": within_accuracy,
        "truncated_accuracy": truncated_accuracy,
        "accuracy_gap": accuracy_gap,
        "within_limit_f1": _safe_f1(within_mask),
        "truncated_f1": _safe_f1(truncated_mask),
        "token_lengths": token_lengths.tolist(),
    }


def confidence_calibration(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    pred_probs: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Assess whether confidence values align with empirical accuracy.

    Args:
        true_labels: Ground-truth labels.
        pred_labels: Predicted labels.
        pred_probs: Class probability matrix.
        n_bins: Number of confidence bins between 0.5 and 1.0.

    Returns:
        Reliability diagram data and expected calibration error.
    """
    confidences = pred_probs.max(axis=1)
    correct = (true_labels == pred_labels).astype(float)
    bin_edges = np.linspace(0.5, 1.0, n_bins + 1)
    bin_accuracies: list[float | None] = []
    bin_confidences: list[float | None] = []
    bin_counts: list[int] = []
    expected_calibration_error = 0.0

    for index in range(n_bins):
        lower = bin_edges[index]
        upper = bin_edges[index + 1]
        if index == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        count = int(mask.sum())
        bin_counts.append(count)
        if count == 0:
            bin_accuracies.append(None)
            bin_confidences.append(None)
            continue

        accuracy = float(correct[mask].mean())
        confidence = float(confidences[mask].mean())
        bin_accuracies.append(accuracy)
        bin_confidences.append(confidence)
        expected_calibration_error += (count / len(true_labels)) * abs(accuracy - confidence)

    return {
        "bin_edges": bin_edges.tolist(),
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
        "expected_calibration_error": float(expected_calibration_error),
    }

