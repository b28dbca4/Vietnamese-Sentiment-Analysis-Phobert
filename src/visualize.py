"""Plotting utilities for model evaluation and comparison."""

from __future__ import annotations

import os
import tempfile
from typing import Any

_MPL_CONFIG_DIR = os.path.join(tempfile.gettempdir(), "matplotlib-cache")
os.makedirs(_MPL_CONFIG_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _MPL_CONFIG_DIR)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.size": 12,
        "figure.figsize": (10, 6),
        "figure.dpi": config.FIGURE_DPI,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
    }
)


def _prepare_save(save_path: str) -> None:
    """Create the parent directory for a figure path."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)


def _display_name(model_key: str) -> str:
    """Return a readable model name."""
    return config.MODEL_DISPLAY_NAMES.get(model_key, model_key)


def _metric_value(values: dict[str, Any], *keys: str) -> float:
    """Read the first available metric key from a dictionary."""
    for key in keys:
        if key in values and values[key] is not None:
            return float(values[key])
    return float("nan")


def plot_confusion_matrix(cm, model_name: str, save_path: str) -> None:
    """Plot a confusion matrix with counts and percentages.

    Args:
        cm: Two-by-two confusion matrix.
        model_name: Model name for the title.
        save_path: Output PNG path.
    """
    _prepare_save(save_path)
    matrix = np.asarray(cm)
    total = matrix.sum()
    labels = np.array(
        [
            [f"{matrix[row, col]}\n{matrix[row, col] / total:.1%}" for col in range(matrix.shape[1])]
            for row in range(matrix.shape[0])
        ]
    )
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=labels,
        fmt="",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        cbar=False,
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()


def plot_roc_curve_single(fpr, tpr, auc_score: float, model_name: str, save_path: str) -> None:
    """Plot one ROC curve with a random-reference diagonal."""
    _prepare_save(save_path)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color=config.COLORS.get("bert"), linewidth=2, label=f"{model_name} (AUC={auc_score:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#6b7280", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()


def plot_roc_curves_comparison(roc_data: dict[str, dict[str, Any]], save_path: str) -> None:
    """Plot ROC curves for multiple models on the same axes."""
    _prepare_save(save_path)
    plt.figure(figsize=(8, 6))
    for model_key, values in roc_data.items():
        if "fpr" not in values or "tpr" not in values:
            continue
        auc_score = values.get("auc", values.get("roc_auc", float("nan")))
        plt.plot(
            values["fpr"],
            values["tpr"],
            linewidth=2,
            color=config.COLORS.get(model_key),
            label=f"{_display_name(model_key)} (AUC={float(auc_score):.4f})",
        )
    plt.plot([0, 1], [0, 1], linestyle="--", color="#6b7280", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()


def plot_training_curves(history: list[dict[str, Any]], model_name: str, save_path: str) -> None:
    """Plot loss, accuracy, F1, and learning rate over epochs."""
    _prepare_save(save_path)
    if not history:
        raise ValueError("Training history is empty.")

    df = pd.DataFrame(history)
    epochs = df["epoch"] if "epoch" in df else np.arange(1, len(df) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(epochs, df["train_loss"], marker="o", label="Train Loss")
    axes[0, 0].plot(epochs, df["val_loss"], marker="o", label="Validation Loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, df["val_accuracy"], marker="o", color=config.COLORS.get("positive"))
    axes[0, 1].set_title("Validation Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")

    axes[1, 0].plot(epochs, df["val_f1"], marker="o", color=config.COLORS.get("bert"))
    axes[1, 0].set_title("Validation F1")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("F1 Macro")

    if "learning_rate" in df:
        axes[1, 1].plot(epochs, df["learning_rate"], marker="o", color=config.COLORS.get("distilbert"))
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning Rate")

    fig.suptitle(f"Training Curves - {model_name}", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves_comparison(histories: dict[str, list[dict[str, Any]]], save_path: str) -> None:
    """Overlay validation curves for Transformer models."""
    _prepare_save(save_path)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [("val_loss", "Validation Loss"), ("val_accuracy", "Validation Accuracy"), ("val_f1", "Validation F1")]

    for metric_key, title in metrics:
        axis = axes[metrics.index((metric_key, title))]
        for model_key, history in histories.items():
            if not history:
                continue
            df = pd.DataFrame(history)
            epochs = df["epoch"] if "epoch" in df else np.arange(1, len(df) + 1)
            axis.plot(
                epochs,
                df[metric_key],
                marker="o",
                label=_display_name(model_key),
                color=config.COLORS.get(model_key),
            )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.legend()

    fig.suptitle("Transformer Training Comparison", y=1.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison_bar(metrics: dict[str, dict[str, Any]], save_path: str) -> None:
    """Create grouped bars for Accuracy, F1, and AUC."""
    _prepare_save(save_path)
    model_keys = list(metrics.keys())
    metric_names = ["Accuracy", "F1", "AUC"]
    metric_keys = [("accuracy",), ("f1_macro", "f1"), ("roc_auc", "auc")]
    x = np.arange(len(metric_names))
    width = 0.8 / max(1, len(model_keys))

    plt.figure(figsize=(11, 6))
    for index, model_key in enumerate(model_keys):
        values = [_metric_value(metrics[model_key], *keys) for keys in metric_keys]
        offset = (index - (len(model_keys) - 1) / 2) * width
        bars = plt.bar(
            x + offset,
            values,
            width,
            label=_display_name(model_key),
            color=config.COLORS.get(model_key),
        )
        for bar in bars:
            height = bar.get_height()
            if np.isfinite(height):
                plt.text(bar.get_x() + bar.get_width() / 2, height + 0.005, f"{height:.3f}", ha="center", va="bottom", fontsize=8)

    plt.xticks(x, metric_names)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()


def plot_model_comparison_radar(metrics: dict[str, dict[str, Any]], save_path: str) -> None:
    """Plot a radar chart across five evaluation metrics."""
    _prepare_save(save_path)
    labels = ["Accuracy", "F1", "Precision", "Recall", "AUC"]
    key_sets = [
        ("accuracy",),
        ("f1_macro", "f1"),
        ("precision_macro", "precision"),
        ("recall_macro", "recall"),
        ("roc_auc", "auc"),
    ]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    axis = fig.add_subplot(111, polar=True)
    for model_key, values in metrics.items():
        scores = [_metric_value(values, *keys) for keys in key_sets]
        scores += scores[:1]
        axis.plot(angles, scores, linewidth=2, label=_display_name(model_key), color=config.COLORS.get(model_key))
        axis.fill(angles, scores, alpha=0.12, color=config.COLORS.get(model_key))

    axis.set_xticks(angles[:-1])
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 1)
    axis.set_title("Radar Comparison", pad=20)
    axis.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    fig.tight_layout()
    fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_confidence_distribution(
    probs: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    model_name: str,
    save_path: str,
) -> None:
    """Plot confidence distributions for correct and incorrect predictions."""
    _prepare_save(save_path)
    confidences = np.asarray(probs).max(axis=1)
    correct = np.asarray(true_labels) == np.asarray(pred_labels)

    plt.figure(figsize=(9, 6))
    plt.hist(confidences[correct], bins=20, alpha=0.7, label="Correct", color=config.COLORS["positive"])
    plt.hist(confidences[~correct], bins=20, alpha=0.7, label="Incorrect", color=config.COLORS["negative"])
    plt.xlabel("Prediction Confidence")
    plt.ylabel("Count")
    plt.title(f"Confidence Distribution - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()


def plot_classification_report_heatmap(report: dict[str, Any], model_name: str, save_path: str) -> None:
    """Plot precision, recall, and F1 from a classification report."""
    _prepare_save(save_path)
    rows = []
    row_labels = []
    for label in ["negative", "positive", "macro avg", "weighted avg"]:
        if label in report:
            row_labels.append(label)
            rows.append([report[label].get("precision", 0), report[label].get("recall", 0), report[label].get("f1-score", 0)])

    df = pd.DataFrame(rows, index=row_labels, columns=["Precision", "Recall", "F1"])
    plt.figure(figsize=(8, 5))
    sns.heatmap(df, annot=True, fmt=".4f", cmap="YlGnBu", vmin=0, vmax=1)
    plt.title(f"Classification Report - {model_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()


def plot_calibration_curve(calibration_data: dict[str, Any], model_name: str, save_path: str) -> None:
    """Plot a reliability diagram for confidence calibration."""
    _prepare_save(save_path)
    confidences = np.asarray([
        value for value in calibration_data["bin_confidences"] if value is not None
    ])
    accuracies = np.asarray([
        value for value in calibration_data["bin_accuracies"] if value is not None
    ])

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="#6b7280", label="Perfect Calibration")
    if len(confidences):
        plt.plot(confidences, accuracies, marker="o", linewidth=2, color=config.COLORS.get("bert"), label=model_name)
    plt.xlabel("Mean Predicted Confidence")
    plt.ylabel("Empirical Accuracy")
    plt.title(f"Calibration Curve - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()


def plot_efficiency_comparison(model_stats: dict[str, dict[str, Any]], save_path: str) -> None:
    """Plot accuracy against parameter count with train-time bubble sizes."""
    _prepare_save(save_path)
    plt.figure(figsize=(9, 6))
    for model_key, stats in model_stats.items():
        params_m = _metric_value(stats, "params_millions", "params") / (1 if "params_millions" in stats else 1_000_000)
        accuracy = _metric_value(stats, "accuracy")
        train_time = max(_metric_value(stats, "train_time_minutes", "train_time_seconds"), 0.1)
        bubble_size = 80 + train_time * 20
        plt.scatter(
            params_m,
            accuracy,
            s=bubble_size,
            color=config.COLORS.get(model_key),
            alpha=0.75,
            label=_display_name(model_key),
            edgecolor="black",
            linewidth=0.5,
        )
        plt.text(params_m, accuracy + 0.003, _display_name(model_key), ha="center", fontsize=9)

    plt.xlabel("Parameters (Millions)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Model Complexity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
