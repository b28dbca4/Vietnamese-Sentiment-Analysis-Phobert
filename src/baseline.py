"""Traditional machine learning baseline for IMDb sentiment analysis."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import config
from src.preprocessing import clean_text


def _json_ready(metrics: dict[str, Any]) -> dict[str, Any]:
    """Convert NumPy values into JSON-serializable Python values."""
    converted: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            converted[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            converted[key] = value.item()
        elif isinstance(value, dict):
            converted[key] = _json_ready(value)
        elif isinstance(value, list):
            converted[key] = [
                item.item() if isinstance(item, (np.integer, np.floating)) else item for item in value
            ]
        else:
            converted[key] = value
    return converted


def _save_json(metrics: dict[str, Any], path: str) -> None:
    """Save a metric dictionary as formatted JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(_json_ready(metrics), file, indent=2)


def _compute_metrics(
    labels: list[int] | np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    """Compute the metric suite used by the baseline model."""
    labels_array = np.asarray(labels)
    report = classification_report(
        labels_array,
        predictions,
        target_names=[config.ID2LABEL[index] for index in sorted(config.ID2LABEL)],
        output_dict=True,
        zero_division=0,
    )
    try:
        auc_score = roc_auc_score(labels_array, probabilities[:, config.LABEL2ID["positive"]])
    except ValueError:
        auc_score = None

    f1_macro = f1_score(labels_array, predictions, average="macro", zero_division=0)
    precision_macro = precision_score(labels_array, predictions, average="macro", zero_division=0)
    recall_macro = recall_score(labels_array, predictions, average="macro", zero_division=0)

    return {
        "accuracy": accuracy_score(labels_array, predictions),
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_score(labels_array, predictions, average="weighted", zero_division=0),
        "recall_weighted": recall_score(labels_array, predictions, average="weighted", zero_division=0),
        "f1_weighted": f1_score(labels_array, predictions, average="weighted", zero_division=0),
        "precision": precision_macro,
        "recall": recall_macro,
        "f1": f1_macro,
        "roc_auc": auc_score,
        "auc": auc_score,
        "confusion_matrix": confusion_matrix(labels_array, predictions, labels=[0, 1]),
        "classification_report": report,
    }


def _artifact_paths() -> tuple[str, str]:
    """Return canonical baseline artifact paths."""
    vectorizer_path = os.path.join(config.BASELINE_DIR, "tfidf_vectorizer.pkl")
    model_path = os.path.join(config.BASELINE_DIR, "logistic_regression.pkl")
    return vectorizer_path, model_path


def train_baseline(
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str],
    val_labels: list[int],
) -> dict[str, Any]:
    """Train a TF-IDF plus Logistic Regression baseline.

    Args:
        train_texts: Training review texts.
        train_labels: Training integer labels.
        val_texts: Validation review texts.
        val_labels: Validation integer labels.

    Returns:
        Validation metrics and training metadata.
    """
    os.makedirs(config.BASELINE_DIR, exist_ok=True)
    start_time = time.time()

    vectorizer = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        min_df=config.TFIDF_MIN_DF,
        max_df=config.TFIDF_MAX_DF,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    train_features = vectorizer.fit_transform(train_texts)
    val_features = vectorizer.transform(val_texts)

    classifier = LogisticRegression(
        C=config.LOGREG_C,
        max_iter=config.LOGREG_MAX_ITER,
        solver="lbfgs",
        random_state=config.SEED,
        n_jobs=-1,
    )
    classifier.fit(train_features, train_labels)

    val_predictions = classifier.predict(val_features)
    val_probabilities = classifier.predict_proba(val_features)
    metrics = _compute_metrics(val_labels, val_predictions, val_probabilities)
    metrics.update(
        {
            "model_key": "baseline",
            "model_name": "TF-IDF + Logistic Regression",
            "training_time_seconds": time.time() - start_time,
            "num_features": int(len(vectorizer.get_feature_names_out())),
            "hyperparameters": {
                "tfidf_max_features": config.TFIDF_MAX_FEATURES,
                "tfidf_ngram_range": list(config.TFIDF_NGRAM_RANGE),
                "tfidf_min_df": config.TFIDF_MIN_DF,
                "tfidf_max_df": config.TFIDF_MAX_DF,
                "logreg_c": config.LOGREG_C,
                "logreg_max_iter": config.LOGREG_MAX_ITER,
            },
        }
    )

    vectorizer_path, model_path = _artifact_paths()
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(classifier, model_path)
    _save_json(metrics, os.path.join(config.LOG_DIR, "baseline_results.json"))
    return _json_ready(metrics)


def evaluate_baseline(test_texts: list[str], test_labels: list[int]) -> dict[str, Any]:
    """Evaluate the saved baseline on the test split.

    Args:
        test_texts: Test review texts.
        test_labels: Test integer labels.

    Returns:
        Test metrics, predictions, and probabilities.
    """
    vectorizer_path, model_path = _artifact_paths()
    if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Baseline artifacts were not found. Run train_baseline first.")

    vectorizer: TfidfVectorizer = joblib.load(vectorizer_path)
    classifier: LogisticRegression = joblib.load(model_path)

    test_features = vectorizer.transform(test_texts)
    predictions = classifier.predict(test_features)
    probabilities = classifier.predict_proba(test_features)

    metrics = _compute_metrics(test_labels, predictions, probabilities)
    metrics.update(
        {
            "model_key": "baseline",
            "model_name": "TF-IDF + Logistic Regression",
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist(),
        }
    )
    _save_json(metrics, os.path.join(config.METRICS_DIR, "baseline_results.json"))
    return _json_ready(metrics)


def predict_baseline(text: str) -> dict[str, Any]:
    """Predict sentiment for one text with the saved baseline.

    Args:
        text: Raw review text.

    Returns:
        Prediction label, confidence, probabilities, and cleaned text.
    """
    vectorizer_path, model_path = _artifact_paths()
    if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Baseline artifacts were not found. Run train_baseline first.")

    vectorizer: TfidfVectorizer = joblib.load(vectorizer_path)
    classifier: LogisticRegression = joblib.load(model_path)

    cleaned_text = clean_text(text)
    features = vectorizer.transform([cleaned_text])
    probabilities = classifier.predict_proba(features)[0]
    prediction_id = int(np.argmax(probabilities))
    label = config.ID2LABEL[prediction_id]
    return {
        "label": label,
        "confidence": float(probabilities[prediction_id]),
        "probabilities": {
            "negative": float(probabilities[config.LABEL2ID["negative"]]),
            "positive": float(probabilities[config.LABEL2ID["positive"]]),
        },
        "cleaned_text": cleaned_text,
    }


def get_top_features(n: int = 20) -> dict[str, list[tuple[str, float]]]:
    """Return the most positive and negative baseline coefficients.

    Args:
        n: Number of features to return per class direction.

    Returns:
        Dictionary containing positive and negative feature coefficient lists.
    """
    vectorizer_path, model_path = _artifact_paths()
    if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Baseline artifacts were not found. Run train_baseline first.")

    vectorizer: TfidfVectorizer = joblib.load(vectorizer_path)
    classifier: LogisticRegression = joblib.load(model_path)
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    coefficients = classifier.coef_[0]

    positive_indices = np.argsort(coefficients)[-n:][::-1]
    negative_indices = np.argsort(coefficients)[:n]
    return {
        "positive": [(str(feature_names[index]), float(coefficients[index])) for index in positive_indices],
        "negative": [(str(feature_names[index]), float(coefficients[index])) for index in negative_indices],
    }

