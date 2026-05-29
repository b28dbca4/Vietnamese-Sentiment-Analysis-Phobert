"""Central configuration for the IMDb sentiment analysis project.

All shared constants, paths, hyperparameters, labels, and visualization settings
live in this module so experiments stay reproducible and easy to audit.
"""

from __future__ import annotations

import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BEST_MODEL_DIR = os.path.join(MODELS_DIR, "best_model")
BASELINE_DIR = os.path.join(MODELS_DIR, "baseline")
LOG_DIR = os.path.join(MODELS_DIR, "training_logs")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

# Model zoo
TRANSFORMER_MODELS = {
    "distilbert": "distilbert-base-uncased",
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
}
MODEL_DISPLAY_NAMES = {
    "baseline": "Baseline",
    "distilbert": "DistilBERT",
    "bert": "BERT-base",
    "roberta": "RoBERTa",
}
NUM_LABELS = 2

# Training hyperparameters
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0
EARLY_STOPPING_PATIENCE = 2

# Baseline hyperparameters
TFIDF_MAX_FEATURES = 50000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
LOGREG_C = 1.0
LOGREG_MAX_ITER = 1000

# Labels
LABEL2ID = {"negative": 0, "positive": 1}
ID2LABEL = {0: "negative", 1: "positive"}

# Reproducibility
SEED = 42

# Visualization
COLORS = {
    "baseline": "#94a3b8",
    "distilbert": "#f59e0b",
    "bert": "#3b82f6",
    "roberta": "#8b5cf6",
    "positive": "#059669",
    "negative": "#dc2626",
}
FIGURE_DPI = 150

