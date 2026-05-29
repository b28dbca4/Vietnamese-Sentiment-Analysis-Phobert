"""Unified multi-model inference engine for the IMDb Sentiment platform.

This module supersedes ``src.predict.SentimentPredictor`` *for the app layer*
because the explainability and arena features need capabilities that a single
"predict one review" wrapper does not expose:

    * batched ``predict_proba`` over many perturbed strings   (occlusion / LIME)
    * access to raw logits, attentions and embedding gradients (IG / attention)
    * the ability to hold several models in memory at once      (Model Arena)

It loads the fine-tuned Transformer checkpoints (RoBERTa / BERT / DistilBERT /
best_model) directly with HuggingFace and the TF-IDF + Logistic-Regression
baseline with joblib. Heavy objects are loaded lazily and cached, so opening the
app does not pull four models into RAM at once.

Nothing here depends on Streamlit; ``app.py`` wraps the singleton in
``st.cache_resource`` so the cache survives reruns.
"""

from __future__ import annotations

import os
import sys
import time
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ── Pull paths / hyper-params from the project config, with safe fallbacks ──
try:
    from src import config as _cfg  # type: ignore

    MAX_LENGTH = int(getattr(_cfg, "MAX_LENGTH", 256))
    MODELS_DIR = Path(getattr(_cfg, "MODELS_DIR", ROOT_DIR / "models"))
except Exception:  # pragma: no cover - config should normally import fine
    MAX_LENGTH = 256
    MODELS_DIR = ROOT_DIR / "models"


# ─────────────────────────────────────────────────────────────────────────────
#  Model registry  —  edit dirs here if your folders are named differently
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    kind: str  # "transformer" | "sklearn"
    rel_dir: str
    tagline: str
    strengths: Tuple[str, ...]
    weaknesses: Tuple[str, ...]


MODEL_SPECS: Tuple[ModelSpec, ...] = (
    ModelSpec(
        "roberta", "RoBERTa", "transformer", "roberta-base",
        "Strongest representation, best accuracy/F1.",
        ("Best contextual understanding", "Top accuracy & F1", "Robust to phrasing"),
        ("Heaviest transformer", "Highest latency"),
    ),
    ModelSpec(
        "bert", "BERT-base", "transformer", "bert-base-uncased",
        "Strong bidirectional context model.",
        ("Strong context modelling", "Well-studied baseline transformer"),
        ("More parameters than DistilBERT", "Slower than DistilBERT"),
    ),
    ModelSpec(
        "distilbert", "DistilBERT", "transformer", "distilbert-base-uncased",
        "Balanced — fast transformer, small footprint.",
        ("~40% smaller than BERT", "Low latency", "Good accuracy/speed trade-off"),
        ("Slightly below BERT/RoBERTa on accuracy",),
    ),
    ModelSpec(
        "baseline", "Logistic Regression", "sklearn", "baseline",
        "TF-IDF + Logistic Regression — fast, lightweight baseline.",
        ("Fastest inference", "Tiny & interpretable", "No GPU needed"),
        ("No contextual understanding", "Bag-of-words only", "Weak on negation/sarcasm"),
    ),
)

SPEC_BY_KEY: Dict[str, ModelSpec] = {s.key: s for s in MODEL_SPECS}
# Arena ordering: cheapest → most expensive reads well in the comparison table.
ARENA_ORDER: Tuple[str, ...] = ("baseline", "distilbert", "bert", "roberta")


@dataclass
class Prediction:
    """One model's verdict on one piece of text."""
    model_key: str
    label: str                       # "positive" | "negative"
    confidence: float                # P(predicted class), 0..1
    p_positive: float
    p_negative: float
    num_tokens: int
    was_truncated: bool
    latency_ms: float
    meta: Dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> Dict[str, Any]:
        return {
            "model": SPEC_BY_KEY.get(self.model_key, None).display_name
            if self.model_key in SPEC_BY_KEY else self.model_key,
            "prediction": self.label,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Loaded-model handles
# ─────────────────────────────────────────────────────────────────────────────
class _TransformerHandle:
    """Wraps a HF model+tokenizer and standardises positive/negative mapping."""

    def __init__(self, model, tokenizer, device, pos_index: int):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.pos_index = pos_index          # which logit column == "positive"
        self.neg_index = 1 - pos_index

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


class _SklearnHandle:
    def __init__(self, vectorizer, classifier, pos_index: int):
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.pos_index = pos_index
        self.neg_index = 1 - pos_index


# ─────────────────────────────────────────────────────────────────────────────
#  The engine
# ─────────────────────────────────────────────────────────────────────────────
class InferenceEngine:
    """Lazy, cached, multi-model sentiment engine."""

    def __init__(self, models_dir: Optional[Path] = None, max_length: int = MAX_LENGTH):
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.max_length = max_length
        self._cache: Dict[str, Any] = {}
        self._torch = None  # imported on first transformer use

    # ---- availability -------------------------------------------------------
    def available_models(self) -> List[str]:
        """Keys whose checkpoint folder exists on disk."""
        out = []
        for spec in MODEL_SPECS:
            path = self.models_dir / spec.rel_dir
            if spec.kind == "transformer" and (path / "config.json").exists():
                out.append(spec.key)
            elif spec.kind == "sklearn" and path.exists():
                out.append(spec.key)
        return out

    def default_model(self) -> str:
        avail = self.available_models()
        for pref in ("roberta", "bert", "distilbert", "baseline"):
            if pref in avail:
                return pref
        return avail[0] if avail else "roberta"

    # ---- lazy loaders -------------------------------------------------------
    def _ensure_torch(self):
        if self._torch is None:
            import torch  # local import keeps cold-start light
            self._torch = torch
        return self._torch

    def _load_transformer(self, spec: ModelSpec) -> _TransformerHandle:
        torch = self._ensure_torch()
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        path = str(self.models_dir / spec.rel_dir)
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(
            path, output_attentions=True
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()

        pos_index = _resolve_positive_index_from_config(model.config)
        return _TransformerHandle(model, tokenizer, device, pos_index)

    def _load_sklearn(self, spec: ModelSpec) -> _SklearnHandle:
        path = self.models_dir / spec.rel_dir
        vectorizer = _load_pickle(path / "tfidf_vectorizer.pkl")
        classifier = _load_pickle(path / "logistic_regression.pkl")
        pos_index = _resolve_positive_index_from_classes(
            getattr(classifier, "classes_", np.array([0, 1]))
        )
        return _SklearnHandle(vectorizer, classifier, pos_index)

    def get(self, key: str):
        """Return a loaded handle for ``key`` (loading + caching on first use)."""
        if key in self._cache:
            return self._cache[key]
        spec = SPEC_BY_KEY[key]
        handle = (
            self._load_transformer(spec)
            if spec.kind == "transformer"
            else self._load_sklearn(spec)
        )
        self._cache[key] = handle
        return handle

    # ---- core inference -----------------------------------------------------
    def predict_proba_batch(self, texts: List[str], key: str) -> np.ndarray:
        """Return an (N, 2) array of [P(negative), P(positive)] for ``texts``.

        This is the workhorse used by occlusion and LIME — both feed many
        perturbed strings through here. Columns are ALWAYS [neg, pos] regardless
        of the model's internal label ordering.
        """
        spec = SPEC_BY_KEY[key]
        if spec.kind == "sklearn":
            return self._proba_sklearn(texts, key)
        return self._proba_transformer(texts, key)

    def _proba_sklearn(self, texts: List[str], key: str) -> np.ndarray:
        h: _SklearnHandle = self.get(key)
        feats = h.vectorizer.transform(texts)
        proba = h.classifier.predict_proba(feats)  # ordered by classes_
        pos = proba[:, h.pos_index]
        neg = proba[:, h.neg_index]
        return np.column_stack([neg, pos])

    def _proba_transformer(
        self, texts: List[str], key: str, batch_size: int = 32
    ) -> np.ndarray:
        torch = self._ensure_torch()
        h: _TransformerHandle = self.get(key)
        out_neg, out_pos = [], []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                chunk = texts[start:start + batch_size]
                enc = h.tokenizer(
                    chunk,
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                ).to(h.device)
                logits = h.model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                out_pos.append(probs[:, h.pos_index])
                out_neg.append(probs[:, h.neg_index])
        neg = np.concatenate(out_neg)
        pos = np.concatenate(out_pos)
        return np.column_stack([neg, pos])

    def predict_score_batch(self, texts: List[str], key: str) -> np.ndarray:
        """Return an (N,) array of *unsaturated* signed scores toward positive.

        For transformers this is the logit margin ``logit_pos - logit_neg``;
        for the sklearn baseline it is the decision-function margin. Unlike
        ``P(positive)`` (which saturates near 0/1 when the model is confident),
        the margin keeps responding to perturbations — so occlusion deltas
        computed on it don't all collapse to ~0.00 on easy, high-confidence
        reviews. This is the standard fix for occlusion under softmax saturation.
        """
        spec = SPEC_BY_KEY[key]
        if spec.kind == "sklearn":
            return self._score_sklearn(texts, key)
        return self._score_transformer(texts, key)

    def _score_sklearn(self, texts: List[str], key: str) -> np.ndarray:
        h: _SklearnHandle = self.get(key)
        feats = h.vectorizer.transform(texts)
        if hasattr(h.classifier, "decision_function"):
            margin = h.classifier.decision_function(feats)
            margin = np.asarray(margin, dtype=float)
            if margin.ndim > 1:  # multiclass shape (N, C)
                margin = margin[:, h.pos_index] - margin[:, h.neg_index]
            # binary decision_function: positive value favours classes_[1]
            return margin if h.pos_index == 1 else -margin
        # fallback: logit of the probability
        proba = self.predict_proba_batch(texts, key)[:, 1]
        proba = np.clip(proba, 1e-6, 1 - 1e-6)
        return np.log(proba / (1 - proba))

    def _score_transformer(
        self, texts: List[str], key: str, batch_size: int = 32
    ) -> np.ndarray:
        torch = self._ensure_torch()
        h: _TransformerHandle = self.get(key)
        out = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                chunk = texts[start:start + batch_size]
                enc = h.tokenizer(
                    chunk, truncation=True, max_length=self.max_length,
                    padding=True, return_tensors="pt",
                ).to(h.device)
                logits = h.model(**enc).logits.cpu().numpy()
                out.append(logits[:, h.pos_index] - logits[:, h.neg_index])
        return np.concatenate(out)

    def predict(self, text: str, key: str) -> Prediction:
        """Single-text prediction with timing + token diagnostics."""
        spec = SPEC_BY_KEY[key]
        start = time.perf_counter()
        proba = self.predict_proba_batch([text], key)[0]
        latency_ms = (time.perf_counter() - start) * 1000.0

        p_neg, p_pos = float(proba[0]), float(proba[1])
        label = "positive" if p_pos >= p_neg else "negative"
        confidence = max(p_pos, p_neg)

        num_tokens, truncated = self._token_diagnostics(text, key)
        return Prediction(
            model_key=key,
            label=label,
            confidence=confidence,
            p_positive=p_pos,
            p_negative=p_neg,
            num_tokens=num_tokens,
            was_truncated=truncated,
            latency_ms=latency_ms,
            meta={"kind": spec.kind},
        )

    def predict_all(self, text: str, keys: Optional[List[str]] = None) -> List[Prediction]:
        """Run several models for the Arena (in a sensible, readable order)."""
        keys = keys or [k for k in ARENA_ORDER if k in self.available_models()]
        return [self.predict(text, k) for k in keys]

    def _token_diagnostics(self, text: str, key: str) -> Tuple[int, bool]:
        spec = SPEC_BY_KEY[key]
        if spec.kind == "sklearn":
            n = len(text.split())
            return n, False
        h: _TransformerHandle = self.get(key)
        ids = h.tokenizer(text, truncation=False)["input_ids"]
        truncated = len(ids) > self.max_length
        return min(len(ids), self.max_length), truncated

    # ---- info ---------------------------------------------------------------
    def model_info(self, key: str) -> Dict[str, Any]:
        spec = SPEC_BY_KEY[key]
        info: Dict[str, Any] = {
            "key": key,
            "display_name": spec.display_name,
            "kind": spec.kind,
            "tagline": spec.tagline,
            "max_length": self.max_length,
        }
        if key in self._cache and spec.kind == "transformer":
            h: _TransformerHandle = self._cache[key]
            info["params"] = h.num_params
            info["device"] = h.device
        return info


# ─────────────────────────────────────────────────────────────────────────────
#  Label-mapping helpers  (the #1 source of silent "positive/negative flipped" bugs)
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_positive_index_from_config(config) -> int:
    """Find which logit index is the positive class from a HF config."""
    id2label = getattr(config, "id2label", None)
    if id2label:
        for idx, name in id2label.items():
            if _looks_positive(str(name)):
                return int(idx)
        for idx, name in id2label.items():
            if _looks_negative(str(name)):
                return 1 - int(idx)
    return 1  # convention: index 1 = positive


def _resolve_positive_index_from_classes(classes) -> int:
    for i, c in enumerate(classes):
        if _looks_positive(str(c)) or str(c) in {"1"}:
            return i
    for i, c in enumerate(classes):
        if _looks_negative(str(c)) or str(c) in {"0"}:
            return 1 - i
    return len(classes) - 1  # last class as positive fallback


def _looks_positive(name: str) -> bool:
    n = name.strip().lower()
    return n in {"positive", "pos", "label_1", "1"} or "pos" in n


def _looks_negative(name: str) -> bool:
    n = name.strip().lower()
    return n in {"negative", "neg", "label_0", "0"} or "neg" in n


def _load_pickle(path: Path):
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as fh:
            return pickle.load(fh)


# Convenience singleton for app.py (wrapped by st.cache_resource there).
_ENGINE: Optional[InferenceEngine] = None


def get_engine() -> InferenceEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = InferenceEngine()
    return _ENGINE
