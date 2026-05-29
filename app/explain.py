"""Explainability core — the heart of the platform.

Every method here answers ONE question with a consistent sign convention:

        "How much does each word push the prediction toward POSITIVE sentiment?"

    score > 0  → the word makes the review *more positive*   (rendered green)
    score < 0  → the word makes the review *more negative*   (rendered red)
    score ≈ 0  → little influence                            (rendered neutral)

Keeping the convention fixed (toward the positive class, NOT toward "whatever
was predicted") means the heatmap reads the same way on every review and every
method, which is what makes a side-by-side method comparison meaningful.

Methods, from most to least faithful for this use-case:

  1. OCCLUSION (leave-one-out)  — model-agnostic, exact, always available.
        Remove word i, measure ΔP(positive). This is a *causal* probe of the
        actual model and is trivial to defend to an examiner.

  2. LIME                       — model-agnostic local surrogate (optional dep).
        The standard academic baseline to name-drop. Approximate but principled.

  3. INTEGRATED GRADIENTS       — gradient attribution in embedding space
        (Captum, transformers only). The "deepest" white-box method.

  4. ATTENTION                  — last-layer attention magnitude (transformers).
        Shown with an explicit caveat: attention is *not* a faithful explanation
        (Jain & Wallace, 2019). Presented as "where the model looks", not "why".

All methods return the same :class:`Explanation` object so the UI has one
rendering path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from engine import InferenceEngine, SPEC_BY_KEY

# A "word" is a run of word-characters (keeping internal apostrophes) OR a
# single piece of punctuation. We keep spans so we can occlude in-place and so
# subword token attributions (IG / attention) can be mapped back onto words.
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\sA-Za-z0-9]")


@dataclass
class WordScore:
    text: str
    start: int
    end: int
    is_word: bool          # False for standalone punctuation
    score: float = 0.0     # signed, toward positive sentiment (display units)
    norm: float = 0.0      # signed, in [-1, 1] for colouring
    pct: float = 0.0       # signed, % of total |attribution| (for readable labels)


@dataclass
class Explanation:
    method: str
    method_label: str
    model_key: str
    predicted_label: str        # "positive" | "negative"
    p_positive: float
    tokens: List[WordScore]
    directional: bool = True    # False → magnitude-only (attention)
    caveat: str = ""
    meta: Dict[str, object] = field(default_factory=dict)

    # -- convenience views ---------------------------------------------------
    def top_positive(self, k: int = 6) -> List[WordScore]:
        words = [t for t in self.tokens if t.is_word and t.score > 0]
        words.sort(key=lambda t: t.score, reverse=True)
        return words[:k]

    def top_negative(self, k: int = 6) -> List[WordScore]:
        words = [t for t in self.tokens if t.is_word and t.score < 0]
        words.sort(key=lambda t: t.score)
        return words[:k]

    def evidence_balance(self) -> Tuple[float, float]:
        """Total positive vs total negative pull (absolute magnitudes)."""
        pos = sum(t.score for t in self.tokens if t.score > 0)
        neg = sum(-t.score for t in self.tokens if t.score < 0)
        return float(pos), float(neg)


# ─────────────────────────────────────────────────────────────────────────────
#  Tokenisation shared by every method
# ─────────────────────────────────────────────────────────────────────────────
def tokenize_words(text: str) -> List[WordScore]:
    out: List[WordScore] = []
    for m in _WORD_RE.finditer(text):
        tok = m.group(0)
        out.append(
            WordScore(
                text=tok,
                start=m.start(),
                end=m.end(),
                is_word=bool(re.match(r"[A-Za-z0-9]", tok)),
            )
        )
    return out


def _normalise(tokens: List[WordScore]) -> None:
    peak = max((abs(t.score) for t in tokens), default=0.0)
    total = sum(abs(t.score) for t in tokens if t.is_word) or 1e-12
    if peak <= 1e-12:
        for t in tokens:
            t.norm = 0.0
            t.pct = 0.0
        return
    for t in tokens:
        t.norm = float(np.clip(t.score / peak, -1.0, 1.0))
        # signed share of the total attribution magnitude → always readable,
        # never collapses to 0.00 the way tiny raw probability deltas do.
        t.pct = float(t.score / total * 100.0) if t.is_word else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Method availability
# ─────────────────────────────────────────────────────────────────────────────
def _has_module(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


def available_methods(model_key: str) -> List[Tuple[str, str]]:
    """Return (method_id, human_label) pairs valid for this model."""
    kind = SPEC_BY_KEY[model_key].kind
    methods = [("occlusion", "Occlusion (leave-one-out)")]
    if _has_module("lime"):
        methods.append(("lime", "LIME (local surrogate)"))
    if kind == "transformer" and _has_module("captum"):
        methods.append(("integrated_gradients", "Integrated Gradients"))
    if kind == "transformer":
        methods.append(("attention", "Attention (where it looks)"))
    return methods


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────────────
def explain(
    engine: InferenceEngine,
    text: str,
    model_key: str,
    method: str = "occlusion",
    *,
    max_words: int = 120,
    lime_samples: int = 400,
    ig_steps: int = 32,
) -> Explanation:
    text = text.strip()
    proba = engine.predict_proba_batch([text], model_key)[0]
    p_pos = float(proba[1])
    predicted = "positive" if p_pos >= 0.5 else "negative"

    if method == "occlusion":
        exp = _explain_occlusion(engine, text, model_key, max_words)
    elif method == "lime":
        exp = _explain_lime(engine, text, model_key, lime_samples)
    elif method == "integrated_gradients":
        exp = _explain_integrated_gradients(engine, text, model_key, ig_steps)
    elif method == "attention":
        exp = _explain_attention(engine, text, model_key)
    else:
        raise ValueError(f"Unknown explanation method: {method!r}")

    exp.predicted_label = predicted
    exp.p_positive = p_pos
    exp.model_key = model_key
    _normalise(exp.tokens)
    return exp


# ─────────────────────────────────────────────────────────────────────────────
#  1) OCCLUSION  — the faithful workhorse
# ─────────────────────────────────────────────────────────────────────────────
def _explain_occlusion(
    engine: InferenceEngine, text: str, model_key: str, max_words: int
) -> Explanation:
    tokens = tokenize_words(text)
    word_idx = [i for i, t in enumerate(tokens) if t.is_word]

    # For very long reviews, occlude only the first `max_words` content words to
    # keep the interaction snappy (each occlusion is one extra forward pass).
    truncated_explain = False
    if len(word_idx) > max_words:
        word_idx = word_idx[:max_words]
        truncated_explain = True

    # Score on the LOGIT MARGIN, not P(positive). When a model is very confident
    # P(positive) sits at ~0.99 and barely moves when a word is removed, so every
    # delta rounds to 0.00. The logit margin doesn't saturate, so contributions
    # stay informative even on easy reviews (Zeiler & Fergus 2014; MATLAB
    # occlusionSensitivity uses the same change-in-score idea).
    base_margin = float(engine.predict_score_batch([text], model_key)[0])
    base_pos = float(engine.predict_proba_batch([text], model_key)[0][1])

    chars = list(text)
    variants: List[str] = []
    for i in word_idx:
        t = tokens[i]
        masked = chars.copy()
        for c in range(t.start, t.end):
            masked[c] = " "
        variants.append("".join(masked))

    if variants:
        margins = engine.predict_score_batch(variants, model_key)
        for j, i in enumerate(word_idx):
            # Δmargin = margin(full) − margin(word removed)
            #   > 0 : removing the word lowered positivity → word was positive
            tokens[i].score = base_margin - float(margins[j])

    caveat = ""
    if truncated_explain:
        caveat = (
            f"Only the first {max_words} content words were probed "
            "(occlusion cost grows with review length)."
        )
    return Explanation(
        method="occlusion",
        method_label="Occlusion (leave-one-out)",
        model_key=model_key,
        predicted_label="",
        p_positive=base_pos,
        tokens=tokens,
        directional=True,
        caveat=caveat,
        meta={
            "scale": "logit-margin",
            "description": (
                "Each word is removed and the model re-run. The change in the "
                "logit margin (logit_pos − logit_neg) is that word's causal "
                "contribution — a faithful probe that, unlike raw probability, "
                "does not saturate when the model is highly confident."
            ),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
#  2) LIME  — local linear surrogate
# ─────────────────────────────────────────────────────────────────────────────
def _explain_lime(
    engine: InferenceEngine, text: str, model_key: str, num_samples: int
) -> Explanation:
    from lime.lime_text import LimeTextExplainer

    def classifier_fn(strings: Sequence[str]) -> np.ndarray:
        # LIME wants columns ordered as class_names below: [negative, positive]
        return engine.predict_proba_batch(list(strings), model_key)

    explainer = LimeTextExplainer(class_names=["negative", "positive"])
    lime_exp = explainer.explain_instance(
        text,
        classifier_fn,
        num_features=40,
        num_samples=num_samples,
        labels=(1,),  # explain the positive class so weights match our sign
    )
    weights: Dict[str, float] = {}
    for word, weight in lime_exp.as_list(label=1):
        weights[word.lower()] = weights.get(word.lower(), 0.0) + float(weight)

    tokens = tokenize_words(text)
    for t in tokens:
        if t.is_word:
            t.score = weights.get(t.text.lower(), 0.0)

    return Explanation(
        method="lime",
        method_label="LIME (local surrogate)",
        model_key=model_key,
        predicted_label="",
        p_positive=0.0,
        tokens=tokens,
        directional=True,
        caveat=(
            "LIME fits a sparse linear model to perturbations of this single "
            "review. It is an approximation of local behaviour, not the model "
            "itself."
        ),
        meta={
            "num_samples": num_samples,
            "description": (
                f"{num_samples} perturbed versions of this review (random word "
                "dropout) were labelled by the model, then a sparse weighted "
                "linear model was fit to that local neighbourhood. The linear "
                "weights are the word contributions shown here."
            ),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
#  3) INTEGRATED GRADIENTS  — white-box gradient attribution (transformers)
# ─────────────────────────────────────────────────────────────────────────────
def _explain_integrated_gradients(
    engine: InferenceEngine, text: str, model_key: str, n_steps: int
) -> Explanation:
    import torch
    from captum.attr import LayerIntegratedGradients

    handle = engine.get(model_key)
    model, tokenizer, device = handle.model, handle.tokenizer, handle.device
    pos_index = handle.pos_index

    enc = tokenizer(
        text,
        truncation=True,
        max_length=engine.max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    ref_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.unk_token_id or 0
    )
    baseline_ids = torch.full_like(input_ids, ref_id)

    def forward_fn(ids, mask):
        return model(input_ids=ids, attention_mask=mask).logits

    lig = LayerIntegratedGradients(forward_fn, model.get_input_embeddings())
    attributions = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attention_mask,),
        target=pos_index,           # attribute toward the positive logit
        n_steps=n_steps,
    )
    # Sum over the embedding dimension → one score per (sub)token.
    token_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()

    tokens = _map_token_scores_to_words(text, offsets, token_scores)
    return Explanation(
        method="integrated_gradients",
        method_label="Integrated Gradients",
        model_key=model_key,
        predicted_label="",
        p_positive=0.0,
        tokens=tokens,
        directional=True,
        caveat=(
            "Gradients are integrated from a baseline (padding) to the real "
            "input. Sub-word attributions are summed back to whole words."
        ),
        meta={
            "n_steps": n_steps,
            "description": (
                f"Gradients of the positive logit w.r.t. the input embeddings "
                f"were integrated over {n_steps} steps along a straight path "
                "from a padding baseline to the real input. Sub-word "
                "attributions were summed back to whole words."
            ),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
#  4) ATTENTION  — where the model looks (magnitude only, caveated)
# ─────────────────────────────────────────────────────────────────────────────
def _explain_attention(
    engine: InferenceEngine, text: str, model_key: str
) -> Explanation:
    import torch

    handle = engine.get(model_key)
    model, tokenizer, device = handle.model, handle.tokenizer, handle.device

    enc = tokenizer(
        text,
        truncation=True,
        max_length=engine.max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    inputs = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**inputs)
    # attentions: tuple(layers) each (batch, heads, q, k). Use the last layer,
    # average over heads, then average the attention each token RECEIVES.
    last = out.attentions[-1][0]              # (heads, q, k)
    attn = last.mean(dim=0).mean(dim=0)       # (k,) mean incoming attention
    token_scores = attn.detach().cpu().numpy()

    tokens = _map_token_scores_to_words(text, offsets, token_scores, signed=False)
    return Explanation(
        method="attention",
        method_label="Attention (where it looks)",
        model_key=model_key,
        predicted_label="",
        p_positive=0.0,
        tokens=tokens,
        directional=False,   # attention has no positive/negative direction
        caveat=(
            "Attention shows which tokens the model attends to, NOT why it "
            "decided. High attention is not evidence of sentiment direction "
            "(see Jain & Wallace, 2019). Use Occlusion or IG for causal claims."
        ),
        meta={
            "layer": "last", "aggregation": "mean heads, incoming",
            "description": (
                "Averaged attention weights from the last transformer layer "
                "(mean over heads) showing how much each token is attended to. "
                "This is magnitude-only — it shows focus, not sentiment "
                "direction, and is not a faithful causal explanation."
            ),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Sub-word → word mapping for token-level methods (IG / attention)
# ─────────────────────────────────────────────────────────────────────────────
def _map_token_scores_to_words(
    text: str,
    offsets: List[List[int]],
    token_scores: np.ndarray,
    signed: bool = True,
) -> List[WordScore]:
    words = tokenize_words(text)
    if not words:
        return words
    for (start, end), score in zip(offsets, token_scores):
        if end <= start:            # special token ([CLS]/[SEP]/pad) → skip
            continue
        val = float(score) if signed else abs(float(score))
        # assign this sub-token to the word whose span contains its start char
        for w in words:
            if w.start <= start < w.end:
                w.score += val
                break
    return words
