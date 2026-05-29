"""Batch analytics + automatic insight generation.

Pure functions that turn a list/DataFrame of predictions into the numbers and
sentences a dashboard needs. All insights are rule-based and derived from the
actual predictions — no fabricated metrics.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Frequent but uninformative tokens we don't want surfacing as "key terms".
_STOP = set(
    """a an and the of to in is it this that was for with as on but be have not
    you i he she they we are were his her its their my your me him them so if or
    at by from movie film one all just very really would could about than then
    out up down who what which when more most some any been being had has do does
    did get got make made see saw watch watched""".split()
)
_TOKEN_RE = re.compile(r"[A-Za-z']{3,}")


@dataclass
class BatchSummary:
    total: int
    positive: int
    negative: int
    positive_ratio: float
    avg_confidence: float
    avg_tokens: float
    truncated: int
    low_confidence: int
    top_positive_terms: List[Tuple[str, int]] = field(default_factory=list)
    top_negative_terms: List[Tuple[str, int]] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)


def summarise(df: pd.DataFrame, low_conf_threshold: float = 0.70) -> BatchSummary:
    """``df`` must have: review, label, confidence, num_tokens, was_truncated."""
    total = len(df)
    if total == 0:
        return BatchSummary(0, 0, 0, 0.0, 0.0, 0.0, 0, 0)

    positive = int((df["label"] == "positive").sum())
    negative = total - positive
    pos_ratio = positive / total
    avg_conf = float(df["confidence"].mean())
    avg_tok = float(df["num_tokens"].mean()) if "num_tokens" in df else 0.0
    truncated = int(df["was_truncated"].sum()) if "was_truncated" in df else 0
    low_conf = int((df["confidence"] < low_conf_threshold).sum())

    top_pos = _top_terms(df.loc[df["label"] == "positive", "review"])
    top_neg = _top_terms(df.loc[df["label"] == "negative", "review"])

    insights = _build_insights(
        total, positive, negative, pos_ratio, avg_conf, low_conf,
        truncated, top_pos, top_neg, low_conf_threshold,
    )
    return BatchSummary(
        total=total,
        positive=positive,
        negative=negative,
        positive_ratio=pos_ratio,
        avg_confidence=avg_conf,
        avg_tokens=avg_tok,
        truncated=truncated,
        low_confidence=low_conf,
        top_positive_terms=top_pos,
        top_negative_terms=top_neg,
        insights=insights,
    )


def _top_terms(reviews: pd.Series, k: int = 6) -> List[Tuple[str, int]]:
    counter: Counter = Counter()
    for text in reviews.astype(str):
        for tok in _TOKEN_RE.findall(text.lower()):
            if tok not in _STOP:
                counter[tok] += 1
    return counter.most_common(k)


def _build_insights(
    total, positive, negative, pos_ratio, avg_conf, low_conf, truncated,
    top_pos, top_neg, low_conf_threshold,
) -> List[str]:
    out: List[str] = []
    out.append(
        f"{pos_ratio * 100:.0f}% of {total} reviews are positive "
        f"({positive} positive / {negative} negative)."
    )
    if top_pos:
        terms = ", ".join(w for w, _ in top_pos[:4])
        out.append(f"Positive reviews frequently mention: {terms}.")
    if top_neg:
        terms = ", ".join(w for w, _ in top_neg[:4])
        out.append(f"Negative reviews frequently mention: {terms}.")
    if low_conf:
        pct = low_conf / total * 100
        out.append(
            f"{pct:.0f}% of predictions are below {low_conf_threshold * 100:.0f}% "
            f"confidence ({low_conf} reviews) and should be routed to human review."
        )
    if truncated:
        out.append(
            f"{truncated} review(s) exceeded the token limit and were truncated — "
            "their endings were not seen by the model."
        )
    out.append(f"Average confidence across the batch is {avg_conf * 100:.1f}%.")
    return out


def auto_insight(label: str, p_positive: float, confidence: float,
                 pos_pull: float = 0.0, neg_pull: float = 0.0) -> str:
    """One-sentence narrative for a single prediction."""
    if confidence < 0.65:
        return ("The model is uncertain about this review — it likely contains "
                "mixed signals or ambiguous language.")
    if pos_pull > 0 and neg_pull > 0 and min(pos_pull, neg_pull) >= 0.4 * max(pos_pull, neg_pull):
        return ("Both positive and negative cues are present; the final label "
                "reflects which side the model weighed more heavily.")
    if label == "positive":
        return ("Clear positive sentiment, driven mainly by approving, "
                "enthusiastic language.")
    return ("Clear negative sentiment, driven mainly by criticism of plot, "
            "pacing, acting or overall quality.")


def arena_consensus(rows: List[Dict]) -> Dict[str, object]:
    """Aggregate Arena predictions into consensus statistics."""
    if not rows:
        return {}
    labels = [r["prediction"] for r in rows]
    majority = Counter(labels).most_common(1)[0]
    agree = majority[1]
    total = len(rows)
    fastest = min(rows, key=lambda r: r["latency_ms"])
    most_conf = max(rows, key=lambda r: r["confidence"])
    consensus_conf = float(np.mean([r["confidence"] for r in rows]))
    return {
        "majority_label": majority[0],
        "agreement": (agree, total),
        "unanimous": agree == total,
        "fastest": fastest["model"],
        "fastest_ms": fastest["latency_ms"],
        "most_confident": most_conf["model"],
        "most_confident_pct": most_conf["confidence"],
        "consensus_confidence": consensus_conf,
    }
