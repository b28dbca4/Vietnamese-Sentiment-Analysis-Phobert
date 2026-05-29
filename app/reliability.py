"""Reliability & risk scoring for a single prediction.

Big-tech sentiment systems don't just emit a label — they emit how much you
should *trust* that label. This module turns the raw signals the engine already
produces into an honest 0-100 reliability score plus human-readable reasons.

Design choice: we deliberately do NOT claim "sarcasm detected". We have no
sarcasm classifier, and pretending to have one is exactly the kind of thing an
examiner will puncture. Instead we surface *honest* cues:

    * confidence band              (how close to the 0.5 decision boundary)
    * length adequacy              (too little evidence to judge)
    * truncation                   (the model didn't see the whole review)
    * model agreement              (do the four models concur?)  ← from Arena
    * mixed-signal flag            (strong positive AND negative words present,
                                    via the explanation — framed as ambiguity,
                                    not as a sarcasm verdict)

The score is a transparent weighted penalty model; every deduction is shown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Lightweight cues. These are *flags that lower confidence*, never standalone
# claims. Negation + contrast are the classic failure modes of bag-of-words and
# even transformers on short text.
_NEGATION = re.compile(
    r"\b(not|no|never|n't|hardly|barely|isn'?t|wasn'?t|don'?t|didn'?t|"
    r"can'?t|couldn'?t|won'?t|wouldn'?t)\b",
    re.I,
)
_CONTRAST = re.compile(r"\b(but|however|although|though|yet|except|despite|whereas)\b", re.I)


@dataclass
class ReliabilityReport:
    score: int                       # 0..100
    level: str                       # "High" | "Medium" | "Low"
    positives: List[str] = field(default_factory=list)   # things that help
    risks: List[str] = field(default_factory=list)       # things that hurt
    flags: List[str] = field(default_factory=list)       # neutral cue badges
    needs_human_review: bool = False


def assess(
    *,
    confidence: float,
    text: str,
    num_tokens: int,
    was_truncated: bool,
    max_length: int,
    model_agreement: Optional[Tuple[int, int]] = None,   # (agree, total)
    pos_pull: float = 0.0,
    neg_pull: float = 0.0,
) -> ReliabilityReport:
    score = 100
    positives: List[str] = []
    risks: List[str] = []
    flags: List[str] = []

    # 1) Confidence band ------------------------------------------------------
    if confidence >= 0.90:
        positives.append(f"High model confidence ({confidence * 100:.1f}%)")
    elif confidence >= 0.70:
        score -= 18
        risks.append(f"Moderate confidence ({confidence * 100:.1f}%) — near the boundary")
    else:
        score -= 40
        risks.append(f"Low confidence ({confidence * 100:.1f}%) — close to a coin flip")

    # 2) Length adequacy ------------------------------------------------------
    words = len(text.split())
    if words < 4:
        score -= 30
        risks.append("Very short text — too little sentiment evidence")
    elif words < 12:
        score -= 10
        risks.append("Short text — limited evidence")
    else:
        positives.append("Sufficient text length")

    # 3) Truncation -----------------------------------------------------------
    if was_truncated:
        score -= 15
        risks.append(f"Review truncated to {max_length} tokens — tail unseen")
    else:
        positives.append("Full review fit within the token limit")

    # 4) Model agreement (only if the Arena was run) --------------------------
    if model_agreement is not None:
        agree, total = model_agreement
        if total > 1 and agree == total:
            positives.append(f"All {total} models agree")
        elif total > 1:
            score -= 20
            risks.append(f"Models disagree ({agree}/{total} for the majority)")

    # 5) Mixed-signal / linguistic cues (flags, not verdicts) -----------------
    if _NEGATION.search(text):
        flags.append("Contains negation")
    if _CONTRAST.search(text):
        flags.append("Contains a contrast word (but / however …)")
    if pos_pull > 0 and neg_pull > 0:
        minor = min(pos_pull, neg_pull)
        major = max(pos_pull, neg_pull)
        if minor >= 0.4 * major:
            score -= 12
            flags.append("Strong positive AND negative evidence — mixed sentiment")

    score = max(0, min(100, score))
    level = "High" if score >= 75 else "Medium" if score >= 50 else "Low"
    needs_human = (
        score < 50
        or confidence < 0.70
        or (model_agreement is not None and model_agreement[0] < model_agreement[1])
    )
    return ReliabilityReport(
        score=score,
        level=level,
        positives=positives,
        risks=risks,
        flags=flags,
        needs_human_review=needs_human,
    )
