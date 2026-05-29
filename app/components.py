"""Presentation helpers — every function returns an HTML/SVG string.

Kept free of Streamlit so the markup is easy to reason about and reuse. All
classes either already exist in style.css or are added in the appended section.
Colour comes from CSS variables; numeric→colour is done inline only where a
gradient depends on a runtime value (heatmap cells, gauges).
"""

from __future__ import annotations

import html
import math
from typing import Dict, List, Optional, Sequence, Tuple

from explain import Explanation, WordScore
from reliability import ReliabilityReport


# ─────────────────────────────────────────────────────────────────────────────
#  Token heatmap  —  the explainability centrepiece
# ─────────────────────────────────────────────────────────────────────────────
def heatmap_html(exp: Explanation) -> str:
    """Colour each word by its contribution.

    Directional methods (occlusion/LIME/IG): green = pushes positive,
    red = pushes negative, intensity = magnitude.
    Non-directional (attention): single-hue intensity = how much focus.

    Text colour is driven by a CSS class (hm-pos / hm-neg / hm-focus) so the
    dark theme can override it — we only set the *background* alpha inline.
    """
    pieces: List[str] = []
    prev_end: Optional[int] = None
    raw_text = _reconstruct_source(exp.tokens)

    for tok in exp.tokens:
        if prev_end is not None and tok.start > prev_end:
            pieces.append(raw_text[prev_end:tok.start].replace(" ", "&nbsp;"))
        prev_end = tok.end
        label = html.escape(tok.text)
        if not tok.is_word or abs(tok.norm) < 0.05:
            pieces.append(f'<span class="hm-tok">{label}</span>')
            continue
        if exp.directional:
            sign_cls = "hm-pos" if tok.norm > 0 else "hm-neg"
            alpha = min(0.82, 0.16 + abs(tok.norm) * 0.6)
            rgb = "5,150,105" if tok.norm > 0 else "220,38,38"
            bg = f"background:rgba({rgb},{alpha:.2f});"
            title = f"{tok.pct:+.1f}% of evidence  (Δlogit {tok.score:+.3f})"
        else:
            sign_cls = "hm-focus"
            alpha = min(0.78, 0.12 + abs(tok.norm) * 0.58)
            bg = f"background:rgba(124,58,237,{alpha:.2f});"
            title = f"attention focus {abs(tok.score):.3f}"
        weight = "700" if abs(tok.norm) > 0.55 else "600"
        pieces.append(
            f'<span class="hm-tok hm-active {sign_cls}" '
            f'style="{bg}font-weight:{weight};" title="{title}">{label}</span>'
        )
    legend = _heatmap_legend(exp.directional)
    return f'<div class="heatmap">{"".join(pieces)}</div>{legend}'


def _reconstruct_source(tokens: Sequence[WordScore]) -> str:
    if not tokens:
        return ""
    end = max(t.end for t in tokens)
    chars = [" "] * end
    for t in tokens:
        for i, ch in enumerate(t.text):
            if t.start + i < end:
                chars[t.start + i] = ch
    return "".join(chars)


def _heatmap_legend(directional: bool) -> str:
    if directional:
        return (
            '<div class="hm-legend">'
            '<span class="hm-swatch neg"></span> pushes negative'
            '<span class="hm-scale"></span>'
            '<span class="hm-swatch pos"></span> pushes positive'
            '<span class="hm-hint">· hover a word for its exact contribution</span>'
            "</div>"
        )
    return (
        '<div class="hm-legend">'
        '<span class="hm-swatch focus"></span> attention focus (magnitude only)'
        "</div>"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Sentiment signals  —  top contributing words as signed bars
# ─────────────────────────────────────────────────────────────────────────────
def signals_html(exp: Explanation, k: int = 6) -> str:
    if not exp.directional:
        return ""
    pos = exp.top_positive(k)
    neg = exp.top_negative(k)
    peak = max([abs(t.score) for t in pos + neg] + [1e-9])

    def column(title: str, items, cls: str) -> str:
        if not items:
            rows = '<div class="sig-empty">No strong signals.</div>'
        else:
            rows = "".join(
                f'<div class="sig-row">'
                f'<span class="sig-word" title="{html.escape(t.text)}">{html.escape(t.text)}</span>'
                f'<span class="sig-bar"><span class="sig-fill {cls}" '
                f'style="width:{min(100, abs(t.score) / peak * 100):.0f}%"></span></span>'
                f'<span class="sig-val {cls}" title="Δlogit {t.score:+.3f}">{abs(t.pct):.1f}%</span>'
                f"</div>"
                for t in items
            )
        return f'<div class="sig-col"><div class="sig-title {cls}">{title}</div>{rows}</div>'

    return (
        '<div class="sig-grid">'
        + column("Top positive signals", pos, "pos")
        + column("Top negative signals", neg, "neg")
        + "</div>"
    )


def evidence_balance_html(pos_pull: float, neg_pull: float) -> str:
    total = pos_pull + neg_pull
    if total <= 1e-9:
        return ""
    pos_pct = pos_pull / total * 100
    neg_pct = 100 - pos_pct
    return (
        '<div class="evi-wrap">'
        '<div class="evi-label"><span>Negative evidence</span><span>Positive evidence</span></div>'
        '<div class="evi-bar">'
        f'<span class="evi-neg" style="width:{neg_pct:.0f}%"></span>'
        f'<span class="evi-pos" style="width:{pos_pct:.0f}%"></span>'
        "</div>"
        f'<div class="evi-num"><span>{neg_pct:.0f}%</span><span>{pos_pct:.0f}%</span></div>'
        "</div>"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Semicircular gauge (reliability / confidence)
# ─────────────────────────────────────────────────────────────────────────────
def gauge_svg(value_0_100: float, caption: str, band_hue: Optional[str] = None) -> str:
    frac = max(0.0, min(1.0, value_0_100 / 100.0))
    cx, cy, r = 90, 92, 70
    start = _polar(cx, cy, r, 180)
    end = _polar(cx, cy, r, 180 - 180 * frac)
    track_end = _polar(cx, cy, r, 0)
    if band_hue is None:
        band_hue = (
            "#059669" if value_0_100 >= 75
            else "#d97706" if value_0_100 >= 50 else "#dc2626"
        )
    sweep = (
        f'<path d="M {start[0]:.1f} {start[1]:.1f} A {r} {r} 0 {1 if frac > 0.5 else 0} 1 '
        f'{end[0]:.1f} {end[1]:.1f}" fill="none" stroke="{band_hue}" '
        f'stroke-width="14" stroke-linecap="round" class="gauge-sweep"/>'
    )
    track = (
        f'<path d="M {start[0]:.1f} {start[1]:.1f} A {r} {r} 0 0 1 '
        f'{track_end[0]:.1f} {track_end[1]:.1f}" fill="none" stroke="rgba(0,0,0,0.07)" '
        f'stroke-width="14" stroke-linecap="round"/>'
    )
    return (
        f'<div class="gauge"><svg viewBox="0 0 180 120" width="180" height="120">'
        f"{track}{sweep}"
        f'<text x="90" y="86" text-anchor="middle" class="gauge-num" '
        f'fill="{band_hue}">{value_0_100:.0f}</text>'
        f'</svg><div class="gauge-cap">{html.escape(caption)}</div></div>'
    )


def _polar(cx: float, cy: float, r: float, deg: float) -> Tuple[float, float]:
    rad = math.radians(deg)
    return cx + r * math.cos(rad), cy - r * math.sin(rad)


# ─────────────────────────────────────────────────────────────────────────────
#  Reliability panel
# ─────────────────────────────────────────────────────────────────────────────
def reliability_panel_html(rep: ReliabilityReport) -> str:
    gauge = gauge_svg(rep.score, f"Reliability · {rep.level}")
    pos = "".join(f'<li class="rl-good">{html.escape(x)}</li>' for x in rep.positives)
    risk = "".join(f'<li class="rl-bad">{html.escape(x)}</li>' for x in rep.risks)
    flags = "".join(f'<span class="rl-flag">{html.escape(f)}</span>' for f in rep.flags)
    hr = (
        '<div class="rl-human">⚑ Flagged for human review</div>'
        if rep.needs_human_review else ""
    )
    return (
        '<div class="rl-panel">'
        f'<div class="rl-gauge">{gauge}{hr}</div>'
        '<div class="rl-lists">'
        f'<ul class="rl-ul">{pos}{risk}</ul>'
        f'<div class="rl-flags">{flags}</div>'
        "</div></div>"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Auto-insight callout
# ─────────────────────────────────────────────────────────────────────────────
def insight_html(text: str) -> str:
    return (
        '<div class="auto-insight"><span class="ai-spark">✦</span>'
        f'<span>{html.escape(text)}</span></div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Reasoning pipeline (real steps, not fake spinners)
# ─────────────────────────────────────────────────────────────────────────────
def pipeline_html(steps: List[Dict[str, str]]) -> str:
    """steps: [{icon, title, detail}] — all rendered as completed."""
    items = []
    for i, s in enumerate(steps):
        items.append(
            f'<div class="pl-step" style="animation-delay:{i * 0.07:.2f}s">'
            f'<span class="pl-dot">{s.get("icon", "•")}</span>'
            f'<div class="pl-body"><div class="pl-title">{html.escape(s["title"])}</div>'
            f'<div class="pl-detail">{html.escape(s.get("detail", ""))}</div></div>'
            "</div>"
        )
    return f'<div class="pipeline">{"".join(items)}</div>'


# ─────────────────────────────────────────────────────────────────────────────
#  Model card
# ─────────────────────────────────────────────────────────────────────────────
def model_card_html(
    display_name: str, tagline: str, strengths: Sequence[str],
    weaknesses: Sequence[str], metrics: Dict[str, str], badge: str = "",
) -> str:
    badge_html = f'<span class="mc-badge">{html.escape(badge)}</span>' if badge else ""
    metric_html = "".join(
        f'<div class="mc-metric"><div class="mc-mval">{html.escape(str(v))}</div>'
        f'<div class="mc-mkey">{html.escape(k)}</div></div>'
        for k, v in metrics.items()
    )
    strong = "".join(f"<li>✓ {html.escape(s)}</li>" for s in strengths)
    weak = "".join(f"<li>– {html.escape(w)}</li>" for w in weaknesses)
    return (
        '<div class="model-card">'
        f'<div class="mc-head"><span class="mc-name">{html.escape(display_name)}</span>{badge_html}</div>'
        f'<div class="mc-tag">{html.escape(tagline)}</div>'
        f'<div class="mc-metrics">{metric_html}</div>'
        f'<ul class="mc-strong">{strong}</ul>'
        f'<ul class="mc-weak">{weak}</ul>'
        "</div>"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Arena comparison table + consensus
# ─────────────────────────────────────────────────────────────────────────────
def arena_table_html(rows: List[Dict], consensus: Dict[str, object]) -> str:
    best_conf = max((r["confidence"] for r in rows), default=0)
    fastest_ms = min((r["latency_ms"] for r in rows), default=0)
    body = ""
    for r in rows:
        cls = "positive" if r["prediction"] == "positive" else "negative"
        conf_badge = " ★" if r["confidence"] == best_conf else ""
        fast_badge = " ⚡" if r["latency_ms"] == fastest_ms else ""
        body += (
            "<tr>"
            f'<td><strong>{html.escape(str(r["model"]))}</strong></td>'
            f'<td><span class="arena-pred {cls}">{r["prediction"].title()}</span></td>'
            f'<td class="arena-num">{r["confidence"] * 100:.1f}%{conf_badge}</td>'
            f'<td class="arena-num">{r["latency_ms"]:.0f} ms{fast_badge}</td>'
            "</tr>"
        )
    table = (
        '<table class="model-info-table arena-table">'
        "<thead><tr><th>Model</th><th>Prediction</th><th>Confidence</th><th>Latency</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )

    if not consensus:
        return table
    agree, total = consensus["agreement"]  # type: ignore[index]
    unanimous = consensus["unanimous"]
    warn = ""
    if not unanimous:
        warn = (
            '<div class="arena-warn">⚠ Models disagree — this review may contain '
            "mixed, sarcastic or ambiguous language. Treat the label with caution.</div>"
        )
    chips = (
        '<div class="arena-consensus">'
        f'<span class="ac-chip">Agreement {agree}/{total}</span>'
        f'<span class="ac-chip">Consensus {float(consensus["consensus_confidence"]) * 100:.1f}%</span>'
        f'<span class="ac-chip">Fastest · {html.escape(str(consensus["fastest"]))}</span>'
        f'<span class="ac-chip">Most confident · {html.escape(str(consensus["most_confident"]))}</span>'
        "</div>"
    )
    return table + chips + warn


# ─────────────────────────────────────────────────────────────────────────────
#  Hero stat strip
# ─────────────────────────────────────────────────────────────────────────────
def donut_html(positive: int, negative: int) -> str:
    total = max(1, positive + negative)
    pos_pct = positive / total * 100
    return (
        '<div class="donut-wrap">'
        f'<div class="donut" style="background:conic-gradient('
        f'var(--color-positive) 0% {pos_pct:.1f}%,'
        f'var(--color-negative) {pos_pct:.1f}% 100%);">'
        f'<div class="donut-hole"><span class="donut-pct">{pos_pct:.0f}%</span>'
        '<span class="donut-sub">positive</span></div></div>'
        '<div class="donut-legend">'
        f'<span><i class="dl pos"></i>Positive · {positive}</span>'
        f'<span><i class="dl neg"></i>Negative · {negative}</span>'
        "</div></div>"
    )


def stat_strip_html(stats: List[Tuple[str, str, str]]) -> str:
    """stats: [(value, label, sublabel)]."""
    cards = ""
    for i, (val, label, sub) in enumerate(stats):
        cards += (
            f'<div class="stat-card" style="animation-delay:{i * 0.06:.2f}s">'
            f'<div class="stat-val">{html.escape(val)}</div>'
            f'<div class="stat-label">{html.escape(label)}</div>'
            f'<div class="stat-sub">{html.escape(sub)}</div>'
            "</div>"
        )
    return f'<div class="stat-strip">{cards}</div>'
