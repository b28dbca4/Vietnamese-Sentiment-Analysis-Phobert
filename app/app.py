"""IMDb Sentiment Intelligence Platform — Streamlit front-end.

Pages
-----
  Analyze          single review → result + reliability + inline explanation
  Model Arena      run 4 models on one review → consensus + disagreement
  Explainability   deep dive: pick model + method, heatmap, signals, compare
  Batch Intel      upload/paste many reviews → analytics dashboard + insights
  Performance      saved metrics + figures for the active model
  Comparison       cross-model table + figures + model cards
  Error Analysis   FP / FN examples mined from the test split
  About            problem, pipeline diagram, methodology

Run from the repo root:  streamlit run app/app.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
for p in (str(ROOT_DIR), str(APP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src import config  # noqa: E402

import components as C          # noqa: E402
import explain as X            # noqa: E402
import insights as I           # noqa: E402
import reliability as R        # noqa: E402
from engine import (           # noqa: E402
    ARENA_ORDER, MODEL_SPECS, SPEC_BY_KEY, InferenceEngine, get_engine,
)

st.set_page_config(page_title="Sentiment Intelligence Platform", layout="wide",
                   initial_sidebar_state="expanded")


# ─────────────────────────────────────────────────────────────────────────────
#  Resources & styling
# ─────────────────────────────────────────────────────────────────────────────
def load_css() -> None:
    css = (APP_DIR / "assets" / "style.css")
    if css.exists():
        st.markdown(f"<style>{css.read_text(encoding='utf-8')}</style>",
                    unsafe_allow_html=True)
    if st.session_state.get("lab_theme"):
        st.markdown(f"<style>{_DARK_OVERRIDES}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Booting inference engine…")
def boot_engine() -> InferenceEngine:
    eng = get_engine()
    # warm the default model so the first prediction isn't cold
    try:
        eng.get(eng.default_model())
    except Exception:
        pass
    return eng


def safe_engine() -> tuple[Optional[InferenceEngine], Optional[str]]:
    try:
        return boot_engine(), None
    except Exception as exc:  # transformers/torch/model missing, etc.
        return None, str(exc)


def read_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="section-heading"><h1>{title}</h1>'
        f'<div class="section-divider"></div><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Examples (expanded set covering the failure modes worth demoing)
# ─────────────────────────────────────────────────────────────────────────────
EXAMPLES = [
    ("Strong positive", "positive",
     "This movie was an absolute masterpiece. The acting was superb, the "
     "cinematography breathtaking, and the story kept me engaged throughout."),
    ("Strong negative", "negative",
     "I cannot believe I wasted two hours on this. The plot was nonsensical, "
     "the dialogue awkward, and the ending made no sense at all."),
    ("Sarcastic", "negative",
     "Best movie ever, if your goal is to fall asleep within ten minutes. "
     "Truly a revolutionary cure for insomnia."),
    ("Negation flip", "positive",
     "I thought it would be terrible, but it was actually amazing and "
     "surprisingly moving by the end."),
    ("Mixed", "negative",
     "The first half had strong performances and a promising setup, but the "
     "second half collapsed into rushed twists and a disappointing ending."),
    ("Ambiguous / short", "negative", "It was fine. Nothing special."),
]


def push_history(model_key: str, pred, reliability_level: str) -> None:
    st.session_state.setdefault("history", [])
    st.session_state.history.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "model": SPEC_BY_KEY[model_key].display_name,
        "label": pred.label,
        "confidence": pred.confidence,
        "latency_ms": pred.latency_ms,
        "reliability": reliability_level,
        "text": pred.meta.get("text", ""),
    })
    st.session_state.history = st.session_state.history[:25]


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE · Analyze
# ─────────────────────────────────────────────────────────────────────────────
def page_analyze(engine: Optional[InferenceEngine], err: Optional[str]) -> None:
    section_header(
        "Sentiment Intelligence Platform",
        "Enterprise-grade movie-review analytics powered by RoBERTa, BERT, "
        "DistilBERT and Logistic Regression — with explainable, reliability-aware predictions.",
    )

    # hero stats
    active = engine.default_model() if engine else "—"
    active_name = SPEC_BY_KEY[active].display_name if engine else "Offline"
    device = "—"
    if engine:
        try:
            device = engine.model_info(active).get("device", "cpu")
        except Exception:
            device = "cpu"
    st.markdown(C.stat_strip_html([
        ("50K", "Reviews trained", "IMDb corpus"),
        ("4", "Models", "LR · DistilBERT · BERT · RoBERTa"),
        (active_name, "Active model", "explainable"),
        (str(getattr(config, "MAX_LENGTH", 256)), "Token limit", "per review"),
        (str(device).upper(), "Device", "local inference"),
    ]), unsafe_allow_html=True)

    st.session_state.setdefault("review_text", "")
    review = st.text_area("Review text", height=180, max_chars=10_000,
                          key="review_text",
                          placeholder="Enter a movie review to analyze…")
    st.caption(f"{len(review)} / 10,000 characters")

    c1, c2 = st.columns([4, 1])
    run = c1.button("Analyze Sentiment", type="primary", use_container_width=True)
    explain_too = c2.toggle("Explain", value=True,
                            help="Run leave-one-out explanation with the prediction")

    if run:
        if engine is None:
            st.error(f"Model is not ready: {err}")
        elif not review.strip():
            st.warning("Enter a review before analyzing.")
        else:
            _run_single(engine, review, explain_too)

    _render_examples()
    _render_history()


def _run_single(engine: InferenceEngine, review: str, do_explain: bool) -> None:
    key = engine.default_model()
    with st.spinner("Running inference…"):
        pred = engine.predict(review, key)
        pred.meta["text"] = review[:300]

    exp = None
    pos_pull = neg_pull = 0.0
    if do_explain:
        with st.spinner("Explaining (leave-one-out probing)…"):
            try:
                exp = X.explain(engine, review, key, method="occlusion", max_words=100)
                pos_pull, neg_pull = exp.evidence_balance()
            except Exception as exc:
                st.info(f"Explanation unavailable: {exc}")

    rep = R.assess(
        confidence=pred.confidence, text=review, num_tokens=pred.num_tokens,
        was_truncated=pred.was_truncated, max_length=engine.max_length,
        pos_pull=pos_pull, neg_pull=neg_pull,
    )
    push_history(key, pred, rep.level)

    # reasoning trace (real values)
    st.markdown(C.pipeline_html([
        {"icon": "①", "title": "Input received", "detail": f"{len(review)} characters"},
        {"icon": "②", "title": "Tokenization", "detail": f"{pred.num_tokens} tokens · limit {engine.max_length}"},
        {"icon": "③", "title": "Transformer inference", "detail": f"{SPEC_BY_KEY[key].display_name} · {pred.latency_ms:.0f} ms"},
        {"icon": "④", "title": "Classification", "detail": f"P(positive)={pred.p_positive:.3f}"},
        {"icon": "⑤", "title": "Reliability check", "detail": f"{rep.level} · {rep.score}/100"},
    ]), unsafe_allow_html=True)

    _render_result_card(pred, engine.max_length)
    st.markdown(C.insight_html(
        I.auto_insight(pred.label, pred.p_positive, pred.confidence, pos_pull, neg_pull)
    ), unsafe_allow_html=True)

    colL, colR = st.columns([1, 1])
    with colL:
        st.markdown("##### Prediction reliability")
        st.markdown(C.reliability_panel_html(rep), unsafe_allow_html=True)
    with colR:
        st.markdown("##### Token usage")
        _render_token_usage(pred, engine.max_length)

    if exp is not None:
        with st.expander("🔍 Why did the model predict this?  ·  Token Contribution Map",
                         expanded=True):
            st.markdown(C.heatmap_html(exp), unsafe_allow_html=True)
            st.markdown(C.evidence_balance_html(pos_pull, neg_pull), unsafe_allow_html=True)
            st.markdown(C.signals_html(exp), unsafe_allow_html=True)
            st.caption(f"Method: {exp.method_label}. {exp.caveat or exp.meta.get('description','')}")
            st.caption("Open the Explainability page to compare LIME / Integrated "
                       "Gradients / Attention on this review.")

    summary = (f"{pred.label.upper()} ({pred.confidence*100:.1f}%) · "
               f"reliability {rep.level} {rep.score}/100 · {pred.latency_ms:.0f} ms")
    st.code(summary, language=None)


def _render_result_card(pred, max_length: int) -> None:
    cls = "positive" if pred.label == "positive" else "negative"
    glow = f"{cls}-glow"
    icon = "✦" if pred.label == "positive" else "✕"
    trunc = "Truncated" if pred.was_truncated else "Full text"
    st.markdown(f"""
        <div class="result-card {glow}">
          <div class="result-topline">
            <span class="sentiment-badge {cls}">{icon} {pred.label.upper()}</span>
            <div class="confidence-value">Confidence {pred.confidence*100:.1f}%</div>
          </div>
          <div class="confidence-track">
            <div class="confidence-fill {cls}" style="width:{pred.confidence*100:.1f}%"></div>
          </div>
          <div class="probability-block">
            <div class="prob-row"><span class="prob-label">Positive</span>
              <span class="prob-track"><span class="prob-fill positive" style="width:{pred.p_positive*100:.1f}%"></span></span>
              <span class="prob-value">{pred.p_positive*100:.1f}%</span></div>
            <div class="prob-row"><span class="prob-label">Negative</span>
              <span class="prob-track"><span class="prob-fill negative" style="width:{pred.p_negative*100:.1f}%"></span></span>
              <span class="prob-value">{pred.p_negative*100:.1f}%</span></div>
          </div>
          <div class="chip-row">
            <span class="info-chip">{pred.num_tokens} tokens</span>
            <span class="info-chip">{trunc}</span>
            <span class="info-chip">{pred.latency_ms:.0f} ms</span>
          </div>
        </div>
    """, unsafe_allow_html=True)


def _render_token_usage(pred, max_length: int) -> None:
    used = min(pred.num_tokens, max_length)
    frac = used / max_length * 100
    color = ("#dc2626" if pred.was_truncated else
             "#d97706" if frac > 70 else "#059669")
    st.markdown(f"""
        <div class="token-usage">
          <div class="tu-head"><span>Token usage</span>
            <span class="tu-num">{used} / {max_length}</span></div>
          <div class="tu-track"><div class="tu-fill" style="width:{min(100,frac):.0f}%;background:{color}"></div></div>
          <div class="tu-note">{"⚠ Truncated — tail unseen" if pred.was_truncated else f"{frac:.0f}% of capacity used"}</div>
        </div>
    """, unsafe_allow_html=True)


def _render_examples() -> None:
    with st.expander("Try example reviews (positive · negative · sarcasm · negation · mixed)",
                     expanded=True):
        cols = st.columns(3)
        for i, (title, expected, text) in enumerate(EXAMPLES):
            with cols[i % 3]:
                st.markdown(f"""
                    <div class="example-card {expected}">
                      <strong>{title}</strong><p>{text[:120]}…</p></div>
                """, unsafe_allow_html=True)
                st.button("Use", key=f"ex_{i}", use_container_width=True,
                          on_click=lambda t=text: st.session_state.update(review_text=t))


def _render_history() -> None:
    hist = st.session_state.get("history", [])
    if not hist:
        return
    with st.expander(f"Recent predictions ({len(hist)})", expanded=False):
        df = pd.DataFrame(hist)[
            ["time", "model", "label", "confidence", "reliability", "latency_ms", "text"]
        ]
        st.dataframe(df, use_container_width=True, hide_index=True)
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE · Model Arena
# ─────────────────────────────────────────────────────────────────────────────
def page_arena(engine: Optional[InferenceEngine], err: Optional[str]) -> None:
    section_header("Model Arena",
                   "Run every model on the same review and compare verdicts, "
                   "confidence and latency — then measure their agreement.")
    if engine is None:
        st.error(f"Model is not ready: {err}")
        return

    st.session_state.setdefault("arena_text", "")
    text = st.text_area("Review to send to all models", height=140, key="arena_text",
                        placeholder="Enter a movie review to run on all models…")
    if st.button("Run Arena", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Enter a review.")
            return
        keys = [k for k in ARENA_ORDER if k in engine.available_models()]
        rows: List[Dict] = []
        bar = st.progress(0.0, text="Running models…")
        for n, k in enumerate(keys, 1):
            with st.spinner(f"{SPEC_BY_KEY[k].display_name}…"):
                rows.append(engine.predict(text, k).as_row())
            bar.progress(n / len(keys), text=f"{SPEC_BY_KEY[k].display_name} done")
        bar.empty()
        consensus = I.arena_consensus(rows)
        st.markdown(C.arena_table_html(rows, consensus), unsafe_allow_html=True)

        # reliability informed by agreement
        agree, total = consensus["agreement"]
        ref = max(rows, key=lambda r: r["confidence"])
        rep = R.assess(confidence=ref["confidence"], text=text,
                       num_tokens=len(text.split()), was_truncated=False,
                       max_length=engine.max_length, model_agreement=(agree, total))
        st.markdown("##### Consensus reliability")
        st.markdown(C.reliability_panel_html(rep), unsafe_allow_html=True)
        if rep.needs_human_review:
            st.warning("This case is flagged for human review.")


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE · Explainability (deep dive)
# ─────────────────────────────────────────────────────────────────────────────
def page_explain(engine: Optional[InferenceEngine], err: Optional[str]) -> None:
    section_header("Explainability Lab",
                   "Token-level evidence behind a prediction — four complementary "
                   "methods on one consistent green-for-positive / red-for-negative scale.")
    if engine is None:
        st.error(f"Model is not ready: {err}")
        return

    models = engine.available_models()
    c1, c2 = st.columns(2)
    model_key = c1.selectbox("Model", models,
                             format_func=lambda k: SPEC_BY_KEY[k].display_name)
    methods = X.available_methods(model_key)
    method = c2.selectbox("Method", [m[0] for m in methods],
                          format_func=lambda m: dict(methods)[m])

    # Tell the user which optional methods are missing + how to enable them.
    missing = []
    if not X._has_module("lime"):
        missing.append(("LIME", "pip install lime"))
    if SPEC_BY_KEY[model_key].kind == "transformer" and not X._has_module("captum"):
        missing.append(("Integrated Gradients", "pip install captum"))
    if missing:
        items = " · ".join(f"<b>{n}</b> (<code>{c}</code>)" for n, c in missing)
        st.markdown(
            f'<div class="xai-missing">Optional methods not installed: {items}. '
            "Occlusion and Attention work without them; install to unlock the rest."
            "</div>", unsafe_allow_html=True)

    st.session_state.setdefault("xai_text", "")
    text = st.text_area("Review", height=140, key="xai_text",
                        placeholder="Enter a movie review to explain…")

    with st.expander("How each method works (and how much to trust it)", expanded=False):
        st.markdown(_METHOD_NOTES, unsafe_allow_html=True)

    cc1, cc2 = st.columns([2, 1])
    go = cc1.button("Explain", type="primary", use_container_width=True)
    compare = cc2.button("Compare all methods", use_container_width=True)

    if go and text.strip():
        _show_explanation(engine, text, model_key, method)
    if compare and text.strip():
        st.markdown("#### Method comparison")
        st.caption("Same review, every available method. Compare where they agree "
                   "(robust evidence) and where they differ (method-specific artefacts).")
        for m_id, m_label in methods:
            st.markdown(f'<div class="xai-method-block"><strong>{m_label}</strong></div>',
                        unsafe_allow_html=True)
            with st.spinner(f"{m_label}…"):
                try:
                    exp = X.explain(engine, text, model_key, method=m_id)
                    st.markdown(C.heatmap_html(exp), unsafe_allow_html=True)
                    if exp.directional:
                        st.markdown(C.signals_html(exp, k=4), unsafe_allow_html=True)
                except ImportError:
                    st.markdown(
                        f'<div class="xai-missing">{m_label} needs an extra library — '
                        "see the notice above.</div>", unsafe_allow_html=True)
                except Exception as exc:
                    st.caption(f"{m_label}: unavailable ({exc})")


def _show_explanation(engine, text, model_key, method) -> None:
    with st.spinner("Computing attributions…"):
        try:
            exp = X.explain(engine, text, model_key, method=method)
        except ImportError as exc:
            st.error(f"This method needs an extra library: {exc}. "
                     "Install it (see the notice above) or pick another method.")
            return
        except Exception as exc:
            st.error(f"Explanation failed: {exc}")
            return

    pred = "POSITIVE" if exp.p_positive >= 0.5 else "NEGATIVE"
    conf = max(exp.p_positive, 1 - exp.p_positive)
    st.markdown(f"Prediction: **{pred}** · P(positive) = {exp.p_positive:.3f} "
                f"· confidence {conf*100:.1f}%")
    st.markdown(C.heatmap_html(exp), unsafe_allow_html=True)
    if exp.directional:
        pos, neg = exp.evidence_balance()
        st.markdown(C.evidence_balance_html(pos, neg), unsafe_allow_html=True)
        st.markdown(C.signals_html(exp), unsafe_allow_html=True)
    # Always explain the mechanism so the result is defensible, not magic.
    desc = exp.meta.get("description", "")
    if desc:
        st.markdown(f'<div class="auto-insight"><span class="ai-spark">✦</span>'
                    f'<span><b>How this was computed:</b> {desc}</span></div>',
                    unsafe_allow_html=True)
    if exp.caveat:
        st.info(exp.caveat)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE · Batch Intelligence
# ─────────────────────────────────────────────────────────────────────────────
def page_batch(engine: Optional[InferenceEngine], err: Optional[str]) -> None:
    section_header("Batch Intelligence",
                   "Score many reviews at once and turn them into an analytics "
                   "dashboard with automatic insights.")
    if engine is None:
        st.error(f"Model is not ready: {err}")
        return

    up = st.file_uploader("CSV with a 'review' column", type=["csv"])
    pasted = st.text_area("…or paste one review per line", height=120)

    if st.button("Run batch analysis", type="primary", use_container_width=True):
        reviews: List[str] = []
        if up is not None:
            d = pd.read_csv(up)
            col = "review" if "review" in d.columns else d.columns[0]
            reviews += d[col].dropna().astype(str).tolist()
        if pasted.strip():
            reviews += [l.strip() for l in pasted.splitlines() if l.strip()]
        if not reviews:
            st.warning("No reviews found.")
            return

        key = engine.default_model()
        rows = []
        bar = st.progress(0.0, text="Scoring…")
        for n, rv in enumerate(reviews, 1):
            p = engine.predict(rv, key)
            rows.append({"review": rv, "label": p.label, "confidence": p.confidence,
                         "p_positive": p.p_positive, "num_tokens": p.num_tokens,
                         "was_truncated": p.was_truncated})
            if n % 5 == 0 or n == len(reviews):
                bar.progress(n / len(reviews), text=f"{n}/{len(reviews)}")
        bar.empty()
        st.session_state.batch_df = pd.DataFrame(rows)

    df = st.session_state.get("batch_df")
    if df is None or df.empty:
        return

    summ = I.summarise(df)
    m = st.columns(4)
    m[0].metric("Reviews", summ.total)
    m[1].metric("Positive", f"{summ.positive} ({summ.positive_ratio*100:.0f}%)")
    m[2].metric("Avg confidence", f"{summ.avg_confidence*100:.1f}%")
    m[3].metric("Low confidence", summ.low_confidence)

    cL, cR = st.columns([1, 1])
    with cL:
        st.markdown(C.donut_html(summ.positive, summ.negative), unsafe_allow_html=True)
    with cR:
        st.markdown("###### Confidence distribution")
        hist, edges = np.histogram(df["confidence"], bins=10, range=(0, 1))
        st.bar_chart(pd.DataFrame({"count": hist},
                     index=[f"{e:.1f}" for e in edges[:-1]]))

    st.markdown("###### Key insights")
    for line in summ.insights:
        st.markdown(C.insight_html(line), unsafe_allow_html=True)

    st.markdown("###### Explore predictions")
    f1, f2, f3 = st.columns(3)
    label_filter = f1.selectbox("Sentiment", ["All", "positive", "negative"])
    only_low = f2.checkbox("Low confidence (<70%)")
    query = f3.text_input("Search text")
    view = df.copy()
    if label_filter != "All":
        view = view[view["label"] == label_filter]
    if only_low:
        view = view[view["confidence"] < 0.70]
    if query.strip():
        view = view[view["review"].str.contains(query, case=False, na=False)]
    st.dataframe(view, use_container_width=True, hide_index=True)

    d1, d2 = st.columns(2)
    d1.download_button("Download CSV", df.to_csv(index=False).encode(),
                       "sentiment_batch.csv", "text/csv", use_container_width=True)
    d2.download_button("Download JSON",
                       df.to_json(orient="records", indent=2).encode(),
                       "sentiment_batch.json", "application/json",
                       use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE · Performance
# ─────────────────────────────────────────────────────────────────────────────
def page_performance() -> None:
    section_header("Model Performance Dashboard",
                   "Metrics and saved figures from the trained models.")
    metrics = (read_json(os.path.join(config.METRICS_DIR, "roberta_results.json"))
               or read_json(os.path.join(config.METRICS_DIR, "distilbert_results.json")))
    c = st.columns(4)
    c[0].metric("Accuracy", _pct(metrics.get("accuracy")))
    c[1].metric("F1 Macro", _pct(metrics.get("f1_macro")))
    c[2].metric("AUC", "N/A" if metrics.get("roc_auc") is None
                else f"{metrics['roc_auc']:.4f}")
    c[3].metric("Max Length", str(getattr(config, "MAX_LENGTH", 256)))

    tabs = st.tabs(["Confusion Matrix", "ROC Curve", "Training", "Calibration", "Confidence"])
    figs = ["confusion_matrix_roberta.png", "roc_curves_comparison.png",
            "training_curves_roberta.png", "calibration_curve_roberta.png",
            "confidence_distribution_roberta.png"]
    for tab, fn in zip(tabs, figs):
        with tab:
            _show_figure(fn)

    rep = metrics.get("classification_report", {})
    if rep:
        rows = [{"class": k, **rep[k]} for k in
                ["negative", "positive", "macro avg", "weighted avg"] if k in rep]
        st.subheader("Classification report")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE · Comparison
# ─────────────────────────────────────────────────────────────────────────────
def page_comparison() -> None:
    section_header("Model Comparison",
                   "Cross-model metrics, figures and capability cards.")
    summary = read_json(os.path.join(config.METRICS_DIR, "comparison_summary.json"))
    if summary:
        _render_summary_table(summary)

    st.markdown("#### Model cards")
    cols = st.columns(2)
    for i, spec in enumerate(MODEL_SPECS):
        metrics = _metrics_for(spec.key, summary)
        with cols[i % 2]:
            badge = ("Best accuracy" if spec.key == "roberta"
                     else "Fastest" if spec.key == "baseline"
                     else "Balanced" if spec.key == "distilbert" else "")
            st.markdown(C.model_card_html(spec.display_name, spec.tagline,
                        spec.strengths, spec.weaknesses, metrics, badge),
                        unsafe_allow_html=True)

    tabs = st.tabs(["Bar", "Radar", "ROC overlay", "Efficiency"])
    for tab, fn in zip(tabs, ["model_comparison_bar.png", "model_comparison_radar.png",
                              "roc_curves_comparison.png", "efficiency_comparison.png"]):
        with tab:
            _show_figure(fn)


def _metrics_for(key: str, summary: Dict) -> Dict[str, str]:
    name_map = {"roberta": "roberta", "bert": "bert", "distilbert": "distilbert",
                "baseline": "baseline"}
    out = {"Accuracy": "—", "F1": "—"}
    for mk, mv in (summary or {}).items():
        if name_map.get(key, key) in mk.lower():
            if "accuracy" in mv:
                out["Accuracy"] = f"{float(mv['accuracy'])*100:.1f}%"
            if "f1_macro" in mv:
                out["F1"] = f"{float(mv['f1_macro'])*100:.1f}%"
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE · Error Analysis
# ─────────────────────────────────────────────────────────────────────────────
def page_errors(engine: Optional[InferenceEngine], err: Optional[str]) -> None:
    section_header("Error Analysis",
                   "Where does the model struggle? Mine false positives and "
                   "false negatives from the held-out test split.")
    if engine is None:
        st.error(f"Model is not ready: {err}")
        return
    test_path = os.path.join(getattr(config, "PROCESSED_DIR",
                             ROOT_DIR / "data" / "processed"), "test.csv")
    if not os.path.exists(test_path):
        st.info("test.csv not found in data/processed/.")
        return

    n = st.slider("Sample size (kept small for speed)", 50, 500, 150, 50)
    if st.button("Run error analysis", type="primary"):
        df = pd.read_csv(test_path)
        rcol = "review" if "review" in df.columns else df.columns[0]
        lcol = ("sentiment" if "sentiment" in df.columns
                else "label" if "label" in df.columns else df.columns[-1])
        df = df.sample(min(n, len(df)), random_state=42).reset_index(drop=True)
        key = engine.default_model()
        recs = []
        bar = st.progress(0.0)
        for i, row in df.iterrows():
            p = engine.predict(str(row[rcol]), key)
            truth = str(row[lcol]).lower()
            truth = "positive" if "pos" in truth or truth == "1" else "negative"
            recs.append({"review": str(row[rcol]), "truth": truth,
                         "pred": p.label, "confidence": p.confidence,
                         "tokens": p.num_tokens})
            bar.progress((i + 1) / len(df))
        bar.empty()
        res = pd.DataFrame(recs)
        st.session_state.err_df = res

    res = st.session_state.get("err_df")
    if res is None:
        return
    acc = (res["pred"] == res["truth"]).mean()
    fp = res[(res["truth"] == "negative") & (res["pred"] == "positive")]
    fn = res[(res["truth"] == "positive") & (res["pred"] == "negative")]
    m = st.columns(3)
    m[0].metric("Sample accuracy", f"{acc*100:.1f}%")
    m[1].metric("False positives", len(fp))
    m[2].metric("False negatives", len(fn))

    st.markdown(C.insight_html(
        f"Errors skew toward {'false positives' if len(fp) > len(fn) else 'false negatives'}. "
        f"Misclassified reviews average {res[res.pred != res.truth]['tokens'].mean():.0f} tokens "
        f"vs {res[res.pred == res.truth]['tokens'].mean():.0f} for correct ones."
    ), unsafe_allow_html=True)

    t1, t2 = st.tabs([f"False positives ({len(fp)})", f"False negatives ({len(fn)})"])
    with t1:
        st.dataframe(fp.sort_values("confidence", ascending=False),
                     use_container_width=True, hide_index=True)
    with t2:
        st.dataframe(fn.sort_values("confidence", ascending=False),
                     use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE · About
# ─────────────────────────────────────────────────────────────────────────────
def page_about() -> None:
    section_header("About This Project",
                   "A defensible, explainable IMDb sentiment system — from raw "
                   "reviews to fine-tuned Transformers, with the reasoning made visible.")

    # 1) Problem & approach
    st.markdown("""
        <div class="result-card stagger-1">
          <h3 class="about-h">📋 Problem & approach</h3>
          <p>This project tackles <b>binary sentiment classification</b> on the
          IMDb 50K movie-review corpus (≈49.6K reviews after de-duplication,
          split 80/10/10). A classical <b>TF-IDF + Logistic-Regression</b>
          baseline establishes a performance floor, then three Transformers —
          <b>DistilBERT</b>, <b>BERT-base</b> and <b>RoBERTa</b> — are fine-tuned
          and compared. On top of the predictions, this platform adds three
          things a raw classifier lacks: <b>explainability</b> (why this label?),
          <b>reliability scoring</b> (how much to trust it?) and
          <b>multi-model consensus</b> (do independent models agree?).</p>
        </div>
    """, unsafe_allow_html=True)

    # 2) Training pipeline
    st.markdown(f"""
        <div class="result-card stagger-2">
          <h3 class="about-h">🔧 Training pipeline</h3>
          <p>From raw text to a deployable checkpoint:</p>
          <div class="flow">
            {''.join(f'<span class="flow-node">{s}</span>' for s in
              ['Raw IMDb 50K','Clean (HTML/URL/space)','Split 80/10/10',
               'TF-IDF baseline','Fine-tune 3 Transformers','Evaluate & compare',
               'Select best_model','Deploy'])}
          </div>
          <p style="margin-top:0.9rem;font-size:0.88rem;color:var(--color-text-secondary)">
          Transformer config: <code>max_length=256</code>, <code>batch_size=16</code>,
          <code>epochs=3</code>, <code>lr=2e-5</code>, <code>warmup_ratio=0.1</code>,
          <code>weight_decay=0.01</code>. Text is <b>not</b> lowercased / stemmed —
          it is left raw so the Transformer tokenizers see natural input.</p>
        </div>
    """, unsafe_allow_html=True)

    # 3) Runtime architecture
    st.markdown(f"""
        <div class="result-card stagger-3">
          <h3 class="about-h">🧠 Runtime architecture (what happens per request)</h3>
          <div class="flow">
            {''.join(f'<span class="flow-node">{s}</span>' for s in
              ['Review','Tokenizer','Transformer','Softmax + logits','Label & confidence',
               'Occlusion / LIME / IG','Reliability score','Dashboard'])}
          </div>
          <p style="margin-top:0.9rem;font-size:0.88rem;color:var(--color-text-secondary)">
          A single multi-model <code>engine</code> lazy-loads each checkpoint on
          first use and caches it, exposing both <code>predict_proba</code> (for
          the verdict) and an unsaturated <code>logit-margin</code> score (for
          faithful occlusion).</p>
        </div>
    """, unsafe_allow_html=True)

    # 4) The four models
    st.markdown('<div class="section-heading" style="margin-top:1.5rem;"><h3 class="about-h">🤖 The model line-up</h3></div>',
                unsafe_allow_html=True)
    cols = st.columns(2)
    for i, spec in enumerate(MODEL_SPECS):
        with cols[i % 2]:
            st.markdown(C.model_card_html(
                spec.display_name, spec.tagline, spec.strengths, spec.weaknesses,
                {"Type": "Transformer" if spec.kind == "transformer" else "Classical ML"},
                badge=("Champion" if spec.key == "roberta" else
                       "Baseline" if spec.key == "baseline" else "")),
                unsafe_allow_html=True)

    # 5) Explainability philosophy
    st.markdown(f"""
        <div class="result-card stagger-4">
          <h3 class="about-h">🔍 Explainability — and how much to trust each method</h3>
          <p>Four methods are offered, deliberately spanning the
          faithful↔approximate spectrum, all on one green-for-positive /
          red-for-negative scale:</p>
          {_METHOD_NOTES}
          <p style="margin-top:0.6rem;"><b>Honesty by design:</b> word
          contributions are <i>computed</i>, never hard-coded; occlusion scores
          the logit margin to avoid probability saturation; and the system flags
          mixed-signal / low-confidence cases for human review instead of
          pretending to "detect sarcasm".</p>
        </div>
    """, unsafe_allow_html=True)

    # 6) Expected metrics
    st.markdown(f"""
        <div class="result-card stagger-5">
          <h3 class="about-h">🎯 Target performance</h3>
          <table class="model-info-table">
            <thead><tr><th>Metric</th><th>Expected range</th></tr></thead>
            <tbody>
              <tr><td><strong>Accuracy</strong></td><td>92–94%</td></tr>
              <tr><td><strong>F1 Macro</strong></td><td>92–94%</td></tr>
              <tr><td><strong>AUC</strong></td><td>0.96–0.98</td></tr>
              <tr><td><strong>Precision</strong></td><td>91–95%</td></tr>
              <tr><td><strong>Recall</strong></td><td>91–95%</td></tr>
            </tbody>
          </table>
        </div>
        <div class="result-card stagger-6">
          <h3 class="about-h">🛠 Tech stack</h3>
          <div class="chip-row">
            {''.join(f'<span class="info-chip">{t}</span>' for t in
              ['Python 3.10+','PyTorch','HF Transformers','scikit-learn',
               'Captum','LIME','Streamlit','pandas','NumPy'])}
          </div>
        </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Small shared helpers
# ─────────────────────────────────────────────────────────────────────────────
def _pct(v) -> str:
    return "N/A" if v is None else f"{float(v)*100:.2f}%"


def _show_figure(filename: str) -> None:
    path = os.path.join(config.FIGURES_DIR, filename)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        st.image(path, use_container_width=True)
        st.caption(f"📄 {Path(path).stem.replace('_', ' ').title()}")
    else:
        st.info(f"Figure not generated yet: {filename}")


def _render_summary_table(summary: Dict) -> None:
    pct_keys = {"accuracy", "f1_macro", "f1_weighted", "precision", "recall",
                "precision_macro", "recall_macro"}
    auc_keys = {"roc_auc", "auc"}
    best = max(summary, key=lambda m: float(summary[m].get("accuracy", 0)))
    sample = next(iter(summary.values()))
    keys = [k for k in sample if isinstance(sample.get(k), (int, float))]
    head = "".join(f'<th>{k.replace("_", " ").title()}</th>' for k in keys)
    body = ""
    for name, mv in summary.items():
        cells = ""
        for k in keys:
            v = mv.get(k)
            cells += ("<td>N/A</td>" if v is None else
                      f"<td>{float(v)*100:.2f}%</td>" if k.lower() in pct_keys else
                      f"<td>{float(v):.4f}</td>" if k.lower() in auc_keys else
                      f"<td>{v}</td>")
        star = " ★" if name == best else ""
        rc = ' class="best-row"' if name == best else ""
        body += f'<tr{rc}><td><strong>{name}{star}</strong></td>{cells}</tr>'
    st.markdown(f'<table class="model-info-table"><thead><tr><th>Model</th>{head}'
                f'</tr></thead><tbody>{body}</tbody></table>', unsafe_allow_html=True)


_METHOD_NOTES = """
<b>Occlusion (leave-one-out)</b> — the most faithful method here. For each word we
build a copy of the review with that word blanked out, re-run the model, and
measure how the <i>logit margin</i> (logit<sub>pos</sub> − logit<sub>neg</sub>)
changes. A drop means the word was pushing the prediction toward positive. We
score on the logit margin rather than P(positive) precisely because probability
<i>saturates</i> near 1.0 on confident reviews — every delta would otherwise round
to ~0. This is a direct, causal, model-agnostic probe (Zeiler &amp; Fergus, 2014).<br><br>
<b>LIME</b> — samples thousands of perturbations of <i>this one</i> review (randomly
dropping words), labels each with the model, and fits a sparse weighted linear
model to that local neighbourhood. The linear weights are the explanation. It is
an <i>approximation</i> of local behaviour — principled, widely cited (Ribeiro et
al., 2016), but less faithful than occlusion.<br><br>
<b>Integrated Gradients</b> — a white-box method (transformers only). It integrates
the gradient of the positive logit w.r.t. the input embeddings along a straight
path from a baseline (padding) to the real input, satisfying the completeness and
sensitivity axioms (Sundararajan et al., 2017). Sub-word attributions are summed
back to whole words.<br><br>
<b>Attention</b> — shows which tokens the last layer attends to, <i>not</i> why the
model decided. High attention is not evidence of sentiment direction; attention is
<i>not</i> a faithful explanation (Jain &amp; Wallace, 2019). Shown magnitude-only,
in purple, and never used for causal claims.
"""

_DARK_OVERRIDES = """
/* ===== Lab (dark) theme ===== */
.stApp{
  --color-bg:#0a1020;--color-surface:#121c33;--color-border:#26334f;
  --color-border-hover:#3a4d73;--color-text:#eaf1fb;--color-text-secondary:#a7b8d6;
  --color-text-tertiary:#7e90b3;
  background-color:#0a1020 !important;
}
.stApp::before{background:radial-gradient(circle,rgba(124,58,237,0.12)0%,rgba(37,99,235,0.08)40%,transparent 70%);}
.main .block-container{border-top-color:#7c3aed;}

/* generic text */
.stApp, .main p, .main li, .main span, .main h1, .main h2, .main h3, .main h4{color:var(--color-text);}
.section-heading h1{color:#f5f9ff !important;}
.section-heading p, .main [data-testid="stCaptionContainer"], .main [data-testid="stCaptionContainer"] *{color:var(--color-text-secondary) !important;}

/* TEXTAREA / INPUT — high specificity beats the original #000 rule */
.stApp .main .stTextArea textarea,
.stApp .main .stTextInput input,
.stApp .main .stNumberInput input{
  background:#0d172b !important;color:#eaf1fb !important;
  -webkit-text-fill-color:#eaf1fb !important;caret-color:#eaf1fb !important;border-color:#2a3a5a !important;}
.stApp .main .stTextArea textarea::placeholder,
.stApp .main .stTextInput input::placeholder{
  color:#8094b8 !important;-webkit-text-fill-color:#8094b8 !important;opacity:1 !important;}

/* SELECTBOX */
.stApp .main .stSelectbox div[data-baseweb="select"] > div{background:#0d172b !important;border-color:#2a3a5a !important;}
.stApp .main .stSelectbox div[data-baseweb="select"] *{color:#eaf1fb !important;}
div[data-baseweb="popover"] [role="option"], div[data-baseweb="popover"] li{background:#121c33 !important;color:#eaf1fb !important;}
div[data-baseweb="popover"] [role="option"]:hover, div[data-baseweb="popover"] li:hover{background:#1b2a48 !important;}

/* WIDGET LABELS (Review text, Explain, Model, Method, slider…) */
.stApp .main .stTextArea label, .stApp .main .stTextArea label p,
.stApp .main .stTextInput label, .stApp .main .stTextInput label p,
.stApp .main .stSelectbox label, .stApp .main .stSelectbox label p,
.stApp .main .stSlider label, .stApp .main .stSlider label p,
.stApp .main .stRadio label, .stApp .main .stRadio label p,
.stApp .main .stCheckbox label, .stApp .main .stCheckbox label p,
.stApp .main .stToggle label, .stApp .main .stToggle label p,
.stApp .main .stFileUploader label, .stApp .main .stFileUploader label p,
.stApp .main [class*="stWidgetLabel"] p{color:#eaf1fb !important;opacity:1 !important;}

/* EXPANDERS */
.stApp .main .stExpander{background:#101a30 !important;border:1px solid #26334f !important;}
.stApp .main .stExpander summary{color:#eaf1fb !important;background:#101a30 !important;}
.stApp .main .stExpander summary:hover{background:#16223c !important;}
.stApp .main .stExpander summary svg{color:#eaf1fb !important;fill:#eaf1fb !important;}
.stApp .main .stExpander details > div{background:#101a30 !important;}
.stApp .main [data-testid="stExpander"]{background:#101a30 !important;border:1px solid #26334f !important;}
.stApp .main [data-testid="stExpander"] summary{background:#101a30 !important;color:#eaf1fb !important;}

/* metrics */
[data-testid="stMetric"]{background:#121c33 !important;border-color:#26334f !important;}
[data-testid="stMetricValue"]{background:linear-gradient(135deg,#eaf1fb,#9fb6e0);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;}
[data-testid="stMetricLabel"]{color:var(--color-text-secondary) !important;}

/* dataframe */
[data-testid="stDataFrame"]{border-color:#26334f !important;background:#0e1830 !important;}
[data-testid="stDataFrame"] *{color:#dce6f7 !important;}

/* custom tables */
.model-info-table th{background:linear-gradient(180deg,#18233c,#101a2e) !important;color:#eaf1fb !important;}
.model-info-table td, .model-info-table td strong{color:#dce6f7 !important;}
.model-info-table tr:hover td{background:rgba(124,58,237,0.10) !important;}

/* tabs */
.stTabs [data-baseweb="tab"]{color:var(--color-text-secondary) !important;}
.stTabs [aria-selected="true"]{color:#a78bfa !important;border-bottom-color:#a78bfa !important;}
.stTabs [data-baseweb="tab-list"]{border-bottom-color:#26334f !important;}

/* file uploader */
[data-testid="stFileUploader"], [data-testid="stFileUploaderDropzone"], .stFileUploader section{background:#0e1830 !important;border-color:#2a3a5a !important;}
.stFileUploader section *{color:var(--color-text-secondary) !important;}

/* slider */
.stSlider [data-testid="stTickBar"]{color:var(--color-text-secondary) !important;}

/* code / alerts */
.stApp .main pre, .stApp .main code{background:#0d172b !important;color:#c8d6f0 !important;}
.stAlert{background:#101a30 !important;border-color:#26334f !important;}
.stAlert *{color:#dce6f7 !important;}

/* result card text */
.result-card, .result-card p, .result-card span, .result-card td, .result-card th,
.result-card strong, .result-card h1, .result-card h2, .result-card h3, .result-card h4{color:#eaf1fb !important;}

/* heatmap & signal text contrast in dark */
.hm-tok{color:#c4d2ec !important;}
.hm-active.hm-pos{color:#eafff6 !important;}
.hm-active.hm-neg{color:#fff0f0 !important;}
.hm-active.hm-focus{color:#f3eaff !important;}
.sig-word, .rl-ul li, .tu-note, .stat-label, .donut-legend, .pl-title{color:#dce6f7 !important;}
.flow-node{background:#0e1830 !important;color:#dce6f7 !important;}
.info-chip, .footer-badge, .ac-chip{color:#c4d2ec !important;}
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar + routing
# ─────────────────────────────────────────────────────────────────────────────
def sidebar(engine: Optional[InferenceEngine]) -> str:
    with st.sidebar:
        st.markdown("""
            <div class="sidebar-brand"><span class="logo-icon">🎬</span>
            <div class="sidebar-title">Sentiment IQ</div>
            <div class="sidebar-subtitle">Intelligence Platform</div></div>
            <div class="sidebar-divider"></div>
        """, unsafe_allow_html=True)
        page = st.radio("Navigation", [
            "🔍  Analyze", "⚔️  Model Arena", "🧠  Explainability",
            "📦  Batch Intelligence", "📊  Performance", "🔬  Comparison",
            "🩺  Error Analysis", "ℹ️  About",
        ], label_visibility="collapsed")
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.toggle("Lab (dark) theme", key="lab_theme")
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        if engine is not None:
            k = engine.default_model()
            try:
                info = engine.model_info(k)
                params = info.get("params")
                dev = info.get("device", "cpu")
            except Exception:
                params, dev = None, "cpu"
            st.markdown('<div style="font-size:0.85rem;margin-bottom:0.4rem;">'
                        '<span class="status-dot online"></span><strong>Model Active</strong>'
                        '</div>', unsafe_allow_html=True)
            st.caption(f"**Model:** {SPEC_BY_KEY[k].display_name}")
            if params:
                st.caption(f"**Params:** {params/1e6:.1f}M")
            st.caption(f"**Device:** {dev}")
            st.caption(f"**Available:** {len(engine.available_models())} models")
        else:
            st.markdown('<div style="font-size:0.85rem;">'
                        '<span class="status-dot offline"></span><strong>Model Offline</strong>'
                        '</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-divider"></div>'
                    '<div style="font-size:0.7rem;color:#475569;font-family:var(--font-mono);">'
                    'v2.0 · Final Project · 2025</div>', unsafe_allow_html=True)
    return page


def footer() -> None:
    st.markdown("""
        <div class="app-footer"><div class="footer-badges">
          <span class="footer-badge">🐍 Python</span>
          <span class="footer-badge">🔥 PyTorch</span>
          <span class="footer-badge">🤗 Transformers</span>
          <span class="footer-badge">🧪 scikit-learn</span>
          <span class="footer-badge">📊 Streamlit</span>
        </div><div>Explainable · Reliability-aware · Multi-model · Fine-tuned on IMDb 50K</div></div>
    """, unsafe_allow_html=True)


def main() -> None:
    load_css()
    engine, err = safe_engine()
    page = sidebar(engine)

    if "Analyze" in page:
        page_analyze(engine, err)
    elif "Arena" in page:
        page_arena(engine, err)
    elif "Explainability" in page:
        page_explain(engine, err)
    elif "Batch" in page:
        page_batch(engine, err)
    elif "Performance" in page:
        page_performance()
    elif "Comparison" in page:
        page_comparison()
    elif "Error" in page:
        page_errors(engine, err)
    else:
        page_about()

    footer()


main()
