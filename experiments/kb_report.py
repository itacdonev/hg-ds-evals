"""Generate an interactive HTML report from a KB-pipeline judge checkpoint CSV.

Works for both Czech (CZKB) and Slovak (SKKB) runs — pick with ``--lang cz`` or
``--lang sk``. The lang choice drives visible labels (button labels, the
``language_compliance`` dimension prompt, etc.) and the config-dir lookup;
the internal data model is identical for both.

Standalone — does NOT require the results-viewer notebook to have run.
Computes the same programmatic derivations the viewer does (weighted_avg,
retrieval_recall, root_cause_category, enum_f1, missing-ENUM counters,
Pearson agreement matrices) and writes a single self-contained HTML file
with tabbed navigation and a clickable per-test-case drill-down.

Usage:
    python kb_report.py --lang cz \\
        --checkpoint checkpoints/evals_czkb_exp_002_<RUN>.csv \\
        --output     /tmp/kb_report.html

    # With a baseline run for per-case + per-metric regression comparison:
    python kb_report.py --lang sk \\
        --checkpoint checkpoints/evals_skkb_exp_001_<RUN>.csv \\
        --baseline   checkpoints/evals_skkb_exp_001_<PRIOR_RUN>.csv \\
        --output     /tmp/kb_report.html
"""
from __future__ import annotations

import argparse
import ast
import html as _html
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots
from kb_checkpoint import read_checkpoint_csv

# Walk up from this file to find the repo root so the report can be run
# straight from `experiments/` without `pip install -e .`.
# Mirrors the bootstrap used in `backfill_latency.py`.
_REPO_ROOT = Path(__file__).resolve().parent
while _REPO_ROOT != _REPO_ROOT.parent and not (_REPO_ROOT / "hg_ds_evals").is_dir():
    _REPO_ROOT = _REPO_ROOT.parent
if (_REPO_ROOT / "hg_ds_evals").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hg_ds_evals.preprocessing.latency import (  # noqa: E402 — needs sys.path tweak above
    RETRY_BASELINE_TOK_PER_S,
    RETRY_MAX_TOK_PER_S,
    RETRY_MIN_DURATION_MS,
)

# g-evals design tokens (mirrored from g-evals/frontend/src/index.css).
GE_FONT = "Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
GE_BLUE = "#135ee2"            # --rgb-blue-300 (primary)
GE_TEXT = "#0a285c"            # --rgb-blue-400 (text primary)
GE_MUTED = "#537090"           # --rgb-gray-400 (text secondary)
GE_GRID = "#edf0f4"
GE_GREEN = "#057f19"
GE_RED = "#cf2a1e"
GE_YELLOW = "#f2a91e"
GE_BLUE_SCALE = [[0, "#f4f8fe"], [0.5, "#7aa6ef"], [1, "#135ee2"]]
GE_GREEN_SCALE = [[0, "#f1faf3"], [0.5, "#74c089"], [1, "#057f19"]]


def _style_fig(fig, *, height: int | None = None):
    """Apply g-evals-inspired styling to any Plotly figure."""
    if fig is None:
        return None
    fig.update_layout(
        template="simple_white",
        font=dict(family=GE_FONT, size=12, color=GE_TEXT),
        title_font=dict(family=GE_FONT, size=13, color=GE_TEXT),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        colorway=[GE_BLUE, GE_GREEN, GE_YELLOW, GE_RED, GE_MUTED],
        legend=dict(font=dict(size=11, color=GE_MUTED), bgcolor="rgba(0,0,0,0)"),
    )
    if height is not None:
        fig.update_layout(height=height)
    fig.update_xaxes(
        showgrid=True, gridcolor=GE_GRID, gridwidth=1,
        zeroline=False, linecolor=GE_GRID, ticks="outside",
        tickcolor=GE_GRID, tickfont=dict(color=GE_MUTED, size=11),
        title_font=dict(color=GE_MUTED, size=11),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=GE_GRID, gridwidth=1,
        zeroline=False, linecolor=GE_GRID, ticks="outside",
        tickcolor=GE_GRID, tickfont=dict(color=GE_MUTED, size=11),
        title_font=dict(color=GE_MUTED, size=11),
    )
    return fig


# Dimension weights — keep in sync with the same dimensions in
# configs/czkb_exp_001_baseline_no_expected_enums.yaml. weighted_avg
# is normalized by sum(weights) and rescaled to [0, 1], so changing weights
# reshapes which dimensions matter most but does NOT shift the pass
# threshold (0.7 always = 70% of the weighted maximum).
DIMENSION_WEIGHTS = {
    # Display order grouped by stage of the pipeline:
    #   query understanding → selection quality → answer quality →
    #   pool adequacy. Dictionary order drives the order of the per-scorer
    #   sub-cards on Summary, the per-scorer distribution table on Scorers,
    #   and the heatmap axes.
    "query_clarity":                       1.0,
    "language_compliance":                 1.0,
    "selection_semantic_relevance":        2.0,
    "selected_context_sufficiency":        2.0,
    "answer_expected_alignment":           2.0,
    "answer_groundedness":                 2.0,
    "optimal_retrieved_context_adequacy":  2.0,
}
PASS_THRESHOLD = 0.7
# Hard veto on top of the threshold: if any *critical* (weight ≥ 2)
# scorer scored 0, the case fails regardless of weighted_avg. Without
# this, a case can still clear 0.7 with one weight-2 dimension at zero
# (e.g. all 2s except groundedness = 0 → 0.83), which is exactly the
# "the agent fabricated something but the rest looked fine" outcome we
# don't want hidden behind a high mean.
CRITICAL_DIMS = tuple(d for d, w in DIMENSION_WEIGHTS.items() if w >= 2.0)

# ── Language scaffolding ────────────────────────────────────────────────────
# Set by `main()` from the ``--lang`` CLI flag. Defaults to "cz" so the module
# stays importable + the existing CZKB call sites keep working. SK runs flip
# these via `_apply_lang("sk")` before any HTML is rendered, which:
#   - substitutes ``{LANG_NAME}`` placeholders in the failure-mode dicts
#     defined below,
#   - swaps the visible language-toggle button label ("CZ" vs "SK") in the
#     rendered HTML,
#   - picks per-lang external benchmarks out of ``BENCHMARKS``.
# NOTE: the internal "native-language" slot in CSS/JS (data-lang="cz",
# .lang-cz, etc.) is kept named "cz" for both CZ and SK runs — it's the
# slot identifier, not a Czech-specific tag. Only labels visible to the
# user are templated by language.
LANG = "cz"
LANG_NAME = "Czech"
LANG_LABEL_UPPER = "CZ"

# Per-lang external benchmarks. Used by the Summary tab's reference-line
# chips (KB Recall, Rel2 Mean) and by the colour-good logic on those cards.
# Setting any value to ``None`` makes the corresponding card skip the
# benchmark chip + threshold-good colouring (matching the API report's
# "if None, just don't render it" pattern).
#
# Internal pass-rate target (`pass_rate_target`) is the same agent-level
# goal for both languages; the CSAS rel2 / recall benchmarks are currently
# treated as applicable to both — replace the SK values when the SK-side
# CSAS-equivalent benchmark exists.
BENCHMARKS = {
    "cz": {
        "csas_rel2":         0.643,
        "csas_recall":       0.656,
        "pass_rate_target":  0.90,
    },
    "sk": {
        # Provisional: re-use the CZ CSAS numbers until the SK benchmark
        # ships. Swap to SK values (or set to None) when available.
        "csas_rel2":         0.643,
        "csas_recall":       0.656,
        "pass_rate_target":  0.90,
    },
}
# These three module-level scalars are kept as a thin compatibility surface
# over BENCHMARKS[LANG] so the rendering layer can read them as plain names
# (`PASS_RATE_TARGET` etc.) without threading the lang code through every
# helper. `_apply_lang()` keeps them in sync.
CSAS_BENCHMARK_REL2   = BENCHMARKS[LANG]["csas_rel2"]
CSAS_BENCHMARK_RECALL = BENCHMARKS[LANG]["csas_recall"]
PASS_RATE_TARGET      = BENCHMARKS[LANG]["pass_rate_target"]


def _apply_lang(lang: str) -> None:
    """Switch the module to ``lang`` ("cz" or "sk").

    Updates the language globals, swaps in the per-lang benchmark scalars,
    and substitutes ``{LANG_NAME}`` placeholders in the two failure-mode
    dicts defined further below. Idempotent — safe to call more than once,
    e.g. when re-running render_html from a notebook.

    Call this exactly once from ``main()`` (or from a notebook that imports
    this module) BEFORE rendering. The substitution is done in-place on
    ``FAILURE_MODE_INFO`` / ``FAILURE_MODE_INFO_LONG`` so the rest of the
    render layer can keep reading them as plain dicts.
    """
    global LANG, LANG_NAME, LANG_LABEL_UPPER
    global CSAS_BENCHMARK_REL2, CSAS_BENCHMARK_RECALL, PASS_RATE_TARGET
    if lang not in BENCHMARKS:
        raise ValueError(f"unknown --lang {lang!r}; expected one of {sorted(BENCHMARKS)}")
    prior_name = LANG_NAME
    LANG = lang
    LANG_NAME = {"cz": "Czech", "sk": "Slovak"}[lang]
    LANG_LABEL_UPPER = lang.upper()
    CSAS_BENCHMARK_REL2   = BENCHMARKS[LANG]["csas_rel2"]
    CSAS_BENCHMARK_RECALL = BENCHMARKS[LANG]["csas_recall"]
    PASS_RATE_TARGET      = BENCHMARKS[LANG]["pass_rate_target"]
    # Substitute placeholders in the two dicts. Also reverse any prior
    # substitution so toggling lang is idempotent across calls.
    for target in (FAILURE_MODE_INFO, FAILURE_MODE_INFO_LONG):
        for key, val in list(target.items()):
            if not isinstance(val, str):
                continue
            if prior_name and prior_name != LANG_NAME:
                val = val.replace(prior_name, "{LANG_NAME}")
            if "{LANG_NAME}" in val:
                target[key] = val.replace("{LANG_NAME}", LANG_NAME)


# ── Derivations ──────────────────────────────────────────────────────────────
def _parse_list(x):
    if isinstance(x, list):
        return x
    if not isinstance(x, str) or not x.strip():
        return []
    try:
        parsed = json.loads(x)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []
    return parsed if isinstance(parsed, list) else []


def _prf(retrieved, expected):
    r, e = set(retrieved), set(expected)
    if not e and not r:
        return (np.nan, np.nan, np.nan)
    tp = len(r & e)
    precision = tp / len(r) if r else np.nan
    recall = tp / len(e) if e else np.nan
    if precision and recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0 if (precision == 0 or recall == 0) else np.nan
    return (precision, recall, f1)


# ── Enum-ID name normalization (deterministic test-set hygiene checks) ──────
# Catches false zeros caused by separator/format differences between the
# expected ENUM IDs in the dataset and the canonical IDs surfaced from the KB
# (e.g. SHARE_PRODUCT_INFO vs SHARE_PRODUCTINFO,
#       CHANGE_CARD_LIMITS-SUFFICIENT_BALANCE vs CHANGE_CARD_LIMITS_SUFFICIENT_BALANCE,
#       "A + B" expected-syntax meant to address two ENUMs).
_ENUM_ATOM_SPLIT_RE = re.compile(r"[+,;&]")


def _normalize_enum_id(s) -> str:
    """Drop every non-alphanumeric character and uppercase. Lets us match
    ``SHARE_PRODUCT_INFO`` with ``SHARE_PRODUCTINFO`` while still being
    case- and separator-tolerant."""
    if not isinstance(s, str):
        return ""
    return re.sub(r"[^A-Za-z0-9]", "", s).upper()


def _split_expected_enum_atoms(eid) -> list[str]:
    """Treat ``A + B`` (or A, B / A; B / A & B) as two atoms. Returns the
    trimmed individual ENUM IDs the test author intended to assert.
    Non-string / empty input → []."""
    if not isinstance(eid, str) or not eid.strip():
        return []
    parts = _ENUM_ATOM_SPLIT_RE.split(eid)
    return [p.strip() for p in parts if p.strip()]


# ── Sentinel normalization ──────────────────────────────────────────────────
# The judge emits an "Answer is not available based on given information"
# string (with several incidental whitespace variants) in three list fields
# when the post-prune pool / reranked context cannot support the answer.
# Treat that as the empty list for any counting / signal extraction.
_SENTINEL_PHRASE_NORMALIZED = "answer is not available based on given information"


def _is_void_string(s) -> bool:
    if not isinstance(s, str):
        return False
    collapsed = " ".join(s.lower().split())
    return _SENTINEL_PHRASE_NORMALIZED in collapsed


def _normalize_array(arr):
    if not isinstance(arr, list):
        return []
    return [x for x in arr if not (isinstance(x, str) and _is_void_string(x))]


# ── Failure-mode taxonomy ────────────────────────────────────────────────────
# Per-case primary tag computed deterministically from existing fields.
# Priority order matters — first match wins. Owners map to teams referenced
# by the YAML's *_improvement_suggestion fields.
FAILURE_MODES = (
    "pass",
    "test_set_defect",
    "enum_name_mismatch",
    "scope_misroute",
    "retrieval_gap",
    "pruning_loss",
    "reranker_miss",
    "pool_content_gap",
    "context_use_failure",
    "hallucination",
    "language_drift",
    "critical_score_zero",
    "other_failure",
)

FAILURE_MODE_OWNER = {
    "pass":                "—",
    "test_set_defect":     "Test set",
    "enum_name_mismatch":  "Test set",
    "scope_misroute":      "Agent",
    "retrieval_gap":       "Retrieval",
    "pruning_loss":        "Pruning",
    "reranker_miss":       "Reranker",
    "pool_content_gap":    "KB content",
    "context_use_failure": "Agent",
    "hallucination":       "Agent",
    "language_drift":      "Agent",
    "critical_score_zero": "Agent",
    "other_failure":       "—",
}

FAILURE_MODE_LABEL = {
    "pass":                "Clean pass",
    "test_set_defect":     "Test-set issue",
    "enum_name_mismatch":  "ENUM name mismatch",
    "scope_misroute":      "Wrong agent routing",
    "retrieval_gap":       "Retrieval gap",
    "pruning_loss":        "Pruning loss",
    "reranker_miss":       "Reranker miss",
    "pool_content_gap":    "Pool content gap",
    "context_use_failure": "Context-use failure",
    "hallucination":       "Hallucination",
    "language_drift":      "Language drift",
    "critical_score_zero": "Critical scorer at 0",
    "other_failure":       "Other issue",
}

# Two-tier descriptions: SHORT (FAILURE_MODE_INFO) goes in the failure-mode
# table — concise, scannable. LONG (FAILURE_MODE_INFO_LONG) feeds the per-row
# info-icon tooltip and the Doc tab so the full reasoning is one click away.
FAILURE_MODE_INFO = {
    "pass":                f"weighted_avg ≥ {PASS_THRESHOLD:g}, no weight-2 scorer at 0, AND no other issue fired. Strict subset of the headline pass rate (which also counts judge-passes where a higher-priority issue fired, e.g. test-set defects).",
    "test_set_defect":     "Judge flagged the gold reference as needing review.",
    "enum_name_mismatch":  "Expected ENUM matches a KB ENUM only after stripping case/separators — false-zero risk.",
    "scope_misroute":      "Judge said KB; system never reached the KB.",
    "retrieval_gap":       "Expected ENUM never entered the pre-prune candidate pool.",
    "pruning_loss":        "Expected ENUM was dropped between pre-prune and post-prune.",
    "reranker_miss":       "Expected ENUM was in the post-prune pool but not picked.",
    "pool_content_gap":    "Pool fragments exist but are too thin to answer.",
    "context_use_failure": "Context was usable; agent answered wrong anyway.",
    "hallucination":       "Severe: agent's answer contains important unsupported / fabricated claims (groundedness = 0).",
    "language_drift":      "Agent answered in non-{LANG_NAME}.",
    "critical_score_zero": "A weight-2 scorer scored 0; weighted_avg cleared 0.7 but the case fails on the hard veto.",
    "other_failure":       "Did not match any earlier category — inspect manually.",
}

FAILURE_MODE_INFO_LONG = {
    "test_set_defect":
        "expected_reference_looks_wrong=True. The agent's behaviour on these "
        "rows says nothing about KB performance — review the gold reference first. "
        "Cases the judge classified as api / out_of_scope / ambiguous are "
        "pre-filtered out of the report; they appear in the Doc tab's "
        "case_scope distribution.",
    "enum_name_mismatch":
        "Deterministic check: expected_enum doesn't match any retrieved / "
        "post-prune ENUM exactly, but matches after normalization (case + "
        "non-alphanumeric stripped). KB and test set use different name "
        "conventions for the same fragment. Fix the spelling on either side, "
        "or normalize before scoring.",
    "scope_misroute":
        "Judge classified the query as KB / KB+API but query_scope ≠ kb — "
        "the agent stopped before retrieval ran.",
    "retrieval_gap":
        "Vector DB did not surface the expected ENUM into the pre-prune pool. "
        "Either embedding/recall failed or the content is missing from the index.",
    "pruning_loss":
        "Expected ENUM(s) were in the pre-prune pool but the pruning step "
        "dropped them before the reranker saw them.",
    "reranker_miss":
        "Expected ENUM(s) were in the post-prune pool offered to the reranker, "
        "but the reranker did not pick them.",
    "pool_content_gap":
        "Judge flagged retrieved_pool_inadequacy_identified AND issued a "
        "kb_improvement_suggestion: post-prune fragments exist but are too "
        "thin / incomplete to answer the query.",
    "context_use_failure":
        "Reranker delivered usable context (sufficiency ≥ 1) but the agent's "
        "answer disagreed with the expected reference (alignment = 0). "
        "Agent prompt or generation issue.",
    "hallucination":
        "Severe hallucination only: answer_groundedness == 0 AND ≥ 1 "
        "hallucinated_claim AND non-empty kb_context. Per the YAML rubric, "
        "score 0 means \"important unsupported, fabricated, or contradictory "
        "claims\" — the agent is genuinely making things up. Cases with "
        "groundedness = 1 (\"one minor unsupported claim or wording "
        "overreach\") are not counted here; those almost always land in "
        "passing cases and would distort the headline. The chip filter "
        "and co-occurrence heatmap still surface the broader signal via "
        "_failure_modes_all.",
    "language_drift":
        "language_compliance ≤ 1 — agent answered in {LANG_NAME} or another "
        "non-{LANG_NAME} language.",
    "critical_score_zero":
        f"weighted_avg ≥ {PASS_THRESHOLD:g} (so the average says \"pass\"), "
        "but at least one weight-2 scorer "
        f"({', '.join(CRITICAL_DIMS)}) "
        "scored 0. We veto pass in that case so the headline doesn't hide "
        "a critical-dimension failure behind a high mean. Cases where a "
        "more specific issue also fired (hallucination, "
        "context-use failure, etc.) are classified under that issue "
        "instead — this bucket only catches the residual.",
}

# Apply the default lang now that the dicts exist, so importers that skip
# main() still see the placeholder substituted in. main() will re-apply if
# the user passes --lang explicitly (idempotent).
_apply_lang(LANG)


def _root_cause(r) -> str:
    if r.get("error") is True or str(r.get("error", "")).lower() == "true":
        return "judge_error"
    if r.get("language_compliance_score") == 0:
        return "language_mismatch"
    if r.get("query_clarity_score") == 0:
        return "query_ambiguous"
    if r.get("retrieval_recall_score", 2) <= 1:
        return "retriever_missed_enums"
    if r.get("selection_semantic_relevance_score") == 0:
        return "reranker_wrong_selection"
    if r.get("optimal_retrieved_context_adequacy_score") == 0:
        return "retrieved_pool_inadequacy"
    if r.get("selected_context_sufficiency_score") <= 1:
        return "selected_context_insufficient"
    if r.get("answer_groundedness_score") <= 1:
        return "hallucination_or_ungrounded_answer"
    if r.get("answer_expected_alignment_score") <= 1:
        return "answer_generation_issue"
    if r.get("language_compliance_score") == 1:
        return "language_mismatch"
    return "no_issue"


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add all programmatic columns consumed by the report."""
    df = df.copy()

    # Every row needs a non-empty, unique test_case_id: the Test Cases tab
    # uses it as the DOM key for selection/highlighting, and every drill-down
    # link in other tabs filters by it. When the checkpoint doesn't carry a
    # real one (or it's blank/NaN), fall back to trace_id, then to a
    # synthetic case_NNNN index — otherwise empty ids collapse all rows
    # onto the same key and selecting one row "selects" all of them.
    if "test_case_id" not in df.columns:
        df["test_case_id"] = ""
    tid = df["test_case_id"].astype("object").where(df["test_case_id"].notna(), "")
    tid = tid.astype(str).str.strip()
    missing = tid.eq("") | tid.str.lower().isin({"nan", "none"})
    if missing.any() and "trace_id" in df.columns:
        trace_fb = df["trace_id"].astype(str).str.strip()
        trace_ok = trace_fb.ne("") & ~trace_fb.str.lower().isin({"nan", "none"})
        tid = tid.mask(missing & trace_ok, trace_fb)
        missing = tid.eq("") | tid.str.lower().isin({"nan", "none"})
    if missing.any():
        synthetic = pd.Series(
            [f"case_{i + 1:04d}" for i in range(len(df))], index=df.index
        )
        tid = tid.mask(missing, synthetic)
    df["test_case_id"] = tid

    for dim in DIMENSION_WEIGHTS:
        col = f"{dim}_score"
        if col not in df.columns:
            raise KeyError(f"missing dimension column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Latency columns arrive as strings from read_checkpoint_csv (which forces
    # dtype=str). Cast to float so the bootstrap CI and summary stats work.
    for col in ("lat_total_ms",
                "lat_routing_ms", "lat_planning_llm_ms", "lat_kb_retrieve_ms",
                "lat_kb_prune_ms", "lat_kb_rerank_ms", "lat_tools_ms",
                "lat_generation_llm_ms", "lat_overhead_ms",
                "lat_retry_overhead_ms", "lat_retry_call_count"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    w_sum = sum(DIMENSION_WEIGHTS.values())
    # Per-dim scores are 0/1/2; the extra `/ 2.0` puts the weighted average in
    # [0, 1] for easier human interpretation. Threshold is scaled accordingly
    # (1.4/2 = 0.7), so pass/fail outcomes are unchanged vs the [0, 2] scale.
    df["weighted_avg"] = (sum(df[f"{d}_score"] * w for d, w in DIMENSION_WEIGHTS.items())
                            / w_sum / 2.0)
    # Critical-zero veto: any weight-≥2 dimension at score 0 blocks pass
    # regardless of the weighted average. See CRITICAL_DIMS comment above.
    df["_critical_zero"] = pd.concat(
        [df[f"{d}_score"].eq(0) for d in CRITICAL_DIMS], axis=1
    ).any(axis=1)
    df["pass"] = (df["weighted_avg"] >= PASS_THRESHOLD) & ~df["_critical_zero"]
    df["total_score_judge"] = sum(df[f"{d}_score"] for d in DIMENSION_WEIGHTS)

    for c in ("reranked_enum_ids", "expected_enums",
              "post_prune_enum_ids", "pre_prune_enum_ids",
              "missing_enums", "missing_enums_in_candidate_pool", "missing_enums_not_in_pool",
              "extra_or_distracting_enums", "optimal_enum_selection",
              "hallucinated_claims", "missing_facts",
              "unavailable_facts_in_selected_context"):
        if c in df.columns:
            df[f"_{c}"] = df[c].apply(_parse_list)

    # Claim-count columns + kb_context_empty flag (used by the Judge Eval tab).
    for src, dst in (
        ("hallucinated_claims",                 "hallucinated_claims_cnt"),
        ("missing_facts",                       "missing_facts_cnt"),
        ("unavailable_facts_in_selected_context","unavailable_facts_cnt"),
        ("extra_or_distracting_enums",          "extra_distracting_enums_cnt"),
    ):
        if f"_{src}" in df.columns:
            df[dst] = df[f"_{src}"].apply(len)
    if "reranked_kb_context" in df.columns:
        df["kb_context_empty"] = df["reranked_kb_context"].fillna("").astype(str).str.strip().eq("")
    else:
        df["kb_context_empty"] = pd.Series([False] * len(df), index=df.index)

    def _recall(r):
        e = set(r.get("_expected_enums", []))
        if not e:
            return np.nan
        return len(e & set(r.get("_post_prune_enum_ids", []))) / len(e)

    df["retrieval_recall"] = df.apply(_recall, axis=1)
    df["retrieval_recall_score"] = df["retrieval_recall"].apply(
        lambda v: np.nan if pd.isna(v) else (2 if v >= 0.999 else (1 if v > 0 else 0))
    )

    df["_expected_enums_missed_by_retriever"] = df.apply(
        lambda r: sorted(set(r.get("_expected_enums", [])) - set(r.get("_post_prune_enum_ids", []))),
        axis=1,
    )
    df["_expected_enums_missed_by_reranker"] = df.apply(
        lambda r: sorted(
            (set(r.get("_expected_enums", [])) & set(r.get("_post_prune_enum_ids", [])))
            - set(r.get("_reranked_enum_ids", []))
        ),
        axis=1,
    )
    df["_missing_enums"] = df.apply(
        lambda r: sorted(set(r.get("_expected_enums", [])) - set(r.get("_reranked_enum_ids", []))),
        axis=1,
    )
    df["_missing_enums_not_in_pool"] = df["_expected_enums_missed_by_retriever"]
    df["_missing_enums_in_candidate_pool"] = df["_expected_enums_missed_by_reranker"]
    df["missing_enums"] = df["_missing_enums"].apply(json.dumps)
    df["missing_enums_not_in_pool"] = df["_missing_enums_not_in_pool"].apply(json.dumps)
    df["missing_enums_in_candidate_pool"] = df["_missing_enums_in_candidate_pool"].apply(json.dumps)
    df["selection_was_optimal"] = df.apply(
        lambda r: set(r.get("_reranked_enum_ids", [])) == set(r.get("_expected_enums", []))
        if r.get("_expected_enums", []) else np.nan,
        axis=1,
    )
    df["root_cause_category"] = df.apply(_root_cause, axis=1)
    df[["enum_precision", "enum_recall", "enum_f1"]] = df.apply(
        lambda r: pd.Series(_prf(r.get("_reranked_enum_ids", []), r.get("_expected_enums", []))),
        axis=1,
    )
    if "expert_score" in df.columns:
        df["expert_score"] = pd.to_numeric(df["expert_score"], errors="coerce")
    if "enum_relevance_score" in df.columns:
        df["enum_relevance_score"] = pd.to_numeric(df["enum_relevance_score"], errors="coerce")

    # ── Sentinel-normalized list columns + per-claim counters ────────────────
    # These override the raw _hallucinated_claims / _unavailable_facts /
    # _missing_facts so any downstream consumer (counters, judge-eval contract
    # checks, failure-mode classifier) treats the void sentinel as [].
    for src in ("hallucinated_claims",
                "unavailable_facts_in_selected_context",
                "missing_facts"):
        col = f"_{src}"
        if col in df.columns:
            df[col] = df[col].apply(_normalize_array)
    for src, dst in (
        ("hallucinated_claims",                 "hallucinated_claims_cnt"),
        ("missing_facts",                       "missing_facts_cnt"),
        ("unavailable_facts_in_selected_context","unavailable_facts_cnt"),
    ):
        if f"_{src}" in df.columns:
            df[dst] = df[f"_{src}"].apply(len)

    # ── Empty user_query flag ────────────────────────────────────────────────
    if "user_query" in df.columns:
        df["_user_query_empty"] = (
            df["user_query"].fillna("").astype(str).str.strip().eq("")
        )
    else:
        df["_user_query_empty"] = pd.Series([True] * len(df), index=df.index)

    # _recall_optimal_vs_final is read by the per-case stage-recall table.
    def _opt_vs_final(r) -> float:
        opt = set(r.get("_optimal_enum_selection", []) or [])
        final = set(r.get("_reranked_enum_ids", []) or [])
        if not opt:
            return np.nan
        return len(opt & final) / len(opt)

    df["_recall_optimal_vs_final"] = df.apply(_opt_vs_final, axis=1)

    # ── Boolean flags used by the failure-mode classifier + summary tiles ───
    cs = df.get("case_scope", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
    qs = df.get("query_scope", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
    df["_case_scope_kb_like"] = cs.isin({"kb", "kb_and_api"})
    df["_case_scope_test_defect"] = cs.isin({"ambiguous", "out_of_scope"})
    df["_gold_defect"] = df.get(
        "expected_reference_looks_wrong", pd.Series([""] * len(df), index=df.index)
    ).astype(str).str.lower().isin({"true", "1", "1.0"})
    df["_pool_inadequate_flag"] = df.get(
        "retrieved_pool_inadequacy_identified", pd.Series([""] * len(df), index=df.index)
    ).astype(str).str.lower().isin({"true", "1", "1.0"})
    df["_kb_suggestion_nonempty"] = df.get(
        "kb_improvement_suggestion", pd.Series([""] * len(df), index=df.index)
    ).fillna("").astype(str).str.strip().ne("")

    df["_agent_skipped_kb"] = df["_case_scope_kb_like"] & (qs != "kb")

    g = lambda c: pd.to_numeric(df.get(c, pd.Series([np.nan]*len(df), index=df.index)),
                                  errors="coerce")
    align = g("answer_expected_alignment_score")
    suff  = g("selected_context_sufficiency_score")
    grd   = g("answer_groundedness_score")
    lang  = g("language_compliance_score")

    df["_context_use_failure"] = (align == 0) & (suff >= 1)
    # Two hallucination flags with different strictness:
    #   _hallucinated_severe  — primary failure-mode classifier uses this.
    #     answer_groundedness == 0 ("important unsupported, fabricated, or
    #     contradictory claims"). Excludes minor / nit-pick cases that
    #     would otherwise dominate the failure-mode table even though the
    #     case itself passed.
    #   _hallucinated_any     — chip filter + co-occurrence heatmap use
    #     this via _failure_modes_all. Any non-empty hallucinated_claims
    #     with non-empty context, regardless of severity. Lets colleagues
    #     still surface every case the judge flagged in the diagnostic
    #     views.
    _has_claim = df.get(
        "hallucinated_claims_cnt", pd.Series([0]*len(df), index=df.index)
    ).gt(0) & ~df["kb_context_empty"]
    df["_hallucinated_any"]    = _has_claim
    df["_hallucinated_severe"] = (grd == 0) & _has_claim
    # Backward-compat alias for downstream code that still reads the old
    # name. Points at the SEVERE definition (matches the classifier).
    df["_hallucinated_with_context"] = df["_hallucinated_severe"]
    df["_wrong_language"] = lang <= 1

    # ── Stage-recall booleans (any expected ENUM missed at this stage) ──────
    def _any_missed(expected, stage) -> bool:
        e = set(expected or [])
        if not e:
            return False
        return not e.issubset(set(stage or []))

    df["_retrieval_gap_flag"] = df.apply(
        lambda r: _any_missed(r.get("_expected_enums", []), r.get("_pre_prune_enum_ids", [])),
        axis=1,
    )
    # Pruning lost what retrieval found.
    df["_pruning_loss_flag"] = df.apply(
        lambda r: (
            bool(set(r.get("_expected_enums", []) or []))
            and set(r.get("_expected_enums", []) or []).issubset(
                set(r.get("_pre_prune_enum_ids", []) or [])
            )
            and not set(r.get("_expected_enums", []) or []).issubset(
                set(r.get("_post_prune_enum_ids", []) or [])
            )
        ),
        axis=1,
    )
    # Reranker lost what pruning kept.
    df["_reranker_miss_flag"] = df.apply(
        lambda r: (
            bool(set(r.get("_expected_enums", []) or []))
            and set(r.get("_expected_enums", []) or []).issubset(
                set(r.get("_post_prune_enum_ids", []) or [])
            )
            and not set(r.get("_expected_enums", []) or []).issubset(
                set(r.get("_reranked_enum_ids", []) or [])
            )
        ),
        axis=1,
    )
    df["_pool_content_gap_flag"] = df["_pool_inadequate_flag"] & df["_kb_suggestion_nonempty"]

    # ── Deterministic test-set hygiene: ENUM name mismatch detection ────────
    # Build a "known KB enum IDs" universe by taking the union of every ID
    # observed in any row's pre-prune / post-prune / reranked stage. That's
    # an approximation of the KB index's surface area as exercised by the
    # eval; good enough to catch separator/format mismatches like
    # SHARE_PRODUCT_INFO vs SHARE_PRODUCTINFO without requiring direct KB
    # CSV access from the report.
    known_enums: set[str] = set()
    for col in ("_pre_prune_enum_ids", "_post_prune_enum_ids", "_reranked_enum_ids"):
        if col in df.columns:
            for lst in df[col]:
                if isinstance(lst, list):
                    for e in lst:
                        if isinstance(e, str) and e.strip():
                            known_enums.add(e.strip())
    # Map normalized form → list of canonical KB IDs (usually 1, sometimes
    # >1 if the KB itself has duplicates that differ only in format).
    known_norm_to_ids: dict[str, list[str]] = {}
    for e in known_enums:
        known_norm_to_ids.setdefault(_normalize_enum_id(e), []).append(e)

    def _find_naming_mismatches(expected_list) -> list[dict]:
        out: list[dict] = []
        if not isinstance(expected_list, list):
            return out
        for raw_eid in expected_list:
            atoms = _split_expected_enum_atoms(raw_eid)
            # If splitting found nothing usable, fall back to the raw eid
            # so we still try to match it.
            if not atoms:
                if isinstance(raw_eid, str) and raw_eid.strip():
                    atoms = [raw_eid.strip()]
                else:
                    continue
            for atom in atoms:
                if atom in known_enums:
                    continue                                   # exact match
                norm = _normalize_enum_id(atom)
                if not norm:
                    continue
                kb_forms = known_norm_to_ids.get(norm, [])
                if kb_forms and atom not in kb_forms:
                    out.append({
                        "expected": atom,
                        "kb_form": kb_forms[0] if len(kb_forms) == 1 else kb_forms,
                        "raw_expected": raw_eid,
                    })
        return out

    df["_enum_naming_mismatches"] = df.get(
        "_expected_enums", pd.Series([[]]*len(df), index=df.index)
    ).apply(_find_naming_mismatches)
    df["_naming_mismatch"] = df["_enum_naming_mismatches"].apply(lambda x: bool(x))

    # NOTE: _rel2_adjusted (per-case adjustment of upstream rel2 for naming
    # mismatches) used to live here. It was dropped once we decided naming
    # mismatches are test-set issues and should be EXCLUDED from KB Recall /
    # KB Precision / Rel2 / funnel rather than adjusted into them. Downstream
    # consumers now read `enum_relevance_score` directly and apply the
    # exclusion at the metric level.

    # ── Primary failure mode (priority order) ────────────────────────────────
    def _classify_failure(r) -> str:
        if r.get("_gold_defect", False) or r.get("_case_scope_test_defect", False):
            return "test_set_defect"
        if r.get("_naming_mismatch", False):
            return "enum_name_mismatch"
        if r.get("_agent_skipped_kb", False):
            return "scope_misroute"
        # The retrieval/pruning/reranker stages only apply when we know the
        # expected ENUM ground truth and the case is genuinely a KB question.
        has_expected = bool(r.get("_expected_enums", []) or [])
        if r.get("_case_scope_kb_like", False) and has_expected:
            if r.get("_retrieval_gap_flag", False):
                return "retrieval_gap"
            if r.get("_pruning_loss_flag", False):
                return "pruning_loss"
            if r.get("_reranker_miss_flag", False):
                return "reranker_miss"
        if r.get("_pool_content_gap_flag", False):
            return "pool_content_gap"
        if r.get("_context_use_failure", False):
            return "context_use_failure"
        if r.get("_hallucinated_with_context", False):
            return "hallucination"
        if r.get("_wrong_language", False):
            return "language_drift"
        # Critical-zero veto sits between language_drift and pass: a case
        # whose weighted_avg cleared the threshold but had a weight-2
        # scorer at 0 is NOT a pass. Cases where a specific failure mode
        # already fired are classified as that mode and never reach this
        # branch, so this bucket only catches the residual where no other
        # signal fired.
        wa_clears = pd.notna(r.get("weighted_avg")) and r["weighted_avg"] >= PASS_THRESHOLD
        if wa_clears and r.get("_critical_zero", False):
            return "critical_score_zero"
        if wa_clears:
            return "pass"
        return "other_failure"

    df["failure_mode"] = df.apply(_classify_failure, axis=1)

    # All applicable failure modes per case (priority order preserved). The
    # primary `failure_mode` is the first-wins value; this list is "every
    # mode whose condition is true". Used by the Test Cases multi-select
    # filter so a case can be matched on a non-primary mode (e.g. a row
    # whose primary is test_set_defect can still be selected when filtering
    # for hallucination if both fired).
    def _all_modes(r) -> list[str]:
        modes: list[str] = []
        if r.get("_gold_defect", False):
            modes.append("test_set_defect")
        if r.get("_naming_mismatch", False):
            modes.append("enum_name_mismatch")
        if r.get("_agent_skipped_kb", False):
            modes.append("scope_misroute")
        has_expected = bool(r.get("_expected_enums", []) or [])
        if r.get("_case_scope_kb_like", False) and has_expected:
            if r.get("_retrieval_gap_flag", False):
                modes.append("retrieval_gap")
            if r.get("_pruning_loss_flag", False):
                modes.append("pruning_loss")
            if r.get("_reranker_miss_flag", False):
                modes.append("reranker_miss")
        if r.get("_pool_content_gap_flag", False):
            modes.append("pool_content_gap")
        if r.get("_context_use_failure", False):
            modes.append("context_use_failure")
        # Use the BROAD hallucination flag here (any judge-listed claim).
        # The primary-failure classifier above uses the strict severe-only
        # variant, but the chip filter + co-occurrence heatmap consume
        # this list and benefit from seeing every flagged case.
        if r.get("_hallucinated_any", False):
            modes.append("hallucination")
        if r.get("_wrong_language", False):
            modes.append("language_drift")
        # Critical-zero veto applies whenever a weight-2 scorer is 0,
        # even if a more specific mode already fired — it's a stand-alone
        # diagnostic signal that the chip filter / co-occurrence matrix
        # benefits from seeing.
        if r.get("_critical_zero", False):
            modes.append("critical_score_zero")
        if not modes:
            if pd.notna(r.get("weighted_avg")) and r["weighted_avg"] >= PASS_THRESHOLD:
                modes.append("pass")
            else:
                modes.append("other_failure")
        return modes

    df["_failure_modes_all"] = df.apply(_all_modes, axis=1)
    return df


# ── Baseline loading (per-run comparison) ────────────────────────────────────
def load_baseline(path: Path) -> dict:
    """Load a prior judge checkpoint CSV and reduce it to the same shape the
    rendering layer expects for the "vs baseline" Δ chips.

    Re-runs ``enrich()`` + ``compute_summary_metrics()`` on the baseline so
    the displayed deltas use the *current* report's enrich rules and pass
    criterion — the baseline number can't drift from the current methodology.
    If you change a threshold or a failure-mode rule, the baseline's
    derived metrics move with it on the next report run.

    Returns a dict with the same keys ``render_html`` reads:
      ``label``         — derived from the file stem; used as the chip title prefix.
      ``metrics``       — ``{pass_rate_clean, dataset_recall, dataset_precision, rel2_mean}``.
      ``issue_counts``  — ``{failure_mode: count}`` for every Top-3 failure mode,
                          counted on the SAME filter the current run uses
                          (``failure_mode == X AND judge_pass == False AND clean``).
      ``n_clean``       — clean-denominator count for context.
      ``cases``         — per-``test_case_id`` lookup for future per-case
                          regression badges. Not used yet; populated so
                          downstream code can add a "was PASS, now FAIL"
                          chip later without re-loading the CSV.

    Returning ``None``-friendly: call sites pass the dict (or None when no
    baseline) directly into ``render_html(baseline=…)`` and ``_inline_delta_html``
    short-circuits to ``""`` when any half of the pair is missing.
    """
    df = read_checkpoint_csv(path)
    df = enrich(df)
    summary = compute_summary_metrics(df, df_all=df)
    metrics = {
        "pass_rate_clean":   summary.get("pass_rate_clean"),
        "dataset_recall":    summary.get("dataset_recall"),
        "dataset_precision": summary.get("dataset_precision"),
        "rel2_mean":         summary.get("rel2_mean"),
    }
    issue_counts = {
        entry["key"]: entry["n"]
        for entry in summary.get("top_failures_clean", [])
        if isinstance(entry, dict) and entry.get("key")
    }
    cases: dict[str, dict] = {}
    if "test_case_id" in df.columns:
        for _, row in df.iterrows():
            tcid = row.get("test_case_id")
            if not isinstance(tcid, str) or not tcid:
                continue
            cases[tcid] = {
                "judge_pass":            bool(row.get("pass")) if "pass" in df.columns else None,
                "weighted_avg":          _num_or_none(row.get("weighted_avg")),
                "failure_mode":          row.get("failure_mode") if isinstance(row.get("failure_mode"), str) else None,
                "expert_score":          _num_or_none(row.get("expert_score")),
                "enum_relevance_score":  _num_or_none(row.get("enum_relevance_score")),
            }
    return {
        "label":        Path(path).stem,
        "metrics":      metrics,
        "issue_counts": issue_counts,
        "n_clean":      summary.get("n_clean", 0),
        "cases":        cases,
    }


def _num_or_none(v):
    """Coerce to float, returning None for NaN / non-numeric so the
    per-case lookup dict stays JSON-friendly without leaking NaN."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


# ── HTML rendering ───────────────────────────────────────────────────────────
def _h(s) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return _html.escape(str(s))



def _case_payload(r: pd.Series, *, baseline_lookup: dict | None = None) -> dict:
    """Build the per-case dict that ships to the JS layer.

    When ``baseline_lookup`` is provided, this function also resolves the
    case's prior-run outcome (keyed by ``test_case_id``) and derives the
    four comparison flags rendered as detail-pane badges:

      - ``regression``         — passed in baseline, fails now (red).
      - ``persistent_failure`` — failed in both runs.
      - ``fixed``              — failed in baseline, passes now (green).
      - ``new_case``           — no entry in baseline (gray).

    When ``baseline_lookup`` is None, every flag is False and the
    ``baseline_*`` fields are None — the JS short-circuits and renders
    no banner. Same "if None, skip" pattern as the API report.
    """
    def g(c, default=""):
        return r[c] if c in r.index and not pd.isna(r[c]) else default
    def b(c, default=False):
        value = g(c, default)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"true", "1", "1.0", "yes"}
    # Parse the per-trace latency breakdown emitted by backfill_latency.py.
    lat_steps_raw = g("lat_steps_json", "")
    try:
        lat_steps = json.loads(lat_steps_raw) if lat_steps_raw else []
    except (json.JSONDecodeError, TypeError):
        lat_steps = []
    if not isinstance(lat_steps, list):
        lat_steps = []
    # Throttling (suspected-retry) detail — emitted by latency.py alongside
    # the step breakdown. Missing on older CSVs; treated as "no retries".
    lat_retries_raw = g("lat_retries_json", "")
    try:
        lat_retries = json.loads(lat_retries_raw) if lat_retries_raw else []
    except (json.JSONDecodeError, TypeError):
        lat_retries = []
    if not isinstance(lat_retries, list):
        lat_retries = []
    # Span-level errors emitted by backfill_span_errors.py. The trace-level
    # state is "OK" whenever the agent returned something, so without this
    # the report would silently hide ConnectTimeout / CancelledError / etc.
    span_errors_raw = g("span_errors_json", "")
    try:
        span_errors = json.loads(span_errors_raw) if span_errors_raw else []
    except (json.JSONDecodeError, TypeError):
        span_errors = []
    if not isinstance(span_errors, list):
        span_errors = []
    span_error_types_raw = g("span_error_types_json", "")
    try:
        span_error_types = json.loads(span_error_types_raw) if span_error_types_raw else []
    except (json.JSONDecodeError, TypeError):
        span_error_types = []
    if not isinstance(span_error_types, list):
        span_error_types = []
    return {
        "id": g("test_case_id"),
        "trace_id": g("trace_id"),
        "lat_total_ms": (float(r["lat_total_ms"])
                         if "lat_total_ms" in r.index and pd.notna(r.get("lat_total_ms"))
                         else None),
        "lat_steps": lat_steps,
        "lat_retry_overhead_ms": (float(r["lat_retry_overhead_ms"])
                                  if "lat_retry_overhead_ms" in r.index and pd.notna(r.get("lat_retry_overhead_ms"))
                                  else None),
        "lat_retry_call_count": (int(r["lat_retry_call_count"])
                                 if "lat_retry_call_count" in r.index and pd.notna(r.get("lat_retry_call_count"))
                                 else 0),
        "lat_retries": lat_retries,
        "has_span_error": b("trace_has_span_error"),
        "span_error_count": (int(r["span_error_count"])
                             if "span_error_count" in r.index and pd.notna(r.get("span_error_count"))
                             else 0),
        "span_error_types": span_error_types,
        "span_errors": span_errors,
        "scope": g("query_scope"),
        "last_agent": g("last_agent"),
        "rerank_empty": b("reranker_selected_empty"),
        "weighted_avg": float(r["weighted_avg"]) if pd.notna(r.get("weighted_avg")) else None,
        # Canonical pass verdict: matches df["pass"] in prepare_df — encodes
        # both weighted_avg ≥ PASS_THRESHOLD and the critical-zero veto.
        # JS uses this directly so the PASS/FAIL badge can't drift from the
        # Python pass-rate / failure-mode tables.
        "judge_pass": bool(r["pass"]) if "pass" in r.index and pd.notna(r.get("pass")) else None,
        "critical_zero": bool(r["_critical_zero"]) if "_critical_zero" in r.index else False,
        "expert_score": float(r["expert_score"]) if pd.notna(r.get("expert_score")) else None,
        "rel2_score": float(r["enum_relevance_score"]) if pd.notna(r.get("enum_relevance_score")) else None,
        "enum_recall": float(r["enum_recall"]) if pd.notna(r.get("enum_recall")) else None,
        "enum_precision": float(r["enum_precision"]) if pd.notna(r.get("enum_precision")) else None,
        "enum_f1": float(r["enum_f1"]) if pd.notna(r.get("enum_f1")) else None,
        "root_cause": g("root_cause_category"),
        "user_query": g("user_query"),
        "user_query_en": g("user_query_en"),
        "agent_response": g("agent_response"),
        "agent_response_en": g("agent_response_en"),
        "expected_response": g("expected_response"),
        "expected_response_en": g("expected_response_en"),
        "reranked_enum_ids": g("reranked_enum_ids"),
        # Cardinalities exposed for chip filters (over-selection, etc.).
        # Derived once on the Python side so the JS doesn't have to parse
        # the JSON list strings on every filter pass.
        "reranked_enum_count": (
            len(r["_reranked_enum_ids"])
            if "_reranked_enum_ids" in r.index
               and isinstance(r["_reranked_enum_ids"], list)
            else None
        ),
        "expected_enum_count": (
            len(r["_expected_enums"])
            if "_expected_enums" in r.index
               and isinstance(r["_expected_enums"], list)
            else None
        ),
        "expected_enums": g("expected_enums"),
        "post_prune_enum_ids": g("post_prune_enum_ids"),
        "reranker_raw_selected_ids": g("reranker_raw_selected_ids"),
        "reranker_valid_selected_ids": g("reranker_valid_selected_ids"),
        "reranker_invalid_selected_ids": g("reranker_invalid_selected_ids"),
        "reranker_unselected_context_ids": g("reranker_unselected_context_ids"),
        "reranker_selection_status": g("reranker_selection_status"),
        "reranker_selection_violations": g("reranker_selection_violations"),
        "missing_enums_in_candidate_pool": g("missing_enums_in_candidate_pool"),
        "missing_enums_not_in_pool": g("missing_enums_not_in_pool"),
        "extra_or_distracting_enums": g("extra_or_distracting_enums"),
        "optimal_enum_selection": g("optimal_enum_selection"),
        "dims": {
            d: {
                "score": int(r[f"{d}_score"]) if pd.notna(r.get(f"{d}_score")) else None,
                "reasoning": g(f"{d}_reasoning"),
            }
            for d in DIMENSION_WEIGHTS
        },
        "overall_explanation": g("overall_explanation"),
        "retrieved_pool_inadequacy_identified": b("retrieved_pool_inadequacy_identified"),
        "retrieved_pool_inadequacy_description": g("retrieved_pool_inadequacy_description"),
        "retrieval_improvement_suggestion": g("retrieval_improvement_suggestion"),
        "reranker_improvement_suggestion": g("reranker_improvement_suggestion"),
        "agent_improvement_suggestion": g("agent_improvement_suggestion"),
        "kb_improvement_suggestion": g("kb_improvement_suggestion"),
        "test_case_improvement_suggestion": g("test_case_improvement_suggestion"),
        "missing_facts": g("missing_facts"),
        "hallucinated_claims": g("hallucinated_claims"),
        "unavailable_facts_in_selected_context": g("unavailable_facts_in_selected_context"),
        "expected_answer_summary_with_optimal_context": (
            g("expected_answer_summary_with_optimal_context")
            or g("Expected_answer_summary_with_optimal_context")
        ),
        "agents_called": g("agents_called"),
        "tools_called": g("tools_called"),
        "case_scope": g("case_scope"),
        "categories_list": g("categories_list"),
        "expected_reference_looks_wrong": b("expected_reference_looks_wrong"),
        "expected_reference_issue_description": g("expected_reference_issue_description"),
        "trace_invariant_violations": g("trace_invariant_violations"),
        "failure_mode": g("failure_mode"),
        "failure_modes_all": (
            list(r["_failure_modes_all"])
            if "_failure_modes_all" in r.index
               and isinstance(r["_failure_modes_all"], list)
            else []
        ),
        # Deterministic test-set hygiene check (computed by the report,
        # not by the judge). List of {expected, kb_form, raw_expected}.
        "enum_naming_mismatches": (
            r["_enum_naming_mismatches"]
            if "_enum_naming_mismatches" in r.index
               and isinstance(r["_enum_naming_mismatches"], list)
            else []
        ),
        # Set when the row was filtered out of the analysis denominator.
        # null for in-scope KB rows. Surfaces as a tag in the
        # sidebar + case-detail title so colleagues can tell at a glance
        # that the row didn't contribute to any pass-rate / failure-mode
        # number, but is still inspectable.
        "excluded_reason": (
            "empty_user_query" if (b("_user_query_empty"))
            else (
                "non_kb_scope" if (g("case_scope") not in ("kb", "kb_and_api")
                                    and not b("_user_query_empty"))
                else None
            )
        ),
        # Baseline comparison fields. When no baseline was loaded these
        # are all None / False so the JS short-circuits and renders no
        # banner. See load_baseline() for the lookup shape.
        **_baseline_compare_fields(g("test_case_id"), r, baseline_lookup),
    }


def _baseline_compare_fields(tcid: str, r: pd.Series,
                              baseline_lookup: dict | None) -> dict:
    """Compute the per-case baseline-comparison subdict.

    Pulled out of ``_case_payload`` so the comparison logic is in one
    place and easy to extend (e.g. if we later want per-dimension deltas
    rendered in the case-detail dim-strip).

    Mirrors the API report's pattern: pass / fail booleans on both sides,
    plus the four mutually-exclusive flags ``regression`` / ``fixed`` /
    ``persistent_failure`` / ``new_case``. ``baseline_known`` lets the JS
    distinguish "no baseline loaded" from "case absent in baseline".
    """
    prev = baseline_lookup.get(str(tcid)) if (baseline_lookup and tcid) else None
    if prev is None:
        return {
            "baseline_loaded":     bool(baseline_lookup),
            "baseline_known":      False,
            "baseline_pass":       None,
            "baseline_weighted_avg": None,
            "baseline_failure_mode": None,
            "regression":          False,
            "persistent_failure":  False,
            "fixed":               False,
            "new_case":            bool(baseline_lookup),  # only "new" if baseline was loaded
        }
    prev_pass = prev.get("judge_pass")
    now_pass = bool(r["pass"]) if "pass" in r.index and pd.notna(r.get("pass")) else None
    # Comparison flags only fire when both sides have a definite verdict.
    can_compare = (prev_pass is not None) and (now_pass is not None)
    regression         = bool(can_compare and prev_pass and not now_pass)
    fixed              = bool(can_compare and (not prev_pass) and now_pass)
    persistent_failure = bool(can_compare and (not prev_pass) and (not now_pass))
    return {
        "baseline_loaded":     True,
        "baseline_known":      True,
        "baseline_pass":       bool(prev_pass) if prev_pass is not None else None,
        "baseline_weighted_avg": prev.get("weighted_avg"),
        "baseline_failure_mode": prev.get("failure_mode"),
        "regression":          regression,
        "persistent_failure":  persistent_failure,
        "fixed":               fixed,
        "new_case":            False,
    }


# ── Bins & contingency matrices for expert / Rel2 / judge comparison ────────
EXPERT_BINS = ["Low (≤3)", "Mid (4-6)", "Good (7-8)", "Excellent (9-10)"]
WAVG_BINS = ["Bad (<0.25)", "Weak (0.25-0.5)", "Mid (0.5-0.7)", "High (≥0.7)"]
REL2_BINS = ["Low (<0.5)", "Mid (0.5-0.8)", "High (≥0.8)"]
ENUM_F1_BINS = ["None (0)", "Weak (<0.5)", "Mid (0.5-0.99)", "Perfect (1.0)"]


def _bin_expert(v):
    if pd.isna(v): return None
    if v <= 3: return EXPERT_BINS[0]
    if v <= 6: return EXPERT_BINS[1]
    if v <= 8: return EXPERT_BINS[2]
    return EXPERT_BINS[3]


def _bin_wavg(v):
    if pd.isna(v): return None
    if v < 0.25: return WAVG_BINS[0]
    if v < 0.5: return WAVG_BINS[1]
    if v < PASS_THRESHOLD: return WAVG_BINS[2]
    return WAVG_BINS[3]


def _bin_rel2(v):
    if pd.isna(v): return None
    if v < 0.5: return REL2_BINS[0]
    if v < 0.8: return REL2_BINS[1]
    return REL2_BINS[2]


def _bin_enum_f1(v):
    if pd.isna(v): return None
    if v == 0: return ENUM_F1_BINS[0]
    if v < 0.5: return ENUM_F1_BINS[1]
    if v < 0.999: return ENUM_F1_BINS[2]
    return ENUM_F1_BINS[3]


def _heatmap(ct: pd.DataFrame, *, x_label: str, y_label: str, color_scale):
    """Plotly imshow heatmap with count-text annotations in each cell."""
    fig = px.imshow(
        ct.values, x=list(ct.columns), y=list(ct.index),
        text_auto=True, aspect="auto",
        color_continuous_scale=color_scale,
        labels=dict(x=x_label, y=y_label, color="count"),
    )
    _style_fig(fig, height=340)
    fig.update_layout(margin=dict(t=20, b=60, l=140, r=30),
                      coloraxis_colorbar=dict(tickfont=dict(color=GE_MUTED, size=10),
                                              thickness=10, len=0.8, outlinewidth=0))
    fig.update_traces(textfont=dict(family=GE_FONT, size=13, color=GE_TEXT))
    fig.update_xaxes(side="bottom", showgrid=False, linecolor=GE_GRID)
    fig.update_yaxes(showgrid=False, linecolor=GE_GRID)
    return fig


def _build_corr_figs(df: pd.DataFrame):
    """Two contingency heatmaps: expert×judge and Rel2×enum_F1."""
    out = {"expert": None, "rel2": None, "expert_pearson": None,
           "rel2_pearson": None, "n_expert": 0, "n_rel2": 0}

    if "expert_score" in df.columns:
        sub = df[df["expert_score"].notna() & df["weighted_avg"].notna()].copy()
        if len(sub):
            sub["__e"] = sub["expert_score"].apply(_bin_expert)
            sub["__w"] = sub["weighted_avg"].apply(_bin_wavg)
            ct = (pd.crosstab(sub["__e"], sub["__w"])
                    .reindex(index=EXPERT_BINS, columns=WAVG_BINS, fill_value=0))
            out["expert"] = _heatmap(
                ct, x_label="Judge weighted_avg",
                y_label="Expert score (1-10)", color_scale=GE_BLUE_SCALE,
            )
            out["expert_pearson"] = float(sub["expert_score"].corr(sub["weighted_avg"]))
            out["n_expert"] = len(sub)

    if "enum_relevance_score" in df.columns and "enum_f1" in df.columns:
        sub = df[df["enum_relevance_score"].notna() & df["enum_f1"].notna()].copy()
        if len(sub):
            sub["__r"] = sub["enum_relevance_score"].apply(_bin_rel2)
            sub["__f"] = sub["enum_f1"].apply(_bin_enum_f1)
            ct = (pd.crosstab(sub["__r"], sub["__f"])
                    .reindex(index=REL2_BINS, columns=ENUM_F1_BINS, fill_value=0))
            out["rel2"] = _heatmap(
                ct, x_label="Judge enum_F1",
                y_label="Rel2 score (0-1)", color_scale=GE_GREEN_SCALE,
            )
            out["rel2_pearson"] = float(sub["enum_relevance_score"].corr(sub["enum_f1"]))
            out["n_rel2"] = len(sub)

    return out


def _build_figs(df: pd.DataFrame, reranker_miss: Counter, retriever_miss: Counter):
    n = len(DIMENSION_WEIGHTS)
    ncols = (n + 1) // 2
    fig_dim = make_subplots(
        rows=2, cols=ncols,
        subplot_titles=list(DIMENSION_WEIGHTS.keys()), shared_yaxes=True,
        horizontal_spacing=0.06, vertical_spacing=0.22,
    )
    for i, dim in enumerate(DIMENSION_WEIGHTS):
        row = (i // ncols) + 1
        col = (i % ncols) + 1
        counts = df[f"{dim}_score"].value_counts().reindex([0, 1, 2]).fillna(0)
        fig_dim.add_trace(
            go.Bar(
                x=[0, 1, 2], y=counts.values, showlegend=False,
                marker=dict(color=[GE_RED, GE_YELLOW, GE_GREEN],
                            line=dict(width=0), cornerradius=5),
            ),
            row=row, col=col,
        )
    _style_fig(fig_dim, height=460)
    fig_dim.update_layout(margin=dict(t=40, b=30, l=30, r=20), bargap=0.35)
    for ann in fig_dim.layout.annotations:
        ann.font = dict(family=GE_FONT, size=11, color=GE_MUTED)
    fig_dim.update_xaxes(tickvals=[0, 1, 2])

    # weighted_avg histogram rendered on the Notes tab next to the Pass
    # rate card. The interactive threshold slider (wired up in JS) drives a
    # dashed vertical reference line on top of the static red
    # PASS_THRESHOLD line so users can dry-run "what if the cut-off were
    # at X?" — the slider's count read-out filters CASES directly and
    # therefore matches the headline Pass-rate definition (wa ≥ X AND
    # NOT critical_zero). Bars stay single-colour so the slider line is
    # the only visual cue moving with user input.
    _wa_series = df["weighted_avg"].dropna()
    if len(_wa_series) > 0:
        _bin_edges = np.linspace(0.0, max(1.0, float(_wa_series.max())), 21)
        _counts, _ = np.histogram(_wa_series, bins=_bin_edges)
        _centers = (_bin_edges[:-1] + _bin_edges[1:]) / 2.0
        fig_hist = go.Figure(go.Bar(
            x=_centers, y=_counts,
            marker=dict(color=GE_BLUE, line=dict(width=0)),
            width=(_bin_edges[1] - _bin_edges[0]) * 0.96,
            hovertemplate="weighted_avg ≈ %{x:.2f}<br>cases: %{y}<extra></extra>",
        ))
    else:
        fig_hist = go.Figure()
    # Static red line = the report's PASS_THRESHOLD (always drawn at 0.7).
    fig_hist.add_vline(x=PASS_THRESHOLD, line_color=GE_RED,
                       line_width=1, annotation_text=f"pass ≥ {PASS_THRESHOLD}",
                       annotation_position="top right",
                       annotation_font=dict(color=GE_RED, size=11))
    # Dashed orange line driven by the slider — starts at PASS_THRESHOLD
    # so the two lines coincide at page load. The JS handler moves this
    # one via Plotly.relayout('shapes[1].x0/x1', sliderValue).
    fig_hist.add_vline(x=PASS_THRESHOLD, line_color="#e57b00",
                       line_width=2, line_dash="dash")
    _style_fig(fig_hist, height=340)
    fig_hist.update_layout(margin=dict(t=30, b=40, l=50, r=20), bargap=0.04,
                           xaxis_title="weighted_avg", yaxis_title="count")

    fig_rc = None
    if "root_cause_category" in df.columns:
        rc = df["root_cause_category"].fillna("(missing)").value_counts().reset_index()
        rc.columns = ["root_cause", "count"]
        # Horizontal bars: categories on the y-axis (easier to read long names,
        # no tick-angle rotation needed). Largest count lands at the top.
        order = rc.sort_values("count", ascending=True)
        fig_rc = px.bar(order, x="count", y="root_cause", text="count",
                         orientation="h")
        fig_rc.update_traces(
            marker=dict(color=GE_BLUE, line=dict(width=0), cornerradius=6),
            textposition="outside", cliponaxis=False,
            textfont=dict(family=GE_FONT, size=11, color=GE_TEXT),
        )
        _style_fig(fig_rc, height=max(280, 34 * len(rc) + 120))
        fig_rc.update_layout(bargap=0.35,
                             margin=dict(t=30, b=40, l=200, r=60),
                             xaxis_title="count", yaxis_title="")
        if len(rc):
            fig_rc.update_xaxes(range=[0, int(rc["count"].max()) * 1.15 + 1])
        fig_rc.update_yaxes(tickfont=dict(family="monospace", size=11, color=GE_TEXT),
                              automargin=True)

    fig_missed = None
    if reranker_miss or retriever_miss:
        # Rank enums by total misses (descending) so the most-missed sits at the top.
        totals = Counter()
        for e, n in reranker_miss.items(): totals[e] += n
        for e, n in retriever_miss.items(): totals[e] += n
        top = [e for e, _ in totals.most_common(20)]
        # Order so bars read top→bottom from most-missed to least-missed.
        top_order = list(reversed(top))
        rows = []
        for e in top:
            rows.append({"enum_id": e, "cause": "reranker miss (in pool, not picked)",
                         "count": reranker_miss.get(e, 0)})
            rows.append({"enum_id": e, "cause": "retriever miss (not in pool)",
                         "count": retriever_miss.get(e, 0)})
        df_miss = pd.DataFrame(rows)
        fig_missed = px.bar(
            df_miss, x="count", y="enum_id", color="cause", orientation="h",
            barmode="stack", category_orders={"enum_id": top_order},
            color_discrete_map={
                "reranker miss (in pool, not picked)": GE_YELLOW,
                "retriever miss (not in pool)": GE_RED,
            },
            text="count",
        )
        _style_fig(fig_missed, height=max(340, 28 * len(top) + 160))
        fig_missed.update_traces(
            marker=dict(line=dict(width=0), cornerradius=4),
            hovertemplate="<b>%{y}</b><br>%{x} × %{fullData.name}<extra></extra>",
            cliponaxis=False,
        )
        # Inside per-segment labels only when the segment is ≥ 2 — otherwise
        # they overlap the total label at the right edge (the bar is too small).
        for tr in fig_missed.data:
            tr.text = [str(v) if v >= 2 else "" for v in tr.x]
            tr.texttemplate = "%{text}"
            tr.textposition = "inside"
            tr.insidetextanchor = "middle"
            tr.textfont = dict(family=GE_FONT, size=10, color="#ffffff")
        # Totals-right annotation
        totals_by_enum = {e: totals[e] for e in top}
        for e in top:
            fig_missed.add_annotation(
                x=totals_by_enum[e], y=e,
                text=f"<b>{totals_by_enum[e]}</b>",
                showarrow=False, xanchor="left", yanchor="middle",
                xshift=6, font=dict(family=GE_FONT, size=11, color=GE_TEXT),
            )
        # Legend above the plot with generous top padding so it never collides
        # with the first row's y-axis label or bar segment.
        fig_missed.update_layout(
            bargap=0.3,
            margin=dict(t=90, b=40, l=220, r=70),
            xaxis_title="count", yaxis_title="",
            legend=dict(orientation="h", yanchor="bottom", y=1.04,
                         xanchor="left", x=0, title_text="",
                         font=dict(family=GE_FONT, size=11, color=GE_TEXT),
                         bgcolor="rgba(255,255,255,0)"),
        )
        fig_missed.update_yaxes(tickfont=dict(family="monospace", size=11, color=GE_TEXT),
                                  automargin=True)
        if totals_by_enum:
            fig_missed.update_xaxes(range=[0, max(totals_by_enum.values()) * 1.15])

    return fig_dim, fig_hist, fig_rc, fig_missed


def _build_failure_mode_cooccurrence_fig(df: pd.DataFrame):
    """Symmetric N×N heatmap of failure-mode co-occurrence — cell (i, j) is
    the number of cases where both mode_i and mode_j fired (using
    _failure_modes_all, not just the priority winner). Shown for every mode
    from `test_set_defect` onwards in priority order ("pass" is dropped
    because co-occurring with itself isn't a failure pattern). The diagonal
    is the number of cases that mode fired in at all.

    Click a cell → filter the Test Cases tab to rows whose
    failure_modes_all contains both clicked modes (handled in JS via the
    customdata on each cell)."""
    if "_failure_modes_all" not in df.columns:
        return None
    if "test_set_defect" in FAILURE_MODES:
        start_idx = FAILURE_MODES.index("test_set_defect")
        modes = list(FAILURE_MODES[start_idx:])
    else:
        modes = [m for m in FAILURE_MODES if m != "pass"]
    if not modes:
        return None

    sets = df["_failure_modes_all"].apply(
        lambda v: set(v) if isinstance(v, list) else set()
    )
    n_modes = len(modes)
    z = [[0] * n_modes for _ in range(n_modes)]
    for i, mi in enumerate(modes):
        for j, mj in enumerate(modes):
            z[i][j] = int(sets.apply(lambda s: (mi in s) and (mj in s)).sum())
    if all(v == 0 for row in z for v in row):
        return None

    labels = [FAILURE_MODE_LABEL.get(m, m) for m in modes]
    text = [[str(v) if v else "" for v in row] for row in z]
    # Per-cell customdata = [row_mode_key, col_mode_key]; the JS click
    # handler filters CASES by intersecting both modes against
    # failure_modes_all so we don't have to ship every per-cell ID list.
    customdata = [[[modes[i], modes[j]] for j in range(n_modes)]
                   for i in range(n_modes)]

    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels,
        colorscale=GE_BLUE_SCALE,
        text=text, texttemplate="%{text}",
        textfont=dict(family=GE_FONT, size=10, color=GE_TEXT),
        hovertemplate=("<b>%{y}</b> AND <b>%{x}</b><br>%{z} cases"
                        "<br><i>click to filter</i><extra></extra>"),
        colorbar=dict(tickfont=dict(color=GE_MUTED, size=10),
                       thickness=10, len=0.8, outlinewidth=0,
                       title=dict(text="cases", font=dict(color=GE_MUTED, size=10))),
        customdata=customdata,
        xgap=1, ygap=1,
    ))
    _style_fig(fig, height=max(440, 36 * n_modes + 200))
    fig.update_layout(margin=dict(t=40, b=140, l=180, r=40))
    fig.update_xaxes(side="bottom", showgrid=False, linecolor=GE_GRID,
                       tickangle=-30, automargin=True,
                       tickfont=dict(family=GE_FONT, size=11, color=GE_TEXT))
    fig.update_yaxes(showgrid=False, linecolor=GE_GRID, autorange="reversed",
                       automargin=True,
                       tickfont=dict(family=GE_FONT, size=11, color=GE_TEXT))
    return fig


def _enum_count_distribution_table_html(df: pd.DataFrame) -> str:
    """Per-case distribution of |expected_enums| vs |reranked_enum_ids| as a
    table — easier to read the actual counts than the overlapping bar
    chart it replaces. Restricted to KB-routed cases (query_scope == 'kb').
    Each row: ENUM-count value, # cases with that many expected, # cases
    with that many selected, and the same numbers as % of n.

    Rows where the ENUM-count is > 6 (the "over-selection" band — the
    reranker has a YAML cap of ≤ 6 ENUMs in the typical configuration)
    are rendered in bold red so they're easy to spot. The Selected ·
    cases cell on every row is clickable: it filters the Test Cases tab
    to exactly those rows. A summary footer cell links to the cumulative
    "> 6 selected" count.
    """
    needed = {"_expected_enums", "_reranked_enum_ids", "query_scope"}
    if not needed.issubset(df.columns):
        return "<p class='placeholder'>no expected/selected ENUM data</p>"
    qs = df["query_scope"].fillna("").astype(str)
    sub = df[qs.eq("kb")].copy()
    if sub.empty:
        return "<p class='placeholder'>no KB-routed cases in this run.</p>"
    sub["_expected_len"] = sub["_expected_enums"].apply(
        lambda v: len(v) if isinstance(v, list) else 0
    )
    sub["_reranked_len"] = sub["_reranked_enum_ids"].apply(
        lambda v: len(v) if isinstance(v, list) else 0
    )
    expected_lens = sub["_expected_len"]
    reranked_lens = sub["_reranked_len"]
    has_tid = "test_case_id" in sub.columns
    n = len(sub)
    max_count = int(max(int(expected_lens.max() or 0), int(reranked_lens.max() or 0)))
    # Over-selection threshold (inclusive). Selected counts strictly above
    # this are flagged: the reranker's standing instruction is "pick the
    # minimum set sufficient to answer", and a typical cap of 6 — so > 6
    # almost always means "padded the answer with noise". The constant is
    # exposed on the chip filter (Test Cases tab) too; keep them in sync.
    OVERSEL_K = 6
    rows_html = ""
    for k in range(max_count + 1):
        e_n = int((expected_lens == k).sum())
        r_n = int((reranked_lens == k).sum())
        e_pct = _fmt_pct(e_n / n) if n else "–"
        r_pct = _fmt_pct(r_n / n) if n else "–"
        is_oversel = k > OVERSEL_K
        # Selected · cases cell — clickable filter into Test Cases tab.
        if has_tid and r_n > 0:
            r_ids = sub.loc[reranked_lens == k, "test_case_id"].astype(str).tolist()
            r_link = _ids_filter_link(
                r_ids, f"selected ENUMs = {k}",
                classes="judge-eval-link", inner=str(r_n),
            )
        else:
            r_link = str(r_n)
        # Over-selection band (k > OVERSEL_K): tint the entire row light
        # red so the eye lands on it immediately. Cell text stays neutral —
        # the row colour is the signal.
        row_cls = " class='enum-count-oversel'" if is_oversel else ""
        rows_html += (
            f"<tr{row_cls}>"
            f"<td style='text-align:right;font-variant-numeric:tabular-nums'>{k}</td>"
            f"<td style='text-align:right'>{e_n}</td>"
            f"<td style='text-align:right;color:#5c7999'>{e_pct}</td>"
            f"<td style='text-align:right'>{r_link}</td>"
            f"<td style='text-align:right;color:#5c7999'>{r_pct}</td>"
            "</tr>"
        )
    # Footer row: cumulative count of cases with > OVERSEL_K selected ENUMs.
    # Single click drills into all over-selected cases at once.
    oversel_mask  = reranked_lens > OVERSEL_K
    oversel_n     = int(oversel_mask.sum())
    oversel_pct   = _fmt_pct(oversel_n / n) if n else "–"
    if has_tid and oversel_n > 0:
        oversel_ids = sub.loc[oversel_mask, "test_case_id"].astype(str).tolist()
        oversel_link = _ids_filter_link(
            oversel_ids, f"selected ENUMs > {OVERSEL_K}",
            classes="judge-eval-link", inner=str(oversel_n),
        )
    else:
        oversel_link = str(oversel_n)
    # Footer row: same light-red tint as the over-selection rows so the
    # rollup line reads as part of the band. enum-count-footer just
    # bumps the font-weight; enum-count-oversel carries the colour.
    footer_html = (
        "<tr class='enum-count-oversel enum-count-footer'>"
        f"<td style='text-align:right'>&gt; {OVERSEL_K}</td>"
        "<td style='text-align:right;color:#5c7999'>—</td>"
        "<td style='text-align:right;color:#5c7999'>—</td>"
        f"<td style='text-align:right'>{oversel_link}</td>"
        f"<td style='text-align:right;color:#5c7999'>{oversel_pct}</td>"
        "</tr>"
    )
    mean_e = float(expected_lens.mean()) if n else float("nan")
    mean_r = float(reranked_lens.mean()) if n else float("nan")
    return (
        f"<p style='font-size:12px;color:#5c7999;margin-bottom:8px'>"
        f"n = <strong>{n}</strong> KB-routed cases. "
        f"Mean expected = <strong>{mean_e:.2f}</strong>, "
        f"mean selected = <strong>{mean_r:.2f}</strong>. "
        f"<span style='background:#fdecea;padding:1px 6px;border-radius:3px'>Light-red rows</span> "
        f"mark over-selection (&gt; {OVERSEL_K} selected ENUMs); "
        f"click any Selected count to drill into those cases on the Test Cases tab.</p>"
        "<table class='tbl funnel-tbl'>"
        "<thead><tr>"
        "<th style='text-align:right'>ENUMs / case</th>"
        "<th style='text-align:right'>Expected · cases</th>"
        "<th style='text-align:right'>%</th>"
        "<th style='text-align:right'>Selected · cases</th>"
        "<th style='text-align:right'>%</th>"
        "</tr></thead>"
        f"<tbody>{rows_html}{footer_html}</tbody></table>"
    )


def _build_dim_heatmap(df: pd.DataFrame):
    """21×21 co-occurrence heatmap of (dimension, score) pairs.

    Both axes list every ``<dimension>=<score>`` combination
    (7 dims × 3 scores = 21 labels by default). The cell at (row=A, col=B)
    is the number of test cases where row A's condition AND col B's condition
    are both true. Diagonal cells = simple count of that single pair.

    Click any cell in the rendered chart to filter the Test Cases tab; the
    JS handler reads the labels and pushes them into ``activeFilters.dim_pairs``.
    """
    # Re-order so query_clarity and language_compliance sit next to each other
    # at the start (they're both meta-axes about the query / output language;
    # easier to read them as a pair than split across the matrix).
    _dim_order = ["query_clarity", "language_compliance",
                   "selection_semantic_relevance", "selected_context_sufficiency",
                   "optimal_retrieved_context_adequacy", "answer_expected_alignment",
                   "answer_groundedness"]
    dims = [d for d in _dim_order if d in DIMENSION_WEIGHTS and f"{d}_score" in df.columns]
    # Fallback: append any DIMENSION_WEIGHTS entries the order list missed.
    for d in DIMENSION_WEIGHTS:
        if d not in dims and f"{d}_score" in df.columns:
            dims.append(d)
    if not dims:
        return None
    pairs = [(d, s) for d in dims for s in (0, 1, 2)]
    labels = [f"{d}={s}" for d, s in pairs]
    n = len(pairs)
    masks = {key: (df[f"{key[0]}_score"] == key[1]) for key in pairs}
    z = [[0] * n for _ in range(n)]
    for i, ki in enumerate(pairs):
        for j, kj in enumerate(pairs):
            z[i][j] = int((masks[ki] & masks[kj]).sum())
    text = [[str(v) if v else "" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels,
        colorscale=GE_BLUE_SCALE,
        text=text, texttemplate="%{text}",
        textfont=dict(family=GE_FONT, size=9, color=GE_TEXT),
        hovertemplate="<b>%{y}</b> &amp; <b>%{x}</b><br>%{z} test cases"
                       "<br><i>click to filter</i><extra></extra>",
        colorbar=dict(tickfont=dict(color=GE_MUTED, size=10),
                       thickness=10, len=0.8, outlinewidth=0,
                       title=dict(text="count", font=dict(color=GE_MUTED, size=10))),
        xgap=1, ygap=1,
    ))
    _style_fig(fig, height=max(560, 26 * n + 240))
    fig.update_layout(margin=dict(t=40, b=120, l=40, r=80),
                       autosize=True, width=None)

    cell_positions = list(range(n))
    # X-axis: vertical (90°) score numbers at every cell; dim names rendered
    # separately as −45° annotations below the score row so each dimension
    # appears once per group without forcing the score numbers to tilt too.
    score_ticktext = [str(s) for d, s in pairs]
    fig.update_xaxes(side="bottom", showgrid=False, linecolor=GE_GRID,
                       tickmode="array", tickvals=cell_positions,
                       ticktext=score_ticktext,
                       tickangle=0, constrain="domain",
                       tickfont=dict(family="monospace", size=10, color=GE_TEXT),
                       automargin=True)
    # Y-axis: keep the dim name embedded on the middle (=1) tick (horizontal
    # text doesn't have the same readability problem as the rotated x-axis).
    y_ticktext = [(f"{d} = {s}" if s == 1 else str(s)) for d, s in pairs]
    fig.update_yaxes(showgrid=False, linecolor=GE_GRID, autorange="reversed",
                       tickmode="array", tickvals=cell_positions,
                       ticktext=y_ticktext,
                       constrain="domain",
                       tickfont=dict(family="monospace", size=10, color=GE_TEXT),
                       automargin=True)

    # X-axis dimension names: very shallow tilt (−15°) — almost horizontal,
    # so the long names take little vertical space and don't overflow the
    # bottom margin. Centred under each group, placed below the score row.
    for i, dim in enumerate(dims):
        center = 3 * i + 1
        fig.add_annotation(
            x=center, y=0, xref="x", yref="paper",
            text=dim, showarrow=False,
            textangle=-15,
            font=dict(family="monospace", size=11, color=GE_TEXT),
            xanchor="right", yanchor="top",
            xshift=40, yshift=-22,
        )

    # Dotted separator lines (vertical / horizontal, no angled segment).
    line_style = dict(color="#3f5b7b", dash="2px,3px", width=1.4)
    for i in range(1, len(dims)):
        boundary = 3 * i - 0.5
        # Vertical: spans the plot AND extends straight down into the label
        # area (passes between the vertical score numbers, not through them).
        fig.add_shape(type="line", xref="x", yref="paper",
                      x0=boundary, x1=boundary, y0=-0.06, y1=1,
                      line=line_style, layer="above")
        # Horizontal extending from the left tick-label area into the plot
        fig.add_shape(type="line", xref="paper", yref="y",
                      x0=-0.20, x1=1, y0=boundary, y1=boundary,
                      line=line_style, layer="above")
    return fig


# ── Summary-tab computations ────────────────────────────────────────────────
def compute_summary_metrics(df: pd.DataFrame, df_all: pd.DataFrame) -> dict:
    """All numbers shown on the Summary tab. Nothing is hard-coded here —
    every value is derived from the input dataframe so the report stays in
    sync with whatever checkpoint it was given.

    ``df`` is the KB subset (empty user_query rows already removed).
    ``df_all`` is the original checkpoint, used only for the "excluded
    empty queries" tally at the top of the report.
    """
    n_total       = int(len(df_all))
    empty_q_series = df_all.get(
        "_user_query_empty", pd.Series([False]*len(df_all), index=df_all.index))
    n_empty_query = int(empty_q_series.sum())
    # case_scope is no longer a Stage 1 exclusion — non-KB cases flow into
    # the failure-mode classifier like every other row. We keep the keys
    # below at zero / empty so any downstream consumer that still reads
    # them continues to render cleanly.
    n_excluded_scope = 0
    excluded_empty_ids = (df_all.loc[empty_q_series, "test_case_id"]
                            .astype(str).tolist()
                          if "test_case_id" in df_all.columns else [])
    excluded_scope_ids: list[str] = []
    n_eval        = int(len(df))

    # Pass = the canonical judge verdict (df["pass"] from prepare_df):
    #     weighted_avg ≥ PASS_THRESHOLD  AND  no weight-2 scorer at 0.
    # This matches the per-case PASS/FAIL badge and the Pass|Fail chip
    # filter exactly, so the headline number always equals the chip-filter
    # row count. It is INTENTIONALLY a looser definition than
    # `failure_mode == "pass"` — a case can be a judge-pass even if a
    # higher-priority failure mode (test_set_defect, retrieval_gap,
    # hallucination, etc.) also fired. The Failure-modes table below
    # remains keyed on the priority cascade so it still attributes each
    # case to a single primary cause; the headline answers the narrower
    # "did the judge pass this case?" question.
    pass_mask     = df.get("pass", pd.Series([False]*len(df), index=df.index)).astype(bool)
    n_pass        = int(pass_mask.sum())
    pass_rate_all = (n_pass / n_eval) if n_eval else float("nan")

    defect_mask   = (
        df.get("_gold_defect", pd.Series([False]*len(df), index=df.index))
        | df.get("_case_scope_test_defect", pd.Series([False]*len(df), index=df.index))
        | df.get("_naming_mismatch", pd.Series([False]*len(df), index=df.index))
    )
    n_defect      = int(defect_mask.sum())
    n_clean       = n_eval - n_defect
    # On the new definition, a defect case CAN be a judge-pass (e.g. the
    # judge agreed with the agent but the gold reference was wrong). So
    # n_pass_clean is no longer trivially equal to n_pass — recompute it
    # by intersecting the judge-pass mask with the non-defect subset.
    n_pass_clean  = int((pass_mask & ~defect_mask).sum())
    pass_rate_clean = (n_pass_clean / n_clean) if n_clean else float("nan")

    # case_scope distribution shown in the Doc tab, computed on the
    # post-empty-query frame so the reader sees what *was* excluded by the
    # in-scope filter (api / out_of_scope / ambiguous) alongside the kept
    # categories (kb / kb_and_api).
    cs_counts = (df_all.loc[~empty_q_series, "case_scope"].fillna("").astype(str).value_counts().to_dict()
                  if "case_scope" in df_all.columns else {})

    # Failure mode roll-up. Two groups with different denominators:
    #   1. Test-set group (test_set_defect, enum_name_mismatch) — % of
    #      n_eval. These rows quantify how much of the analyzed sample is
    #      unusable ground truth.
    #   2. Valid-cases group (everything else, including pass) — % of
    #      n_clean (= n_eval − test_set_defect − enum_name_mismatch).
    #      Once defects are excluded, percentages reflect agent-side
    #      performance on the trustworthy subset, matching the Top-3
    #      failure-reasons card AND the "Pass rate · excl. test-set
    #      issues" headline card.
    # The renderer inserts an "All valid cases" denominator row between the
    # two groups so the switch is explicit.
    fm_counts = df["failure_mode"].value_counts().to_dict() if "failure_mode" in df.columns else {}
    test_set_group = ("test_set_defect", "enum_name_mismatch")
    valid_group_order = ("pass", "scope_misroute", "retrieval_gap", "pruning_loss",
                          "reranker_miss", "pool_content_gap", "context_use_failure",
                          "hallucination", "language_drift",
                          "critical_score_zero", "other_failure")
    fm_rows = []
    for fm in test_set_group + valid_group_order:
        n = int(fm_counts.get(fm, 0))
        denom = n_eval if fm in test_set_group else n_clean
        pct = (n / denom) if denom else float("nan")
        ids = df.loc[df["failure_mode"] == fm, "test_case_id"].astype(str).tolist() \
            if ("test_case_id" in df.columns and "failure_mode" in df.columns) else []
        fm_rows.append({
            "key":   fm,
            "label": FAILURE_MODE_LABEL[fm],
            "owner": FAILURE_MODE_OWNER[fm],
            "info":  FAILURE_MODE_INFO[fm],
            "n":     n,
            "pct":   pct,
            "ids":   ids,
        })

    # Funnel — micro-averaged expected-ENUM recall at each pipeline stage.
    # Mask matches KB Recall / KB Precision / Rel2 exactly:
    #   query_scope == 'kb' AND NOT _naming_mismatch.
    # ENUM-naming-mismatch cases are excluded because the upstream gold
    # ENUM IDs use a different naming convention than the KB — that's a
    # test-set issue, not an agent failure, and counting them here would
    # drag the agent's metrics down for something the agent can't fix.
    qs_kb_route = df.get("query_scope", pd.Series([""]*len(df), index=df.index)) \
                    .fillna("").astype(str).eq("kb")
    naming_mismatch_mask = df.get("_naming_mismatch",
        pd.Series([False]*len(df), index=df.index)).astype(bool)
    funnel_mask = qs_kb_route & ~naming_mismatch_mask
    funnel_sub = df[funnel_mask]
    # Micro-averaged recall AND precision per stage: sum TPs, |expected|,
    # and |stage selection| across cases first, then divide. Matches the
    # KB Recall + KB Precision headlines exactly at the reranked stage —
    # the funnel and the headlines speak the same units. Recall trends
    # down through the pipeline (you can only lose gold ENUMs as the set
    # shrinks); precision trends up (later stages drop noise).
    if len(funnel_sub):
        total_expected = sum(len(e or []) for e in funnel_sub["_expected_enums"])
        def _stage_micro(stage_col: str) -> tuple[float, float]:
            tp = sum(
                len(set(e or []) & set(s or []))
                for e, s in zip(funnel_sub["_expected_enums"], funnel_sub[stage_col])
            )
            total_in_stage = sum(len(s or []) for s in funnel_sub[stage_col])
            recall    = (tp / total_expected) if total_expected else float("nan")
            precision = (tp / total_in_stage) if total_in_stage else float("nan")
            return (recall, precision)
        funnel = {
            "n": int(len(funnel_sub)),
            "stages": [
                ("pre-prune",  *_stage_micro("_pre_prune_enum_ids")),
                ("post-prune", *_stage_micro("_post_prune_enum_ids")),
                ("reranked",   *_stage_micro("_reranked_enum_ids")),
            ],
        }
    else:
        funnel = {
            "n": 0,
            "stages": [("pre-prune", float("nan"), float("nan")),
                        ("post-prune", float("nan"), float("nan")),
                        ("reranked", float("nan"), float("nan"))],
        }

    # Optimal vs final selection — judge's optimal_enum_selection ∩ reranked_enum_ids.
    opt_series = df.get("_recall_optimal_vs_final",
                         pd.Series([np.nan]*len(df), index=df.index))
    opt_defined = opt_series.dropna()
    opt_vs_final = {
        "n":    int(len(opt_defined)),
        "mean": float(opt_defined.mean()) if len(opt_defined) else float("nan"),
        # Bin for histogram-style distribution (computed, not hard-coded list).
        "buckets": [
            ("0%",         int(((opt_defined == 0)).sum())),
            ("(0, 50%)",   int(((opt_defined > 0) & (opt_defined < 0.5)).sum())),
            ("[50, 100%)", int(((opt_defined >= 0.5) & (opt_defined < 1)).sum())),
            ("100%",       int((opt_defined >= 0.999).sum())),
        ],
    }

    # Per-dimension card data (label, mean, counts at 0/1/2, ids per bucket).
    dim_data = []
    for dim in DIMENSION_WEIGHTS:
        col = f"{dim}_score"
        if col not in df.columns:
            continue
        scores = pd.to_numeric(df[col], errors="coerce")
        n_def = int(scores.notna().sum())
        bucket_ids = {
            s: df.loc[scores == s, "test_case_id"].astype(str).tolist()
               if "test_case_id" in df.columns else []
            for s in (0, 1, 2)
        }
        dim_data.append({
            "key":     dim,
            "label":   _dim_plain_label(dim),
            "owner":   _dim_owner(dim),
            "mean":    float(scores.mean()) if n_def else float("nan"),
            "n":       n_def,
            "counts":  {s: int((scores == s).sum()) for s in (0, 1, 2)},
            "ids":     bucket_ids,
        })

    # Rel2 is the semantic-overlap metric between expected_enums and the
    # system's selected ENUM IDs. We use the *adjusted* series produced by
    # enrich() — for rows with a detected naming mismatch the upstream
    # value (a false zero against the raw string) is replaced with a
    # normalized-recall calculation. Other rows keep the upstream value.
    # Restricted to cases where the reranker actually ran (query_scope == 'kb').
    # ── Expected vs final ENUM count comparison ──────────────────────────────
    # Limited to in-scope rows with a non-empty expected_enums (otherwise the
    # comparison is undefined). Buckets are mutually exclusive — a given row
    # falls into exactly one. The mean(expected) vs mean(selected) numbers
    # show whether the reranker has a systemic over- or under-selection bias.
    expected_lens = df.get("_expected_enums", pd.Series([[]]*len(df), index=df.index)).apply(
        lambda v: len(v) if isinstance(v, list) else 0
    )
    reranked_lens = df.get("_reranked_enum_ids", pd.Series([[]]*len(df), index=df.index)).apply(
        lambda v: len(v) if isinstance(v, list) else 0
    )
    cmp_mask = expected_lens > 0
    cmp_n = int(cmp_mask.sum())
    if cmp_n:
        e = expected_lens[cmp_mask]
        r = reranked_lens[cmp_mask]
        cmp_mean_expected = float(e.mean())
        cmp_mean_selected = float(r.mean())
        def _bucket(re_pair):
            rv, ev = re_pair
            if rv == 0:
                return "Selected = 0 (empty)"
            if rv < ev:
                return "Selected < Expected"
            if rv == ev:
                return "Selected = Expected"
            return "Selected > Expected"
        labels_in_order = ("Selected = 0 (empty)", "Selected < Expected",
                            "Selected = Expected", "Selected > Expected")
        cats = pd.Series([_bucket((rv, ev)) for rv, ev in zip(r, e)],
                          index=e.index)
        cmp_buckets = []
        for lbl in labels_in_order:
            idx = cats.index[cats == lbl]
            ids = df.loc[idx, "test_case_id"].astype(str).tolist() \
                if "test_case_id" in df.columns else []
            cmp_buckets.append({"label": lbl, "n": int(len(idx)), "ids": ids})
    else:
        cmp_mean_expected = float("nan")
        cmp_mean_selected = float("nan")
        cmp_buckets = []

    # ── Dataset-level (micro-averaged) recall on ENUM selection ──────────────
    # Aggregates true positives, expected, and selected across all qualifying
    # cases — gives a single "how well did the pipeline retrieve the gold
    # ENUMs over the whole run" number, distinct from the per-case mean.
    # Basis: cases the system routed through the KB pipeline (query_scope
    # == 'kb') AND that don't have an ENUM-naming-mismatch. Naming-
    # mismatch cases are test-set issues (gold ENUM IDs use a different
    # naming convention than the KB), so they shouldn't drag down agent
    # metrics for something the agent can't fix.
    qs_kb_mask = df.get("query_scope", pd.Series([""]*len(df), index=df.index)) \
                    .fillna("").astype(str).eq("kb")
    recall_mask = qs_kb_mask & ~naming_mismatch_mask
    dataset_recall_n = int(recall_mask.sum())
    if dataset_recall_n:
        sub_e = df.loc[recall_mask, "_expected_enums"]
        sub_r = df.loc[recall_mask, "_reranked_enum_ids"]
        tp = sum(len(set(e or []) & set(r or [])) for e, r in zip(sub_e, sub_r))
        total_expected = sum(len(e or []) for e in sub_e)
        total_reranked = sum(len(r or []) for r in sub_r)
        dataset_recall    = (tp / total_expected) if total_expected else float("nan")
        dataset_precision = (tp / total_reranked) if total_reranked else float("nan")
        if (dataset_recall == dataset_recall  # not NaN
            and dataset_precision == dataset_precision
            and (dataset_recall + dataset_precision)):
            dataset_f1 = 2 * dataset_recall * dataset_precision / (dataset_recall + dataset_precision)
        else:
            dataset_f1 = float("nan")
        dataset_recall_tp = int(tp)
        dataset_recall_total_expected = int(total_expected)
        dataset_recall_total_reranked = int(total_reranked)
    else:
        dataset_recall = float("nan")
        dataset_precision = float("nan")
        dataset_f1 = float("nan")
        dataset_recall_tp = 0
        dataset_recall_total_expected = 0
        dataset_recall_total_reranked = 0

    # Rel2 — same basis as KB Recall: KB-routed cases with no naming
    # mismatch. We use the upstream enum_relevance_score directly (no
    # per-case adjustment) since naming-mismatch rows are now removed
    # from the population entirely rather than recomputed.
    rel2_series = pd.to_numeric(df.get("enum_relevance_score",
        pd.Series([np.nan]*len(df), index=df.index)), errors="coerce")
    reranker_ran_mask = qs_kb_mask & ~naming_mismatch_mask
    rel2_defined = rel2_series[reranker_ran_mask].dropna()
    n_rel2_naming_excluded = int((qs_kb_mask & naming_mismatch_mask).sum())
    rel2_mean   = float(rel2_defined.mean())   if len(rel2_defined) else float("nan")
    rel2_median = float(rel2_defined.median()) if len(rel2_defined) else float("nan")
    rel2_std    = float(rel2_defined.std(ddof=0)) if len(rel2_defined) else float("nan")
    rel2_n      = int(len(rel2_defined))

    # Bucket distribution mirrors the Optimal-vs-final card so the two
    # ENUM-overlap metrics on the Summary page can be compared by eye.
    rel2_buckets = []
    if rel2_n:
        # Map back to row IDs so the bucket counts can become click-through
        # filters into the Test Cases tab.
        rel2_for_ids = rel2_series.copy()
        valid_kb_mask = reranker_ran_mask & rel2_for_ids.notna()
        bucket_specs = [
            ("0",          lambda v: v == 0),
            ("(0, 0.5)",   lambda v: (v > 0) & (v < 0.5)),
            ("[0.5, 1)",   lambda v: (v >= 0.5) & (v < 1)),
            ("1.0",        lambda v: v >= 0.999),
        ]
        for label, pred in bucket_specs:
            mask = valid_kb_mask & pred(rel2_for_ids)
            ids = df.loc[mask, "test_case_id"].astype(str).tolist() \
                    if "test_case_id" in df.columns else []
            rel2_buckets.append((label, int(mask.sum()), ids))

    # ── Top failure reasons (excluding test-set issues + passes) ─────────────
    # Surfaces the largest agent-side / pipeline-side issues for a
    # one-glance "what's hurting us most" answer on the Summary tab.
    #
    # Population filter: `failure_mode == X  AND  judge_pass == False`
    # over the clean (defect-free) subset. A case primarily classified
    # as `reranker_miss` that nevertheless judge-passed isn't a real
    # failure attributable to the reranker — the agent still produced
    # a passing answer, so it's excluded.
    #
    # Denominator: total clean FAILURES (n_fail_clean), NOT total clean
    # cases. The percentage on each card therefore reads as "share of
    # clean failures attributable to this issue", which lets the three
    # numbers be compared on equal footing (they don't have to sum to
    # 100% because a case has one primary issue and other issues can
    # still fail outside the Top-3, but each card's percentage is
    # interpretable on its own).
    #
    # The Issues tab still uses the unfiltered `failure_modes` data so
    # the per-issue breakdown there isn't affected.
    clean_mask = ~defect_mask
    fail_mask  = ~pass_mask                                  # judge_pass == False
    top_filter_mask = clean_mask & fail_mask                 # clean AND failed
    n_fail_clean = int(top_filter_mask.sum())                # denominator for Top-3
    fm_in_scope = (df.loc[top_filter_mask, "failure_mode"]
                    if "failure_mode" in df.columns
                    else pd.Series(dtype=object))
    excluded_for_top = {"pass", "test_set_defect"}
    fm_in_scope_failures = fm_in_scope[~fm_in_scope.isin(excluded_for_top)]
    fm_top_counts = fm_in_scope_failures.value_counts()
    top_failures_clean = []
    for fm in fm_top_counts.index[:3]:
        n = int(fm_top_counts[fm])
        pct = (n / n_fail_clean) if n_fail_clean else float("nan")
        ids = df.loc[top_filter_mask & (df["failure_mode"] == fm), "test_case_id"] \
                .astype(str).tolist() if "test_case_id" in df.columns else []
        top_failures_clean.append({
            "key":   fm,
            "label": FAILURE_MODE_LABEL.get(fm, fm),
            "owner": FAILURE_MODE_OWNER.get(fm, "—"),
            "info":  FAILURE_MODE_INFO.get(fm, ""),
            "n":     n,
            "pct":   pct,
            "ids":   ids,
        })

    return {
        "n_total":         n_total,
        "n_empty_query":   n_empty_query,
        "n_excluded_scope": n_excluded_scope,
        "excluded_empty_ids":  excluded_empty_ids,
        "excluded_scope_ids":  excluded_scope_ids,
        "n_eval":          n_eval,
        "n_pass":          n_pass,
        "pass_rate_all":   pass_rate_all,
        "n_defect":        n_defect,
        "n_clean":         n_clean,
        "n_pass_clean":    n_pass_clean,
        "pass_rate_clean": pass_rate_clean,
        "case_scope_counts": cs_counts,
        "failure_modes":   fm_rows,
        "funnel":          funnel,
        "opt_vs_final":    opt_vs_final,
        "dimensions":      dim_data,
        "rel2_mean":       rel2_mean,
        "rel2_median":     rel2_median,
        "rel2_std":        rel2_std,
        "rel2_n":          rel2_n,
        "rel2_buckets":    rel2_buckets,
        "rel2_naming_excluded": n_rel2_naming_excluded,
        "top_failures_clean": top_failures_clean,
        "n_fail_clean":    n_fail_clean,
        "count_cmp_n":              cmp_n,
        "count_cmp_mean_expected":  cmp_mean_expected,
        "count_cmp_mean_selected":  cmp_mean_selected,
        "count_cmp_buckets":        cmp_buckets,
        "dataset_recall":           dataset_recall,
        "dataset_precision":        dataset_precision,
        "dataset_f1":               dataset_f1,
        "dataset_recall_n":         dataset_recall_n,
        "dataset_recall_tp":        dataset_recall_tp,
        "dataset_recall_total_expected": dataset_recall_total_expected,
        "dataset_recall_total_reranked": dataset_recall_total_reranked,
    }


# Plain-English re-labels for the 7 judge dimensions.
def _dim_plain_label(dim: str) -> str:
    return {
        "query_clarity":                       "Was the user's question clear?",
        "selection_semantic_relevance":        "Did the reranker pick relevant content?",
        "selected_context_sufficiency":        "Did the chosen context contain the answer?",
        "optimal_retrieved_context_adequacy":  "Was the right content available to pick from?",
        "answer_expected_alignment":           "Did the agent give the expected answer?",
        "answer_groundedness":                 "Did the agent stick to the provided context?",
        "language_compliance":                 f"Did the agent answer in {LANG_NAME}?",
    }.get(dim, dim)


def _dim_owner(dim: str) -> str:
    return {
        "query_clarity":                       "Test set",
        "selection_semantic_relevance":        "Reranker",
        "selected_context_sufficiency":        "Reranker / KB",
        "optimal_retrieved_context_adequacy":  "Retrieval / KB",
        "answer_expected_alignment":           "Agent",
        "answer_groundedness":                 "Agent",
        "language_compliance":                 "Agent",
    }.get(dim, "—")


# ── Render helpers for redesigned Summary tab ───────────────────────────────
_INFO_ICON = "&#9432;"


def _info_icon(tip: str) -> str:
    """Inline (i) icon — uses the existing tip-bubble JS for hover content."""
    return f'<span class="info-icon" data-tip="{_h(tip)}">{_INFO_ICON}</span>'


# ─── Prompts tab (run-level system prompts + tool descriptions) ─────────
# Sidecar JSON `prompt_{mlflow_run_id}.json` is produced by the import
# notebook (write_prompt_sidecar). It carries the supervisor prompt,
# sub-agent prompts, and the union of tool descriptions. The per-trace
# *_prompt_hash + tool_descriptions_hash columns on the checkpoint let
# us flag runs that mixed multiple deploys.

import hashlib as _hashlib  # noqa: E402

_EMPTY_PROMPT_HASH = _hashlib.md5(b"").hexdigest()[:10]

_PROMPT_HASH_COLS = (
    ("main_agent_prompt_hash", "Supervisor (main_agent) prompt"),
    ("daily_banking_agent_prompt_hash", "daily_banking_agent prompt"),
    ("tool_descriptions_hash", "Tool descriptions"),
)


def _load_prompt_sidecar(prompts_path: Path | None,
                          checkpoint_path: Path | None,
                          run_id: str) -> dict | None:
    """Find ``prompt_{run_id}.json``: explicit flag → next to checkpoint."""
    candidate: Path | None = None
    if prompts_path is not None:
        if prompts_path.is_dir():
            candidate = prompts_path / f"prompt_{run_id}.json" if run_id else None
        else:
            candidate = prompts_path
    elif checkpoint_path is not None and run_id:
        candidate = checkpoint_path.parent / f"prompt_{run_id}.json"
    if candidate is None or not candidate.exists():
        return None
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _prompt_hash_warnings(df: pd.DataFrame) -> list[dict]:
    warnings: list[dict] = []
    for col, label in _PROMPT_HASH_COLS:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        series = series[series.astype(str).str.len() > 0]
        if series.empty:
            continue
        counts = series.value_counts()
        has_missing = _EMPTY_PROMPT_HASH in counts.index
        if len(counts) == 1 and not has_missing:
            continue
        id_col = "trace_id" if "trace_id" in df.columns else (
            "id" if "id" in df.columns else None
        )
        breakdown: list[dict] = []
        for hash_value, count in counts.items():
            if id_col is not None:
                sample_ids = df.loc[df[col] == hash_value, id_col].head(3).tolist()
            else:
                sample_ids = []
            breakdown.append({
                "hash": str(hash_value),
                "count": int(count),
                "is_missing": str(hash_value) == _EMPTY_PROMPT_HASH,
                "sample_trace_ids": [str(t) for t in sample_ids if t],
            })
        warnings.append({
            "column": col,
            "label": label,
            "distinct": int(len(counts)),
            "total": int(series.shape[0]),
            "has_missing": has_missing,
            "breakdown": breakdown,
        })
    return warnings


def _prompt_warning_card(warnings: list[dict]) -> str:
    if not warnings:
        return ""
    items: list[str] = []
    for w in warnings:
        rows: list[str] = []
        for b in w["breakdown"]:
            sample = ", ".join(b["sample_trace_ids"]) or "–"
            tag = (
                "<span class='prompt-missing-tag'>missing</span>"
                if b["is_missing"] else ""
            )
            rows.append(
                "<tr>"
                f"<td><code>{_h(b['hash'])}</code> {tag}</td>"
                f"<td class='num-cell'>{b['count']}</td>"
                f"<td><code>{_h(sample)}</code></td>"
                "</tr>"
            )
        headline = (
            f"<strong>{_h(w['label'])}</strong>: "
            f"{w['distinct']} distinct hashes across {w['total']} traces"
        )
        if w["has_missing"]:
            headline += " — includes traces with no prompt extracted"
        items.append(
            "<div class='prompt-warning-item'>"
            f"<div>{headline}</div>"
            "<table class='tbl prompt-warning-tbl'>"
            "<thead><tr><th>Hash</th><th>Traces</th><th>Sample trace IDs</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            "</table>"
            "</div>"
        )
    return (
        "<div class='card prompt-warning-card'>"
        "<div class='card-title prompt-warning-title'>"
        "⚠ Prompt / tool-registry mismatch detected"
        f"{_info_icon('A uniform eval run produces one hash per agent-prompt and one for the tool-registry. More than one — or any trace with no prompt at all — means the run mixed deploys or had span-extraction failures.')}"
        "</div>"
        f"{''.join(items)}"
        "</div>"
    )


def _prompts_tab(sidecar: dict | None, run_id: str) -> str:
    if not sidecar:
        return (
            "<div class='card'>"
            "<div class='card-title'>Prompts</div>"
            "<p class='prompt-empty'>No prompt sidecar found for this run. "
            f"Expected <code>prompt_{_h(run_id) or '&lt;run_id&gt;'}.json</code> "
            "next to the checkpoint CSV (or pass <code>--prompts</code>). Re-run "
            "the import notebook to produce one for new runs.</p>"
            "</div>"
        )

    supervisor = sidecar.get("main_agent_system_prompt") or ""
    dba_prompt = sidecar.get("daily_banking_agent_system_prompt") or ""
    tool_descriptions = sidecar.get("tool_descriptions") or {}

    def _block(label: str, body: str, *, open_: bool = False) -> str:
        attr = " open" if open_ else ""
        empty_note = " <em class='prompt-empty-note'>(not extracted from any trace)</em>" if not body.strip() else ""
        return (
            f"<details class='prompt-block'{attr}>"
            f"<summary>{_h(label)}{empty_note}</summary>"
            f"<pre class='prompt-pre'>{_h(body)}</pre>"
            "</details>"
        )

    tool_blocks = "".join(
        _block(name, tool_descriptions.get(name) or "")
        for name in sorted(tool_descriptions.keys())
    )
    tool_section = tool_blocks or "<p class='prompt-empty'>No tool descriptions captured for this run.</p>"

    return (
        "<div class='card prompts-card'>"
        "<div class='card-title'>Supervisor prompt</div>"
        f"{_block('main_agent', supervisor, open_=True)}"
        "</div>"
        "<div class='card prompts-card'>"
        "<div class='card-title'>Agent prompts</div>"
        f"{_block('daily_banking_agent', dba_prompt)}"
        "</div>"
        "<div class='card prompts-card'>"
        "<div class='card-title'>"
        f"Tool descriptions <span class='tool-count'>({len(tool_descriptions)})</span>"
        "</div>"
        f"{tool_section}"
        "</div>"
    )


def _ids_filter_link(ids: list[str], label: str, *, classes: str = "judge-eval-link",
                       inner: str = "") -> str:
    """Inline link that filters the Test Cases tab to ``ids`` when clicked."""
    ids_attr = _h(json.dumps([str(x) for x in ids]))
    return (f"<a href='#' class='{classes}' data-ids='{ids_attr}' "
            f"data-label='{_h(label)}' "
            f"title='Show these {len(ids)} cases in the Test Cases tab'>"
            f"{inner}</a>")


def _fmt_pct(v: float) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or not np.isfinite(v))):
        return "–"
    return f"{v:.1%}"


def _fmt_score(v: float) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or not np.isfinite(v))):
        return "–"
    return f"{v:.2f}"


def _failure_mode_table_html(metrics: dict) -> str:
    """Render the full Total → exclusions → Evaluable → failure-modes breakdown.

    Composition rows show their share of n_total (the entire checkpoint).
    Failure-mode rows show their share of n_eval (the in-scope KB
    cases used for every other Summary metric). Counts on failure-mode
    rows link into the Test Cases tab; the "All analyzed cases" link
    clears any active filter.
    """
    n_total      = metrics["n_total"]
    n_empty      = metrics["n_empty_query"]
    n_excl_scope = metrics["n_excluded_scope"]
    n_eval       = metrics["n_eval"]
    n_clean      = metrics["n_clean"]
    fm_rows      = metrics["failure_modes"]
    excl_empty_ids = metrics.get("excluded_empty_ids", []) or []
    excl_scope_ids = metrics.get("excluded_scope_ids", []) or []

    pct = lambda n, d: _fmt_pct(n / d) if d else "–"

    head = (
        "<thead><tr>"
        "<th>Row</th>"
        "<th style='text-align:right'>N</th>"
        "<th style='text-align:right'>%</th>"
        "<th>Owner</th>"
        "<th>Description</th>"
        "</tr></thead>"
    )

    body = ""
    # ── Composition: percentages relative to n_total ─────────────────────
    body += (
        "<tr class='fm-composition-total'>"
        "<td><strong>Total test cases</strong></td>"
        f"<td style='text-align:right'><strong>{n_total}</strong></td>"
        f"<td style='text-align:right'><strong>{pct(n_total, n_total)}</strong></td>"
        "<td>—</td>"
        "<td>All rows processed by evaluation.</td>"
        "</tr>"
        "<tr class='fm-composition-excluded'>"
        f"<td><span class='fm-indent'>↳</span> Empty <code>user_query</code> (excluded)</td>"
        f"<td style='text-align:right'>{(_ids_filter_link(excl_empty_ids, 'excluded: empty user_query', classes='judge-eval-link', inner=str(n_empty)) if n_empty and excl_empty_ids else str(n_empty))}</td>"
        f"<td style='text-align:right'>{pct(n_empty, n_total)}</td>"
        "<td>Test set</td>"
        "<td>Judge had nothing to score; excluded from analysis.</td>"
        "</tr>"
    )

    # ── Analyzed-cases denominator row (clear-filters link) ───────────────
    eval_link = (f"<a href='#' class='fm-clear-link' "
                  f"title='Clear filters and show all analyzed cases'>{n_eval}</a>"
                  if n_eval > 0 else "0")
    body += (
        "<tr class='fm-total-row fm-evaluable-row'>"
        "<td><strong>All analyzed cases</strong> (denominator below)</td>"
        f"<td style='text-align:right'><strong>{eval_link}</strong></td>"
        f"<td style='text-align:right'><strong>{pct(n_eval, n_total)}</strong></td>"
        "<td>—</td>"
        f"<td>Cases with non-empty user_query (the only Stage-1 exclusion). "
        f"Includes non-KB case_scope rows — they go through the issues "
        f"classifier rather than being silently dropped. Rows below partition this set.</td>"
        "</tr>"
    )

    # ── Failure modes ─────────────────────────────────────────────────────
    # Two groups: test-set rows (test_set_defect, enum_name_mismatch) use
    # n_eval as their denominator and come first; remaining rows use
    # n_clean to match the Top-3 failure-reasons card and the "Pass rate ·
    # excl. test-set issues" headline. After the second test-set row we
    # drop in an "All valid cases" denominator marker so the switch is
    # explicit. Each row's row["pct"] is already computed against the right
    # denominator in compute_summary_metrics, so we just format it.
    for row in fm_rows:
        n = row["n"]
        if n > 0:
            count_cell = _ids_filter_link(
                row["ids"], f"issue: {row['label']}",
                classes="judge-eval-link", inner=str(n),
            )
        else:
            count_cell = "0"
        row_cls = (" class='fm-row-defect'"
                    if row["key"] in {"test_set_defect", "enum_name_mismatch"}
                    else "")
        long_info = FAILURE_MODE_INFO_LONG.get(row["key"])
        info_cell = _h(row['info'])
        if long_info:
            info_cell += " " + _info_icon(long_info)
        body += (
            f"<tr{row_cls}>"
            f"<td><span class='fm-indent'>↳</span> {_h(row['label'])}</td>"
            f"<td style='text-align:right'>{count_cell}</td>"
            f"<td style='text-align:right'>{_fmt_pct(row['pct'])}</td>"
            f"<td>{_h(row['owner'])}</td>"
            f"<td style='font-size:11px;color:#5c7999'>{info_cell}</td></tr>"
        )
        # After ENUM name mismatch, insert the "All valid cases" denominator
        # marker. Everything below it is rendered as % of n_clean.
        if row["key"] == "enum_name_mismatch":
            valid_link = (f"<a href='#' class='fm-clear-link' "
                           f"title='Clear filters and show all valid cases'>{n_clean}</a>"
                           if n_clean > 0 else "0")
            body += (
                "<tr class='fm-total-row fm-evaluable-row'>"
                "<td><strong>All valid cases</strong> (denominator for rows below)</td>"
                f"<td style='text-align:right'><strong>{valid_link}</strong></td>"
                f"<td style='text-align:right'><strong>{pct(n_clean, n_total)}</strong></td>"
                "<td>—</td>"
                "<td>All analyzed cases minus test-set issues and ENUM name mismatches. "
                "Same denominator as the Top-3 issues card above and the "
                "\"Pass rate · excl. test-set issues\" headline; rows below report "
                "% of this number so the values reflect agent-side performance on "
                "the trustworthy subset.</td>"
                "</tr>"
            )
    return f"<table class='tbl fm-table'>{head}<tbody>{body}</tbody></table>"


def _funnel_html(funnel: dict) -> str:
    n_funnel = funnel["n"]
    if n_funnel == 0:
        return ("<p class='placeholder'>no rows match: "
                "<code>query_scope == 'kb'</code> (reranker actually ran).</p>")
    def _bar_pct(v: float) -> float:
        return 0 if (v is None or np.isnan(v)) else max(0.0, min(1.0, v)) * 100
    rows_html = ""
    for stage, recall, precision in funnel["stages"]:
        r_bar = _bar_pct(recall)
        p_bar = _bar_pct(precision)
        rows_html += (
            "<tr>"
            f"<td>{_h(stage)}</td>"
            f"<td style='text-align:right'>{_fmt_pct(recall)}</td>"
            f"<td><div class='funnel-bar'><div class='funnel-bar-fill funnel-bar-fill-recall' style='width:{r_bar:.1f}%'></div></div></td>"
            f"<td style='text-align:right'>{_fmt_pct(precision)}</td>"
            f"<td><div class='funnel-bar'><div class='funnel-bar-fill funnel-bar-fill-precision' style='width:{p_bar:.1f}%'></div></div></td>"
            "</tr>"
        )
    # NOTE: the descriptive paragraph that used to sit above the table now
    # lives in the info-icon tooltip of every card that calls _funnel_html
    # (Summary tab "Stage funnel" pane + Retrieval Findings "Stage funnel"
    # card). One source of truth, less visual clutter under the bars.
    return (
        f"<table class='tbl funnel-tbl'>"
        f"<thead><tr><th>Stage</th>"
        f"<th style='text-align:right'>Recall</th>"
        f"<th>&nbsp;</th>"
        f"<th style='text-align:right'>Precision</th>"
        f"<th>&nbsp;</th></tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
    )


def _empty_user_queries_card_html(df_all: pd.DataFrame) -> str:
    """Compact card showing how many cases have an empty user_query.
    The number is a click-through that filters the Test Cases tab to
    those rows. Returns "" when there are no empty-query rows."""
    if "_user_query_empty" not in df_all.columns:
        return ""
    sub = df_all[df_all["_user_query_empty"]]
    if sub.empty:
        return ""

    ids = sub["test_case_id"].astype(str).tolist() \
        if "test_case_id" in sub.columns else []
    n = len(sub)
    count_link = (
        _ids_filter_link(ids, "excluded: empty user_query",
                          classes="judge-eval-link",
                          inner=f"<strong>{n}</strong>")
        if ids else f"<strong>{n}</strong>"
    )
    return (
        "<div class='card'>"
        f"<div class='card-title'>Empty user queries "
        f"{_info_icon('Test cases whose user_query is empty. The judge has nothing to score on these rows; they are excluded from every metric in this report. Click the count to inspect them in the Test Cases tab.')}"
        f"</div>"
        f"<p style='font-size:13px;color:#0a285c;margin:4px 0'>"
        f"{count_link} test case{'s' if n != 1 else ''} have an empty "
        f"<code>user_query</code>.</p>"
        "</div>"
    )


def _kb_findings_table_html(df: pd.DataFrame) -> str:
    """Filterable review queue for the KB / dataset team.

    Surfaces every case that has at least one of:
      * failure_mode in {test_set_defect, enum_name_mismatch, pool_content_gap}
      * non-empty kb_improvement_suggestion or test_case_improvement_suggestion
      * detected ENUM naming mismatches
    Each row links into the Test Cases tab. The table reuses the existing
    `tbl-filterable` JS so per-column filters work out of the box.
    """
    if df is None or len(df) == 0:
        return "<p class='placeholder'>no rows</p>"

    fm = df.get("failure_mode", pd.Series([""]*len(df), index=df.index)).fillna("").astype(str)
    tc_sugg = df.get("test_case_improvement_suggestion",
                      pd.Series([""]*len(df), index=df.index)).fillna("").astype(str)
    kb_sugg = df.get("kb_improvement_suggestion",
                      pd.Series([""]*len(df), index=df.index)).fillna("").astype(str)
    naming = df.get("_naming_mismatch", pd.Series([False]*len(df), index=df.index))

    issue_modes = {"test_set_defect", "enum_name_mismatch", "pool_content_gap"}
    mask = (fm.isin(issue_modes)
            | tc_sugg.str.strip().ne("")
            | kb_sugg.str.strip().ne("")
            | naming.astype(bool))
    sub = df[mask]
    if sub.empty:
        return "<p class='placeholder'>no test-case issues detected in this run.</p>"

    # Stable sort: defects first, then naming mismatches, then KB content gaps,
    # then everything else; secondary sort by test_case_id for deterministic order.
    rank = {"test_set_defect": 0, "enum_name_mismatch": 1, "pool_content_gap": 2}
    sub = sub.assign(_rank=fm[mask].map(lambda k: rank.get(k, 9))) \
              .sort_values(by=["_rank", "test_case_id"], na_position="last") \
              .drop(columns="_rank")

    def _issue_summary(r) -> str:
        parts = []
        v = r.get("expected_reference_issue_description", "")
        if isinstance(v, str) and v.strip():
            parts.append(f"GOLD: {v.strip()}")
        misses = r.get("_enum_naming_mismatches") or []
        if isinstance(misses, list) and misses:
            shown = "; ".join(
                f"{m.get('expected', '')}↔{(m.get('kb_form') if isinstance(m.get('kb_form'), str) else '/'.join(m.get('kb_form') or []))}"
                for m in misses[:3]
            )
            extra = "" if len(misses) <= 3 else f" (+{len(misses) - 3})"
            parts.append(f"NAMING: {shown}{extra}")
        v = r.get("retrieved_pool_inadequacy_description", "")
        if isinstance(v, str) and v.strip():
            parts.append(f"POOL: {v.strip()}")
        v = r.get("kb_improvement_suggestion", "")
        if isinstance(v, str) and v.strip():
            parts.append(f"KB SUGG: {v.strip()}")
        v = r.get("test_case_improvement_suggestion", "")
        if isinstance(v, str) and v.strip():
            parts.append(f"TEST-SET SUGG: {v.strip()}")
        return " · ".join(parts) if parts else "(no specific suggestion)"

    # Build the filter-row widgets. case_scope as multi-select; text filters
    # on test_case_id, user query, and issue.
    cs_in_sub = sub.get("case_scope", pd.Series([""]*len(sub), index=sub.index)).fillna("").astype(str)
    cs_options = sorted(v for v in cs_in_sub.dropna().unique().tolist() if v)

    def _multi_widget(col: str, opts: list[str]) -> str:
        items = "".join(
            f'<label><input type="checkbox" value="{_h(v)}"> {_h(v)}</label>'
            for v in opts
        )
        return (f'<div class="multi-filter" data-col="{col}">'
                f'<button type="button" class="multi-toggle">all</button>'
                f'<div class="multi-menu">{items}</div>'
                f'</div>')

    def _text_widget(col: str) -> str:
        return (f'<input class="col-filter" data-col="{col}" type="text" '
                f'placeholder="filter…">')

    issue_tip = (
        "Concatenated, prefixed by source: "
        "GOLD: judge's expected_reference_issue_description (gold reference flagged for review). "
        "NAMING: ENUM name mismatches detected by the deterministic check "
        "(expected ↔ KB-form). "
        "POOL: judge's retrieved_pool_inadequacy_description (post-prune fragments too thin). "
        "KB SUGG: judge's kb_improvement_suggestion. "
        "TEST-SET SUGG: judge's test_case_improvement_suggestion."
    )
    head = (
        "<thead>"
        "<tr>"
        "<th>Test case</th>"
        "<th>case_scope</th>"
        "<th>User query (en)</th>"
        f"<th>Issue / suggestion {_info_icon(issue_tip)}</th>"
        "</tr>"
        f"<tr class='filter-row'>"
        f"<th>{_text_widget('test_case_id')}</th>"
        f"<th>{_multi_widget('case_scope', cs_options)}</th>"
        f"<th>{_text_widget('user_query_en')}</th>"
        f"<th>{_text_widget('issue_text')}</th>"
        "</tr>"
        "</thead>"
    )

    body = ""
    for _, r in sub.iterrows():
        tid = _h(r.get("test_case_id", ""))
        cs = str(r.get("case_scope", "") or "")
        cs_cls = "cs-" + (cs if cs in {"kb","kb_and_api","api","out_of_scope","ambiguous"} else "other")
        uq = str(r.get("user_query_en", "") or "")
        issue = _issue_summary(r)
        body += (
            "<tr>"
            f"<td data-col='test_case_id' data-val='{tid}'>"
            f"<a class='case-link' href='#' data-case='{tid}'>{tid}</a></td>"
            f"<td data-col='case_scope' data-val='{_h(cs)}'>"
            f"<span class='cs-badge {cs_cls}'>{_h(cs)}</span></td>"
            f"<td data-col='user_query_en' data-val='{_h(uq)}' "
            f"class='kb-findings-query'>{_h(uq[:200])}</td>"
            f"<td data-col='issue_text' data-val='{_h(issue)}' "
            f"class='kb-findings-issue'>{_h(issue)}</td>"
            "</tr>"
        )
    return (
        "<table class='tbl tbl-filterable kb-findings-table' id='kb-findings-table'>"
        f"{head}<tbody>{body}</tbody></table>"
    )


def _aggregate_naming_mismatches(df: pd.DataFrame) -> list[dict]:
    """Collect distinct (expected, kb_form) pairs across every row, with the
    list of test cases each pair affects. Stable sort by impact (most cases
    affected first) so the largest-blast-radius offenders surface at the top.
    """
    if "_enum_naming_mismatches" not in df.columns:
        return []
    bucket: dict[tuple, dict] = {}
    for _, row in df.iterrows():
        misses = row.get("_enum_naming_mismatches") or []
        if not isinstance(misses, list):
            continue
        case_id = "" if pd.isna(row.get("test_case_id")) else str(row.get("test_case_id"))
        for m in misses:
            if not isinstance(m, dict):
                continue
            kb_form = m.get("kb_form")
            kb_key = " / ".join(kb_form) if isinstance(kb_form, list) else str(kb_form or "")
            key = (str(m.get("expected", "")), kb_key)
            entry = bucket.setdefault(key, {
                "expected": key[0],
                "kb_form":  kb_form if isinstance(kb_form, list) else (kb_form or ""),
                "cases":    [],
            })
            if case_id and case_id not in entry["cases"]:
                entry["cases"].append(case_id)
    rows = list(bucket.values())
    rows.sort(key=lambda x: (-len(x["cases"]), x["expected"]))
    return rows


def _naming_mismatches_card_html(agg: list[dict]) -> str:
    if not agg:
        return ""
    body_rows = ""
    for entry in agg:
        kb_form = entry["kb_form"]
        kb_html = (" / ".join(_h(x) for x in kb_form)
                    if isinstance(kb_form, list) else _h(kb_form))
        case_ids = entry["cases"]
        n = len(case_ids)
        cases_link = _ids_filter_link(
            case_ids, f"naming mismatch: {entry['expected']} ↔ {kb_form}",
            classes="judge-eval-link",
            inner=f"{n} {'case' if n == 1 else 'cases'}",
        ) if n else "0"
        body_rows += (
            "<tr>"
            f"<td><code class='naming-expected'>{_h(entry['expected'])}</code></td>"
            f"<td><code class='naming-kb-form'>{kb_html}</code></td>"
            f"<td style='text-align:right'>{cases_link}</td>"
            "</tr>"
        )
    return (
        "<div class='card'>"
        f"<div class='card-title'>Detected ENUM name mismatches "
        f"{_info_icon('Deterministic check on every expected_enum across the run: drop case and non-alphanumeric chars, look for any KB ENUM that matches the normalized form. Pairs that differ only in separator / casing produce false zeros in Rel2 / enum-recall and are worth fixing on either side.')}"
        f"</div>"
        "<p style='font-size:12px;color:#5c7999;margin-bottom:8px'>"
        "Each row is a distinct expected/KB pair found in this run. Click the case "
        "count to drill into the affected test cases."
        "</p>"
        f"<table class='tbl naming-table'>"
        "<thead><tr>"
        "<th>Expected (test set)</th>"
        "<th>KB form</th>"
        "<th style='text-align:right'>Cases affected</th>"
        "</tr></thead>"
        f"<tbody>{body_rows}</tbody></table></div>"
    )


def _rel2_distribution_html(metrics: dict) -> str:
    """Bucket histogram of Rel2 over the cases where the search tool was used.
    Mirrors the Optimal-vs-final card's structure so the two ENUM-overlap
    metrics on the Summary page line up visually."""
    n      = metrics["rel2_n"]
    mean   = metrics["rel2_mean"]
    median = metrics["rel2_median"]
    std    = metrics["rel2_std"]
    buckets = metrics["rel2_buckets"]
    if n == 0:
        return "<p class='placeholder'>no cases reached the reranker.</p>"
    rows = ""
    for label, count, ids in buckets:
        pct = (count / n) if n else float("nan")
        if count and ids:
            cell = _ids_filter_link(ids, f"Rel2 bucket: {label}",
                                      classes="judge-eval-link",
                                      inner=str(count))
        else:
            cell = str(count)
        rows += (
            "<tr>"
            f"<td>{_h(label)}</td>"
            f"<td style='text-align:right'>{cell}</td>"
            f"<td style='text-align:right'>{_fmt_pct(pct)}</td>"
            "</tr>"
        )
    return (
        f"<p style='font-size:12px;color:#5c7999;margin-bottom:8px'>"
        f"Distribution of <code>enum_relevance_score</code> across the "
        f"<strong>{n}</strong> cases where the search tool was used. "
        f"Mean = <strong>{_fmt_score(mean)}</strong> · "
        f"median = {_fmt_score(median)} · σ = {_fmt_score(std)}.</p>"
        f"<table class='tbl funnel-tbl'>"
        f"<thead><tr><th>Rel2 bucket</th><th style='text-align:right'>N</th>"
        f"<th style='text-align:right'>%</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _inline_delta_html(baseline_val, current_val, fmt: str, good_dir: str,
                        *, title_prefix: str = "vs baseline") -> str:
    """Small inline "↑ +X.X pp" chip rendered under a card's value.

    Args:
        baseline_val: the prior run's value (any numeric, or None).
        current_val:  this run's value (any numeric, or None).
        fmt:  ``"pct"`` (× 100, suffix ``pp``), ``"score"`` (raw 2-decimal
              delta), or ``"count"`` (integer delta).
        good_dir: ``"up"`` if higher = better (pass rate, recall);
                  ``"down"`` if lower = better (issue counts).
        title_prefix: leading words for the ``title=`` tooltip so each
              chip can say "vs baseline {label}: was X.X%".

    Returns ``""`` when either side is missing — keeps the call sites
    branch-free.
    """
    if baseline_val is None or current_val is None:
        return ""
    try:
        delta = float(current_val) - float(baseline_val)
    except (TypeError, ValueError):
        return ""
    if abs(delta) < 1e-9:
        arrow, cls = "●", "timeline-delta-flat"
    elif delta > 0:
        arrow = "↑"
        cls = "timeline-delta-good" if good_dir == "up" else "timeline-delta-bad"
    else:
        arrow = "↓"
        cls = "timeline-delta-good" if good_dir == "down" else "timeline-delta-bad"
    if fmt == "pct":
        delta_str = f"{delta * 100:+.1f} pp"
        was_str   = f"{baseline_val * 100:.1f}%"
    elif fmt == "score":
        delta_str = f"{delta:+.2f}"
        was_str   = f"{baseline_val:.2f}"
    else:  # "count"
        delta_str = f"{int(delta):+d}"
        was_str   = f"{int(baseline_val)}"
    tip = f"{title_prefix}: was {was_str}"
    return (
        f"<div class='hc-delta-inline {cls}' title='{_h(tip)}'>"
        f"{arrow} {delta_str}"
        "</div>"
    )


def _top_failures_html(top: list[dict], n_fail_clean: int,
                         *, baseline: dict | None = None) -> str:
    """Render the top-3 failure reasons (excl. test-set issues) as a row of
    headline-style cards. Click any card to drill into the matching cases.

    ``n_fail_clean`` is the denominator that each card's pre-computed
    percentage was divided by — the total number of clean failures
    (cases that judge-failed AND aren't test-set defects). The cards
    therefore read as "share of clean failures attributable to this
    issue", not "share of all clean cases".

    Returns "" when there's nothing to show (no failures, or no clean
    rows after defect exclusion).
    """
    if not top or n_fail_clean <= 0:
        return ""
    # Baseline-count lookup for the inline Δ chip — only failure modes that
    # appeared in the baseline run's own Top-3 (and therefore carry an entry
    # in ``baseline['issue_counts']``) get a Δ chip. Anything else, including
    # all failure modes when no ``--baseline`` was passed, simply omits the
    # chip via _inline_delta_html's None-short-circuit.
    baseline_issue_counts = (baseline or {}).get("issue_counts", {}) if baseline else {}
    baseline_label = (baseline or {}).get("label", "")
    cards = []
    for entry in top:
        n = entry["n"]
        pct = entry["pct"]
        click_link = _ids_filter_link(
            entry["ids"], f"top failure: {entry['label']}",
            classes="judge-eval-link top-failure-link",
            inner=f"{_fmt_pct(pct)}",
        ) if n else _fmt_pct(pct)
        # Inline Δ chip: count change vs baseline. Lower count = good
        # direction for issue cards.
        baseline_n = baseline_issue_counts.get(entry["key"])
        delta_chip = _inline_delta_html(
            baseline_n, n, fmt="count", good_dir="down",
            title_prefix=f"vs baseline {baseline_label}" if baseline_label else "vs baseline",
        )
        # Tooltip carries the short failure-mode description + the
        # count line that used to render as a footer (hc-detail).
        # NB: numerator and denominator are BOTH filtered to "clean
        # AND judge-failed" — the percentage on the card reads as
        # "share of clean failures attributable to this issue".
        tip_parts = []
        if entry.get("info"):
            tip_parts.append(str(entry["info"]))
        tip_parts.append(
            f"This run: {n} clean failures attributed to this issue / "
            f"{n_fail_clean} total clean failures. Cases that carry "
            f"this issue as their primary cause but still judge-passed "
            f"aren't counted here — see the Issues tab for the "
            f"unfiltered per-issue breakdown."
        )
        tip = _info_icon("\n\n".join(tip_parts))
        cards.append(
            "<div class='headline-card top-failure-card'>"
            f"<div class='hc-label'>{_h(entry['label'])} "
            f"<span class='top-failure-owner'>· {_h(entry['owner'])}</span> "
            f"{tip}</div>"
            f"<div class='hc-value'>{click_link}</div>"
            f"{delta_chip}"
            "</div>"
        )
    section_tooltip = (
        f"Top issues among clean cases that ALSO failed the judge "
        f"(judge_pass = False). A case is counted only when it both "
        f"carries the issue as its primary cause AND has a FAIL tag. "
        f"The percentage on each card divides by the TOTAL clean "
        f"failures ({n_fail_clean} this run), not by all clean cases — "
        f"so it reads as 'share of clean failures attributable to this "
        f"issue', letting the three numbers be compared on equal "
        f"footing. They don't have to sum to 100% because failures "
        f"outside the Top-3 (other_failure, language_drift, …) live in "
        f"the same denominator.\n\n"
        f"Cases with the issue that nevertheless judge-passed (e.g. the "
        f"reranker missed an ENUM but the agent still produced a "
        f"passing answer) are excluded. Test-set defects / ambiguous / "
        f"out-of-scope rows are also excluded. Click a card to drill "
        f"in.\n\n"
        f"The Issues tab uses the unfiltered failure-mode breakdown "
        f"and is unaffected by this filter."
    )
    note = (
        f"<div class='top-failure-title'>Top {len(cards)} "
        f"issue{'s' if len(cards) != 1 else ''} · failures only "
        f"(excl. test-set issues) {_info_icon(section_tooltip)}</div>"
    )
    grid_cls = "headline-row"
    if len(cards) == 3:
        grid_cls += " headline-row-3"
    elif len(cards) == 2:
        grid_cls += " headline-row-2"
    return note + f"<div class='{grid_cls}'>{''.join(cards)}</div>"


def _dimension_cards_html(dim_data: list[dict],
                            dim_full_info: dict[str, dict] | None = None) -> str:
    if not dim_data:
        return "<p class='placeholder'>no dimension scores in this checkpoint.</p>"
    dim_full_info = dim_full_info or {}
    cards = []
    for d in dim_data:
        n = d["n"]
        c0, c1, c2 = d["counts"][0], d["counts"][1], d["counts"][2]
        # Stacked bar widths in % of n (computed; never hard-coded).
        if n > 0:
            w0, w1, w2 = c0 * 100 / n, c1 * 100 / n, c2 * 100 / n
        else:
            w0 = w1 = w2 = 0
        link0 = _ids_filter_link(d["ids"][0], f"{d['key']}=0",
                                   classes="judge-eval-link", inner=f"<strong>{c0}</strong> fail") if c0 else f"<span style='color:#a3b5c9'>0 fail</span>"
        link1 = _ids_filter_link(d["ids"][1], f"{d['key']}=1",
                                   classes="judge-eval-link", inner=f"<strong>{c1}</strong> partial") if c1 else f"<span style='color:#a3b5c9'>0 partial</span>"
        link2 = _ids_filter_link(d["ids"][2], f"{d['key']}=2",
                                   classes="judge-eval-link", inner=f"<strong>{c2}</strong> pass") if c2 else f"<span style='color:#a3b5c9'>0 pass</span>"
        # Info icon: verbatim YAML rubric (name / weight / description /
        # scale) for this dimension. Tooltip uses the existing tip-bubble JS.
        yaml_tip = _format_dimension_yaml_tip(dim_full_info.get(d["key"], {}))
        info_icon_html = _info_icon(yaml_tip) if yaml_tip else ""
        cards.append(
            "<div class='dim-summary-card'>"
            f"<div class='dim-summary-head'>"
            f"<span class='dim-summary-q'>{_h(d['label'])} {info_icon_html}</span>"
            f"<span class='dim-summary-tag' title='Owner team'>{_h(d['owner'])}</span>"
            f"</div>"
            f"<div class='dim-summary-meta'>"
            f"<code>{_h(d['key'])}</code> · mean = <strong>{_fmt_score(d['mean'])}</strong>"
            f" · n = {n}</div>"
            f"<div class='dim-summary-bar'>"
            f"<div class='dim-bar-seg dim-bar-bad'  style='width:{w0:.2f}%'></div>"
            f"<div class='dim-bar-seg dim-bar-mid'  style='width:{w1:.2f}%'></div>"
            f"<div class='dim-bar-seg dim-bar-good' style='width:{w2:.2f}%'></div>"
            f"</div>"
            f"<div class='dim-summary-links'>"
            f"<span class='dim-link-bad'>{link0}</span>"
            f"<span class='dim-link-mid'>{link1}</span>"
            f"<span class='dim-link-good'>{link2}</span>"
            f"</div>"
            "</div>"
        )
    return f"<div class='dim-summary-grid'>{''.join(cards)}</div>"


def _scope_distribution_html(cs_counts: dict, n_eval: int) -> str:
    """Render the case_scope distribution. The yaml expects every case to be
    KB; anything else flags a test-set composition issue."""
    if not cs_counts or n_eval == 0:
        return ""
    rows = ""
    declared = ("kb", "kb_and_api", "api", "out_of_scope", "ambiguous")
    seen = set()
    for k in declared:
        if k in cs_counts:
            seen.add(k)
            n = int(cs_counts[k])
            rows += (f"<tr><td><code>{_h(k)}</code></td>"
                      f"<td style='text-align:right'>{n}</td>"
                      f"<td style='text-align:right'>{_fmt_pct(n / n_eval)}</td></tr>")
    # Anything not in the declared enum (would be a judge schema violation,
    # but report it here so it isn't silently hidden).
    for k, n in cs_counts.items():
        if k in seen or not k:
            continue
        rows += (f"<tr><td><code>{_h(k)}</code> <em>(unexpected)</em></td>"
                  f"<td style='text-align:right'>{int(n)}</td>"
                  f"<td style='text-align:right'>{_fmt_pct(int(n) / n_eval)}</td></tr>")
    return (
        f"<table class='tbl funnel-tbl'>"
        f"<thead><tr><th>case_scope (judge)</th><th style='text-align:right'>N</th>"
        f"<th style='text-align:right'>%</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _build_threshold_explainer_html() -> str:
    """Notes-tab card: how the pass threshold interacts with the scorer
    weights. Renders the weight table and a set of pass/fail example
    configurations so colleagues can see the boundary concretely.

    The weighted_avg distribution histogram lives on the Notes tab next
    to the Pass-rate card (with a draggable threshold slider) — that's
    the more discoverable home for an interactive sanity-check of the
    cut-off, alongside the definition that explains it."""
    w = DIMENSION_WEIGHTS
    w_sum = sum(w.values())
    max_score = 2 * w_sum
    threshold_raw = PASS_THRESHOLD * max_score

    def _wavg(scores: dict) -> tuple[float, float]:
        s = sum(scores[k] * w[k] for k in w)
        return s, s / w_sum / 2.0

    # Display order of scorers in the rows (uses DIMENSION_WEIGHTS order).
    weight_rows = ""
    for k, v in w.items():
        weight_rows += (
            f"<tr><td><code>{_h(k)}</code></td>"
            f"<td style='text-align:right;font-variant-numeric:tabular-nums'>{v:g}</td></tr>"
        )
    weight_rows += (
        f"<tr style='border-top:2px solid #135ee2'>"
        f"<td><strong>Σweight</strong></td>"
        f"<td style='text-align:right;font-weight:700'>{w_sum:g}</td></tr>"
    )

    # Pre-build a couple of useful "shape" dicts for the example configurations.
    weight2_keys = [k for k, v in w.items() if v >= 2]
    weight1_keys = [k for k, v in w.items() if v < 2]

    def _shape(values_for_w2: list[int], values_for_w1: list[int]) -> dict:
        return ({k: values_for_w2[i] for i, k in enumerate(weight2_keys)}
                | {k: values_for_w1[i] for i, k in enumerate(weight1_keys)})

    examples = [
        ("All scorers at 2 (perfect)",
            {k: 2 for k in w}),
        ("All scorers at 1 (every dim 'partial')",
            {k: 1 for k in w}),
        ("All scorers at 0",
            {k: 0 for k in w}),
        ("3 of 5 weight-2 scorers at 2, rest at 1",
            _shape([2, 2, 2, 1, 1], [1, 1])),
        ("2 of 5 weight-2 scorers at 2, rest at 1",
            _shape([2, 2, 1, 1, 1], [1, 1])),
        ("All at 2 except one weight-1 scorer at 0",
            _shape([2, 2, 2, 2, 2], [0, 2])),
        ("All at 2 except one weight-2 scorer at 0",
            _shape([0, 2, 2, 2, 2], [2, 2])),
        ("All at 2 except two weight-2 scorers at 0",
            _shape([0, 0, 2, 2, 2], [2, 2])),
    ]

    example_rows = ""
    for label, scores in examples:
        raw, wa = _wavg(scores)
        critical_zero = any(scores[k] == 0 for k in CRITICAL_DIMS)
        wa_clears = wa >= PASS_THRESHOLD
        passes = wa_clears and not critical_zero
        if passes:
            mark, cls = "PASS", "num-good"
        elif wa_clears and critical_zero:
            mark, cls = "FAIL (veto)", "num-bad"
        else:
            mark, cls = "FAIL", "num-bad"
        example_rows += (
            f"<tr>"
            f"<td>{_h(label)}</td>"
            f"<td style='text-align:right;font-variant-numeric:tabular-nums'>{raw:g}</td>"
            f"<td style='text-align:right;font-variant-numeric:tabular-nums'>{wa:.3f}</td>"
            f"<td class='{cls}' style='text-align:center;font-weight:700'>{mark}</td>"
            f"</tr>"
        )

    return f"""
    <div class="card">
      <div class="card-title">Pass threshold &amp; scorer weights</div>
      <p>For each case, the weighted average of the seven scorer scores
      (each in 0/1/2) is normalized to [0, 1] and compared to the
      threshold. A hard veto blocks pass when any critical (weight-2)
      scorer scored 0 — that prevents the headline from hiding a
      catastrophic single-dimension failure behind a high mean (e.g.
      groundedness = 0 with everything else at 2 still gives weighted_avg
      = 0.83).</p>
      <pre style="font-size:12px;color:#0a285c;background:#f4f6fa;padding:8px 10px;border-radius:6px;margin:6px 0">
weighted_avg = Σ (score × weight) / Σ weight / 2
case passes  ⇔ weighted_avg ≥ {PASS_THRESHOLD:g} AND
                no weight-2 scorer == 0
</pre>
      <p>For this run, <code>Σ weight = {w_sum:g}</code>, so the maximum
      possible <code>Σ (score × weight)</code> is <strong>{max_score:g}</strong>
      (all scorers at 2). The pass cut-off in raw weighted units is
      <strong>{threshold_raw:g}</strong> — equivalent to "70% of the
      weighted maximum". The critical scorers (weight ≥ 2) the veto
      applies to: <code>{', '.join(CRITICAL_DIMS)}</code>.
      Each scorer's weight from the YAML rubric:</p>
      <table class="tbl funnel-tbl" style="max-width:480px">
        <thead><tr><th>Scorer</th><th style="text-align:right">Weight</th></tr></thead>
        <tbody>{weight_rows}</tbody>
      </table>

      <p style="margin-top:14px"><strong>Reference configurations</strong> —
      where the threshold sits relative to common scoring patterns. The
      "minimum to pass with no zeros" is roughly <em>three of the five
      weight-2 scorers at 2, everything else at 1</em>; one fewer
      upgrade and the case fails. Weight-1 scorers (query_clarity,
      language_compliance) carry less leverage, so dropping them to 1
      from 2 affects the score less than dropping a weight-2 scorer.</p>
      <table class="tbl funnel-tbl">
        <thead><tr>
          <th>Configuration</th>
          <th style="text-align:right">Σ (score × weight)</th>
          <th style="text-align:right">weighted_avg</th>
          <th style="text-align:center">Result</th>
        </tr></thead>
        <tbody>{example_rows}</tbody>
      </table>
    </div>
    """


def _doc_tab_html(metrics: dict, wavg_hist_fig=None, *,
                  include_latency: bool = False) -> str:
    """Notes page — companion text to the rest of the report. Every count
    or percentage referenced in the body comes from ``metrics`` so the
    text stays in sync with the run on display.

    When ``wavg_hist_fig`` is provided, the weighted_avg distribution
    (with an interactive threshold slider) renders as a side-card to the
    first "Pass rate" card so readers can do a what-if on the cut-off
    while reading the definition."""
    cs_html = _scope_distribution_html(metrics["case_scope_counts"], metrics["n_eval"])
    fm_terms = "".join(
        f"<dt><strong>{_h(FAILURE_MODE_LABEL[k])}</strong> "
        f"<span style='color:#5c7999'>(owner: {_h(FAILURE_MODE_OWNER[k])})</span></dt>"
        f"<dd>{_h(FAILURE_MODE_INFO_LONG.get(k, FAILURE_MODE_INFO[k]))}</dd>"
        for k in FAILURE_MODES
    )
    # Build the optional weighted_avg histogram + slider side-card.
    # Sits next to the "Pass rate" card so the reader can dry-run the
    # cut-off in the same field of view as the definition.
    if wavg_hist_fig is not None:
        wavg_side_card = f"""
    <div class="card wavg-hist-card">
      <div class="card-title">weighted_avg distribution {_info_icon(
          f"Histogram of weighted_avg across the analyzed cases. The "
          f"solid red line marks the report's pass cut-off "
          f"({PASS_THRESHOLD}); the dashed orange line follows the "
          f"slider below, letting you ask 'how many cases would pass if "
          f"the threshold were X?'. The slider's PASS / FAIL read-out "
          f"applies the same definition as the headline cards (wa ≥ "
          f"threshold AND no weight-2 scorer at 0), so the counts at "
          f"threshold {PASS_THRESHOLD} match the headline Pass-rate card "
          f"exactly. Reset returns the slider to {PASS_THRESHOLD}.")}
      </div>
      <div class="wavg-chart-wrap">
        {_plot(wavg_hist_fig, div_id="wavg-hist-chart")}
      </div>
      <div class="wavg-slider-wrap">
        <div class="wavg-slider-row">
          <span>threshold</span>
          <input type="range" id="wavg-slider" min="0" max="1" step="0.01" value="{PASS_THRESHOLD}">
          <span class="wavg-slider-value" id="wavg-slider-value">{PASS_THRESHOLD:.2f}</span>
          <button type="button" class="wavg-slider-reset" id="wavg-slider-reset" title="Reset to default ({PASS_THRESHOLD})">reset</button>
        </div>
        <div class="wavg-slider-readout" id="wavg-slider-readout"></div>
      </div>
    </div>"""
        pass_row_open  = '<div class="summary-grid-2 doc-passrate-row">'
        pass_row_close = "</div>"
    else:
        wavg_side_card = ""
        pass_row_open  = ""
        pass_row_close = ""

    # The latency-definitions card is only rendered when the latency
    # surfaces are enabled. Built as a variable so the f-string below
    # stays readable and the condition lives in one place.
    if include_latency:
        latency_definitions_card = _latency_definitions_card_html()
    else:
        latency_definitions_card = ""

    return f"""
    {pass_row_open}
    <div class="card">
      <div class="card-title">Pass rate</div>
      <p><strong>Definition.</strong> A case is a <em>pass</em> when
      <code>weighted_avg ≥ {PASS_THRESHOLD}</code> <em>and</em> no
      weight-2 (critical) scorer is at 0. The first half is the headline
      gate; the second is a hard veto that prevents one catastrophic
      dimension (e.g. <code>answer_groundedness = 0</code>) from hiding
      behind an otherwise-high mean. <code>weighted_avg</code> is the
      weighted mean of the seven judge scorers, normalized to [0, 1].
      This single rule drives the per-case PASS/FAIL badge, the Pass | Fail
      chip filter on the Test Cases tab, and the pass-rate headline card
      on the Summary tab — they always agree on the count.</p>

      <p><strong>Denominator.</strong> Every case in the checkpoint with
      a non-empty <code>user_query</code> (the only Stage-1 exclusion).
      Rows the judge classified as <code>api</code>,
      <code>out_of_scope</code>, or <code>ambiguous</code> are kept in
      the analysis and flow through the Issues classifier
      (<code>ambiguous</code> / <code>out_of_scope</code> become
      <em>Test-set issue</em>; <code>api</code> passes through to
      whatever applies given its scores).</p>

      <p>The Summary tab shows the agent-side pass rate (trustworthy
      ground truth only) compared against an internal target:</p>
      <ul style="margin-left:20px;margin-bottom:8px">
        <li><strong>Pass rate · excl. test-set issues</strong>
            ({metrics['n_pass_clean']} / {metrics['n_clean']} =
            {_fmt_pct(metrics['pass_rate_clean'])}, target
            {PASS_RATE_TARGET:.0%}) — drops rows the judge flagged with
            <code>expected_reference_looks_wrong</code>, rows classified
            as <code>ambiguous</code> / <code>out_of_scope</code>, and
            rows the deterministic check flagged as ENUM naming
            mismatches; measures the agent against trustworthy ground
            truth only. The value turns green on the card when it meets
            or exceeds the target.</li>
        <li><strong>Pass rate · all cases</strong>
            ({metrics['n_pass']} / {metrics['n_eval']} =
            {_fmt_pct(metrics['pass_rate_all'])}) — for reference only;
            includes cases with gold-reference issues and ENUM-naming
            mismatches in the denominator, so it's a "production
            realistic" view that the test-set hygiene drags down.</li>
      </ul>

      <p style="font-size:12px;color:#5c7999;margin:0">
      <strong>Why the Clean pass row in the Issues table is smaller than
      the headline pass count.</strong> The Issues table assigns each
      case to one <em>primary issue</em> in priority order (test-set
      defects first, then retrieval / pruning / reranker losses, then
      agent-side problems, then language). The <em>Clean pass</em> row
      is the residual: a case that not only met the judge-pass rule
      above but also had no upstream/agent issue flagged. A case can
      still be a judge-pass and be filed under, say, <em>Test-set
      issue</em> — it counts in the headline pass rate but lands in the
      Test-set issue row, not in Clean pass. This is intentional: the
      headline answers "what fraction did the judge pass?"; the Issues
      table answers "where do the rest land, and is the gold trustworthy?".</p>
    </div>
    {wavg_side_card}
    {pass_row_close}

    {_build_threshold_explainer_html()}

    <div class="card">
      <div class="card-title">KB Recall · KB Precision · Stage funnel</div>
      <p>Three metrics that share the same basis: cases where the search
      tool actually ran (<code>query_scope == "kb"</code>) AND the
      deterministic check did NOT flag an ENUM-naming mismatch. The
      headline KB Recall and KB Precision numbers are micro-averaged
      (Σ TP / Σ expected and Σ TP / Σ selected respectively, summed across
      cases first then divided). The Stage funnel shows the same
      calculation at each pipeline stage — pre-prune (vector DB output)
      → post-prune (after dedup/filtering) → reranked (final selection).
      The reranked row equals the headline numbers exactly. Recall trends
      down through the pipeline (gold ENUMs can only be lost as the set
      shrinks); precision trends up (later stages drop noise). Run-level
      n = <strong>{metrics['dataset_recall_n']}</strong>.</p>
    </div>

    <div class="card">
      <div class="card-title">case_scope distribution (judge classification)</div>
      <p style="font-size:12px;color:#5c7999;margin-bottom:8px">
      The judge classifies each query independently of how the system
      handled it. The test set is supposed to be 100 % KB; any other label
      indicates a test-set composition issue.</p>
      {cs_html}
    </div>

    <div class="card">
      <div class="card-title">Issues</div>
      <p style="font-size:12px;color:#5c7999;margin-bottom:10px">
      Each analyzed case is assigned one primary issue in priority
      order: test-set issues first (because they invalidate downstream
      analysis), then ENUM name mismatch, then wrong agent routing,
      then per-stage ENUM losses, then pool/agent/language issues, with
      <em>Clean pass</em> and <em>Other issue</em> as the residual
      buckets. The Issues tab's table shows two denominators: the
      test-set group (test-set issue, ENUM name mismatch) is a share of
      <code>n_eval</code>; everything below the "All valid cases"
      divider is a share of <code>n_clean = n_eval − test-set issue −
      ENUM name mismatch</code>, matching the Top-3 issues card on the
      Summary tab and the "Pass rate · excl. test-set issues" headline.</p>
      <dl class="doc-dl">{fm_terms}</dl>
    </div>

    <div class="card">
      <div class="card-title">Placeholder handling</div>
      <p>The judge emits the literal string
      <code>"Answer is not available based on given information"</code>
      in <code>hallucinated_claims</code>,
      <code>unavailable_facts_in_selected_context</code>,
      <code>missing_facts</code>, and
      <code>expected_answer_summary_with_optimal_context</code> when the
      post-prune pool / reranked context cannot support an answer. We
      normalize that placeholder out of every list before counting, and
      surface it as a separate "no achievable answer" signal — counting
      it as a real claim would inflate hallucination and missing-fact
      metrics.</p>
    </div>

    {latency_definitions_card}
    """


def _latency_definitions_card_html() -> str:
    """Notes-tab card explaining what each latency step represents and how
    the retry-overhead heuristic works. Only rendered when the latency
    surfaces are enabled (--include-latency on the CLI)."""
    # Bucket threshold table rendered from the live constants so the doc
    # stays in sync with the heuristic if someone retunes the dicts.
    from hg_ds_evals.preprocessing.latency import (
        RETRY_DUR_THRESHOLDS_MS,
        RETRY_MAX_TOK_PER_S_BY_BUCKET,
        RETRY_BASELINE_TOK_PER_S as _BASE,
    )
    rows = ""
    for b in ("routing", "sub_agent_llm", "kb_rerank", "kb_other", "other"):
        rows += (
            f"<tr><td><code>{b}</code></td>"
            f"<td style='text-align:right'>{RETRY_DUR_THRESHOLDS_MS[b]/1000:.0f}s</td>"
            f"<td style='text-align:right'>{RETRY_MAX_TOK_PER_S_BY_BUCKET[b]:.1f}</td>"
            f"</tr>"
        )
    return f"""
    <div class="card">
      <div class="card-title">Latency · step definitions</div>
      <p style="font-size:12px;color:#5c7999;margin-bottom:10px">
      Per-step wall-clock latency is parsed from the MLflow trace spans by
      <code>hg_ds_evals.preprocessing.latency.extract_latency_breakdown</code>.
      Each step is matched to a specific LangGraph span by name; spans are
      bucketed so the resulting groups are <strong>mutually non-overlapping</strong>
      and sum exactly to <code>lat_total_ms</code> (the duration of the
      root <code>eval_item</code> / <code>eval.predict_item</code> span).
      <em>Overhead</em> is the residual that captures LangGraph wiring,
      <code>tools_condition</code> routing decisions, and any
      eval-framework processing that runs <em>outside</em> the agent's
      LangGraph (e.g. trace serialization happening after the agent
      produced its final answer).</p>

      <p style="font-size:12px;color:#b46504;margin-bottom:10px">
      <strong>Note: these surfaces are approximations.</strong> Retry
      detection is a tok/s-based heuristic with no ground-truth signal
      from the spans themselves. Use the numbers as <em>indicators</em>,
      not facts.</p>

      <dl class="doc-dl">
        <dt><strong>Routing (<code>main_agent</code>)</strong></dt>
        <dd>Every span named <code>main_agent</code> at the top level
            (excluding any that nests inside a sub-agent). In CZKB this
            normally fires twice per trace: once at the start when the
            router LLM decides which sub-agent to invoke, and once at
            the end when control returns. The bulk of the time is one
            <code>ChatDatabricks</code> call.</dd>

        <dt><strong>Planning LLM</strong></dt>
        <dd>Every <code>llm</code>-named span inside a sub-agent
            (<code>daily_banking_agent</code>, <code>hg-invest-phase2</code>)
            <em>except</em> the last one in temporal order. These are the
            LLM calls that decide which tool to call next — their output
            messages carry <code>tool_calls</code> and feed into a
            <code>tools_condition</code> branch. Multiple iterations are
            summed.</dd>

        <dt><strong>KB retrieve</strong></dt>
        <dd>Every span named <code>retrieve</code> inside a
            <code>knowledge_search</code> tool invocation. Contains the
            parallel vector-DB queries (<code>HTTP POST
            /admin/knowledge-base/query</code>) — the dominant cost here
            is network round-trips, not local computation.</dd>

        <dt><strong>KB prune</strong></dt>
        <dd>Every <code>prune</code> span inside <code>knowledge_search</code>.
            Filters and deduplicates the candidate pool from KB retrieve.
            Pure local computation — typically tens of milliseconds.</dd>

        <dt><strong>KB rerank</strong></dt>
        <dd>Every <code>rerank</code> span inside <code>knowledge_search</code>.
            This is an LLM-based reranker (its time includes the inner
            <code>ChatDatabricks</code> call), <strong>not</strong> the
            generation step. Often the single largest sub-step of
            <code>knowledge_search</code>.</dd>

        <dt><strong>Tools (non-KB)</strong></dt>
        <dd>Every span typed <code>TOOL</code> whose name is not
            <code>knowledge_search</code>. Empty for KB-only runs.
            Becomes meaningful in the API report where tools like
            <code>george-gcg-product_getLoans</code> are invoked.</dd>

        <dt><strong>Generation LLM</strong></dt>
        <dd>The last <code>llm</code> span (by end time) inside a
            sub-agent invocation — the LLM call that produces the final
            user-visible answer and is immediately followed by
            <code>agent_answer</code>. If the sub-agent makes no tool
            call at all, this is also the only LLM span for that
            sub-agent. Typically the largest single contributor and a
            primary target for optimization (prompt length, model choice,
            reasoning budget).</dd>

        <dt><strong>Overhead</strong></dt>
        <dd>The residual: <code>lat_total_ms</code> minus the sum of the
            above. Includes LangGraph DAG-wiring, <code>tools_condition</code>
            edge spans, and (importantly) any eval-framework time spent
            <em>after</em> the agent produced its answer but
            <em>before</em> the root span closed — e.g. trace
            serialization. A high overhead share for a run is a signal
            that the bottleneck is outside the agent itself.</dd>

        <dt><strong>Retry overhead</strong> <em style="color:#b46504">(approximation, overlapping with LLM buckets)</em></dt>
        <dd>Estimated wall-clock time spent in silent SDK retry / throttle
            back-off. The Databricks / LangChain client swallows 429s and
            retries internally, so the back-off sleep is invisibly
            included in the <code>CHAT_MODEL</code> span duration — there
            is no explicit retry marker on the span. The detector
            (<code>_evaluate_chat_model_for_retry</code> in
            <code>hg_ds_evals.preprocessing.latency</code>) flags a call
            as a probable retry when its wall time and observed
            throughput cross <strong>bucket-specific</strong> thresholds
            — flat thresholds mis-fired on the reranker (whose output is
            intrinsically tiny, so tok/s is naturally low). The current
            defaults are:
            <table class="tbl" style="max-width:520px;margin-top:8px;font-size:11px">
              <thead><tr>
                <th>Bucket</th>
                <th style='text-align:right'>min duration</th>
                <th style='text-align:right'>max tok/s</th>
              </tr></thead>
              <tbody>{rows}</tbody>
            </table>
            <p style='margin-top:8px'>Estimated overhead =
            <code>duration − output_tokens / {_BASE}</code>
            (baseline expected gen time, clamped at 0).
            <strong>Retry overhead overlaps with the LLM step buckets above</strong>
            (each flagged call's wall time is already counted under
            routing / planning / generation / kb_rerank depending on
            where the <code>CHAT_MODEL</code> span sits), so it is
            reported as its own row below the breakdown table and is
            <em>not</em> summed into the total. Read it as "of the LLM
            time, how much was probably back-off, not real work?".</p></dd>
      </dl>
      <p style="font-size:12px;color:#5c7999;margin-top:10px">
      <strong>Mean / 95% CI / p50 / p95.</strong> The summary card shows
      one row per step plus the run total. The 95% CI is a bootstrap
      percentile CI on the mean (1000 resamples, seed 0) — the
      right-skewed distribution of latency violates the normality
      assumption of t-based CIs, so the bootstrap is the correct
      choice. p50 / p95 are per-case order statistics on the step's
      duration (not on the bootstrap distribution); p95 is the practical
      tail metric — it's what users experience as "slow".</p>
    </div>
    """


def _foldable_card(title_html: str, desc_html: str, body_html: str,
                     *, expanded: bool = False) -> str:
    """Render a fold/unfold card. Default collapsed — clicking the title
    toggles the body. Title + desc stay visible when collapsed."""
    cls = "card foldable" + ("" if expanded else " collapsed")
    desc_block = (f"<p class='card-desc'>{desc_html}</p>") if desc_html else ""
    return (f"<div class='{cls}'>"
            f"<div class='card-title'>{title_html}</div>"
            f"{desc_block}"
            f"<div class='card-body'>{body_html}</div>"
            f"</div>")


def _crosstab_clickable(sub: "pd.DataFrame", row_col: str, col_col: str,
                          context_label: str) -> str:
    """Render a pd.crosstab as an HTML table whose non-zero counts are links
    that filter the Test Cases tab via the judge-eval-link mechanism."""
    if row_col not in sub.columns or col_col not in sub.columns or len(sub) == 0:
        return "<p class='placeholder'>(no rows)</p>"
    ct = pd.crosstab(sub[row_col], sub[col_col])
    if ct.empty:
        return "<p class='placeholder'>(empty)</p>"
    cols = list(ct.columns)
    head = "<tr><th>" + _h(row_col) + " \\ " + _h(col_col) + "</th>" + "".join(
        f"<th style='text-align:right'>{_h(c)}</th>" for c in cols
    ) + "</tr>"
    body = ""
    for idx, row in ct.iterrows():
        cells = []
        for c in cols:
            count = int(row[c]) if c in row else 0
            if count > 0 and "test_case_id" in sub.columns:
                cell_mask = (sub[row_col] == idx) & (sub[col_col] == c)
                cell_ids = sub.loc[cell_mask, "test_case_id"].astype(str).tolist()
                ids_attr = _h(json.dumps(cell_ids))
                label = _h(f"{row_col}={idx} & {col_col}={c} ({context_label})")
                cells.append(
                    f"<td style='text-align:right'>"
                    f"<a href='#' class='judge-eval-link' data-ids='{ids_attr}' "
                    f"data-label='{label}' "
                    f"title='Show these {count} cases in the Test Cases tab'>{count}</a></td>"
                )
            else:
                cells.append(f"<td style='text-align:right'>{count}</td>")
        body += f"<tr><td><strong>{_h(idx)}</strong></td>{''.join(cells)}</tr>"
    return f"<table class='tbl'><thead>{head}</thead><tbody>{body}</tbody></table>"


def _judge_eval_html(df: pd.DataFrame) -> str:
    """Render the Judge Eval tab: schema-contract violations + claim/score crosstabs.

    Logic mirrors the `Schema-contract violations` and `Compare counts of claims
    and judge score` cells in `czkb_001_results_viewer_local.ipynb`.
    """
    SENTINEL = "Answer is not available based on given information"

    def _isn_true(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series([False] * len(df), index=df.index)
        return df[col].astype(str).str.lower().eq("true")

    def _nonempty(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series([False] * len(df), index=df.index)
        return df[col].fillna("").astype(str).str.strip().ne("")

    ctx_empty    = df.get("kb_context_empty", pd.Series([False] * len(df), index=df.index))
    ctx_nonempty = ~ctx_empty
    pool_flag    = _isn_true("retrieved_pool_inadequacy_identified")
    ref_wrong    = _isn_true("expected_reference_looks_wrong")
    in_kb_scope  = (df.get("case_scope", pd.Series([""] * len(df), index=df.index))
                      .isin({"kb", "kb_and_api"}))
    adeq_low     = pd.to_numeric(
        df.get("optimal_retrieved_context_adequacy_score", pd.Series([2] * len(df), index=df.index)),
        errors="coerce",
    ) <= 1

    opt_sel  = df.get("_optimal_enum_selection", pd.Series([[]] * len(df), index=df.index))
    pool_ids = df.get("_post_prune_enum_ids",    pd.Series([[]] * len(df), index=df.index))
    opt_outside_pool = pd.Series(
        [not set(o or []).issubset(set(p or [])) for o, p in zip(opt_sel, pool_ids)],
        index=df.index,
    )

    hc = df.get("_hallucinated_claims", pd.Series([[]] * len(df), index=df.index))
    uf = df.get("_unavailable_facts_in_selected_context", pd.Series([[]] * len(df), index=df.index))
    is_sentinel = lambda s: s.map(lambda v: v == [SENTINEL])

    oe_len = df.get("overall_explanation", pd.Series([""] * len(df), index=df.index)) \
        .fillna("").astype(str).str.len()

    g = lambda col: pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series([np.nan]*len(df), index=df.index)
    cnt = lambda col: df[col] if col in df.columns else pd.Series([0]*len(df), index=df.index)

    checks = {
        "groundedness<=1 & ctx non-empty but hallucinated_claims=[]":
            (g("answer_groundedness_score") <= 1) & ctx_nonempty & (cnt("hallucinated_claims_cnt") == 0),
        "groundedness==2 but hallucinated_claims non-empty":
            (g("answer_groundedness_score") == 2) & (cnt("hallucinated_claims_cnt") > 0),
        "alignment==2 but missing_facts non-empty":
            (g("answer_expected_alignment_score") == 2) & (cnt("missing_facts_cnt") > 0),
        "sufficiency==2 but unavailable_facts non-empty":
            (g("selected_context_sufficiency_score") == 2) & (cnt("unavailable_facts_cnt") > 0),
        "selection_relevance==2 but extra_or_distracting_enums non-empty":
            (g("selection_semantic_relevance_score") == 2) & (cnt("extra_distracting_enums_cnt") > 0),
        "ctx empty but hallucinated_claims is not the placeholder array":
            ctx_empty & ~is_sentinel(hc),
        "ctx empty but unavailable_facts is not the placeholder array":
            ctx_empty & ~is_sentinel(uf),
        "ctx non-empty but hallucinated_claims is the placeholder array":
            ctx_nonempty & is_sentinel(hc),
        "ctx non-empty but unavailable_facts is the placeholder array":
            ctx_nonempty & is_sentinel(uf),
        "retrieved_pool_inadequacy=True but contract conditions not met":
            pool_flag & ~(in_kb_scope & adeq_low),
        "retrieved_pool_inadequacy=False but contract conditions met":
            ~pool_flag & in_kb_scope & adeq_low,
        "retrieved_pool_inadequacy=True but description empty":
            pool_flag & ~_nonempty("retrieved_pool_inadequacy_description"),
        "retrieved_pool_inadequacy=False but description non-empty":
            ~pool_flag & _nonempty("retrieved_pool_inadequacy_description"),
        "expected_reference_looks_wrong=True but issue description empty":
            ref_wrong & ~_nonempty("expected_reference_issue_description"),
        "expected_reference_looks_wrong=True but test_case suggestion empty":
            ref_wrong & ~_nonempty("test_case_improvement_suggestion"),
        "expected_reference_looks_wrong=False but issue description non-empty":
            ~ref_wrong & _nonempty("expected_reference_issue_description"),
        "optimal_enum_selection contains IDs outside post_prune pool":
            opt_outside_pool,
        "case_scope value not in declared enum":
            ~df.get("case_scope", pd.Series([""]*len(df), index=df.index))
              .isin({"kb", "api", "kb_and_api", "out_of_scope", "ambiguous"}),
    }

    # Sort by count desc. Each row keeps the matching test_case_id list so the
    # rendered count can become a clickable filter link.
    tc_ids = df["test_case_id"].astype(str) if "test_case_id" in df.columns else pd.Series([""]*len(df), index=df.index)
    rows = []
    for name, mask in checks.items():
        m = mask.fillna(False)
        ids = tc_ids[m].tolist()
        rows.append((name, len(ids), float(m.mean()) if len(m) else 0.0, ids))
    rows.sort(key=lambda x: -x[1])

    schema_rows_html = ""
    for name, n, rate, ids in rows:
        cls = "num-bad" if n > 0 else "num-good"
        if n > 0:
            ids_attr = _h(json.dumps(ids))
            label = _h(f"check: {name}")
            count_cell = (
                f"<td class='dim-count {cls}' style='text-align:right'>"
                f"<a href='#' class='judge-eval-link' data-ids='{ids_attr}' "
                f"data-label='{label}' "
                f"title='Show these {n} cases in the Test Cases tab'>{n}</a></td>"
            )
        else:
            count_cell = (
                f"<td class='dim-count {cls}' style='text-align:right'>{n}</td>"
            )
        schema_rows_html += (
            f"<tr><td>{_h(name)}</td>{count_cell}"
            f"<td style='text-align:right'>{rate:.1%}</td></tr>"
        )
    schema_table = (
        "<table class='tbl'><thead>"
        "<tr><th>check</th><th style='text-align:right'>n</th><th style='text-align:right'>rate</th></tr>"
        f"</thead><tbody>{schema_rows_html}</tbody></table>"
    )

    # Crosstabs
    pairs = [
        ("hallucinated_claims_cnt",     "answer_groundedness_score"),
        ("missing_facts_cnt",           "answer_expected_alignment_score"),
        ("unavailable_facts_cnt",       "selected_context_sufficiency_score"),
        ("extra_distracting_enums_cnt", "selection_semantic_relevance_score"),
    ]
    grounded = df[~ctx_empty]
    no_ctx   = df[ctx_empty]

    crosstabs_html = ""
    for cnt_col, score_col in pairs:
        sub_body = (
            f"<div class='judge-eval-sub'>kb_context present (n={len(grounded)})</div>"
            f"{_crosstab_clickable(grounded, cnt_col, score_col, 'kb_context present')}"
        )
        if len(no_ctx):
            sub_body += (
                f"<div class='judge-eval-sub' style='margin-top:10px'>"
                f"kb_context empty (n={len(no_ctx)} — routing/agent path)</div>"
                f"{_crosstab_clickable(no_ctx, cnt_col, score_col, 'kb_context empty')}"
            )
        crosstabs_html += _foldable_card(
            f"{_h(cnt_col)} vs {_h(score_col)}",
            "",
            sub_body,
        )

    schema_card = _foldable_card(
        "Schema-contract violations",
        "Hard rules from the YAML <code>output_schema</code> + "
        "<code>additional_instructions</code>. Any non-zero count is a judge bug.",
        schema_table,
    )
    claims_card = _foldable_card(
        "Compare counts of claims and judge scores",
        "Each pair: judge claim list vs the rubric score it should track. "
        "Rows split by <code>kb_context_empty</code> — empty-context cases reflect "
        "routing/agent issues rather than judge groundedness.",
        crosstabs_html,
    )
    rel2_card = _judge_eval_rel2_html(df)
    return schema_card + claims_card + rel2_card


def _judge_eval_rel2_html(df: pd.DataFrame) -> str:
    """Compare Rel2 score and judge output (mirrors notebook cells 18–20).

    Funnel-filters to KB-scope queries that actually reached KB search, then
    cross-tabs the rounded Rel2 score against three judge scorers. Cells
    are clickable. A final illogical-combinations link surfaces obvious
    Rel2/judge disagreements for inspection.
    """
    if "enum_relevance_score" not in df.columns or "query_scope" not in df.columns:
        return ""
    rel2 = pd.to_numeric(df["enum_relevance_score"], errors="coerce")
    bucket = pd.Series("kb routed", index=df.index)
    bucket[df["query_scope"] != "kb"] = "not kb scope (drop)"
    if "kb_context_empty" in df.columns:
        bucket[(df["query_scope"] == "kb") & df["kb_context_empty"]] = (
            "kb scope, routing failure"
        )
    bucket_counts = bucket.value_counts()
    funnel_rows = ""
    for name, n in bucket_counts.items():
        ids = df.loc[bucket == name, "test_case_id"].astype(str).tolist() \
            if "test_case_id" in df.columns else []
        n_int = int(n)
        if n_int > 0 and ids:
            ids_attr = _h(json.dumps(ids))
            label = _h(f"Rel2 funnel: {name}")
            cell = (f"<a href='#' class='judge-eval-link' data-ids='{ids_attr}' "
                    f"data-label='{label}' title='Show these {n_int} cases'>{n_int}</a>")
        else:
            cell = str(n_int)
        funnel_rows += (
            f"<tr><td>{_h(name)}</td>"
            f"<td style='text-align:right'>{cell}</td></tr>"
        )
    funnel_table = (
        "<table class='tbl'><thead>"
        "<tr><th>bucket</th><th style='text-align:right'>n</th></tr>"
        f"</thead><tbody>{funnel_rows}</tbody></table>"
    )

    sub = df[bucket == "kb routed"].copy()
    if not sub.empty:
        sub["rel2"] = rel2[bucket == "kb routed"].round(2)

    judge_cols = [
        "selection_semantic_relevance_score",
        "selected_context_sufficiency_score",
        "optimal_retrieved_context_adequacy_score",
    ]
    crosstabs_html = ""
    for jcol in judge_cols:
        if jcol not in sub.columns or sub.empty:
            continue
        crosstabs_html += (
            f"<div class='judge-eval-sub' style='margin-top:14px'>"
            f"rel2 (rounded) vs <code>{_h(jcol)}</code> "
            f"— kb-routed n={len(sub)}</div>"
            f"{_crosstab_clickable(sub, 'rel2', jcol, 'rel2 vs ' + jcol)}"
        )

    # Illogical combinations: Rel2 and the judge disagree strongly.
    illogical_link = ""
    if not sub.empty:
        sel_score = pd.to_numeric(sub.get("selection_semantic_relevance_score"),
                                    errors="coerce")
        illogical_mask = (
            ((sub["rel2"] >= 0.99) & (sel_score <= 1))
            | ((sub["rel2"] <= 0.01) & (sel_score == 2))
        )
        illogical_ids = sub.loc[illogical_mask.fillna(False), "test_case_id"] \
            .astype(str).tolist() if "test_case_id" in sub.columns else []
        n = len(illogical_ids)
        if n > 0:
            ids_attr = _h(json.dumps(illogical_ids))
            label = _h("Rel2 vs judge: illogical combinations")
            illogical_link = (
                f"<a href='#' class='judge-eval-link' data-ids='{ids_attr}' "
                f"data-label='{label}' title='Show these {n} cases'>{n}</a>"
            )
        else:
            illogical_link = "0"
    illogical_block = (
        "<p style='font-size:12px;color:#537090;margin-top:14px'>"
        "<strong>Illogical combinations</strong> — rows where Rel2 ≥ 0.99 but "
        "<code>selection_semantic_relevance ≤ 1</code>, OR Rel2 ≤ 0.01 but "
        "<code>selection_semantic_relevance == 2</code>. "
        f"Count: {illogical_link}"
        "</p>"
    )

    body = (
        "<div class='judge-eval-sub'>Funnel — query-scope filter</div>"
        f"{funnel_table}"
        f"{crosstabs_html}"
        f"{illogical_block}"
    )

    return _foldable_card(
        "Compare Rel2 score and judge output",
        "Filter to KB-scope queries that actually reached KB search, then compare the "
        "pre-existing Rel2 score against the judge's selection / context-quality "
        "dimensions. Click any cell or count to drill into the matching test cases.",
        body,
    )


def _build_kb_data(df: pd.DataFrame) -> dict:
    """Aggregate every distinct KB entry referenced by any case, keyed by
    ``enum_id``. Each entry holds:

      - ``cz`` / ``en``       — language descriptions parsed from the
        ``reranked_enums_kb_*`` and ``post_prune_candidates_kb_*`` columns.
        The native-language slot is named ``cz`` regardless of LANG — for
        SK runs it carries the Slovak text picked up from ``_sk``-suffixed
        columns. Keeping one slot name lets the CSS/JS language toggle
        stay shared between CZ and SK reports.
        We ingest from both reranked + post-prune columns so an expected
        ENUM that's only ever present in the post-prune pool (never
        reranker-selected) still picks up its description.
      - ``reranked_cases``    — cases that picked this entry into their
        reranked selection.
      - ``expected_cases``    — cases that had this entry in their gold
        ``expected_enums`` list, regardless of whether the reranker
        actually selected it.

    The KB tab's per-case filter uses these two sets to decide which
    rows to show when filtered to a single case, and to colour each row
    by its status for that case:
      - in BOTH expected and reranked → correctly selected (green)
      - in expected only              → missed by reranker (blue)
      - in reranked only              → wrongly selected / distractor (red)

    Empty dict when none of the relevant columns are present.
    """
    # The native-language column is suffixed with the LANG code on the
    # producing notebook side (`_cz` for CZKB, `_sk` for SKKB). Prefer the
    # current LANG's suffix, but fall back to the other so a checkpoint
    # produced by the "other" pipeline still renders without forcing the
    # caller to set --lang correctly.
    def _pick_native(prefix: str) -> str | None:
        primary = f"{prefix}_{LANG}"
        if primary in df.columns:
            return primary
        for alt in ("cz", "sk"):
            if alt == LANG:
                continue
            candidate = f"{prefix}_{alt}"
            if candidate in df.columns:
                return candidate
        return None

    native_col_re = _pick_native("reranked_enums_kb")
    en_col_re     = "reranked_enums_kb_en"        if "reranked_enums_kb_en"        in df.columns else None
    native_col_pp = _pick_native("post_prune_candidates_kb")
    en_col_pp     = "post_prune_candidates_kb_en" if "post_prune_candidates_kb_en" in df.columns else None
    if not (native_col_re or en_col_re or native_col_pp or en_col_pp):
        return {}

    def _unescape(desc: str) -> str:
        # Upstream JSON often double-escapes whitespace (\\n / \\r / \\t
        # in the source instead of \n / \r / \t), so json.loads gives
        # us a literal backslash+letter. Convert to real whitespace so
        # CSS white-space:pre-wrap renders proper line breaks.
        return (desc.replace("\\r\\n", "\n")
                       .replace("\\n", "\n")
                       .replace("\\r", "\n")
                       .replace("\\t", "\t"))

    # ── Step 1: build a global enum_id → {cz, en} description index ──────
    # Walk every column that could carry a description; first non-empty
    # value per (enum, lang) wins. This lets us back-fill descriptions
    # for expected ENUMs that were never reranker-selected anywhere.
    descriptions: dict[str, dict[str, str]] = {}
    for col, lang in ((native_col_re, "cz"), (en_col_re, "en"),
                       (native_col_pp, "cz"), (en_col_pp, "en")):
        if not col:
            continue
        for raw in df[col].dropna():
            try:
                items = json.loads(raw) if isinstance(raw, str) else raw
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(items, list):
                continue
            for it in items:
                if not isinstance(it, dict):
                    continue
                eid = (it.get("enum_id") or "").strip()
                if not eid:
                    continue
                desc = it.get("description") or ""
                if not desc:
                    continue
                bucket = descriptions.setdefault(eid, {"cz": "", "en": ""})
                if not bucket[lang]:
                    bucket[lang] = _unescape(desc)

    # ── Step 2: tag expected / reranked membership per enum per case ─────
    kb: dict[str, dict] = {}

    def _ensure(eid: str) -> dict:
        return kb.setdefault(eid, {
            "cz": descriptions.get(eid, {}).get("cz", ""),
            "en": descriptions.get(eid, {}).get("en", ""),
            "reranked_cases": set(),
            "expected_cases": set(),
        })

    for _, r in df.iterrows():
        tc = "" if pd.isna(r.get("test_case_id")) else str(r.get("test_case_id"))
        if not tc:
            continue

        reranked_ids = r.get("_reranked_enum_ids") or []
        if isinstance(reranked_ids, list):
            for eid in reranked_ids:
                eid = str(eid).strip()
                if eid:
                    _ensure(eid)["reranked_cases"].add(tc)

        expected_ids = r.get("_expected_enums") or []
        if isinstance(expected_ids, list):
            for eid in expected_ids:
                eid = str(eid).strip()
                if eid:
                    _ensure(eid)["expected_cases"].add(tc)

    # Stabilize case-list order — by numeric tail of test_case_id, then string.
    def _tc_key(t: str):
        m = re.search(r"\d+", t)
        return (int(m.group()) if m else 10**9, t)
    for v in kb.values():
        v["reranked_cases"] = sorted(v["reranked_cases"], key=_tc_key)
        v["expected_cases"] = sorted(v["expected_cases"], key=_tc_key)
    return kb


def _render_kb_html(kb_data: dict) -> str:
    if not kb_data:
        return (
            "<div class='card'>"
            "<div class='card-title'>Knowledge base</div>"
            f"<p class='placeholder'>No <code>reranked_enums_kb_{LANG}</code> / "
            "<code>reranked_enums_kb_en</code> columns in the checkpoint.</p>"
            "</div>"
        )
    # Sort by total unique cases (union of expected + reranked), then by
    # enum_id. This keeps the most-referenced entries at the top whether
    # they were referenced as gold expectations, selections, or both.
    def _sort_key(item):
        eid, entry = item
        all_cases = set(entry["reranked_cases"]) | set(entry["expected_cases"])
        return (-len(all_cases), eid)
    items = sorted(kb_data.items(), key=_sort_key)
    # Same numeric-tail-aware key used by _build_kb_data for stable ordering.
    def _tc_key(t: str):
        m = re.search(r"\d+", t)
        return (int(m.group()) if m else 10**9, t)
    rows_html = ""
    for eid, entry in items:
        reranked_cases = entry["reranked_cases"]
        expected_cases = entry["expected_cases"]
        all_cases = sorted(set(reranked_cases) | set(expected_cases), key=_tc_key)
        n_total      = len(all_cases)
        n_reranked   = len(reranked_cases)
        n_expected   = len(expected_cases)
        all_ids_attr      = _h(json.dumps(all_cases))
        reranked_ids_attr = _h(json.dumps(reranked_cases))
        expected_ids_attr = _h(json.dumps(expected_cases))
        label = _h(f"KB enum: {eid}")
        # Case-count link surfaces the union of cases (most useful for
        # drilling-in); the tooltip explains the breakdown so the reader
        # knows how many of those cases had it as expected vs selected.
        cases_link = (
            f"<a href='#' class='judge-eval-link' "
            f"data-ids='{all_ids_attr}' data-label='{label}' "
            f"title='Show the {n_total} cases that referenced this entry "
            f"({n_expected} as expected · {n_reranked} as reranker selection)'>"
            f"{n_total} {'case' if n_total == 1 else 'cases'}</a>"
        )
        cz_html = (
            _h(entry["cz"]) if entry["cz"]
            else f"<em class='lang-fallback'>(no {LANG_LABEL_UPPER} text)</em>"
        )
        en_html = _h(entry["en"]) if entry["en"] else "<em class='lang-fallback'>(no EN text)</em>"
        # Searchable haystack: enum_id + both descriptions, lower-cased.
        search_text = _h((eid + " " + entry["cz"] + " " + entry["en"]).lower())
        rows_html += (
            f"<div class='kb-row collapsed' data-search='{search_text}' "
            f"data-ids='{all_ids_attr}' "
            f"data-reranked-cases='{reranked_ids_attr}' "
            f"data-expected-cases='{expected_ids_attr}'>"
            f"<div class='kb-row-head'>"
            f"<span class='kb-chev' aria-hidden='true'>▶</span>"
            f"<code class='kb-id'>{_h(eid)}</code>"
            # Empty badge — JS fills in (and the row class drives the colour)
            # only when the KB tab is filtered to a specific case.
            f"<span class='kb-status-badge' aria-hidden='true'></span>"
            f"<span class='kb-cases-count'>{cases_link}</span>"
            f"</div>"
            f"<div class='kb-desc lang-cz'>{cz_html}</div>"
            f"<div class='kb-desc lang-en'>{en_html}</div>"
            f"</div>"
        )
    return (
        "<div class='card'>"
        f"<div class='card-title'>Knowledge base — {len(items)} unique entries appearing in reranked context</div>"
        "<p style='font-size:12px;color:#537090;margin-bottom:10px'>"
        f"Use the {LANG_LABEL_UPPER} / EN switch on the right to flip language. "
        "Click the case-count link to filter the Test Cases tab to those rows."
        "</p>"
        "<div class='kb-toolbar'>"
        "<input id='kb-search' type='text' class='case-search kb-search' "
        "placeholder='Search enum_id or description…'>"
        "<span id='kb-count' class='kb-count'></span>"
        "<span class='lang-switch kb-lang-switch' role='group' aria-label='language'>"
        f"<button type='button' class='lang-btn' data-lang='cz'>{LANG_LABEL_UPPER}</button>"
        "<button type='button' class='lang-btn active' data-lang='en'>EN</button>"
        "</span>"
        "<button type='button' id='kb-reset' class='kb-reset' "
        "title='Show all KB entries (clear test-case filter)'>Show all</button>"
        "</div>"
        "<div id='kb-case-banner' class='kb-case-banner'></div>"
        f"<div id='kb-list' class='kb-list'>{rows_html}</div>"
        "</div>"
    )


def _build_rel2_expert_scatter(df: pd.DataFrame):
    """Scatter of Rel2 × expert_score, colored by judge weighted_avg.

    Expert score is integer 1-10 and Rel2 clusters on a few discrete values,
    so points pile on top of each other. Add a small deterministic jitter
    so every test case is visible and individually hoverable. The hover
    tooltip still shows the *original* (un-jittered) values.
    """
    if "expert_score" not in df.columns or "enum_relevance_score" not in df.columns:
        return None
    sub = df[df["expert_score"].notna() & df["enum_relevance_score"].notna()].copy()
    if sub.empty:
        return None
    rng = np.random.default_rng(42)
    x_jit = 0.018
    y_jit = 0.18
    sub["_rel2_plot"] = sub["enum_relevance_score"] + rng.uniform(-x_jit, x_jit, len(sub))
    sub["_expert_plot"] = sub["expert_score"] + rng.uniform(-y_jit, y_jit, len(sub))

    hover_cols = [c for c in ("test_case_id", "query_scope", "root_cause_category",
                               "weighted_avg", "enum_relevance_score", "expert_score")
                   if c in sub.columns]
    fig = px.scatter(
        sub, x="_rel2_plot", y="_expert_plot",
        color="weighted_avg", color_continuous_scale=GE_BLUE_SCALE,
        hover_data={c: True for c in hover_cols} | {"_rel2_plot": False, "_expert_plot": False},
        labels={"_rel2_plot": "Rel2 score (0-1)",
                "_expert_plot": "Expert score (1-10)",
                "weighted_avg": "judge w.avg",
                "enum_relevance_score": "Rel2",
                "expert_score": "Expert"},
    )
    fig.update_traces(marker=dict(size=9, line=dict(color="#ffffff", width=1),
                                   opacity=0.85))
    _style_fig(fig, height=400)
    fig.update_layout(margin=dict(t=20, b=50, l=60, r=20),
                      coloraxis_colorbar=dict(tickfont=dict(color=GE_MUTED, size=10),
                                              thickness=10, len=0.8, outlinewidth=0,
                                              title=dict(text="judge<br>w.avg",
                                                          font=dict(color=GE_MUTED, size=10))))
    fig.update_xaxes(range=[-0.05, 1.05], title="Rel2 score (0-1)")
    fig.update_yaxes(range=[0.5, 10.5], dtick=1, title="Expert score (1-10)")
    return fig


def _build_rel2_wavg_scatter(df: pd.DataFrame):
    """Scatter of Rel2 × judge weighted_avg, colored by expert_score."""
    if "weighted_avg" not in df.columns or "enum_relevance_score" not in df.columns:
        return None
    sub = df[df["weighted_avg"].notna() & df["enum_relevance_score"].notna()].copy()
    if sub.empty:
        return None
    rng = np.random.default_rng(42)
    x_jit = 0.018
    y_jit = 0.03
    sub["_rel2_plot"] = sub["enum_relevance_score"] + rng.uniform(-x_jit, x_jit, len(sub))
    sub["_wavg_plot"] = sub["weighted_avg"] + rng.uniform(-y_jit, y_jit, len(sub))

    hover_cols = [c for c in ("test_case_id", "query_scope", "root_cause_category",
                               "weighted_avg", "enum_relevance_score", "expert_score")
                   if c in sub.columns]
    has_expert = "expert_score" in sub.columns and sub["expert_score"].notna().any()
    fig = px.scatter(
        sub, x="_rel2_plot", y="_wavg_plot",
        color="expert_score" if has_expert else None,
        color_continuous_scale=GE_BLUE_SCALE,
        range_color=[1, 10] if has_expert else None,
        hover_data={c: True for c in hover_cols} | {"_rel2_plot": False, "_wavg_plot": False},
        labels={"_rel2_plot": "Rel2 score (0-1)",
                "_wavg_plot": "Judge weighted_avg",
                "weighted_avg": "judge w.avg",
                "enum_relevance_score": "Rel2",
                "expert_score": "Expert"},
    )
    fig.update_traces(marker=dict(size=9, line=dict(color="#ffffff", width=1),
                                   opacity=0.85))
    _style_fig(fig, height=400)
    cb = (dict(tickfont=dict(color=GE_MUTED, size=10),
                thickness=10, len=0.8, outlinewidth=0,
                title=dict(text="expert", font=dict(color=GE_MUTED, size=10)))
            if has_expert else None)
    fig.update_layout(margin=dict(t=20, b=50, l=60, r=20),
                      coloraxis_colorbar=cb)
    fig.add_hline(y=PASS_THRESHOLD, line_color=GE_RED, line_width=1,
                    annotation_text=f"pass ≥ {PASS_THRESHOLD}",
                    annotation_position="top left",
                    annotation_font=dict(color=GE_RED, size=10))
    fig.update_xaxes(range=[-0.05, 1.05], title="Rel2 score (0-1)")
    fig.update_yaxes(range=[-0.05, 1.05], dtick=0.1, title="Judge weighted_avg (0-1)")
    return fig


def _plot(fig, first: bool = False, *, interactive: bool = True,
          div_id: str | None = None) -> str:
    """Render a fig without its own plotly.js — the script tag lives in <head>.

    ``interactive=True`` shows a lightweight hover-activated modebar with the
    **reset axes** button so users don't have to refresh the page after zooming.

    Pass ``div_id`` to get a stable DOM id so JS can attach listeners (click,
    hover, etc.) to the chart later.
    """
    if fig is None:
        return ""
    config: dict = {"displaylogo": False}
    if interactive:
        config.update({
            "displayModeBar": "hover",
            "modeBarButtonsToRemove": [
                "lasso2d", "select2d", "toggleSpikelines",
                "hoverClosestCartesian", "hoverCompareCartesian",
            ],
            "responsive": True,
        })
    else:
        config["displayModeBar"] = False
    kwargs = dict(
        include_plotlyjs=False,
        full_html=False,
        config=config,
    )
    if div_id is not None:
        kwargs["div_id"] = div_id
    return fig.to_html(**kwargs)


CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; font-size: 14px;
       line-height: 1.5; color: #0a285c; background: #f4f6fa; }
code { font-family: monospace; font-size: 12px; background: #e7effd; padding: 1px 4px; border-radius: 4px; }
.sticky-top { position: sticky; top: 0; z-index: 20; }
.report-header { background: #2870ed; color: #fff; padding: 12px 24px;
                  box-shadow: 0 2px 6px rgba(10,40,92,0.15); }
.report-header code { background: rgba(255,255,255,0.18); color: #fff;
                      padding: 1px 6px; border-radius: 4px; font-weight: 500; }
.header-inner { display: flex; align-items: center; justify-content: space-between;
                max-width: 1280px; margin: 0 auto; flex-wrap: wrap; gap: 14px; }
.report-header h1 { font-size: 26px; font-weight: 700; line-height: 1.15;
                      white-space: nowrap; letter-spacing: -0.01em; }
.header-meta { display: flex; gap: 18px; font-size: 12px; color: rgba(255,255,255,.92); flex-wrap: wrap; }
/* Two-column layout for the header meta, with Checkpoint sitting on its
   own row underneath. Falls back to a single column on narrow viewports
   so the columns don't crush each other. */
.header-meta-grid { display: grid;
                     grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
                     row-gap: 4px; column-gap: 32px;
                     align-items: start; margin-top: 6px; }
.header-meta-grid .header-col { display: flex; flex-direction: column; gap: 3px; min-width: 0; }
.header-meta-grid .header-footer { grid-column: 1 / -1; margin-top: 4px;
                                     padding-top: 4px;
                                     border-top: 1px solid rgba(255,255,255,.18); }
.header-meta-grid .header-aux { color: rgba(255,255,255,.7); font-weight: 400; }
.header-meta-grid code { word-break: break-all; }
@media (max-width: 720px) {
  .header-meta-grid { grid-template-columns: minmax(0, 1fr); }
}
.tab-nav { background: #fff; border-bottom: 1px solid #e4eaf0;
           display: flex; padding: 0 24px;
           box-shadow: 0 1px 3px rgba(10,40,92,0.05); }
.tab-nav-inner { display: flex; max-width: 1280px; width: 100%; margin: 0 auto;
                 align-items: stretch; }
.tab-nav .lang-switch { margin-left: auto; }
.tab-btn { background: none; border: none; cursor: pointer; font: inherit;
           color: #5c7999; font-size: 13px; font-weight: 500;
           padding: 12px 18px; border-bottom: 2px solid transparent; }
.tab-btn.active, .tab-btn:hover { color: #1d69ec; border-bottom-color: #1d69ec; }
/* Summary tab gets a soft blue background so it reads as the "home" tab. */
.tab-btn.tab-home { background: #eef4fd; }
.tab-btn.tab-home:hover { background: #e0eafd; }
/* Prompts tab + summary mismatch warning. */
.prompts-card { margin-bottom: 16px; }
.prompts-card .tool-count { color: #5c7999; font-weight: 400; font-size: 12px; }
.prompt-block { margin: 6px 0; border: 1px solid #e4eaf0; border-radius: 4px;
                background: #fafbfc; }
.prompt-block > summary { cursor: pointer; padding: 8px 12px; font-weight: 500;
                          font-family: ui-monospace, Menlo, monospace; font-size: 13px; }
.prompt-block[open] > summary { border-bottom: 1px solid #e4eaf0; background: #f3f6fa; }
.prompt-pre { margin: 0; padding: 12px 14px; font-family: ui-monospace, Menlo, monospace;
              font-size: 12.5px; line-height: 1.5; white-space: pre-wrap; word-break: break-word;
              max-height: 60vh; overflow-y: auto; }
.prompt-empty { color: #5c7999; font-style: italic; margin: 8px 0; }
.prompt-empty-note { color: #5c7999; font-weight: 400; font-size: 12px; }
.prompt-warning-card { border-left: 3px solid #d83a3a; background: #fff5f5; margin-bottom: 16px; }
.prompt-warning-title { color: #b32424; }
.prompt-warning-item { margin-top: 8px; }
.prompt-warning-item + .prompt-warning-item { border-top: 1px dashed #f0c5c5; padding-top: 8px; }
.prompt-warning-tbl { margin-top: 4px; }
.prompt-missing-tag { display: inline-block; padding: 1px 6px; border-radius: 3px;
                       background: #fbe2e2; color: #b32424; font-size: 11px; margin-left: 6px; }
.tab-btn.tab-home.active { background: #d8e6fc; }
/* KB browser tab — very pale hooker-green tint to mark it as the
   reference / browse-knowledge surface (separate from the Summary "home"
   blue and from the diagnostic Findings tabs). */
.tab-btn.tab-kb-highlight { background: #e6f2ea; }
.tab-btn.tab-kb-highlight:hover { background: #d6ead7; }
.tab-btn.tab-kb-highlight.active { background: #c8e0c9; color: #125b2c; border-bottom-color: #125b2c; }
/* Push the eval-tools group (KB + Notes) to the right with grey bg. */
.tab-tools { margin-left: auto; display: flex; background: #f4f6fa;
             border-left: 1px solid #e4eaf0; }
.tab-tools .tab-btn { background: transparent; }
.tab-tools .tab-btn:hover { background: #edf0f4; }
.tab-tools .tab-btn.active { background: #e4eaf0; color: #1d69ec; }
/* Personal / debugging tab (Judge Eval) — sits at the far right with its
   own separator and de-emphasized greyed-out label so it reads as
   "internal use" relative to the rest of the tab bar. */
.tab-personal { display: flex; border-left: 1px solid #c4ccd5; }
.tab-personal .tab-btn.tab-personal-btn { color: #a3b5c9; font-style: italic; }
.tab-personal .tab-btn.tab-personal-btn:hover { color: #5c7999;
             border-bottom-color: #a3b5c9; }
.tab-personal .tab-btn.tab-personal-btn.active { color: #5c7999;
             border-bottom-color: #a3b5c9; background: #f4f6fa; }
.content { max-width: 1280px; margin: 0 auto; padding: 20px 24px; }
.tab-panel { display: none; } .tab-panel.active { display: block; }
.card { background: #fff; border: 1px solid #e4eaf0; border-radius: 10px;
        box-shadow: 0 1px 4px rgba(10,40,92,.07); padding: 18px 20px; margin-bottom: 16px; }
.card-title { font-size: 15px; font-weight: 600; margin-bottom: 12px; }
.dim-cell-name { font-weight: 600; color: #0a285c; }
.dim-cell-desc { font-size: 11px; color: #537090; margin-top: 2px; line-height: 1.4;
                  max-width: 520px; }
.dim-cell-desc code { background: #f3f5f8; padding: 0 4px; border-radius: 3px;
                        font-size: 10.5px; color: #0a285c; }
.scatter-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.scatter-row > .card { margin-bottom: 16px; min-width: 0; }
@media (max-width: 1100px) { .scatter-row { grid-template-columns: 1fr; } }
.metrics { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
           gap: 12px; margin-bottom: 16px; }
.metric { background: #fff; border: 1px solid #e4eaf0; border-radius: 10px; padding: 12px 14px; }
.metric-label { font-size: 11px; color: #a3b5c9; text-transform: uppercase; letter-spacing: .5px; }
.metric-value { font-size: 22px; font-weight: 700; color: #1d69ec; margin-top: 3px; }
.badge { display: inline-block; padding: 1px 8px; border-radius: 10px;
         font-size: 11px; font-weight: 600; min-width: 22px; text-align: center; }
.badge-bad  { background: #fde5e3; color: #cf2a1e; }
.badge-mid  { background: #fef4e2; color: #b46504; }
.badge-good { background: #dff5ea; color: #028661; }
.badge-na   { background: #edf0f4; color: #5c7999; }
.scope-badge { display: inline-block; padding: 1px 8px; border-radius: 10px;
               font-size: 11px; font-weight: 600; }
.scope-kb           { background: #dff5ea; color: #028661; }
.scope-mock_tool    { background: #fef4e2; color: #b46504; }
.scope-dba_no_tools { background: #fde5e3; color: #cf2a1e; }
.scope-main_agent   { background: #e7effd; color: #1d69ec; }
.scope-other        { background: #edf0f4; color: #5c7999; }
/* All hg_invest_* fan-out scopes share one palette to mark the agent at a
   glance; the badge text (hg_invest_kb / hg_invest_no_tools / ...) names the bucket. */
.scope-hg_invest    { background: #ede5fb; color: #5b3da3; }
/* `last_agent` badge — the last agent slug visited in the trace's graph. */
.agent-badge { display: inline-block; padding: 1px 8px; border-radius: 10px;
               font-size: 11px; font-weight: 600; }
.agent-main      { background: #e7effd; color: #1d69ec; }
.agent-dba       { background: #dff5ea; color: #028661; }
.agent-hg_invest { background: #ede5fb; color: #5b3da3; }
.agent-other     { background: #edf0f4; color: #5c7999; }
/* case_scope (judge-output) badge — distinct palette from query_scope. */
.cs-badge { display: inline-block; padding: 1px 8px; border-radius: 10px;
            font-size: 11px; font-weight: 600; }
.cs-kb         { background: #e7effd; color: #1d69ec; }
.cs-kb_and_api { background: #ede5fb; color: #5b3da3; }
.cs-api        { background: #fef4e2; color: #b46504; }
.cs-out_of_scope { background: #fde5e3; color: #cf2a1e; }
.cs-ambiguous  { background: #edf0f4; color: #5c7999; }
.cs-other      { background: #edf0f4; color: #5c7999; }
.tbl { width: 100%; border-collapse: collapse; font-size: 12px; }
.tbl th, .tbl td { border-bottom: 1px solid #edf0f4; padding: 6px 10px; text-align: left;
                   vertical-align: top; max-width: 480px; }
/* Latency breakdown (Summary tab + Test Cases tab). */
.lat-tbl td, .lat-tbl th { vertical-align: middle; }
.lat-total-row td { font-weight: 600; border-top: 2px solid #d3dce6; background: #fafbfd; }
.lat-bar { display: inline-block; vertical-align: middle; margin-left: 6px;
           width: 60px; height: 6px; background: #eef2f7; border-radius: 3px; overflow: hidden; }
.lat-bar-fill { height: 100%; background: #5c7999; }
/* Latency score-box value uses a slightly smaller weight so "28.70s" lines
   up visually with the 0.xx ratios in the neighbouring boxes. */
.score-box .sb-value-lat { font-variant-numeric: tabular-nums;
                            letter-spacing: -0.01em; }
/* 95% CI column. The mini-bar's full width corresponds to the run's mean
   total latency, so a step at 80% of total sits on the right and a step
   at 5% sits squashed on the left — readable at a glance. */
.lat-tbl .lat-ci-cell { width: 180px; min-width: 140px; padding-top: 8px; padding-bottom: 8px;
                         vertical-align: middle; max-width: none; }
.lat-ci-vis { position: relative; height: 10px; width: 100%; margin: 2px 0 4px;
              background: linear-gradient(to right, #eef2f7, #eef2f7); border-radius: 5px; }
.lat-ci-band { position: absolute; top: 3px; height: 4px; background: #7aa6ef;
               border-radius: 2px; min-width: 2px; }
.lat-ci-mark { position: absolute; top: 0; width: 10px; height: 10px;
               margin-left: -5px; background: #135ee2; border: 2px solid #fff;
               border-radius: 50%; box-shadow: 0 0 0 1px #135ee2; }
.lat-ci-text { font-size: 11px; color: #5c7999; font-variant-numeric: tabular-nums;
               text-align: right; }
.lat-ci-text-only { font-size: 11px; color: #5c7999; font-variant-numeric: tabular-nums;
                     display: inline-block; padding: 4px 0; }
.lat-axis-hint { font-size: 9px; color: #a3b5c9; font-weight: 400;
                  text-transform: none; letter-spacing: 0; margin-left: 4px; }
/* "Without throttling" sub-line shown under a step's mean/p95 cell when
   the retry-adjusted value differs from the observed value by ≥1%.
   Subtle amber so the eye treats it as commentary on the main number,
   not a peer of it. */
.lat-adj-line { font-size: 10px; color: #ad5700; font-weight: 500;
                 font-variant-numeric: tabular-nums; margin-top: 1px;
                 line-height: 1.2; }
.lat-adj-delta { color: #057f19; font-weight: 600; }
/* Suspected-retry block sits below the breakdown table — separate styling
   to make the visual separation clear (retries OVERLAP with the LLM step
   buckets and are not summed into the total). */
.lat-retry-row { margin-top: 14px; padding-top: 10px;
                  border-top: 2px dashed #d3dce6; }
.lat-retry-title { font-size: 11px; color: #b46504; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 0.4px;
                    margin-bottom: 6px; }
.lat-retry-tbl th { background: #fef4e2; color: #b46504; }
.lat-retry-tbl td { font-variant-numeric: tabular-nums; }
/* Headline latency card on Summary row 2. Centred big mean + CI bracket. */
.lat-headline-card { align-items: center; text-align: center; }
.lat-headline-card .hc-label { justify-content: center; }
.lat-headline-card .hc-value { color: #135ee2; }
.lat-hc-ci { width: 100%; max-width: 240px; margin: 8px auto 0; }
.lat-hc-ci-bracket { display: flex; align-items: center; justify-content: center;
                      width: 100%; height: 16px; }
.lat-hc-ci-tick { width: 2px; height: 12px; background: #135ee2; border-radius: 1px; }
.lat-hc-ci-line { flex: 1 1 auto; height: 2px; background: #7aa6ef; }
.lat-hc-ci-dot { width: 10px; height: 10px; background: #135ee2; border: 2px solid #fff;
                 border-radius: 50%; box-shadow: 0 0 0 1px #135ee2; flex: 0 0 auto;
                 margin: 0 -2px; }
.lat-hc-ci-labels { display: flex; justify-content: space-between; margin-top: 4px;
                    font-size: 11px; color: #5c7999; font-variant-numeric: tabular-nums; }
.lat-hc-ci-mid { color: #a3b5c9; font-size: 10px; text-transform: uppercase;
                  letter-spacing: 0.5px; font-weight: 600; }
.lat-hc-detail { display: flex; justify-content: center; align-items: center;
                  gap: 8px; flex-wrap: wrap; }
.lat-hc-detail strong { color: #0a285c; font-variant-numeric: tabular-nums; }
.lat-hc-detail-sep { color: #c4cfdc; }
.lat-hc-throttle { font-size: 11px; color: #ad5700; margin-top: 6px;
                   padding-top: 6px; border-top: 1px dashed #f2dca3;
                   justify-content: center; }
.lat-hc-throttle strong { color: #ad5700; }
.lat-hc-label-aux { font-size: 10px; color: #a3b5c9; font-weight: 500;
                     text-transform: uppercase; letter-spacing: 0.04em; }
/* "Without throttling" row directly below the throttling line. Same
   amber palette so the reader sees them as a paired before/after, but
   slightly muted so it doesn't compete with the main observed value. */
.lat-hc-adjusted { font-size: 11px; color: #5c7999; margin-top: 4px;
                    justify-content: center; flex-direction: column;
                    gap: 2px; text-align: center; }
.lat-hc-adjusted-title { color: #ad5700; font-weight: 600;
                          text-transform: uppercase; letter-spacing: 0.04em;
                          font-size: 10px; }
.lat-hc-adjusted strong { color: #0a285c; font-variant-numeric: tabular-nums;
                           font-weight: 700; }
.lat-hc-adjusted-delta { color: #057f19; font-variant-numeric: tabular-nums;
                          font-weight: 600; margin-left: 2px; }
.lat-hc-detail-n { color: #a3b5c9; font-variant-numeric: tabular-nums;
                    margin-left: 8px; }
/* Latency breakdown sub-block, rendered INSIDE the score-strip grey box,
   styled to match `.score-strip-sugg` (same tiny uppercase title rule).
   Implemented as <details> so it collapses by default — the table is bulky. */
.score-strip-lat { border-top: 1px solid #edf0f4; padding-top: 10px; }
.score-strip-lat > summary { list-style: none; cursor: pointer;
                              user-select: none; outline: none; }
.score-strip-lat > summary::-webkit-details-marker { display: none; }
.score-strip-lat-title { font-size: 9px; color: #537090; font-weight: 600;
                         text-transform: uppercase; letter-spacing: 0.12em;
                         margin-bottom: 6px; display: flex; align-items: baseline;
                         gap: 8px; }
.score-strip-lat-title::before { content: "▸"; display: inline-block;
                                  width: 10px; font-size: 9px; color: #5c7999;
                                  transition: transform 0.15s ease; }
.score-strip-lat[open] > .score-strip-lat-title::before { transform: rotate(90deg); }
.score-strip-lat[open] > .score-strip-lat-title { margin-bottom: 8px; }
.lat-inline-total { font-size: 11px; color: #0a285c; font-weight: 600;
                     text-transform: none; letter-spacing: 0;
                     font-variant-numeric: tabular-nums; }
.lat-tbl-inline { width: 100%; border-collapse: collapse; font-size: 11px;
                  font-variant-numeric: tabular-nums; }
.lat-tbl-inline td { padding: 2px 0; border: none; vertical-align: middle; }
.lat-tbl-inline tr + tr td { border-top: 1px solid #edf0f4; }
.lat-inline-name { color: #0a285c; width: 50%; }
.lat-inline-ms { color: #0a285c; text-align: right; width: 70px;
                  padding-right: 12px !important; }
.lat-inline-share { text-align: right; white-space: nowrap; }
.lat-inline-pct { color: #5c7999; min-width: 42px; display: inline-block;
                   text-align: right; }
.lat-n { font-size: 9px; color: #a3b5c9; font-variant-numeric: tabular-nums;
          margin-left: 4px; }

/* Throttling chip on the latency-breakdown summary + nested retry table. */
.lat-inline-retry { margin-left: 8px; padding: 1px 7px; border-radius: 8px;
                    font-size: 10.5px; font-weight: 600;
                    background: #fff3da; border: 1px solid #f2a91e;
                    color: #ad5700; font-variant-numeric: tabular-nums; }
.score-strip-lat-retry { margin-top: 8px; padding-top: 8px;
                         border-top: 1px dashed #edf0f4; }
.score-strip-lat-retry > summary { list-style: none; cursor: pointer;
                                   font-size: 9px; color: #ad5700; font-weight: 600;
                                   text-transform: uppercase; letter-spacing: .12em; }
.score-strip-lat-retry > summary::-webkit-details-marker { display: none; }
.score-strip-lat-retry > summary::before { content: "▸"; display: inline-block;
                                            margin-right: 4px;
                                            transition: transform 120ms; }
.score-strip-lat-retry[open] > summary::before { transform: rotate(90deg); }
.score-strip-lat-retry[open] > summary { margin-bottom: 6px; }
.lat-retry-note { color: #a3b5c9; font-weight: 500; margin-left: 6px;
                  text-transform: none; letter-spacing: 0; }
.lat-tbl-retry thead th { font-size: 9px; color: #5c7999; font-weight: 600;
                          text-transform: uppercase; letter-spacing: .08em;
                          text-align: left; padding: 2px 0; border-bottom: 1px solid #edf0f4; }
.lat-tbl-retry thead th:nth-child(2),
.lat-tbl-retry thead th:nth-child(4) { text-align: right; padding-right: 12px; }
.lat-retry-toks { color: #5c7999; font-size: 10.5px; }
.lat-retry-overhead { color: #ad5700; font-weight: 600; }

/* Span-level error surfaces. Trace info.state stays "OK" whenever the
   agent returned an answer, so a child span crashing (ConnectTimeout,
   CancelledError, etc.) is silent unless we surface it here. Chip on
   the case-detail header is always rendered when has_span_error;
   standalone <details> block lives next to the latency breakdown so
   the reader can drill into exception type + trimmed stacktrace. */
.span-err-chip { display: inline-flex; align-items: center; gap: 4px;
                  padding: 2px 8px; border-radius: 10px;
                  background: #cf2a1e; color: #fff;
                  font-size: 10px; font-weight: 700; letter-spacing: 0.02em;
                  white-space: nowrap; cursor: help; }
.span-err-chip .span-err-icon { font-size: 11px; line-height: 1; }
.score-strip-err { border-top: 1px solid #f3d6d2; padding-top: 10px; }
.score-strip-err > summary { list-style: none; cursor: pointer;
                              user-select: none; outline: none; }
.score-strip-err > summary::-webkit-details-marker { display: none; }
.score-strip-err-title { font-size: 9px; color: #8b1a10; font-weight: 700;
                          text-transform: uppercase; letter-spacing: 0.12em;
                          display: inline-flex; align-items: baseline; gap: 8px; }
.score-strip-err-title::before { content: "▸"; display: inline-block;
                                  width: 10px; font-size: 9px; color: #cf2a1e;
                                  transition: transform 0.15s ease; }
.score-strip-err[open] > summary > .score-strip-err-title::before { transform: rotate(90deg); }
.score-strip-err-summary { font-size: 11px; color: #8b1a10; font-weight: 600;
                            text-transform: none; letter-spacing: 0; margin-left: 4px; }
.score-strip-err-list { margin: 8px 0 0; padding: 0; list-style: none;
                         border: 1px solid #f3d6d2; border-radius: 6px;
                         background: #fff8f7; }
.score-strip-err-list li { padding: 8px 10px; border-bottom: 1px solid #f3d6d2;
                            font-size: 11px; color: #0a285c; }
.score-strip-err-list li:last-child { border-bottom: none; }
.span-err-type { display: inline-block; padding: 1px 6px; border-radius: 4px;
                  background: #cf2a1e; color: #fff; font-weight: 700;
                  font-size: 10px; letter-spacing: 0.02em;
                  font-variant-numeric: tabular-nums; }
.span-err-where { color: #5c7999; font-size: 10px; margin-left: 6px; }
.span-err-msg { display: block; margin-top: 4px; color: #8b1a10;
                 font-style: italic; }
.span-err-stack { display: block; margin-top: 4px; padding: 6px 8px;
                   background: #fff; border: 1px solid #f3d6d2; border-radius: 4px;
                   font-family: ui-monospace, Menlo, Consolas, monospace;
                   font-size: 10px; color: #5c7999; white-space: pre-wrap;
                   max-height: 160px; overflow: auto; }

/* Mark a latency-breakdown row that we attributed an error to. Name
   goes red and an exception-type chip lands inline so the operator
   sees both *which step crashed* and *what crashed* in one glance. */
.lat-inline-name.lat-row-err { color: #8b1a10; font-weight: 600; }
.lat-row-err-types { margin-left: 6px; display: inline-flex; gap: 4px;
                      vertical-align: middle; }
.lat-row-err-types .span-err-type { font-size: 9px; padding: 0 5px; }
.tbl th { background: #f4f6fa; color: #5c7999; font-weight: 600; text-transform: uppercase;
          font-size: 11px; letter-spacing: .5px; }
.tbl tbody tr:hover { background: #f9fbff; }
.tbl .filter-row th { padding: 4px 6px; background: #f8fafc; vertical-align: top; }
.tbl .col-filter { width: 100%; padding: 3px 6px; font: inherit; font-size: 11px;
                   border: 1px solid #e4eaf0; border-radius: 4px; color: #0a285c;
                   background: #fff; }
.tbl .col-filter:focus { outline: none; border-color: #135ee2; }
.range-filter { display: flex; align-items: center; gap: 4px; }
.range-filter input { width: 46px; padding: 3px 4px; font: inherit; font-size: 10.5px;
                      border: 1px solid #e4eaf0; border-radius: 4px;
                      color: #0a285c; background: #fff; text-align: right;
                      font-variant-numeric: tabular-nums; }
.range-filter input:focus { outline: none; border-color: #135ee2; }
.range-filter .range-sep { color: #a3b5c9; font-size: 11px; }
.multi-filter { position: relative; }
.multi-filter .multi-toggle { width: 100%; padding: 3px 8px; font: inherit;
                               font-size: 11px; border: 1px solid #e4eaf0;
                               border-radius: 4px; color: #0a285c; background: #fff;
                               cursor: pointer; text-align: left; }
.multi-filter .multi-toggle:hover { border-color: #135ee2; }
.multi-filter .multi-toggle.has-selection { color: #135ee2; font-weight: 600; }
.multi-filter .multi-menu { display: none; position: absolute; top: 100%; left: 0;
                             z-index: 30; background: #fff; border: 1px solid #e4eaf0;
                             border-radius: 6px; box-shadow: 0 4px 12px rgba(10,40,92,0.08);
                             padding: 6px; min-width: 180px; max-height: 220px;
                             overflow-y: auto; margin-top: 2px; }
.multi-filter.open .multi-menu { display: block; }
.multi-filter .multi-menu label { display: flex; align-items: center; gap: 6px;
                                   padding: 3px 6px; font-size: 11px; cursor: pointer;
                                   border-radius: 4px; white-space: nowrap; }
.multi-filter .multi-menu label:hover { background: #f4f6fa; }
.multi-filter .multi-menu input[type=checkbox] { margin: 0; }
.tbl td.num-cell { text-align: right; }
.tbl .num-pill { display: inline-block; min-width: 46px; padding: 1px 8px;
                 border-radius: 10px; font-weight: 600; font-size: 11px;
                 font-variant-numeric: tabular-nums; text-align: center; }
.tbl .num-bad  .num-pill { background: #fde5e3; color: #cf2a1e; }
.tbl .num-mid  .num-pill { background: #fef4e2; color: #ad5700; }
.tbl .num-good .num-pill { background: #dff5ea; color: #057f19; }
.tbl .num-na   .num-pill { background: #edf0f4; color: #a3b5c9; }
.dim-count { text-align: center; font-variant-numeric: tabular-nums; font-weight: 600; }
.dim-count.empty { color: #cbd6e3; }
.dim-count.num-bad  { background: rgba(207,42,30,0.06); }
.dim-count.num-mid  { background: rgba(242,169,30,0.08); }
.dim-count.num-good { background: rgba(5,127,25,0.06); }
.dim-link { text-decoration: none; color: inherit; display: inline-block;
            min-width: 28px; padding: 2px 8px; border-radius: 10px;
            transition: background 0.12s ease; }
.dim-link:hover { background: #135ee2; color: #fff; }
.active-filter-banner { display: none; align-items: center; gap: 8px;
                        padding: 6px 10px; margin-bottom: 8px;
                        background: #eef4fd; border: 1px solid #dbe5f8;
                        border-radius: 6px; font-size: 11px; color: #0a285c;
                        flex-shrink: 0; }
.active-filter-banner.visible { display: flex; }
.active-filter-banner button { margin-left: auto; background: transparent;
                                border: none; color: #cf2a1e; font: inherit;
                                font-size: 11px; font-weight: 600; cursor: pointer; }
a.case-link { color: #1d69ec; text-decoration: none; font-weight: 600; }
a.case-link:hover { text-decoration: underline; }
.cases-layout { display: grid; grid-template-columns: 360px 1fr; gap: 16px;
                 height: calc(100vh - 160px); min-height: 520px;
                 position: relative; }
@media (max-width: 900px) { .cases-layout { grid-template-columns: 1fr; height: auto; } }
.cases-layout.sidebar-hidden { grid-template-columns: 1fr; gap: 0; }
.cases-layout.sidebar-hidden .cases-sidebar { display: none; }
/* Hover-peek: a 14-px hover strip on the left edge re-shows the sidebar
   as an overlay (position: absolute) without resizing the detail panel. */
.peek-zone { display: none; }
.cases-layout.sidebar-hidden .peek-zone {
  display: block; position: absolute; left: 0; top: 0; bottom: 0;
  width: 14px; z-index: 4; }
.cases-layout.sidebar-hidden.peek-on .cases-sidebar {
  display: flex;
  position: absolute; top: 0; left: 0; bottom: 0;
  width: 360px; z-index: 25;
  box-shadow: 4px 0 18px rgba(10,40,92,0.12); }
.sidebar-toggle {
  position: absolute; top: 8px; left: 8px; z-index: 30;
  width: 26px; height: 22px; padding: 0;
  background: #fff; border: 1px solid #e4eaf0; border-radius: 6px;
  color: #5c7999; font-size: 14px; line-height: 1; cursor: pointer;
  display: flex; align-items: center; justify-content: center; }
.sidebar-toggle:hover { background: #f4f8fe; color: #135ee2; border-color: #dbe5f8; }
.cases-sidebar { background: #fff; border: 1px solid #e4eaf0; border-radius: 10px;
                 padding: 10px; display: flex; flex-direction: column; min-height: 0; }
.case-search { width: 100%; padding: 8px 10px 8px 38px; border: 1px solid #e4eaf0;
               border-radius: 6px; margin-bottom: 8px; font-size: 13px; flex-shrink: 0; }
/* Filter section sizes itself to its content — no internal scrollbar.
   The case-list below uses flex: 1 1 auto, so it gets whatever vertical
   room the filters don't take. Expanding/collapsing the Judge scorers
   matrix grows/shrinks this section dynamically. */
.filters-wrap { flex-shrink: 0; border-bottom: 1px solid #edf0f4;
                padding-bottom: 8px; margin-bottom: 8px; }
.filter-group { display: flex; align-items: flex-start; gap: 4px; margin-bottom: 4px; flex-wrap: wrap; }
/* Foldable filter group (used by Failure mode, Flags, etc. in the
   sidebar). Same chevron pattern as the per-scorer matrix below. */
.foldable-filter { width: 100%; flex-direction: column; }
.foldable-filter-body { display: flex; flex-wrap: wrap; gap: 4px;
                          margin-top: 6px; padding-left: 4px; }
/* Per-scorer 0/1/2 matrix — foldable section in the Test Cases sidebar.
   <details>/<summary> handle the open/close toggle natively. */
.dim-matrix-group { width: 100%; flex-direction: column; }
.dim-matrix-details { width: 100%; }
.dim-matrix-summary { list-style: none; cursor: pointer; user-select: none;
                       display: flex; align-items: center; gap: 6px;
                       padding: 2px 0; }
.dim-matrix-summary::-webkit-details-marker { display: none; }
.dim-matrix-summary::before { content: "▶"; font-size: 9px; color: #5c7999;
                                display: inline-block;
                                transition: transform 0.15s ease; }
.dim-matrix-details[open] > .dim-matrix-summary::before { transform: rotate(90deg); }
.dim-matrix-active-count { font-size: 10px; color: #135ee2; font-weight: 600;
                             letter-spacing: .04em; }
.dim-matrix { display: flex; flex-direction: column; gap: 3px;
              margin-top: 6px; padding-left: 4px; }
.dim-matrix-row { display: grid;
                   grid-template-columns: 1fr auto;
                   align-items: center; gap: 6px; }
.dim-matrix-name { font-size: 10.5px; color: #5c7999;
                    overflow: hidden; text-overflow: ellipsis;
                    white-space: nowrap; }
.dim-matrix-chips { display: inline-flex; gap: 3px; }
.dim-chip { min-width: 22px; padding: 1px 6px; font-size: 10px;
             font-variant-numeric: tabular-nums; }
.filter-label { font-size: 9px; color: #a3b5c9; text-transform: uppercase;
                letter-spacing: .5px; min-width: 60px; flex-shrink: 0;
                padding-top: 4px; font-weight: 600; white-space: nowrap; }
.chip { background: #edf0f4; color: #5c7999; border: 1px solid transparent;
        border-radius: 10px; font-size: 10px; padding: 2px 8px; cursor: pointer;
        font-family: inherit; white-space: nowrap; }
.chip:hover { background: #e7effd; }
.chip.active { background: #1d69ec; color: #fff; border-color: #1d69ec; }
.chip-clear { background: transparent; border: 1px solid #e4eaf0; color: #cf2a1e;
              font-weight: 600; margin-left: auto; }
.chip-clear:hover { background: #fde5e3; }
/* Flag chip dedicated to span-level errors. Visible-red even inactive so
   the reader knows it filters to crashes specifically. Active state is
   solid red (vs the blue used by every other chip) to underscore that
   the result set is the broken cases. */
.chip-err { background: #fde5e3; color: #8b1a10; border: 1px solid #f3d6d2; }
.chip-err:hover { background: #fbd0cc; }
.chip-err.active { background: #cf2a1e; color: #fff; border-color: #cf2a1e; }
.list-count { font-size: 10px; color: #a3b5c9; padding: 0 4px 6px;
              text-transform: uppercase; letter-spacing: .5px; flex-shrink: 0; }
.cases-list-wrap { flex: 1 1 auto; overflow-y: auto; min-height: 0; }
.cases-list { list-style: none; }
.cases-list li { padding: 8px 10px; border-radius: 6px; cursor: pointer; font-size: 12px;
                 display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.cases-list li:hover { background: #f4f6fa; }
.cases-list li.active { background: #e7effd; }
.case-id { font-weight: 600; color: #0a285c; }
.case-q { color: #5c7999; font-size: 11px; width: 100%; }
.case-detail { background: #fff; border: 1px solid #e4eaf0; border-radius: 10px;
               padding: 18px 20px; overflow-y: auto; min-height: 0; }
.corr-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 900px) { .corr-grid { grid-template-columns: 1fr; } }
.corr-stat { font-size: 12px; color: #5c7999; margin-top: 6px; }
.detail-section { margin-bottom: 16px; }
.detail-row { display: grid; grid-template-columns: 1fr 1fr; gap: 14px;
              margin-bottom: 16px; }
.detail-row .detail-section { margin-bottom: 0; min-width: 0; }
@media (max-width: 900px) { .detail-row { grid-template-columns: 1fr; } }
.bodywrap { max-height: 280px; overflow-y: auto; position: relative;
            border-radius: 6px; }
.bodywrap.expanded { max-height: none; overflow: visible; }
.body-toggle { background: transparent; border: 1px solid #dbe5f8; color: #135ee2;
               font: inherit; font-size: 10px; font-weight: 600;
               text-transform: uppercase; letter-spacing: .04em;
               padding: 1px 8px; border-radius: 10px; cursor: pointer;
               margin-left: 8px; vertical-align: middle; }
.body-toggle:hover { background: #eef4fd; }
.detail-section h3 { font-size: 13px; color: #5c7999; text-transform: uppercase;
                     letter-spacing: .5px; margin-bottom: 6px; }
.detail-section .body { background: #f4f6fa; border-radius: 6px; padding: 10px 12px;
                        white-space: pre-wrap; font-size: 13px; }
.detail-section .body.mono { font-family: monospace; font-size: 12px; }
.enum-chip { display: inline-block; padding: 1px 7px; border-radius: 10px;
             font-size: 11px; font-weight: 600; margin: 1px 2px;
             font-family: monospace; }
.enum-chip.match   { background: #dff5ea; color: #0a285c; font-weight: 700; }
.enum-chip.miss    { background: #fde5e3; color: #0a285c; font-weight: 700; }
.enum-chip.neutral { background: #edf0f4; color: #5c7999; font-weight: 500; }
.enum-chip.expected-row { background: #e0eafd; color: #0a285c;
                          font-weight: 800; }
.enum-chip.expected-missed { background: #edf0f4; color: #5c7999;
                             font-weight: 600; }
.enum-rows { display: flex; flex-direction: column; gap: 4px;
             background: #f4f6fa; border-radius: 6px;
             padding: 10px 12px; font-size: 12px; font-family: monospace;
             color: #0a285c; }
.enum-row { display: grid; grid-template-columns: 180px 1fr; gap: 8px;
            align-items: baseline; }
.enum-row .enum-label { color: #5c7999; font-weight: 600;
                          display: inline-flex; align-items: center; gap: 4px; }
.enum-row .info-icon { cursor: help; color: #a3b5c9; font-style: normal;
                          font-family: -apple-system, sans-serif;
                          font-size: 11px; line-height: 1; user-select: none; }
.enum-row .info-icon:hover { color: #135ee2; }
.tip-bubble {
  position: fixed; left: 0; top: -9999px;
  background: #0a285c; color: #fff;
  padding: 8px 10px; border-radius: 6px;
  font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif;
  font-size: 11px; font-weight: 500; line-height: 1.45;
  max-width: 520px; width: max-content;
  box-shadow: 0 4px 12px rgba(10,40,92,0.25);
  pointer-events: none; z-index: 10000;
  opacity: 0; transition: opacity 0.12s ease;
  /* Preserve the "\n" newlines we put in the verbatim-YAML tooltips for
     each judge scorer (name / weight / description / scale rows). */
  white-space: pre-wrap;
}
.tip-bubble.visible { opacity: 1; }
.judge-eval-sub { font-size: 11px; color: #5c7999; margin-bottom: 6px;
                  text-transform: uppercase; letter-spacing: .04em;
                  font-weight: 600; }
/* Foldable card — title + .card-desc remain visible when collapsed. */
.card.foldable .card-title { cursor: pointer; user-select: none;
                              display: flex; align-items: center; gap: 8px; }
.card.foldable .card-title::before { content: "▼"; font-size: 9px;
                                      color: #5c7999;
                                      transition: transform 0.15s ease;
                                      display: inline-block; }
.card.foldable.collapsed .card-title::before { transform: rotate(-90deg); }
.card.foldable.collapsed .card-body { display: none; }
.card.foldable .card-desc { font-size: 12px; color: #537090;
                             margin-bottom: 10px; }
.card.foldable.collapsed .card-desc { margin-bottom: 0; }
.kb-toolbar { display: flex; gap: 12px; align-items: center; margin-bottom: 12px;
              flex-wrap: wrap; }
.kb-toolbar .kb-search { flex: 1 1 280px; min-width: 200px; }
.kb-count { font-size: 11px; color: #a3b5c9; text-transform: uppercase;
            letter-spacing: .04em; font-weight: 600; }
.kb-list { display: flex; flex-direction: column; gap: 10px;
           max-height: calc(100vh - 280px); min-height: 320px; overflow-y: auto;
           padding-right: 4px; }
.kb-row { background: #f8fafc; border: 1px solid #edf0f4; border-radius: 8px;
          padding: 12px 14px; }
.kb-row-head { display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
               cursor: pointer; user-select: none; }
.kb-row:not(.collapsed) .kb-row-head { margin-bottom: 8px; }
.kb-row .kb-chev { display: inline-block; width: 12px; color: #5c7999;
                   font-size: 9px; transition: transform 0.15s ease; }
.kb-row:not(.collapsed) .kb-chev { transform: rotate(90deg); }
.kb-row.collapsed .kb-desc { display: none; }
.kb-id { font-family: monospace; font-size: 12px; font-weight: 700;
         color: #135ee2; background: #e7effd;
         padding: 3px 8px; border-radius: 6px; }
.kb-cases-count { font-size: 11px; color: #5c7999; margin-left: auto; }
.kb-cases-count a { font-weight: 600; }
.kb-desc { font-size: 12px; color: #0a285c; line-height: 1.55;
           white-space: pre-wrap; word-wrap: break-word; }
.kb-reset { background: #fff; border: 1px solid #e4eaf0; border-radius: 6px;
            padding: 4px 10px; font: inherit; font-size: 11px; font-weight: 600;
            color: #cf2a1e; cursor: pointer; display: none; }
.kb-reset.visible { display: inline-block; }
.kb-reset:hover { background: #fde5e3; border-color: #f4b2ab; }
.kb-case-banner { display: none; padding: 6px 10px; margin-bottom: 10px;
                  background: #eef4fd; border: 1px solid #dbe5f8;
                  border-radius: 6px; font-size: 11px; color: #0a285c; }
.kb-case-banner.visible { display: block; }
.kb-case-banner strong { font-weight: 700; }
/* Per-case status colouring on the KB tab. Same palette as the per-case
   ENUMs panel on the Test Cases tab so the two views read consistently:
     correct    = green  (in expected AND reranked — selected & wanted)
     missed     = blue   (in expected only — should have been selected)
     distractor = red    (in reranked only — selected but not expected)
   Tints applied only when the KB tab is filtered to a single case; the
   JS in bindKbSearch sets the class + badge text. */
.kb-row.kb-status-correct     { background: #f1faf3; border-color: #c8e0c9; }
.kb-row.kb-status-correct    .kb-id { background: #dff5ea; color: #057f19; }
.kb-row.kb-status-missed      { background: #f4f8fd; border-color: #c6d4ee; }
.kb-row.kb-status-missed     .kb-id { background: #e0eafd; color: #135ee2; }
.kb-row.kb-status-distractor  { background: #fdf3f2; border-color: #efc8c4; }
.kb-row.kb-status-distractor .kb-id { background: #fde5e3; color: #cf2a1e; }
.kb-status-badge { display: inline-block; font-size: 10px; font-weight: 700;
                    text-transform: uppercase; letter-spacing: .04em;
                    padding: 1px 7px; border-radius: 10px; }
.kb-row:not(.kb-status-correct):not(.kb-status-missed):not(.kb-status-distractor) .kb-status-badge {
                    display: none; }
.kb-row.kb-status-correct    .kb-status-badge { background: #dff5ea; color: #057f19; }
.kb-row.kb-status-missed     .kb-status-badge { background: #e0eafd; color: #135ee2; }
.kb-row.kb-status-distractor .kb-status-badge { background: #fde5e3; color: #cf2a1e; }
/* Inline tallies in the kb-case-banner. Plain text + a glyph + colour
   accent — small enough to coexist with the surrounding banner copy. */
.kb-banner-tally     { font-weight: 700; font-variant-numeric: tabular-nums; }
.kb-banner-correct   { color: #057f19; }
.kb-banner-missed    { color: #135ee2; }
.kb-banner-distractor{ color: #cf2a1e; }
a.judge-eval-link { color: #1d69ec; text-decoration: none; font-weight: 600;
                    cursor: pointer; padding: 0 4px; border-radius: 4px; }
a.judge-eval-link:hover { background: #135ee2; color: #fff; }
.triage-card-title { display: flex; justify-content: space-between;
                     align-items: center; gap: 12px; flex-wrap: wrap; }
.triage-card-title a.triage-show { font-size: 12px; font-weight: 600;
                                    background: #eef4fd; color: #1d69ec;
                                    padding: 4px 10px; border-radius: 6px;
                                    border: 1px solid #dbe5f8; flex-shrink: 0; }
.triage-card-title a.triage-show:hover { background: #135ee2;
                                          color: #fff; border-color: #135ee2; }
hr.enum-divider { border: none; border-top: 1px solid #c8d3e1;
                  margin: 2px 0; }
.dim-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
            gap: 10px; }
.dim-card { border: 1px solid #e4eaf0; border-radius: 8px; padding: 10px 12px;
            min-width: 0; overflow: hidden; }
.dim-card .dim-head { display: flex; justify-content: space-between; align-items: center;
                      margin-bottom: 4px; gap: 8px; }
.dim-card .dim-name { font-weight: 600; font-size: 12px;
                      overflow-wrap: anywhere; word-break: break-word; min-width: 0; }
.dim-card .dim-reason { font-size: 12px; color: #5c7999;
                        overflow-wrap: anywhere; word-break: break-word; }
.placeholder { color: #a3b5c9; font-style: italic; padding: 20px 0; }
.sugg { background: #fff8e7; border-left: 3px solid #f2a91e; padding: 8px 12px;
        border-radius: 4px; font-size: 12px; margin-bottom: 6px; }
.rel2-range { display: flex; align-items: center; gap: 6px; font-size: 11px;
              color: #537090; flex-wrap: wrap; }
.rel2-range input { width: 58px; padding: 3px 6px; font: inherit; font-size: 11px;
                    border: 1px solid #e4eaf0; border-radius: 6px; color: #0a285c; }
.rel2-range input:focus { outline: none; border-color: #135ee2; }
.rel2-range .range-reset { background: transparent; border: none; color: #cf2a1e;
                           cursor: pointer; font-size: 10px; font-weight: 600;
                           padding: 0 4px; }
.trace-id { font-size: 11px; font-weight: 500; color: #5c7999; letter-spacing: 0; }
.trace-id code { font-size: 11px; background: #f1f4f8; color: #0a285c;
                 padding: 1px 6px; border-radius: 4px; }
.lang-switch { display: inline-flex; gap: 0; margin-left: 8px;
               border: 1px solid #dbe5f8; border-radius: 6px; overflow: hidden;
               vertical-align: middle; }
.case-detail-title { display: flex; align-items: center; gap: 8px;
                     justify-content: space-between; flex-wrap: wrap; }
.case-detail-title .case-detail-title-main { display: inline-flex;
                     align-items: center; gap: 8px; flex-wrap: wrap; }
.case-detail-title .lang-switch { margin-left: auto; flex-shrink: 0; }
.lang-switch .lang-btn { background: #fff; border: none; padding: 3px 9px;
                         font: inherit; font-size: 10.5px; font-weight: 600;
                         color: #537090; cursor: pointer; letter-spacing: .04em; }
.lang-switch .lang-btn + .lang-btn { border-left: 1px solid #dbe5f8; }
.lang-switch .lang-btn:hover { background: #f4f8fe; }
.lang-switch .lang-btn.active { background: #135ee2; color: #fff; }
#case-detail[data-lang="en"] .lang-cz,
#case-detail[data-lang="cz"] .lang-en,
#tab-kb[data-lang="en"] .lang-cz,
#tab-kb[data-lang="cz"] .lang-en { display: none; }
.lang-fallback { color: #a3b5c9; font-style: italic; margin-left: 4px; font-size: 11px; }
.score-strip { display: flex; flex-direction: column; gap: 10px; margin-bottom: 14px;
               padding: 12px; background: #f8fafc; border: 1px solid #edf0f4;
               border-radius: 10px; }
.score-strip-row { display: flex; gap: 8px; flex-wrap: wrap; }
.score-strip-sugg { border-top: 1px solid #edf0f4; padding-top: 10px; }
.score-strip-sugg-title { font-size: 9px; color: #537090; font-weight: 600;
                          text-transform: uppercase; letter-spacing: .12em;
                          margin-bottom: 6px; }
.score-box { flex: 1 1 90px; min-width: 90px; background: #fff;
             border: 1px solid #edf0f4; border-radius: 8px; padding: 8px 10px;
             text-align: left; position: relative; }
.score-box .sb-label { font-size: 9px; color: #537090; font-weight: 600;
                       text-transform: uppercase; letter-spacing: .12em; }
.score-box .sb-value { font-size: 22px; font-weight: 600; color: #0a285c;
                       margin-top: 4px; font-family: Inter, -apple-system, sans-serif;
                       letter-spacing: -0.02em; }
.score-box .sb-accent { position: absolute; top: 0; left: 0; bottom: 0; width: 3px;
                        border-radius: 8px 0 0 8px; }
.score-box.s-bad  .sb-accent { background: #cf2a1e; }
.score-box.s-bad  .sb-value  { color: #cf2a1e; }
.score-box.s-mid  .sb-accent { background: #f2a91e; }
.score-box.s-mid  .sb-value  { color: #ad5700; }
.score-box.s-good .sb-accent { background: #057f19; }
.score-box.s-good .sb-value  { color: #057f19; }
.score-box.s-na   .sb-accent { background: #e4eaf0; }
.score-box.s-na   .sb-value  { color: #a3b5c9; }
.metric-value { font-family: Inter, -apple-system, sans-serif; letter-spacing: -0.02em; }
.schema-grid { display: flex; flex-direction: column; gap: 4px;
               border: 1px solid #edf0f4; border-radius: 8px; overflow: hidden; }
.schema-row { display: grid; grid-template-columns: 240px 1fr;
              border-bottom: 1px solid #edf0f4; font-size: 12px;
              align-items: stretch; }
.schema-row:last-child { border-bottom: none; }
.schema-key { background: #f8fafc; color: #537090; padding: 6px 10px;
              font-weight: 600; font-size: 10.5px; letter-spacing: .02em;
              font-family: monospace; overflow-wrap: anywhere; word-break: break-word;
              border-right: 1px solid #edf0f4; }
@media (max-width: 720px) {
  .schema-row { grid-template-columns: 1fr; }
  .schema-key { border-right: none; border-bottom: 1px solid #edf0f4; }
}
.schema-val { padding: 6px 12px; color: #0a285c; white-space: pre-wrap;
              word-break: break-word; }
.schema-val.mono { font-family: monospace; font-size: 11px; color: #537090; }
.route-chips { display: inline-flex; flex-wrap: wrap; gap: 6px; width: 100%;
               margin-top: 4px; }
.route-chip { display: inline-flex; align-items: center; gap: 6px;
              padding: 2px 8px; border-radius: 10px; font-size: 10.5px;
              font-family: monospace; line-height: 1.5;
              border: 1px solid transparent; max-width: 100%;
              overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.route-chip .route-label { font-family: Inter, -apple-system, sans-serif;
                           font-size: 9px; font-weight: 600; letter-spacing: .12em;
                           text-transform: uppercase; opacity: .7; }
.route-agent { background: #eef4fd; color: #135ee2; border-color: #dbe5f8; }
.route-tool  { background: #fff3df; color: #ad5700; border-color: #fde0b3; }
.cases-list li .route-chips { margin-top: 4px; }
.cases-list li .route-chip { font-size: 10px; padding: 1px 6px; }

/* ── Redesigned Summary tab ─────────────────────────────────────────── */
.summary-banner { background: #fff; border: 1px solid #e4eaf0; border-radius: 10px;
                   padding: 14px 18px; margin-bottom: 16px;
                   box-shadow: 0 1px 4px rgba(10,40,92,.07); }
.summary-banner .sb-title { font-size: 11px; color: #a3b5c9; text-transform: uppercase;
                              letter-spacing: .5px; font-weight: 600; margin-bottom: 6px; }
.summary-banner p { font-size: 13px; color: #0a285c; line-height: 1.5; }
.summary-banner .sb-warn { color: #b46504; font-weight: 600; }
.headline-row { display: grid; grid-template-columns: 1fr 1fr;
                 gap: 14px; margin-bottom: 16px; }
.headline-row.headline-row-3 { grid-template-columns: 1fr 1fr 1fr; }
.headline-row.headline-row-2 { grid-template-columns: 1fr 1fr; }
/* Row variant for "small left card + wide right card" — used on Summary
   row 2 so the Test-cases card sits at ~1/4 width next to the wide
   Stage funnel. */
.headline-row.headline-row-1-3 { grid-template-columns: 1fr 3fr; }
@media (max-width: 980px) { .headline-row.headline-row-1-3 { grid-template-columns: 1fr; } }
/* Row 2 with three cards: Test cases | Latency headline | Stage funnel. */
.headline-row.headline-row-1-1-3 { grid-template-columns: 1fr 1.4fr 3fr; }
@media (max-width: 980px) { .headline-row.headline-row-1-1-3 { grid-template-columns: 1fr; } }
/* weighted_avg histogram card on the Notes tab — slider + readout below the chart.
   Lives in a summary-grid-2 next to the "Pass rate" card. */
.doc-passrate-row { align-items: stretch; }
.wavg-hist-card { display: flex; flex-direction: column; }
.wavg-hist-card .hc-label { margin-bottom: 6px; }
.wavg-chart-wrap { flex: 1 1 auto; min-height: 0; }
.wavg-slider-wrap { padding: 8px 10px 4px; border-top: 1px solid #e5eaf2;
                     margin-top: 6px; display: flex; flex-direction: column; gap: 6px; }
.wavg-slider-row { display: flex; align-items: center; gap: 10px;
                    font-size: 12px; color: #5c7999; }
.wavg-slider-row input[type="range"] { flex: 1 1 auto; accent-color: #e57b00; }
.wavg-slider-row .wavg-slider-value { font-variant-numeric: tabular-nums;
                                       font-weight: 700; color: #0a285c;
                                       min-width: 38px; text-align: right; }
.wavg-slider-row .wavg-slider-reset { font-size: 11px; cursor: pointer;
                                       background: transparent; border: 1px solid #c4cfdc;
                                       color: #5c7999; padding: 1px 6px; border-radius: 4px; }
.wavg-slider-row .wavg-slider-reset:hover { color: #0a285c; border-color: #5c7999; }
.wavg-slider-readout { font-size: 12px; color: #0a285c;
                        font-variant-numeric: tabular-nums; }
.wavg-slider-readout .badge { font-size: 11px; padding: 1px 6px; }
.wavg-slider-readout .delta { font-size: 11px; color: #5c7999; margin-left: 6px; }
/* KB Recall card needs more horizontal room — it has 4-col funnel + 2 stat
   columns inside. Rel2 stats card is naturally compact (3 small numbers). */
.headline-row.headline-row-recall-rel2 {
  grid-template-columns: minmax(0, 2fr) minmax(0, 1fr); }
@media (max-width: 980px) {
  .headline-row.headline-row-recall-rel2 { grid-template-columns: minmax(0, 1fr); }
}
/* Pass / Pass-clean / Rel2 row: shrink the two single-number cards a bit
   so the multi-stat Rel2 card has more horizontal room (so Mean | Median
   | STDEV stay big and don't crowd each other). */
.headline-row.headline-row-pass-rel2 { grid-template-columns: 0.85fr 0.85fr 1.3fr; }
@media (max-width: 980px) { .headline-row.headline-row-3 { grid-template-columns: 1fr; }
                             .headline-row.headline-row-pass-rel2 { grid-template-columns: 1fr; } }
.top-failure-title { font-size: 12px; font-weight: 600; color: #5c7999;
                      text-transform: uppercase; letter-spacing: .04em;
                      margin: 4px 2px 8px;
                      display: flex; align-items: center; gap: 6px; }
.top-failure-card { border-left: 3px solid #cf2a1e; }
.top-failure-card .hc-value { color: #cf2a1e; }
.top-failure-card .hc-value a { color: #cf2a1e; text-decoration: none; }
.top-failure-card .hc-value a:hover { text-decoration: underline; color: #a8211a; }
.top-failure-owner { color: #a3b5c9; font-weight: 500; font-size: 10px;
                      letter-spacing: .04em; }
@media (max-width: 760px) { .headline-row { grid-template-columns: 1fr; } }
.sample-comp-tbl td:first-child { width: 60%; }
.sample-comp-tbl tr:last-child td { background: #eef4fd; }
.sample-note { font-size: 12px; color: #5c7999; margin: 4px 2px 12px;
                line-height: 1.5; }
.sample-note code { background: #eef2f8; }
/* All headline cards use flex column so the detail line always sits at
   the bottom of the card. Combined with the grid's default
   align-items:stretch, this lines up labels, values, and detail rows
   across cards even when the middle content has different heights. */
.headline-card { background: #fff; border: 1px solid #e4eaf0; border-radius: 10px;
                  padding: 18px 20px; box-shadow: 0 1px 4px rgba(10,40,92,.07);
                  display: flex; flex-direction: column; }
.headline-card .hc-label { font-size: 11px; color: #5c7999; text-transform: uppercase;
                            letter-spacing: .5px; font-weight: 600;
                            display: flex; align-items: center; gap: 6px; }
.headline-card .hc-value { font-size: 38px; font-weight: 700; color: #135ee2;
                            margin-top: 6px; line-height: 1.1;
                            font-family: Inter, -apple-system, sans-serif;
                            letter-spacing: -0.02em; }
.headline-card .hc-detail { font-size: 12px; color: #5c7999; margin-top: auto;
                              padding-top: 8px; }
/* Inline Δ chip rendered under a card's main value (Pass rate, KB
   Recall, Rel2 mean, and each Top-issue card). The arrow encodes the
   numeric direction of change vs the baseline; the colour encodes
   whether that direction is good for the metric. Sized large enough
   to read as a peer trend marker next to the 38px main value, but
   still clearly secondary (~half size). */
.hc-delta-inline { font-size: 18px; font-weight: 700;
                    font-variant-numeric: tabular-nums;
                    line-height: 1.2; margin-top: 6px;
                    letter-spacing: 0.01em; }
.timeline-delta-good { color: #057f19; }
.timeline-delta-bad  { color: #cf2a1e; }
.timeline-delta-flat { color: #8092a8; font-weight: 500; }
/* Stat cells centre their content; centre the Δ chip too so it sits
   neatly under the big number rather than hugging the left edge. */
.headline-card .hc-stat .hc-delta-inline { text-align: center; }
/* Top-issues cards left-align their value, so left-align the Δ chip too. */
.top-failure-card .hc-delta-inline { text-align: left; }

/* Pass-rate card: same two-stat shape as the Rel2 card so the headline
   row reads consistently — Run rate on the left, Target on the right,
   vertical divider between them. Label + value typography is inherited
   from the shared selectors above. */
.pass-rate-card .hc-stats-row { display: grid;
                                  grid-template-columns: 1fr 1fr;
                                  margin-top: 8px; }
.pass-rate-card .hc-stat { padding: 0 10px; text-align: center; min-width: 0; }
.pass-rate-card .hc-stat:first-child { padding-left: 0; }
.pass-rate-card .hc-stat:nth-child(2) { padding-right: 0;
                                          border-left: 1px solid #e4eaf0; }
/* Run rate turns green when it meets or exceeds the target. The Target
   stat itself uses the shared .hc-stat-reference rule above. */
.pass-rate-card .hc-stat-value.pass-rate-good { color: #057f19; }
/* Multi-stat headline. Test-cases card stays single-row (2 stats fit fine).
   Rel2 card uses a 2-column grid so 3 stats wrap onto 2 rows: Mean + Median
   share row 1, STDEV spans the full second row underneath. KB Recall has
   only 2 stats (Recall + Precision) and stacks them vertically because
   its column is narrow. */
.test-cases-stats-card .hc-stats-row { display: flex; align-items: stretch;
                                          margin-top: 8px; }
.rel2-stats-card .hc-stats-row { display: grid;
                                  grid-template-columns: 1fr 1fr;
                                  margin-top: 8px; }
.rel2-stats-card .hc-stat:nth-child(3) { grid-column: 1 / -1;
                                            margin-top: 12px;
                                            padding-top: 12px;
                                            border-top: 1px solid #e4eaf0; }
/* Two-stat layout (Recall | Precision) — same shape as the Rel2 card so
   the row-1 cards read consistently. Grid + a left border on the second
   stat gives the vertical divider between the two numbers. */
.kb-recall-card .hc-stats-row { display: grid;
                                  grid-template-columns: 1fr 1fr;
                                  margin-top: 8px; }
.rel2-stats-card .hc-stat,
.test-cases-stats-card .hc-stat { min-width: 0;
                                    padding: 0 10px;
                                    text-align: center; }
.test-cases-stats-card .hc-stat { flex: 1 1 0; border-left: 1px solid #e4eaf0; }
.rel2-stats-card .hc-stat:nth-child(2) { border-left: 1px solid #e4eaf0; }
.rel2-stats-card .hc-stat:first-child,
.test-cases-stats-card .hc-stat:first-child { padding-left: 0; }
.test-cases-stats-card .hc-stat:first-child { border-left: none; }
.test-cases-stats-card .hc-stat:last-child { padding-right: 0; }
/* Horizontal two-stat variant — centered cells, vertical divider between
   them to match the Rel2 card styling. */
.kb-recall-card .hc-stat { padding: 0 10px; text-align: center; min-width: 0; }
.kb-recall-card .hc-stat:first-child { padding-left: 0; }
.kb-recall-card .hc-stat:nth-child(2) { padding-right: 0;
                                          border-left: 1px solid #e4eaf0; }
.rel2-stats-card .hc-stat-label,
.test-cases-stats-card .hc-stat-label,
.kb-recall-card .hc-stat-label,
.pass-rate-card .hc-stat-label { font-size: 10px; color: #5c7999;
                                   text-transform: uppercase;
                                   letter-spacing: .08em; font-weight: 600;
                                   margin-bottom: 4px; }
/* Rel2 Mean / KB Recall turn green when the run value strictly exceeds
   its CSAS benchmark. Same #057f19 used by the dim-bar / num-good
   styling so it reads consistently as "good" across the report. */
.rel2-stats-card .hc-stat-value.rel2-mean-good,
.kb-recall-card .hc-stat-value.kb-recall-good { color: #057f19; }
/* Generic "reference / benchmark / target" stat value — rendered in
   dark grey so it reads as a fixed comparison anchor rather than a
   measured run value. Used by the Pass-rate Target, the Rel2 CSAS
   benchmark, and the KB Recall CSAS benchmark. */
.headline-card .hc-stat-value.hc-stat-reference { color: #4a5566; }
.rel2-stats-card .hc-stat-value,
.test-cases-stats-card .hc-stat-value,
.kb-recall-card .hc-stat-value,
.pass-rate-card .hc-stat-value { font-size: 38px; font-weight: 700;
                                   color: #135ee2; line-height: 1.1;
                                   font-family: Inter, -apple-system, sans-serif;
                                   letter-spacing: -0.02em;
                                   font-variant-numeric: tabular-nums; }
.kb-recall-card .hc-stat-detail { font-size: 11px; color: #5c7999;
                                    margin-top: 4px; line-height: 1.4; }
.fm-table tr.fm-total-row td { background: #eef4fd; }
/* Test-set-side failures (gold-defect + naming mismatch) get a soft cream
   tint so the same maintainer team can see at a glance which rows belong
   to them. Light enough to coexist with the orange accent strip. */
.fm-table tr.fm-row-defect td { background: #fffaeb; }
.fm-table tr.fm-row-defect td:first-child { box-shadow: inset 3px 0 0 #f2a91e;
                                              font-weight: 600; }
.fm-table tr.fm-row-defect:hover td { background: #fff3d6; }
/* Over-selection band on the ENUMs-per-case distribution table (Retrieval
   Findings): light-red wash signals "reranker picked > 6 ENUMs", which the
   YAML rubric / agent config considers an over-pick. Subtle enough to not
   compete with the bar charts beside it, strong enough to spot at a glance. */
.funnel-tbl tr.enum-count-oversel td { background: #fdecea; }
.funnel-tbl tr.enum-count-oversel:hover td { background: #fad8d4; }
.funnel-tbl tr.enum-count-footer td { font-weight: 600; }
/* Composition rows live above the "All analyzed cases" divider — they
   describe how the denominator was built, so we style them lighter than
   the failure-mode rows that follow. */
.fm-table tr.fm-composition-total td { background: #f4f6fa; }
.fm-table tr.fm-composition-excluded td { background: #fafbfd; color: #5c7999; }
.fm-table tr.fm-composition-excluded td:first-child { padding-left: 18px; }
.fm-table .fm-indent { color: #a3b5c9; font-family: monospace;
                        margin-right: 4px; }
.fm-table th, .fm-table td { vertical-align: top; }
a.fm-clear-link { color: #1d69ec; text-decoration: none; font-weight: 700; }
a.fm-clear-link:hover { text-decoration: underline; }
.dim-summary-grid { display: grid;
                     grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
                     gap: 12px; }
.dim-summary-card { background: #fff; border: 1px solid #e4eaf0; border-radius: 8px;
                     padding: 12px 14px; }
.dim-summary-head { display: flex; justify-content: space-between; gap: 8px;
                     align-items: flex-start; margin-bottom: 4px; }
.dim-summary-q { font-size: 13px; font-weight: 600; color: #0a285c; }
.dim-summary-tag { font-size: 10px; font-weight: 600; color: #5c7999;
                    background: #f4f6fa; padding: 2px 8px; border-radius: 10px;
                    white-space: nowrap; flex-shrink: 0; }
.dim-summary-meta { font-size: 11px; color: #5c7999; margin-bottom: 8px; }
.dim-summary-meta code { background: #eef2f8; font-size: 10.5px; }
.dim-summary-bar { display: flex; height: 10px; border-radius: 5px;
                    overflow: hidden; background: #edf0f4; margin-bottom: 8px; }
.dim-bar-seg { height: 100%; }
.dim-bar-bad  { background: #cf2a1e; }
.dim-bar-mid  { background: #f2a91e; }
.dim-bar-good { background: #057f19; }
.dim-summary-links { display: flex; justify-content: space-between; gap: 8px;
                      font-size: 11px; color: #5c7999; }
.dim-summary-links a { color: inherit; text-decoration: none;
                        padding: 1px 6px; border-radius: 4px; }
.dim-summary-links a:hover { background: #135ee2; color: #fff; }
.dim-summary-links .dim-link-bad  a { color: #cf2a1e; }
.dim-summary-links .dim-link-mid  a { color: #ad5700; }
.dim-summary-links .dim-link-good a { color: #057f19; }
.funnel-tbl td, .funnel-tbl th { padding: 6px 10px; }
/* KB Recall card: number/detail on the left, embedded Stage-recall funnel
   on the right, separated by a thin vertical divider. Use minmax(0, 1fr)
   so neither side can be pushed past its share by long unbroken content
   (the detail line on the left otherwise eats the funnel column). */
.kb-recall-row { display: grid;
                  grid-template-columns: minmax(0, 0.55fr) minmax(0, 1.45fr);
                  gap: 16px; align-items: start; margin-top: 8px; }
.kb-recall-left, .kb-recall-right { min-width: 0; }
.kb-recall-right { border-left: 1px solid #edf0f4; padding-left: 16px; }
.kb-recall-left .hc-detail { word-break: break-word; overflow-wrap: anywhere; }
.hc-funnel-title { font-size: 11px; color: #5c7999; font-weight: 600;
                    text-transform: uppercase; letter-spacing: .04em;
                    margin-bottom: 6px;
                    display: flex; align-items: center; gap: 6px; }
.kb-recall-right .funnel-tbl td,
.kb-recall-right .funnel-tbl th { padding: 4px 8px; font-size: 11px; }
.kb-recall-right .funnel-bar { min-width: 40px; }
@media (max-width: 760px) {
  .kb-recall-row { grid-template-columns: minmax(0, 1fr); }
  .kb-recall-right { border-left: none; padding-left: 0;
                      border-top: 1px solid #edf0f4; padding-top: 10px; }
}
.funnel-bar { width: 100%; height: 10px; background: #edf0f4; border-radius: 5px;
              overflow: hidden; min-width: 80px; }
.funnel-bar-fill { background: #135ee2; height: 100%; border-radius: 5px;
                    transition: width .2s ease; }
/* Differentiated colors so the recall (down-trend) and precision (up-trend)
   bars read as complementary metrics. */
.funnel-bar-fill-recall    { background: #135ee2; }
.funnel-bar-fill-precision { background: #057f19; }
/* `minmax(0, 1fr)` keeps each column at its share of the row regardless of
   intrinsic content width — without it, a Plotly chart's min-width would
   blow the first column past its allocated 1fr and squash the others. */
.summary-grid-2 { display: grid;
                   grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
                   gap: 16px; margin-bottom: 16px; }
.summary-grid-3 { display: grid;
                   grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) minmax(0, 1fr);
                   gap: 16px; margin-bottom: 16px; }
.summary-grid-2 > .card, .summary-grid-3 > .card { min-width: 0; }
.summary-grid-2 .js-plotly-plot, .summary-grid-3 .js-plotly-plot { width: 100% !important; }
@media (max-width: 900px) {
  .summary-grid-2 { grid-template-columns: minmax(0, 1fr); }
  .summary-grid-3 { grid-template-columns: minmax(0, 1fr); }
}
@media (max-width: 1100px) and (min-width: 901px) {
  .summary-grid-3 { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }
}
.doc-dl dt { font-size: 13px; color: #0a285c; margin-top: 10px; }
.doc-dl dd { font-size: 12px; color: #5c7999; margin: 2px 0 0 16px;
             line-height: 1.5; }
.info-icon-headline { color: #a3b5c9; font-size: 13px; cursor: help; }
.info-icon-headline:hover { color: #135ee2; }

/* Failure-mode badges (sidebar list + case-detail title). Colour groups:
   pass → green; test_set_defect → grey-blue (informational, not an agent
   issue); pipeline failures (scope/retrieval/pruning/reranker/pool) → orange;
   agent failures (context-use/hallucination/language) → red; other → grey. */
.fm-badge { display: inline-block; padding: 1px 8px; border-radius: 10px;
             font-size: 11px; font-weight: 600; white-space: nowrap; }
.fm-pass                { background: #dff5ea; color: #028661; }
.fm-test_set_defect     { background: #e7effd; color: #1d69ec; }
.fm-scope_misroute      { background: #fef4e2; color: #b46504; }
.fm-retrieval_gap       { background: #fef4e2; color: #b46504; }
.fm-pruning_loss        { background: #fef4e2; color: #b46504; }
.fm-reranker_miss       { background: #fef4e2; color: #b46504; }
.fm-pool_content_gap    { background: #fff3df; color: #ad5700; }
.fm-context_use_failure { background: #fde5e3; color: #cf2a1e; }
.fm-hallucination       { background: #fde5e3; color: #cf2a1e; }
.fm-language_drift      { background: #fde5e3; color: #cf2a1e; }
.fm-other_failure       { background: #edf0f4; color: #5c7999; }
.cases-list li .fm-badge { font-size: 10px; padding: 1px 6px; }
/* Excluded-from-analysis badge — informational, not actionable. Greys out
   the sidebar row a touch so it reads as "not part of the metrics" while
   still being inspectable. */
.excl-badge { display: inline-block; padding: 1px 8px; border-radius: 10px;
              font-size: 10px; font-weight: 700; letter-spacing: .04em;
              background: #edf0f4; color: #5c7999;
              border: 1px dashed #cbd6e3; }
.cases-list li.li-excluded { opacity: 0.78; }
.cases-list li.li-excluded .case-q { font-style: italic; color: #a3b5c9; }

/* Per-case baseline-comparison chip rendered next to the PASS/FAIL badge
   when --baseline was passed to the report. Colour follows the same
   semantic as the API report: red for regressions, dark red for ongoing
   failures, green for fixes, gray for new cases. */
.baseline-chip { display: inline-block; padding: 1px 8px; border-radius: 10px;
                font-size: 10px; font-weight: 700; letter-spacing: .04em; }
.baseline-regression { background: #fde5e3; color: #8a1a12; }
.baseline-persistent { background: #f3d9d6; color: #6b1009; }
.baseline-fixed      { background: #dff5ea; color: #036c4d; }
.baseline-new        { background: #eef3f8; color: #5c7999; }

/* Deterministic naming-mismatch tables (Summary tab + per-case detail). */
.naming-table td { vertical-align: top; }
.naming-table .naming-expected { background: #fde5e3; color: #0a285c;
                                  font-weight: 700; padding: 2px 7px;
                                  border-radius: 10px; font-size: 11px; }
.naming-table .naming-kb-form  { background: #dff5ea; color: #0a285c;
                                  font-weight: 700; padding: 2px 7px;
                                  border-radius: 10px; font-size: 11px; }
.naming-source { font-size: 10px; color: #a3b5c9; margin-top: 2px; }

/* Card title with a right-aligned ownership tag (e.g. DATASET). */
.card-title.card-title-with-tag { display: flex; align-items: center;
                                    justify-content: space-between;
                                    gap: 12px; flex-wrap: wrap; }
.card-title-with-tag .card-title-text { display: inline-flex;
                                          align-items: center; gap: 6px; }
.owner-tag { display: inline-block; padding: 2px 10px; border-radius: 12px;
              font-size: 10px; font-weight: 700; letter-spacing: .08em;
              white-space: nowrap; }
.owner-tag-dataset { background: #fef4e2; color: #b46504;
                      border: 1px solid #f4d59b; }
.owner-tag-kbsearch { background: #ede5fb; color: #5b3da3;
                       border: 1px solid #cdb4eb; }

/* KB Findings row 1 — small Empty-queries card (1/4) next to the wider
   naming-mismatches card (3/4). Drops to single column on narrow screens. */
.kb-findings-row1 { display: grid;
                     grid-template-columns: minmax(0, 1fr) minmax(0, 3fr);
                     gap: 16px; margin-bottom: 16px; }
.kb-findings-row1 > .card { min-width: 0; }
@media (max-width: 900px) {
  .kb-findings-row1 { grid-template-columns: minmax(0, 1fr); }
}

/* KB findings review-queue table — wider issue column with proper wrapping. */
.kb-findings-table td.kb-findings-query { max-width: 280px; white-space: normal;
                                            overflow-wrap: anywhere; }
.kb-findings-table td.kb-findings-issue { max-width: 520px; white-space: normal;
                                            overflow-wrap: anywhere;
                                            font-size: 11.5px; line-height: 1.45; }
"""


JS = r"""
const CASES = __CASES__;
const DIM_NAMES = __DIM_NAMES__;
const PASS_THRESHOLD = __PASS_THRESHOLD__;
// Native-language label rendered on the language-toggle button next to
// "EN". "CZ" for CZKB runs, "SK" for SKKB runs. The data-lang attribute
// stays "cz" for both — it identifies the slot, not the language.
const LANG_LABEL_UPPER = __LANG_LABEL_UPPER__;
// Run mean latency in ms (null when no latency columns OR latency surface
// is suppressed). Drives the per-case latency-chip colour.
const MEAN_LAT_MS = __MEAN_LAT_MS__;
const LAT_GREEN_CEILING_MS = 10000;
// Master switch — when false, the case-detail score strip skips the
// Latency score-box and the in-strip Latency breakdown sub-block.
// Driven by --include-latency on the report CLI.
const INCLUDE_LATENCY = __INCLUDE_LATENCY__;
// Throughput threshold used by latency.py's retry heuristic — kept in sync
// with hg_ds_evals.preprocessing.latency.RETRY_MAX_TOK_PER_S so the report's
// "tok/s < X" caption can't drift from the actual rule.
const RETRY_MAX_TOK_PER_S = __RETRY_MAX_TOK_PER_S__;

// Display labels for the per-step latency breakdown. Mirrors LAT_STEP_DISPLAY
// in the Python side so the case-detail table reads identically to the
// Summary-tab card.
const LAT_STEP_DISPLAY = {
  routing:        "Routing (main_agent)",
  planning_llm:   "Planning LLM",
  kb_retrieve:    "KB retrieve",
  kb_prune:       "KB prune",
  kb_rerank:      "KB rerank",
  tools:          "Tools (non-KB)",
  generation_llm: "Generation LLM",
  overhead:       "Overhead",
};

function formatMs(ms) {
  if (ms == null || !isFinite(ms)) return "—";
  if (ms >= 60000) return (ms / 60000).toFixed(1) + "m";
  if (ms >= 1000) return (ms / 1000).toFixed(2) + "s";
  return Math.round(ms) + "ms";
}

// Three-band colour for per-case latency, matching the score-box palette so
// the Latency card sits visually alongside Judge w.avg / Recall / Precision
// without introducing a new colour vocabulary.
//   green (s-good): < 10s
//   yellow (s-mid): >= 10s and <= run mean
//   red   (s-bad):  > run mean
// When MEAN_LAT_MS is null (no latency data) the red band falls back to
// "more than 2 × the 10s floor" so the card still reads as a heat map.
function latencyClass(ms) {
  if (ms == null || !isFinite(ms)) return "s-na";
  if (MEAN_LAT_MS != null) {
    if (ms > MEAN_LAT_MS) return "s-bad";
    if (ms > LAT_GREEN_CEILING_MS) return "s-mid";
    return "s-good";
  }
  if (ms > 2 * LAT_GREEN_CEILING_MS) return "s-bad";
  if (ms > LAT_GREEN_CEILING_MS) return "s-mid";
  return "s-good";
}

function latencyBoxTitle(ms) {
  if (ms == null || !isFinite(ms)) return "trace wall-clock latency · not available";
  const meanTxt = (MEAN_LAT_MS != null) ? formatMs(MEAN_LAT_MS) : "—";
  const cls = latencyClass(ms);
  if (cls === "s-bad")  return `trace wall-clock latency · above run mean (${meanTxt})`;
  if (cls === "s-mid")  return `trace wall-clock latency · above 10s, at-or-below run mean (${meanTxt})`;
  if (cls === "s-good") return "trace wall-clock latency · below 10s";
  return "trace wall-clock latency · not available";
}

// Map a span_errors[] entry to a lat_step label so the breakdown row for
// the crashing step can be highlighted. langgraph_node wins when present
// (more specific); falls back to span_name for HTTP / Runnable spans
// that don't carry one. null = unmappable → only surfaced in the
// standalone errors block.
const SPAN_NODE_TO_STEP = {
  retrieve:        "kb_retrieve",
  rerank:          "kb_rerank",
  knowledge_prune: "kb_prune",
  kb_prune:        "kb_prune",
};
const SPAN_NAME_TO_STEP = {
  "retrieve":                              "kb_retrieve",
  "rerank":                                "kb_rerank",
  "RunnableSequence":                      "kb_rerank",
  "HTTP POST /admin/knowledge-base/query": "kb_retrieve",
  "knowledge_search":                      "kb_retrieve",
  "prune":                                 "kb_prune",
  "knowledge_prune":                       "kb_prune",
  "kb_prune":                              "kb_prune",
};
function mapSpanErrorToStep(err) {
  if (!err) return null;
  const node = (err.langgraph_node != null) ? String(err.langgraph_node).trim() : "";
  if (node && SPAN_NODE_TO_STEP[node]) return SPAN_NODE_TO_STEP[node];
  const name = (err.span_name != null) ? String(err.span_name).trim() : "";
  if (name && SPAN_NAME_TO_STEP[name]) return SPAN_NAME_TO_STEP[name];
  return null;
}
function buildStepErrorIndex(spanErrors) {
  const idx = {};
  const errs = Array.isArray(spanErrors) ? spanErrors : [];
  for (const err of errs) {
    const step = mapSpanErrorToStep(err);
    if (!step) continue;
    if (!idx[step]) idx[step] = [];
    const t = String(err.exception_type || err.status_code || "error").trim() || "error";
    if (!idx[step].includes(t)) idx[step].push(t);
  }
  return idx;
}

// Red chip rendered on the case-detail header. Always on when the
// trace has a span-level error — the trace state column says OK so
// without this the reader has no signal.
function spanErrorChipHtml(c) {
  if (!c || !c.has_span_error) return "";
  const errs = Array.isArray(c.span_errors) ? c.span_errors : [];
  const n = errs.length || (c.span_error_count || 0);
  const types = Array.isArray(c.span_error_types) ? c.span_error_types : [];
  const label = types.length ? types.join(", ") : (n + " span error" + (n === 1 ? "" : "s"));
  const where = errs.map(e =>
    `${e.exception_type || e.status_code || "error"} @ ${e.span_name || "?"}`
  ).join("\n");
  const tip = "Span-level errors hidden by trace state=OK:\n" + where;
  return `<span class="span-err-chip" title="${esc(tip)}">` +
         `<span class="span-err-icon">⚠</span>${esc(label)}</span>`;
}

// Standalone errors <details> block inside the score-strip. Rendered
// whenever c.has_span_error so the reader has a clear "what crashed
// where" surface regardless of whether --include-latency is set.
function spanErrorsBlockHtml(c) {
  if (!c || !c.has_span_error) return "";
  const errs = Array.isArray(c.span_errors) ? c.span_errors : [];
  if (!errs.length) return "";
  const items = errs.map(e => {
    const type = String(e.exception_type || e.status_code || "error");
    const whereParts = [e.span_name, e.langgraph_node]
      .map(v => (v == null ? "" : String(v).trim()))
      .filter((v, i, a) => v && a.indexOf(v) === i);
    const where = whereParts.join(" · ") || "(unknown span)";
    const msg = String(e.exception_message || e.status_message || "").trim();
    const stack = String(e.stacktrace_tail || "").trim();
    return `<li>
      <span class="span-err-type">${esc(type)}</span>` +
        `<span class="span-err-where">at ${esc(where)}</span>` +
        (msg ? `<span class="span-err-msg">${esc(msg)}</span>` : "") +
        (stack ? `<pre class="span-err-stack">${esc(stack)}</pre>` : "") +
    `</li>`;
  }).join("");
  const n = errs.length;
  const summary = n === 1
    ? "1 span crashed — trace state was OK so this would otherwise be hidden"
    : `${n} spans crashed — trace state was OK so these would otherwise be hidden`;
  return `<details class="score-strip-err">
    <summary>
      <span class="score-strip-err-title">Span errors</span>` +
      `<span class="score-strip-err-summary">${esc(summary)}</span>
    </summary>
    <ul class="score-strip-err-list">${items}</ul>
  </details>`;
}

function latencyDetailHtml(c) {
  if (!INCLUDE_LATENCY) return "";
  const total = c.lat_total_ms;
  const steps = Array.isArray(c.lat_steps) ? c.lat_steps : [];
  if (total == null || !steps.length) return "";
  // Sort steps by ms descending so the bottleneck for this case lands on top.
  // Drop zero-ms rows — they're noise inside the compact in-box layout.
  const sorted = steps.slice()
    .filter(s => (s.ms != null) && Number(s.ms) > 0)
    .sort((a, b) => (b.ms || 0) - (a.ms || 0));
  const stepErrIdx = buildStepErrorIndex(c.span_errors);
  const rows = sorted.map(s => {
    const ms = Number(s.ms);
    const share = (total > 0) ? (ms / total) : 0;
    const pct = Math.max(0, Math.min(100, share * 100));
    const display = LAT_STEP_DISPLAY[s.label] || s.label;
    // ×N = number of times this step's span fired in the trace (e.g. routing
    // typically ×2: one main_agent span at the start, one at the end; KB
    // retrieve ×2 if knowledge_search ran twice). Shown for every non-zero
    // step so the count is consistent — a "×1" tells the reader the step
    // was actually present, which is meaningful when reading the breakdown.
    const nStr = (s.n == null || s.n <= 0) ? "" : ` <span class="lat-n">×${s.n}</span>`;
    const errTypes = stepErrIdx[s.label] || [];
    const errChips = errTypes.length
      ? `<span class="lat-row-err-types">` +
          errTypes.map(t => `<span class="span-err-type">${esc(t)}</span>`).join("") +
        `</span>`
      : "";
    const nameCls = errTypes.length ? "lat-inline-name lat-row-err" : "lat-inline-name";
    return `<tr>
      <td class="${nameCls}">${esc(display)}${nStr}${errChips}</td>
      <td class="lat-inline-ms">${formatMs(ms)}</td>
      <td class="lat-inline-share"><span class="lat-inline-pct">${pct.toFixed(1)}%</span>` +
        `<div class="lat-bar"><div class="lat-bar-fill" style="width:${pct.toFixed(1)}%"></div></div></td>
    </tr>`;
  }).join("");

  // Throttling chip + per-call detail — emitted by latency.py when the
  // tok/s heuristic flags one or more CHAT_MODEL spans as a hidden SDK
  // retry. The chip is visible on the collapsed summary so the reader
  // sees the cost without expanding; per-call detail is a nested
  // <details> so it doesn't crowd the step table.
  const retryCount = c.lat_retry_call_count || 0;
  const retryMs = c.lat_retry_overhead_ms || 0;
  const retries = Array.isArray(c.lat_retries) ? c.lat_retries : [];
  let retryChip = "";
  let retryDetail = "";
  if (retryCount > 0 && retryMs > 0) {
    const share = (total > 0) ? (100 * retryMs / total) : 0;
    retryChip = ` <span class="lat-inline-retry" title="suspected SDK retry overhead — see breakdown">` +
                `↻ ${retryCount} · ${formatMs(retryMs)} (${share.toFixed(0)}%)</span>`;
    const retryRows = retries.map(r => `
      <tr>
        <td class="lat-inline-name">${esc(r.bucket || "other")}</td>
        <td class="lat-inline-ms">${formatMs(Number(r.dur_ms) || 0)}</td>
        <td class="lat-retry-toks">${r.output_tokens || 0} out / ${(r.tok_per_s || 0).toFixed(1)} tok/s</td>
        <td class="lat-inline-ms lat-retry-overhead">~${formatMs(Number(r.overhead_ms) || 0)}</td>
      </tr>`).join("");
    retryDetail = `<details class="score-strip-lat-retry">
      <summary class="score-strip-lat-retry-title">Throttling detail (heuristic) <span class="lat-retry-note">tok/s &lt; ${RETRY_MAX_TOK_PER_S}</span></summary>
      <table class="lat-tbl-inline lat-tbl-retry">
        <thead><tr><th>bucket</th><th>duration</th><th>output / rate</th><th>overhead</th></tr></thead>
        <tbody>${retryRows}</tbody>
      </table>
    </details>`;
  }

  // Rendered INSIDE the score-strip grey box, styled to match the
  // Improvement-suggestions sub-block (same tiny uppercase title rule).
  // Collapsed by default — the breakdown table is bulky and most readers
  // only want it when something looks off.
  return `<details class="score-strip-lat">
    <summary class="score-strip-lat-title">Latency breakdown <span class="lat-inline-total">${formatMs(total)} total</span>${retryChip}</summary>
    <table class="lat-tbl-inline">
      <tbody>${rows}</tbody>
    </table>
    ${retryDetail}
  </details>`;
}

const SCOPE_CLS = {kb:"scope-kb", mock_tool:"scope-mock_tool",
                    dba_no_tools:"scope-dba_no_tools", main_agent:"scope-main_agent",
                    other_tools:"scope-other",
                    hg_invest_kb:"scope-hg_invest", hg_invest_mock_tool:"scope-hg_invest",
                    hg_invest_no_tools:"scope-hg_invest", hg_invest_other_tools:"scope-hg_invest"};
const AGENT_CLS = {main_agent:"agent-main", daily_banking_agent:"agent-dba",
                    "hg-invest-phase2":"agent-hg_invest"};
function agentBadge(slug) {
  // Return empty string when no last agent was extracted from the trace —
  // a placeholder dash badge is just visual noise.
  if (!slug) return "";
  const cls = AGENT_CLS[slug] || "agent-other";
  return `<span class="agent-badge ${cls}" title="last agent visited">${esc(slug)}</span>`;
}

// Plain-English label + colour class per failure_mode (mirrors the Python
// FAILURE_MODE_LABEL dict). Anything missing falls back to the raw key.
const FAILURE_MODE_LABELS = {
  pass: "Pass",
  test_set_defect: "Test-set issue",
  enum_name_mismatch: "ENUM name mismatch",
  scope_misroute: "Wrong agent routing",
  retrieval_gap: "Retrieval gap",
  pruning_loss: "Pruning loss",
  reranker_miss: "Reranker miss",
  pool_content_gap: "Pool content gap",
  context_use_failure: "Context-use failure",
  hallucination: "Hallucination",
  language_drift: "Language drift",
  critical_score_zero: "Critical scorer at 0",
  other_failure: "Other failure",
};
function fmBadge(key) {
  if (!key) return "";
  const label = FAILURE_MODE_LABELS[key] || key;
  return `<span class="fm-badge fm-${esc(key)}" title="primary issue">${esc(label)}</span>`;
}

const EXCLUSION_LABELS = {
  empty_user_query: "EXCLUDED · empty user_query",
  non_kb_scope:     "EXCLUDED · non-KB case_scope",
};
function exclusionBadge(reason) {
  if (!reason) return "";
  const label = EXCLUSION_LABELS[reason] || ("EXCLUDED · " + reason);
  return `<span class="excl-badge" title="Row was excluded from the analysis denominator. Shown for inspection only.">${esc(label)}</span>`;
}

// Baseline comparison chip on the case-detail header. Empty string when no
// --baseline was loaded; otherwise one of regression / persistent / fixed /
// new chips. Mirrors the API report's per-case "was PASS, now FAIL" surface.
function baselineCompareChip(c) {
  if (!c || !c.baseline_loaded) return "";
  if (c.regression) {
    const title = "Passed in baseline run; fails in this run. Click PASS/FAIL badge for details.";
    return `<span class="baseline-chip baseline-regression" title="${esc(title)}">PASS → FAIL</span>`;
  }
  if (c.persistent_failure) {
    const title = "Failed in baseline AND in this run — unresolved.";
    return `<span class="baseline-chip baseline-persistent" title="${esc(title)}">FAIL · still failing</span>`;
  }
  if (c.fixed) {
    const title = "Failed in baseline; passes in this run.";
    return `<span class="baseline-chip baseline-fixed" title="${esc(title)}">FAIL → PASS</span>`;
  }
  if (c.new_case) {
    const title = "Not present in baseline run (added since).";
    return `<span class="baseline-chip baseline-new" title="${esc(title)}">new</span>`;
  }
  return "";
}

// Tab switching. Plotly charts inside hidden panels can't measure their
// container on first render, so we resize each chart in the panel that
// just became active. Cheap when the chart is already correctly sized.
function resizePanelCharts(panel) {
  if (!panel || typeof Plotly === "undefined") return;
  // requestAnimationFrame ensures the panel has been laid out before measuring.
  requestAnimationFrame(() => {
    panel.querySelectorAll(".js-plotly-plot").forEach(plot => {
      try { Plotly.Plots.resize(plot); } catch (e) {}
    });
  });
}
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    const panel = document.getElementById(btn.dataset.target);
    panel.classList.add("active");
    resizePanelCharts(panel);
  });
});
// Also resize on window resize (covers the case where the user has the
// Dimensions tab open and resizes the browser). Throttled via rAF.
let _resizeRaf = null;
window.addEventListener("resize", () => {
  if (_resizeRaf) return;
  _resizeRaf = requestAnimationFrame(() => {
    _resizeRaf = null;
    document.querySelectorAll(".tab-panel.active").forEach(resizePanelCharts);
  });
});
// Resize once on initial load so the default-active Summary tab's charts
// fill the card width on first paint.
window.addEventListener("load", () => {
  document.querySelectorAll(".tab-panel.active").forEach(resizePanelCharts);
  initWavgSlider();
});

// ── weighted_avg threshold slider on the Notes tab ──────────────────────────
// Lets the user drag the pass cut-off and see live PASS / FAIL counts under
// the same definition the headline uses (wa ≥ threshold AND NOT critical_zero).
// Moves shapes[1] (the dashed orange line) on the histogram via Plotly.relayout.
function initWavgSlider() {
  const slider   = document.getElementById("wavg-slider");
  const valueEl  = document.getElementById("wavg-slider-value");
  const readout  = document.getElementById("wavg-slider-readout");
  const resetBtn = document.getElementById("wavg-slider-reset");
  const chartEl  = document.getElementById("wavg-hist-chart");
  if (!slider || !readout) return;

  const wavgCases = CASES.filter(c => c.weighted_avg != null);
  const total = wavgCases.length;
  // Baseline (threshold = PASS_THRESHOLD) — matches the headline card.
  const baseline = wavgCases.filter(c => c.weighted_avg >= PASS_THRESHOLD && !c.critical_zero).length;

  const fmtPct = n => total ? `${(n / total * 100).toFixed(1)}%` : "–";

  function update() {
    const t = parseFloat(slider.value);
    valueEl.textContent = t.toFixed(2);
    const passes = wavgCases.filter(c => c.weighted_avg >= t && !c.critical_zero).length;
    const fails  = total - passes;
    const delta  = passes - baseline;
    const deltaStr = delta === 0 ? "" :
      `<span class="delta">(${delta > 0 ? "+" : ""}${delta} vs default)</span>`;
    readout.innerHTML =
      `<span class="badge badge-good">PASS ${passes} (${fmtPct(passes)})</span> ` +
      `<span class="badge badge-bad">FAIL ${fails} (${fmtPct(fails)})</span>` +
      deltaStr +
      `<div style="margin-top:4px;color:#5c7999;font-size:11px">` +
      `Pass rule applied: <code>weighted_avg ≥ ${t.toFixed(2)}</code> AND no weight-2 scorer at 0. ` +
      `Total wa-scored cases: ${total}.</div>`;
    if (chartEl && window.Plotly) {
      // shapes[0] is the static red PASS_THRESHOLD line; shapes[1] is the dashed orange slider line.
      Plotly.relayout(chartEl, {"shapes[1].x0": t, "shapes[1].x1": t}).catch(() => {});
    }
  }
  slider.addEventListener("input", update);
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      slider.value = PASS_THRESHOLD;
      update();
    });
  }
  update();
}

const listEl = document.getElementById("cases-list");
const countEl = document.getElementById("list-count");
const searchEl = document.getElementById("case-search");
const detailEl = document.getElementById("case-detail");
// Language toggle: "en" (default) or "cz". Scoped to the case-detail panel
// (Test Cases tab); the toggle button lives in the case-detail header.
let currentLang = "en";
detailEl.dataset.lang = currentLang;
function applyLang(lang) {
  if (lang !== "cz" && lang !== "en") return;
  currentLang = lang;
  detailEl.dataset.lang = currentLang;
  const kbEl = document.getElementById("tab-kb");
  if (kbEl) kbEl.dataset.lang = currentLang;
  // Sync the active state of every lang switch on the page (case-detail + KB).
  document.querySelectorAll(".lang-switch .lang-btn").forEach(b =>
    b.classList.toggle("active", b.dataset.lang === currentLang));
}
document.addEventListener("click", (ev) => {
  const langBtn = ev.target.closest && ev.target.closest(".lang-btn");
  if (langBtn) { applyLang(langBtn.dataset.lang); }
});
detailEl.addEventListener("click", (ev) => {
  const expandBtn = ev.target.closest(".body-toggle");
  if (expandBtn) {
    const wrap = document.getElementById(expandBtn.dataset.target);
    if (!wrap) return;
    const expanded = wrap.classList.toggle("expanded");
    expandBtn.textContent = expanded ? "collapse" : "expand";
  }
});

// Filter state — one active value per group; null means "no filter".
// `failure_modes` is the only multi-select group: a Set of mode keys; the
// case must have at least one of the selected modes in failure_modes_all.
const activeFilters = { rc: null, flag: null, rel2_min: null, rel2_max: null,
                          wavg_min: null, wavg_max: null,
                          lat_min: null, lat_max: null,
                          dim_pairs: [], missed_enum: null,
                          case_id_set: null, case_id_label: null,
                          failure_modes: new Set() };

function esc(s) {
  if (s == null) return "";
  return String(s).replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));
}

function dimBadge(score) {
  if (score == null) return '<span class="badge badge-na">–</span>';
  const cls = score === 0 ? "badge-bad" : score === 1 ? "badge-mid" : "badge-good";
  return `<span class="badge ${cls}">${score}</span>`;
}

function matchesFilters(c) {
  if (activeFilters.rc && c.root_cause !== activeFilters.rc) return false;
  if (activeFilters.flag) {
    const f = activeFilters.flag;
    if (f === "rerank_empty" && !c.rerank_empty) return false;
    if (f === "kb_gap" && !c.retrieved_pool_inadequacy_identified) return false;
    if (f === "dba_no_tools" && c.scope !== "dba_no_tools") return false;
    if (f === "hg_invest_scope" && !(typeof c.scope === "string" && c.scope.indexOf("hg_invest_") === 0)) return false;
    if (f === "hg_invest_no_tools" && c.scope !== "hg_invest_no_tools") return false;
    // Use the canonical judge_pass (which includes the critical-zero veto)
    // so the chip filter matches the PASS/FAIL badge and the Python-side
    // pass-rate card. Fall back to the raw threshold check only when the
    // payload predates this field.
    if (f === "pass") {
      const isPass = (c.judge_pass != null)
        ? c.judge_pass
        : (c.weighted_avg != null && c.weighted_avg >= PASS_THRESHOLD && !c.critical_zero);
      if (!isPass) return false;
    }
    if (f === "fail") {
      if (c.weighted_avg == null) return false;
      const isPass = (c.judge_pass != null)
        ? c.judge_pass
        : (c.weighted_avg >= PASS_THRESHOLD && !c.critical_zero);
      if (isPass) return false;
    }
    if (f === "kb_scope" && c.scope !== "kb") return false;
    // Over-selection chip: reranker picked more than 6 ENUMs. Matches the
    // red rows + "> 6" footer in the Retrieval Findings tab's ENUMs-per-case
    // distribution table. Threshold kept in sync with OVERSEL_K on the Python side.
    if (f === "oversel" && !(c.reranked_enum_count != null && c.reranked_enum_count > 6)) return false;
    // Span errors: surfaces ConnectTimeout / CancelledError / etc. that
    // are otherwise hidden by trace info.state=OK. has_span_error is set
    // by parse_trace_skkb whenever any span carries STATUS_CODE_ERROR or
    // an "exception" event.
    if (f === "span_errors" && !c.has_span_error) return false;
  }
  if (activeFilters.rel2_min != null || activeFilters.rel2_max != null) {
    if (c.rel2_score == null) return false;
    if (activeFilters.rel2_min != null && c.rel2_score < activeFilters.rel2_min) return false;
    if (activeFilters.rel2_max != null && c.rel2_score > activeFilters.rel2_max) return false;
  }
  if (activeFilters.wavg_min != null || activeFilters.wavg_max != null) {
    if (c.weighted_avg == null) return false;
    if (activeFilters.wavg_min != null && c.weighted_avg < activeFilters.wavg_min) return false;
    if (activeFilters.wavg_max != null && c.weighted_avg > activeFilters.wavg_max) return false;
  }
  // Latency range (ms) — sidebar inputs are in seconds; bindLatencyRange
  // converts to ms before writing here so the comparison is direct.
  if (activeFilters.lat_min != null || activeFilters.lat_max != null) {
    if (c.lat_total_ms == null) return false;
    if (activeFilters.lat_min != null && c.lat_total_ms < activeFilters.lat_min) return false;
    if (activeFilters.lat_max != null && c.lat_total_ms > activeFilters.lat_max) return false;
  }
  if (activeFilters.dim_pairs && activeFilters.dim_pairs.length) {
    // Group entries by dim — within a dim the selected scores are OR-ed,
    // across dims they are AND-ed. So {clarity:0} + {clarity:1} matches
    // cases scoring 0 or 1 on clarity; adding {language:2} further
    // restricts to those that ALSO score 2 on language.
    const grouped = {};
    for (const p of activeFilters.dim_pairs) {
      if (!grouped[p.dim]) grouped[p.dim] = new Set();
      grouped[p.dim].add(p.score);
    }
    for (const dim in grouped) {
      const dd = c.dims[dim];
      if (!dd || !grouped[dim].has(dd.score)) return false;
    }
  }
  if (activeFilters.missed_enum) {
    const target = activeFilters.missed_enum;
    const inPool = parseList(c.missing_enums_in_candidate_pool).map(String);
    const notPool = parseList(c.missing_enums_not_in_pool).map(String);
    if (!inPool.includes(target) && !notPool.includes(target)) return false;
  }
  if (activeFilters.case_id_set && !activeFilters.case_id_set.has(String(c.id))) return false;
  if (activeFilters.failure_modes && activeFilters.failure_modes.size) {
    const all = Array.isArray(c.failure_modes_all) ? c.failure_modes_all : [];
    let hit = false;
    for (const m of all) {
      if (activeFilters.failure_modes.has(m)) { hit = true; break; }
    }
    if (!hit) return false;
  }
  return true;
}

function updateActiveFilterBanner() {
  const el = document.getElementById("active-filter");
  const txt = document.getElementById("active-filter-text");
  if (!el || !txt) return;
  const msgs = [];
  if (activeFilters.dim_pairs && activeFilters.dim_pairs.length) {
    const parts = activeFilters.dim_pairs.map(p =>
      `<strong>${esc(p.dim)}</strong>=<strong>${p.score}</strong>`);
    msgs.push(parts.join(" &amp; "));
  }
  if (activeFilters.missed_enum) {
    msgs.push(`missed ENUM = <strong>${esc(activeFilters.missed_enum)}</strong>`);
  }
  if (activeFilters.case_id_set) {
    const lbl = activeFilters.case_id_label || "judge eval";
    msgs.push(`<strong>${esc(lbl)}</strong> (${activeFilters.case_id_set.size} cases)`);
  }
  if (msgs.length) {
    txt.innerHTML = "Filtered by " + msgs.join(" · ");
    el.classList.add("visible");
  } else {
    el.classList.remove("visible");
  }
}

function syncDimMatrixState() {
  const pairs = activeFilters.dim_pairs || [];
  document.querySelectorAll(".chip.dim-chip").forEach(btn => {
    const dim   = btn.dataset.dim;
    const score = parseInt(btn.dataset.score, 10);
    const active = pairs.some(p => p.dim === dim && p.score === score);
    btn.classList.toggle("active", active);
  });
  const countEl = document.querySelector(".dim-matrix-active-count");
  if (countEl) {
    countEl.textContent = pairs.length ? ` (${pairs.length} active)` : "";
  }
}

function bindDimMatrix() {
  // Per-scorer matrix in the sidebar: each chip toggles a {dim, score}
  // entry in activeFilters.dim_pairs. Multi-select per dim — clicking
  // multiple scores for the same scorer adds them all; matchesFilters
  // OR-s scores within a dim and AND-s across dims.
  document.querySelectorAll(".chip.dim-chip").forEach(btn => {
    btn.addEventListener("click", () => {
      const dim   = btn.dataset.dim;
      const score = parseInt(btn.dataset.score, 10);
      const idx = activeFilters.dim_pairs.findIndex(
        p => p.dim === dim && p.score === score);
      if (idx >= 0) {
        activeFilters.dim_pairs.splice(idx, 1);
      } else {
        activeFilters.dim_pairs.push({dim, score});
      }
      syncDimMatrixState();
      renderList();
    });
  });
}

function renderList() {
  const q = (searchEl.value || "").toLowerCase();
  updateActiveFilterBanner();
  syncDimMatrixState();
  listEl.innerHTML = "";
  const filtered = CASES.filter(c => {
    if (!matchesFilters(c)) return false;
    if (!q) return true;
    return (c.id || "").toLowerCase().includes(q)
        || (c.user_query || "").toLowerCase().includes(q)
        || (c.user_query_en || "").toLowerCase().includes(q);
  });
  countEl.textContent = `showing ${filtered.length} of ${CASES.length}`;
  filtered.forEach(c => {
    const li = document.createElement("li");
    li.dataset.id = c.id;
    const scopeCls = SCOPE_CLS[c.scope] || "scope-other";
    const wa = (c.weighted_avg == null) ? "–" : c.weighted_avg.toFixed(2);
    let passTag = "";
    if (c.weighted_avg != null) {
      const isPass = (c.judge_pass != null)
        ? c.judge_pass
        : (c.weighted_avg >= PASS_THRESHOLD && !c.critical_zero);
      const vetoNote = c.critical_zero ? " · critical scorer at 0" : "";
      passTag = `<span class="badge ${isPass ? 'badge-good' : 'badge-bad'}" title="judge weighted_avg ${c.weighted_avg.toFixed(2)} ${c.weighted_avg >= PASS_THRESHOLD ? '≥' : '<'} ${PASS_THRESHOLD}${vetoNote}">${isPass ? 'PASS' : 'FAIL'}</span>`;
    }
    const csVal = (c.case_scope || "").toString();
    const csCls = csVal ? ("cs-" + (["kb","kb_and_api","api","out_of_scope","ambiguous"].includes(csVal) ? csVal : "other")) : "";
    const csTag = csVal ? `<span class="cs-badge ${csCls}" title="case_scope (judge): ${esc(csVal)}">${esc(csVal)}</span>` : "";
    li.innerHTML = `<span class="case-id">${c.id}</span>` +
                   csTag +
                   `<span class="scope-badge ${scopeCls}" title="query_scope (graph): ${esc(c.scope || "")}">${c.scope || ""}</span>` +
                   agentBadge(c.last_agent) +
                   passTag +
                   baselineCompareChip(c) +
                   (c.excluded_reason ? exclusionBadge(c.excluded_reason) : (c.failure_mode && c.failure_mode !== "pass" ? fmBadge(c.failure_mode) : "")) +
                   `<span style="margin-left:auto;font-weight:600;color:#135ee2">${wa}</span>` +
                   `<span class="case-q">${(c.user_query_en || c.user_query || (c.excluded_reason === "empty_user_query" ? "(empty user_query)" : "")).slice(0, 90)}</span>`;
    if (c.excluded_reason) li.classList.add("li-excluded");
    li.addEventListener("click", () => selectCase(c.id));
    listEl.appendChild(li);
  });
}

function syncChipStates() {
  document.querySelectorAll(".chip[data-group]").forEach(btn => {
    const group = btn.dataset.group;
    const value = btn.dataset.value;
    if (group === "failure_modes") {
      btn.classList.toggle("active", activeFilters.failure_modes.has(value));
    } else {
      btn.classList.toggle("active", activeFilters[group] === value);
    }
  });
}

function bindChips() {
  document.querySelectorAll(".chip[data-group]").forEach(btn => {
    btn.addEventListener("click", () => {
      const group = btn.dataset.group;
      const value = btn.dataset.value;
      if (group === "failure_modes") {
        // Multi-select: clicking toggles membership in the Set.
        if (activeFilters.failure_modes.has(value)) {
          activeFilters.failure_modes.delete(value);
        } else {
          activeFilters.failure_modes.add(value);
        }
      } else {
        // Single-select: clicking the active chip deselects it.
        activeFilters[group] = (activeFilters[group] === value) ? null : value;
      }
      syncChipStates();
      renderList();
    });
  });
  const clearBtn = document.getElementById("chip-clear");
  if (clearBtn) clearBtn.addEventListener("click", () => {
    activeFilters.rc = null; activeFilters.flag = null;
    activeFilters.rel2_min = null; activeFilters.rel2_max = null;
    activeFilters.wavg_min = null; activeFilters.wavg_max = null;
    activeFilters.lat_min = null; activeFilters.lat_max = null;
    activeFilters.dim_pairs = [];
    activeFilters.missed_enum = null;
    activeFilters.case_id_set = null;
    activeFilters.case_id_label = null;
    activeFilters.failure_modes.clear();
    ["rel2-min","rel2-max","wavg-min","wavg-max","lat-min","lat-max"].forEach(id => {
      const el = document.getElementById(id); if (el) el.value = "";
    });
    syncChipStates();
    renderList();
  });
}

function bindRange(minId, maxId, resetId, minKey, maxKey) {
  const mn = document.getElementById(minId);
  const mx = document.getElementById(maxId);
  const rs = document.getElementById(resetId);
  function onChange() {
    const a = mn.value === "" ? null : parseFloat(mn.value);
    const b = mx.value === "" ? null : parseFloat(mx.value);
    activeFilters[minKey] = Number.isFinite(a) ? a : null;
    activeFilters[maxKey] = Number.isFinite(b) ? b : null;
    renderList();
  }
  if (mn) mn.addEventListener("input", onChange);
  if (mx) mx.addEventListener("input", onChange);
  if (rs) rs.addEventListener("click", () => {
    if (mn) mn.value = ""; if (mx) mx.value = "";
    activeFilters[minKey] = null; activeFilters[maxKey] = null;
    renderList();
  });
}

function bindRel2Range() {
  bindRange("rel2-min", "rel2-max", "rel2-reset", "rel2_min", "rel2_max");
  bindRange("wavg-min", "wavg-max", "wavg-reset", "wavg_min", "wavg_max");
}

// Latency range — inputs are in SECONDS for readability; convert to ms
// before writing into activeFilters since c.lat_total_ms is millis.
// Only wired up when --include-latency was passed (the inputs aren't
// even rendered otherwise).
function bindLatencyRange() {
  if (!INCLUDE_LATENCY) return;
  const mn = document.getElementById("lat-min");
  const mx = document.getElementById("lat-max");
  const rs = document.getElementById("lat-reset");
  if (!mn || !mx) return;
  function onChange() {
    const a = mn.value === "" ? null : parseFloat(mn.value);
    const b = mx.value === "" ? null : parseFloat(mx.value);
    activeFilters.lat_min = Number.isFinite(a) ? a * 1000 : null;
    activeFilters.lat_max = Number.isFinite(b) ? b * 1000 : null;
    renderList();
  }
  mn.addEventListener("input", onChange);
  mx.addEventListener("input", onChange);
  if (rs) rs.addEventListener("click", () => {
    mn.value = ""; mx.value = "";
    activeFilters.lat_min = null; activeFilters.lat_max = null;
    renderList();
  });
}

function fmt(v) { return v == null ? "–" : v.toFixed(2); }

// Parse a payload value that might be JSON/Python-list/plain string into an array.
function parseList(x) {
  if (Array.isArray(x)) return x;
  if (x == null) return [];
  let s = String(x).trim();
  if (!s) return [];
  try { const p = JSON.parse(s); if (Array.isArray(p)) return p; }
  catch (_) { /* fall through */ }
  try { const p = JSON.parse(s.replace(/'/g, '"')); if (Array.isArray(p)) return p; }
  catch (_) { return []; }
  return [];
}

// Color rule (default, used by most rows):
//   green = id is in BOTH expected and reranked (the right pick)
//   red   = id is in expected XOR reranked (missed or wrong pick)
//   grey  = id is in neither (pool noise)
// Modes:
//   "vs_expected" — green if in expected, red otherwise (used for optimal selection).
//   "expected"    — default rule but bold (used for the expected row itself).
function enumChipClass(id, expectedSet, rerankedSet, mode) {
  const inExp = expectedSet.has(id);
  const inRer = rerankedSet.has(id);
  if (mode === "vs_expected") {
    return inExp ? "enum-chip match" : "enum-chip miss";
  }
  if (mode === "expected") {
    return inRer ? "enum-chip expected-row" : "enum-chip expected-missed";
  }
  if (inExp && inRer) return "enum-chip match";
  if (inExp || inRer) return "enum-chip miss";
  return "enum-chip neutral";
}

function enumChips(rawIds, rawExpected, rawReranked, mode) {
  const ids = parseList(rawIds).map(String);
  if (!ids.length) return '<em class="lang-fallback">[]</em>';
  const expected = new Set(parseList(rawExpected).map(String));
  const reranked = new Set(parseList(rawReranked).map(String));
  return ids.map(id =>
    `<span class="${enumChipClass(id, expected, reranked, mode)}">${esc(id)}</span>`
  ).join(" ");
}

// One row in the ENUMs panel: label + concise tooltip + chip list.
function enumRow(label, info, chipsHtml) {
  return `<div class="enum-row">` +
         `<span class="enum-label">${esc(label)} ` +
         `<span class="info-icon" data-tip="${esc(info)}">&#9432;</span></span>` +
         `<span class="enum-chips">${chipsHtml}</span>` +
         `</div>`;
}

// Tooltip helper. Native `title` is unreliable inside scrollable containers
// (overflow-y:auto can clip CSS pseudo-element tooltips). We render a single
// fixed-position bubble appended to <body> and position it near the hovered
// element on each pointer event, so clipping never applies.
const _tipBubble = document.createElement("div");
_tipBubble.className = "tip-bubble";
document.body.appendChild(_tipBubble);
function _showTip(target) {
  const tip = target.dataset.tip;
  if (!tip) return;
  _tipBubble.textContent = tip;
  _tipBubble.classList.add("visible");
  const rect = target.getBoundingClientRect();
  // Lay out off-screen first to measure, then position.
  _tipBubble.style.left = "0px";
  _tipBubble.style.top = "-9999px";
  const tw = _tipBubble.offsetWidth;
  const th = _tipBubble.offsetHeight;
  let left = rect.left + rect.width / 2 - tw / 2;
  let top  = rect.bottom + 8;
  // Keep bubble inside viewport.
  const margin = 6;
  if (left < margin) left = margin;
  if (left + tw > window.innerWidth - margin) left = window.innerWidth - tw - margin;
  // If there's no room below, flip above the icon.
  if (top + th > window.innerHeight - margin) {
    top = rect.top - th - 8;
  }
  _tipBubble.style.left = left + "px";
  _tipBubble.style.top = top + "px";
}
function _hideTip() {
  _tipBubble.classList.remove("visible");
}
document.addEventListener("mouseover", e => {
  const t = e.target.closest && e.target.closest("[data-tip]");
  if (t) _showTip(t);
});
document.addEventListener("mouseout", e => {
  const t = e.target.closest && e.target.closest("[data-tip]");
  if (t) _hideTip();
});
window.addEventListener("scroll", _hideTip, true);

function toolName(t) {
  if (t == null) return "";
  if (typeof t === "string") return t;
  if (typeof t === "object") return t.name || t.tool || t.tool_name || JSON.stringify(t);
  return String(t);
}

function routeChips(c, opts) {
  opts = opts || {};
  const agents = parseList(c.agents_called);
  const tools  = parseList(c.tools_called).map(toolName).filter(Boolean);
  if (!agents.length && !tools.length) return "";
  const parts = [];
  if (agents.length) {
    parts.push(`<span class="route-chip route-agent" title="agents_called">` +
               `<span class="route-label">graph</span>${esc(agents.join(" → "))}</span>`);
  }
  if (tools.length) {
    const shown = opts.compact ? tools.slice(0, 2) : tools;
    const extra = opts.compact && tools.length > shown.length ? ` +${tools.length - shown.length}` : "";
    parts.push(`<span class="route-chip route-tool" title="tools_called: ${esc(tools.join(', '))}">` +
               `<span class="route-label">tools</span>${esc(shown.join(", "))}${extra}</span>`);
  }
  return `<span class="route-chips">${parts.join("")}</span>`;
}

// Render a CZ/EN pair inside a .body container. The #case-detail[data-lang]
// attribute drives which one is visible via CSS.
function bodyLangPair(cz, en, opts) {
  opts = opts || {};
  const mono = opts.mono ? " mono" : "";
  const czRaw = cz == null ? "" : String(cz).trim();
  const enRaw = en == null ? "" : String(en).trim();
  const czBody = czRaw
    ? esc(czRaw)
    : '<em class="lang-fallback">(no CZ text)</em>';
  const enBody = enRaw
    ? esc(enRaw)
    : (czRaw
        ? `${esc(czRaw)}<span class="lang-fallback">(EN missing — showing CZ)</span>`
        : '<em class="lang-fallback">(empty)</em>');
  return `<div class="body${mono} lang-cz" lang="cz">${czBody}</div>` +
         `<div class="body${mono} lang-en" lang="en">${enBody}</div>`;
}

function scoreClass(score) {
  if (score == null) return "s-na";
  if (score === 0) return "s-bad";
  if (score === 1) return "s-mid";
  return "s-good";
}

function summaryClass(v, thresholds) {
  if (v == null || !Number.isFinite(v)) return "s-na";
  if (v < thresholds[0]) return "s-bad";
  if (v < thresholds[1]) return "s-mid";
  return "s-good";
}

function scoreStripHtml(c, suggHtml, latHtml) {
  const wa = c.weighted_avg;
  const waStr = wa == null ? "–" : wa.toFixed(2);
  const waCls = summaryClass(wa, [0.5, PASS_THRESHOLD]);
  const rc = c.enum_recall;
  const rcStr = rc == null ? "–" : rc.toFixed(2);
  const rcCls = summaryClass(rc, [0.5, 0.8]);
  const pr = c.enum_precision;
  const prStr = pr == null ? "–" : pr.toFixed(2);
  const prCls = summaryClass(pr, [0.5, 0.8]);
  const r2 = c.rel2_score;
  const r2Str = r2 == null ? "–" : r2.toFixed(2);
  const r2Cls = summaryClass(r2, [0.5, 0.8]);
  const latMs = c.lat_total_ms;
  const latStr = (latMs == null || !isFinite(latMs)) ? "–" : formatMs(latMs);
  const latCls = latencyClass(latMs);
  const latTip = latencyBoxTitle(latMs);
  // When the latency surface is disabled by --include-latency=false the
  // box itself is dropped from the score strip so the report doesn't
  // show approximated data to readers who shouldn't see it.
  const latBoxHtml = INCLUDE_LATENCY
    ? `<div class="score-box ${latCls}" title="${latTip}"><span class="sb-accent"></span>
       <div class="sb-label">Latency</div><div class="sb-value sb-value-lat">${latStr}</div></div>`
    : "";
  const suggBlock = suggHtml
    ? `<div class="score-strip-sugg"><div class="score-strip-sugg-title">Improvement suggestions</div>${suggHtml}</div>`
    : "";
  return `<div class="score-strip">` +
    `<div class="score-strip-row">` +
    `<div class="score-box ${waCls}"><span class="sb-accent"></span>
       <div class="sb-label">Judge w.avg (0–1)</div><div class="sb-value">${waStr}</div></div>` +
    `<div class="score-box ${rcCls}" title="Per-case ENUM recall: |expected ∩ reranker-selected| / |expected|"><span class="sb-accent"></span>
       <div class="sb-label">Recall (0–1)</div><div class="sb-value">${rcStr}</div></div>` +
    `<div class="score-box ${prCls}" title="Per-case ENUM precision: |expected ∩ reranker-selected| / |reranker-selected|"><span class="sb-accent"></span>
       <div class="sb-label">Precision (0–1)</div><div class="sb-value">${prStr}</div></div>` +
    `<div class="score-box ${r2Cls}"><span class="sb-accent"></span>
       <div class="sb-label">Rel2 (0–1)</div><div class="sb-value">${r2Str}</div></div>` +
    latBoxHtml +
    `</div>` +
    suggBlock +
    (latHtml || "") +
    spanErrorsBlockHtml(c) +
    `</div>`;
}

function selectCase(id) {
  document.querySelectorAll("#cases-list li").forEach(li => li.classList.toggle("active", li.dataset.id === id));
  const c = CASES.find(x => x.id === id);
  if (!c) return;
  // Push the selection to the KB tab so its list auto-filters to the
  // entries this case actually used. The KB "Show all" button clears it.
  if (window._setKbCase) window._setKbCase(id);
  const dimHtml = DIM_NAMES.map(d => {
    const dd = c.dims[d] || {};
    return `<div class="dim-card">
      <div class="dim-head"><span class="dim-name">${d}</span>${dimBadge(dd.score)}</div>
      <div class="dim-reason">${esc(dd.reasoning)}</div>
    </div>`;
  }).join("");
  const sugg = [
    ["Retrieval", c.retrieval_improvement_suggestion],
    ["Reranker", c.reranker_improvement_suggestion],
    ["Agent", c.agent_improvement_suggestion],
    ["KB", c.kb_improvement_suggestion],
    ["Test case", c.test_case_improvement_suggestion],
  ].filter(x => x[1]).map(x => `<div class="sugg"><strong>${x[0]}:</strong> ${esc(x[1])}</div>`).join("");
    const scopeCls = SCOPE_CLS[c.scope] || "scope-other";
  // Pass / fail tag — fail when judge_pass is False (weighted_avg below
  // PASS_THRESHOLD OR any critical (weight-2) scorer at 0).
  let passTag = "";
  if (c.weighted_avg != null) {
    const isPass = (c.judge_pass != null)
      ? c.judge_pass
      : (c.weighted_avg >= PASS_THRESHOLD && !c.critical_zero);
    const vetoNote = c.critical_zero ? " · critical scorer at 0" : "";
    passTag = `<span class="badge ${isPass ? 'badge-good' : 'badge-bad'}" title="judge weighted_avg ${c.weighted_avg.toFixed(2)} ${c.weighted_avg >= PASS_THRESHOLD ? '≥' : '<'} ${PASS_THRESHOLD}${vetoNote}">${isPass ? 'PASS' : 'FAIL'}</span>`;
  }
  // CZ / EN switch lives on the right side of the case-detail header.
  const czCls = currentLang === "cz" ? " active" : "";
  const enCls = currentLang === "en" ? " active" : "";
  const langSwitch = `<span class="lang-switch case-detail-lang" role="group" aria-label="language">` +
                     `<button type="button" class="lang-btn${czCls}" data-lang="cz">${LANG_LABEL_UPPER}</button>` +
                     `<button type="button" class="lang-btn${enCls}" data-lang="en">EN</button>` +
                     `</span>`;
  detailEl.dataset.lang = currentLang;
  detailEl.innerHTML = `
    <h2 class="case-detail-title" style="font-size:16px;margin-bottom:8px">
      <span class="case-detail-title-main">${esc(c.id)} · <span class="scope-badge ${scopeCls}">${esc(c.scope)}</span>${c.last_agent ? ` · ${agentBadge(c.last_agent)}` : ""}${passTag ? ` · ${passTag}` : ""}${baselineCompareChip(c) ? ` · ${baselineCompareChip(c)}` : ""}${c.has_span_error ? ` · ${spanErrorChipHtml(c)}` : ""}${c.excluded_reason ? ` · ${exclusionBadge(c.excluded_reason)}` : ((c.failure_mode && c.failure_mode !== "pass") ? ` · ${fmBadge(c.failure_mode)}` : "")}${c.trace_id ? ` · <span class="trace-id" title="trace_id">trace: <code>${esc(c.trace_id)}</code></span>` : ""}</span>
      ${langSwitch}
    </h2>
    <div style="margin-bottom:10px">${routeChips(c)}</div>
    ${scoreStripHtml(c, sugg, latencyDetailHtml(c))}
    <div class="detail-section">
      <h3>User query</h3>
      ${bodyLangPair(c.user_query, c.user_query_en)}
    </div>
    <div class="detail-row">
      <div class="detail-section detail-col">
        <h3>Agent response <button type="button" class="body-toggle" data-target="agent-resp-body">expand</button></h3>
        <div class="bodywrap" id="agent-resp-body">${bodyLangPair(c.agent_response, c.agent_response_en)}</div>
      </div>
      <div class="detail-section detail-col">
        <h3>Expected response <button type="button" class="body-toggle" data-target="expected-resp-body">expand</button></h3>
        <div class="bodywrap" id="expected-resp-body">${bodyLangPair(c.expected_response, c.expected_response_en)}</div>
      </div>
    </div>
    <div class="detail-section"><h3>Judge scores</h3><div class="dim-grid">${dimHtml}</div></div>
    <div class="detail-section">
      <h3>ENUMs</h3>
      <div class="enum-rows">
        ${enumRow("expected", "Ground-truth ENUM IDs that should be retrieved/selected for this query (from the test set).", enumChips(c.expected_enums, c.expected_enums, c.reranked_enum_ids, "expected"))}
        <hr class="enum-divider">
        ${enumRow("final selected", "ENUM IDs the reranker actually picked — what the agent used (reranked_enum_ids).", enumChips(c.reranked_enum_ids, c.expected_enums, c.reranked_enum_ids))}
        <hr class="enum-divider">
        ${enumRow("post-prune", "Candidate ENUM pool after dedup/pruning, before reranking (post_prune_enum_ids).", enumChips(c.post_prune_enum_ids, c.expected_enums, c.reranked_enum_ids))}
        <hr class="enum-divider">
        ${enumRow("retriever miss", "Expected ENUMs that never made it into the post-prune pool — the retriever didn't surface them. Computed as expected_enums − post_prune_enum_ids.", enumChips(c.missing_enums_not_in_pool, c.expected_enums, c.reranked_enum_ids))}
        ${enumRow("reranker miss", "Expected ENUMs that WERE in the post-prune pool but the reranker did NOT pick them. Computed as (expected_enums ∩ post_prune_enum_ids) − reranked_enum_ids.", enumChips(c.missing_enums_in_candidate_pool, c.expected_enums, c.reranked_enum_ids))}
        ${enumRow("extra or distracting", "Selected ENUMs the judge flagged as irrelevant / distracting (extra_or_distracting_enums).", enumChips(c.extra_or_distracting_enums, c.expected_enums, c.reranked_enum_ids))}
        <hr class="enum-divider">
        ${enumRow("optimal selection", "Judge-picked optimal subset of the post-prune pool that would answer the query (optimal_enum_selection).", enumChips(c.optimal_enum_selection, c.expected_enums, c.reranked_enum_ids, "vs_expected"))}
      </div>
    </div>
    ${c.overall_explanation ? `<div class="detail-section"><h3>Overall explanation</h3><div class="body">${esc(c.overall_explanation)}</div></div>` : ""}
    ${c.retrieved_pool_inadequacy_description ? `<div class="detail-section"><h3>KB gap</h3><div class="body">${esc(c.retrieved_pool_inadequacy_description)}</div></div>` : ""}
    ${namingMismatchesHtml(c)}
    ${outputSchemaHtml(c)}
  `;
}

function schemaRow(label, value, opts) {
  opts = opts || {};
  const raw = value == null ? "" : String(value).trim();
  if (!raw && !opts.showEmpty) return "";
  const body = raw ? esc(raw) : '<em style="color:#a3b5c9">(empty)</em>';
  return `<div class="schema-row">
    <div class="schema-key">${esc(label)}</div>
    <div class="schema-val${opts.mono ? ' mono' : ''}">${body}</div>
  </div>`;
}

function namingMismatchesHtml(c) {
  const list = Array.isArray(c.enum_naming_mismatches) ? c.enum_naming_mismatches : [];
  if (!list.length) return "";
  const rows = list.map(m => {
    const kb = Array.isArray(m.kb_form) ? m.kb_form.join(" / ") : (m.kb_form || "");
    const fromList = m.raw_expected && m.raw_expected !== m.expected
      ? `<div class='naming-source'>from <code>${esc(m.raw_expected)}</code></div>`
      : "";
    return `<tr>
      <td><code class='enum-chip miss'>${esc(m.expected)}</code>${fromList}</td>
      <td><code class='enum-chip match'>${esc(kb)}</code></td>
    </tr>`;
  }).join("");
  return `<div class="detail-section">
    <h3>Deterministic checks · ENUM name mismatches</h3>
    <p style="font-size:11px;color:#a3b5c9;margin-bottom:6px">
      Computed by the report (not by the judge): expected ENUM IDs that don't match
      any retrieved KB ENUM exactly, but DO match after stripping case + non-alphanumeric
      characters. These are false-zero risks for Rel2 / enum_F1 — fix by aligning the
      naming on either the test set or the KB side.
    </p>
    <table class="tbl naming-table">
      <thead><tr><th>Expected (test set)</th><th>KB form</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  </div>`;
}

function outputSchemaHtml(c) {
  // Order matches the YAML output_schema definition. Trace metadata follows.
  const rows =
    schemaRow("case_scope", c.case_scope) +
    schemaRow("categories_list", c.categories_list, {mono: true, showEmpty: true}) +
    schemaRow("expected_reference_looks_wrong", c.expected_reference_looks_wrong ? "true" : "false") +
    schemaRow("expected_reference_issue_description", c.expected_reference_issue_description, {showEmpty: true}) +
    schemaRow("optimal_enum_selection", c.optimal_enum_selection, {mono: true, showEmpty: true}) +
    schemaRow("expected_answer_summary_with_optimal_context", c.expected_answer_summary_with_optimal_context, {showEmpty: true}) +
    schemaRow("unavailable_facts_in_selected_context", c.unavailable_facts_in_selected_context, {showEmpty: true}) +
    schemaRow("missing_facts", c.missing_facts, {showEmpty: true}) +
    schemaRow("hallucinated_claims", c.hallucinated_claims, {showEmpty: true}) +
    schemaRow("retrieved_pool_inadequacy_identified", c.retrieved_pool_inadequacy_identified ? "true" : "false") +
    schemaRow("retrieved_pool_inadequacy_description", c.retrieved_pool_inadequacy_description, {showEmpty: true}) +
    schemaRow("retrieval_improvement_suggestion", c.retrieval_improvement_suggestion, {showEmpty: true}) +
    schemaRow("reranker_improvement_suggestion", c.reranker_improvement_suggestion, {showEmpty: true}) +
    schemaRow("agent_improvement_suggestion", c.agent_improvement_suggestion, {showEmpty: true}) +
    schemaRow("kb_improvement_suggestion", c.kb_improvement_suggestion, {showEmpty: true}) +
    schemaRow("test_case_improvement_suggestion", c.test_case_improvement_suggestion, {showEmpty: true}) +
    schemaRow("overall_explanation", c.overall_explanation, {showEmpty: true}) +
    schemaRow("agents_called", c.agents_called, {mono: true, showEmpty: true}) +
    schemaRow("tools_called", c.tools_called, {mono: true, showEmpty: true}) +
    schemaRow("trace_invariant_violations", c.trace_invariant_violations, {mono: true, showEmpty: true});
  return `<div class="detail-section">
    <h3>Judge findings</h3>
    <p style="font-size:11px;color:#a3b5c9;margin-bottom:6px">
      All structured fields emitted by the judge (per the YAML output_schema): scope classification,
      defect flags, optimal selection, missing/hallucinated lists, per-team improvement suggestions,
      and the overall explanation. Per-dimension scores + reasonings are shown above; trace metadata
      is at the bottom.
    </p>
    <div class="schema-grid">${rows}</div>
  </div>`;
}

// KB tab filter state — must be declared before bindKbSearch() runs (the
// function captures it from this enclosing scope; `let` would otherwise be
// in the temporal dead zone if bindKbSearch was called first).
let kbCaseFilter = null;
searchEl.addEventListener("input", renderList);
bindChips();
bindRel2Range();
bindLatencyRange();
bindTableFilters();
bindSidebarToggle();
bindKbSearch();
bindDimMatrix();
renderList();

// kbCaseFilter is declared further up so it's already initialised when
// bindKbSearch() is called (avoids a TDZ ReferenceError).
function bindKbSearch() {
  const search = document.getElementById("kb-search");
  const list = document.getElementById("kb-list");
  const counter = document.getElementById("kb-count");
  const reset = document.getElementById("kb-reset");
  const banner = document.getElementById("kb-case-banner");
  if (!search || !list) return;
  const rows = Array.from(list.querySelectorAll(".kb-row"));
  // Pre-parse per-row case sets so update() stays cheap.
  //   rowReranked  — cases that picked this entry into reranked
  //   rowExpected  — cases that had this entry as gold ground truth
  //   rowAny       — union (data-ids), used by case-count link & fallback
  const _parseIds = ds => {
    try { return new Set((JSON.parse(ds || "[]")).map(String)); }
    catch (_) { return new Set(); }
  };
  const rowReranked = rows.map(r => _parseIds(r.dataset.rerankedCases));
  const rowExpected = rows.map(r => _parseIds(r.dataset.expectedCases));
  const rowAny      = rows.map(r => _parseIds(r.dataset.ids));

  function update() {
    const q = (search.value || "").trim().toLowerCase();
    const fid = kbCaseFilter ? String(kbCaseFilter) : null;
    // Tallies for the banner readout when a case filter is active.
    let nCorrect = 0, nMissed = 0, nDistractor = 0;
    let shown = 0;
    rows.forEach((r, i) => {
      let visible = true;
      if (q && !(r.dataset.search || "").includes(q)) visible = false;

      // Reset per-case status before reapplying (so flipping cases or
      // clearing the filter leaves no stale class behind).
      r.classList.remove("kb-status-correct", "kb-status-missed", "kb-status-distractor");
      const badge = r.querySelector(".kb-status-badge");
      if (badge) badge.textContent = "";

      if (fid) {
        const inExp = rowExpected[i].has(fid);
        const inRer = rowReranked[i].has(fid);
        // Visibility under filter: expected ∪ reranked for that case.
        if (visible && !inExp && !inRer) visible = false;
        if (visible) {
          let cls = "", txt = "";
          if (inExp && inRer)      { cls = "kb-status-correct";    txt = "Correctly selected"; nCorrect++; }
          else if (inExp)          { cls = "kb-status-missed";     txt = "Missed";             nMissed++; }
          else if (inRer)          { cls = "kb-status-distractor"; txt = "Distractor";         nDistractor++; }
          if (cls) {
            r.classList.add(cls);
            if (badge) badge.textContent = txt;
          }
        }
      }

      r.style.display = visible ? "" : "none";
      if (visible) shown++;
    });
    const tail = fid ? ` · case ${fid}` : "";
    if (counter) counter.textContent = `showing ${shown} of ${rows.length}${tail}`;
    if (reset) reset.classList.toggle("visible", !!fid);
    if (banner) {
      if (fid) {
        const total = nCorrect + nMissed + nDistractor;
        banner.innerHTML =
          `Filtered to KB entries linked to <strong>${esc(fid)}</strong> ` +
          `(expected ∪ reranker selection · ${total} entries). ` +
          `<span class="kb-banner-tally kb-banner-correct">✓ ${nCorrect} correctly selected</span> · ` +
          `<span class="kb-banner-tally kb-banner-missed">○ ${nMissed} missed</span> · ` +
          `<span class="kb-banner-tally kb-banner-distractor">✗ ${nDistractor} distractor</span>. ` +
          `Click <em>Show all</em> to clear.`;
        banner.classList.add("visible");
      } else {
        banner.classList.remove("visible");
      }
    }
  }

  search.addEventListener("input", update);
  if (reset) reset.addEventListener("click", () => { kbCaseFilter = null; update(); });

  // Fold / unfold rows when the header is clicked. Don't fold when the
  // user clicks the inner case-count link.
  list.addEventListener("click", e => {
    if (e.target.closest("a")) return;
    const head = e.target.closest(".kb-row-head");
    if (!head) return;
    head.parentElement.classList.toggle("collapsed");
  });

  // Expose so selectCase can push the selected test case into the filter.
  window._setKbCase = (caseId) => {
    kbCaseFilter = caseId || null;
    update();
  };
  update();
}

function bindSidebarToggle() {
  const layout = document.querySelector("#tab-cases .cases-layout");
  const btn = document.getElementById("sidebar-toggle");
  const peek = document.querySelector("#tab-cases .peek-zone");
  const sidebar = document.querySelector("#tab-cases .cases-sidebar");
  if (!layout || !btn) return;
  function setHidden(hidden) {
    layout.classList.toggle("sidebar-hidden", hidden);
    if (!hidden) layout.classList.remove("peek-on");
    btn.textContent = hidden ? "›" : "‹";
    btn.title = hidden ? "Show sidebar" : "Hide sidebar";
  }
  btn.addEventListener("click", () => {
    setHidden(!layout.classList.contains("sidebar-hidden"));
  });
  // Hover-to-peek with a small leave-grace so moving cursor between
  // peek-zone and sidebar doesn't flicker.
  let leaveTimer = null;
  function showPeek() {
    if (!layout.classList.contains("sidebar-hidden")) return;
    clearTimeout(leaveTimer);
    layout.classList.add("peek-on");
  }
  function hidePeek() {
    clearTimeout(leaveTimer);
    leaveTimer = setTimeout(() => layout.classList.remove("peek-on"), 120);
  }
  if (peek) {
    peek.addEventListener("mouseenter", showPeek);
    peek.addEventListener("mouseleave", hidePeek);
  }
  if (sidebar) {
    sidebar.addEventListener("mouseenter", showPeek);
    sidebar.addEventListener("mouseleave", hidePeek);
  }
}

function bindTableFilters() {
  document.querySelectorAll("table.tbl-filterable").forEach(table => {
    const rows = Array.from(table.querySelectorAll("tbody tr"));
    // filterState[col] = {type: 'text'|'range'|'multi', q|min|max|set}
    const state = {};

    function matchCol(tr, col, f) {
      const td = tr.querySelector(`td[data-col="${col}"]`);
      if (!td) return true;
      if (f.type === "text") {
        if (!f.q) return true;
        const hay = (td.dataset.val || td.textContent || "").toLowerCase();
        return hay.includes(f.q);
      }
      if (f.type === "range") {
        if (f.min == null && f.max == null) return true;
        const raw = td.dataset.numeric;
        if (raw === "" || raw == null) return false;
        const v = parseFloat(raw);
        if (!Number.isFinite(v)) return false;
        if (f.min != null && v < f.min) return false;
        if (f.max != null && v > f.max) return false;
        return true;
      }
      if (f.type === "multi") {
        if (!f.set || f.set.size === 0) return true;
        const val = td.dataset.val || td.textContent.trim();
        return f.set.has(val);
      }
      return true;
    }

    function apply() {
      rows.forEach(tr => {
        let ok = true;
        for (const [col, f] of Object.entries(state)) {
          if (!matchCol(tr, col, f)) { ok = false; break; }
        }
        tr.style.display = ok ? "" : "none";
      });
    }

    // Text filters
    table.querySelectorAll("input.col-filter").forEach(inp => {
      state[inp.dataset.col] = {type: "text", q: ""};
      inp.addEventListener("input", () => {
        state[inp.dataset.col] = {type: "text", q: inp.value.trim().toLowerCase()};
        apply();
      });
    });

    // Range filters
    table.querySelectorAll(".range-filter").forEach(rf => {
      const col = rf.dataset.col;
      state[col] = {type: "range", min: null, max: null};
      const mn = rf.querySelector(".col-range-min");
      const mx = rf.querySelector(".col-range-max");
      function upd() {
        const a = mn.value === "" ? null : parseFloat(mn.value);
        const b = mx.value === "" ? null : parseFloat(mx.value);
        state[col] = {
          type: "range",
          min: Number.isFinite(a) ? a : null,
          max: Number.isFinite(b) ? b : null,
        };
        apply();
      }
      mn.addEventListener("input", upd);
      mx.addEventListener("input", upd);
    });

    // Multi-select filters
    table.querySelectorAll(".multi-filter").forEach(mf => {
      const col = mf.dataset.col;
      state[col] = {type: "multi", set: new Set()};
      const toggle = mf.querySelector(".multi-toggle");
      const menu = mf.querySelector(".multi-menu");
      function updateLabel() {
        const n = state[col].set.size;
        toggle.textContent = n === 0 ? "all" : `${n} selected`;
        toggle.classList.toggle("has-selection", n > 0);
      }
      toggle.addEventListener("click", e => {
        e.stopPropagation();
        document.querySelectorAll(".multi-filter.open").forEach(o => {
          if (o !== mf) o.classList.remove("open");
        });
        mf.classList.toggle("open");
      });
      menu.addEventListener("click", e => e.stopPropagation());
      menu.querySelectorAll("input[type=checkbox]").forEach(cb => {
        cb.addEventListener("change", () => {
          if (cb.checked) state[col].set.add(cb.value);
          else state[col].set.delete(cb.value);
          updateLabel();
          apply();
        });
      });
    });
  });

  // Close any open multi-filter on outside click.
  document.addEventListener("click", () => {
    document.querySelectorAll(".multi-filter.open").forEach(o => o.classList.remove("open"));
  });
}

// Event delegation so anchors rendered inside filterable tables (which may be
// missed by direct querySelectorAll if table bodies are rebuilt) always work.
document.addEventListener("click", e => {
  const caseA = e.target.closest && e.target.closest("a.case-link");
  if (caseA) {
    e.preventDefault();
    // Clear filters so the case is guaranteed to appear in the sidebar.
    activeFilters.rc = null; activeFilters.flag = null;
    activeFilters.rel2_min = null; activeFilters.rel2_max = null;
    activeFilters.wavg_min = null; activeFilters.wavg_max = null;
    activeFilters.lat_min = null; activeFilters.lat_max = null;
    activeFilters.dim_pairs = [];
    activeFilters.missed_enum = null;
    activeFilters.case_id_set = null;
    activeFilters.case_id_label = null;
    activeFilters.failure_modes.clear();
    syncChipStates();
    document.querySelector('[data-target="tab-cases"]').click();
    renderList();
    selectCase(caseA.dataset.case);
    return;
  }
  const dimA = e.target.closest && e.target.closest("a.dim-link");
  if (dimA) {
    e.preventDefault();
    activeFilters.dim_pairs = [{dim: dimA.dataset.dim,
                                  score: parseInt(dimA.dataset.score, 10)}];
    document.querySelector('[data-target="tab-cases"]').click();
    renderList();
  }
  const fmClearA = e.target.closest && e.target.closest("a.fm-clear-link");
  if (fmClearA) {
    e.preventDefault();
    activeFilters.rc = null; activeFilters.flag = null;
    activeFilters.rel2_min = null; activeFilters.rel2_max = null;
    activeFilters.wavg_min = null; activeFilters.wavg_max = null;
    activeFilters.lat_min = null; activeFilters.lat_max = null;
    activeFilters.dim_pairs = [];
    activeFilters.missed_enum = null;
    activeFilters.case_id_set = null;
    activeFilters.case_id_label = null;
    activeFilters.failure_modes.clear();
    syncChipStates();
    document.querySelector('[data-target="tab-cases"]').click();
    renderList();
    return;
  }
  const judgeA = e.target.closest && e.target.closest("a.judge-eval-link");
  if (judgeA) {
    e.preventDefault();
    let ids = [];
    try { ids = JSON.parse(judgeA.dataset.ids || "[]"); } catch (_) { ids = []; }
    activeFilters.case_id_set = new Set(ids.map(String));
    activeFilters.case_id_label = judgeA.dataset.label || "";
    // Drop conflicting filters so the resulting set is purely the cases that
    // matched the clicked check / crosstab cell.
    activeFilters.dim_pairs = [];
    activeFilters.missed_enum = null;
    document.querySelector('[data-target="tab-cases"]').click();
    renderList();
  }
});

const afClear = document.getElementById("active-filter-clear");
if (afClear) afClear.addEventListener("click", () => {
  activeFilters.dim_pairs = [];
  activeFilters.missed_enum = null;
  activeFilters.case_id_set = null;
  activeFilters.case_id_label = null;
  renderList();
});

// Foldable cards — clicking the title toggles the .collapsed class. Skip
// when the click originates from a link / chart / interactive element so
// drill-down links inside the title still work. When we *expand* a card,
// trigger Plotly.Plots.resize on any charts inside; charts that rendered
// while the card was collapsed (display: none) measured their container
// at width 0 and would otherwise stay squashed.
document.addEventListener("click", e => {
  if (e.target.closest("a, button, input, select, textarea")) return;
  const title = e.target.closest(".card.foldable > .card-title");
  if (!title) return;
  const card = title.parentElement;
  card.classList.toggle("collapsed");
  if (!card.classList.contains("collapsed") && typeof Plotly !== "undefined") {
    requestAnimationFrame(() => {
      card.querySelectorAll(".js-plotly-plot").forEach(plot => {
        try { Plotly.Plots.resize(plot); } catch (_) {}
      });
    });
  }
});

// Cell click on the Summary failure-mode co-occurrence heatmap → filter
// Test Cases to rows where BOTH modes (row + col) fired in
// failure_modes_all. Per-cell customdata is [rowModeKey, colModeKey].
(function bindFailureModeCooccurrenceClick() {
  const el = document.getElementById("fm-cooccurrence-chart");
  if (!el || typeof Plotly === "undefined") return;
  function attach() {
    if (!el.on) { setTimeout(attach, 60); return; }
    el.on("plotly_click", evt => {
      if (!evt || !evt.points || !evt.points.length) return;
      const p = evt.points[0];
      if (!p.z || p.z === 0) return;
      const cd = p.customdata;
      if (!Array.isArray(cd) || cd.length < 2) return;
      const [rowKey, colKey] = cd;
      const ids = CASES.filter(c => {
        const fma = Array.isArray(c.failure_modes_all) ? c.failure_modes_all : [];
        return fma.indexOf(rowKey) >= 0 && fma.indexOf(colKey) >= 0;
      }).map(c => String(c.id));
      if (!ids.length) return;
      const sameMode = (rowKey === colKey);
      activeFilters.case_id_set = new Set(ids);
      activeFilters.case_id_label = sameMode
        ? ("failure mode: " + (FAILURE_MODE_LABELS[rowKey] || rowKey))
        : ((FAILURE_MODE_LABELS[rowKey] || rowKey) + " ∩ " + (FAILURE_MODE_LABELS[colKey] || colKey));
      activeFilters.dim_pairs = [];
      activeFilters.missed_enum = null;
      document.querySelector('[data-target="tab-cases"]').click();
      renderList();
    });
  }
  attach();
})();

// Bar click on Judge-detected failure items (Scorers tab) → filter Test
// Cases to the rows where that field had at least one detected item. The
// per-bar id list is encoded in customdata[1] (customdata[0] is the case
// count used by the hover template).
(function bindJudgeFailureFieldsClick() {
  const el = document.getElementById("judge-failure-fields-chart");
  if (!el || typeof Plotly === "undefined") return;
  function attach() {
    if (!el.on) { setTimeout(attach, 60); return; }
    el.on("plotly_click", evt => {
      if (!evt || !evt.points || !evt.points.length) return;
      const p = evt.points[0];
      const cd = p.customdata;
      const ids = Array.isArray(cd) ? cd[1] : null;
      if (!Array.isArray(ids) || !ids.length) return;
      activeFilters.case_id_set = new Set(ids.map(String));
      activeFilters.case_id_label = "judge field: " + String(p.y || "");
      activeFilters.dim_pairs = [];
      activeFilters.missed_enum = null;
      document.querySelector('[data-target="tab-cases"]').click();
      renderList();
    });
  }
  attach();
})();

// Bar click on Missed ENUMs → filter Test Cases by that ENUM ID.
(function bindMissedEnumClick() {
  const el = document.getElementById("missed-enums-chart");
  if (!el || typeof Plotly === "undefined") return;
  function attach() {
    if (!el.on) { setTimeout(attach, 60); return; }
    el.on("plotly_click", evt => {
      if (!evt || !evt.points || !evt.points.length) return;
      const enumId = evt.points[0].y;
      if (!enumId) return;
      activeFilters.missed_enum = String(enumId);
      document.querySelector('[data-target="tab-cases"]').click();
      renderList();
    });
  }
  attach();
})();

// Point click on the Latency-vs-#ENUMs scatter → filter Test Cases to
// the single clicked case. The point's customdata is [case_id]. Clicks
// on the binned-mean diamond line don't carry customdata (no case-id
// attached at the trace level) so we simply ignore them.
(function bindLatencyEnumScatterClick() {
  const el = document.getElementById("latency-vs-enums");
  if (!el || typeof Plotly === "undefined") return;
  function attach() {
    if (!el.on) { setTimeout(attach, 60); return; }
    el.on("plotly_click", evt => {
      if (!evt || !evt.points || !evt.points.length) return;
      const p = evt.points[0];
      const cd = p.customdata;
      if (!Array.isArray(cd) || !cd.length) return;
      const cid = String(cd[0] || "");
      if (!cid) return;
      activeFilters.case_id_set = new Set([cid]);
      activeFilters.case_id_label = "case " + cid;
      activeFilters.dim_pairs = [];
      activeFilters.missed_enum = null;
      document.querySelector('[data-target="tab-cases"]').click();
      renderList();
    });
  }
  attach();
})();

// Bar click on the Latency-distribution histogram → filter Test Cases to
// the rows whose total wall-clock latency falls in that bin. Per-bar
// customdata is [bin_lo_s, bin_hi_s, [case_ids]]; the last bin holds the
// p99+ tail (we clip the visual at p99 but assign all overflow cases to
// the rightmost bar so the click still reveals them).
(function bindLatencyDistributionClick() {
  const el = document.getElementById("latency-distribution");
  if (!el || typeof Plotly === "undefined") return;
  function attach() {
    if (!el.on) { setTimeout(attach, 60); return; }
    el.on("plotly_click", evt => {
      if (!evt || !evt.points || !evt.points.length) return;
      const p = evt.points[0];
      const cd = p.customdata;
      if (!Array.isArray(cd) || cd.length < 3) return;
      const lo = Number(cd[0]);
      const hi = Number(cd[1]);
      const ids = cd[2];
      if (!Array.isArray(ids) || !ids.length) return;
      activeFilters.case_id_set = new Set(ids.map(String));
      activeFilters.case_id_label = "latency " + lo.toFixed(2) + "–" + hi.toFixed(2) + "s";
      activeFilters.dim_pairs = [];
      activeFilters.missed_enum = null;
      document.querySelector('[data-target="tab-cases"]').click();
      renderList();
    });
  }
  attach();
})();

// Cell click on Dimension × score heatmap → filter Test Cases.
// Each axis label is "<dim>=<score>". Diagonal click = one pair, off-diagonal = two.
(function bindDimHeatmapClick() {
  const el = document.getElementById("dim-heatmap-chart");
  if (!el || typeof Plotly === "undefined") return;
  function parseLabel(s) {
    const m = String(s || "").match(/^(.+)=(\d)$/);
    if (!m) return null;
    return {dim: m[1], score: parseInt(m[2], 10)};
  }
  function attach() {
    if (!el.on) { setTimeout(attach, 60); return; }
    el.on("plotly_click", evt => {
      if (!evt || !evt.points || !evt.points.length) return;
      const p = evt.points[0];
      const py = parseLabel(p.y);
      const px = parseLabel(p.x);
      if (!py || !px) return;
      const pairs = [py];
      if (py.dim !== px.dim || py.score !== px.score) pairs.push(px);
      activeFilters.dim_pairs = pairs;
      document.querySelector('[data-target="tab-cases"]').click();
      renderList();
    });
  }
  attach();
})();
"""


# ─── Latency (per-step wall-clock breakdown parsed from MLflow spans) ──────
# Columns are added to the checkpoint CSV by backfill_latency.py before
# the report runs; the report degrades gracefully when they are absent.
LAT_STEP_LABELS = (
    "routing", "planning_llm", "kb_retrieve", "kb_prune",
    "kb_rerank", "tools", "generation_llm", "overhead",
)
LAT_STEP_DISPLAY = {
    "routing":        "Routing (main_agent)",
    "planning_llm":   "Planning LLM",
    "kb_retrieve":    "KB retrieve",
    "kb_prune":       "KB prune",
    "kb_rerank":      "KB rerank",
    "tools":          "Tools (non-KB)",
    "generation_llm": "Generation LLM",
    "overhead":       "Overhead",
}


def _fmt_ms(ms) -> str:
    if ms is None or (isinstance(ms, float) and (np.isnan(ms) or not np.isfinite(ms))):
        return "—"
    try:
        ms = float(ms)
    except (TypeError, ValueError):
        return "—"
    if ms >= 60_000:
        return f"{ms / 60_000:.1f}m"
    if ms >= 1_000:
        return f"{ms / 1_000:.2f}s"
    return f"{ms:.0f}ms"


def _bootstrap_mean_ci(arr: np.ndarray, *, n_resamples: int = 1000,
                       seed: int = 0) -> tuple[float, float]:
    """Bootstrap percentile 95% CI on the mean. Skewed distributions like
    latency violate the t-CI normality assumption; the percentile bootstrap
    handles them correctly."""
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_resamples, arr.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _latency_stats(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None,
                "p50": None, "p95": None}
    arr = s.to_numpy(dtype=float)
    ci_low, ci_high = _bootstrap_mean_ci(arr)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def _per_step_retry_overhead(df: pd.DataFrame) -> dict[str, "pd.Series"]:
    """Attribute each flagged retry's estimated back-off back to a step bucket.

    Returns a dict ``{step_label: pd.Series}`` where each Series is per-case
    overhead in ms attributed to that step. Buckets that don't carry LLM
    time (``kb_retrieve``, ``kb_prune``, ``tools``, ``overhead``) get the
    zero Series so callers can treat the dict uniformly.

    Attribution rules — driven by the ``bucket`` field that
    ``_classify_llm_call`` writes into ``lat_retries_json``:

    * ``routing``       — counted under ``routing``.
    * ``kb_rerank`` / ``kb_other`` — counted under ``kb_rerank`` (the only
                          knowledge-search bucket that contains LLM time).
    * ``sub_agent_llm`` — split proportionally between ``planning_llm``
                          and ``generation_llm`` based on the per-case
                          ratio of those two step times. When only
                          ``generation_llm`` is non-zero (no tool call),
                          all of it lands there. Defensible heuristic
                          because the retry detector cannot tell
                          planning- from generation-CHAT_MODEL spans
                          from ancestor chain alone (both nest under
                          ``llm`` → ``LangGraph`` → sub-agent).
    * ``other``         — unattributable; not subtracted from any step.

    NB: Per-step overhead does not have to sum to ``lat_retry_overhead_ms``
    on a case (the ``other`` bucket gets dropped). That's by design — we
    only adjust step buckets we can defend.
    """
    overhead = {label: pd.Series(0.0, index=df.index) for label in LAT_STEP_LABELS}
    if "lat_retries_json" not in df.columns:
        return overhead
    plan = pd.to_numeric(df.get("lat_planning_llm_ms"), errors="coerce").fillna(0.0)
    gen = pd.to_numeric(df.get("lat_generation_llm_ms"), errors="coerce").fillna(0.0)
    for idx, retries_json in df["lat_retries_json"].items():
        if not isinstance(retries_json, str) or not retries_json:
            continue
        try:
            retries = json.loads(retries_json)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(retries, list):
            continue
        plan_ms = float(plan.at[idx]) if idx in plan.index else 0.0
        gen_ms = float(gen.at[idx]) if idx in gen.index else 0.0
        denom = plan_ms + gen_ms
        plan_share = (plan_ms / denom) if denom > 0 else 0.0
        gen_share = 1.0 - plan_share if denom > 0 else 1.0
        for r in retries:
            try:
                o = float(r.get("overhead_ms", 0))
            except (TypeError, ValueError):
                continue
            if o <= 0:
                continue
            bucket = r.get("bucket", "")
            if bucket == "routing":
                overhead["routing"].at[idx] += o
            elif bucket in ("kb_rerank", "kb_other"):
                overhead["kb_rerank"].at[idx] += o
            elif bucket == "sub_agent_llm":
                overhead["planning_llm"].at[idx] += o * plan_share
                overhead["generation_llm"].at[idx] += o * gen_share
            # "other": skip — can't attribute.
    return overhead


def _latency_summary_rows(df: pd.DataFrame) -> list[dict]:
    if "lat_total_ms" not in df.columns:
        return []
    total = _latency_stats(df["lat_total_ms"])
    if total["n"] == 0:
        return []
    # Adjusted total: observed - per-case retry overhead, clamped at 0.
    clean_total_series = _adjusted_latency_series(df)
    total_adj = (
        _latency_stats(clean_total_series) if clean_total_series is not None
        else {"n": 0, "mean": None, "ci_low": None, "ci_high": None,
              "p50": None, "p95": None}
    )
    # Per-step retry overhead (attributed per the rules in
    # `_per_step_retry_overhead`). When no retry data is present every
    # series is zero, so `adj_*` ends up equal to `*` and the report
    # silently skips the "without throttling" line per row.
    overhead_by_step = _per_step_retry_overhead(df)
    rows: list[dict] = []
    for label in LAT_STEP_LABELS:
        col = f"lat_{label}_ms"
        if col not in df.columns:
            continue
        obs_series = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        st = _latency_stats(obs_series)
        adj_series = (obs_series - overhead_by_step.get(label, 0.0)).clip(lower=0.0)
        adj = _latency_stats(adj_series)
        share = ((st["mean"] / total["mean"])
                 if (total["mean"] and st["mean"] is not None) else None)
        rows.append({
            "label": label, **st, "share": share,
            "adj_mean": adj["mean"], "adj_ci_low": adj["ci_low"],
            "adj_ci_high": adj["ci_high"], "adj_p50": adj["p50"],
            "adj_p95": adj["p95"],
        })
    # Bottleneck on top: descending share, with missing shares last.
    rows.sort(key=lambda r: (r["share"] is None, -(r["share"] or 0)))
    rows.append({
        "label": "total", **total, "share": 1.0,
        "adj_mean": total_adj["mean"], "adj_ci_low": total_adj["ci_low"],
        "adj_ci_high": total_adj["ci_high"], "adj_p50": total_adj["p50"],
        "adj_p95": total_adj["p95"],
    })
    return rows


def _lat_ci_bar_html(mean, ci_low, ci_high, *, total_mean: float | None) -> str:
    """Horizontal mini-bar visualising a step's 95% CI relative to the run's
    mean total latency. The full bar width corresponds to ``total_mean``; the
    coloured band marks ``[ci_low, ci_high]`` and the diamond marks ``mean``.

    Falls back to a text-only range when any value or the reference total is
    missing — the column still renders, just without the visual.
    """
    if (mean is None or ci_low is None or ci_high is None
            or not total_mean or total_mean <= 0):
        if ci_low is not None and ci_high is not None:
            return (
                "<span class='lat-ci-text-only'>"
                f"[{_fmt_ms(ci_low)} … {_fmt_ms(ci_high)}]</span>"
            )
        return "—"
    lo_pct = max(0.0, min(100.0, (ci_low / total_mean) * 100))
    hi_pct = max(0.0, min(100.0, (ci_high / total_mean) * 100))
    mean_pct = max(0.0, min(100.0, (mean / total_mean) * 100))
    band_w = max(hi_pct - lo_pct, 0.4)  # 0.4% min so a tight CI is still visible
    label = f"[{_fmt_ms(ci_low)} … {_fmt_ms(ci_high)}]"
    return (
        f"<div class='lat-ci-vis' title='{label} · scale: full bar = mean total'>"
        f"<div class='lat-ci-band' style='left:{lo_pct:.2f}%;width:{band_w:.2f}%'></div>"
        f"<div class='lat-ci-mark' style='left:{mean_pct:.2f}%'></div>"
        f"</div>"
        f"<div class='lat-ci-text'>{label}</div>"
    )


def _latency_retry_summary(df: pd.DataFrame) -> dict | None:
    """Run-level retry metrics. Returns ``None`` when the columns are
    absent or no traces carry retry data.

    Computed separately from the step rows because retries OVERLAP with the
    LLM buckets (each retried call's wall time is already counted in
    routing / planning_llm / generation_llm / kb_rerank). Summing them
    into the total would double-count, so the report keeps them as their
    own row below the step table.
    """
    if "lat_retry_call_count" not in df.columns or "lat_retry_overhead_ms" not in df.columns:
        return None
    counts = pd.to_numeric(df["lat_retry_call_count"], errors="coerce").fillna(0).astype(int)
    overhead = pd.to_numeric(df["lat_retry_overhead_ms"], errors="coerce").fillna(0.0)
    n_total = int(counts.notna().sum())
    if n_total == 0:
        return None
    n_with = int((counts > 0).sum())
    # Stats on the subset of cases that actually had a retry — the
    # zero-retry cases dominate the population and would drown out the
    # mean otherwise.
    with_mask = counts > 0
    if with_mask.any():
        stats = _latency_stats(overhead[with_mask])
    else:
        stats = {"n": 0, "mean": None, "ci_low": None, "ci_high": None,
                 "p50": None, "p95": None}
    return {
        "n_total": n_total,
        "n_with_retries": n_with,
        "share_with_retries": (n_with / n_total) if n_total else 0.0,
        "calls_total": int(counts.sum()),
        **stats,
    }


def _latency_summary_card_html(df: pd.DataFrame) -> str:
    rows = _latency_summary_rows(df)
    if not rows:
        return ""
    # The Total row sets the scale for every per-step CI bar. Pull it once.
    total_row = next((r for r in rows if r["label"] == "total"), None)
    total_mean = total_row["mean"] if total_row else None
    body_rows: list[str] = []
    for r in rows:
        is_total = r["label"] == "total"
        cls = " class='lat-total-row'" if is_total else ""
        display = "Total" if is_total else LAT_STEP_DISPLAY.get(r["label"], r["label"])
        share_str = f"{r['share'] * 100:.1f}%" if r.get("share") is not None else "—"
        bar_html = ""
        if not is_total and r.get("share") is not None:
            pct = max(0.0, min(100.0, r["share"] * 100))
            bar_html = (f"<div class='lat-bar' title='{pct:.1f}% of mean total'>"
                        f"<div class='lat-bar-fill' style='width:{pct:.1f}%'></div></div>")
        # The Total row's CI is shown text-only — visualising it against
        # itself is uninformative (band would always span the whole bar).
        if is_total:
            if r.get("ci_low") is not None and r.get("ci_high") is not None:
                ci_html = (
                    f"<span class='lat-ci-text-only'>"
                    f"[{_fmt_ms(r['ci_low'])} … {_fmt_ms(r['ci_high'])}]</span>"
                )
            else:
                ci_html = "—"
        else:
            ci_html = _lat_ci_bar_html(
                r.get("mean"), r.get("ci_low"), r.get("ci_high"),
                total_mean=total_mean,
            )
        # When a step's retry-adjusted mean differs meaningfully from the
        # observed mean (≥1%) we tuck a small "without throttling" line
        # under the observed number — same column, smaller font, amber
        # delta, so a reader scanning the Mean column sees both the as-
        # observed value and what it would be if back-off was stripped.
        # Steps with no LLM time (kb_retrieve, kb_prune, tools, overhead)
        # never carry retry overhead, so this line just doesn't render
        # for them.
        adj_mean_html = ""
        if r.get("adj_mean") is not None and r.get("mean"):
            adj_mean = r["adj_mean"]
            mean_val = r["mean"]
            if mean_val > 0 and abs(adj_mean - mean_val) / mean_val >= 0.01:
                delta_pct = 100.0 * (adj_mean - mean_val) / mean_val
                adj_mean_html = (
                    f"<div class='lat-adj-line' title='Same mean with the "
                    f"estimated per-step retry back-off subtracted "
                    f"(clamped at 0)'>"
                    f"{_fmt_ms(adj_mean)} "
                    f"<span class='lat-adj-delta'>({delta_pct:+.1f}%)</span>"
                    f"</div>"
                )
        adj_p95_html = ""
        if r.get("adj_p95") is not None and r.get("p95"):
            ap = r["adj_p95"]
            pv = r["p95"]
            if pv > 0 and abs(ap - pv) / pv >= 0.01:
                adj_p95_html = (
                    f"<div class='lat-adj-line'>{_fmt_ms(ap)}</div>"
                )
        body_rows.append(
            f"<tr{cls}>"
            f"<td>{display}</td>"
            f"<td style='text-align:right'>{r['n']}</td>"
            f"<td style='text-align:right'>{_fmt_ms(r['mean'])}{adj_mean_html}</td>"
            f"<td class='lat-ci-cell'>{ci_html}</td>"
            f"<td style='text-align:right'>{_fmt_ms(r['p50'])}</td>"
            f"<td style='text-align:right'>{_fmt_ms(r['p95'])}{adj_p95_html}</td>"
            f"<td style='text-align:right;white-space:nowrap'>{share_str}{bar_html}</td>"
            f"</tr>"
        )
    # Retry overhead — rendered as its own row below the table because it
    # OVERLAPS with the LLM step buckets (it's a slice of routing /
    # planning_llm / generation_llm / kb_rerank wall time, not an
    # independent step). Adding it to the table would visually invite
    # double-counting.
    retry = _latency_retry_summary(df)
    retry_block = ""
    if retry is not None and retry["n_with_retries"] > 0:
        share_pct = retry["share_with_retries"] * 100
        ci_html = (
            f"[{_fmt_ms(retry['ci_low'])} … {_fmt_ms(retry['ci_high'])}]"
            if retry["ci_low"] is not None and retry["ci_high"] is not None
            else "—"
        )
        retry_block = (
            '<div class="lat-retry-row">'
            '<div class="lat-retry-title">Retry overhead <span class="lat-axis-hint">— overlaps with LLM buckets, not summed into total</span></div>'
            '<table class="tbl lat-tbl lat-retry-tbl">'
            '<thead><tr>'
            '<th>—</th>'
            '<th style="text-align:right">cases w/ retries</th>'
            '<th style="text-align:right">Mean overhead</th>'
            '<th style="text-align:right">95% CI (mean)</th>'
            '<th style="text-align:right">p50</th>'
            '<th style="text-align:right">p95</th>'
            '<th style="text-align:right">Retry rate</th>'
            '</tr></thead>'
            '<tbody>'
            '<tr>'
            '<td>Suspected throttle / retry</td>'
            f'<td style="text-align:right">{retry["n_with_retries"]:,} / {retry["n_total"]:,}'
            f' <span class="lat-axis-hint">({retry["calls_total"]:,} call(s))</span></td>'
            f'<td style="text-align:right">{_fmt_ms(retry["mean"])}</td>'
            f'<td style="text-align:right;color:#5c7999;font-size:11px">{ci_html}</td>'
            f'<td style="text-align:right">{_fmt_ms(retry["p50"])}</td>'
            f'<td style="text-align:right">{_fmt_ms(retry["p95"])}</td>'
            f'<td style="text-align:right">{share_pct:.1f}%</td>'
            '</tr>'
            '</tbody></table></div>'
        )
    return (
        '<div class="card">'
        '<div class="card-title">Latency breakdown ' + _info_icon(
            "Per-step wall-clock latency parsed from MLflow trace spans. "
            "Mean: arithmetic mean across cases. 95% CI: bootstrap percentile "
            "(1000 resamples) on the mean — robust to the right-skew typical "
            "of latency data. The CI column is drawn to scale: the full bar "
            "width equals the mean total latency, so you can read each step's "
            "absolute position and CI tightness at a glance. "
            "p50 / p95: per-case order statistics. "
            "Share: mean step time / mean total time. Steps are mutually "
            "non-overlapping; overhead absorbs LangGraph wiring time. "
            "Sorted by share desc — the bottleneck is on top. "
            "A small amber line under Mean / p95 shows the same number "
            "without the estimated retry back-off (when it differs by "
            "≥1%) — answers 'what would this step look like with no "
            "throttling?'. Retry-overhead row below the table aggregates "
            "the same overhead across the run."
        ) + "</div>"
        '<table class="tbl lat-tbl">'
        '<thead><tr>'
        '<th>Step</th>'
        '<th style="text-align:right">n</th>'
        '<th style="text-align:right">Mean</th>'
        '<th>95% CI (mean) <span class="lat-axis-hint">— full bar = mean total</span></th>'
        '<th style="text-align:right">p50</th>'
        '<th style="text-align:right">p95</th>'
        '<th style="text-align:right">Share</th>'
        '</tr></thead>'
        '<tbody>' + "".join(body_rows) + '</tbody>'
        '</table>'
        + retry_block +
        '</div>'
    )


def _build_latency_distribution_fig(df: pd.DataFrame):
    """Histogram of per-trace total latency with vertical lines marking the
    bootstrap 95% CI on the mean, plus p50 / p95 reference lines.

    X-axis is in seconds (latencies usually run multi-second so milliseconds
    would be hard to read). The histogram is clipped at the p99 mark to keep
    a few extreme outliers from squashing the rest of the distribution; a
    note in the card title makes the clipping explicit.
    """
    if "lat_total_ms" not in df.columns:
        return None
    s = pd.to_numeric(df["lat_total_ms"], errors="coerce").dropna()
    if s.empty:
        return None
    arr_s = (s.to_numpy(dtype=float) / 1000.0)
    stats = _latency_stats(s)
    if stats["mean"] is None:
        return None
    mean_s = stats["mean"] / 1000.0
    ci_lo_s = stats["ci_low"] / 1000.0
    ci_hi_s = stats["ci_high"] / 1000.0
    p50_s = stats["p50"] / 1000.0
    p95_s = stats["p95"] / 1000.0

    n = len(arr_s)
    upper = float(np.percentile(arr_s, 99))
    if upper <= 0:
        upper = float(arr_s.max() or 1.0)
    n_bins = max(20, min(60, int(np.sqrt(n) * 2)))
    bin_edges = np.linspace(0.0, upper, n_bins + 1)
    clipped = np.clip(arr_s, 0.0, upper)
    counts, _ = np.histogram(clipped, bins=bin_edges)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Build per-bin lists of test_case_ids so clicking a bar can filter the
    # Test Cases tab. Outliers above the p99 clip land in the last bin —
    # matches the visual: the bar at the right edge holds the long tail.
    case_ids = (df["test_case_id"].astype(str).tolist()
                if "test_case_id" in df.columns
                else [""] * len(arr_s))
    bin_idx = np.digitize(clipped, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, len(bin_edges) - 2)
    per_bin_ids: list[list[str]] = [[] for _ in range(len(bin_edges) - 1)]
    for cid, bi in zip(case_ids, bin_idx, strict=False):
        per_bin_ids[int(bi)].append(cid)
    # customdata schema per bar: [bin_lo_s, bin_hi_s, [case_ids…]]. The JS
    # click handler reads cd[2] as the filter set and cd[0]/cd[1] for the
    # range label shown on the active-filter banner.
    customdata = [
        [float(bin_edges[i]), float(bin_edges[i + 1]), per_bin_ids[i]]
        for i in range(len(bin_edges) - 1)
    ]

    fig = go.Figure(go.Bar(
        x=centers, y=counts,
        marker=dict(color=GE_BLUE, line=dict(width=0)),
        width=(bin_edges[1] - bin_edges[0]) * 0.96,
        customdata=customdata,
        hovertemplate=(
            "<b>%{customdata[0]:.2f}s – %{customdata[1]:.2f}s</b><br>"
            "cases: %{y}<br>"
            "<i>click to filter Test Cases</i><extra></extra>"
        ),
        showlegend=False,
    ))
    # Mean + 95% CI annotations all live above the plot area on the same
    # horizontal line as the rest. Annotation y is anchored to "paper"
    # coordinates with a small yshift so labels sit just above the top axis
    # and never collide with bars (which is what the bottom-anchored p50/p95
    # used to do).
    def _vline_at_top(x, *, color, dash="solid", width=1, label, size=10, yshift=0,
                       xanchor="center"):
        fig.add_shape(type="line", xref="x", yref="paper",
                      x0=x, x1=x, y0=0, y1=1,
                      line=dict(color=color, width=width, dash=dash))
        fig.add_annotation(x=x, xref="x", y=1.0, yref="paper",
                            yshift=10 + yshift,
                            text=label, showarrow=False,
                            font=dict(color=color, size=size),
                            xanchor=xanchor, yanchor="bottom",
                            bgcolor="rgba(255,255,255,0.85)")

    # Lines drawn in x-axis order so closer labels can stagger via yshift if
    # they're too close together (p50/CI-lo cluster, mean/CI-hi cluster).
    _vline_at_top(p50_s, color=GE_MUTED, dash="dot", width=1,
                   label=f"p50 {p50_s:.2f}s", size=10, xanchor="right")
    _vline_at_top(ci_lo_s, color=GE_BLUE, dash="dash", width=1,
                   label=f"CI lo {ci_lo_s:.2f}s", size=10, xanchor="right",
                   yshift=14)
    _vline_at_top(mean_s, color=GE_BLUE, dash="solid", width=2,
                   label=f"mean {mean_s:.2f}s", size=11)
    _vline_at_top(ci_hi_s, color=GE_BLUE, dash="dash", width=1,
                   label=f"CI hi {ci_hi_s:.2f}s", size=10, xanchor="left",
                   yshift=14)
    _vline_at_top(p95_s, color=GE_RED, dash="dot", width=1,
                   label=f"p95 {p95_s:.2f}s", size=10, xanchor="left")

    _style_fig(fig, height=380)
    fig.update_layout(
        # Extra top margin so the staggered annotations have room.
        margin=dict(t=70, b=40, l=50, r=20),
        bargap=0.04,
        xaxis_title="trace latency (s) · clipped at p99",
        yaxis_title="count",
    )
    return fig


def _adjusted_latency_series(df: pd.DataFrame) -> "pd.Series | None":
    """Per-case latency with the estimated retry back-off subtracted.

    Returns ``None`` when the retry columns are missing on the dataframe.
    Clamped at 0 so an over-estimate of overhead cannot push a case below
    zero. Used to compute the "without estimated throttling" mean / CI /
    p50 / p95 shown on the headline Latency card.
    """
    if "lat_total_ms" not in df.columns or "lat_retry_overhead_ms" not in df.columns:
        return None
    total = pd.to_numeric(df["lat_total_ms"], errors="coerce")
    overhead = pd.to_numeric(df["lat_retry_overhead_ms"], errors="coerce").fillna(0.0)
    clean = (total - overhead).clip(lower=0.0)
    return clean


def _build_latency_vs_enum_count_fig(df: pd.DataFrame):
    """Scatter of reranked-ENUM count vs *generation-LLM* latency.

    Directly tests the hypothesis: "more selected ENUMs → bigger
    generation-LLM context → longer generation time". Plotting against
    ``lat_total_ms`` was misleading because cases that never invoked the
    KB pipeline are pinned at enum_count=0 with their latency dominated
    by main-agent routing time. We filter to cases where the KB
    pipeline ran (``post_prune_enum_count > 0``) so the relationship
    being plotted is the one the user is actually asking about.

    Retry-flagged cases are coloured red so any clustering of retries
    at high enum counts (which would support the "more context → more
    TPM pressure → more throttling" side of the hypothesis) is
    immediately visible. A binned mean line (per integer enum count,
    dropping bins with fewer than 3 cases) shows the trend, and the
    Pearson r between count and generation latency is printed in the
    x-axis label.

    Customdata on each point is ``[case_id]`` so clicking a point opens
    that case in the Test Cases tab (see
    ``bindLatencyEnumScatterClick``).
    """
    needed = ("lat_generation_llm_ms", "_reranked_enum_ids", "post_prune_enum_count")
    if any(c not in df.columns for c in needed):
        return None
    gen_lat = pd.to_numeric(df["lat_generation_llm_ms"], errors="coerce")
    counts = df["_reranked_enum_ids"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    post_prune = pd.to_numeric(df["post_prune_enum_count"], errors="coerce").fillna(0)
    rc = pd.to_numeric(
        df.get("lat_retry_call_count", pd.Series(0, index=df.index)),
        errors="coerce",
    ).fillna(0).astype(int)
    case_ids = (df["test_case_id"].astype(str)
                if "test_case_id" in df.columns
                else pd.Series(["?"] * len(df), index=df.index))
    # Filter: only cases where the KB pipeline actually ran. For cases
    # that took the chit-chat / no-tool path, reranked_enum_count is
    # structurally 0 and the generation latency reflects routing-LLM
    # rather than context-driven generation — including them would
    # bias the correlation toward 0.
    valid = gen_lat.notna() & (post_prune > 0)
    if not valid.any():
        return None
    lat_s = gen_lat[valid].to_numpy(dtype=float) / 1000.0
    counts_v = counts[valid].to_numpy(dtype=int)
    rc_v = rc[valid].to_numpy(dtype=int)
    case_ids_v = case_ids[valid].to_numpy()
    has_retry = rc_v > 0
    no_retry = ~has_retry

    fig = go.Figure()
    # Background layer: cases with no flagged retry. Semi-transparent so
    # the red retry points and the binned-mean line still read clearly.
    if no_retry.any():
        fig.add_trace(go.Scatter(
            x=counts_v[no_retry], y=lat_s[no_retry],
            mode="markers",
            name=f"no retry (n={int(no_retry.sum())})",
            marker=dict(color=GE_BLUE, size=6, opacity=0.35,
                        line=dict(width=0)),
            customdata=case_ids_v[no_retry].reshape(-1, 1),
            hovertemplate=("<b>%{customdata[0]}</b><br>"
                            "ENUMs: %{x}<br>latency: %{y:.2f}s"
                            "<extra></extra>"),
        ))
    # Foreground layer: retry-flagged cases. Red, larger, bordered.
    if has_retry.any():
        fig.add_trace(go.Scatter(
            x=counts_v[has_retry], y=lat_s[has_retry],
            mode="markers",
            name=f"retry flagged (n={int(has_retry.sum())})",
            marker=dict(color=GE_RED, size=8, opacity=0.75,
                        line=dict(color="#ffffff", width=0.5)),
            customdata=case_ids_v[has_retry].reshape(-1, 1),
            hovertemplate=("<b>%{customdata[0]}</b><br>"
                            "ENUMs: %{x}<br>latency: %{y:.2f}s"
                            "<br><i>retry flagged</i><extra></extra>"),
        ))
    # Binned mean trend line — only buckets with n>=3 so a single outlier
    # doesn't yank a bucket's mean.
    bin_df = pd.DataFrame({"c": counts_v, "l": lat_s})
    grouped = (bin_df.groupby("c")
                       .agg(mean=("l", "mean"), n=("l", "count"))
                       .reset_index())
    grouped = grouped[grouped["n"] >= 3].sort_values("c")
    if not grouped.empty:
        fig.add_trace(go.Scatter(
            x=grouped["c"], y=grouped["mean"],
            mode="lines+markers", name="mean per bin (n≥3)",
            line=dict(color=GE_TEXT, width=2),
            marker=dict(color=GE_TEXT, size=9, symbol="diamond"),
            hovertemplate=("ENUMs: %{x}<br>"
                            "mean latency: %{y:.2f}s<extra></extra>"),
        ))
    # Correlation summary printed in the x-axis label so the reader gets
    # a numeric "is there a trend?" answer alongside the visual.
    if len(lat_s) >= 3:
        r = float(np.corrcoef(counts_v.astype(float), lat_s)[0, 1])
        r_text = f"Pearson r = {r:+.2f}"
    else:
        r_text = "Pearson r = n/a"
    _style_fig(fig, height=420)
    fig.update_layout(
        margin=dict(t=20, b=70, l=60, r=20),
        xaxis_title=f"# reranked ENUMs · {r_text} · KB-running cases only (n={len(lat_s)})",
        yaxis_title="generation LLM latency (s)",
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
    )
    return fig


def _throttling_detail_row(df: pd.DataFrame) -> str:
    """Throttling info appended inside the headline Latency card.

    Two pieces, only rendered when at least one retry was flagged:

    1. How many traces / calls were affected and how much wall time the
       estimator attributes to back-off — the same line that was there
       before, kept for the run-level scope.
    2. A *retry-adjusted* mean / CI / p50 / p95, shown right under the
       throttling line. This answers the question "if we could remove
       the throttle back-off entirely, what would the run look like?".
       Computed per-case (``lat_total_ms - lat_retry_overhead_ms``,
       clamped at 0) so the bootstrap CI matches the same arithmetic as
       the unadjusted headline mean.
    """
    if "lat_retry_overhead_ms" not in df.columns or "lat_retry_call_count" not in df.columns:
        return ""
    rc = pd.to_numeric(df["lat_retry_call_count"], errors="coerce").fillna(0)
    ro = pd.to_numeric(df["lat_retry_overhead_ms"], errors="coerce").fillna(0)
    total_overhead_ms = float(ro.sum())
    n_calls = int(rc.sum())
    n_traces = int((rc > 0).sum())
    if n_calls == 0 or total_overhead_ms <= 0:
        return ""
    lat_sum = float(pd.to_numeric(df["lat_total_ms"], errors="coerce").fillna(0).sum())
    pct = (100.0 * total_overhead_ms / lat_sum) if lat_sum > 0 else 0.0
    tip = (
        "Heuristic — APPROXIMATION, not ground truth. CHAT_MODEL spans "
        "are flagged when their duration crosses a bucket-specific "
        "minimum AND their output_tokens/sec sits below a bucket-specific "
        "cap. Per-bucket thresholds (default): routing dur≥6s, tok/s&lt;5; "
        "sub_agent_llm dur≥35s, tok/s&lt;8; kb_rerank dur≥18s, "
        "tok/s&lt;4. The OTEL spans carry no explicit retry marker "
        "(the Databricks/LangChain SDK swallows 429s), so an estimated "
        f"overhead = duration − output_tokens / {RETRY_BASELINE_TOK_PER_S} "
        "is reported per flagged call. See Notes for full definitions."
    )
    # Run-level: how much overhead was observed.
    throttle_line = (
        '<div class="hc-detail lat-hc-throttle">'
        f'<span>Throttling {_info_icon(tip)} <strong>{n_traces}</strong> traces · '
        f'<strong>{n_calls}</strong> calls · '
        f'<strong>{_fmt_ms(total_overhead_ms)}</strong> overhead '
        f'(<strong>{pct:.1f}%</strong> of total wall)</span>'
        '</div>'
    )
    # What the same stats look like with the estimated overhead subtracted
    # per-case. p50 usually shifts very little (most cases have no retry,
    # so the median is set by an unretried case); the mean and p95 are
    # where the difference shows up.
    clean = _adjusted_latency_series(df)
    if clean is None:
        return throttle_line
    clean_st = _latency_stats(clean)
    obs_st = _latency_stats(pd.to_numeric(df["lat_total_ms"], errors="coerce"))
    if clean_st["n"] == 0 or obs_st["mean"] is None:
        return throttle_line
    delta = clean_st["mean"] - obs_st["mean"]
    delta_pct = (100.0 * delta / obs_st["mean"]) if obs_st["mean"] else 0.0
    # _fmt_ms is sign-blind (treats -1439 as a sub-second value), so format
    # the absolute drop and prepend the sign explicitly.
    delta_sign = "−" if delta < 0 else "+"
    delta_str = f"{delta_sign}{_fmt_ms(abs(delta))}"
    adjusted_tip = (
        "Same population, re-computed with the estimated retry back-off "
        "subtracted per case (clamped at 0). Read it as: "
        "'if the SDK never had to back off, this is what the run would "
        "look like'. The CI here uses the same 1000-resample percentile "
        "bootstrap as the observed mean. p50 usually moves very little "
        "(most cases have no retry, so the median is unaffected); the "
        "biggest change is in the mean and p95."
    )
    adjusted_line = (
        '<div class="hc-detail lat-hc-adjusted">'
        f'<span class="lat-hc-adjusted-title">Without throttling {_info_icon(adjusted_tip)}</span>'
        '<span class="lat-hc-adjusted-stats">'
        f'mean <strong>{_fmt_ms(clean_st["mean"])}</strong> '
        f'<span class="lat-hc-adjusted-delta">({delta_pct:+.1f}%, '
        f'{delta_str})</span> · '
        f'CI <strong>[{_fmt_ms(clean_st["ci_low"])} – {_fmt_ms(clean_st["ci_high"])}]</strong> · '
        f'p50 <strong>{_fmt_ms(clean_st["p50"])}</strong> · '
        f'p95 <strong>{_fmt_ms(clean_st["p95"])}</strong>'
        '</span>'
        '</div>'
    )
    return throttle_line + adjusted_line


def _latency_headline_card_html(df: pd.DataFrame) -> str:
    """Compact headline-style card with the run's overall mean latency,
    a centred CI bracket visualisation, and p50/p95 below. Designed to sit
    on Summary row 2 alongside the Test-cases card.
    """
    if "lat_total_ms" not in df.columns:
        return ""
    st = _latency_stats(df["lat_total_ms"])
    if st["n"] == 0 or st["mean"] is None:
        return ""
    return (
        '<div class="headline-card lat-headline-card">'
        '<div class="hc-label">Latency · mean <span class="lat-hc-label-aux">(observed)</span> ' + _info_icon(
            "Mean end-to-end latency across all parsed traces — exactly "
            "what the user/system experienced, back-off sleep included. "
            "The bracket below shows the 95% bootstrap CI (1000 resamples) on "
            "the mean — wide bracket means the run-level mean is noisy. "
            "p50 / p95 are per-case order statistics; p95 is the right-tail "
            "experience that dominates user complaints. "
            "If a throttling line is shown below, the 'Without throttling' "
            "row is the same arithmetic with the estimated retry back-off "
            "subtracted per case — a what-if for the same population. "
            "See the Notes tab for the full per-step definitions."
        ) + '</div>'
        f'<div class="hc-value lat-hc-value">{_fmt_ms(st["mean"])}</div>'
        '<div class="lat-hc-ci">'
        '<div class="lat-hc-ci-bracket">'
        '<span class="lat-hc-ci-tick lat-hc-ci-tick-left"></span>'
        '<span class="lat-hc-ci-line"></span>'
        '<span class="lat-hc-ci-dot" title="mean"></span>'
        '<span class="lat-hc-ci-line"></span>'
        '<span class="lat-hc-ci-tick lat-hc-ci-tick-right"></span>'
        '</div>'
        '<div class="lat-hc-ci-labels">'
        f'<span>{_fmt_ms(st["ci_low"])}</span>'
        '<span class="lat-hc-ci-mid">95% CI (mean)</span>'
        f'<span>{_fmt_ms(st["ci_high"])}</span>'
        '</div>'
        '</div>'
        '<div class="hc-detail lat-hc-detail">'
        f'<span>p50 <strong>{_fmt_ms(st["p50"])}</strong></span>'
        '<span class="lat-hc-detail-sep">·</span>'
        f'<span>p95 <strong>{_fmt_ms(st["p95"])}</strong></span>'
        f'<span class="lat-hc-detail-n">n = {st["n"]:,}</span>'
        '</div>'
        + _throttling_detail_row(df) +
        '</div>'
    )


def render_html(df: pd.DataFrame, *, df_all: pd.DataFrame | None = None,
                yaml_name: str, reasoning_effort: str,
                checkpoint_label: str, judge_model: str = "unknown",
                experiment_name: str | None = None,
                mlflow_run_id: str = "",
                mlflow_experiment_id: str = "",
                mlflow_run_timestamp: str = "",
                include_latency: bool = False,
                checkpoint_path: Path | None = None,
                prompts_path: Path | None = None,
                baseline: dict | None = None) -> str:
    if df_all is None:
        df_all = df
    if not experiment_name:
        experiment_name = yaml_name
    reranker_miss, retriever_miss = Counter(), Counter()
    if "_missing_enums_in_candidate_pool" in df.columns:
        for lst in df["_missing_enums_in_candidate_pool"]:
            for e in lst:
                reranker_miss[e] += 1
    if "_missing_enums_not_in_pool" in df.columns:
        for lst in df["_missing_enums_not_in_pool"]:
            for e in lst:
                retriever_miss[e] += 1

    # _build_figs returns (fig_dim, fig_hist, fig_rc, fig_missed). fig_hist
    # is rendered next to the Pass-threshold card on the Notes tab to show
    # the run's weighted_avg distribution at a glance; fig_missed feeds the
    # Retrieval Findings tab. fig_dim and fig_rc are unused but kept here.
    _, fig_wavg_hist, _, fig_missed = _build_figs(df, reranker_miss, retriever_miss)
    # The latency figures are only built when the surface is included so
    # we don't pay the plotly cost for a shareable report that suppresses
    # everything latency-related anyway.
    fig_lat_dist = _build_latency_distribution_fig(df) if include_latency else None
    fig_lat_vs_enums = _build_latency_vs_enum_count_fig(df) if include_latency else None
    fig_dim_heatmap = _build_dim_heatmap(df)
    enum_count_dist_html = _enum_count_distribution_table_html(df)
    fig_fm_cooccurrence = _build_failure_mode_cooccurrence_fig(df)
    judge_eval_html = _judge_eval_html(df)
    kb_html = _render_kb_html(_build_kb_data(df))
    fig_rel2_expert = _build_rel2_expert_scatter(df)
    fig_rel2_wavg = _build_rel2_wavg_scatter(df)
    dim_descriptions = _load_dimension_descriptions(yaml_name)
    dim_full_info = _load_dimension_full_info(yaml_name)
    corr = _build_corr_figs(df)

    summary_metrics = compute_summary_metrics(df, df_all)
    naming_mismatches_agg = _aggregate_naming_mismatches(df)
    pass_rate = df["pass"].mean()

    prompts_sidecar = _load_prompt_sidecar(prompts_path, checkpoint_path, mlflow_run_id)
    prompt_warnings = _prompt_hash_warnings(df_all if df_all is not None else df)
    prompt_warning_card = _prompt_warning_card(prompt_warnings)
    prompts_tab_html = _prompts_tab(prompts_sidecar, mlflow_run_id)

    def _count_cell(dim: str, score: int, n: int) -> str:
        """Clickable count cell — jumps to Test Cases with dim+score applied."""
        if n == 0:
            return f"<td class='dim-count empty'>{n}</td>"
        cls = {0: "num-bad", 1: "num-mid", 2: "num-good"}[score]
        return (f'<td class="dim-count {cls}">'
                f'<a class="dim-link" href="#" data-dim="{dim}" data-score="{score}"'
                f' title="Show these {n} cases in the Test Cases tab">{n}</a></td>')

    dim_rows_html = ""
    for dim in DIMENSION_WEIGHTS:
        col = f"{dim}_score"
        counts = df[col].value_counts().reindex([0, 1, 2]).fillna(0).astype(int)
        desc = dim_descriptions.get(dim, "")
        desc_html = _h(desc).replace("\n", "<br>")
        # Render YAML backtick spans as <code> for readability (after escaping).
        desc_html = re.sub(r"`([^`]+)`", r"<code>\1</code>", desc_html)
        dim_rows_html += (
            f"<tr><td><div class='dim-cell-name'>{dim}</div>"
            f"<div class='dim-cell-desc'>{desc_html}</div></td>"
            f"<td>{df[col].mean():.2f}</td>"
            f"{_count_cell(dim, 0, int(counts[0]))}"
            f"{_count_cell(dim, 1, int(counts[1]))}"
            f"{_count_cell(dim, 2, int(counts[2]))}"
            f"<td>{(df[col] == 0).mean():.1%}</td>"
            f"<td>{(df[col] == 1).mean():.1%}</td>"
            f"<td>{(df[col] == 2).mean():.1%}</td></tr>"
        )

    def _tid_key(tid: str) -> tuple[int, str]:
        m = re.search(r"\d+", str(tid))
        return (int(m.group()) if m else 10**9, str(tid))

    # Build the per-case JSON from df_all (not df) so the Test Cases tab can
    # surface excluded rows too — empty user_query and non-KB scope cases
    # carry an `excluded_reason` tag so they're visually distinguishable but
    # still inspectable when a Summary failure-mode link drills into them.
    baseline_cases_lookup = (baseline or {}).get("cases") if baseline else None
    cases_payload = sorted(
        (_case_payload(r, baseline_lookup=baseline_cases_lookup)
         for _, r in df_all.iterrows()),
        key=lambda c: _tid_key(c.get("id", "")),
    )
    # Run-level mean latency feeds the per-case latency-chip threshold:
    # green < 10s · yellow [10s, mean] · red > mean. Falls back to null
    # when latency columns are absent so the JS goes to the trivial fast path.
    if "lat_total_ms" in df.columns:
        mean_lat_ms = pd.to_numeric(df["lat_total_ms"], errors="coerce").dropna().mean()
        mean_lat_payload = float(mean_lat_ms) if pd.notna(mean_lat_ms) else None
    else:
        mean_lat_payload = None

    js = (JS.replace("__CASES__", json.dumps(cases_payload))
            .replace("__DIM_NAMES__", json.dumps(list(DIMENSION_WEIGHTS.keys())))
            .replace("__PASS_THRESHOLD__", json.dumps(PASS_THRESHOLD))
            .replace("__LANG_LABEL_UPPER__", json.dumps(LANG_LABEL_UPPER))
            .replace("__MEAN_LAT_MS__", json.dumps(mean_lat_payload if include_latency else None))
            .replace("__RETRY_MAX_TOK_PER_S__", json.dumps(RETRY_MAX_TOK_PER_S))
            .replace("__INCLUDE_LATENCY__", json.dumps(bool(include_latency))))


    # ── Filter chips (Test-Cases tab) ────────────────────────────────────────
    rc_values = sorted(df["root_cause_category"].dropna().unique().tolist()) if "root_cause_category" in df.columns else []
    rc_chips = "".join(
        f'<button class="chip" data-group="rc" data-value="{_h(v)}">{_h(v)}</button>'
        for v in rc_values
    )
    pass_fail_chips = (
        '<button class="chip" data-group="flag" data-value="pass">Pass</button>'
        '<button class="chip" data-group="flag" data-value="fail">Fail</button>'
    )
    flag_chips = (
        '<button class="chip" data-group="flag" data-value="kb_scope">KB-scope</button>'
        '<button class="chip" data-group="flag" data-value="dba_no_tools">DBA no tools</button>'
        '<button class="chip" data-group="flag" data-value="hg_invest_scope">HG-invest</button>'
        '<button class="chip" data-group="flag" data-value="hg_invest_no_tools">HG-invest no tools</button>'
        '<button class="chip" data-group="flag" data-value="rerank_empty">Rerank ∅</button>'
        '<button class="chip" data-group="flag" data-value="kb_gap">KB gap</button>'
        '<button class="chip" data-group="flag" data-value="oversel" title="Reranker selected more than 6 ENUMs — over-selection band on Retrieval Findings">ENUMs &gt; 6</button>'
        '<button class="chip chip-err" data-group="flag" data-value="span_errors" title="Cases with at least one span-level error (info.state=OK but a child span crashed — e.g. CancelledError, ConnectTimeout)">Span errors</button>'
    )
    # Latency range filter — gated on --include-latency since lat_total_ms
    # is only populated when the backfill ran. Inputs are in seconds for
    # readability; the JS multiplies by 1000 to match c.lat_total_ms.
    latency_range_html = (
        '<div class="filter-group">'
        '<span class="filter-label">Latency (s)</span>'
        '<div class="rel2-range">'
        '<input id="lat-min" type="number" min="0" step="0.5" placeholder="min">'
        '<span>–</span>'
        '<input id="lat-max" type="number" min="0" step="0.5" placeholder="max">'
        '<button class="range-reset" id="lat-reset">reset</button>'
        '</div>'
        '</div>'
    ) if include_latency else ""
    # Multi-select chips for failure mode. Matches against c.failure_modes_all
    # (every mode that fires for the case), so a row whose primary mode is
    # test_set_defect can still be selected when filtering for hallucination
    # if both conditions were true. "pass" is omitted — that distinction is
    # handled by the Pass | Fail row at the top of the sidebar.
    failure_mode_chips = "".join(
        f'<button class="chip" data-group="failure_modes" data-value="{_h(k)}">'
        f'{_h(FAILURE_MODE_LABEL[k])}</button>'
        for k in FAILURE_MODES if k != "pass"
    )
    # Per-scorer 0/1/2 matrix. Each chip writes a {dim, score} pair into
    # activeFilters.dim_pairs (the same filter the Scorer-heatmap click
    # already uses). Single-select per dim — the JS handler replaces the
    # existing entry for that dim if a different score is clicked. Wrapped
    # in <details> so the section is foldable; default collapsed since a
    # 7×3 matrix is a lot to show always.
    scorer_matrix_rows = ""
    for dim_key in DIMENSION_WEIGHTS:
        name = (dim_full_info.get(dim_key, {}).get("name") or dim_key).strip()
        chips = "".join(
            f'<button type="button" class="chip dim-chip" '
            f'data-dim="{_h(dim_key)}" data-score="{s}">{s}</button>'
            for s in (0, 1, 2)
        )
        scorer_matrix_rows += (
            f"<div class='dim-matrix-row'>"
            f"<span class='dim-matrix-name' title='{_h(dim_key)}'>{_h(name)}</span>"
            f"<span class='dim-matrix-chips'>{chips}</span>"
            f"</div>"
        )
    scorer_matrix_html = (
        "<div class='filter-group dim-matrix-group'>"
        "<details class='dim-matrix-details'>"
        "<summary class='dim-matrix-summary'>"
        "<span class='filter-label'>Judge scorers</span>"
        "<span class='dim-matrix-active-count'></span>"
        "</summary>"
        f"<div class='dim-matrix'>{scorer_matrix_rows}</div>"
        "</details>"
        "</div>"
    )

    # ── Rel2 mean for summary card ───────────────────────────────────────────
    rel2_mean_str = (
        f"{df['enum_relevance_score'].mean():.2f}"
        if "enum_relevance_score" in df.columns and df["enum_relevance_score"].notna().any()
        else "–"
    )

    # ── Rel2 mean vs CSAS benchmark — green tint when the run beats it ───────
    # Kept as a precomputed flag so the f-string below stays readable and
    # the comparison is done once, in one place.
    _rel2_mean_val = summary_metrics.get("rel2_mean")
    _rel2_beats_csas = (
        isinstance(_rel2_mean_val, (int, float))
        and not (isinstance(_rel2_mean_val, float) and np.isnan(_rel2_mean_val))
        and _rel2_mean_val > CSAS_BENCHMARK_REL2
    )
    rel2_mean_cls = " rel2-mean-good" if _rel2_beats_csas else ""
    rel2_mean_title = (
        f"Run mean ({_fmt_score(_rel2_mean_val)}) beats the CSAS benchmark "
        f"({CSAS_BENCHMARK_REL2:.3f})."
        if _rel2_beats_csas else
        f"Run mean vs CSAS benchmark ({CSAS_BENCHMARK_REL2:.3f}). "
        f"Green when the run mean strictly exceeds the benchmark."
    )

    # ── KB Recall vs CSAS benchmark recall ───────────────────────────────────
    # Same pattern as the Rel2 mean above: turn the Recall cell green
    # when the run's micro-averaged recall strictly exceeds the
    # external benchmark.
    _recall_val = summary_metrics.get("dataset_recall")
    _recall_beats_csas = (
        isinstance(_recall_val, (int, float))
        and not (isinstance(_recall_val, float) and np.isnan(_recall_val))
        and _recall_val > CSAS_BENCHMARK_RECALL
    )
    kb_recall_cls = " kb-recall-good" if _recall_beats_csas else ""
    kb_recall_title = (
        f"Run recall ({_fmt_pct(_recall_val)}) beats the CSAS benchmark "
        f"({CSAS_BENCHMARK_RECALL:.1%})."
        if _recall_beats_csas else
        f"Run recall vs CSAS benchmark ({CSAS_BENCHMARK_RECALL:.1%}). "
        f"Green when the run recall strictly exceeds the benchmark."
    )

    # ── Pass-rate (excl. test-set issues) vs internal target ─────────────────
    # Pre-computed flag so the f-string below stays readable. Met when the
    # clean pass rate is at or above the target.
    _pass_rate_clean_val = summary_metrics.get("pass_rate_clean")
    _pass_rate_meets_target = (
        isinstance(_pass_rate_clean_val, (int, float))
        and not (isinstance(_pass_rate_clean_val, float) and np.isnan(_pass_rate_clean_val))
        and _pass_rate_clean_val >= PASS_RATE_TARGET
    )

    # ── Inline Δ chips vs the --baseline run ─────────────────────────────────
    # Rendered under each headline number when a baseline CSV was passed.
    # When no baseline is loaded these expressions evaluate to {} / ""
    # and every _inline_delta_html() call below silently returns "".
    _baseline_metrics = (baseline or {}).get("metrics", {})
    _baseline_label_str = (baseline or {}).get("label", "")
    _baseline_title_prefix = (
        f"vs baseline {_baseline_label_str}".rstrip()
        if _baseline_label_str else "vs baseline"
    )

    # ── Headline-card tooltip strings ────────────────────────────────────────
    # Precomputed because Python 3.9 f-string expressions can't contain
    # backslashes (so we can't write "\n\n" inline inside the f"{ ... }"
    # blocks below). Each string starts with the metric definition and
    # ends with the run-specific footnote that used to render as a
    # separate "hc-detail" footer line under the card.
    _NL = "\n\n"
    pass_rate_tooltip = (
        f"Share of cases the judge passed (weighted_avg ≥ "
        f"{PASS_THRESHOLD} AND no weight-2 scorer at 0), restricted to "
        f"cases with trustworthy ground truth — i.e. dropping rows the "
        f"judge flagged with expected_reference_looks_wrong=True, rows "
        f"classified as ambiguous/out_of_scope, and rows the "
        f"deterministic check flagged as ENUM naming mismatches. "
        f"Matches the per-case PASS/FAIL badge and the Pass | Fail chip "
        f"filter on the Test Cases tab. Compared against an internal "
        f"target of {PASS_RATE_TARGET:.0%}; the run value turns green "
        f"when it meets or exceeds the target."
        + _NL
        + f"This run: {summary_metrics['n_pass_clean']} / "
          f"{summary_metrics['n_clean']} cases · "
          f"{summary_metrics['n_defect']} test-set-issue rows excluded."
    )
    test_cases_tooltip = (
        "Total cases in the source checkpoint. The non-empty count "
        "(rows the report actually scores against) and its empty-query "
        "exclusion details live in the Methodology section at the "
        "bottom of this tab."
        + _NL
        + "Source checkpoint · see Methodology for exclusions."
    )
    kb_recall_tooltip = (
        "Micro-averaged recall (Σ TP / Σ expected) over the whole run, "
        "where TP is the count of expected ENUMs that appeared in the "
        "reranker's final selection. Restricted to cases where the "
        "search tool was used (query_scope == 'kb') AND the "
        "deterministic check did NOT flag an ENUM-naming-mismatch "
        "(those are test-set issues, not agent failures). The Stage "
        "funnel card in the row below uses the same basis at each "
        "pipeline stage; its reranked-stage recall equals this "
        "headline number."
        + _NL
        + f"This run: {summary_metrics['dataset_recall_tp']} true "
          f"positives · denominator = "
          f"{summary_metrics['dataset_recall_total_expected']} expected "
          f"ENUMs. CSAS benchmark recall = "
          f"{CSAS_BENCHMARK_RECALL:.1%} (external baseline; the Recall "
          f"cell turns green when the run value strictly exceeds it)."
    )
    _rel2_excluded_note = (
        f" Excluded {summary_metrics['rel2_naming_excluded']} "
        f"ENUM-naming-mismatch "
        f"case{'s' if summary_metrics['rel2_naming_excluded'] != 1 else ''}."
        if summary_metrics['rel2_naming_excluded'] else ""
    )
    rel2_tooltip = (
        "Upstream semantic-overlap metric between expected_enums and "
        "the system's selected ENUM IDs over cases where the search "
        "tool was actually used (query_scope == 'kb'). Cases the "
        "deterministic check flagged as ENUM-naming-mismatch are "
        "excluded — those are test-set issues (gold ENUM IDs use a "
        "different naming convention than the KB), not agent failures, "
        "so they shouldn't drag down the score. Range [0, 1]; higher "
        "is better. See the Doc tab for the exact computation."
        + _NL
        + f"This run: n = {summary_metrics['rel2_n']} of "
          f"{summary_metrics['n_eval']} cases "
          f"(the rest didn't reach the reranker).{_rel2_excluded_note}"
    )
    pass_rate_delta_chip = _inline_delta_html(
        _baseline_metrics.get("pass_rate_clean"),
        summary_metrics.get("pass_rate_clean"),
        fmt="pct", good_dir="up",
        title_prefix=_baseline_title_prefix,
    )
    recall_delta_chip = _inline_delta_html(
        _baseline_metrics.get("dataset_recall"),
        summary_metrics.get("dataset_recall"),
        fmt="pct", good_dir="up",
        title_prefix=_baseline_title_prefix,
    )
    rel2_mean_delta_chip = _inline_delta_html(
        _baseline_metrics.get("rel2_mean"),
        summary_metrics.get("rel2_mean"),
        fmt="score", good_dir="up",
        title_prefix=_baseline_title_prefix,
    )

    # ── Comparison section (Summary tab) ─────────────────────────────────────
    corr_cards = ""
    if corr["expert"] is not None:
        corr_cards += f"""
        <div class="card">
          <div class="card-title">Expert score (1–10) × Judge weighted_avg  —  agreement matrix</div>
          {_plot(corr["expert"])}
          <p class="corr-stat">n = {corr["n_expert"]} &middot; Pearson r = {corr["expert_pearson"]:.3f}</p>
        </div>"""
    if corr["rel2"] is not None:
        corr_cards += f"""
        <div class="card">
          <div class="card-title">Rel2 score (0–1) × Judge enum_F1  —  ENUM-selection agreement</div>
          {_plot(corr["rel2"])}
          <p class="corr-stat">n = {corr["n_rel2"]} &middot; Pearson r = {corr["rel2_pearson"]:.3f}</p>
        </div>"""
    kb_versions = (
        sorted({v for v in df["kb_version"].dropna().astype(str) if v.strip()})
        if "kb_version" in df.columns else []
    )
    if not kb_versions:
        kb_version_label = "–"
    elif len(kb_versions) == 1:
        kb_version_label = kb_versions[0]
    else:
        kb_version_label = f"{len(kb_versions)} versions (mixed)"

    comparison_block = (
        f'<div class="card"><div class="card-title">Comparison: pre-existing grades vs this eval</div>'
        f'<p style="font-size:12px;color:#5c7999;margin-bottom:10px">'
        f'Two contingency matrices. <strong>Left</strong>: answer-level — how the expert\'s 1–10 score '
        f'(human) lines up with the judge\'s weighted_avg (this eval). Diagonal = agreement; off-diagonal = disagreement to investigate. '
        f'<strong>Right</strong>: ENUM-selection — how the pre-existing Rel2 score (semantic overlap) compares with the judge\'s set-based enum_F1 '
        f'(retrieved vs expected). Cell labels = row counts.</p>'
        f'<div class="corr-grid">{corr_cards}</div></div>'
    ) if corr_cards else ""

    # Build the Latency tab panel only when the surface is enabled. Kept
    # as a variable rather than an inline expression because the panel
    # is multi-card and easier to read out-of-band.
    if include_latency:
        latency_tab_panel = f"""
  <div id="tab-latency" class="tab-panel">
    <div class="card">
      <div class="card-title">Latency distribution {_info_icon(
          "Histogram of per-trace total wall-clock latency across all "
          "parsed traces. Vertical lines: solid blue = mean, dashed blue = "
          "95% bootstrap CI on the mean, dotted grey = p50 (median), dotted "
          "red = p95 (the tail experience that dominates user complaints). "
          "X-axis is clipped at p99 so extreme outliers do not squash the "
          "rest of the distribution.")}
      </div>
      {_plot(fig_lat_dist, div_id="latency-distribution") if fig_lat_dist is not None else "<p class='placeholder'>no latency data on this checkpoint</p>"}
    </div>

    <div class="card">
      <div class="card-title">Generation latency vs # selected ENUMs {_info_icon(
          "Scatter of the reranker's final ENUM count (x) against the "
          "generation-LLM step latency (y). Filtered to cases where the "
          "KB pipeline actually ran (post_prune_enum_count &gt; 0) — "
          "chit-chat cases would otherwise pile up at x=0 with their "
          "latency driven by routing rather than context size, biasing "
          "the correlation toward 0. "
          "Directly tests the hypothesis 'more selected ENUMs → bigger "
          "generation context → longer generation time'. "
          "Retry-flagged cases are drawn in red so any clustering at "
          "high enum counts (which would support the TPM-throttling "
          "side of the hypothesis) is visible. "
          "The diamond line is the mean per enum-count bucket (bins "
          "with fewer than 3 cases omitted). Pearson r in the x-axis "
          "label is the linear correlation. "
          "Click any point to open that case in the Test Cases tab.")}
      </div>
      {_plot(fig_lat_vs_enums, div_id="latency-vs-enums") if fig_lat_vs_enums is not None else "<p class='placeholder'>need lat_generation_llm_ms, reranked_enum_ids, and post_prune_enum_count columns</p>"}
    </div>

    {_latency_summary_card_html(df)}
  </div>"""
    else:
        latency_tab_panel = ""

    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>{LANG_LABEL_UPPER}KB – {_h(yaml_name)}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{CSS}</style></head>
<body>

<div class="sticky-top">
<header class="report-header">
  <div class="header-inner">
    <h1>{LANG_LABEL_UPPER}KB Evaluation Report</h1>
    <div class="header-meta header-meta-grid">
      <div class="header-col">
        <span>Run time: <strong>{_h(mlflow_run_timestamp) if mlflow_run_timestamp else '—'}</strong></span>
        {f'<span>MLflow experiment: <code>{_h(mlflow_experiment_id)}</code></span>' if mlflow_experiment_id else '<span>MLflow experiment: <code>—</code></span>'}
        {f'<span>MLflow run: <code>{_h(mlflow_run_id)}</code></span>' if mlflow_run_id else '<span>MLflow run: <code>—</code></span>'}
      </div>
      <div class="header-col">
        <span>KB version: <code>{_h(kb_version_label)}</code></span>
        <span>Experiment: <strong>{_h(experiment_name)}</strong></span>
        <span>Judge model: <strong>{_h(judge_model)}</strong> <span class="header-aux">({_h(reasoning_effort)})</span></span>
      </div>
      <div class="header-footer">
        <span>Checkpoint: <code>{_h(checkpoint_label)}</code></span>
      </div>
    </div>
  </div>
</header>

<nav class="tab-nav">
  <div class="tab-nav-inner">
    <button class="tab-btn tab-home active" data-target="tab-summary">Summary</button>
    <button class="tab-btn" data-target="tab-cases">Test Cases</button>
    <button class="tab-btn" data-target="tab-issues">Issues</button>
    <button class="tab-btn" data-target="tab-prompts">Prompts</button>
    {'<button class="tab-btn" data-target="tab-latency">Latency</button>' if include_latency else ''}
    <button class="tab-btn" data-target="tab-dims">Scorers</button>
    <button class="tab-btn" data-target="tab-missed">Retrieval Findings</button>
    <button class="tab-btn" data-target="tab-kb-findings">KB Findings</button>
    <span class="tab-tools">
      <button class="tab-btn tab-kb-highlight" data-target="tab-kb">KB</button>
      <button class="tab-btn" data-target="tab-doc">Notes</button>
    </span>
    <span class="tab-personal" title="Personal / debugging tab — judge-output sanity checks">
      <button class="tab-btn tab-personal-btn" data-target="tab-judge">Judge Eval</button>
    </span>
  </div>
</nav>
</div>

<main class="content">

  <div id="tab-summary" class="tab-panel active">
    {prompt_warning_card}
    <div class="headline-row headline-row-3">
      <div class="headline-card pass-rate-card">
        <div class="hc-label">Pass rate · excl. test-set issues {_info_icon(pass_rate_tooltip)}
        </div>
        <div class="hc-stats-row">
          <div class="hc-stat">
            <div class="hc-stat-label">Run rate</div>
            <div class="hc-stat-value{(' pass-rate-good' if _pass_rate_meets_target else '')}">{_fmt_pct(summary_metrics['pass_rate_clean'])}</div>
            {pass_rate_delta_chip}
          </div>
          <div class="hc-stat" title="Internal target the agent is benchmarked against. The Run rate turns green when it meets or exceeds this value.">
            <div class="hc-stat-label">Target</div>
            <div class="hc-stat-value hc-stat-reference">{PASS_RATE_TARGET:.0%}</div>
          </div>
        </div>
      </div>
      <div class="headline-card kb-recall-card">
        <div class="hc-label">KB Coverage (Recall) {_info_icon(kb_recall_tooltip)}
        </div>
        <div class="hc-stats-row">
          <div class="hc-stat" title="{kb_recall_title}">
            <div class="hc-stat-label">Recall</div>
            <div class="hc-stat-value{kb_recall_cls}">{_fmt_pct(summary_metrics['dataset_recall'])}</div>
            {recall_delta_chip}
          </div>
          <div class="hc-stat" title="External baseline recall (Σ TP / Σ expected) from the CSAS benchmark, kept here for at-a-glance comparison with the run's recall.">
            <div class="hc-stat-label">CSAS benchmark</div>
            <div class="hc-stat-value hc-stat-reference">{CSAS_BENCHMARK_RECALL:.1%}</div>
          </div>
        </div>
      </div>
      <div class="headline-card rel2-stats-card">
        <div class="hc-label">Rel2 score (search tool used) {_info_icon(rel2_tooltip)}
        </div>
        <div class="hc-stats-row">
          <div class="hc-stat" title="{rel2_mean_title}">
            <div class="hc-stat-label">Mean</div>
            <div class="hc-stat-value{rel2_mean_cls}">{_fmt_score(summary_metrics['rel2_mean'])}</div>
            {rel2_mean_delta_chip}
          </div>
          <div class="hc-stat" title="External baseline rel2 score from the CSAS benchmark, kept here for at-a-glance comparison with the run's mean.">
            <div class="hc-stat-label">CSAS benchmark</div>
            <div class="hc-stat-value hc-stat-reference">{CSAS_BENCHMARK_REL2:.3f}</div>
          </div>
        </div>
      </div>
    </div>

    <div class="headline-row {('headline-row-1-1-3' if include_latency else 'headline-row-1-3')}">
      <div class="headline-card">
        <div class="hc-label">Test cases {_info_icon(test_cases_tooltip)}
        </div>
        <div class="hc-value">{summary_metrics['n_total']}</div>
      </div>
      {_latency_headline_card_html(df) if include_latency else ""}
      <div class="headline-card kb-recall-card">
        <div class="hc-label">Stage funnel — recall &amp; precision {_info_icon(
            "Micro-averaged recall (Σ TP / Σ expected) and precision (Σ TP / "
            "Σ selected) of expected_enums at each pipeline stage. Same basis "
            "as the KB Recall · Precision card above: query_scope == 'kb' AND "
            "no ENUM-naming mismatch. Recall trends down through the pipeline "
            "because the gold ENUM set can only shrink as candidates are "
            "dropped; precision trends up because pre-prune precision is "
            "naturally low — the vector DB returns many candidates by design "
            "— and the later stages drop noise. The reranked row equals the "
            "KB Recall and KB Precision headlines above.")}
        </div>
        {_funnel_html(summary_metrics['funnel'])}
      </div>
    </div>

    {_top_failures_html(summary_metrics['top_failures_clean'], summary_metrics['n_fail_clean'], baseline=baseline)}

    <div class="card foldable collapsed">
      <div class="card-title">Methodology {_info_icon(
          "How the source checkpoint is filtered before any metric is "
          "computed, and which subset each headline number sits on. "
          "Open this to see the exact denominators behind the cards "
          "above.")}
      </div>
      <div class="card-body">
        <p style="font-size:12px;color:#5c7999;margin-bottom:10px">
          Stage-1 filtering, sample sizes, and the denominators each
          headline metric sits on.
        </p>
        <table class="tbl funnel-tbl" style="max-width:640px">
          <thead><tr>
            <th>Step</th>
            <th style="text-align:right">N</th>
            <th>Notes</th>
          </tr></thead>
          <tbody>
            <tr>
              <td><strong>Source checkpoint</strong></td>
              <td style="text-align:right;font-variant-numeric:tabular-nums"><strong>{summary_metrics['n_total']}</strong></td>
              <td>All rows the judge processed (matches the <em>Test cases</em> card above).</td>
            </tr>
            <tr>
              <td><span class='fm-indent'>↳</span> Excluded · empty <code>user_query</code></td>
              <td style="text-align:right;font-variant-numeric:tabular-nums;color:#5c7999">{summary_metrics['n_empty_query']}</td>
              <td>Judge had nothing to score; dropped from every metric.</td>
            </tr>
            <tr class="fm-total-row">
              <td><strong>Non-empty</strong> (= <code>n_eval</code>)</td>
              <td style="text-align:right;font-variant-numeric:tabular-nums"><strong>{summary_metrics['n_total'] - summary_metrics['n_empty_query']}</strong></td>
              <td>Denominator for the Pass-rate · all cases card and the Issues table.</td>
            </tr>
            <tr>
              <td><span class='fm-indent'>↳</span> Excluded · test-set issues (gold-defect, naming mismatch, ambiguous, out_of_scope)</td>
              <td style="text-align:right;font-variant-numeric:tabular-nums;color:#5c7999">{summary_metrics['n_defect']}</td>
              <td>Untrustworthy ground truth; dropped from the agent-side denominator.</td>
            </tr>
            <tr class="fm-total-row">
              <td><strong>Clean</strong> (= <code>n_clean</code>)</td>
              <td style="text-align:right;font-variant-numeric:tabular-nums"><strong>{summary_metrics['n_clean']}</strong></td>
              <td>Denominator for the Pass-rate · excl. test-set issues card and the Top issues card.</td>
            </tr>
          </tbody>
        </table>
        <p style="font-size:11px;color:#5c7999;margin-top:10px">
          KB Recall, KB Precision and the Stage funnel apply a tighter
          internal filter (<code>query_scope == 'kb'</code> AND
          <code>expected_enums</code> non-empty AND no naming mismatch)
          so that only rows the reranker actually ran on contribute. See
          the <em>Notes</em> tab for the full per-metric definitions.
        </p>
      </div>
    </div>

  </div>

  <div id="tab-cases" class="tab-panel">
    <div class="cases-layout">
      <button id="sidebar-toggle" class="sidebar-toggle" type="button"
              title="Hide sidebar" aria-label="Toggle sidebar">‹</button>
      <div class="peek-zone" aria-hidden="true"></div>
      <aside class="cases-sidebar">
        <input id="case-search" class="case-search" type="text" placeholder="Search id or query…">
        <div class="filters-wrap">
          <div class="filter-group">
            <span class="filter-label">Pass | Fail</span>
            {pass_fail_chips}
            <button class="chip chip-clear" id="chip-clear">Clear</button>
          </div>
          {scorer_matrix_html}
          <div class="filter-group foldable-filter">
            <details class="dim-matrix-details">
              <summary class="dim-matrix-summary">
                <span class="filter-label">Issue</span>
              </summary>
              <div class="foldable-filter-body">{failure_mode_chips}</div>
            </details>
          </div>
          <div class="filter-group foldable-filter">
            <details class="dim-matrix-details">
              <summary class="dim-matrix-summary">
                <span class="filter-label">Flags</span>
              </summary>
              <div class="foldable-filter-body">{flag_chips}</div>
            </details>
          </div>
          <div class="filter-group">
            <span class="filter-label">Rel2</span>
            <div class="rel2-range">
              <input id="rel2-min" type="number" min="0" max="1" step="0.05" placeholder="min">
              <span>–</span>
              <input id="rel2-max" type="number" min="0" max="1" step="0.05" placeholder="max">
              <button class="range-reset" id="rel2-reset">reset</button>
            </div>
          </div>
          <div class="filter-group">
            <span class="filter-label">Judge w.avg</span>
            <div class="rel2-range">
              <input id="wavg-min" type="number" min="0" max="1" step="0.05" placeholder="min">
              <span>–</span>
              <input id="wavg-max" type="number" min="0" max="1" step="0.05" placeholder="max">
              <button class="range-reset" id="wavg-reset">reset</button>
            </div>
          </div>
          {latency_range_html}
        </div>
        <div id="active-filter" class="active-filter-banner">
          <span id="active-filter-text"></span>
          <button id="active-filter-clear">clear</button>
        </div>
        <div id="list-count" class="list-count"></div>
        <div class="cases-list-wrap">
          <ul id="cases-list" class="cases-list"></ul>
        </div>
      </aside>
      <article id="case-detail" class="case-detail">
        <div class="placeholder">Select a test case from the list.</div>
      </article>
    </div>
  </div>

  <div id="tab-issues" class="tab-panel">
    <div class="card">
      <div class="card-title">Issues — primary cause per case {_info_icon(
          "Every test case is assigned exactly one primary issue. Click any count to filter the "
          "Test Cases tab to those rows. See the Doc tab for the full taxonomy.")}
      </div>
      <p style="font-size:12px;color:#5c7999;margin-bottom:10px">
        Priority order: test-set issue → wrong agent routing → retrieval / pruning / reranker losses →
        pool content gap → context-use / hallucination / language drift → clean pass / other.
        Multiple secondary issues may also be present per case — drill into individual cases for the full picture.
      </p>
      {_failure_mode_table_html(summary_metrics)}
    </div>

    <div class="card foldable collapsed">
      <div class="card-title">Issue co-occurrence {_info_icon(
          "Symmetric matrix of how often each pair of issues fires "
          "for the same case (using failure_modes_all — every issue that "
          "applied to a row, not just the priority winner). Diagonal cells = "
          "total cases that issue fired in. Off-diagonal cells = cases where "
          "both issues fired together. Click any non-zero cell to filter the "
          "Test Cases tab to those rows. Issues from the Issues table "
          "are shown starting at 'Test-set issue' (Clean pass is omitted).")}
      </div>
      <div class="card-body">
        <p style="font-size:12px;color:#5c7999;margin-bottom:10px">
          This matrix uses <code>failure_modes_all</code> — every applicable
          issue per case, not just the priority winner shown in the
          Issues table. So a case primarily classified as
          <em>test-set issue</em> that also triggered <em>hallucination</em>
          appears in both diagonal cells and in the off-diagonal pair.
          Use it to spot "clusters" — e.g. if Test-set issue occurs across
          the whole row, the test set is producing noise that's confounding
          several other issues.
        </p>
        {_plot(fig_fm_cooccurrence, div_id="fm-cooccurrence-chart") if fig_fm_cooccurrence is not None else "<p class='placeholder'>not enough data to compute co-occurrence</p>"}
      </div>
    </div>
  </div>

  <div id="tab-prompts" class="tab-panel">
    {prompts_tab_html}
  </div>

  {latency_tab_panel}

  <div id="tab-dims" class="tab-panel">
    <div class="card">
      <div class="card-title">Judge scorers {_info_icon(
          "The seven judge dimensions (scorers). "
          "Each card shows a stacked bar (red = fail, amber = partial, green = pass). "
          "Click-through links to drill into the cases that scored at each level.")}
      </div>
      {_dimension_cards_html(summary_metrics['dimensions'], dim_full_info)}
    </div>
    <div class="card">
      <div class="card-title">Per-scorer distribution</div>
      <p style="font-size:12px;color:#537090;margin-bottom:10px">
        Each test case is graded by the LLM judge on every dimension on a <strong>0 / 1 / 2</strong>
        scale (<span style="color:#cf2a1e">0 = fail</span>, <span style="color:#f2a91e">1 = partial</span>,
        <span style="color:#057f19">2 = pass</span>). Click any count to see those test cases.
      </p>
      <table class="tbl">
        <thead><tr><th>Dimension</th><th>Mean</th><th>0 count</th><th>1 count</th><th>2 count</th><th>0 rate</th><th>1 rate</th><th>2 rate</th></tr></thead>
        <tbody>{dim_rows_html}</tbody>
      </table>
    </div>
    <div class="card">
      <div class="card-title">Scorer heatmap</div>
      <p style="font-size:12px;color:#537090;margin-bottom:10px">
        Counts of test cases per dimension at each judge score (0 / 1 / 2).
        <strong>Click any cell</strong> to jump to the Test Cases tab filtered to those rows.
      </p>
      {_plot(fig_dim_heatmap, div_id="dim-heatmap-chart") if fig_dim_heatmap else "<p class='placeholder'>no dimension scores</p>"}
    </div>
  </div>

  <div id="tab-missed" class="tab-panel">
    <div class="summary-grid-2">
      <div class="card">
        <div class="card-title">Stage funnel — recall &amp; precision {_info_icon(
            "Micro-averaged recall (Σ TP / Σ expected) and precision (Σ TP / Σ selected) "
            "of the expected ENUMs at each pipeline stage: "
            "pre-prune (vector DB output) → post-prune (after dedup/filtering) → reranked (final selection). "
            "Recall trends down because the gold ENUM set can only shrink as candidates are dropped; "
            "precision trends up because pre-prune precision is naturally low — the vector DB returns "
            "many candidates by design — and the later stages drop that noise. "
            "The reranked row equals the KB Recall and KB Precision headlines on the Summary page. "
            "Restricted to cases where the search tool was used (query_scope == 'kb') AND the "
            "deterministic check did NOT flag an ENUM-naming-mismatch (those are test-set issues).")}
        </div>
        {_funnel_html(summary_metrics['funnel'])}
      </div>
      <div class="card">
        <div class="card-title">Rel2 distribution {_info_icon(
            "Bucketed distribution of enum_relevance_score over the cases "
            "where the reranker ran. Mean alone hides whether cases cluster "
            "tight or split into great/terrible halves; bucket counts make "
            "that shape visible. Click any count to drill into those cases.")}
        </div>
        {_rel2_distribution_html(summary_metrics)}
      </div>
    </div>
    <div class="summary-grid-2">
      <div class="card">
        <div class="card-title">Top missed ENUMs — reranker miss (in pool, not picked) vs retriever miss (not in pool)</div>
        <p style="font-size:11px;color:#537090;margin-bottom:10px">
          Click a bar to jump to the Test Cases tab filtered to the rows that missed that ENUM.
        </p>
        {_plot(fig_missed, div_id="missed-enums-chart") if fig_missed else "<p class='placeholder'>no missed-enum data</p>"}
      </div>
      <div class="card">
        <div class="card-title">ENUMs per case — expected vs selected {_info_icon(
            "Per-case distribution of |expected_enums| vs |reranked_enum_ids|. "
            "Each row is an ENUM-count value, with the number of cases that had "
            "that many expected ENUMs and that many selected ENUMs (and the % "
            "share of n on each side). Use it to spot systemic over- or "
            "under-selection: if the Selected column is heavier on higher counts "
            "than the Expected column, the reranker is over-picking. Restricted "
            "to cases where the reranker actually ran (query_scope == 'kb').")}
        </div>
        {enum_count_dist_html}
      </div>
    </div>
  </div>

  <div id="tab-kb-findings" class="tab-panel">
    <div class="kb-findings-row1">
      {_empty_user_queries_card_html(df_all)}
      {_naming_mismatches_card_html(naming_mismatches_agg)}
    </div>
    <div class="card">
      <div class="card-title">Test-case issues — review queue {_info_icon(
          "Every case the report flagged for KB / dataset-side review: "
          "test-set issues (judge said gold reference needs review), ENUM "
          "naming mismatches, pool content gaps, and any case the judge "
          "left a kb_improvement_suggestion or test_case_improvement_suggestion "
          "for. Filterable per column. Click the test_case_id to drill into "
          "the case detail.")}
      </div>
      {_kb_findings_table_html(df)}
    </div>
  </div>

  <div id="tab-judge" class="tab-panel">
    {judge_eval_html}
  </div>

  <div id="tab-kb" class="tab-panel" data-lang="en">
    {kb_html}
  </div>

  <div id="tab-doc" class="tab-panel">
    {_doc_tab_html(summary_metrics, wavg_hist_fig=fig_wavg_hist, include_latency=include_latency)}
  </div>

</main>

<script>{js}</script>
</body></html>
"""


# ── CLI ──────────────────────────────────────────────────────────────────────
# Override for the YAML config directory. Set by ``--config-dir`` in main();
# when None, the resolver below searches a list of known layouts so the
# script works for CZKB (configs at experiments/czkb/configs/) and SKKB
# (configs at experiments/skkb/configs/skkb/) without further arguments.
_CONFIG_DIR_OVERRIDE: Path | None = None


def _candidate_config_dirs() -> list[Path]:
    """Ordered list of directories the YAML resolver searches.

    Built dynamically per call so that ``_CONFIG_DIR_OVERRIDE`` (set by
    ``main()``) wins. After the override comes a per-lang shortlist, then
    a broader recursive sweep under the repo for safety. The first hit
    is returned by ``_resolve_yaml_path``.
    """
    here = Path(__file__).resolve().parent
    candidates: list[Path] = []
    if _CONFIG_DIR_OVERRIDE is not None:
        candidates.append(_CONFIG_DIR_OVERRIDE)
    # Per-lang well-known layouts. Includes both the CZKB layout
    # (experiments/czkb/configs/) and the SKKB one (experiments/skkb/configs/skkb/).
    for lang in (LANG, *(c for c in BENCHMARKS if c != LANG)):
        candidates.append(here / f"{lang}kb" / "configs")
        candidates.append(here / f"{lang}kb" / "configs" / f"{lang}kb")
    # Legacy fallbacks that worked in the older single-lang scripts.
    candidates.append(here / "configs")
    candidates.append(here.parents[0] / "configs")
    return candidates


def _resolve_yaml_path(yaml_name: str) -> Path | None:
    """Return the first existing ``<dir>/<yaml_name>.yaml`` in the
    configured search list, or None when nothing matches."""
    for d in _candidate_config_dirs():
        cfg = d / f"{yaml_name}.yaml"
        if cfg.exists():
            return cfg
    return None


def _load_dimension_descriptions(yaml_name: str) -> dict[str, str]:
    """Extract each dimension's `description: >` block verbatim from the YAML.

    Avoids a pyyaml dependency; parses by tracking indent of the `description:` key
    and folding the subsequent more-indented lines (YAML's `>` style).
    """
    cfg = _resolve_yaml_path(yaml_name)
    if cfg is None:
        return {}
    try:
        lines = cfg.read_text().splitlines()
    except OSError:
        return {}
    descs: dict[str, str] = {}
    current_id: str | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        m_id = re.match(r'^(\s*)-\s*id:\s*([A-Za-z_][\w]*)\s*$', line)
        if m_id:
            current_id = m_id.group(2)
            i += 1
            continue
        m_desc = re.match(r'^(\s*)description:\s*>\s*$', line)
        if m_desc and current_id and current_id not in descs:
            indent = len(m_desc.group(1))
            i += 1
            buf: list[str] = []
            while i < len(lines):
                nxt = lines[i]
                stripped = nxt.lstrip()
                nxt_indent = len(nxt) - len(stripped)
                if stripped == "" or nxt_indent > indent:
                    buf.append(stripped)
                    i += 1
                else:
                    break
            # fold (YAML '>' folds newlines to spaces, blank lines become \n).
            folded_parts: list[str] = []
            cur = ""
            for s in buf:
                if s == "":
                    if cur:
                        folded_parts.append(cur.strip())
                        cur = ""
                else:
                    cur = f"{cur} {s}" if cur else s
            if cur:
                folded_parts.append(cur.strip())
            descs[current_id] = "\n".join(folded_parts)
            continue
        i += 1
    return descs


def _load_dimension_full_info(yaml_name: str) -> dict[str, dict]:
    """Parse each dimension's full block from configs/{yaml_name}.yaml.

    Returns a dict keyed by dimension id, where each value carries:
        name, weight, description, scale (list of {score, label, description}).
    Used to populate the per-scorer info-icon tooltips on the Summary tab
    with the verbatim YAML content.
    """
    cfg = _resolve_yaml_path(yaml_name)
    if cfg is None:
        return {}
    try:
        text = cfg.read_text()
    except OSError:
        return {}
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    rubric = parsed.get("rubric") or {}
    dims = rubric.get("dimensions") or []
    out: dict[str, dict] = {}
    for d in dims:
        if not isinstance(d, dict) or "id" not in d:
            continue
        scale_raw = d.get("scale") or []
        scale = []
        if isinstance(scale_raw, list):
            for s in scale_raw:
                if not isinstance(s, dict):
                    continue
                scale.append({
                    "score":       s.get("score"),
                    "label":       s.get("label", "") or "",
                    "description": (s.get("description") or "").strip(),
                })
        out[d["id"]] = {
            "name":        d.get("name", "") or "",
            "weight":      d.get("weight"),
            "description": (d.get("description") or "").strip(),
            "scale":       scale,
        }
    return out


def _format_dimension_yaml_tip(info: dict) -> str:
    """Format a dimension's YAML info as a multi-line tooltip string."""
    if not info:
        return ""
    weight = info.get("weight")
    weight_str = f"{weight:g}" if isinstance(weight, (int, float)) else str(weight or "")
    parts = [
        f"name: {info.get('name', '')}",
        f"weight: {weight_str}",
        f"description: {info.get('description', '')}",
        "",
        "score | label | description",
    ]
    for s in info.get("scale", []) or []:
        score = s.get("score")
        score_str = str(score) if score is not None else ""
        parts.append(f"{score_str} | {s.get('label', '')} | {s.get('description', '')}")
    return "\n".join(parts)


def _detect_experiment_meta(yaml_name: str) -> dict:
    """Pull the header-display fields from configs/{yaml_name}.yaml.

    Returns ``{"name", "judge_model", "reasoning_effort", "mlflow_run_id",
    "mlflow_experiment_id", "mlflow_run_timestamp"}``. The yaml has
    several ``name:`` keys (input_fields / dimensions / etc.); we
    therefore extract from the *top-level* ``experiment:``, ``model:``,
    and ``dataset:`` blocks only, so we don't accidentally pick up an
    unrelated nested name. Avoids a pyyaml dependency for a small lookup.
    """
    out = {"name": yaml_name, "judge_model": "unknown", "reasoning_effort": "unknown",
           "mlflow_run_id": "", "mlflow_experiment_id": "",
           "mlflow_run_timestamp": ""}
    cfg = _resolve_yaml_path(yaml_name)
    if cfg is None:
        return out
    try:
        text = cfg.read_text()
    except OSError:
        return out

    # A top-level block runs from `^block:` until the next non-indented,
    # non-comment, non-blank line. Capturing only that slice keeps the
    # later regexes restricted to the right scope.
    def _block(name: str) -> str:
        m = re.search(
            rf"^{re.escape(name)}\s*:\s*\n((?:[ \t]+.*\n|[ \t]*\n|[ \t]*#.*\n)*)",
            text, re.MULTILINE,
        )
        return m.group(1) if m else ""

    exp_block = _block("experiment")
    if exp_block:
        m = re.search(r'^\s+name\s*:\s*"?([^"\n#]+?)"?\s*(?:#.*)?$',
                       exp_block, re.MULTILINE)
        if m:
            out["name"] = m.group(1).strip()

    model_block = _block("model")
    if model_block:
        m = re.search(r'^\s+model_deployment_name\s*:\s*"?([^"\n#]+?)"?\s*(?:#.*)?$',
                       model_block, re.MULTILINE)
        if m:
            out["judge_model"] = m.group(1).strip()
        m = re.search(r'^\s+reasoning_effort\s*:\s*"?([^"\n#]+?)"?\s*(?:#.*)?$',
                       model_block, re.MULTILINE)
        if m:
            out["reasoning_effort"] = m.group(1).strip()

    dataset_block = _block("dataset")
    if dataset_block:
        m = re.search(r'^\s+mlflow_run_id\s*:\s*"?([^"\n#]+?)"?\s*(?:#.*)?$',
                       dataset_block, re.MULTILINE)
        if m:
            out["mlflow_run_id"] = m.group(1).strip()
        m = re.search(r'^\s+mlflow_experiment_id\s*:\s*"?([^"\n#]+?)"?\s*(?:#.*)?$',
                       dataset_block, re.MULTILINE)
        if m:
            out["mlflow_experiment_id"] = m.group(1).strip()
        m = re.search(r'^\s+mlflow_run_timestamp\s*:\s*"?([^"\n#]+?)"?\s*(?:#.*)?$',
                       dataset_block, re.MULTILINE)
        if m:
            out["mlflow_run_timestamp"] = m.group(1).strip()
    return out


def _auto_yaml_name(checkpoint_path: Path) -> str | None:
    """Pick the configs/<name>.yaml whose stem is the longest prefix of
    the checkpoint stem (after stripping the ``evals_`` prefix). Lets users
    skip ``--yaml-name`` when the checkpoint follows the ``evals_<name>_*``
    convention used by ``hg_ds_evals``. Searches every candidate config
    directory (per-lang well-known layouts + ``--config-dir`` override).
    """
    seen_stems: set[str] = set()
    candidates: list[str] = []
    for cfg_dir in _candidate_config_dirs():
        if not cfg_dir.exists():
            continue
        for p in cfg_dir.glob("*.yaml"):
            if p.stem not in seen_stems:
                seen_stems.add(p.stem)
                candidates.append(p.stem)
    if not candidates:
        return None
    candidates.sort(key=len, reverse=True)
    stem = checkpoint_path.stem
    if stem.startswith("evals_"):
        stem = stem[len("evals_"):]
    for name in candidates:
        if stem.startswith(name):
            return name
    return None


def _default_paths(yaml_name: str, reasoning_effort: str, suffix: str):
    here = Path(__file__).resolve().parent
    ckpt = here / "checkpoints" / f"evals_{yaml_name}_{reasoning_effort}{suffix}.csv"
    out = here.parents[1] / "reports" / yaml_name / f"report_clickable{suffix}.html"
    return ckpt, out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lang", choices=sorted(BENCHMARKS.keys()), default="cz",
                    help="Language of the KB run. Drives visible labels "
                         "(language-toggle button text, the "
                         "language_compliance dimension prompt) and the "
                         "default config-dir lookup. Internal CSS/JS slot "
                         "names stay 'cz' for both — only what the user "
                         "sees is templated.")
    ap.add_argument("--checkpoint", help="path to judge checkpoint CSV")
    ap.add_argument("--output", help="path to write the HTML report")
    ap.add_argument("--baseline", default=None, type=Path,
                    help="Optional: path to a prior judge checkpoint CSV "
                         "to compare against. Re-runs enrich() on the "
                         "baseline so the deltas always use the current "
                         "report's pass criterion. Adds Δ chips to the "
                         "Summary headline cards and Top-3 failure cards. "
                         "Omit to render a single-run report with no "
                         "baseline comparison surfaces.")
    ap.add_argument("--config-dir", default=None, type=Path,
                    help="Directory containing experiment YAML configs. "
                         "When omitted, the resolver searches the per-lang "
                         "well-known layouts (experiments/{cz,sk}kb/configs/...).")
    ap.add_argument("--yaml-name", default=None,
                    help="config name (e.g. czkb_exp_002_baseline_no_expected_enums). "
                         "Auto-detected from the checkpoint filename if omitted.")
    ap.add_argument("--reasoning-effort", default="medium",
                    help="used only to derive default checkpoint paths; "
                         "the header reads reasoning_effort directly from the YAML.")
    ap.add_argument("--suffix", default="",
                    help="optional filename suffix (e.g. '_test' for a test run)")
    ap.add_argument("--prompts", default=None, type=Path,
                    help="Path to the prompt sidecar JSON (or its directory). "
                         "If omitted, the report looks for "
                         "prompt_{mlflow_run_id}.json next to the checkpoint.")
    ap.add_argument("--include-latency", action="store_true",
                    help="Include the experimental Latency surfaces "
                         "(headline card on Summary row 2, Latency tab, "
                         "case-detail Latency score-box, in-case Latency "
                         "breakdown, Notes-tab definitions). These are "
                         "approximations — retry detection is a tok/s-based "
                         "heuristic and can mis-fire. Off by default so the "
                         "shareable report stays clean; turn on for "
                         "internal perf reviews.")
    args = ap.parse_args()

    # Apply language before any rendering so the FAILURE_MODE_INFO_LONG
    # placeholders and the visible button labels reflect the run's language.
    _apply_lang(args.lang)
    if args.config_dir is not None:
        global _CONFIG_DIR_OVERRIDE
        _CONFIG_DIR_OVERRIDE = args.config_dir

    # If --yaml-name is omitted, try to auto-detect it from the checkpoint
    # filename (so colleagues don't need to remember the exact config name).
    yaml_name = args.yaml_name
    if not yaml_name and args.checkpoint:
        yaml_name = _auto_yaml_name(Path(args.checkpoint))
    if not yaml_name:
        # Per-lang last-resort fallback when the checkpoint stem doesn't
        # match any YAML in the search list.
        yaml_name = f"{args.lang}kb_exp_001_baseline_no_expected_enums"

    ckpt_default, out_default = _default_paths(yaml_name, args.reasoning_effort, args.suffix)
    ckpt = Path(args.checkpoint) if args.checkpoint else ckpt_default
    out = Path(args.output) if args.output else out_default

    if not ckpt.exists():
        sys.exit(f"checkpoint not found: {ckpt}")

    # Load the optional baseline run BEFORE the main df so any load failure
    # surfaces before the slow enrich+render path runs.
    baseline_payload: dict | None = None
    if args.baseline is not None:
        if not args.baseline.exists():
            sys.exit(f"--baseline not found: {args.baseline}")
        baseline_payload = load_baseline(args.baseline)
        print(
            f"baseline: {args.baseline.name} · "
            f"n_clean={baseline_payload.get('n_clean')} · "
            f"pass_rate_clean={baseline_payload.get('metrics', {}).get('pass_rate_clean')}"
        )

    df_all = enrich(read_checkpoint_csv(ckpt))
    # Pre-filter to drop only the rows we genuinely cannot score: those
    # with an empty user_query (the judge had nothing to grade). All other
    # rows — including non-KB case_scope (api / out_of_scope / ambiguous)
    # — go through the full failure-mode classifier so they're visible in
    # the Failure modes table rather than silently dropped.
    empty_mask = df_all.get(
        "_user_query_empty", pd.Series([False]*len(df_all), index=df_all.index))
    df = df_all[~empty_mask].reset_index(drop=True)
    n_empty_total = int(empty_mask.sum())

    # Header display values come straight from the YAML (experiment.name,
    # model.model_deployment_name, model.reasoning_effort) so the report
    # always names the actual run being scored, regardless of CLI defaults.
    meta = _detect_experiment_meta(yaml_name)
    html_str = render_html(
        df,
        df_all=df_all,
        yaml_name=yaml_name,
        reasoning_effort=meta["reasoning_effort"],
        checkpoint_label=str(ckpt.name),
        judge_model=meta["judge_model"],
        experiment_name=meta["name"],
        mlflow_run_id=meta["mlflow_run_id"],
        mlflow_experiment_id=meta["mlflow_experiment_id"],
        mlflow_run_timestamp=meta["mlflow_run_timestamp"],
        include_latency=args.include_latency,
        checkpoint_path=ckpt,
        prompts_path=args.prompts,
        baseline=baseline_payload,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_str)
    print(f"rows: {len(df)} of {len(df_all)} "
          f"(dropped {n_empty_total} empty user_query)   → {out}   "
          f"({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
