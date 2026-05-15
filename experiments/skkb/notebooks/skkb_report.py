"""Generate an interactive HTML report from a SKKB judge checkpoint CSV.

Standalone — does NOT require the results-viewer notebook to have run.
Computes the same programmatic derivations the viewer does (weighted_avg,
retrieval_recall, root_cause_category, enum_f1, missing-ENUM counters,
Pearson agreement matrices) and writes a single self-contained HTML file
with tabbed navigation and a clickable per-test-case drill-down.

Usage:
    python skkb_report.py \\
        --checkpoint checkpoints/evals_skkb_exp_001_baseline_medium_test.csv \\
        --output     /tmp/skkb_report.html

    python skkb_report.py \\
        --yaml-name skkb_exp_001_baseline \\
        --reasoning-effort medium \\
        --suffix _test
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
from skkb_checkpoint import read_checkpoint_csv

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
# configs/skkb/skkb_exp_001_baseline_no_expected_enums.yaml. weighted_avg
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
    "pass":                "Pass",
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
    "other_failure":       "Other failure",
}

# Two-tier descriptions: SHORT (FAILURE_MODE_INFO) goes in the failure-mode
# table — concise, scannable. LONG (FAILURE_MODE_INFO_LONG) feeds the per-row
# info-icon tooltip and the Doc tab so the full reasoning is one click away.
FAILURE_MODE_INFO = {
    "pass":                f"weighted_avg ≥ {PASS_THRESHOLD:g}, no weight-2 scorer at 0, and no failure category fired.",
    "test_set_defect":     "Judge flagged the gold reference as needing review.",
    "enum_name_mismatch":  "Expected ENUM matches a KB ENUM only after stripping case/separators — false-zero risk.",
    "scope_misroute":      "Judge said KB; system never reached the KB.",
    "retrieval_gap":       "Expected ENUM never entered the pre-prune candidate pool.",
    "pruning_loss":        "Expected ENUM was dropped between pre-prune and post-prune.",
    "reranker_miss":       "Expected ENUM was in the post-prune pool but not picked.",
    "pool_content_gap":    "Pool fragments exist but are too thin to answer.",
    "context_use_failure": "Context was usable; agent answered wrong anyway.",
    "hallucination":       "Severe: agent's answer contains important unsupported / fabricated claims (groundedness = 0).",
    "language_drift":      "Agent answered in non-Slovak.",
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
        "language_compliance ≤ 1 — agent answered in Czech or another "
        "non-Slovak language.",
    "critical_score_zero":
        f"weighted_avg ≥ {PASS_THRESHOLD:g} (so the average says \"pass\"), "
        "but at least one weight-2 scorer "
        f"({', '.join(CRITICAL_DIMS)}) "
        "scored 0. We veto pass in that case so the headline doesn't hide "
        "a critical-dimension failure behind a high mean. Cases where a "
        "more specific failure mode also fired (hallucination, "
        "context-use failure, etc.) are classified under that mode "
        "instead — this bucket only catches the residual.",
}


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

    for dim in DIMENSION_WEIGHTS:
        col = f"{dim}_score"
        if col not in df.columns:
            raise KeyError(f"missing dimension column: {col}")
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

    # `expected_answer_summary_with_optimal_context = sentinel` is the judge's
    # "the post-prune pool cannot answer this query" signal — independent of
    # the dimension scores.
    if "expected_answer_summary_with_optimal_context" in df.columns:
        df["_no_achievable_answer"] = df["expected_answer_summary_with_optimal_context"].apply(
            lambda v: _is_void_string(v) if isinstance(v, str) else False
        )
    else:
        df["_no_achievable_answer"] = pd.Series([False] * len(df), index=df.index)

    # ── Empty user_query flag ────────────────────────────────────────────────
    if "user_query" in df.columns:
        df["_user_query_empty"] = (
            df["user_query"].fillna("").astype(str).str.strip().eq("")
        )
    else:
        df["_user_query_empty"] = pd.Series([True] * len(df), index=df.index)

    # ── Stage-by-stage expected-ENUM recall (unweighted) ─────────────────────
    def _recall_from_to(expected, candidate) -> float:
        e = set(expected or [])
        if not e:
            return np.nan
        return len(e & set(candidate or [])) / len(e)

    if "_pre_prune_enum_ids" not in df.columns:
        df["_pre_prune_enum_ids"] = df.get(
            "pre_prune_enum_ids", pd.Series([""] * len(df), index=df.index)
        ).apply(_parse_list)

    df["_expected_recall_pre_prune"] = df.apply(
        lambda r: _recall_from_to(r.get("_expected_enums", []), r.get("_pre_prune_enum_ids", [])),
        axis=1,
    )
    df["_expected_recall_post_prune"] = df.apply(
        lambda r: _recall_from_to(r.get("_expected_enums", []), r.get("_post_prune_enum_ids", [])),
        axis=1,
    )
    df["_expected_recall_reranked"] = df.apply(
        lambda r: _recall_from_to(r.get("_expected_enums", []), r.get("_reranked_enum_ids", [])),
        axis=1,
    )

    def _opt_vs_final(r) -> float:
        opt = set(r.get("_optimal_enum_selection", []) or [])
        final = set(r.get("_reranked_enum_ids", []) or [])
        if not opt:
            return np.nan
        return len(opt & final) / len(opt)

    df["_recall_optimal_vs_final"] = df.apply(_opt_vs_final, axis=1)

    def _distractor_rate(r) -> float:
        final = r.get("_reranked_enum_ids", []) or []
        extra = r.get("_extra_or_distracting_enums", []) or []
        if not final:
            return np.nan
        return len(extra) / len(final)

    df["_distractor_rate"] = df.apply(_distractor_rate, axis=1)

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
    df["_reranked_empty_kb"] = df["_case_scope_kb_like"] & df["_reranked_enum_ids"].apply(
        lambda v: isinstance(v, list) and len(v) == 0
    )

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


# ── HTML rendering ───────────────────────────────────────────────────────────
def _h(s) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return _html.escape(str(s))


def _scope_badge(scope: str) -> str:
    cls = {
        "kb": "scope-kb", "mock_tool": "scope-mock_tool",
        "dba_no_tools": "scope-dba_no_tools", "main_agent": "scope-main_agent",
        "other_tools": "scope-other",
        "hg_invest_kb": "scope-hg_invest",
        "hg_invest_mock_tool": "scope-hg_invest",
        "hg_invest_no_tools": "scope-hg_invest",
        "hg_invest_other_tools": "scope-hg_invest",
    }.get(str(scope), "scope-other")
    return f'<span class="scope-badge {cls}">{_h(scope)}</span>'


def _agent_badge(agent: str) -> str:
    cls = {
        "main_agent": "agent-main",
        "daily_banking_agent": "agent-dba",
        "hg-invest-phase2": "agent-hg_invest",
    }.get(str(agent), "agent-other")
    label = _h(agent) if agent else "—"
    return f'<span class="agent-badge {cls}">{label}</span>'


_TABLE_COLS = ("test_case_id", "query_scope", "last_agent", "weighted_avg", "expert_score",
                "enum_relevance_score", "root_cause_category", "user_query")
_COL_LABELS = {"enum_relevance_score": "rel2", "weighted_avg": "judge_w.avg",
                "root_cause_category": "root_cause"}

# Filter widget type per column when ``filterable=True``. Anything not listed
# falls back to a plain substring text filter.
_FILTER_TYPES = {
    "test_case_id":           "text",
    "user_query":             "text",
    "weighted_avg":           "range",
    "expert_score":           "range",
    "enum_relevance_score":   "range",
    "query_scope":            "multi",
    "last_agent":             "multi",
    "root_cause_category":    "multi",
}

# Numeric step/min/max defaults for range inputs.
_RANGE_CFG = {
    "weighted_avg":         (0, 1, 0.05),
    "expert_score":         (0, 10, 1),
    "enum_relevance_score": (0, 1, 0.05),
}


# Color-coding thresholds for numeric cells.
_NUM_THRESHOLDS = {
    "weighted_avg": (0.5, PASS_THRESHOLD),          # bad < 0.5, mid < 0.7, good ≥ 0.7
    "expert_score": (4.0, 7.0),                      # bad < 4, mid < 7, good ≥ 7
    "enum_relevance_score": (0.5, 0.8),              # bad < 0.5, mid < 0.8, good ≥ 0.8
}


def _num_class(col: str, v) -> str:
    t = _NUM_THRESHOLDS.get(col)
    if t is None or pd.isna(v):
        return "num-na"
    lo, hi = t
    if v < lo: return "num-bad"
    if v < hi: return "num-mid"
    return "num-good"


def _filter_widget(col: str, series: pd.Series) -> str:
    """Render the per-column filter widget."""
    ftype = _FILTER_TYPES.get(col, "text")
    if ftype == "range":
        lo, hi, step = _RANGE_CFG.get(col, (0, 10, 0.1))
        return (f'<div class="range-filter" data-col="{col}">'
                f'<input class="col-range-min" data-col="{col}" type="number" '
                f'min="{lo}" max="{hi}" step="{step}" placeholder="min">'
                f'<span class="range-sep">–</span>'
                f'<input class="col-range-max" data-col="{col}" type="number" '
                f'min="{lo}" max="{hi}" step="{step}" placeholder="max">'
                f'</div>')
    if ftype == "multi":
        values = sorted(str(v) for v in series.dropna().unique())
        opts = "".join(
            f'<label><input type="checkbox" value="{_h(v)}"> {_h(v)}</label>'
            for v in values
        )
        return (f'<div class="multi-filter" data-col="{col}">'
                f'<button type="button" class="multi-toggle">all</button>'
                f'<div class="multi-menu">{opts}</div>'
                f'</div>')
    return (f'<input class="col-filter" data-col="{col}" type="text" '
            f'placeholder="filter…">')


def _table_rows(sub: pd.DataFrame, cols: tuple = _TABLE_COLS,
                *, table_id: str | None = None, filterable: bool = False) -> str:
    cols = [c for c in cols if c in sub.columns]
    if not cols or sub.empty:
        return "<p class='placeholder'>no rows</p>"
    thead_rows = ["<tr>" + "".join(
        f'<th data-col="{c}">{_COL_LABELS.get(c, c)}</th>' for c in cols
    ) + "</tr>"]
    if filterable:
        thead_rows.append(
            "<tr class='filter-row'>" + "".join(
                f'<th>{_filter_widget(c, sub[c])}</th>' for c in cols
            ) + "</tr>"
        )
    body = ""
    for _, r in sub.iterrows():
        tid = _h(r.get("test_case_id"))
        cells = []
        for c in cols:
            v = r[c]
            if c == "test_case_id":
                cells.append(
                    f'<td data-col="{c}" data-val="{tid}">'
                    f'<a class="case-link" href="#" data-case="{tid}">{tid}</a></td>'
                )
            elif c == "query_scope":
                cells.append(f'<td data-col="{c}" data-val="{_h(v)}">{_scope_badge(v)}</td>')
            elif c == "last_agent":
                cells.append(f'<td data-col="{c}" data-val="{_h(v)}">{_agent_badge(v)}</td>')
            elif c in ("weighted_avg", "expert_score", "enum_relevance_score"):
                if pd.isna(v):
                    cells.append(f'<td data-col="{c}" data-val="" data-numeric="">–</td>')
                else:
                    cls = _num_class(c, v)
                    cells.append(
                        f'<td data-col="{c}" data-val="{v:.4f}" data-numeric="{v:.6f}" '
                        f'class="num-cell {cls}">'
                        f'<span class="num-pill">{v:.2f}</span></td>'
                    )
            else:
                text = _h(v)[:240]
                cells.append(f'<td data-col="{c}" data-val="{_h(v)}">{text}</td>')
        body += "<tr>" + "".join(cells) + "</tr>"
    id_attr = f' id="{table_id}"' if table_id else ""
    cls = "tbl" + (" tbl-filterable" if filterable else "")
    return (f'<table class="{cls}"{id_attr}>'
            f'<thead>{"".join(thead_rows)}</thead><tbody>{body}</tbody></table>')


def _case_payload(r: pd.Series) -> dict:
    def g(c, default=""):
        return r[c] if c in r.index and not pd.isna(r[c]) else default
    def b(c, default=False):
        value = g(c, default)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"true", "1", "1.0", "yes"}
    return {
        "id": g("test_case_id"),
        "trace_id": g("trace_id"),
        "scope": g("query_scope"),
        "last_agent": g("last_agent"),
        "rerank_empty": b("reranker_selected_empty"),
        "weighted_avg": float(r["weighted_avg"]) if pd.notna(r.get("weighted_avg")) else None,
        "expert_score": float(r["expert_score"]) if pd.notna(r.get("expert_score")) else None,
        "rel2_score": float(r["enum_relevance_score"]) if pd.notna(r.get("enum_relevance_score")) else None,
        "enum_f1": float(r["enum_f1"]) if pd.notna(r.get("enum_f1")) else None,
        "root_cause": g("root_cause_category"),
        "user_query": g("user_query"),
        "user_query_en": g("user_query_en"),
        "agent_response": g("agent_response"),
        "agent_response_en": g("agent_response_en"),
        "expected_response": g("expected_response"),
        "expected_response_en": g("expected_response_en"),
        "reranked_enum_ids": g("reranked_enum_ids"),
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
    }


# ── Bins & contingency matrices for expert / Rel2 / judge comparison ────────
EXPERT_BINS = ["Low (≤3)", "Mid (4-6)", "Good (7-8)", "Excellent (9-10)"]
WAVG_BINS = ["Bad (<0.25)", "Weak (0.25-0.5)", "Mid (0.5-0.7)", "Pass (≥0.7)"]
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

    fig_hist = px.histogram(df, x="weighted_avg", nbins=20)
    fig_hist.update_traces(marker=dict(color=GE_BLUE, line=dict(width=0),
                                          cornerradius=4))
    fig_hist.add_vline(x=PASS_THRESHOLD, line_color=GE_RED,
                       line_width=1, annotation_text=f"pass ≥ {PASS_THRESHOLD}",
                       annotation_position="top right",
                       annotation_font=dict(color=GE_RED, size=11))
    _style_fig(fig_hist, height=320)
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


def _build_judge_failure_field_fig(df: pd.DataFrame):
    """Bar chart of how often the judge populated each failure-detection
    field. Each judge output array (hallucinated_claims, missing_facts,
    unavailable_facts_in_selected_context, extra_or_distracting_enums) has
    a *_cnt numeric column built in enrich(). For every in-scope KB
    case (df is already filtered to non-empty user_query AND case_scope
    in {kb, kb_and_api}) we sum the per-row counts. The result is the
    total number of detected items across the run for each failure type.
    Sentinel-normalized arrays were already collapsed to [] in enrich,
    so the void-marker rows don't inflate the totals."""
    fields = (
        ("hallucinated_claims_cnt",     "Hallucinated claims"),
        ("missing_facts_cnt",           "Missing facts (vs expected response)"),
        ("unavailable_facts_cnt",       "Unavailable facts in selected context"),
        ("extra_distracting_enums_cnt", "Extra / distracting ENUMs"),
    )
    rows = []
    for col, label in fields:
        if col not in df.columns:
            continue
        counts = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        affected = counts > 0
        ids = df.loc[affected, "test_case_id"].astype(str).tolist() \
              if "test_case_id" in df.columns else []
        rows.append({
            "label":  label,
            "total":  int(counts.sum()),
            "cases":  int(affected.sum()),
            "ids":    ids,
        })
    if not rows or all(r["total"] == 0 for r in rows):
        return None
    rows.sort(key=lambda r: -r["total"])
    fig = go.Figure(go.Bar(
        x=[r["total"] for r in rows],
        y=[r["label"] for r in rows],
        orientation="h",
        marker=dict(color=GE_BLUE, line=dict(width=0), cornerradius=6),
        text=[r["total"] for r in rows],
        textposition="outside",
        cliponaxis=False,
        textfont=dict(family=GE_FONT, size=11, color=GE_TEXT),
        # customdata[0] = case count for hover; customdata[1] = ID list
        # surfaced to the click handler so the bar can drill into Test Cases.
        customdata=[[r["cases"], r["ids"]] for r in rows],
        hovertemplate=("<b>%{y}</b><br>%{x} items detected"
                        "<br>across %{customdata[0]} cases — click to filter<extra></extra>"),
    ))
    _style_fig(fig, height=max(220, 44 * len(rows) + 80))
    fig.update_layout(margin=dict(t=20, b=40, l=260, r=80), bargap=0.4,
                      xaxis_title="total items detected",
                      yaxis_title="")
    if rows:
        fig.update_xaxes(range=[0, max(r["total"] for r in rows) * 1.18 + 1])
    fig.update_yaxes(tickfont=dict(family="monospace", size=11, color=GE_TEXT),
                      automargin=True)
    return fig


def _build_enum_count_distribution_fig(df: pd.DataFrame):
    """DEPRECATED — replaced by `_enum_count_distribution_table_html` for a
    clearer numeric presentation. Kept as a stub returning None so any
    leftover call sites render a placeholder rather than blowing up.
    """
    return None


def _enum_count_distribution_table_html(df: pd.DataFrame) -> str:
    """Per-case distribution of |expected_enums| vs |reranked_enum_ids| as a
    table — easier to read the actual counts than the overlapping bar
    chart it replaces. Restricted to KB-routed cases (query_scope == 'kb').
    Each row: ENUM-count value, # cases with that many expected, # cases
    with that many selected, and the same numbers as % of n.
    """
    needed = {"_expected_enums", "_reranked_enum_ids", "query_scope"}
    if not needed.issubset(df.columns):
        return "<p class='placeholder'>no expected/selected ENUM data</p>"
    qs = df["query_scope"].fillna("").astype(str)
    sub = df[qs.eq("kb")]
    if sub.empty:
        return "<p class='placeholder'>no KB-routed cases in this run.</p>"
    expected_lens = sub["_expected_enums"].apply(
        lambda v: len(v) if isinstance(v, list) else 0
    )
    reranked_lens = sub["_reranked_enum_ids"].apply(
        lambda v: len(v) if isinstance(v, list) else 0
    )
    n = len(sub)
    max_count = int(max(int(expected_lens.max() or 0), int(reranked_lens.max() or 0)))
    rows_html = ""
    for k in range(max_count + 1):
        e_n = int((expected_lens == k).sum())
        r_n = int((reranked_lens == k).sum())
        e_pct = _fmt_pct(e_n / n) if n else "–"
        r_pct = _fmt_pct(r_n / n) if n else "–"
        rows_html += (
            "<tr>"
            f"<td style='text-align:right;font-variant-numeric:tabular-nums'><strong>{k}</strong></td>"
            f"<td style='text-align:right'>{e_n}</td>"
            f"<td style='text-align:right;color:#5c7999'>{e_pct}</td>"
            f"<td style='text-align:right'>{r_n}</td>"
            f"<td style='text-align:right;color:#5c7999'>{r_pct}</td>"
            "</tr>"
        )
    mean_e = float(expected_lens.mean()) if n else float("nan")
    mean_r = float(reranked_lens.mean()) if n else float("nan")
    return (
        f"<p style='font-size:12px;color:#5c7999;margin-bottom:8px'>"
        f"n = <strong>{n}</strong> KB-routed cases. "
        f"Mean expected = <strong>{mean_e:.2f}</strong>, "
        f"mean selected = <strong>{mean_r:.2f}</strong>.</p>"
        "<table class='tbl funnel-tbl'>"
        "<thead><tr>"
        "<th style='text-align:right'>ENUMs / case</th>"
        "<th style='text-align:right'>Expected · cases</th>"
        "<th style='text-align:right'>%</th>"
        "<th style='text-align:right'>Selected · cases</th>"
        "<th style='text-align:right'>%</th>"
        "</tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
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

    # Pass = cases whose primary failure_mode is "pass" — i.e. the case
    # cleared the threshold AND no failure mode (test-set issue, naming
    # mismatch, agent-side issue, etc.) fired ahead of it. This makes the
    # headline match the Pass row in the Failure-modes table exactly.
    pass_mode_mask = (df.get("failure_mode",
                              pd.Series([""]*len(df), index=df.index)) == "pass")
    n_pass        = int(pass_mode_mask.sum())
    pass_rate_all = (n_pass / n_eval) if n_eval else float("nan")

    defect_mask   = (
        df.get("_gold_defect", pd.Series([False]*len(df), index=df.index))
        | df.get("_case_scope_test_defect", pd.Series([False]*len(df), index=df.index))
        | df.get("_naming_mismatch", pd.Series([False]*len(df), index=df.index))
    )
    n_defect      = int(defect_mask.sum())
    n_clean       = n_eval - n_defect
    # n_pass_clean equals n_pass: failure_mode=="pass" already excludes
    # test-set defects and naming mismatches by priority order. Keep both
    # names for downstream tooltips that read each independently.
    n_pass_clean  = n_pass
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
    # Surfaces the two largest agent-side / pipeline-side issues for a
    # one-glance "what's hurting us most" answer on the Summary tab. We
    # rank by count within the *clean* subset (defects excluded) so the
    # numbers reflect the agent's real performance, not test-set hygiene.
    clean_mask = ~defect_mask
    fm_in_scope = df.loc[clean_mask, "failure_mode"] if "failure_mode" in df.columns else pd.Series(dtype=object)
    excluded_for_top = {"pass", "test_set_defect"}
    fm_in_scope_failures = fm_in_scope[~fm_in_scope.isin(excluded_for_top)]
    fm_top_counts = fm_in_scope_failures.value_counts()
    top_failures_clean = []
    for fm in fm_top_counts.index[:3]:
        n = int(fm_top_counts[fm])
        pct = (n / n_clean) if n_clean else float("nan")
        ids = df.loc[clean_mask & (df["failure_mode"] == fm), "test_case_id"] \
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
        "language_compliance":                 "Did the agent answer in Slovak?",
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
        f"Includes non-KB case_scope rows — they go through the failure-mode "
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
                row["ids"], f"failure mode: {row['label']}",
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
                "Same denominator as the Top-3 failure-reasons card above and the "
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
    return (
        f"<p style='font-size:12px;color:#5c7999;margin-bottom:8px'>"
        f"Micro-averaged recall and precision (Σ TP / Σ expected · Σ TP / Σ selected) "
        f"at each pipeline stage. "
        f"Pre-prune precision is naturally "
        f"low — the vector DB returns many candidates by design; the later "
        f"stages drop noise to improve it.</p>"
        f"<table class='tbl funnel-tbl'>"
        f"<thead><tr><th>Stage</th>"
        f"<th style='text-align:right'>Recall</th>"
        f"<th>&nbsp;</th>"
        f"<th style='text-align:right'>Precision</th>"
        f"<th>&nbsp;</th></tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
    )


def _opt_vs_final_html(opt: dict) -> str:
    n = opt["n"]
    if n == 0:
        return "<p class='placeholder'>no rows have a non-empty <code>optimal_enum_selection</code>.</p>"
    bucket_rows = ""
    for label, count in opt["buckets"]:
        pct = (count / n) if n else float("nan")
        bucket_rows += (
            "<tr>"
            f"<td>{_h(label)}</td>"
            f"<td style='text-align:right'>{count}</td>"
            f"<td style='text-align:right'>{_fmt_pct(pct)}</td>"
            "</tr>"
        )
    return (
        f"<p style='font-size:12px;color:#5c7999;margin-bottom:8px'>"
        f"Per-case recall of the judge's <code>optimal_enum_selection</code> against "
        f"the actual <code>reranked_enum_ids</code>. n = {n}. Mean = "
        f"<strong>{_fmt_pct(opt['mean'])}</strong>.</p>"
        f"<table class='tbl funnel-tbl'>"
        f"<thead><tr><th>Recall bucket</th><th style='text-align:right'>N</th>"
        f"<th style='text-align:right'>%</th></tr></thead>"
        f"<tbody>{bucket_rows}</tbody></table>"
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


def _enum_count_comparison_html(metrics: dict) -> str:
    """Compare how many ENUMs the reranker selected against how many the
    test set expected. Surfaces systemic over/under-selection bias and
    empty-selection rates."""
    n = metrics.get("count_cmp_n", 0)
    if n == 0:
        return ("<p class='placeholder'>no rows have both case_scope ∈ "
                "{kb, kb_and_api} AND a non-empty <code>expected_enums</code>.</p>")
    mean_e = metrics.get("count_cmp_mean_expected", float("nan"))
    mean_s = metrics.get("count_cmp_mean_selected", float("nan"))
    rows = ""
    for entry in metrics.get("count_cmp_buckets", []):
        n_b = entry["n"]
        pct = (n_b / n) if n else float("nan")
        if n_b and entry["ids"]:
            cell = _ids_filter_link(
                entry["ids"], f"count comparison: {entry['label']}",
                classes="judge-eval-link", inner=str(n_b),
            )
        else:
            cell = str(n_b)
        rows += (
            "<tr>"
            f"<td>{_h(entry['label'])}</td>"
            f"<td style='text-align:right'>{cell}</td>"
            f"<td style='text-align:right'>{_fmt_pct(pct)}</td>"
            "</tr>"
        )
    return (
        f"<p style='font-size:12px;color:#5c7999;margin-bottom:8px'>"
        f"<strong>{n}</strong> in-scope cases have a non-empty <code>expected_enums</code>. "
        f"Mean expected = <strong>{mean_e:.2f}</strong> · "
        f"mean selected = <strong>{mean_s:.2f}</strong>. "
        f"A persistent gap (mean selected ≪ mean expected, or many empty "
        f"selections) is reranker-prompt evidence.</p>"
        f"<table class='tbl funnel-tbl'>"
        f"<thead><tr><th>Pattern</th><th style='text-align:right'>N</th>"
        f"<th style='text-align:right'>%</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
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


def _top_failures_html(top: list[dict], n_clean: int) -> str:
    """Render the top-2 failure reasons (excl. test-set issues) as a row of
    headline-style cards. Click any card to drill into the matching cases.
    Returns "" when there's nothing to show (everything passed, or no clean
    rows after defect exclusion)."""
    if not top or n_clean <= 0:
        return ""
    cards = []
    for entry in top:
        n = entry["n"]
        pct = entry["pct"]
        click_link = _ids_filter_link(
            entry["ids"], f"top failure: {entry['label']}",
            classes="judge-eval-link top-failure-link",
            inner=f"{_fmt_pct(pct)}",
        ) if n else _fmt_pct(pct)
        cards.append(
            "<div class='headline-card top-failure-card'>"
            f"<div class='hc-label'>{_h(entry['label'])} "
            f"<span class='top-failure-owner'>· {_h(entry['owner'])}</span></div>"
            f"<div class='hc-value'>{click_link}</div>"
            f"<div class='hc-detail'>{n} cases / {n_clean} (test-set issues excluded).</div>"
            "</div>"
        )
    note = (
        f"<div class='top-failure-title'>Top {len(cards)} failure "
        f"reason{'s' if len(cards) != 1 else ''} (excl. test-set issues) "
        f"{_info_icon('After removing the cases the judge flagged for gold-reference review / ambiguous / out-of-scope, these are the failure modes that account for the most cases. Click a card to drill in.')}</div>"
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
    configurations so colleagues can see the boundary concretely."""
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


def _doc_tab_html(metrics: dict) -> str:
    """Notes page — companion text to the rest of the report. Every count
    or percentage referenced in the body comes from ``metrics`` so the
    text stays in sync with the run on display."""
    cs_html = _scope_distribution_html(metrics["case_scope_counts"], metrics["n_eval"])
    fm_terms = "".join(
        f"<dt><strong>{_h(FAILURE_MODE_LABEL[k])}</strong> "
        f"<span style='color:#5c7999'>(owner: {_h(FAILURE_MODE_OWNER[k])})</span></dt>"
        f"<dd>{_h(FAILURE_MODE_INFO_LONG.get(k, FAILURE_MODE_INFO[k]))}</dd>"
        for k in FAILURE_MODES
    )
    return f"""
    <div class="card">
      <div class="card-title">Pass rate</div>
      <p>The headline pass rate is the share of analyzed cases whose
      <code>weighted_avg</code> meets the pass threshold of
      <strong>{PASS_THRESHOLD}</strong> <em>and</em> have no critical
      (weight-2) scorer at 0. <code>weighted_avg</code> is the weighted
      mean of the seven judge scorers, normalized to the [0, 1] range;
      the critical-zero veto prevents a single catastrophic dimension
      (e.g. groundedness = 0) from hiding behind an otherwise-high mean.
      The denominator is every case in the checkpoint with a non-empty
      <code>user_query</code> — that's the only Stage-1 filter. Rows the
      judge classified as api / out_of_scope / ambiguous are kept in the
      analysis and flow through the failure-mode classifier (ambiguous /
      out_of_scope land in <em>Test-set issue</em>; api passes through
      to whatever applies given its scores).</p>
      <p>The Summary tab shows two pass rates side by side:</p>
      <ul style="margin-left:20px;margin-bottom:8px">
        <li><strong>Pass rate · all cases</strong>
            ({metrics['n_pass']} / {metrics['n_eval']} =
            {_fmt_pct(metrics['pass_rate_all'])}) — production-realistic,
            includes cases with gold-reference issues and ENUM-naming
            mismatches.</li>
        <li><strong>Pass rate · excl. test-set issues</strong>
            ({metrics['n_pass_clean']} / {metrics['n_clean']} =
            {_fmt_pct(metrics['pass_rate_clean'])}) — drops rows the judge
            flagged with <code>expected_reference_looks_wrong</code>,
            rows classified as <code>ambiguous</code>/<code>out_of_scope</code>,
            and rows the deterministic check flagged as ENUM naming
            mismatches; measures the agent against trustworthy ground
            truth only.</li>
      </ul>
    </div>

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
      <div class="card-title">Failure modes</div>
      <p style="font-size:12px;color:#5c7999;margin-bottom:10px">
      Each analyzed case is assigned one primary failure mode in priority
      order: test-set issues first (because they invalidate downstream
      analysis), then ENUM name mismatch, then wrong agent routing,
      then per-stage ENUM losses, then pool/agent/language issues, with
      <em>pass</em> and <em>other_failure</em> as the residual buckets.
      The Summary's failure-mode table shows two denominators: the
      test-set group (test-set issue, ENUM name mismatch) is a share of
      <code>n_eval</code>; everything below the "All valid cases"
      divider is a share of <code>n_clean = n_eval − test-set issue −
      ENUM name mismatch</code>, matching the Top-3 failure-reasons
      card above and the "Pass rate · excl. test-set issues" headline.</p>
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
    and judge score` cells in `skkb_001_results_viewer_local.ipynb`.
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
    """Aggregate every distinct KB entry that appeared in any case's reranked
    context, keyed by ``enum_id``. Each entry holds the SK + EN description
    (parsed from ``reranked_enums_kb_sk`` / ``_en`` JSON columns) and the
    sorted list of test cases that referenced it. Empty dict if the columns
    are missing.
    """
    sk_col = "reranked_enums_kb_sk" if "reranked_enums_kb_sk" in df.columns else None
    en_col = "reranked_enums_kb_en" if "reranked_enums_kb_en" in df.columns else None
    if not (sk_col or en_col):
        return {}
    kb: dict[str, dict] = {}
    for _, r in df.iterrows():
        tc = "" if pd.isna(r.get("test_case_id")) else str(r.get("test_case_id"))
        for col, lang in ((sk_col, "sk"), (en_col, "en")):
            if not col:
                continue
            raw = r.get(col)
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                continue
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
                # Upstream JSON often double-escapes whitespace (\\n / \\r / \\t
                # in the source instead of \n / \r / \t), so json.loads gives
                # us a literal backslash+letter. Convert to real whitespace so
                # CSS white-space:pre-wrap renders proper line breaks.
                desc = (desc.replace("\\r\\n", "\n")
                              .replace("\\n", "\n")
                              .replace("\\r", "\n")
                              .replace("\\t", "\t"))
                entry = kb.setdefault(eid, {"sk": "", "en": "", "cases": set()})
                if not entry[lang] and desc:
                    entry[lang] = desc
                if tc:
                    entry["cases"].add(tc)
    # Stabilize order — by numeric tail of test_case_id, then string.
    def _tc_key(t: str):
        m = re.search(r"\d+", t)
        return (int(m.group()) if m else 10**9, t)
    for v in kb.values():
        v["cases"] = sorted(v["cases"], key=_tc_key)
    return kb


def _render_kb_html(kb_data: dict) -> str:
    if not kb_data:
        return (
            "<div class='card'>"
            "<div class='card-title'>Knowledge base</div>"
            "<p class='placeholder'>No <code>reranked_enums_kb_sk</code> / "
            "<code>reranked_enums_kb_en</code> columns in the checkpoint.</p>"
            "</div>"
        )
    items = sorted(kb_data.items(), key=lambda x: (-len(x[1]["cases"]), x[0]))
    rows_html = ""
    for eid, entry in items:
        n_cases = len(entry["cases"])
        ids_attr = _h(json.dumps(entry["cases"]))
        label = _h(f"KB enum: {eid}")
        cases_link = (
            f"<a href='#' class='judge-eval-link' "
            f"data-ids='{ids_attr}' data-label='{label}' "
            f"title='Show the {n_cases} cases that referenced this entry'>"
            f"{n_cases} {'case' if n_cases == 1 else 'cases'}</a>"
        )
        sk_html = _h(entry["sk"]) if entry["sk"] else "<em class='lang-fallback'>(no SK text)</em>"
        en_html = _h(entry["en"]) if entry["en"] else "<em class='lang-fallback'>(no EN text)</em>"
        # Searchable haystack: enum_id + both descriptions, lower-cased.
        search_text = _h((eid + " " + entry["sk"] + " " + entry["en"]).lower())
        rows_html += (
            f"<div class='kb-row collapsed' data-search='{search_text}' "
            f"data-ids='{ids_attr}'>"
            f"<div class='kb-row-head'>"
            f"<span class='kb-chev' aria-hidden='true'>▶</span>"
            f"<code class='kb-id'>{_h(eid)}</code>"
            f"<span class='kb-cases-count'>{cases_link}</span>"
            f"</div>"
            f"<div class='kb-desc lang-sk'>{sk_html}</div>"
            f"<div class='kb-desc lang-en'>{en_html}</div>"
            f"</div>"
        )
    return (
        "<div class='card'>"
        f"<div class='card-title'>Knowledge base — {len(items)} unique entries appearing in reranked context</div>"
        "<p style='font-size:12px;color:#537090;margin-bottom:10px'>"
        "Aggregated from each case's <code>reranked_enums_kb_sk</code> / "
        "<code>reranked_enums_kb_en</code>. Use the SK / EN switch on the right to flip language. "
        "Click the case-count link to filter the Test Cases tab to those rows."
        "</p>"
        "<div class='kb-toolbar'>"
        "<input id='kb-search' type='text' class='case-search kb-search' "
        "placeholder='Search enum_id or description…'>"
        "<span id='kb-count' class='kb-count'></span>"
        "<span class='lang-switch kb-lang-switch' role='group' aria-label='language'>"
        "<button type='button' class='lang-btn' data-lang='sk'>SK</button>"
        "<button type='button' class='lang-btn active' data-lang='en'>EN</button>"
        "</span>"
        "<button type='button' id='kb-reset' class='kb-reset' "
        "title='Show all KB entries (clear test-case filter)'>Show all</button>"
        "</div>"
        "<div id='kb-case-banner' class='kb-case-banner'></div>"
        f"<div id='kb-list' class='kb-list'>{rows_html}</div>"
        "</div>"
    )


def _build_agents_called_fig(df: pd.DataFrame):
    """Horizontal bar of agents_called values (each row's agent-trajectory)."""
    if "agents_called" not in df.columns:
        return None, 0
    seqs = []
    for v in df["agents_called"]:
        items = _parse_list(v)
        if items:
            seqs.append(" → ".join(str(x) for x in items))
        else:
            seqs.append("(none)")
    vc = Counter(seqs)
    if not vc:
        return None, 0
    items = vc.most_common(15)
    y_labels = [k for k, _ in items][::-1]
    x_counts = [n for _, n in items][::-1]
    fig = go.Figure(go.Bar(
        x=x_counts, y=y_labels, orientation="h",
        marker=dict(color=GE_BLUE, line=dict(width=0), cornerradius=6),
        text=x_counts, textposition="outside", cliponaxis=False,
        textfont=dict(family=GE_FONT, size=11, color=GE_TEXT),
        hovertemplate="<b>%{y}</b><br>%{x} rows<extra></extra>",
    ))
    _style_fig(fig, height=max(260, 30 * len(items) + 80))
    fig.update_layout(margin=dict(t=20, b=40, l=220, r=40), bargap=0.35,
                      xaxis_title="count", yaxis_title="")
    if x_counts:
        fig.update_xaxes(range=[0, max(x_counts) * 1.12])
    fig.update_yaxes(tickfont=dict(family="monospace", size=11, color=GE_TEXT))
    return fig, len(vc)


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
.report-header h1 { font-size: 18px; font-weight: 700; }
.header-meta { display: flex; gap: 18px; font-size: 12px; color: rgba(255,255,255,.92); flex-wrap: wrap; }
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
#case-detail[data-lang="en"] .lang-sk,
#case-detail[data-lang="sk"] .lang-en,
#tab-kb[data-lang="en"] .lang-sk,
#tab-kb[data-lang="sk"] .lang-en { display: none; }
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
.kb-recall-card .hc-stats-row { display: flex; flex-direction: column;
                                  gap: 14px; margin-top: 8px; }
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
/* KB Recall stacked variant — each stat sits in its own row, left-aligned.
   Horizontal divider between the two so they read as separate metrics. */
.kb-recall-card .hc-stat { padding: 0; text-align: left; min-width: 0; }
.kb-recall-card .hc-stat + .hc-stat { padding-top: 12px;
                                        border-top: 1px solid #e4eaf0; }
.rel2-stats-card .hc-stat-label,
.test-cases-stats-card .hc-stat-label,
.kb-recall-card .hc-stat-label { font-size: 10px; color: #5c7999;
                                   text-transform: uppercase;
                                   letter-spacing: .08em; font-weight: 600;
                                   margin-bottom: 4px; }
.rel2-stats-card .hc-stat-value,
.test-cases-stats-card .hc-stat-value,
.kb-recall-card .hc-stat-value { font-size: 38px; font-weight: 700;
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
  return `<span class="fm-badge fm-${esc(key)}" title="primary failure mode">${esc(label)}</span>`;
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
});

const listEl = document.getElementById("cases-list");
const countEl = document.getElementById("list-count");
const searchEl = document.getElementById("case-search");
const detailEl = document.getElementById("case-detail");
// Language toggle: "en" (default) or "sk". Scoped to the case-detail panel
// (Test Cases tab); the toggle button lives in the case-detail header.
let currentLang = "en";
detailEl.dataset.lang = currentLang;
function applyLang(lang) {
  if (lang !== "sk" && lang !== "en") return;
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
    if (f === "pass" && !(c.weighted_avg != null && c.weighted_avg >= PASS_THRESHOLD)) return false;
    if (f === "fail" && !(c.weighted_avg != null && c.weighted_avg < PASS_THRESHOLD)) return false;
    if (f === "kb_scope" && c.scope !== "kb") return false;
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
      const isPass = c.weighted_avg >= PASS_THRESHOLD;
      passTag = `<span class="badge ${isPass ? 'badge-good' : 'badge-bad'}" title="judge weighted_avg ${c.weighted_avg.toFixed(2)} ${isPass ? '≥' : '<'} ${PASS_THRESHOLD}">${isPass ? 'PASS' : 'FAIL'}</span>`;
    }
    const csVal = (c.case_scope || "").toString();
    const csCls = csVal ? ("cs-" + (["kb","kb_and_api","api","out_of_scope","ambiguous"].includes(csVal) ? csVal : "other")) : "";
    const csTag = csVal ? `<span class="cs-badge ${csCls}" title="case_scope (judge): ${esc(csVal)}">${esc(csVal)}</span>` : "";
    li.innerHTML = `<span class="case-id">${c.id}</span>` +
                   csTag +
                   `<span class="scope-badge ${scopeCls}" title="query_scope (graph): ${esc(c.scope || "")}">${c.scope || ""}</span>` +
                   agentBadge(c.last_agent) +
                   passTag +
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
    activeFilters.dim_pairs = [];
    activeFilters.missed_enum = null;
    activeFilters.case_id_set = null;
    activeFilters.case_id_label = null;
    activeFilters.failure_modes.clear();
    ["rel2-min","rel2-max","wavg-min","wavg-max"].forEach(id => {
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
    return "enum-chip expected-row";
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

// Render a SK/EN pair inside a .body container. The #case-detail[data-lang]
// attribute drives which one is visible via CSS.
function bodyLangPair(sk, en, opts) {
  opts = opts || {};
  const mono = opts.mono ? " mono" : "";
  const skRaw = sk == null ? "" : String(sk).trim();
  const enRaw = en == null ? "" : String(en).trim();
  const skBody = skRaw
    ? esc(skRaw)
    : '<em class="lang-fallback">(no SK text)</em>';
  const enBody = enRaw
    ? esc(enRaw)
    : (skRaw
        ? `${esc(skRaw)}<span class="lang-fallback">(EN missing — showing SK)</span>`
        : '<em class="lang-fallback">(empty)</em>');
  return `<div class="body${mono} lang-sk" lang="sk">${skBody}</div>` +
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

function scoreStripHtml(c, suggHtml) {
  const wa = c.weighted_avg;
  const waStr = wa == null ? "–" : wa.toFixed(2);
  const waCls = summaryClass(wa, [0.5, PASS_THRESHOLD]);
  const exp = c.expert_score;
  const expStr = exp == null ? "–" : exp.toFixed(1);
  const expCls = summaryClass(exp, [4, 7]);
  const r2 = c.rel2_score;
  const r2Str = r2 == null ? "–" : r2.toFixed(2);
  const r2Cls = summaryClass(r2, [0.5, 0.8]);
  const suggBlock = suggHtml
    ? `<div class="score-strip-sugg"><div class="score-strip-sugg-title">Improvement suggestions</div>${suggHtml}</div>`
    : "";
  return `<div class="score-strip">` +
    `<div class="score-strip-row">` +
    `<div class="score-box ${waCls}"><span class="sb-accent"></span>
       <div class="sb-label">Judge w.avg (0–1)</div><div class="sb-value">${waStr}</div></div>` +
    `<div class="score-box ${expCls}"><span class="sb-accent"></span>
       <div class="sb-label">Expert (1–10)</div><div class="sb-value">${expStr}</div></div>` +
    `<div class="score-box ${r2Cls}"><span class="sb-accent"></span>
       <div class="sb-label">Rel2 (0–1)</div><div class="sb-value">${r2Str}</div></div>` +
    `</div>` +
    suggBlock +
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
  // Pass / fail tag — fail when the judge weighted_avg is below PASS_THRESHOLD.
  let passTag = "";
  if (c.weighted_avg != null) {
    const isPass = c.weighted_avg >= PASS_THRESHOLD;
    passTag = `<span class="badge ${isPass ? 'badge-good' : 'badge-bad'}" title="judge weighted_avg ${c.weighted_avg.toFixed(2)} ${isPass ? '≥' : '<'} ${PASS_THRESHOLD}">${isPass ? 'PASS' : 'FAIL'}</span>`;
  }
  // SK / EN switch lives on the right side of the case-detail header.
  const skCls = currentLang === "sk" ? " active" : "";
  const enCls = currentLang === "en" ? " active" : "";
  const langSwitch = `<span class="lang-switch case-detail-lang" role="group" aria-label="language">` +
                     `<button type="button" class="lang-btn${skCls}" data-lang="sk">SK</button>` +
                     `<button type="button" class="lang-btn${enCls}" data-lang="en">EN</button>` +
                     `</span>`;
  detailEl.dataset.lang = currentLang;
  detailEl.innerHTML = `
    <h2 class="case-detail-title" style="font-size:16px;margin-bottom:8px">
      <span class="case-detail-title-main">${esc(c.id)} · <span class="scope-badge ${scopeCls}">${esc(c.scope)}</span>${c.last_agent ? ` · ${agentBadge(c.last_agent)}` : ""}${passTag ? ` · ${passTag}` : ""}${c.excluded_reason ? ` · ${exclusionBadge(c.excluded_reason)}` : ((c.failure_mode && c.failure_mode !== "pass") ? ` · ${fmBadge(c.failure_mode)}` : "")}${c.trace_id ? ` · <span class="trace-id" title="trace_id">trace: <code>${esc(c.trace_id)}</code></span>` : ""}</span>
      ${langSwitch}
    </h2>
    <div style="margin-bottom:10px">${routeChips(c)}</div>
    ${scoreStripHtml(c, sugg)}
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
  // Pre-parse data-ids per row for the case filter (keeps update() cheap).
  const rowIds = rows.map(r => {
    try { return new Set((JSON.parse(r.dataset.ids || "[]")).map(String)); }
    catch (_) { return new Set(); }
  });

  function update() {
    const q = (search.value || "").trim().toLowerCase();
    let shown = 0;
    rows.forEach((r, i) => {
      let visible = true;
      if (q && !(r.dataset.search || "").includes(q)) visible = false;
      if (visible && kbCaseFilter && !rowIds[i].has(String(kbCaseFilter))) visible = false;
      r.style.display = visible ? "" : "none";
      if (visible) shown++;
    });
    const tail = kbCaseFilter ? ` · case ${kbCaseFilter}` : "";
    if (counter) counter.textContent = `showing ${shown} of ${rows.length}${tail}`;
    if (reset) reset.classList.toggle("visible", !!kbCaseFilter);
    if (banner) {
      if (kbCaseFilter) {
        banner.innerHTML = `Filtered to KB entries used by <strong>${esc(kbCaseFilter)}</strong>. ` +
                            `Click <em>Show all</em> to see every entry.`;
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


def render_html(df: pd.DataFrame, *, df_all: pd.DataFrame | None = None,
                yaml_name: str, reasoning_effort: str,
                checkpoint_label: str, judge_model: str = "unknown",
                experiment_name: str | None = None) -> str:
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

    # _build_figs returns (fig_dim, fig_hist, fig_rc, fig_missed). Only the
    # last one is still rendered after the Triage and Root Causes tabs were
    # removed; the others are kept inside _build_figs for now.
    _, _, _, fig_missed = _build_figs(df, reranker_miss, retriever_miss)
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
    cases_payload = sorted(
        (_case_payload(r) for _, r in df_all.iterrows()),
        key=lambda c: _tid_key(c.get("id", "")),
    )
    js = (JS.replace("__CASES__", json.dumps(cases_payload))
            .replace("__DIM_NAMES__", json.dumps(list(DIMENSION_WEIGHTS.keys())))
            .replace("__PASS_THRESHOLD__", json.dumps(PASS_THRESHOLD)))


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
    )
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

    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>SKKB – {_h(yaml_name)}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{CSS}</style></head>
<body>

<div class="sticky-top">
<header class="report-header">
  <div class="header-inner">
    <h1>SKKB Evaluation Report</h1>
    <div class="header-meta">
      <span>Experiment: <strong>{_h(experiment_name)}</strong></span>
      <span>Judge model: <strong>{_h(judge_model)}</strong></span>
      <span>Reasoning: <strong>{_h(reasoning_effort)}</strong></span>
      <span>Test Cases: <strong>{len(df)}</strong></span>
      <span>KB version: <code>{_h(kb_version_label)}</code></span>
      <span>Checkpoint: <code>{_h(checkpoint_label)}</code></span>
    </div>
  </div>
</header>

<nav class="tab-nav">
  <div class="tab-nav-inner">
    <button class="tab-btn tab-home active" data-target="tab-summary">Summary</button>
    <button class="tab-btn" data-target="tab-cases">Test Cases</button>
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
    <div class="headline-row headline-row-3">
      <div class="headline-card test-cases-stats-card">
        <div class="hc-label">Test cases {_info_icon(
            "Total cases in the source checkpoint vs. cases with a non-empty "
            "user_query. Empty-query rows are excluded from every metric "
            "in the report — that's the only Stage-1 filter. KB Recall and "
            "the Stage-recall funnel apply a tighter internal filter "
            "(query_scope == 'kb' AND expected_enums non-empty) for the "
            "metrics that genuinely need the reranker to have run.")}
        </div>
        <div class="hc-stats-row">
          <div class="hc-stat">
            <div class="hc-stat-label">Total</div>
            <div class="hc-stat-value">{summary_metrics['n_total']}</div>
          </div>
          <div class="hc-stat">
            <div class="hc-stat-label">Non-empty</div>
            <div class="hc-stat-value">{summary_metrics['n_total'] - summary_metrics['n_empty_query']}</div>
          </div>
        </div>
        <div class="hc-detail">{summary_metrics['n_empty_query']} excluded for empty <code>user_query</code>.</div>
      </div>
      <div class="headline-card">
        <div class="hc-label">Pass rate · all cases {_info_icon(
            f"Share of cases whose primary failure_mode is 'pass' — i.e. "
            f"weighted_avg ≥ {PASS_THRESHOLD}, no weight-2 scorer at 0, "
            f"AND no failure mode (test-set issue, naming mismatch, "
            f"agent-side issue) fired ahead of it. The weight-2 veto "
            f"prevents a single critical-dimension zero from hiding "
            f"behind a high mean. Denominator is n_eval (all analyzed "
            f"cases, including test-set issues).")}
        </div>
        <div class="hc-value">{_fmt_pct(summary_metrics['pass_rate_all'])}</div>
        <div class="hc-detail">{summary_metrics['n_pass']} / {summary_metrics['n_eval']} cases · (includes test-set issues; excludes empty queries).</div>
      </div>
      <div class="headline-card">
        <div class="hc-label">Pass rate · excl. test-set issues {_info_icon(
            "Same definition as the all-cases pass rate, but the denominator drops cases the judge "
            "flagged with expected_reference_looks_wrong=True or classified as ambiguous/out_of_scope, "
            "plus deterministic ENUM-naming-mismatch cases. Measures the agent against trustworthy "
            "ground truth only — matches the Pass row in the failure-modes table directly.")}
        </div>
        <div class="hc-value">{_fmt_pct(summary_metrics['pass_rate_clean'])}</div>
        <div class="hc-detail">{summary_metrics['n_pass_clean']} / {summary_metrics['n_clean']} cases · {summary_metrics['n_defect']} test-set-issue rows excluded.</div>
      </div>
    </div>

    <div class="headline-row headline-row-recall-rel2">
      <div class="headline-card kb-recall-card">
        <div class="hc-label">KB Recall {_info_icon(
            "Micro-averaged recall over the whole run: Σ TP / Σ expected, "
            "where TP is the count of expected ENUMs that appeared in the "
            "reranker's final selection and Σ expected is the sum of "
            "|expected_enums| across qualifying cases. Restricted to cases "
            "where the search tool was used (query_scope == 'kb') AND the "
            "deterministic check did NOT flag an ENUM-naming-mismatch "
            "(those are test-set issues, not agent failures). The Stage "
            "funnel beside it uses the same basis at each pipeline stage — "
            "its reranked row equals this headline number.")}
        </div>
        <div class="kb-recall-row">
          <div class="kb-recall-left">
            <div class="hc-stats-row">
              <div class="hc-stat">
                <div class="hc-stat-label">Recall</div>
                <div class="hc-stat-value">{_fmt_pct(summary_metrics['dataset_recall'])}</div>
                <div class="hc-stat-detail">TP {summary_metrics['dataset_recall_tp']} / {summary_metrics['dataset_recall_total_expected']} expected</div>
              </div>
              <div class="hc-stat">
                <div class="hc-stat-label">Precision</div>
                <div class="hc-stat-value">{_fmt_pct(summary_metrics['dataset_precision'])}</div>
                <div class="hc-stat-detail">TP {summary_metrics['dataset_recall_tp']} / {summary_metrics['dataset_recall_total_reranked']} selected</div>
              </div>
            </div>
          </div>
          <div class="kb-recall-right">
            <div class="hc-funnel-title">Stage funnel — recall &amp; precision {_info_icon(
                "Micro-averaged recall (Σ TP / Σ expected) and precision (Σ TP / "
                "Σ selected) of expected_enums at each pipeline stage. Same basis "
                "as KB Recall on the left: query_scope == 'kb' AND no ENUM-naming "
                "mismatch. Recall trends down through the pipeline (you can only "
                "lose gold ENUMs as the set shrinks); precision trends up (later "
                "stages drop noise). The reranked row equals the KB Recall and KB "
                "Precision headlines on the left.")}
            </div>
            {_funnel_html(summary_metrics['funnel'])}
          </div>
        </div>
      </div>
      <div class="headline-card rel2-stats-card">
        <div class="hc-label">Rel2 score (search tool used) {_info_icon(
            "Upstream semantic-overlap metric between expected_enums and the "
            "system's selected ENUM IDs over cases where the search tool was "
            "actually used (query_scope == 'kb'). Cases the deterministic "
            "check flagged as ENUM-naming-mismatch are excluded — those are "
            "test-set issues (gold ENUM IDs use a different naming convention "
            "than the KB), not agent failures, so they shouldn't drag down "
            "the score. Range [0, 1]; higher is better. See the Doc tab for "
            "the exact computation.")}
        </div>
        <div class="hc-stats-row">
          <div class="hc-stat">
            <div class="hc-stat-label">Mean</div>
            <div class="hc-stat-value">{_fmt_score(summary_metrics['rel2_mean'])}</div>
          </div>
          <div class="hc-stat">
            <div class="hc-stat-label">Median</div>
            <div class="hc-stat-value">{_fmt_score(summary_metrics['rel2_median'])}</div>
          </div>
          <div class="hc-stat">
            <div class="hc-stat-label">STDEV</div>
            <div class="hc-stat-value">{_fmt_score(summary_metrics['rel2_std'])}</div>
          </div>
        </div>
        <div class="hc-detail">
          n = {summary_metrics['rel2_n']} of {summary_metrics['n_eval']} cases (the rest didn't reach the reranker).
          {(f"<br>Excluded {summary_metrics['rel2_naming_excluded']} ENUM-naming-mismatch case{'s' if summary_metrics['rel2_naming_excluded'] != 1 else ''}." if summary_metrics['rel2_naming_excluded'] else "")}
        </div>
      </div>
    </div>

    {_top_failures_html(summary_metrics['top_failures_clean'], summary_metrics['n_clean'])}

    <div class="card">
      <div class="card-title">Judge scorers {_info_icon(
          "The seven judge dimensions (scorers)."
          "Each card shows a stacked bar (red = fail, amber = partial, green = pass)."
          "Click-through links to drill into the cases that scored at each level.")}
      </div>
      {_dimension_cards_html(summary_metrics['dimensions'], dim_full_info)}
    </div>

    <div class="card">
      <div class="card-title">Failure modes — primary cause per case {_info_icon(
          "Every test case is assigned exactly one primary failure mode. Click any count to filter the "
          "Test Cases tab to those rows. See the Doc tab for the full taxonomy.")}
      </div>
      <p style="font-size:12px;color:#5c7999;margin-bottom:10px">
        Priority order: test-set issue → wrong agent routing → retrieval / pruning / reranker losses → 
        pool content gap → context-use / hallucination / language drift → pass / other.
        Multiple secondary issues may also be present per case — drill into individual cases for the full picture.
      </p>
      {_failure_mode_table_html(summary_metrics)}
    </div>

    <div class="card foldable collapsed">
      <div class="card-title">Failure-mode co-occurrence {_info_icon(
          "Symmetric matrix of how often each pair of failure modes fires "
          "for the same case (using failure_modes_all — every mode that "
          "applied to a row, not just the priority winner). Diagonal cells = "
          "total cases that mode fired in. Off-diagonal cells = cases where "
          "both modes fired together. Click any non-zero cell to filter the "
          "Test Cases tab to those rows. Modes from the failure-mode table "
          "are shown starting at 'Test-set issue' (pass is omitted).")}
      </div>
      <div class="card-body">
        <p style="font-size:12px;color:#5c7999;margin-bottom:10px">
          This matrix uses <code>failure_modes_all</code> — every applicable
          failure mode per case, not just the priority winner shown in the
          failure-modes table. So a case primarily classified as
          <em>test-set issue</em> that also triggered <em>hallucination</em>
          appears in both diagonal cells and in the off-diagonal pair.
          Use it to spot "clusters" — e.g. if Test-set issue occurs across
          the whole row, the test set is producing noise that's confounding
          several other failure types.
        </p>
        {_plot(fig_fm_cooccurrence, div_id="fm-cooccurrence-chart") if fig_fm_cooccurrence is not None else "<p class='placeholder'>not enough data to compute co-occurrence</p>"}
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
                <span class="filter-label">Failure mode</span>
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

  <div id="tab-dims" class="tab-panel">
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
        <div class="card-title">Stage funnel — recall &amp; precision {_info_icon(
            "Micro-averaged recall (Σ TP / Σ expected) and precision (Σ TP / Σ selected) "
            "of the expected ENUMs at each pipeline stage: "
            "pre-prune (vector DB output) → post-prune (after dedup/filtering) → reranked (final selection). "
            "Recall trends down (you only lose gold ENUMs as the set shrinks); precision trends up "
            "(later stages drop noise). The reranked row equals the KB Recall and KB Precision headlines "
            "on the Summary page. Restricted to cases where the search tool was used (query_scope == 'kb') "
            "AND the deterministic check did NOT flag an ENUM-naming-mismatch (those are test-set issues).")}
        </div>
        {_funnel_html(summary_metrics['funnel'])}
      </div>
    </div>
    <div class="card">
      <div class="card-title">Top missed ENUMs — reranker miss (in pool, not picked) vs retriever miss (not in pool)</div>
      <p style="font-size:11px;color:#537090;margin-bottom:10px">
        Click a bar to jump to the Test Cases tab filtered to the rows that missed that ENUM.
      </p>
      {_plot(fig_missed, div_id="missed-enums-chart") if fig_missed else "<p class='placeholder'>no missed-enum data</p>"}
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
    {_doc_tab_html(summary_metrics)}
  </div>

</main>

<script>{js}</script>
</body></html>
"""


# ── CLI ──────────────────────────────────────────────────────────────────────
def _load_dimension_descriptions(yaml_name: str) -> dict[str, str]:
    """Extract each dimension's `description: >` block verbatim from the YAML.

    Avoids a pyyaml dependency; parses by tracking indent of the `description:` key
    and folding the subsequent more-indented lines (YAML's `>` style).
    """
    here = Path(__file__).resolve().parent
    cfg = here.parents[0] / "configs" / "skkb" / f"{yaml_name}.yaml"
    if not cfg.exists():
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
    """Parse each dimension's full block from configs/skkb/{yaml_name}.yaml.

    Returns a dict keyed by dimension id, where each value carries:
        name, weight, description, scale (list of {score, label, description}).
    Used to populate the per-scorer info-icon tooltips on the Summary tab
    with the verbatim YAML content.
    """
    here = Path(__file__).resolve().parent
    cfg = here.parents[0] / "configs" / "skkb" / f"{yaml_name}.yaml"
    if not cfg.exists():
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
    """Pull the header-display fields from configs/skkb/{yaml_name}.yaml.

    Returns ``{"name", "judge_model", "reasoning_effort"}``. The yaml has
    several ``name:`` keys (input_fields / dimensions / etc.); we therefore
    extract from the *top-level* ``experiment:`` and ``model:`` blocks only,
    so we don't accidentally pick up an unrelated nested name.
    Avoids a pyyaml dependency for a three-field lookup.
    """
    out = {"name": yaml_name, "judge_model": "unknown", "reasoning_effort": "unknown"}
    here = Path(__file__).resolve().parent
    cfg = here.parents[0] / "configs" / "skkb" / f"{yaml_name}.yaml"
    if not cfg.exists():
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
    return out


def _auto_yaml_name(checkpoint_path: Path) -> str | None:
    """Pick the configs/skkb/<name>.yaml whose stem is the longest prefix of
    the checkpoint stem (after stripping the ``evals_`` prefix). Lets users
    skip ``--yaml-name`` when the checkpoint follows the ``evals_<name>_*``
    convention used by ``hg_ds_evals``.
    """
    here = Path(__file__).resolve().parent
    cfg_dir = here.parents[0] / "configs" / "skkb"
    if not cfg_dir.exists():
        return None
    candidates = sorted(
        (p.stem for p in cfg_dir.glob("*.yaml")),
        key=len, reverse=True,
    )
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
    ap.add_argument("--checkpoint", help="path to judge checkpoint CSV")
    ap.add_argument("--output", help="path to write the HTML report")
    ap.add_argument("--yaml-name", default=None,
                    help="config name (e.g. skkb_exp_001_baseline_no_expected_enums). "
                         "Auto-detected from the checkpoint filename if omitted.")
    ap.add_argument("--reasoning-effort", default="medium",
                    help="used only to derive default checkpoint paths; "
                         "the header reads reasoning_effort directly from the YAML.")
    ap.add_argument("--suffix", default="",
                    help="optional filename suffix (e.g. '_test' for a test run)")
    args = ap.parse_args()

    # If --yaml-name is omitted, try to auto-detect it from the checkpoint
    # filename (so colleagues don't need to remember the exact config name).
    yaml_name = args.yaml_name
    if not yaml_name and args.checkpoint:
        yaml_name = _auto_yaml_name(Path(args.checkpoint))
    if not yaml_name:
        yaml_name = "skkb_exp_001_baseline"  # last-resort fallback

    ckpt_default, out_default = _default_paths(yaml_name, args.reasoning_effort, args.suffix)
    ckpt = Path(args.checkpoint) if args.checkpoint else ckpt_default
    out = Path(args.output) if args.output else out_default

    if not ckpt.exists():
        sys.exit(f"checkpoint not found: {ckpt}")

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
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_str)
    print(f"rows: {len(df)} of {len(df_all)} "
          f"(dropped {n_empty_total} empty user_query)   → {out}   "
          f"({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
