"""Generate a self-contained HTML report for API trace scorer results.

Modeled on the CZKB report's design system (experiments/czkb/notebooks/
czkb_report.py): same color palette, info-icon + tip-bubble tooltips,
sticky tab nav, sidebar with chip filters and foldable groups, score-strip
case detail, active-filter banner. Three scorers (agent_routing,
tool_usage, tool_parameter) replace CZKB's seven judge dimensions.

Usage:
    python api_report.py

    python api_report.py \
        --input traces/traces_enriched_offline_smoke_pr12_kb_smoke_infer.csv \
        --output reports/api_smoke_report.html

    # With a baseline run for regression comparison:
    python api_report.py \
        --input    .../enriched_traces_<CURRENT_RUN_ID>.csv \
        --baseline .../enriched_traces_<PREVIOUS_RUN_ID>.csv \
        --output reports/api_smoke_report.html
"""
from __future__ import annotations

import argparse
import ast
import html
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


SCORERS = ("agent_routing", "tool_usage", "tool_parameter")
BINARY_SCORERS = ("agent_routing", "tool_usage")

# Tools whose ``parameters`` are not meaningfully comparable by the
# deterministic tool_parameter scorer (fuzzy semantic strings — search
# queries, expected facets, question text). Mirrors the notebook's
# ``EXCLUDE_PARAM_TOOLS`` so the report doesn't display a stale params
# mean computed from the one mixed row that still carries the scorer.
EXCLUDE_PARAM_TOOLS = {"knowledge_search"}

SCORER_LABEL = {
    "agent_routing": "Agent routing",
    "tool_usage": "Tool usage",
    "tool_parameter": "Tool parameter",
}


# ─── Utility functions ───────────────────────────────────────────────────

def _h(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return html.escape(str(value), quote=True)


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _short(value: Any, limit: int = 160) -> str:
    text = " ".join(_safe_str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _parse_obj(value: Any) -> Any:
    """Parse JSON/Python-literal cells emitted by the notebook parser."""
    if isinstance(value, (list, dict)):
        return value
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def _as_list(value: Any) -> list[Any]:
    parsed = _parse_obj(value)
    if parsed is None:
        return []
    if isinstance(parsed, list):
        return parsed
    return [parsed]


def _as_int_or_none(value: Any) -> int | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return int(f)


def _as_dict(value: Any) -> dict[str, Any]:
    parsed = _parse_obj(value)
    return parsed if isinstance(parsed, dict) else {}


def _json_pretty(value: Any, limit: int | None = None) -> str:
    parsed = _parse_obj(value)
    if parsed is None:
        text = ""
    elif isinstance(parsed, (dict, list)):
        text = json.dumps(parsed, ensure_ascii=False, indent=2, default=str)
    else:
        text = str(parsed)
    if limit is not None and len(text) > limit:
        return text[: limit - 1].rstrip() + "..."
    return text


def _norm_tool(name: Any) -> str:
    return _safe_str(name).strip().lower()


def _tool_names(value: Any) -> list[str]:
    names: list[str] = []
    for item in _as_list(value):
        if isinstance(item, dict):
            name = item.get("tool") or item.get("name")
            if name:
                names.append(str(name))
        elif item:
            names.append(str(item))
    return names


def _agent(value: Any) -> str:
    return _safe_str(value).strip()


def _fmt_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "–"
    return f"{float(value):.1%}"


def _fmt_num(value: float | int | None, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "–"
    return f"{float(value):.{digits}f}"


def _status_counts(series: pd.Series) -> dict[str, int]:
    counts = Counter()
    for value in series.fillna("").astype(str):
        key = value.strip() or "not_scored"
        counts[key] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _status_counts_text(series: pd.Series) -> str:
    counts = _status_counts(series)
    return ", ".join(f"{k}: {v}" for k, v in counts.items())


def _score_columns(df: pd.DataFrame) -> list[str]:
    return [f"{s}_score" for s in SCORERS if f"{s}_score" in df.columns]


# ─── Data enrichment ─────────────────────────────────────────────────────

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for score_col in _score_columns(out):
        out[score_col] = pd.to_numeric(out[score_col], errors="coerce")

    # Parser contract: these columns must exist on the input CSV. Direct
    # access (no .get default) so a missing column raises a clear KeyError
    # at enrich time rather than silently producing an empty UI.
    out["_expected_agent"] = out["expected_agent"].apply(_agent)
    out["_actual_agent"] = out["actual_agent"].apply(_agent)
    out["_expected_tools"] = out["expected_tool_calls"].apply(_tool_names)
    out["_actual_tools"] = out["actual_tool_calls"].apply(_tool_names)
    # Cached expected-norm column: also read by _tool_frequency_table to mask
    # per-tool subsets. The actual side is only needed for the seq-match
    # comparison so it's computed inline.
    out["_expected_tool_seq_norm"] = out["_expected_tools"].apply(lambda xs: [_norm_tool(x) for x in xs])
    actual_tool_seq_norm = out["_actual_tools"].apply(lambda xs: [_norm_tool(x) for x in xs])
    out["_tool_seq_match"] = out["_expected_tool_seq_norm"] == actual_tool_seq_norm

    out["_agent_routing_pass"] = out.get("agent_routing_score", pd.Series(index=out.index)).eq(1)
    out["_tool_usage_pass"] = out.get("tool_usage_score", pd.Series(index=out.index)).eq(1)
    param_score = out.get("tool_parameter_score", pd.Series(index=out.index, dtype="float64"))
    out["_tool_parameter_scored"] = param_score.notna()
    out["_tool_parameter_full_pass"] = param_score.eq(1)
    out["_tool_parameter_partial"] = param_score.gt(0) & param_score.lt(1)
    out["_tool_parameter_fail"] = param_score.eq(0)

    out["_core_pass"] = out["_agent_routing_pass"] & out["_tool_usage_pass"]
    out["_strict_pass"] = out["_core_pass"] & (
        ~out["_tool_parameter_scored"] | out["_tool_parameter_full_pass"]
    )

    score_cols = _score_columns(out)
    out["_score_mean"] = out[score_cols].mean(axis=1, skipna=True) if score_cols else 0.0
    out["_score_min"] = out[score_cols].min(axis=1, skipna=True) if score_cols else 0.0

    def issue_bucket(row: pd.Series) -> str:
        if not row["_agent_routing_pass"]:
            return "agent_routing"
        if not row["_tool_usage_pass"]:
            return "tool_usage"
        if row["_tool_parameter_scored"] and not row["_tool_parameter_full_pass"]:
            return "tool_parameter"
        return "pass"

    out["_issue_bucket"] = out.apply(issue_bucket, axis=1)

    def issue_labels(row: pd.Series) -> list[str]:
        labels: list[str] = []
        if not row["_agent_routing_pass"]:
            labels.append("routing")
        if not row["_tool_usage_pass"]:
            labels.append("tool usage")
        if row["_tool_parameter_scored"] and not row["_tool_parameter_full_pass"]:
            labels.append("tool params")
        return labels

    out["_issue_labels"] = out.apply(issue_labels, axis=1)
    out["_issue_count"] = out["_issue_labels"].apply(len)

    def param_bucket(row: pd.Series) -> str:
        if not row["_tool_parameter_scored"]:
            return "na"
        if row["_tool_parameter_full_pass"]:
            return "pass"
        if row["_tool_parameter_partial"]:
            return "partial"
        return "fail"

    out["_tool_parameter_bucket"] = out.apply(param_bucket, axis=1)
    return out


def load_baseline(path: Path) -> dict[str, dict[str, Any]]:
    """Load a previously-scored enriched-traces CSV and reduce it to a
    per-test-case lookup of strict-pass / issue-bucket.

    Reuses ``enrich`` so the pass criterion is identical to the current
    run (``_strict_pass`` = routing + tool_usage + tool_parameter when
    scored). Duplicate ``test_case_id`` rows keep the last one and print
    a one-line warning so it's visible.
    """
    df = pd.read_csv(path)
    df = enrich(df)
    if "test_case_id" not in df.columns:
        raise KeyError(f"Baseline CSV missing 'test_case_id' column: {path}")
    dup_mask = df.duplicated(subset="test_case_id", keep="last")
    if int(dup_mask.sum()):
        print(f"[baseline] {int(dup_mask.sum())} duplicate test_case_id rows in "
              f"{path.name}; keeping last occurrence each.")
    lookup: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        tcid = _safe_str(row.get("test_case_id"))
        if not tcid:
            continue
        lookup[tcid] = {
            "strict_pass": bool(row.get("_strict_pass")),
            "issue_bucket": _safe_str(row.get("_issue_bucket")),
        }
    return lookup


def scorer_summary(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scorer in SCORERS:
        score_col = f"{scorer}_score"
        status_col = f"{scorer}_status"
        if score_col not in df.columns:
            continue
        values = pd.to_numeric(df[score_col], errors="coerce")
        scored = int(values.notna().sum())
        row: dict[str, Any] = {
            "scorer": scorer,
            "scored": scored,
            "not_scored": int(values.isna().sum()),
            "status_counts": _status_counts_text(df.get(status_col, pd.Series(index=df.index))),
        }
        if scorer in BINARY_SCORERS:
            passed = int(values.eq(1).sum())
            failed = int(values.eq(0).sum())
            row.update(
                kind="binary",
                passed=passed,
                failed=failed,
                partial=0,
                mean=values.mean(),
                pass_rate=(passed / scored if scored else None),
            )
        else:
            passed = int(values.eq(1).sum())
            partial = int(values.gt(0).where(values.lt(1), False).sum())
            failed = int(values.eq(0).sum())
            row.update(
                kind="numeric",
                passed=passed,
                failed=failed,
                partial=partial,
                mean=values.mean(),
                pass_rate=(passed / scored if scored else None),
            )
        rows.append(row)
    return rows


def case_payload(df: pd.DataFrame, *,
                 baseline: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    # Sort by the numeric portion of test_case_id ascending (smoke-1, smoke-2, …)
    # so the sidebar reads in dataset order rather than failure-first order.
    def _tcid_key(value: Any) -> tuple[int, str]:
        s = _safe_str(value)
        m = re.search(r"\d+", s)
        return (int(m.group()) if m else 10**9, s)

    df = df.assign(_tcid_key=df["test_case_id"].apply(_tcid_key))
    sorted_df = df.sort_values(
        by="_tcid_key",
        ascending=True,
        kind="stable",
    )
    for _, row in sorted_df.iterrows():
        scores = {}
        statuses = {}
        rationales = {}
        for scorer in SCORERS:
            score = row.get(f"{scorer}_score")
            scores[scorer] = None if pd.isna(score) else float(score)
            statuses[scorer] = _safe_str(row.get(f"{scorer}_status"))
            rationales[scorer] = _safe_str(row.get(f"{scorer}_rationale"))

        # Per-row scorer plan from the test-case definition (scorers_to_run).
        # Lets the UI distinguish "scorer not configured for this case" from
        # "scorer configured but produced no value" (the latter is a bug).
        configured_raw = _as_list(row.get("scorers_to_run"))
        configured_scorers = sorted({
            str(s).strip() for s in configured_raw if str(s).strip()
        })

        # tool_usage metadata (lists of tool names, or empty if not scored)
        tool_class = {
            "correct":          _as_list(row.get("tool_usage_correct")),
            "incorrect":        _as_list(row.get("tool_usage_incorrect")),
            "hallucinated":     _as_list(row.get("tool_usage_hallucinated")),
            "missing_expected": _as_list(row.get("tool_usage_missing_expected")),
        }
        # tool_parameter metadata
        def _num_or_none(v):
            try:
                f = float(v)
                return None if pd.isna(f) else f
            except (TypeError, ValueError):
                return None
        param_breakdown = {
            "key_score":       _num_or_none(row.get("tool_parameter_key_score")),
            "value_score":     _num_or_none(row.get("tool_parameter_value_score")),
            "expected_keys":   _num_or_none(row.get("tool_parameter_expected_keys")),
            "matched_keys":    _num_or_none(row.get("tool_parameter_matched_keys")),
            "correct_values":  _num_or_none(row.get("tool_parameter_correct_values")),
            "wrong_values":    _num_or_none(row.get("tool_parameter_wrong_values")),
            "missing_keys":    _num_or_none(row.get("tool_parameter_missing_keys")),
        }
        # Per-entry param breakdown — one record per expected tool call,
        # with matched / wrong / missing_keys lists. JSON-serialized in the
        # CSV (cell may be missing on older runs).
        per_entry = _as_list(row.get("tool_parameter_per_entry"))
        extra_by_tool_raw = _as_dict(row.get("tool_parameter_extra_by_tool"))
        # Coerce values to ints; drop zero entries so the case-detail badge list is tight.
        extra_by_tool: dict[str, int] = {}
        for k, v in extra_by_tool_raw.items():
            try:
                n = int(v)
            except (TypeError, ValueError):
                continue
            if n > 0:
                extra_by_tool[str(k)] = n

        # Span-level errors (emitted by parse_trace_mlflow → enriched CSV).
        # info.state stays "OK" even when child spans crash, so without
        # these columns the report would silently hide ConnectTimeout /
        # CancelledError / etc. Missing on older CSVs → empty list.
        span_errors = _as_list(row.get("span_errors_json"))
        span_error_types = _as_list(row.get("span_error_types_json"))
        has_span_error_raw = row.get("trace_has_span_error")
        if isinstance(has_span_error_raw, bool):
            has_span_error = has_span_error_raw
        elif has_span_error_raw is None or (
            isinstance(has_span_error_raw, float) and math.isnan(has_span_error_raw)
        ):
            has_span_error = bool(span_errors)
        else:
            has_span_error = str(has_span_error_raw).strip().lower() in {"true", "1", "1.0", "yes"}
        span_error_count_raw = row.get("span_error_count")
        try:
            span_error_count = int(span_error_count_raw)
        except (TypeError, ValueError):
            span_error_count = len(span_errors)

        # Baseline comparison fields. When no baseline is provided every
        # case gets `prev_known=False` and all comparison flags `False`,
        # which the JS reads as "no comparison loaded" (no chips, no
        # badge, no detail banner).
        tcid_str = _safe_str(row.get("test_case_id"))
        strict_pass_now = bool(row.get("_strict_pass"))
        if baseline is not None and tcid_str in baseline:
            prev = baseline[tcid_str]
            prev_known = True
            prev_pass = bool(prev["strict_pass"])
            prev_issue_bucket = str(prev.get("issue_bucket") or "")
        else:
            prev_known = False
            prev_pass = None
            prev_issue_bucket = ""
        regression = (baseline is not None) and prev_known and prev_pass and (not strict_pass_now)
        persistent_failure = (baseline is not None) and prev_known and (not prev_pass) and (not strict_pass_now)
        fixed = (baseline is not None) and prev_known and (not prev_pass) and strict_pass_now
        new_case = (baseline is not None) and (not prev_known)

        cases.append(
            {
                "id": _safe_str(row.get("test_case_id")),
                "trace_id": _safe_str(row.get("trace_id")),
                "session_id": _safe_str(row.get("session_id")),
                "domain": _safe_str(row.get("eval_domain")),
                "persona": _safe_str(row.get("eval_persona")),
                "state": _safe_str(row.get("state")),
                "has_span_error": has_span_error,
                "span_error_count": span_error_count,
                "span_error_types": span_error_types,
                "span_errors": span_errors,
                "query": _safe_str(row.get("user_query")),
                "query_en": _safe_str(row.get("user_query_en")),
                "expected_agent": row.get("_expected_agent", ""),
                "actual_agent": row.get("_actual_agent", ""),
                "expected_tools": row.get("_expected_tools", []),
                "actual_tools": row.get("_actual_tools", []),
                "tool_seq_match": bool(row.get("_tool_seq_match")),
                "scores": scores,
                "statuses": statuses,
                "rationales": rationales,
                "core_pass": bool(row.get("_core_pass")),
                "strict_pass": bool(row.get("_strict_pass")),
                "issue_bucket": _safe_str(row.get("_issue_bucket")),
                "issue_labels": row.get("_issue_labels", []),
                "param_bucket": _safe_str(row.get("_tool_parameter_bucket")),
                "score_mean": float(row.get("_score_mean", 0) or 0),
                "score_min": float(row.get("_score_min", 0) or 0),
                "guidelines": _as_list(row.get("guidelines")),
                "expected_tool_calls": _as_list(row.get("expected_tool_calls")),
                "actual_tool_calls": _as_list(row.get("actual_tool_calls")),
                "expected_tool_calls_pretty": _json_pretty(row.get("expected_tool_calls")),
                "actual_tool_calls_pretty": _json_pretty(row.get("actual_tool_calls"), limit=12000),
                "expected_response": _safe_str(row.get("expected_response")),
                "actual_response": _safe_str(row.get("actual_response")),
                # English translation of actual_response, when available
                # (notebook step writes this column). Empty string falls
                # back gracefully — the UI only shows the EN toggle when
                # this is non-empty.
                "actual_response_en": _safe_str(row.get("actual_response_en")),
                "available_tools": _as_list(row.get("available_tools")),
                "agents_path": _as_list(row.get("actual_agents_path")),
                "tool_classification": tool_class,
                "param_breakdown": param_breakdown,
                "param_per_entry": per_entry,
                "extra_by_tool": extra_by_tool,
                "configured_scorers": configured_scorers,
                # Per-call list of parameter keys excused from the
                # tool_parameter scorer by the notebook's relaxation
                # rules (e.g. ``size``, ``visualization_type``). Same
                # length & order as expected_tool_calls / actual_tool_calls.
                # Old CSVs without these columns get empty lists.
                "expected_excused": _coerce_excused(
                    row.get("tool_parameter_expected_excused"),
                    len(row.get("_expected_tools") or []),
                ),
                "actual_excused": _coerce_excused(
                    row.get("tool_parameter_actual_excused"),
                    len(row.get("_actual_tools") or []),
                ),
                # KB-retrieval ENUM stages (populated only for KB / KB&API
                # cases; empty lists otherwise). The case-detail UI shows
                # the section when any of these are non-empty.
                "expected_enums":       _as_list(row.get("expected_enums")),
                "pre_prune_enum_ids":   _as_list(row.get("pre_prune_enum_ids")),
                "post_prune_enum_ids":  _as_list(row.get("post_prune_enum_ids")),
                "reranked_enum_ids":    _as_list(row.get("reranked_enum_ids")),
                "pre_prune_enum_count":  _as_int_or_none(row.get("pre_prune_enum_count")),
                "post_prune_enum_count": _as_int_or_none(row.get("post_prune_enum_count")),
                # Baseline comparison (see load_baseline / --baseline CLI flag).
                # All False / None when no baseline was loaded.
                "prev_known": prev_known,
                "prev_pass": prev_pass,
                "prev_issue_bucket": prev_issue_bucket,
                "regression": bool(regression),
                "persistent_failure": bool(persistent_failure),
                "fixed": bool(fixed),
                "new_case": bool(new_case),
            }
        )
    return cases


def _coerce_excused(value: Any, n_calls: int) -> list[list[str]]:
    """Parse the excused-per-call column into a list of lists of str.
    Pads or truncates to ``n_calls`` so the JS can index by call index
    without bounds checks."""
    parsed = _as_list(value)
    out: list[list[str]] = []
    for entry in parsed:
        if isinstance(entry, list):
            out.append([str(x) for x in entry if x])
        else:
            out.append([])
    while len(out) < n_calls:
        out.append([])
    return out[:n_calls] if n_calls else out


def _unique_label(df: pd.DataFrame, column: str) -> str:
    if column not in df.columns:
        return "–"
    values = sorted({str(v) for v in df[column].dropna().unique() if str(v).strip()})
    if not values:
        return "–"
    if len(values) == 1:
        return values[0]
    return f"{len(values)} values"


def compute_metrics(df: pd.DataFrame, *,
                    baseline: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    routing = pd.to_numeric(df.get("agent_routing_score", pd.Series(index=df.index)), errors="coerce")
    usage = pd.to_numeric(df.get("tool_usage_score", pd.Series(index=df.index)), errors="coerce")
    param = pd.to_numeric(df.get("tool_parameter_score", pd.Series(index=df.index)), errors="coerce")
    routing_scored = int(routing.notna().sum())
    usage_scored = int(usage.notna().sum())
    param_scored = int(param.notna().sum())
    routing_pass = int(routing.eq(1).sum())
    usage_pass = int(usage.eq(1).sum())
    param_pass = int(param.eq(1).sum())
    out: dict[str, Any] = {
        "n_total": len(df),
        "routing_scored": routing_scored,
        "routing_pass": routing_pass,
        "routing_pass_rate": (routing_pass / routing_scored) if routing_scored else None,
        "usage_scored": usage_scored,
        "usage_pass": usage_pass,
        "usage_pass_rate": (usage_pass / usage_scored) if usage_scored else None,
        "param_scored": param_scored,
        "param_pass": param_pass,
        "param_pass_rate": (param_pass / param_scored) if param_scored else None,
        "baseline_loaded": baseline is not None,
    }
    if baseline is not None:
        # Count regression / persistent / fixed / new against the
        # baseline lookup. Strict pass criterion (same as case_payload).
        n_reg = n_persistent = n_fixed = n_new = n_comparable = 0
        for _, row in df.iterrows():
            tcid = _safe_str(row.get("test_case_id"))
            strict_now = bool(row.get("_strict_pass"))
            prev = baseline.get(tcid)
            if prev is None:
                n_new += 1
                continue
            n_comparable += 1
            prev_pass = bool(prev["strict_pass"])
            if prev_pass and not strict_now:
                n_reg += 1
            elif (not prev_pass) and (not strict_now):
                n_persistent += 1
            elif (not prev_pass) and strict_now:
                n_fixed += 1
        out.update(
            n_regression=n_reg,
            n_persistent=n_persistent,
            n_fixed=n_fixed,
            n_new=n_new,
            n_comparable=n_comparable,
        )
    return out


# ─── Info-icon tooltips ──────────────────────────────────────────────────

_INFO_ICON = "&#9432;"


def _info_icon(tip: str) -> str:
    return f'<span class="info-icon" data-tip="{_h(tip)}">{_INFO_ICON}</span>'


ROUTING_PASS_INFO = (
    "Share of cases where agent_routing_score == 1 — the actual "
    "top-level agent matched the expected agent."
)

TOOL_PASS_INFO = (
    "Share of cases where tool_usage_score == 1 — the actual tool-call "
    "sequence exactly matched the expected sequence (ordered, "
    "case-insensitive)."
)

PARAM_PASS_INFO = (
    "Share of cases where tool_parameter_score == 1 — every expected "
    "tool call was made with the correct parameters. Denominator is "
    "rows where the parameter scorer ran (missing-param rows excluded)."
)

SCORER_MODE_INFO = {
    "agent_routing": (
        "Binary (0/1). The actual top-level agent reached by the trace must "
        "equal the expected agent. No partial credit."
    ),
    "tool_usage": (
        "Binary (0/1), exact match (ordered). The sequence of called tool "
        "names must equal the expected sequence, in the same order. Same "
        "set in a different order scores 0; rationales surface as "
        "'Order mismatch'."
    ),
    "tool_parameter": (
        "Numeric in [0, 1]. Per-call parameter match averaged across the "
        "expected tool calls. Only rows with expected_tool_calls are "
        "scored; rows without are reported as not-scored, not as failures."
    ),
}


# ─── Render helpers ──────────────────────────────────────────────────────

def _bar(width_pct: float, color: str = "#1d69ec") -> str:
    pct = max(0.0, min(100.0, width_pct))
    return (
        f"<div class='mini-bar'><span style='width:{pct:.2f}%;"
        f"background:{color}'></span></div>"
    )


def _summary_cards(metrics: dict[str, Any]) -> str:
    # Per-card tooltips now also carry the small "X / Y cases scored"
    # footnote that used to sit visibly under the value (the visible
    # .hc-detail line is gone — the info-icon hover shows it instead).
    cards: list[dict[str, Any]] = [
        {
            "label": "Test cases",
            "value": str(metrics["n_total"]),
            "tip": "Rows in the scored CSV.",
        },
        {
            "label": "Routing pass",
            "value": _fmt_pct(metrics["routing_pass_rate"]),
            "tip": (
                f"{ROUTING_PASS_INFO}\n\n"
                f"{metrics['routing_pass']} / {metrics['routing_scored']} cases scored."
            ),
        },
        {
            "label": "Tool pass",
            "value": _fmt_pct(metrics["usage_pass_rate"]),
            "tip": (
                f"{TOOL_PASS_INFO}\n\n"
                f"{metrics['usage_pass']} / {metrics['usage_scored']} cases scored."
            ),
        },
        {
            "label": "Params pass",
            "value": _fmt_pct(metrics["param_pass_rate"]),
            "tip": (
                f"{PARAM_PASS_INFO}\n\n"
                f"{metrics['param_pass']} / {metrics['param_scored']} cases scored."
            ),
        },
    ]
    if metrics.get("baseline_loaded"):
        n_reg = int(metrics.get("n_regression", 0))
        n_persistent = int(metrics.get("n_persistent", 0))
        n_fixed = int(metrics.get("n_fixed", 0))
        n_new = int(metrics.get("n_new", 0))
        n_comp = int(metrics.get("n_comparable", 0))
        cards.append({
            "label": "Regressions",
            "value": str(n_reg),
            "tip": (
                "Cases that passed in the baseline run but fail in this "
                "run (strict pass = routing + tool_usage + tool_parameter "
                "when scored).\n\n"
                f"{n_reg} regression · {n_persistent} persistent fail · "
                f"{n_fixed} fixed · {n_new} new · {n_comp} comparable.\n"
                "Click the card to filter the Test Cases tab to regressions."
            ),
            "filter_link": "regression",
            "tone": "bad" if n_reg else "good",
        })
    out = []
    for c in cards:
        tip = _info_icon(c["tip"]) if c["tip"] else ""
        if c.get("filter_link"):
            tone_cls = f" hc-{c.get('tone', '')}" if c.get("tone") else ""
            out.append(
                f"<a href='#' class='headline-card hc-clickable{tone_cls}' "
                f"data-compare-filter='{_h(c['filter_link'])}'>"
                f"<div class='hc-label'>{_h(c['label'])} {tip}</div>"
                f"<div class='hc-value'>{c['value']}</div>"
                "</a>"
            )
        else:
            out.append(
                "<div class='headline-card'>"
                f"<div class='hc-label'>{_h(c['label'])} {tip}</div>"
                f"<div class='hc-value'>{c['value']}</div>"
                "</div>"
            )
    return "".join(out)


def _scorer_count_link(scorer: str, bucket: str, count: int, tone: str) -> str:
    """Render a clickable count linking to Test Cases filtered by scorer+bucket."""
    if count <= 0:
        return f"<td class='num-cell {tone}'>0</td>"
    return (
        f"<td class='num-cell {tone}'>"
        f"<a class='scorer-filter-link' href='#' "
        f"data-scorer='{_h(scorer)}' data-bucket='{_h(bucket)}' "
        f"title='Filter Test Cases to {_h(scorer)} = {_h(bucket)}'>{count}</a>"
        "</td>"
    )


def _scorer_summary_table(rows: list[dict[str, Any]]) -> str:
    body_rows = []
    for row in rows:
        tip = _info_icon(SCORER_MODE_INFO.get(row["scorer"], ""))
        scorer = row["scorer"]
        # Binary scorers (0/1): partial is not a possible outcome, and mean
        # equals pass_rate exactly (same number, different format) — so we
        # show "–" in both columns to avoid implying redundant signal.
        if row["kind"] == "binary":
            partial_cell = "<td class='num-cell rate-na'>–</td>"
            mean_cell = "<td class='num-cell rate-na'>–</td>"
        else:
            partial_cell = _scorer_count_link(scorer, "partial", row["partial"], "mid")
            # Mean is reported on the scorer's native scale ([0, 1] for
            # tool_parameter), not as a percentage — the column is "Mean",
            # not "Pass rate". Pass rate is its own column to the left.
            mean_cell = f"<td class='num-cell'>{_fmt_num(row['mean'], 3)}</td>"
        pass_cell = _scorer_count_link(scorer, "pass", row["passed"], "good")
        fail_cell = _scorer_count_link(scorer, "fail", row["failed"], "bad")
        body_rows.append(
            "<tr>"
            f"<td class='scorer-cell'><div class='scorer-cell-head'>"
            f"<code>{_h(scorer)}</code>{tip}</div>"
            f"<span class='muted-small'>{_h(row['kind'])}</span></td>"
            f"<td class='num-cell'>{row['scored']}</td>"
            f"{pass_cell}"
            f"{partial_cell}"
            f"{fail_cell}"
            f"<td class='num-cell'>{_fmt_pct(row['pass_rate'])}</td>"
            f"{mean_cell}"
            "</tr>"
        )
    return (
        "<table class='tbl'>"
        "<thead><tr>"
        "<th>Scorer</th>"
        "<th class='num-head'>Scored</th>"
        "<th class='num-head'>Pass</th>"
        "<th class='num-head'>Partial</th>"
        "<th class='num-head'>Fail</th>"
        "<th class='num-head'>Pass rate</th>"
        "<th class='num-head'>Mean</th>"
        "</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
        "<p class='card-footnote'><strong>Tool usage scorer mode: exact</strong></p>"
    )


def _issue_table(df: pd.DataFrame) -> str:
    counts = df["_issue_bucket"].value_counts().reindex(
        ["pass", "agent_routing", "tool_usage", "tool_parameter"], fill_value=0
    )
    labels = {
        "pass": "Pass",
        "agent_routing": "Agent routing fail",
        "tool_usage": "Tool usage fail",
        "tool_parameter": "Tool parameter mismatch",
    }
    badge_cls = {
        "pass": "fm-pass",
        "agent_routing": "fm-routing",
        "tool_usage": "fm-usage",
        "tool_parameter": "fm-params",
    }
    bar_color = {
        "pass": "#057f19",
        "agent_routing": "#cf2a1e",
        "tool_usage": "#cf2a1e",
        "tool_parameter": "#f2a91e",
    }
    body_rows = []
    n = len(df)
    for key, count in counts.items():
        rate = count / n if n else 0
        count_html = (
            f"<a class='fm-clear-link' href='#' data-outcome='{_h(key)}'>{int(count)}</a>"
            if count else "0"
        )
        body_rows.append(
            "<tr>"
            f"<td><span class='fm-badge {badge_cls[key]}'>{_h(labels[key])}</span></td>"
            f"<td class='num-cell'>{count_html}</td>"
            f"<td class='num-cell'>{_fmt_pct(rate)}</td>"
            f"<td>{_bar(rate * 100, bar_color[key])}</td>"
            "</tr>"
        )
    return (
        "<table class='tbl'>"
        "<thead><tr>"
        "<th>Primary outcome</th>"
        "<th class='num-head'>Cases</th>"
        "<th class='num-head'>Share</th>"
        "<th>Distribution</th>"
        "</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def _rate_class(value: float | None) -> str:
    """Map a 0–1 rate to a tinted cell class (high / mid / low / n-a)."""
    if value is None or pd.isna(value):
        return "rate-na"
    v = float(value)
    if v >= 0.8:
        return "rate-hi"
    if v >= 0.5:
        return "rate-mid"
    return "rate-lo"


GROUP_COL_TO_FILTER = {
    "eval_domain": "domains",
    "eval_persona": "personas",
}


def _group_table(df: pd.DataFrame, group_col: str, label: str, tip: str | None = None,
                 col_label: str | None = None) -> str:
    if group_col not in df.columns:
        return f"<div class='card'><div class='card-title'>{_h(label)}</div><p class='placeholder'>Column not available.</p></div>"
    agg_spec: dict[str, tuple[str, str]] = {
        "cases": ("test_case_id", "count"),
    }
    if "agent_routing_score" in df.columns:
        agg_spec["routing"] = ("agent_routing_score", "mean")
    if "tool_usage_score" in df.columns:
        agg_spec["usage"] = ("tool_usage_score", "mean")
    has_params = "tool_parameter_score" in df.columns
    if has_params:
        agg_spec["params"] = ("tool_parameter_score", "mean")
    grouped = (
        df.groupby(group_col, dropna=False)
        .agg(**agg_spec)
        .reset_index()
        .sort_values(["cases", group_col], ascending=[False, True])
    )
    filter_group = GROUP_COL_TO_FILTER.get(group_col)
    body_rows = []
    for _, row in grouped.iterrows():
        raw_name = _safe_str(row[group_col])
        name = raw_name or "(blank)"
        routing_val = row["routing"] if "routing" in grouped.columns else None
        usage_val = row["usage"] if "usage" in grouped.columns else None
        routing_cls = _rate_class(routing_val)
        usage_cls = _rate_class(usage_val)
        cases_n = int(row["cases"])
        if filter_group and raw_name:
            cases_cell = (
                f"<a class='group-filter-link' href='#' "
                f"data-group='{_h(filter_group)}' data-value='{_h(raw_name)}' "
                f"title='Filter Test Cases to {_h(raw_name)}'>{cases_n}</a>"
            )
        else:
            cases_cell = str(cases_n)
        if has_params:
            params_cls = _rate_class(row["params"])
            params_cell = f"<td class='num-cell {params_cls}'>{_fmt_pct(row['params'])}</td>"
        else:
            params_cell = "<td class='num-cell rate-na' title='tool_parameter_score not present in this run'>–</td>"
        body_rows.append(
            "<tr>"
            f"<td>{_h(name)}</td>"
            f"<td class='num-cell'>{cases_cell}</td>"
            f"<td class='num-cell {routing_cls}'>{_fmt_pct(routing_val)}</td>"
            f"<td class='num-cell {usage_cls}'>{_fmt_pct(usage_val)}</td>"
            f"{params_cell}"
            "</tr>"
        )
    info = f" {_info_icon(tip)}" if tip else ""
    return (
        "<div class='card'>"
        f"<div class='card-title'>{_h(label)}{info}</div>"
        "<table class='tbl'>"
        "<thead><tr>"
        f"<th>{_h(col_label or group_col)}</th>"
        "<th class='num-head'>Cases</th>"
        "<th class='num-head col-score'>Routing</th>"
        "<th class='num-head col-score'>Tool</th>"
        "<th class='num-head col-score th-wrap'>Params mean</th>"
        "</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
        "</div>"
    )


def _tool_frequency_table(df: pd.DataFrame) -> str:
    expected = Counter()
    actual = Counter()
    for names in df["_expected_tools"]:
        expected.update(names)
    for names in df["_actual_tools"]:
        actual.update(names)
    tools = sorted(set(expected) | set(actual), key=lambda t: (-(expected[t] + actual[t]), t.lower()))

    # Per-tool tool_usage pass rate and tool_parameter mean — computed over
    # cases where this tool appears in expected_tools (i.e., the cases where
    # the scorers were actually evaluating this tool's plan / parameters).
    expected_norm_per_row = df["_expected_tool_seq_norm"]
    body_rows = []
    for tool in tools:
        tool_norm = _norm_tool(tool)
        mask = expected_norm_per_row.apply(lambda names: tool_norm in names)
        sub = df[mask]
        if len(sub):
            usage_vals = pd.to_numeric(sub.get("tool_usage_score", pd.Series(dtype=float)), errors="coerce")
            usage_scored = int(usage_vals.notna().sum())
            usage_pass = (usage_vals.eq(1).sum() / usage_scored) if usage_scored else None
            param_vals = pd.to_numeric(sub.get("tool_parameter_score", pd.Series(dtype=float)), errors="coerce")
            param_mean = param_vals.mean() if param_vals.notna().any() else None
        else:
            usage_pass = None
            param_mean = None
        usage_cls = _rate_class(usage_pass)
        # Tools we explicitly exclude from tool_parameter scoring (fuzzy
        # semantic params) get "–" even when a stray mixed row carries a
        # score, so the column reads as "not measured" instead of a one-row
        # mean that misrepresents the tool.
        if _norm_tool(tool) in EXCLUDE_PARAM_TOOLS:
            param_cell = "<td class='num-cell rate-na'>–</td>"
        else:
            param_cls = _rate_class(param_mean)
            param_cell = f"<td class='num-cell {param_cls}'>{_fmt_pct(param_mean)}</td>"
        body_rows.append(
            "<tr>"
            f"<td><code>{_h(tool)}</code></td>"
            f"<td class='num-cell'><span title='expected calls'>"
            f"<strong>{expected[tool]}</strong></span>"
            f" <span class='muted-small' style='display:inline;color:#a3b5c9'>/ "
            f"<span title='actual calls'>{actual[tool]}</span></span></td>"
            f"<td class='num-cell {usage_cls}'>{_fmt_pct(usage_pass)}</td>"
            f"{param_cell}"
            "</tr>"
        )
    return (
        "<table class='tbl'>"
        "<thead><tr>"
        "<th>Tool</th>"
        "<th class='num-head'>Calls (exp / act)</th>"
        "<th class='num-head'>Tool usage pass</th>"
        "<th class='num-head'>Params mean</th>"
        "</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def _top_problem_cases(df: pd.DataFrame, limit: int = 15) -> str:
    sub = df[df["_issue_bucket"] != "pass"].copy()
    if sub.empty:
        return "<p class='placeholder'>No failing cases under the strict report definition.</p>"
    sub = sub.sort_values(
        by=["_issue_count", "_score_min", "_score_mean", "test_case_id"],
        ascending=[False, True, True, True],
    ).head(limit)
    body_rows = []
    for _, row in sub.iterrows():
        issue = ", ".join(row["_issue_labels"]) or row["_issue_bucket"]
        cid = _h(row.get("test_case_id"))
        body_rows.append(
            "<tr>"
            f"<td><a class='case-link' href='#' data-case='{cid}'>{cid}</a></td>"
            f"<td>{_h(issue)}</td>"
            f"<td>{_h(row.get('eval_domain'))}</td>"
            f"<td>{_h(_short(row.get('user_query'), 110))}</td>"
            f"<td class='num-cell'>{_fmt_num(row.get('_score_mean'), 3)}</td>"
            f"<td class='num-cell'>{_fmt_num(row.get('agent_routing_score'), 3)}</td>"
            f"<td class='num-cell'>{_fmt_num(row.get('tool_usage_score'), 3)}</td>"
            f"<td class='num-cell'>{_fmt_num(row.get('tool_parameter_score'), 3)}</td>"
            "</tr>"
        )
    return (
        "<table class='tbl'>"
        "<thead><tr>"
        "<th>Case</th><th>Issue</th><th>Domain</th><th>Query</th>"
        "<th class='num-head'>Mean</th>"
        "<th class='num-head'>Routing</th>"
        "<th class='num-head'>Tools</th>"
        "<th class='num-head'>Params</th>"
        "</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


# ─── Prompts tab ────────────────────────────────────────────────────────
# Reads `prompt_{source_run_id}.json` (written by write_prompt_sidecar in
# the import notebook) and surfaces the supervisor + sub-agent system
# prompts and the run's tool descriptions on a dedicated tab. Per-trace
# hash columns let us flag runs that mix multiple deploys (uniform run ⇒
# nunique == 1 on each hash column).

import hashlib as _hashlib  # noqa: E402

_EMPTY_PROMPT_HASH = _hashlib.md5(b"").hexdigest()[:10]

_PROMPT_HASH_COLS = (
    ("main_agent_prompt_hash", "Supervisor (main_agent) prompt"),
    ("daily_banking_agent_prompt_hash", "daily_banking_agent prompt"),
    ("tool_descriptions_hash", "Tool descriptions"),
)


def _load_prompt_sidecar(input_path: Path, source_run_id: str,
                          prompts_path: Path | None = None) -> dict[str, Any] | None:
    """Locate the prompt sidecar JSON: explicit flag → next to input CSV."""
    candidate: Path | None = None
    if prompts_path is not None:
        candidate = prompts_path
        if candidate.is_dir() and source_run_id:
            candidate = candidate / f"prompt_{source_run_id}.json"
    elif source_run_id and source_run_id not in ("–", "—", "mixed"):
        candidate = input_path.parent / f"prompt_{source_run_id}.json"
    if candidate is None or not candidate.exists():
        return None
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _prompt_hash_warnings(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Per-column breakdown for any prompt/tool hash that isn't uniform.

    Empty rows for a given hash column → no warning for that column (the
    column was never populated, e.g. older CSV). Mixed values OR any row
    with the empty-string hash (== missing prompt) → warning entry.
    """
    warnings: list[dict[str, Any]] = []
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
        breakdown: list[dict[str, Any]] = []
        for hash_value, count in counts.items():
            sample_ids = df.loc[df[col] == hash_value, "trace_id"].head(3).tolist()
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


def _prompt_warning_card(warnings: list[dict[str, Any]]) -> str:
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


def _prompts_tab(sidecar: dict[str, Any] | None, source_run_id: str) -> str:
    """Render the Prompts tab body.

    Empty sidecar → small explainer + path of where the file is expected.
    """
    if not sidecar:
        return (
            "<div class='card'>"
            "<div class='card-title'>Prompts</div>"
            "<p class='prompt-empty'>No prompt sidecar found for this run. "
            f"Expected <code>prompt_{_h(source_run_id) or '&lt;run_id&gt;'}.json</code> "
            "next to the input CSV. Re-run the import notebook (it now writes "
            "the sidecar automatically) to populate this tab.</p>"
            "</div>"
        )

    supervisor = sidecar.get("main_agent_system_prompt") or ""
    dba_prompt = sidecar.get("daily_banking_agent_system_prompt") or ""
    tool_descriptions = sidecar.get("tool_descriptions") or {}

    def _prompt_block(label: str, body: str, *, open_: bool = False) -> str:
        attr = " open" if open_ else ""
        empty_note = " <em class='prompt-empty-note'>(not extracted from any trace)</em>" if not body.strip() else ""
        return (
            f"<details class='prompt-block'{attr}>"
            f"<summary>{_h(label)}{empty_note}</summary>"
            f"<pre class='prompt-pre'>{_h(body)}</pre>"
            "</details>"
        )

    tool_blocks: list[str] = []
    for name in sorted(tool_descriptions.keys()):
        desc = tool_descriptions.get(name) or ""
        tool_blocks.append(_prompt_block(name, desc))
    tool_section = (
        "".join(tool_blocks)
        if tool_blocks
        else "<p class='prompt-empty'>No tool descriptions captured for this run.</p>"
    )

    return (
        "<div class='card prompts-card'>"
        "<div class='card-title'>Supervisor prompt</div>"
        f"{_prompt_block('main_agent', supervisor, open_=True)}"
        "</div>"
        "<div class='card prompts-card'>"
        "<div class='card-title'>Agent prompts</div>"
        f"{_prompt_block('daily_banking_agent', dba_prompt)}"
        "</div>"
        "<div class='card prompts-card'>"
        "<div class='card-title'>"
        f"Tool descriptions <span class='tool-count'>({len(tool_descriptions)})</span>"
        "</div>"
        f"{tool_section}"
        "</div>"
    )


def _chip_group(label: str, group: str, options: list[tuple[str, str]], *, foldable: bool = False) -> str:
    chips = "".join(
        f'<button class="chip" data-group="{_h(group)}" data-value="{_h(value)}">{_h(text)}</button>'
        for value, text in options
    )
    if foldable:
        return (
            "<div class='filter-group foldable-filter'>"
            "<details class='dim-matrix-details'>"
            "<summary class='dim-matrix-summary'>"
            f"<span class='filter-label'>{_h(label)}</span>"
            "</summary>"
            f"<div class='foldable-filter-body'>{chips}</div>"
            "</details>"
            "</div>"
        )
    return (
        "<div class='filter-group'>"
        f"<span class='filter-label'>{_h(label)}</span>"
        f"{chips}"
        "</div>"
    )


def _scorer_matrix(df: pd.DataFrame) -> str:
    """Foldable per-scorer chip matrix in the Test Cases sidebar.

    Each scorer row exposes pass/fail/(partial)/(n-a) chips that filter
    the case list by that scorer's outcome. Mirrors the CZKB sidebar's
    Judge-scorers matrix; here the chip set varies per scorer.
    """
    rows = []
    chip_defs = {
        "agent_routing": [("pass", "pass"), ("fail", "fail")],
        "tool_usage": [("pass", "pass"), ("fail", "fail")],
        "tool_parameter": [("pass", "pass"), ("partial", "partial"), ("fail", "fail"), ("na", "n/a")],
    }
    for scorer in SCORERS:
        chips = "".join(
            f'<button type="button" class="chip dim-chip" '
            f'data-dim="{_h(scorer)}" data-score="{_h(value)}">{_h(text)}</button>'
            for value, text in chip_defs[scorer]
        )
        rows.append(
            "<div class='dim-matrix-row'>"
            f"<span class='dim-matrix-name' title='{_h(scorer)}'>{_h(SCORER_LABEL[scorer])}</span>"
            f"<span class='dim-matrix-chips'>{chips}</span>"
            "</div>"
        )
    return (
        "<div class='filter-group dim-matrix-group'>"
        "<details class='dim-matrix-details'>"
        "<summary class='dim-matrix-summary'>"
        "<span class='filter-label'>Scorers</span>"
        "<span class='dim-matrix-active-count'></span>"
        "</summary>"
        f"<div class='dim-matrix'>{''.join(rows)}</div>"
        "</details>"
        "</div>"
    )


# ─── CSS (ported from czkb_report.py) ────────────────────────────────────

CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif;
       font-size: 14px; line-height: 1.5; color: #0a285c; background: #f4f6fa; }
code { font-family: monospace; font-size: 12px; background: #e7effd;
       padding: 1px 4px; border-radius: 4px; }
.sticky-top { position: sticky; top: 0; z-index: 20; }
.report-header { background: #2870ed; color: #fff; padding: 12px 24px;
                 box-shadow: 0 2px 6px rgba(10,40,92,0.15); }
.report-header code { background: rgba(255,255,255,0.18); color: #fff;
                      padding: 1px 6px; border-radius: 4px; font-weight: 500; }
.header-inner { display: flex; align-items: center; justify-content: space-between;
                max-width: 1280px; margin: 0 auto; flex-wrap: wrap; gap: 14px; }
.report-header h1 { font-size: 26px; font-weight: 700; line-height: 1.15;
                    white-space: nowrap; letter-spacing: -0.01em; }
.header-meta { display: flex; gap: 18px; font-size: 12px; color: rgba(255,255,255,.92);
               flex-wrap: wrap; }
/* Two-column meta grid (mirrors czkb_report.py header). The footer row
   spans both columns and gets a thin top divider so long input paths
   sit clearly under the per-column lists. */
.header-meta-grid { display: grid;
                     grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
                     row-gap: 4px; column-gap: 32px;
                     align-items: start; margin-top: 6px; }
@media (max-width: 720px) {
  .header-meta-grid { grid-template-columns: minmax(0, 1fr); }
}
.header-meta-grid .header-col { display: flex; flex-direction: column;
                                 gap: 3px; min-width: 0; }
.header-meta-grid .header-footer { grid-column: 1 / -1; margin-top: 4px;
                                     padding-top: 4px;
                                     border-top: 1px solid rgba(255,255,255,.18); }
.header-meta-grid .header-aux { color: rgba(255,255,255,.7); font-weight: 400; }
.header-meta-grid code { word-break: break-all; }
.tab-nav { background: #fff; border-bottom: 1px solid #e4eaf0;
           display: flex; padding: 0 24px;
           box-shadow: 0 1px 3px rgba(10,40,92,0.05); }
.tab-nav-inner { display: flex; max-width: 1280px; width: 100%; margin: 0 auto;
                 align-items: stretch; }
.tab-btn { background: none; border: none; cursor: pointer; font: inherit;
           color: #5c7999; font-size: 13px; font-weight: 500;
           padding: 12px 18px; border-bottom: 2px solid transparent; }
.tab-btn.active, .tab-btn:hover { color: #1d69ec; border-bottom-color: #1d69ec; }
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
.content { max-width: 1280px; margin: 0 auto; padding: 20px 24px; }
.tab-panel { display: none; } .tab-panel.active { display: block; }
.card { background: #fff; border: 1px solid #e4eaf0; border-radius: 10px;
        box-shadow: 0 1px 4px rgba(10,40,92,.07); padding: 18px 20px;
        margin-bottom: 16px; }
.card-title { font-size: 15px; font-weight: 600; margin-bottom: 12px;
              display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.card-note { font-size: 12px; color: #5c7999; margin-bottom: 10px;
             line-height: 1.5; }
.card-footnote { font-size: 11px; color: #5c7999; line-height: 1.5;
                 margin-top: 10px; padding-top: 8px;
                 border-top: 1px dashed #e4eaf0; }
.card-footnote code { font-size: 10.5px; }

.metrics { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
           gap: 12px; margin-bottom: 16px; }
.metric { background: #fff; border: 1px solid #e4eaf0; border-radius: 10px;
          padding: 12px 14px; }
.metric-label { font-size: 11px; color: #a3b5c9; text-transform: uppercase;
                letter-spacing: .5px; }
.metric-value { font-size: 22px; font-weight: 700; color: #1d69ec; margin-top: 3px;
                font-family: Inter, -apple-system, sans-serif; letter-spacing: -0.02em; }

.headline-row { display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 14px; margin-bottom: 16px; }
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

.grid-2 { display: grid;
          grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
          gap: 16px; margin-bottom: 16px; }
.grid-2 > .card { min-width: 0; }
@media (max-width: 900px) { .grid-2 { grid-template-columns: minmax(0, 1fr); } }

.badge { display: inline-block; padding: 1px 8px; border-radius: 10px;
         font-size: 11px; font-weight: 600; min-width: 22px; text-align: center;
         white-space: nowrap; }
.badge-bad  { background: #fde5e3; color: #cf2a1e; }
.badge-mid  { background: #fef4e2; color: #b46504; }
.badge-good { background: #dff5ea; color: #028661; }
.badge-na   { background: #edf0f4; color: #5c7999; }

.fm-badge { display: inline-block; padding: 1px 8px; border-radius: 10px;
            font-size: 11px; font-weight: 600; white-space: nowrap; }
.fm-pass       { background: #dff5ea; color: #028661; }
.fm-routing    { background: #fde5e3; color: #cf2a1e; }
.fm-usage      { background: #fde5e3; color: #cf2a1e; }
.fm-params     { background: #fef4e2; color: #b46504; }
/* Baseline-comparison badges (see compareBadge / compareBannerHtml). */
.fm-regression { background: #fde5e3; color: #cf2a1e; }
.fm-persistent { background: #fef4e2; color: #b46504; }
.fm-fixed      { background: #dff5ea; color: #028661; }
.fm-new        { background: #e7effd; color: #1d69ec; }

/* Detail-pane banner showing PASS → FAIL etc. against the baseline. */
.compare-banner { display: inline-flex; align-items: center; gap: 6px;
                  padding: 4px 10px; border-radius: 6px;
                  font-size: 12px; margin-bottom: 10px; }
.compare-banner.fm-regression { background: #fde5e3; color: #8a1a12; }
.compare-banner.fm-persistent { background: #fef4e2; color: #6a3c01; }
.compare-banner.fm-fixed      { background: #dff5ea; color: #024a3a; }
.compare-banner.fm-new        { background: #e7effd; color: #0a285c; }
.compare-banner.fm-pass       { background: #dff5ea; color: #024a3a; }
.compare-banner code { background: rgba(255,255,255,0.6); color: inherit; }

/* Clickable headline card (Regressions). Mirrors the static card
   style but adds a hover lift + tone-coded left border. */
a.headline-card.hc-clickable { text-decoration: none; cursor: pointer;
                                color: inherit; transition: box-shadow 0.15s ease,
                                transform 0.15s ease; }
a.headline-card.hc-clickable:hover { box-shadow: 0 4px 12px rgba(10,40,92,0.12);
                                      transform: translateY(-1px); }
a.headline-card.hc-bad  { border-left: 3px solid #cf2a1e; }
a.headline-card.hc-bad  .hc-value  { color: #cf2a1e; }
a.headline-card.hc-good { border-left: 3px solid #057f19; }
a.headline-card.hc-good .hc-value  { color: #057f19; }

.tbl { width: 100%; border-collapse: collapse; font-size: 12px;
       table-layout: auto; }
.tbl th, .tbl td { border-bottom: 1px solid #edf0f4; padding: 6px 10px;
                   text-align: left; vertical-align: top;
                   overflow-wrap: anywhere; word-break: break-word; }
.tbl th { background: #f4f6fa; color: #5c7999; font-weight: 600;
          text-transform: uppercase; font-size: 11px; letter-spacing: .5px;
          /* Single-word column headers should not break across letters
             (overflow-wrap: anywhere from .tbl td would otherwise apply
             when a column is narrow). Cells that explicitly want to wrap
             opt in via `.th-wrap`. */
          white-space: nowrap; overflow-wrap: normal; word-break: normal; }
.tbl th.th-wrap { white-space: normal; }
/* Equal-width score columns for Scores By Domain. The "Params mean"
   header opts into `th-wrap` so its label fits this width on two lines. */
.tbl th.col-score { width: 78px; min-width: 78px; }
.tbl th.num-head { text-align: right; }
.tbl tbody tr:hover { background: #f9fbff; }
.tbl td.num-cell { text-align: right; font-variant-numeric: tabular-nums; }
.num-cell.good { color: #057f19; font-weight: 600; }
.num-cell.mid  { color: #ad5700; font-weight: 600; }
.num-cell.bad  { color: #cf2a1e; font-weight: 600; }
/* Rate-based cell highlights (used by Core / Strict pass columns). */
.num-cell.rate-hi  { background: rgba(5,127,25,0.08);   color: #057f19; font-weight: 600; }
.num-cell.rate-mid { background: rgba(242,169,30,0.12); color: #ad5700; font-weight: 600; }
.num-cell.rate-lo  { background: rgba(207,42,30,0.08);  color: #cf2a1e; font-weight: 600; }
.num-cell.rate-na  { color: #a3b5c9; }
.muted-small { display: block; color: #a3b5c9; font-size: 11px; margin-top: 2px; }
.scorer-cell-head { display: inline-flex; align-items: center; gap: 6px; }
.tbl td.scorer-cell { white-space: nowrap; overflow-wrap: normal;
                      word-break: normal; }

.mini-bar { height: 8px; width: 100%; background: #edf0f4; border-radius: 4px;
            overflow: hidden; min-width: 60px; }
.mini-bar span { display: block; height: 100%; border-radius: 4px; }

a.case-link { color: #1d69ec; text-decoration: none; font-weight: 600; }
a.case-link:hover { text-decoration: underline; }
a.fm-clear-link { color: #1d69ec; text-decoration: none; font-weight: 700;
                  cursor: pointer; }
a.fm-clear-link:hover { text-decoration: underline; }
a.group-filter-link { color: #1d69ec; text-decoration: none; font-weight: 700;
                      cursor: pointer; }
a.group-filter-link:hover { text-decoration: underline; }
a.scorer-filter-link { color: inherit; text-decoration: none; font-weight: 700;
                       cursor: pointer; }
a.scorer-filter-link:hover { text-decoration: underline; }

/* ─── Info-icon + global tip-bubble (matches czkb_report.py) ─── */
.info-icon { cursor: help; color: #a3b5c9; font-style: normal;
             font-family: -apple-system, sans-serif;
             font-size: 13px; line-height: 1; user-select: none;
             display: inline-block; vertical-align: middle; }
.info-icon:hover { color: #135ee2; }
.tip-bubble {
  position: fixed; left: 0; top: -9999px;
  background: #0a285c; color: #fff;
  padding: 8px 10px; border-radius: 6px;
  font-family: Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif;
  font-size: 11px; font-weight: 500; line-height: 1.45;
  max-width: 520px; width: max-content;
  box-shadow: 0 4px 12px rgba(10,40,92,0.25);
  pointer-events: none; z-index: 10000;
  opacity: 0; transition: opacity 0.12s ease;
  white-space: pre-wrap;
}
.tip-bubble.visible { opacity: 1; }

/* ─── Test Cases tab layout ─── */
.cases-layout { display: grid; grid-template-columns: 360px 1fr; gap: 16px;
                height: calc(100vh - 160px); min-height: 520px;
                position: relative; }
@media (max-width: 900px) { .cases-layout { grid-template-columns: 1fr; height: auto; } }
.cases-layout.sidebar-hidden { grid-template-columns: 1fr; gap: 0; }
.cases-layout.sidebar-hidden .cases-sidebar { display: none; }
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
                 padding: 10px; display: flex; flex-direction: column; min-height: 0;
                 position: relative; }
.case-search { width: 100%; padding: 8px 10px; border: 1px solid #e4eaf0;
               border-radius: 6px; margin-bottom: 8px; font-size: 13px;
               flex-shrink: 0; font-family: inherit; }
.filters-wrap { flex-shrink: 0; border-bottom: 1px solid #edf0f4;
                padding-bottom: 8px; margin-bottom: 8px; }
.filter-group { display: flex; align-items: flex-start; gap: 4px;
                margin-bottom: 4px; flex-wrap: wrap; }
.foldable-filter { width: 100%; flex-direction: column; }
.foldable-filter-body { display: flex; flex-wrap: wrap; gap: 4px;
                        margin-top: 6px; padding-left: 4px; }

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
   the reader knows it filters to crashes specifically. */
.chip-err { background: #fde5e3; color: #8b1a10; border: 1px solid #f3d6d2; }
.chip-err:hover { background: #fbd0cc; }
.chip-err.active { background: #cf2a1e; color: #fff; border-color: #cf2a1e; }

.range-filter { display: flex; align-items: center; gap: 6px; font-size: 11px;
                color: #537090; flex-wrap: wrap; }
.range-filter input { width: 58px; padding: 3px 6px; font: inherit; font-size: 11px;
                      border: 1px solid #e4eaf0; border-radius: 6px;
                      color: #0a285c; text-align: right;
                      font-variant-numeric: tabular-nums; }
.range-filter input:focus { outline: none; border-color: #135ee2; }
.range-reset { background: transparent; border: none; color: #cf2a1e;
               cursor: pointer; font-size: 10px; font-weight: 600;
               padding: 0 4px; }

.active-filter-banner { display: none; align-items: center; gap: 8px;
                        padding: 6px 10px; margin-bottom: 8px;
                        background: #eef4fd; border: 1px solid #dbe5f8;
                        border-radius: 6px; font-size: 11px; color: #0a285c;
                        flex-shrink: 0; }
.active-filter-banner.visible { display: flex; }
.active-filter-banner button { margin-left: auto; background: transparent;
                                border: none; color: #cf2a1e; font: inherit;
                                font-size: 11px; font-weight: 600; cursor: pointer; }

.list-count { font-size: 10px; color: #a3b5c9; padding: 0 4px 6px;
              text-transform: uppercase; letter-spacing: .5px; flex-shrink: 0; }
.cases-list-wrap { flex: 1 1 auto; overflow-y: auto; min-height: 0; }
.cases-list { list-style: none; }
.cases-list li { padding: 8px 10px; border-radius: 6px; cursor: pointer;
                 font-size: 12px; display: flex; align-items: center;
                 gap: 6px; flex-wrap: wrap; }
.cases-list li:hover { background: #f4f6fa; }
.cases-list li.active { background: #e7effd; }
.case-id { font-weight: 600; color: #0a285c; }
.case-q { color: #5c7999; font-size: 11px; width: 100%; }

.case-detail { background: #fff; border: 1px solid #e4eaf0; border-radius: 10px;
               padding: 18px 20px; overflow-y: auto; min-height: 0; }
.case-detail-title { display: flex; align-items: center; gap: 8px;
                     flex-wrap: wrap; }
.detail-section { margin-bottom: 16px; }
.detail-row { display: grid; grid-template-columns: 1fr 1fr; gap: 14px;
              margin-bottom: 16px; }
.detail-row .detail-section { margin-bottom: 0; min-width: 0; }
@media (max-width: 900px) { .detail-row { grid-template-columns: 1fr; } }
.detail-section h3 { font-size: 13px; color: #5c7999; text-transform: uppercase;
                     letter-spacing: .5px; margin-bottom: 6px;
                     display: flex; align-items: center; gap: 8px; }
.detail-section .body { background: #f4f6fa; border-radius: 6px;
                        padding: 10px 12px; white-space: pre-wrap;
                        font-size: 13px; word-break: break-word; }
.detail-section .body.mono { font-family: monospace; font-size: 12px;
                              line-height: 1.45; }
.bodywrap { max-height: 280px; overflow-y: auto; position: relative;
            border-radius: 6px; }
.bodywrap.expanded { max-height: none; overflow: visible; }
.body-toggle { background: transparent; border: 1px solid #dbe5f8; color: #135ee2;
               font: inherit; font-size: 10px; font-weight: 600;
               text-transform: uppercase; letter-spacing: .04em;
               padding: 1px 8px; border-radius: 10px; cursor: pointer;
               margin-left: 8px; vertical-align: middle; }
.body-toggle:hover { background: #eef4fd; }

/* Segmented language switch — two pills sharing a border. The
   highlighted segment shows which body is currently visible; clicking
   the other segment switches to that body. Both labels stay visible
   so it's unambiguous which one is the action and which is the state. */
.lang-switch { display: inline-flex; margin-left: 6px;
               border: 1px solid #dbe5f8; border-radius: 10px;
               overflow: hidden; vertical-align: middle; }
.lang-seg { background: transparent; border: none; color: #135ee2;
            font: inherit; font-size: 10px; font-weight: 600;
            text-transform: uppercase; letter-spacing: .04em;
            padding: 1px 8px; cursor: pointer; }
.lang-seg + .lang-seg { border-left: 1px solid #dbe5f8; }
.lang-seg:hover:not(.active) { background: #eef4fd; }
.lang-seg.active { background: #135ee2; color: #fff; cursor: default; }

/* ─── ENUM section (KB / KB&API cases only) ─── */
.enum-chip { display: inline-block; padding: 1px 7px; border-radius: 10px;
             font-size: 11px; font-weight: 600; margin: 1px 2px;
             font-family: monospace; }
.enum-chip.match   { background: #dff5ea; color: #0a285c; font-weight: 700; }
.enum-chip.miss    { background: #fde5e3; color: #0a285c; font-weight: 700; }
.enum-chip.neutral { background: #edf0f4; color: #5c7999; font-weight: 500; }
.enum-chip.expected-row    { background: #e0eafd; color: #0a285c; font-weight: 800; }
.enum-chip.expected-missed { background: #edf0f4; color: #5c7999; font-weight: 600; }
.enum-rows { display: flex; flex-direction: column; gap: 4px;
             background: #f4f6fa; border-radius: 6px;
             padding: 10px 12px; font-size: 12px; font-family: monospace;
             color: #0a285c; }
.enum-row { display: grid; grid-template-columns: 180px 1fr; gap: 8px;
            align-items: baseline; }
.enum-row .enum-label { color: #5c7999; font-weight: 600;
                        display: inline-flex; align-items: center; gap: 4px;
                        font-family: Inter, -apple-system, sans-serif; }
.enum-row .enum-count { color: #a3b5c9; font-weight: 500; font-size: 11px; }
hr.enum-divider { border: none; border-top: 1px solid #c8d3e1;
                  margin: 2px 0; }

.score-strip { display: flex; flex-direction: column; gap: 10px;
               margin-bottom: 14px; padding: 12px; background: #f8fafc;
               border: 1px solid #edf0f4; border-radius: 10px; }
.score-strip-row { display: flex; gap: 8px; flex-wrap: wrap; }
.score-box { flex: 1 1 90px; min-width: 90px; background: #fff;
             border: 1px solid #edf0f4; border-radius: 8px;
             padding: 8px 10px 8px 14px; text-align: left; position: relative; }
.score-box .sb-label { font-size: 9px; color: #537090; font-weight: 600;
                       text-transform: uppercase; letter-spacing: .12em;
                       display: flex; align-items: center; gap: 6px; }
.score-box .sb-value { font-size: 22px; font-weight: 600; color: #0a285c;
                       margin-top: 4px; font-family: Inter, -apple-system, sans-serif;
                       letter-spacing: -0.02em; }
.score-box .sb-accent { position: absolute; top: 0; left: 0; bottom: 0;
                        width: 3px; border-radius: 8px 0 0 8px; }
.score-box.s-bad  .sb-accent { background: #cf2a1e; }
.score-box.s-bad  .sb-value  { color: #cf2a1e; }
.score-box.s-mid  .sb-accent { background: #f2a91e; }
.score-box.s-mid  .sb-value  { color: #ad5700; }
.score-box.s-good .sb-accent { background: #057f19; }
.score-box.s-good .sb-value  { color: #057f19; }
.score-box.s-na   .sb-accent { background: #e4eaf0; }
.score-box.s-na   .sb-value  { color: #a3b5c9; }
/* "Configured but no score" — should not happen; flag in amber. */
.score-box.s-warn .sb-accent { background: #f2a91e; }
.score-box.s-warn .sb-value  { color: #ad5700; }
/* "Not configured for this case" — by design; dim with dashed border. */
.score-box.s-skip { background: repeating-linear-gradient(
                    135deg, #fafbfd, #fafbfd 6px, #f3f5f9 6px, #f3f5f9 12px);
                    border-style: dashed; border-color: #d6dde7; }
.score-box.s-skip .sb-accent { background: transparent; }
.score-box.s-skip .sb-value  { color: #94a4b6; font-size: 18px; }
.score-box .sb-sub { font-size: 9px; color: #7c8ca0; font-weight: 500;
                     text-transform: uppercase; letter-spacing: .1em;
                     margin-top: 2px; }
.score-box.s-warn .sb-sub { color: #ad5700; }

/* Span-level error surfaces. Trace info.state stays "OK" whenever the
   agent returned an answer, so a child span crashing (ConnectTimeout,
   CancelledError, etc.) is silent unless we surface it here. Chip on
   the case-detail header is always rendered when has_span_error;
   standalone <details> block lives in the score-strip so the reader
   can drill into exception type + trimmed stacktrace. */
.span-err-chip { display: inline-flex; align-items: center; gap: 4px;
                  padding: 2px 8px; border-radius: 10px;
                  background: #cf2a1e; color: #fff;
                  font-size: 10px; font-weight: 700; letter-spacing: 0.02em;
                  white-space: nowrap; cursor: help; }
.span-err-chip .span-err-icon { font-size: 11px; line-height: 1; }
.score-strip-err { border-top: 1px solid #f3d6d2; padding-top: 10px;
                    margin-top: 4px; }
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

.scorer-plan { display: flex; flex-wrap: wrap; align-items: center;
               gap: 6px; margin: 6px 0 10px; font-size: 11px; }
.scorer-plan-label { color: #5c7999; font-weight: 600;
                     text-transform: uppercase; letter-spacing: .08em;
                     font-size: 10px; }

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

.kv { display: grid; grid-template-columns: 140px minmax(0, 1fr);
      gap: 6px 12px; margin: 4px 0; font-size: 12px; }
.kv dt { color: #5c7999; font-weight: 600; }
.kv dd { color: #0a285c; word-break: break-word; }

.rationale-block { display: flex; flex-direction: column; gap: 4px;
                   padding: 8px 10px; background: #f8fafc;
                   border: 1px solid #edf0f4; border-radius: 6px; }
.rationale-head { display: flex; align-items: center; gap: 8px;
                  font-size: 11px; }
.rationale-name { font-family: monospace; color: #0a285c; }
.rationale-status { font-size: 10px; color: #5c7999; }
.rationale-text { font-size: 12px; color: #0a285c; }

.guideline-line { font-size: 12px; color: #0a285c; line-height: 1.5;
                  margin-bottom: 4px; word-break: break-word; }
.guideline-line:last-child { margin-bottom: 0; }

/* Tool-call cards — replaces the raw-JSON ``<pre>`` block. One card per
   expected / actual entry, with structured sections for arguments,
   output, status. */
.tc-list { display: flex; flex-direction: column; gap: 8px; }
.tc-card { background: #fff; border: 1px solid #e4eaf0; border-radius: 8px;
           padding: 10px 12px; }
.tc-head { display: flex; align-items: center; gap: 8px;
           padding-bottom: 6px; border-bottom: 1px solid #edf0f4;
           margin-bottom: 8px; flex-wrap: wrap; }
.tc-step { color: #a3b5c9; font-size: 11px; font-weight: 700;
           font-variant-numeric: tabular-nums; min-width: 18px; text-align: right; }
.tc-tool { font-size: 12px; font-weight: 700; color: #135ee2;
           background: #e7effd; padding: 2px 8px; border-radius: 6px;
           word-break: break-all; }
.tc-reason { font-size: 11.5px; color: #5c7999; font-style: italic;
             margin-bottom: 6px; line-height: 1.45; }
.tc-section-label { font-size: 9px; color: #5c7999; font-weight: 700;
                    text-transform: uppercase; letter-spacing: .12em;
                    margin: 6px 0 4px; }
.tc-pre { background: #f8fafc; border: 1px solid #edf0f4;
          border-radius: 6px; padding: 8px 10px;
          font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          font-size: 11px; line-height: 1.45; color: #0a285c;
          white-space: pre-wrap; word-break: break-word;
          max-height: 240px; overflow: auto; }
.tc-pre.tc-error { background: #fff1ef; border-color: #f7d1cc; color: #cf2a1e; }
.tc-empty { color: #a3b5c9; font-style: italic; font-size: 11.5px;
            padding: 4px 0; }

/* Per-output toggle inside an actual tool-call card. The <summary>
   reuses the small section-label style; clicking expands the <pre> below. */
details.tc-output { margin-top: 8px; }
details.tc-output > summary { list-style: none; cursor: pointer; user-select: none;
                              font-size: 9px; color: #5c7999; font-weight: 700;
                              text-transform: uppercase; letter-spacing: .12em;
                              padding: 4px 0; display: flex; align-items: center; gap: 6px; }
details.tc-output > summary::-webkit-details-marker { display: none; }
details.tc-output > summary::before { content: "▶"; font-size: 9px; color: #5c7999;
                                       display: inline-block;
                                       transition: transform 0.15s ease; }
details.tc-output[open] > summary::before { transform: rotate(90deg); }
details.tc-output > summary:hover { color: #135ee2; }
details.tc-output > summary:hover::before { color: #135ee2; }

/* Bordered detail-section: a soft outline + padding for the high-value
   Expected / Actual summary blocks and the Parameter Breakdown card.
   Applied via class — the unbordered sections (responses, rationales,
   tool calls) keep their open layout. */
.detail-section.bordered { border: 1px solid #e4eaf0; border-radius: 8px;
                           padding: 12px 14px; background: #fff;
                           box-shadow: 0 1px 2px rgba(10,40,92,.04); }
.detail-section.bordered h3 { margin-bottom: 8px; }

/* Per-parameter breakdown — one card per expected tool call. Each row is
   a key/value pair colored by outcome: green=matched, red=wrong (with
   actual ≠ expected shown side-by-side), default=missing. */
.pe-list { display: flex; flex-direction: column; gap: 8px; }
.pe-card { background: #f8fafc; border: 1px solid #edf0f4; border-radius: 6px;
           padding: 8px 12px; }
.pe-head { display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
           padding-bottom: 6px; border-bottom: 1px solid #edf0f4;
           margin-bottom: 6px; }
.pe-scores { margin-left: auto; font-size: 10.5px; color: #5c7999;
             font-variant-numeric: tabular-nums; }
.pe-rows { display: flex; flex-direction: column; gap: 3px; }
.pe-row { display: grid; grid-template-columns: minmax(140px, max-content) 1fr;
          gap: 12px; align-items: baseline; font-size: 12px; line-height: 1.5;
          padding: 2px 0; }
.pe-key { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          font-size: 11.5px; font-weight: 600; word-break: break-all; }
.pe-val { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          font-size: 11.5px; word-break: break-all; }
.pe-val code { background: transparent; padding: 0; color: inherit; }
.pe-matched .pe-key, .pe-matched .pe-val,
.pe-matched .pe-val code { color: #057f19; }
.pe-wrong   .pe-key, .pe-wrong   .pe-val,
.pe-wrong   .pe-val code { color: #cf2a1e; }
.pe-missing .pe-key, .pe-missing .pe-val,
.pe-missing .pe-val code { color: #ad5700; }
/* Excused — key was dropped from scoring by the relaxation rules.
   Greyed out + slight strikethrough on the key so it reads as
   "the scorer never saw this", not "this matched/failed". */
.pe-excused .pe-key, .pe-excused .pe-val,
.pe-excused .pe-val code { color: #94a4b6; }
.pe-excused .pe-key { text-decoration: line-through;
                      text-decoration-color: #c9d1dc; }
.pe-excused-swatch { color: #94a4b6; text-decoration: line-through;
                     text-decoration-color: #c9d1dc; }
.pe-null { color: #a3b5c9; font-style: italic; }
.pe-empty { font-size: 11.5px; color: #a3b5c9; font-style: italic;
            padding: 2px 0; }
.pe-legend { font-size: 10.5px; color: #a3b5c9; font-style: italic;
             margin-top: 4px; padding: 2px 0; }

.placeholder { color: #a3b5c9; font-style: italic; padding: 20px 0; }
"""


# ─── JS ──────────────────────────────────────────────────────────────────

JS = r"""
const CASES = __CASES__;
let activeCaseId = CASES.length ? CASES[0].id : null;

// Filter state. Single-select groups carry a string value (or null).
// `scorer_pairs` is the per-scorer matrix: array of {scorer, value} entries.
// `domains`/`personas` are Sets — multi-select.
const activeFilters = {
  outcome: null,
  flag: null,
  scorer_pairs: [],
  domains: new Set(),
  personas: new Set(),
  tools: new Set(),
  // `compare` is multi-select (Set), populated only when --baseline was
  // provided. Values: "regression", "persistent_failure", "fixed",
  // "new_case". Empty Set = no comparison filter.
  compare: new Set(),
  param_min: null,
  param_max: null,
};

// Baseline filename injected by render_html. Empty string means
// no comparison was loaded, so the detail banner stays hidden.
const BASELINE_NAME = __BASELINE_NAME__;

function esc(s) {
  if (s == null) return "";
  return String(s).replace(/[&<>"']/g, ch =>
    ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));
}

function shortText(value, n = 140) {
  const text = String(value ?? "").replace(/\s+/g, " ").trim();
  return text.length > n ? text.slice(0, n - 1).trimEnd() + "..." : text;
}

function parseList(x) {
  if (Array.isArray(x)) return x;
  if (x == null) return [];
  return [];
}

// ─── Tab switching ─────────────────────────────────────────────────
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.target).classList.add("active");
  });
});

// ─── Tip-bubble (global tooltip for any [data-tip] element) ──────────
// Native title is unreliable inside scrollable containers; we render a
// single fixed-position bubble appended to body and place it near the
// hovered element on each pointer event.
const _tipBubble = document.createElement("div");
_tipBubble.className = "tip-bubble";
document.body.appendChild(_tipBubble);
function _showTip(target) {
  const tip = target.dataset.tip;
  if (!tip) return;
  _tipBubble.textContent = tip;
  _tipBubble.classList.add("visible");
  const rect = target.getBoundingClientRect();
  _tipBubble.style.left = "0px";
  _tipBubble.style.top = "-9999px";
  const tw = _tipBubble.offsetWidth;
  const th = _tipBubble.offsetHeight;
  let left = rect.left + rect.width / 2 - tw / 2;
  let top  = rect.bottom + 8;
  const margin = 6;
  if (left < margin) left = margin;
  if (left + tw > window.innerWidth - margin) left = window.innerWidth - tw - margin;
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

// ─── Case-list filtering ────────────────────────────────────────────
const listEl = document.getElementById("cases-list");
const countEl = document.getElementById("list-count");
const searchEl = document.getElementById("case-search");
const detailEl = document.getElementById("case-detail");

function scorerBucket(c, scorer) {
  if (scorer === "tool_parameter") return c.param_bucket;
  const v = c.scores[scorer];
  if (v == null) return "na";
  return v >= 1 ? "pass" : "fail";
}

function matchesFilters(c) {
  if (activeFilters.outcome) {
    // Three classes of value can land here:
    //   "pass"           — sidebar Outcome chip / Primary-Outcomes "Pass" row
    //   "fail"           — sidebar Outcome chip
    //   <issue_bucket>   — Primary-Outcomes drill-down on a specific bucket
    //                       ("agent_routing", "tool_usage", "tool_parameter")
    // The third case requires an explicit bucket match — without it the
    // click does nothing because the value is neither "pass" nor "fail".
    const f = activeFilters.outcome;
    if (f === "pass") {
      if (c.issue_bucket !== "pass") return false;
    } else if (f === "fail") {
      if (c.issue_bucket === "pass") return false;
    } else if (c.issue_bucket !== f) {
      return false;
    }
  }
  if (activeFilters.flag === "span_errors" && !c.has_span_error) return false;
  if (activeFilters.compare && activeFilters.compare.size) {
    // OR within the group: case matches if any selected baseline-comparison
    // flag is true on the case. Flags are mutually exclusive per case
    // (regression / persistent / fixed / new_case) so OR is what we want.
    let hit = false;
    for (const v of activeFilters.compare) {
      if (c[v]) { hit = true; break; }
    }
    if (!hit) return false;
  }
  if (activeFilters.scorer_pairs && activeFilters.scorer_pairs.length) {
    // Group by scorer: OR within a scorer, AND across scorers.
    const grouped = {};
    for (const p of activeFilters.scorer_pairs) {
      if (!grouped[p.scorer]) grouped[p.scorer] = new Set();
      grouped[p.scorer].add(p.value);
    }
    for (const scorer in grouped) {
      const bucket = scorerBucket(c, scorer);
      if (!grouped[scorer].has(bucket)) return false;
    }
  }
  if (activeFilters.domains.size && !activeFilters.domains.has(c.domain || "")) return false;
  if (activeFilters.personas.size && !activeFilters.personas.has(c.persona || "")) return false;
  if (activeFilters.tools.size) {
    // A case matches if any selected tool appears in either expected or actual.
    const involved = new Set([...(c.expected_tools || []), ...(c.actual_tools || [])]);
    let hit = false;
    for (const t of activeFilters.tools) {
      if (involved.has(t)) { hit = true; break; }
    }
    if (!hit) return false;
  }
  if (activeFilters.param_min != null || activeFilters.param_max != null) {
    const v = c.scores.tool_parameter;
    if (v == null) return false;
    if (activeFilters.param_min != null && v < activeFilters.param_min) return false;
    if (activeFilters.param_max != null && v > activeFilters.param_max) return false;
  }
  return true;
}

function matchesSearch(c) {
  const q = (searchEl?.value || "").trim().toLowerCase();
  if (!q) return true;
  return [
    c.id, c.trace_id, c.domain, c.persona, c.query, c.query_en,
    c.expected_agent, c.actual_agent,
    (c.expected_tools || []).join(" "),
    (c.actual_tools || []).join(" "),
    c.actual_response,
  ].join(" ").toLowerCase().includes(q);
}

function outcomeBadge(c) {
  if (c.issue_bucket === "pass") return `<span class="fm-badge fm-pass">pass</span>`;
  if (c.issue_bucket === "agent_routing") return `<span class="fm-badge fm-routing">routing fail</span>`;
  if (c.issue_bucket === "tool_usage") return `<span class="fm-badge fm-usage">tool usage fail</span>`;
  if (c.issue_bucket === "tool_parameter") return `<span class="fm-badge fm-params">tool params</span>`;
  // Any other value (only possible if enrich.issue_bucket regresses) is
  // surfaced as a visible "?" so the bug doesn't hide as a missing badge.
  return `<span class="fm-badge badge-na" title="issue_bucket=${esc(c.issue_bucket || "(empty)")}">?</span>`;
}

// One-line banner in the case-detail header showing how this case
// compares to the baseline run. Only rendered when a baseline was
// loaded (BASELINE_NAME is non-empty).
function compareBannerHtml(c) {
  if (!BASELINE_NAME) return "";
  let arrow, cls;
  if (c.new_case) {
    arrow = "(no baseline)";
    cls = "fm-new";
  } else {
    const prev = c.prev_pass ? "PASS" : "FAIL";
    const now = c.issue_bucket === "pass" ? "PASS" : "FAIL";
    arrow = `${prev} → ${now}`;
    if (c.regression) cls = "fm-regression";
    else if (c.persistent_failure) cls = "fm-persistent";
    else if (c.fixed) cls = "fm-fixed";
    else cls = "fm-pass";
  }
  return `<div class="compare-banner ${cls}">
    Compared to <code>${esc(BASELINE_NAME)}</code>:
    <strong>${esc(arrow)}</strong>
  </div>`;
}

// ─── KB-retrieval ENUMs (KB / KB&API cases) ──────────────────────────
// Mirrors the CZKB report's ENUM section. Coloring rules:
//   • default     — green = in BOTH expected and reranked (right pick)
//                   red   = in expected XOR reranked (missed / wrong)
//                   grey  = in neither (pool noise)
//   • "expected"  — used for the ground-truth row itself; bolds IDs the
//                   reranker actually picked, greys out the ones it
//                   didn't (expected-row vs expected-missed).
function enumChipClass(id, expectedSet, rerankedSet, mode) {
  const inExp = expectedSet.has(id);
  const inRer = rerankedSet.has(id);
  if (mode === "expected") {
    return inRer ? "enum-chip expected-row" : "enum-chip expected-missed";
  }
  if (inExp && inRer) return "enum-chip match";
  if (inExp || inRer) return "enum-chip miss";
  return "enum-chip neutral";
}

function enumChipsApi(rawIds, expectedSet, rerankedSet, mode) {
  if (!Array.isArray(rawIds) || !rawIds.length) {
    return '<em class="lang-fallback">[]</em>';
  }
  return rawIds.map(id => {
    const s = String(id);
    return `<span class="${enumChipClass(s, expectedSet, rerankedSet, mode)}">${esc(s)}</span>`;
  }).join(" ");
}

function enumRow(label, info, chipsHtml, countText) {
  const count = countText != null ? ` <span class="enum-count">(${esc(String(countText))})</span>` : "";
  return `<div class="enum-row">
    <span class="enum-label">${esc(label)}${count}
      <span class="info-icon" data-tip="${esc(info)}">&#9432;</span>
    </span>
    <span class="enum-chips">${chipsHtml}</span>
  </div>`;
}

function enumSectionHtml(c) {
  // Only KB / KB&API cases carry meaningful retrieval data. Hide the
  // section entirely otherwise so api-only cases stay uncluttered.
  const dom = String(c.domain || "").toLowerCase();
  if (dom !== "kb" && dom !== "kb&api") return "";
  const exp  = Array.isArray(c.expected_enums)      ? c.expected_enums      : [];
  const pre  = Array.isArray(c.pre_prune_enum_ids)  ? c.pre_prune_enum_ids  : [];
  const post = Array.isArray(c.post_prune_enum_ids) ? c.post_prune_enum_ids : [];
  const rer  = Array.isArray(c.reranked_enum_ids)   ? c.reranked_enum_ids   : [];
  if (!exp.length && !pre.length && !post.length && !rer.length) return "";
  const expectedSet = new Set(exp.map(String));
  const rerankedSet = new Set(rer.map(String));
  const preCount  = c.pre_prune_enum_count  != null ? c.pre_prune_enum_count  : pre.length;
  const postCount = c.post_prune_enum_count != null ? c.post_prune_enum_count : post.length;
  // Computed miss/extra rows (mirrors CZKB layout).
  const postSet = new Set(post.map(String));
  const retrieverMiss = exp.filter(id => !postSet.has(String(id)));
  const rerankerMiss  = exp.filter(id => postSet.has(String(id)) && !rerankedSet.has(String(id)));
  const extraSelected = rer.filter(id => !expectedSet.has(String(id)));
  return `<div class="detail-section">
    <h3>ENUMs</h3>
    <div class="enum-rows">
      ${enumRow("expected",
                "Ground-truth ENUM IDs that should be retrieved/selected for this query (from target_enums_to_relevance on the test set). Bold blue = picked by the reranker; grey = not picked.",
                enumChipsApi(exp, expectedSet, rerankedSet, "expected"),
                exp.length)}
      <hr class="enum-divider">
      ${enumRow("final selected",
                "ENUM IDs the reranker actually picked — what the agent used (reranked_enum_ids). Green = also expected; red = picked but not in expected.",
                enumChipsApi(rer, expectedSet, rerankedSet),
                rer.length)}
      <hr class="enum-divider">
      ${enumRow("post-prune",
                "Candidate pool after dedup/pruning, before reranking (post_prune_enum_ids).",
                enumChipsApi(post, expectedSet, rerankedSet),
                postCount)}
      <hr class="enum-divider">
      ${enumRow("retriever miss",
                "Expected ENUMs that never made it into the post-prune pool — the retriever didn't surface them. Computed as expected_enums − post_prune_enum_ids.",
                enumChipsApi(retrieverMiss, expectedSet, rerankedSet))}
      ${enumRow("reranker miss",
                "Expected ENUMs that WERE in the post-prune pool but the reranker did NOT pick them. Computed as (expected_enums ∩ post_prune_enum_ids) − reranked_enum_ids.",
                enumChipsApi(rerankerMiss, expectedSet, rerankedSet))}
      ${enumRow("extra selected",
                "Reranker picks that aren't in the expected set — potential noise / distractors.",
                enumChipsApi(extraSelected, expectedSet, rerankedSet))}
      <hr class="enum-divider">
      ${enumRow("pre-prune",
                "Initial candidate pool returned by the retriever (pre_prune_enum_ids).",
                enumChipsApi(pre, expectedSet, rerankedSet),
                preCount)}
    </div>
  </div>`;
}

// Pill rendered next to the outcome badge in the case list and in
// the case-detail header. Mutually exclusive — at most one of
// regression / persistent / fixed / new_case is true per case.
function compareBadge(c) {
  if (c.regression) return `<span class="fm-badge fm-regression" title="Passed in baseline, fails now">regression</span>`;
  if (c.persistent_failure) return `<span class="fm-badge fm-persistent" title="Failed in both runs">persistent</span>`;
  if (c.fixed) return `<span class="fm-badge fm-fixed" title="Failed in baseline, passes now">fixed</span>`;
  if (c.new_case) return `<span class="fm-badge fm-new" title="Not present in baseline">new</span>`;
  return "";
}

function scoreBadge(label, value, kind) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return `<span class="badge badge-na" title="${esc(label)}: not scored">${esc(label)}: –</span>`;
  }
  let cls = "badge-mid";
  if (kind === "binary") {
    cls = value >= 1 ? "badge-good" : "badge-bad";
  } else {
    if (value >= 1) cls = "badge-good";
    else if (value <= 0) cls = "badge-bad";
  }
  const txt = Number(value).toFixed(value === Math.floor(value) ? 0 : 2);
  return `<span class="badge ${cls}">${esc(label)}: ${txt}</span>`;
}

// Readable labels for the issue_bucket values surfaced as outcome
// filters by the Primary Outcomes table. Falls through to the raw
// value for anything else (pass / fail / future buckets).
const OUTCOME_LABELS = {
  pass: "Pass",
  fail: "Fail",
  agent_routing: "Agent routing fail",
  tool_usage: "Tool usage fail",
  tool_parameter: "Tool parameter mismatch",
};

function updateActiveFilterBanner() {
  const el = document.getElementById("active-filter");
  const txt = document.getElementById("active-filter-text");
  if (!el || !txt) return;
  const msgs = [];
  if (activeFilters.outcome) {
    const label = OUTCOME_LABELS[activeFilters.outcome] || activeFilters.outcome;
    msgs.push(`outcome = <strong>${esc(label)}</strong>`);
  }
  if (activeFilters.flag) {
    msgs.push(`flag = <strong>${esc(activeFilters.flag)}</strong>`);
  }
  if (activeFilters.scorer_pairs.length) {
    const parts = activeFilters.scorer_pairs.map(p =>
      `<strong>${esc(p.scorer)}</strong>=<strong>${esc(p.value)}</strong>`);
    msgs.push(parts.join(" &amp; "));
  }
  if (activeFilters.domains.size) {
    msgs.push(`domain ∈ {<strong>${[...activeFilters.domains].map(esc).join(", ")}</strong>}`);
  }
  if (activeFilters.personas.size) {
    msgs.push(`persona ∈ {<strong>${[...activeFilters.personas].map(esc).join(", ")}</strong>}`);
  }
  if (activeFilters.tools.size) {
    msgs.push(`tool ∈ {<strong>${[...activeFilters.tools].map(esc).join(", ")}</strong>}`);
  }
  if (activeFilters.compare && activeFilters.compare.size) {
    const labels = {regression: "regression", persistent_failure: "persistent fail",
                    fixed: "fixed", new_case: "new"};
    const parts = [...activeFilters.compare].map(v => esc(labels[v] || v));
    msgs.push(`vs. baseline ∈ {<strong>${parts.join(", ")}</strong>}`);
  }
  if (activeFilters.param_min != null || activeFilters.param_max != null) {
    const a = activeFilters.param_min == null ? "" : activeFilters.param_min;
    const b = activeFilters.param_max == null ? "" : activeFilters.param_max;
    msgs.push(`tool_parameter ∈ [<strong>${a}</strong>, <strong>${b}</strong>]`);
  }
  if (msgs.length) {
    txt.innerHTML = "Filtered by " + msgs.join(" · ");
    el.classList.add("visible");
  } else {
    el.classList.remove("visible");
  }
}

function syncChipStates() {
  document.querySelectorAll(".chip[data-group]").forEach(btn => {
    const group = btn.dataset.group;
    const value = btn.dataset.value;
    if (group === "domains" || group === "personas" || group === "tools" || group === "compare") {
      btn.classList.toggle("active", activeFilters[group].has(value));
    } else {
      btn.classList.toggle("active", activeFilters[group] === value);
    }
  });
  // Per-scorer matrix chips
  document.querySelectorAll(".chip.dim-chip").forEach(btn => {
    const scorer = btn.dataset.dim;
    const value = btn.dataset.score;
    const active = activeFilters.scorer_pairs.some(p =>
      p.scorer === scorer && p.value === value);
    btn.classList.toggle("active", active);
  });
  const matrixCount = document.querySelector(".dim-matrix-active-count");
  if (matrixCount) {
    matrixCount.textContent = activeFilters.scorer_pairs.length
      ? ` (${activeFilters.scorer_pairs.length} active)` : "";
  }
}

function renderList() {
  updateActiveFilterBanner();
  syncChipStates();
  const filtered = CASES.filter(c => matchesFilters(c) && matchesSearch(c));
  countEl.textContent = `showing ${filtered.length} of ${CASES.length}`;
  listEl.innerHTML = filtered.map(c => {
    const routing = c.scores.agent_routing;
    const usage = c.scores.tool_usage;
    const params = c.scores.tool_parameter;
    return `
      <li data-id="${esc(c.id)}" class="${c.id === activeCaseId ? 'active' : ''}">
        <span class="case-id">${esc(c.id)}</span>
        ${outcomeBadge(c)}
        ${compareBadge(c)}
        ${c.domain ? `<span class="badge badge-na">${esc(c.domain)}</span>` : ""}
        <span style="margin-left:auto;font-weight:600;color:#135ee2">${c.score_mean != null ? c.score_mean.toFixed(2) : '–'}</span>
        <span class="case-q">${esc(shortText(c.query, 110))}</span>
      </li>
    `;
  }).join("");
  listEl.querySelectorAll("li").forEach(li => {
    li.addEventListener("click", () => selectCase(li.dataset.id));
  });
  if (filtered.length && !filtered.some(c => c.id === activeCaseId)) {
    selectCase(filtered[0].id);
  } else if (!filtered.length) {
    detailEl.innerHTML = `<div class="placeholder">No cases match the active filters.</div>`;
  }
}

function bindChips() {
  document.querySelectorAll(".chip[data-group]").forEach(btn => {
    btn.addEventListener("click", () => {
      const group = btn.dataset.group;
      const value = btn.dataset.value;
      if (group === "domains" || group === "personas" || group === "tools" || group === "compare") {
        if (activeFilters[group].has(value)) activeFilters[group].delete(value);
        else activeFilters[group].add(value);
      } else {
        activeFilters[group] = (activeFilters[group] === value) ? null : value;
      }
      renderList();
    });
  });
  // Per-scorer matrix chips
  document.querySelectorAll(".chip.dim-chip").forEach(btn => {
    btn.addEventListener("click", () => {
      const scorer = btn.dataset.dim;
      const value = btn.dataset.score;
      const idx = activeFilters.scorer_pairs.findIndex(
        p => p.scorer === scorer && p.value === value);
      if (idx >= 0) activeFilters.scorer_pairs.splice(idx, 1);
      else activeFilters.scorer_pairs.push({scorer, value});
      renderList();
    });
  });
  const clearBtn = document.getElementById("chip-clear");
  if (clearBtn) clearBtn.addEventListener("click", clearAllFilters);
  const afClear = document.getElementById("active-filter-clear");
  if (afClear) afClear.addEventListener("click", clearAllFilters);
}

function clearAllFilters() {
  activeFilters.outcome = null;
  activeFilters.flag = null;
  activeFilters.scorer_pairs = [];
  activeFilters.domains.clear();
  activeFilters.personas.clear();
  activeFilters.tools.clear();
  if (activeFilters.compare) activeFilters.compare.clear();
  activeFilters.param_min = null;
  activeFilters.param_max = null;
  ["param-min", "param-max"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.value = "";
  });
  renderList();
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

// Red chip rendered on the case-detail header. Always on when the
// trace has a span-level error — info.state="OK" lies whenever the
// orchestrator returned *some* response, even if a child span crashed.
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

// Standalone errors <details> block inside the score-strip. Surfaces
// exception type + trimmed stacktrace so the reader can see what
// crashed without leaving the case-detail pane.
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

function scoreClass(value, kind) {
  if (value == null) return "s-na";
  if (kind === "binary") return value >= 1 ? "s-good" : "s-bad";
  if (value >= 1) return "s-good";
  if (value <= 0) return "s-bad";
  return "s-mid";
}

function scoreStripHtml(c) {
  const configured = new Set(c.configured_scorers || []);
  // If the test case has no scorers_to_run metadata, fall back to assuming
  // every SCORERS entry is configured (so old CSVs render unchanged).
  const hasPlan = (c.configured_scorers || []).length > 0;
  function box(scorer, label, kind, fmt) {
    const v = c.scores[scorer];
    const isConfigured = !hasPlan || configured.has(scorer);
    if (v == null) {
      if (!isConfigured) {
        return `<div class="score-box s-skip" title="${esc(label)} is not in this case's scorers_to_run">
          <span class="sb-accent"></span>
          <div class="sb-label">${esc(label)}</div>
          <div class="sb-value">n/a</div>
          <div class="sb-sub">not configured</div>
        </div>`;
      }
      return `<div class="score-box s-warn" title="${esc(label)} was configured to run but has no score">
        <span class="sb-accent"></span>
        <div class="sb-label">${esc(label)}</div>
        <div class="sb-value">–</div>
        <div class="sb-sub">no score</div>
      </div>`;
    }
    return `<div class="score-box ${scoreClass(v, kind)}"><span class="sb-accent"></span>
      <div class="sb-label">${esc(label)}</div>
      <div class="sb-value">${fmt(v)}</div>
    </div>`;
  }
  return `<div class="score-strip">
    <div class="score-strip-row">
      ${box("agent_routing", "Agent routing (0/1)", "binary", v => v.toFixed(0))}
      ${box("tool_usage", "Tool usage (0/1)", "binary", v => v.toFixed(0))}
      ${box("tool_parameter", "Tool parameter (0–1)", "numeric", v => v.toFixed(2))}
    </div>
    ${spanErrorsBlockHtml(c)}
  </div>`;
}

function scorerPlanLine(c) {
  const plan = c.configured_scorers || [];
  if (!plan.length) return "";
  const labels = {
    agent_routing: "routing",
    tool_usage: "tool usage",
    tool_parameter: "tool parameters",
  };
  const chips = plan.map(s =>
    `<span class="badge badge-na" title="from scorers_to_run">${esc(labels[s] || s)}</span>`
  ).join(" ");
  return `<div class="scorer-plan">
    <span class="scorer-plan-label">Scorers configured:</span>
    ${chips}
  </div>`;
}

function routeChips(c) {
  const expected = parseList(c.expected_tools);
  const actual = parseList(c.actual_tools);
  const parts = [];
  if (c.actual_agent) {
    parts.push(`<span class="route-chip route-agent"><span class="route-label">agent</span>${esc(c.actual_agent)}</span>`);
  }
  if (actual.length) {
    parts.push(`<span class="route-chip route-tool"><span class="route-label">tools</span>${esc(actual.join(", "))}</span>`);
  }
  return parts.length ? `<span class="route-chips">${parts.join("")}</span>` : "";
}

function selectCase(id) {
  activeCaseId = id;
  document.querySelectorAll("#cases-list li").forEach(li =>
    li.classList.toggle("active", li.dataset.id === id));
  const c = CASES.find(x => x.id === id);
  if (!c) return;
  const expectedAgent = esc(c.expected_agent || "–");
  const actualAgent = esc(c.actual_agent || "–");
  const expectedTools = (c.expected_tools || []).map(t => `<code>${esc(t)}</code>`).join(", ") || "–";
  const actualTools = (c.actual_tools || []).map(t => `<code>${esc(t)}</code>`).join(", ") || "–";
  // Guidelines as plain paragraphs — no bullets per UI request.
  const guidelines = (c.guidelines || [])
    .map(g => `<div class="guideline-line">${esc(g)}</div>`)
    .join("") || "–";
  const seqMatch = c.tool_seq_match
    ? `<span class="badge badge-good">matches expected</span>`
    : `<span class="badge badge-bad">differs from expected</span>`;

  const rationale = (scorer, label) => {
    const text = c.rationales[scorer] || "";
    const status = c.statuses[scorer] || "";
    if (!text && !status) return "";
    return `<div class="rationale-block">
      <div class="rationale-head">
        <span class="rationale-name">${esc(label)}</span>
        ${status ? `<span class="rationale-status">status: ${esc(status)}</span>` : ""}
      </div>
      <div class="rationale-text">${esc(text || "–")}</div>
    </div>`;
  };

  // ── Tool classification (from tool_usage metadata) ──────────────────
  const tc = c.tool_classification || {};
  const tcSections = [
    {key: "correct",          label: "Correct",        cls: "badge-good"},
    {key: "incorrect",        label: "Incorrect",      cls: "badge-bad"},
    {key: "hallucinated",     label: "Hallucinated",   cls: "badge-bad"},
    {key: "missing_expected", label: "Missing",        cls: "badge-mid"},
  ];
  const tcAny = tcSections.some(s => (tc[s.key] || []).length);
  const tcHtml = tcAny ? `
    <div class="detail-section">
      <h3>Tool classification</h3>
      <dl class="kv">
        ${tcSections.map(s => {
          const items = tc[s.key] || [];
          if (!items.length) return "";
          const badges = items.map(t => `<span class="badge ${s.cls}">${esc(t)}</span>`).join(" ");
          return `<dt>${esc(s.label)}</dt><dd>${badges}</dd>`;
        }).join("")}
      </dl>
    </div>` : "";

  // ── Parameter breakdown (from tool_parameter metadata) ──────────────
  const pb = c.param_breakdown || {};
  const fmtNum = v => (v == null || Number.isNaN(v)) ? "–" : Number(v).toFixed(0);
  const fmtPct = v => (v == null || Number.isNaN(v)) ? "–" : (Number(v) * 100).toFixed(1) + "%";
  const pbAny = pb.expected_keys != null || pb.key_score != null;
  const pbHtml = pbAny ? `
    <div class="detail-section bordered">
      <h3>Parameter breakdown</h3>
      <dl class="kv">
        <dt>Key score</dt><dd>${fmtPct(pb.key_score)} <span class="muted-small" style="display:inline">(matched names / expected)</span></dd>
        <dt>Value score</dt><dd>${fmtPct(pb.value_score)} <span class="muted-small" style="display:inline">(correct values / expected)</span></dd>
        <dt>Totals</dt><dd>
          ${fmtNum(pb.expected_keys)} expected ·
          ${fmtNum(pb.matched_keys)} matched ·
          ${fmtNum(pb.correct_values)} correct ·
          ${fmtNum(pb.wrong_values)} wrong ·
          ${fmtNum(pb.missing_keys)} missing
        </dd>
      </dl>
    </div>` : "";

  // ── Extra invocations (tools called but not expected) ───────────────
  const eb = c.extra_by_tool || {};
  const ebEntries = Object.entries(eb).sort((a, b) => b[1] - a[1]);
  const ebHtml = ebEntries.length ? `
    <div class="detail-section">
      <h3>Extra invocations (called but not expected)</h3>
      <div>
        ${ebEntries.map(([tool, n]) => `<span class="badge badge-mid"><code>${esc(tool)}</code> × ${n}</span>`).join(" ")}
      </div>
    </div>` : "";


  detailEl.innerHTML = `
    <h2 class="case-detail-title" style="font-size:16px;margin-bottom:8px">
      <span style="font-weight:700">${esc(c.id)}</span>
      ${outcomeBadge(c)}
      ${compareBadge(c)}
      ${c.domain ? `<span class="badge badge-na">${esc(c.domain)}</span>` : ""}
      ${c.persona ? `<span class="badge badge-na">${esc(c.persona)}</span>` : ""}
      ${spanErrorChipHtml(c)}
      ${c.trace_id ? `<span style="font-size:11px;color:#5c7999">trace: <code>${esc(c.trace_id)}</code></span>` : ""}
    </h2>
    ${compareBannerHtml(c)}
    <div style="margin-bottom:10px">${routeChips(c)}</div>
    ${scorerPlanLine(c)}
    ${scoreStripHtml(c)}

    <div class="detail-section">
      <h3>User query</h3>
      <div class="body">${esc(c.query || "–")}</div>
    </div>

    <div class="detail-row">
      <div class="detail-section bordered">
        <h3>Expected</h3>
        <dl class="kv">
          <dt>Agent</dt><dd><code>${expectedAgent}</code></dd>
          <dt>Tools</dt><dd>${expectedTools}</dd>
          <dt>Guidelines</dt><dd>${guidelines}</dd>
        </dl>
      </div>
      <div class="detail-section bordered">
        <h3>Actual</h3>
        <dl class="kv">
          <dt>Agent</dt><dd><code>${actualAgent}</code></dd>
          <dt>Tools</dt><dd>${actualTools}</dd>
          <dt>Tool sequence</dt><dd>${seqMatch}</dd>
        </dl>
      </div>
    </div>

    ${pbHtml}
    ${tcHtml}
    ${ebHtml}

    <div class="detail-row">
      <div class="detail-section">
        <h3>Expected tool calls</h3>
        ${expectedToolCallsHtml(c)}
      </div>
      <div class="detail-section">
        <h3>Actual tool calls</h3>
        ${actualToolCallsHtml(c)}
      </div>
    </div>

    <div class="detail-row">
      <div class="detail-section">
        <h3>Expected response</h3>
        <div class="body">${c.expected_response ? esc(c.expected_response) : '<em class="lang-fallback">(none)</em>'}</div>
      </div>
      <div class="detail-section">
        <h3>Actual response
          ${c.actual_response_en ? `<span class="lang-switch" data-target="actual-resp-body">
            <button type="button" class="lang-seg active" data-lang="orig">orig</button>
            <button type="button" class="lang-seg" data-lang="en">EN</button>
          </span>` : ""}
          <button type="button" class="body-toggle" data-target="actual-resp-body">expand</button>
        </h3>
        <div class="bodywrap" id="actual-resp-body">
          <div class="body lang-orig">${esc(c.actual_response || "–")}</div>
          ${c.actual_response_en ? `<div class="body lang-en" style="display:none">${esc(c.actual_response_en)}</div>` : ""}
        </div>
      </div>
    </div>

    ${enumSectionHtml(c)}
  `;
}

// ── Pretty tool-call rendering ──────────────────────────────────────────
// `expected_tool_calls` entries have {step, tool, parameters, reason}.
// `actual_tool_calls` entries are richer: {tool, step, arguments,
// parameters, outputs_text/outputs_obj, tool_call_id, error, ...}.
// Both render as a stack of cards so the structure is scannable.

function fmtJson(value, indent) {
  indent = indent == null ? 2 : indent;
  if (value == null) return "–";
  try { return JSON.stringify(value, null, indent); }
  catch (_) { return String(value); }
}

function tryParseJson(s) {
  // Tool outputs are stored as JSON strings inside the trace. Parse them
  // so we can pretty-print; fall back to the raw string on failure.
  if (s == null) return null;
  if (typeof s !== "string") return s;
  try { return JSON.parse(s); }
  catch (_) { return s; }
}

function paramsBlock(params) {
  if (!params || (typeof params === "object" && !Array.isArray(params) && Object.keys(params).length === 0)) {
    return `<div class="tc-empty">no parameters</div>`;
  }
  return `<pre class="tc-pre">${esc(fmtJson(params))}</pre>`;
}

function outputBlock(entry) {
  // Pick the richest representation available, parse JSON if it's a string.
  const raw = entry.outputs_obj && entry.outputs_obj.content
    ? entry.outputs_obj.content
    : (entry.outputs_text || entry.outputs_preview || null);
  if (!raw) return `<div class="tc-empty">no output</div>`;
  const parsed = tryParseJson(raw);
  if (typeof parsed === "string") {
    return `<pre class="tc-pre">${esc(parsed)}</pre>`;
  }
  return `<pre class="tc-pre">${esc(fmtJson(parsed))}</pre>`;
}

function statusBadge(entry) {
  if (entry.error) return `<span class="badge badge-bad" title="error">error</span>`;
  const s = (entry.span_status_code || entry.tool_message_status || "").toString().toLowerCase();
  if (s.startsWith("ok") || s.startsWith("success")) return `<span class="badge badge-good">${esc(s)}</span>`;
  if (s) return `<span class="badge badge-mid">${esc(s)}</span>`;
  return "";
}

function fmtParamVal(v) {
  if (v == null) return `<span class="pe-null">null</span>`;
  if (typeof v === "object") return `<code>${esc(JSON.stringify(v))}</code>`;
  return `<code>${esc(String(v))}</code>`;
}

// Render an expected entry's parameters as key/value rows, preserving
// the original expected order and values. Coloring (when a per_entry
// record is available):
//   matched (right value passed)        → green
//   wrong   (key passed but wrong value) → red
//   missing (key never passed)           → orange
// Falls back to default-color rows when there's no per_entry record
// (e.g. tool_parameter scoring didn't run for this row).
function expectedParamBlock(expectedEntry, per, excusedKeys) {
  const params = (expectedEntry && expectedEntry.parameters) || {};
  const keys = Object.keys(params);
  if (!keys.length) return `<div class="tc-empty">no parameters</div>`;

  const matchedKeys = new Set(((per && per.matched) || []).map(m => m.key));
  const wrongKeys   = new Set(((per && per.wrong)   || []).map(w => w.key));
  const missingKeys = new Set((per && per.missing_keys) || []);
  const excused     = new Set(excusedKeys || []);

  const rows = keys.map(k => {
    let cls = "";
    // Excused keys win over the matched/wrong/missing buckets — the
    // scorer never saw them, so colouring them as "matched" or "wrong"
    // would be misleading.
    if (excused.has(k)) {
      cls = "pe-excused";
    } else if (per) {
      if (matchedKeys.has(k))      cls = "pe-matched";
      else if (wrongKeys.has(k))   cls = "pe-wrong";
      else if (missingKeys.has(k)) cls = "pe-missing";
    }
    const tip = excused.has(k) ? ` title="excused by relaxation rules — not scored"` : "";
    return `<div class="pe-row ${cls}"${tip}>
      <span class="pe-key">${esc(k)}</span>
      <span class="pe-val">${fmtParamVal(params[k])}</span>
    </div>`;
  });
  return `<div class="pe-rows">${rows.join("")}</div>`;
}

// Walk expected_tool_calls and per_entry in tandem. per_entry is one
// record per scorable expected entry; match by tool name in order so a
// row with multiple calls to the same tool aligns correctly.
function buildPerEntryMap(expectedList, perList) {
  const map = new Map();
  if (!perList || !perList.length) return map;
  let pi = 0;
  for (let i = 0; i < expectedList.length; i++) {
    const e = expectedList[i];
    const expectedTool = (e.tool || e.name || "").toLowerCase();
    while (pi < perList.length && (perList[pi].tool || "").toLowerCase() !== expectedTool) {
      pi++;
    }
    if (pi < perList.length) {
      map.set(i, perList[pi]);
      pi++;
    }
  }
  return map;
}

function expectedToolCallsHtml(c) {
  const list = c.expected_tool_calls || [];
  if (!list.length) {
    return `<div class="tc-empty">(no expected tool calls)</div>`;
  }
  const perMap = buildPerEntryMap(list, c.param_per_entry || []);
  const hasAnyPerEntry = perMap.size > 0;
  const excusedPerCall = c.expected_excused || [];
  const hasAnyExcused = excusedPerCall.some(xs => (xs || []).length);
  return `<div class="tc-list">${list.map((e, i) => {
    const per = perMap.get(i);
    const excused = excusedPerCall[i] || [];
    const scoreTag = per && per.value_score != null
      ? `<span class="pe-scores">values ${(Number(per.value_score) * 100).toFixed(0)}%</span>`
      : "";
    return `
    <div class="tc-card">
      <div class="tc-head">
        <span class="tc-step">${esc(String(e.step != null ? e.step : (i + 1)))}.</span>
        <code class="tc-tool">${esc(e.tool || e.name || "?")}</code>
        ${scoreTag}
      </div>
      ${e.reason ? `<div class="tc-reason">${esc(e.reason)}</div>` : ""}
      <div class="tc-section-label">parameters</div>
      ${expectedParamBlock(e, per, excused)}
    </div>`;
  }).join("")}${(hasAnyPerEntry || hasAnyExcused) ? `<div class="pe-legend">green = correct · red = wrong value · orange = missing key${hasAnyExcused ? ` · <span class="pe-excused-swatch">grey</span> = excused (not scored)` : ""}</div>` : ""}</div>`;
}

// For an actual call's `arguments` dict, render the same key/value row
// format used by Expected. When a per_entry record is available for the
// matching expected entry, color-code:
//   matched key (right value)  → green
//   wrong key (wrong value)    → red
//   extra key (not in expected) → default
function actualParamRows(args, per, excusedKeys) {
  if (!args || (typeof args === "object" && !Array.isArray(args) && Object.keys(args).length === 0)) {
    return `<div class="tc-empty">no parameters</div>`;
  }
  const matched = new Set((per && per.matched || []).map(m => m.key));
  const wrong   = new Set((per && per.wrong   || []).map(w => w.key));
  const excused = new Set(excusedKeys || []);
  const rows = Object.entries(args).map(([k, v]) => {
    let cls = "";
    if (excused.has(k))      cls = "pe-excused";
    else if (matched.has(k)) cls = "pe-matched";
    else if (wrong.has(k))   cls = "pe-wrong";
    const tip = excused.has(k) ? ` title="excused by relaxation rules — not scored"` : "";
    return `<div class="pe-row ${cls}"${tip}>
      <span class="pe-key">${esc(k)}</span>
      <span class="pe-val">${fmtParamVal(v)}</span>
    </div>`;
  });
  return `<div class="pe-rows">${rows.join("")}</div>`;
}

// per_entry.matched_actual_index points into the actual_tool_calls list,
// so build {actual_index → per_entry} for the actual side.
function buildActualPerMap(perList) {
  const map = new Map();
  for (const p of (perList || [])) {
    if (p && p.matched_actual_index != null) {
      map.set(p.matched_actual_index, p);
    }
  }
  return map;
}

function actualToolCallsHtml(c) {
  const list = c.actual_tool_calls || [];
  if (!list.length) return `<div class="tc-empty">(no actual tool calls)</div>`;
  const actMap = buildActualPerMap(c.param_per_entry || []);
  const excusedPerCall = c.actual_excused || [];
  return `<div class="tc-list">${list.map((e, i) => {
    const per = actMap.get(i);
    const excused = excusedPerCall[i] || [];
    return `
    <div class="tc-card">
      <div class="tc-head">
        <span class="tc-step">${esc(String(e.step != null ? e.step : (i + 1)))}.</span>
        <code class="tc-tool">${esc(e.tool || e.name || "?")}</code>
        ${statusBadge(e)}
      </div>
      <div class="tc-section-label">arguments</div>
      ${actualParamRows(e.arguments || e.parameters, per, excused)}
      <details class="tc-output">
        <summary>output</summary>
        ${outputBlock(e)}
      </details>
      ${e.error_message ? `<div class="tc-section-label">error</div><pre class="tc-pre tc-error">${esc(e.error_message)}</pre>` : ""}
    </div>`;
  }).join("")}</div>`;
}

// Expand/collapse buttons inside the detail panel.
document.addEventListener("click", e => {
  const expandBtn = e.target.closest && e.target.closest(".body-toggle");
  if (expandBtn) {
    const wrap = document.getElementById(expandBtn.dataset.target);
    if (!wrap) return;
    const expanded = wrap.classList.toggle("expanded");
    expandBtn.textContent = expanded ? "collapse" : "expand";
  }
});

// Language switch (Actual response). Segmented two-button control —
// `data-lang` on the clicked segment decides which body is shown. The
// non-clicked segment becomes the next click target, so the label
// always describes a switch action rather than a current state.
document.addEventListener("click", e => {
  const seg = e.target.closest && e.target.closest(".lang-seg");
  if (!seg) return;
  const switchEl = seg.closest(".lang-switch");
  if (!switchEl) return;
  const wrap = document.getElementById(switchEl.dataset.target);
  if (!wrap) return;
  const orig = wrap.querySelector(".lang-orig");
  const en   = wrap.querySelector(".lang-en");
  if (!orig || !en) return;
  const lang = seg.dataset.lang;
  if (lang === "en") {
    orig.style.display = "none";
    en.style.display = "";
  } else {
    orig.style.display = "";
    en.style.display = "none";
  }
  switchEl.querySelectorAll(".lang-seg").forEach(b =>
    b.classList.toggle("active", b === seg));
});

// Drill-down links: Summary table cells + Top Problem case links.
document.addEventListener("click", e => {
  const caseA = e.target.closest && e.target.closest("a.case-link");
  if (caseA) {
    e.preventDefault();
    clearAllFilters();
    document.querySelector('[data-target="tab-cases"]').click();
    renderList();
    selectCase(caseA.dataset.case);
    return;
  }
  const outA = e.target.closest && e.target.closest("a.fm-clear-link");
  if (outA) {
    e.preventDefault();
    clearAllFilters();
    activeFilters.outcome = outA.dataset.outcome;
    document.querySelector('[data-target="tab-cases"]').click();
    renderList();
  }
  const groupA = e.target.closest && e.target.closest("a.group-filter-link");
  if (groupA) {
    e.preventDefault();
    const group = groupA.dataset.group;
    const value = groupA.dataset.value;
    if (group === "domains" || group === "personas") {
      clearAllFilters();
      activeFilters[group].add(value);
      document.querySelector('[data-target="tab-cases"]').click();
      renderList();
    }
  }
  const compareA = e.target.closest && e.target.closest("[data-compare-filter]");
  if (compareA) {
    e.preventDefault();
    const v = compareA.dataset.compareFilter;
    if (v) {
      clearAllFilters();
      activeFilters.compare.add(v);
      document.querySelector('[data-target="tab-cases"]').click();
      renderList();
    }
    return;
  }
  const scorerA = e.target.closest && e.target.closest("a.scorer-filter-link");
  if (scorerA) {
    e.preventDefault();
    const scorer = scorerA.dataset.scorer;
    const bucket = scorerA.dataset.bucket;
    if (scorer && bucket) {
      clearAllFilters();
      activeFilters.scorer_pairs = [{scorer, value: bucket}];
      // Pop open the Scorers matrix in the sidebar so the active filter is visible.
      const det = document.querySelector(".dim-matrix-details");
      if (det) det.open = true;
      document.querySelector('[data-target="tab-cases"]').click();
      renderList();
    }
  }
});

searchEl.addEventListener("input", renderList);
bindChips();
bindRange("param-min", "param-max", "param-reset", "param_min", "param_max");
bindSidebarToggle();
renderList();
"""


# ─── render_html ────────────────────────────────────────────────────────

def render_html(df: pd.DataFrame, *, input_path: Path, output_path: Path,
                baseline: dict[str, dict[str, Any]] | None = None,
                baseline_path: Path | None = None,
                prompts_path: Path | None = None) -> str:
    metrics = compute_metrics(df, baseline=baseline)
    summaries = scorer_summary(df)
    cases = case_payload(df, baseline=baseline)
    # MLflow run time = earliest trace request_time (ms-since-epoch). For
    # a single-run CSV this equals when the run started; for a mixed CSV
    # it's the start of the earliest trace, which is still the most useful
    # anchor. Falls back to "–" when the column is missing or unparseable.
    rt = pd.to_numeric(df.get("request_time", pd.Series(dtype="float64")), errors="coerce")
    if rt.notna().any():
        run_time = datetime.fromtimestamp(rt.min() / 1000).strftime("%b %d, %Y, %I:%M %p")
    else:
        run_time = "–"
    source_run_id_raw = _unique_label(df, "source_run_id")
    mlflow_run_id = source_run_id_raw
    sidecar = _load_prompt_sidecar(input_path, source_run_id_raw, prompts_path)
    prompt_warnings = _prompt_hash_warnings(df)
    prompt_warning_card = _prompt_warning_card(prompt_warnings)
    prompts_tab_html = _prompts_tab(sidecar, source_run_id_raw)
    mlflow_user = _unique_label(df, "mlflow_user")
    kb_version_label = _unique_label(df, "kb_version")
    git_branch = _unique_label(df, "git_branch")
    _git_commit_full = _unique_label(df, "git_commit")
    git_commit = (_git_commit_full[:10] if _git_commit_full not in ("–", None) and len(_git_commit_full) >= 10
                  else _git_commit_full)
    js = JS.replace("__CASES__", json.dumps(cases, ensure_ascii=False))
    # Inject the baseline filename into the JS (used by the case-detail
    # comparison banner). Empty string when no baseline was loaded — the
    # JS reads that as "skip the banner".
    baseline_name = baseline_path.name if baseline_path is not None else ""
    js = js.replace("__BASELINE_NAME__", json.dumps(baseline_name, ensure_ascii=False))

    # Header line surfacing the baseline filename — only when --baseline
    # was provided so the existing single-input report is byte-identical.
    if baseline_path is not None:
        baseline_header_line = (
            f"<span>Baseline: <code>{_h(baseline_path.name)}</code></span>"
        )
    else:
        baseline_header_line = ""

    domains = sorted({str(v) for v in df.get("eval_domain", pd.Series(dtype=str)).dropna().unique() if str(v).strip()})
    personas = sorted({str(v) for v in df.get("eval_persona", pd.Series(dtype=str)).dropna().unique() if str(v).strip()})
    tools_set: set[str] = set()
    for lst in df.get("_expected_tools", []):
        tools_set.update(lst or [])
    for lst in df.get("_actual_tools", []):
        tools_set.update(lst or [])
    tools = sorted(t for t in tools_set if t)

    outcome_group = _chip_group(
        "Outcome",
        "outcome",
        [
            ("pass", "Pass"),
            ("fail", "Fail"),
        ],
    )
    domain_group = _chip_group("Domain", "domains", [(d, d) for d in domains], foldable=True) if domains else ""
    persona_group = _chip_group("Persona", "personas", [(p, p) for p in personas], foldable=True) if personas else ""
    tool_group = _chip_group("Tool", "tools", [(t, t) for t in tools], foldable=True) if tools else ""

    # "Vs. baseline" filter group — only rendered when --baseline was provided.
    # Multi-select (Set semantics in JS, matching domains / personas / tools),
    # so e.g. "Regression + Persistent fail" can be combined.
    if baseline is not None:
        compare_group = _chip_group(
            "Vs. baseline",
            "compare",
            [
                ("regression",         "Regression"),
                ("persistent_failure", "Persistent fail"),
                ("fixed",              "Fixed"),
                ("new_case",           "New"),
            ],
        )
    else:
        compare_group = ""

    # Tooltip strings hoisted out of the f-string template — newlines
    # inside an `_info_icon(...)` call inside an f-string expression are
    # rejected by the f-string parser ("backslash in expression part").
    tool_freq_tip = (
        "One row per tool that appears in expected_tool_calls or "
        "actual_tool_calls.\n"
        "  • Calls (exp / act) = total times this tool was expected vs "
        "actually called across all cases (multi-step cases contribute "
        "multiple calls). Bold = expected, light = actual.\n"
        "  • Tool usage pass = pass rate of tool_usage_score, restricted "
        "to cases that include this tool in expected_tools.\n"
        "  • Params mean = mean of tool_parameter_score, restricted to "
        "the same case subset. knowledge_search is shown as “–” because "
        "its parameters (fuzzy semantic strings — queries, facets) are "
        "not meaningfully comparable by the deterministic scorer."
    )
    scores_by_domain_tip = (
        "Per-domain breakdown. Cases = number of test cases in "
        "that domain. The three score columns are the mean of "
        "each scorer's score column, grouped by domain:\n"
        "  • Routing = mean of agent_routing_score (binary 0/1) "
        "→ equivalent to the pass rate.\n"
        "  • Tool = mean of tool_usage_score (binary 0/1) → "
        "equivalent to the pass rate.\n"
        "  • Params = mean of tool_parameter_score (numeric "
        "0–1), computed only over rows where the parameter "
        "scorer ran (not-scored rows excluded). Cell tint: "
        "green ≥ 80%, amber ≥ 50%, red < 50%."
    )

    return f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>API Evaluation Report</title>
<style>{CSS}</style>
</head>
<body>

<div class="sticky-top">
<header class="report-header">
  <div class="header-inner">
    <h1>API Evaluation Report</h1>
    <div class="header-meta header-meta-grid">
      <div class="header-col">
        <span>Run time: <strong>{_h(run_time)}</strong></span>
        <span>MLflow run: <code>{_h(mlflow_run_id)}</code></span>
        <span>MLflow user: <strong>{_h(mlflow_user)}</strong></span>
      </div>
      <div class="header-col">
        <span>Test cases: <strong>{len(df)}</strong></span>
        <span>KB version: <code>{_h(kb_version_label)}</code></span>
        <span>Git branch: <code>{_h(git_branch)}</code></span>
        <span>Git commit: <code>{_h(git_commit)}</code></span>
      </div>
      <div class="header-footer">
        <span>Input: <code>{_h(input_path.name)}</code></span>
        {baseline_header_line}
      </div>
    </div>
  </div>
</header>

<nav class="tab-nav">
  <div class="tab-nav-inner">
    <button class="tab-btn tab-home active" data-target="tab-summary">Summary</button>
    <button class="tab-btn" data-target="tab-cases">Test Cases</button>
    <button class="tab-btn" data-target="tab-prompts">Prompts</button>
  </div>
</nav>
</div>

<main class="content">

  <div id="tab-summary" class="tab-panel active">
    {prompt_warning_card}
    <div class="headline-row">
      {_summary_cards(metrics)}
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-title">Primary Outcomes {_info_icon(
            "Priority assignment per case: routing failure → tool usage failure → "
            "tool-parameter mismatch → pass. Parameter rows that were not scored "
            "are treated as not applicable (count toward pass). Click a count to "
            "filter the Test Cases tab to those rows.")}
        </div>
        {_issue_table(df)}
      </div>
      {_group_table(df, "eval_domain", "Scores By Domain",
                    col_label="domain", tip=scores_by_domain_tip)}
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-title">Per-Scorer Summary {_info_icon(
            "One row per scorer. Hover the info icon next to each scorer code for "
            "the exact match rule. Pass / Partial / Fail are counts; Pass rate is "
            "Pass / Scored; Mean is the mean of the numeric score column.")}
        </div>
        {_scorer_summary_table(summaries)}
      </div>
      <div class="card">
        <div class="card-title">Tool Frequency {_info_icon(tool_freq_tip)}
        </div>
        {_tool_frequency_table(df)}
      </div>
    </div>

    <div class="card">
      <div class="card-title">Top Problem Cases {_info_icon(
          "Sorted by number of failed scorer areas (descending), then by worst "
          "individual score. Click a case id to open it in the Test Cases tab.")}
      </div>
      {_top_problem_cases(df)}
    </div>
  </div>

  <div id="tab-cases" class="tab-panel">
    <div class="cases-layout">
      <button id="sidebar-toggle" class="sidebar-toggle" type="button"
              title="Hide sidebar" aria-label="Toggle sidebar">‹</button>
      <div class="peek-zone" aria-hidden="true"></div>
      <aside class="cases-sidebar">
        <input id="case-search" class="case-search" type="text" placeholder="Search id, query, agent, tool, response…">
        <div class="filters-wrap">
          {outcome_group}
          <div style="margin-top:6px"></div>
          {_scorer_matrix(df)}
          {domain_group}
          {persona_group}
          {tool_group}
          <div class="filter-group">
            <span class="filter-label">Tool param</span>
            <div class="range-filter">
              <input id="param-min" type="number" min="0" max="1" step="0.05" placeholder="min">
              <span>–</span>
              <input id="param-max" type="number" min="0" max="1" step="0.05" placeholder="max">
              <button class="range-reset" id="param-reset">reset</button>
            </div>
          </div>
          {compare_group}
          <div class="filter-group">
            <span class="filter-label">Flags</span>
            <button class="chip chip-err" data-group="flag" data-value="span_errors"
                    title="Cases with at least one span-level error (info.state=OK but a child span crashed — e.g. CancelledError, ConnectTimeout)">Span errors</button>
          </div>
          <div class="filter-group" style="justify-content:flex-end">
            <button class="chip chip-clear" id="chip-clear">Clear all</button>
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

  <div id="tab-prompts" class="tab-panel">
    {prompts_tab_html}
  </div>

</main>

<script>{js}</script>
</body>
</html>
"""


# ─── CLI ────────────────────────────────────────────────────────────────

def default_input_path() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent / "input" / "traces_offline_smoke_pr12_kb_smoke_infer_score.csv"


def default_output_path(input_path: Path) -> Path:
    here = Path(__file__).resolve().parent
    return here / "reports" / f"{input_path.stem}_api_report.html"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_input_path(), help="Scored traces CSV.")
    parser.add_argument("--output", type=Path, default=None, help="Output HTML report path.")
    parser.add_argument("--baseline", type=Path, default=None,
                        help="Optional previous-run enriched-traces CSV. When set, "
                             "the report adds Regression / Persistent / Fixed / New "
                             "filters on the Test Cases tab and a Regressions card "
                             "on the Summary tab. Cases are joined by test_case_id.")
    parser.add_argument("--prompts", type=Path, default=None,
                        help="Path to the prompt sidecar JSON (or its directory). "
                             "If omitted, the report looks for "
                             "prompt_{source_run_id}.json next to the input CSV.")
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")
    output_path = (args.output.expanduser().resolve() if args.output else default_output_path(input_path))

    baseline_path: Path | None = None
    baseline_lookup: dict[str, dict[str, Any]] | None = None
    if args.baseline is not None:
        baseline_path = args.baseline.expanduser().resolve()
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline CSV does not exist: {baseline_path}")
        baseline_lookup = load_baseline(baseline_path)
        print(f"[baseline] loaded {len(baseline_lookup)} cases from {baseline_path.name}")

    df = pd.read_csv(input_path)
    df = enrich(df)
    html_text = render_html(df, input_path=input_path, output_path=output_path,
                            baseline=baseline_lookup, baseline_path=baseline_path,
                            prompts_path=args.prompts)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
