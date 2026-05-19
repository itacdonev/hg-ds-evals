"""Per-trace latency breakdown extracted from MLflow span trees.

Given a list of MLflow spans (the normalised shape used elsewhere in this
package — see :func:`hg_ds_evals.preprocessing.skkb_traces._normalize_span`)
this module bucketises wall-clock time into a small fixed set of pipeline
steps so the reports can surface where each trace spent its time.

The bucket definitions are based on inspection of real
``daily_banking_agent`` + ``knowledge_search`` traces:

* ``routing``         — all ``main_agent`` spans (the top-level router LLM).
* ``planning_llm``    — every ``llm`` span inside a sub-agent that is NOT
                        the last ``llm`` for that sub-agent invocation,
                        i.e. the LLM calls that decide which tool to call.
* ``generation_llm``  — the last ``llm`` per sub-agent invocation,
                        i.e. the LLM call that produces the final answer.
* ``tools``           — ``TOOL``-typed spans excluding ``knowledge_search``.
* ``kb_retrieve``     — ``retrieve`` nodes inside ``knowledge_search``.
* ``kb_prune``        — ``prune`` nodes inside ``knowledge_search``.
* ``kb_rerank``       — ``rerank`` nodes inside ``knowledge_search`` (this
                        wraps a reranker LLM call, so its time is already
                        accounted for here and we do not also count it under
                        ``generation_llm``).
* ``overhead``        — total minus the sum of the above (LangGraph wiring,
                        ``tools_condition`` routing, etc.).

The buckets are designed to be **mutually non-overlapping**: each named span
is a langgraph DAG node whose parent edge encodes *runs-after-this-node*
rather than temporal containment, so we can sum their durations safely.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from typing import Any


# Step labels in the canonical order used by the reports.
STEP_LABELS = (
    "routing",
    "planning_llm",
    "kb_retrieve",
    "kb_prune",
    "kb_rerank",
    "tools",
    "generation_llm",
    "overhead",
)

# Retry-overhead heuristic. There is no explicit retry marker on OTEL
# CHAT_MODEL spans — the Databricks/LangChain SDK swallows 429s and
# retries internally, so the back-off sleep is invisibly included in the
# span duration. We approximate by flagging calls whose duration is
# substantially above what is normal for their pipeline bucket AND whose
# output-tokens-per-second is well below the bucket's healthy floor.
#
# Per-bucket thresholds matter because the buckets have *very* different
# natural distributions:
#   - routing      : small structured output (tool call ~26 tokens),
#                    healthy dur p95 ≈ 3s, healthy tok/s p10 ≈ 12.
#   - sub_agent_llm: variable output, healthy dur p95 ≈ 17s, tok/s p10 ≈ 19.
#   - kb_rerank    : tiny structured output (~40 tokens — list of selected
#                    IDs), tok/s is intrinsically low (p10 ≈ 6) because the
#                    work is reading large input candidates, not generating.
#                    A flat "tok/s < 5" rule misfires here. Healthy dur
#                    p95 ≈ 9s.
# Defaults are calibrated against the May 2026 czkb online run.
RETRY_BASELINE_TOK_PER_S = 30.0   # used to estimate "expected" generation time
# Legacy scalar — kept for backwards compatibility and as the "other"
# bucket fallback. Per-bucket overrides live in the dicts below.
RETRY_MIN_DURATION_MS = 5000
RETRY_MAX_TOK_PER_S = 5.0

# Bucket-specific MINIMUM wall-clock duration before a span is even
# considered for retry-flagging. Roughly 2× the bucket's healthy p95 so
# normal-busy calls are not flagged. Override via
# :func:`extract_latency_breakdown(thresholds=...)` if calibrating for a
# different model / endpoint.
RETRY_DUR_THRESHOLDS_MS: dict[str, float] = {
    "routing":        6_000.0,
    "sub_agent_llm":  35_000.0,
    "kb_rerank":      18_000.0,
    "kb_other":       18_000.0,
    "other":          float(RETRY_MIN_DURATION_MS),
}

# Bucket-specific UPPER bound on output_tokens / second. A call with
# tok/s above this can't be a retry — it's producing tokens at a healthy
# rate. Sits just below the bucket's healthy p10 on calibration data.
RETRY_MAX_TOK_PER_S_BY_BUCKET: dict[str, float] = {
    "routing":        5.0,
    "sub_agent_llm":  8.0,
    "kb_rerank":      4.0,
    "kb_other":       4.0,
    "other":          RETRY_MAX_TOK_PER_S,
}

# These overlap with the LLM buckets above (routing / planning_llm /
# generation_llm / kb_rerank all contain CHAT_MODEL spans whose retries
# count toward the overhead). Reported separately so consumers can choose
# whether to subtract from the bucket totals to get "clean" LLM time.
LATENCY_COLUMNS = (
    "lat_total_ms",
    *(f"lat_{label}_ms" for label in STEP_LABELS),
    "lat_steps_json",
    "lat_retry_overhead_ms",
    "lat_retry_call_count",
    "lat_retries_json",
)

_SUB_AGENT_NAMES = frozenset({"daily_banking_agent", "hg-invest-phase2"})
_ROUTING_NAMES = frozenset({"main_agent"})
_KB_RETRIEVE_NAMES = frozenset({"retrieve"})
_KB_PRUNE_NAMES = frozenset({"prune"})
_KB_RERANK_NAMES = frozenset({"rerank"})
_KB_WRAPPER_NAME = "knowledge_search"
_LLM_NAME = "llm"


def _empty_breakdown() -> dict[str, Any]:
    out: dict[str, Any] = {"lat_total_ms": None}
    for label in STEP_LABELS:
        out[f"lat_{label}_ms"] = None
    out["lat_steps_json"] = json.dumps([])
    out["lat_retry_overhead_ms"] = None
    out["lat_retry_call_count"] = None
    out["lat_retries_json"] = json.dumps([])
    return out


def _parse_token_usage(attrs: Mapping[str, Any] | None) -> dict[str, int]:
    """Decode ``mlflow.chat.tokenUsage`` attribute into a dict.

    The attribute is usually a JSON-encoded string ({input_tokens,
    output_tokens, total_tokens, cache_read_input_tokens}) but some
    exporters round-trip it as a plain dict. Returns ``{}`` when
    unparseable so callers can short-circuit cleanly.
    """
    if not attrs:
        return {}
    raw = attrs.get("mlflow.chat.tokenUsage")
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return {k: int(v) for k, v in raw.items() if isinstance(v, (int, float))}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {}
        if isinstance(parsed, Mapping):
            return {k: int(v) for k, v in parsed.items() if isinstance(v, (int, float))}
    return {}


def _classify_llm_call(
    span: Mapping[str, Any],
    ancestor_names: list[str],
) -> str:
    """Coarse bucket label for one CHAT_MODEL span — matches the LLM-side
    STEP_LABELS plus ``other``. Used to attribute retry overhead so the
    case-detail panel can say "this was a retry on the routing call"."""
    ancestor_set = set(ancestor_names)
    if _KB_RERANK_NAMES & ancestor_set or _KB_WRAPPER_NAME in ancestor_set:
        # Reranker LLM (or any other LLM inside knowledge_search).
        return "kb_rerank" if _KB_RERANK_NAMES & ancestor_set else "kb_other"
    if _SUB_AGENT_NAMES & ancestor_set:
        # Inside a sub-agent — could be planning or generation; we don't
        # distinguish here because that ordering is done elsewhere.
        return "sub_agent_llm"
    if _ROUTING_NAMES & ancestor_set:
        return "routing"
    return "other"


def _evaluate_chat_model_for_retry(
    span: Mapping[str, Any],
    ancestor_names: list[str],
    *,
    dur_thresholds: Mapping[str, float] | None = None,
    tok_per_s_caps: Mapping[str, float] | None = None,
) -> dict[str, Any] | None:
    """Decide whether one CHAT_MODEL span looks like a hidden retry.

    Bucket-aware: the duration cutoff and tok/s cap are looked up by the
    span's pipeline bucket (``routing``, ``sub_agent_llm``, ``kb_rerank``,
    ``kb_other``, ``other``). This prevents the rerank false-positive
    problem where rerank LLM calls have intrinsically low tok/s (small
    structured output) and would otherwise be flagged by a flat threshold.

    Returns a per-call payload with overhead estimate when flagged,
    ``None`` when the span is too short / missing tokens / above the
    bucket's throughput cap.

    ``dur_thresholds`` and ``tok_per_s_caps`` default to module-level
    constants (:data:`RETRY_DUR_THRESHOLDS_MS`,
    :data:`RETRY_MAX_TOK_PER_S_BY_BUCKET`); pass overrides to calibrate
    for a different model / endpoint without editing the module.
    """
    dt = RETRY_DUR_THRESHOLDS_MS if dur_thresholds is None else dur_thresholds
    tc = RETRY_MAX_TOK_PER_S_BY_BUCKET if tok_per_s_caps is None else tok_per_s_caps
    bucket = _classify_llm_call(span, ancestor_names)
    min_dur = float(dt.get(bucket, dt.get("other", RETRY_MIN_DURATION_MS)))
    max_tps = float(tc.get(bucket, tc.get("other", RETRY_MAX_TOK_PER_S)))
    dur_ms = _span_duration_ms(span)
    if dur_ms < min_dur:
        return None
    usage = _parse_token_usage(span.get("attributes"))
    output_tokens = usage.get("output_tokens", 0)
    if output_tokens <= 0 or dur_ms <= 0:
        return None
    tok_per_s = output_tokens * 1000.0 / dur_ms
    if tok_per_s >= max_tps:
        return None
    # Estimated overhead = wall time minus what generation "should" have
    # taken at the baseline rate. Clamped at 0 to avoid negatives when a
    # heuristic-flagged call is faster than baseline (shouldn't happen
    # given the threshold, but defensive).
    expected_gen_ms = output_tokens * 1000.0 / RETRY_BASELINE_TOK_PER_S
    overhead_ms = max(0.0, dur_ms - expected_gen_ms)
    return {
        "name": str(span.get("name") or ""),
        "bucket": bucket,
        "dur_ms": round(dur_ms, 3),
        "output_tokens": output_tokens,
        "input_tokens": usage.get("input_tokens", 0),
        "tok_per_s": round(tok_per_s, 3),
        "overhead_ms": round(overhead_ms, 3),
        # Carry the thresholds that decided this flag so consumers can
        # explain the call ("flagged because dur > 18s AND tok/s < 4")
        # without re-importing the constants. Kept short so the JSON
        # payload doesn't bloat.
        "dur_threshold_ms": min_dur,
        "tok_per_s_cap": max_tps,
    }


def _span_duration_ms(span: Mapping[str, Any]) -> float:
    start = span.get("start_time_unix_nano")
    end = span.get("end_time_unix_nano")
    if start is None or end is None:
        return 0.0
    try:
        return max(0.0, (int(end) - int(start)) / 1e6)
    except (TypeError, ValueError):
        return 0.0


def _parse_span_type(attrs: Mapping[str, Any] | None) -> str:
    if not attrs:
        return ""
    raw = attrs.get("mlflow.spanType")
    if raw is None:
        return ""
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return str(parsed) if parsed is not None else ""
        except (json.JSONDecodeError, ValueError):
            return raw
    return str(raw)


def _ancestor_names(
    span_id: str | None, parent_of: Mapping[str, str | None], by_id: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    cur = parent_of.get(span_id) if span_id else None
    while cur and cur not in seen:
        seen.add(cur)
        sp = by_id.get(cur)
        if not sp:
            break
        names.append(str(sp.get("name") or ""))
        cur = parent_of.get(cur)
    return names


def _find_root(spans: Iterable[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    no_parent = [s for s in spans if not s.get("parent_span_id")]
    if no_parent:
        return max(no_parent, key=_span_duration_ms)
    spans_list = list(spans)
    return max(spans_list, key=_span_duration_ms) if spans_list else None


def extract_latency_breakdown(spans: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute per-step latency in ms for one trace.

    Returns a dict with one ``lat_<step>_ms`` column per :data:`STEP_LABELS`,
    plus ``lat_total_ms`` and a JSON-encoded ``lat_steps_json`` list of
    ``{label, ms, n}`` entries (used by the test-case detail tooltip).

    If the trace is empty or has no usable timestamps every numeric value is
    ``None`` and ``lat_steps_json`` is ``"[]"``.
    """
    spans_list = [s for s in spans if isinstance(s, Mapping)]
    if not spans_list:
        return _empty_breakdown()

    by_id: dict[str, Mapping[str, Any]] = {}
    for sp in spans_list:
        sid = sp.get("span_id")
        if sid:
            by_id[str(sid)] = sp
    parent_of: dict[str, str | None] = {
        str(sp["span_id"]): (str(sp["parent_span_id"]) if sp.get("parent_span_id") else None)
        for sp in spans_list
        if sp.get("span_id")
    }

    root = _find_root(spans_list)
    if root is None:
        return _empty_breakdown()
    total_ms = _span_duration_ms(root)

    routing_ms = 0.0
    tools_ms = 0.0
    kb_retrieve_ms = 0.0
    kb_prune_ms = 0.0
    kb_rerank_ms = 0.0
    n_routing = 0
    n_tools = 0
    n_kb_retrieve = 0
    n_kb_prune = 0
    n_kb_rerank = 0

    # Collected separately so we can split last-in-time per sub-agent into generation.
    llm_spans: list[tuple[str, Mapping[str, Any]]] = []  # (sub_agent_name, span)

    # Retry detection runs in the same loop. CHAT_MODEL is where the
    # ``mlflow.chat.tokenUsage`` attribute lives, and where the throttling
    # back-off sleep is silently absorbed. We walk every CHAT_MODEL span
    # in the trace regardless of which bucket it belongs to (routing,
    # planning, generation, reranker), so the total here covers all LLM
    # retry overhead in the trace.
    retry_calls: list[dict[str, Any]] = []

    for sp in spans_list:
        name = str(sp.get("name") or "")
        sid = str(sp.get("span_id") or "")
        attrs = sp.get("attributes") or {}
        ancestor_names = _ancestor_names(sid, parent_of, by_id)
        ancestor_set = set(ancestor_names)
        inside_kb = _KB_WRAPPER_NAME in ancestor_set or name == _KB_WRAPPER_NAME
        span_type = _parse_span_type(attrs)

        if span_type == "CHAT_MODEL":
            retry = _evaluate_chat_model_for_retry(sp, ancestor_names)
            if retry is not None:
                retry_calls.append(retry)
            # Don't ``continue`` — CHAT_MODEL spans don't match any of
            # the step-bucket name filters below, so falling through is
            # safe and keeps the next pass intact.

        if name in _ROUTING_NAMES:
            # main_agent spans only appear at the top level; if one ever
            # nests inside a sub-agent treat it as overhead, not routing.
            if not (ancestor_set & _SUB_AGENT_NAMES):
                routing_ms += _span_duration_ms(sp)
                n_routing += 1
            continue

        if name in _KB_RETRIEVE_NAMES and inside_kb:
            kb_retrieve_ms += _span_duration_ms(sp)
            n_kb_retrieve += 1
            continue
        if name in _KB_PRUNE_NAMES and inside_kb:
            kb_prune_ms += _span_duration_ms(sp)
            n_kb_prune += 1
            continue
        if name in _KB_RERANK_NAMES and inside_kb:
            kb_rerank_ms += _span_duration_ms(sp)
            n_kb_rerank += 1
            continue

        if name == _LLM_NAME and not inside_kb:
            sub_agent = next(
                (a for a in ancestor_names if a in _SUB_AGENT_NAMES),
                None,
            )
            if sub_agent is not None:
                llm_spans.append((sub_agent, sp))
            continue

        if _parse_span_type(attrs) == "TOOL" and name != _KB_WRAPPER_NAME and not inside_kb:
            tools_ms += _span_duration_ms(sp)
            n_tools += 1
            continue

    # Split llm spans into planning vs generation, per sub-agent invocation.
    # In LangGraph agent loops the last LLM call per agent is the one whose
    # output (no tool_calls) is the final answer — we use end-time order as a
    # robust proxy because the parent_span_id graph is data-flow shaped, not
    # temporal.
    planning_llm_ms = 0.0
    generation_llm_ms = 0.0
    n_planning = 0
    n_generation = 0

    by_agent: dict[str, list[Mapping[str, Any]]] = {}
    for sub_agent, sp in llm_spans:
        by_agent.setdefault(sub_agent, []).append(sp)
    for agent_spans in by_agent.values():
        agent_spans.sort(key=lambda s: int(s.get("end_time_unix_nano") or 0))
        for sp in agent_spans[:-1]:
            planning_llm_ms += _span_duration_ms(sp)
            n_planning += 1
        # The last llm span in temporal order is generation.
        last = agent_spans[-1]
        generation_llm_ms += _span_duration_ms(last)
        n_generation += 1

    bucket_total = (
        routing_ms
        + planning_llm_ms
        + generation_llm_ms
        + tools_ms
        + kb_retrieve_ms
        + kb_prune_ms
        + kb_rerank_ms
    )
    overhead_ms = max(0.0, total_ms - bucket_total)

    step_ms = {
        "routing": (routing_ms, n_routing),
        "planning_llm": (planning_llm_ms, n_planning),
        "kb_retrieve": (kb_retrieve_ms, n_kb_retrieve),
        "kb_prune": (kb_prune_ms, n_kb_prune),
        "kb_rerank": (kb_rerank_ms, n_kb_rerank),
        "tools": (tools_ms, n_tools),
        "generation_llm": (generation_llm_ms, n_generation),
        "overhead": (overhead_ms, 1 if total_ms else 0),
    }
    steps_payload = [
        {"label": label, "ms": round(step_ms[label][0], 3), "n": step_ms[label][1]}
        for label in STEP_LABELS
    ]

    out: dict[str, Any] = {"lat_total_ms": round(total_ms, 3)}
    for label in STEP_LABELS:
        out[f"lat_{label}_ms"] = round(step_ms[label][0], 3)
    out["lat_steps_json"] = json.dumps(steps_payload, ensure_ascii=False)

    # Retry overhead — NOT a step (overlaps with the LLM buckets). Sum
    # of estimated back-off time across all CHAT_MODEL spans flagged by
    # the throughput heuristic.
    retry_overhead_ms = round(sum(r["overhead_ms"] for r in retry_calls), 3)
    out["lat_retry_overhead_ms"] = retry_overhead_ms
    out["lat_retry_call_count"] = len(retry_calls)
    out["lat_retries_json"] = json.dumps(retry_calls, ensure_ascii=False)
    return out


def build_latency_dataframe(traces_df) -> "pd.DataFrame":  # type: ignore[name-defined]
    """Apply :func:`extract_latency_breakdown` to a mlflow traces DataFrame.

    ``traces_df`` must have a ``trace_id`` column and a ``spans`` column whose
    cells are lists (as returned by ``mlflow.search_traces``) or string-encoded
    lists (as written by ``DataFrame.to_csv``). Returns a new DataFrame keyed
    by ``trace_id`` with the latency columns.
    """
    import ast
    import pandas as pd

    def _coerce_spans(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(text)
                except Exception:  # noqa: BLE001 — try the other parser
                    continue
                if isinstance(parsed, list):
                    return parsed
                return []
        return []

    records: list[dict[str, Any]] = []
    for trace_id, spans in zip(traces_df["trace_id"], traces_df["spans"], strict=True):
        breakdown = extract_latency_breakdown(_coerce_spans(spans))
        breakdown["trace_id"] = trace_id
        records.append(breakdown)
    return pd.DataFrame.from_records(records).set_index("trace_id")
