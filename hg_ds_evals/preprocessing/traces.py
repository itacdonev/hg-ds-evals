"""Unified MLflow trace preprocessing.

Replaces the two legacy modules ``skkb_traces`` and ``mlflow_traces``.
Both public entry points and their dataclasses live here:

- :func:`build_skkb_dataframe_from_mlflow_search_traces` — SKKB / CZKB
  KB-pipeline analytics. One row per trace with KB retrieval, prune,
  reranker, query-scope, invariants, and prompt-hash columns.
- :func:`build_dataframe_from_mlflow_traces` — canonical eval scoring.
  One row per trace with ``expected_*`` / ``actual_*`` columns matching
  the ai-data-science scorer contract.

The legacy module paths still work via thin re-export shims.

Single source of truth for the agent's final answer
---------------------------------------------------
``agent_response`` / ``actual_response`` come **only** from the
``agent_answer`` AGENT span (``mlflow.spanOutputs.answer``) emitted by
the orchestrator post GCAI-2566 / GCAI-3581. Missing span ⇒ ``""``.

The previous "longest stopped AI message" heuristic was dropped — it
silently selected sub-agent reasoning or tool-prep narration when those
happened to be longer than the user-facing reply, contaminating eval
scores. Coverage is now honest: rows with no canonical span surface as
empty rather than as a wrong guess.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


# ── Constants ──────────────────────────────────────────────────────────

AI_TYPE_STRINGS = frozenset({"ai", "AIMessage", "AIMessageChunk"})

HG_INVEST_AGENT_NAME = "hg-invest-phase2"
KNOWN_AGENT_NAMES = frozenset({"main_agent", "daily_banking_agent", HG_INVEST_AGENT_NAME})
NO_AGENT = ""

KB_TOOL_NAMES = frozenset({"knowledge_search"})
BANKING_DATA_TOOL_PREFIX = "mock_banking_"
# Placeholder — confirm once hg-invest tools are wired (mirrors banking prefix).
HG_INVEST_DATA_TOOL_PREFIX = "mock_invest_"

ENUM_RE = re.compile(r"(?m)^([A-Z][A-Z0-9_@-]*(?:\s*\+\s*[A-Z][A-Z0-9_@-]*)?)(?=:)")
RERANK_CANDIDATE_RE = re.compile(
    r"(?ms)^- ID:\s*(?P<enum_id>[^|\n]+?)\s*\|\s*"
    r"Group:\s*(?P<group_name>[^|\n]+?)\s*\|\s*"
    r"Description:\s*(?P<description>.*?)(?=^- ID:\s|\Z)"
)
_TS_TAIL_RE = re.compile(r"Current date and time:.*$", re.DOTALL)
_TEST_CASE_ID_RE = re.compile(r"(?i)\btest case \d+\b")

# Span-type literals (after JSON-decoding ``mlflow.spanType``).
_TOOL = "TOOL"
_CHAT_MODEL = "CHAT_MODEL"
_CHAIN = "CHAIN"
_AGENT = "AGENT"

# Post-#388 supervisor (ReAct tool-calling pattern) delegates to a child
# agent via a synthetic TOOL whose name is ``transfer-to-<slug>``. From
# the eval-scoring contract's perspective these are not real tool calls
# — they're agent invocations — so we filter them out of
# ``actual_tool_calls`` and surface them via ``actual_agents_path``
# instead. The slug after the prefix may be kebab-case (commit
# ``100f2a2b`` allowed kebab-case agent slugs); the canonical agent
# name comes from the ``SubagentToolResult.agent`` field inside the
# TOOL span's output payload.
_TRANSFER_TOOL_PREFIX = "transfer-to-"

# Trace-level keys carrying a native test-case id.
TRACE_TEST_CASE_KEYS = (
    "eval.test_case_id",   # NEW orchestrator (trace_schema v3 / GCAI-3581)
    "test_case_id",         # legacy
    "testCaseId",
    "test_case",
    "dataset_example_id",
    "datasetExampleId",
)

_PRUNE_LEAF_SPAN_NAMES = (
    "knowledge_prune",   # NEW orchestrator (GCAI-3581+)
    "kb_prune",          # legacy
)


# ── Assessment names + aliases (canonical eval surface) ────────────────

class AssessmentNames:
    """Assessment names attached to each eval_item trace.

    Contract between ``ai-orchestrator/run_evals.py`` and downstream
    scoring. Stored via ``mlflow.update_current_trace`` in the eval
    harness.
    """
    EVAL_ITEM_ID = "eval_item_id"
    EVAL_DOMAIN = "eval_domain"
    EVAL_PERSONA = "eval_persona"
    EXPECTED_AGENT = "expected_agent"
    EXPECTED_TOOL_CALLS = "expected_tool_calls"
    EXPECTED_RESPONSE = "expected_response"
    GUIDELINES = "guidelines"
    SCORERS = "scorers"
    # Per-ENUM relevance weights set by the test author for KB cases.
    # Same payload shape used by SKKB (dict[str, int|float]); the keys
    # are the expected ENUM IDs and the values are relevance weights.
    TARGET_ENUMS_TO_RELEVANCE = "target_enums_to_relevance"


# Read the canonical assessment name first, fall back to legacy. Each
# entry is canonical → legacy.
_LEGACY_ASSESSMENT_NAMES: dict[str, str] = {
    AssessmentNames.EXPECTED_TOOL_CALLS: "expected_tools",
}

# Scorer-name aliases (eval team sometimes ships under a different name).
_SCORER_NAME_ALIASES: dict[str, str] = {
    "tool_input_parameters": "tool_parameter",
}


# ── Result dataclasses ─────────────────────────────────────────────────

@dataclass
class ParseError:
    trace_id: str
    error: str


@dataclass
class SKKBParseResult:
    """SKKB/CZKB flat table + diagnostics."""

    dataframe: pd.DataFrame
    parse_errors: list[tuple[str, str]]
    unmapped_trace_ids: list[str]


@dataclass
class MlflowParseResult:
    """Eval-scoring DataFrame + diagnostics + cross-trace tool registry."""

    dataframe: pd.DataFrame
    parse_errors: list[ParseError] = field(default_factory=list)
    untagged_trace_ids: list[str] = field(default_factory=list)
    run_tool_registry: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class _CandidateDocument:
    enum_id: str
    description: str
    group_name: str = ""


# ── Generic conversion helpers ─────────────────────────────────────────

def _to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _to_plain_data(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {key: _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain_data(item) for item in value]
    if isinstance(value, (tuple, set)):
        return [_to_plain_data(item) for item in value]
    for attr_name in ("to_dictionary", "to_dict", "model_dump", "dict"):
        attr = getattr(value, attr_name, None)
        if callable(attr):
            try:
                return _to_plain_data(attr())
            except TypeError:
                continue
    value_dict = getattr(value, "__dict__", None)
    if isinstance(value_dict, dict) and value_dict:
        return {
            key: _to_plain_data(item)
            for key, item in value_dict.items()
            if not key.startswith("_")
        }
    return value


def _coerce_mapping(value: Any) -> dict[str, Any]:
    parsed = _parse_json_like_payload(_to_plain_data(value))
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _coerce_list(value: Any) -> list[Any]:
    parsed = _parse_json_like_payload(_to_plain_data(value))
    return parsed if isinstance(parsed, list) else []


def _decode_json(value: Any) -> Any:
    """Best-effort JSON decode; returns the original on failure."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _parse_attr_json(attrs: Mapping[str, Any], key: str) -> Any:
    """Parse a span-attribute value that may be a JSON string."""
    raw = attrs.get(key)
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    return raw


def _parse_json_like_payload(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(value)
            except (SyntaxError, ValueError):
                return None
    return None


def _coerce_mapping_payload(value: Any) -> dict[str, Any]:
    parsed = _parse_json_like_payload(value)
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _walk_dicts(obj: Any) -> Iterable[dict[str, Any]]:
    """Yield every dict reachable inside a nested dict/list structure."""
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _walk_dicts(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_dicts(item)


def _md5_short(value: str) -> str:
    return hashlib.md5((value or "").encode("utf-8", errors="ignore")).hexdigest()[:10]


# ── Span normalization & tree helpers ──────────────────────────────────

def normalize_mlflow_trace_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one ``mlflow.search_traces`` row to ``{info, data}`` shape.

    Accepts either:

    - JSONL shape with top-level ``info`` and ``data`` keys (passthrough), or
    - the flat ``search_traces`` shape (``trace_id``, ``trace_metadata``,
      ``assessments``, ``spans``, …).
    """
    if "info" in row and "data" in row:
        return {
            "info": _coerce_mapping(row.get("info")),
            "data": _coerce_mapping(row.get("data")),
        }
    return {
        "info": {
            "trace_id": row.get("trace_id"),
            "client_request_id": row.get("client_request_id"),
            "state": row.get("state"),
            "request_time": row.get("request_time"),
            "execution_duration_ms": row.get("execution_duration_ms")
            or row.get("execution_duration"),
            "trace_metadata": _coerce_mapping(row.get("trace_metadata")),
            "tags": _coerce_mapping(row.get("tags")),
            "assessments": [
                _to_plain_data(a) for a in _coerce_list(row.get("assessments"))
            ],
        },
        "data": {
            "spans": [
                _normalize_span(span)
                for span in _coerce_list(row.get("spans"))
                if isinstance(_to_plain_data(span), dict)
            ],
        },
    }


def _normalize_span(span: Mapping[str, Any]) -> dict[str, Any]:
    span = _to_plain_data(span)
    return {
        "trace_id": span.get("trace_id"),
        "span_id": span.get("span_id"),
        "parent_span_id": span.get("parent_span_id"),
        "name": span.get("name"),
        "start_time_unix_nano": span.get("start_time_unix_nano"),
        "end_time_unix_nano": span.get("end_time_unix_nano"),
        "events": _coerce_list(span.get("events")),
        "status": _coerce_mapping(span.get("status")),
        "attributes": _coerce_mapping(span.get("attributes")),
        "inputs": _to_plain_data(span.get("inputs")),
        "outputs": _to_plain_data(span.get("outputs")),
    }


def _build_span_children(spans: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    children: dict[str, list[dict[str, Any]]] = {}
    for span in spans:
        parent_span_id = span.get("parent_span_id")
        if parent_span_id:
            children.setdefault(str(parent_span_id), []).append(span)
    return children


def _sort_spans_by_dependency_order(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(indexed_span: tuple[int, dict[str, Any]]) -> tuple[int, int, int]:
        index, span = indexed_span
        start_time = span.get("start_time_unix_nano")
        try:
            return (0, int(start_time), index)
        except (TypeError, ValueError):
            return (1, index, index)

    return [span for _, span in sorted(enumerate(spans), key=sort_key)]


def _find_descendant_by_name(
    root: dict[str, Any],
    children: dict[str, list[dict[str, Any]]],
    name: str,
) -> dict[str, Any] | None:
    for child in children.get(_to_str(root.get("span_id")), []):
        if child.get("name") == name:
            return child
        deeper = _find_descendant_by_name(child, children, name)
        if deeper is not None:
            return deeper
    return None


def _collect_span_descendants(
    root: dict[str, Any],
    children: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    descendants = [root]
    stack = list(children.get(_to_str(root.get("span_id")), []))
    while stack:
        current = stack.pop()
        descendants.append(current)
        stack.extend(children.get(_to_str(current.get("span_id")), []))
    return _sort_spans_by_dependency_order(descendants)


def _span_time_value(span: Mapping[str, Any], key: str) -> int | None:
    try:
        return int(span.get(key))
    except (TypeError, ValueError):
        return None


def _spans_in_time_window(
    root: Mapping[str, Any],
    spans: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    start = _span_time_value(root, "start_time_unix_nano")
    end = _span_time_value(root, "end_time_unix_nano")
    if start is None or end is None:
        return []
    scoped: list[dict[str, Any]] = []
    for span in spans:
        span_start = _span_time_value(span, "start_time_unix_nano")
        span_end = _span_time_value(span, "end_time_unix_nano") or span_start
        if span_start is None or span_end is None:
            continue
        if start <= span_start and span_end <= end:
            scoped.append(span)
    return _sort_spans_by_dependency_order(scoped)


def _merge_spans_by_id(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for span in spans:
        span_key = _to_str(span.get("span_id")) or str(id(span))
        if span_key in seen:
            continue
        seen.add(span_key)
        merged.append(span)
    return _sort_spans_by_dependency_order(merged)


def _span_payload(span: Mapping[str, Any], attr_key: str, direct_key: str) -> Any:
    parsed = _parse_attr_json(span.get("attributes") or {}, attr_key)
    if parsed is not None:
        return parsed
    return span.get(direct_key)


def _span_inputs(span: Mapping[str, Any]) -> Any:
    return _span_payload(span, "mlflow.spanInputs", "inputs")


def _span_outputs(span: Mapping[str, Any]) -> Any:
    return _span_payload(span, "mlflow.spanOutputs", "outputs")


def _span_type(span: Mapping[str, Any]) -> str:
    """Return the JSON-decoded ``mlflow.spanType`` (e.g. ``"TOOL"``)."""
    raw = (span.get("attributes") or {}).get("mlflow.spanType")
    if isinstance(raw, str) and raw.startswith('"'):
        try:
            return _to_str(json.loads(raw))
        except json.JSONDecodeError:
            return raw
    return _to_str(raw)


def _span_langgraph_node(span: Mapping[str, Any]) -> str:
    metadata = _parse_attr_json(span.get("attributes") or {}, "metadata")
    if not isinstance(metadata, Mapping):
        return ""
    node = metadata.get("langgraph_node")
    return node if isinstance(node, str) else ""


def _span_matches_node(span: Mapping[str, Any], node_names: set[str]) -> bool:
    return span.get("name") in node_names or _span_langgraph_node(span) in node_names


# ── Span-level errors ──────────────────────────────────────────────────
#
# Trace ``info.state`` reports the orchestrator-level outcome and stays
# ``OK`` whenever the agent returned *some* answer, even if a child span
# crashed (KB endpoint timeout, asyncio CancelledError on a retry, etc.).
# These helpers surface error signals from two places per span and OR
# them together:
#
#   - ``status.code == "STATUS_CODE_ERROR"`` (OpenTelemetry status), and
#   - ``events[].name == "exception"`` (OpenTelemetry exception event,
#     carrying ``exception.type`` / ``exception.message`` /
#     ``exception.stacktrace`` attributes).
#
# Output is a flat list of per-span error records that downstream
# reports can chip / drill into.

SPAN_ERROR_COLUMNS = (
    "trace_has_span_error",
    "span_error_count",
    "span_error_types_json",
    "span_errors_json",
)

_STACKTRACE_TAIL_LINES = 10


def _stacktrace_tail(value: Any, max_lines: int = _STACKTRACE_TAIL_LINES) -> str:
    """Trim a stacktrace to its last ``max_lines`` non-empty lines.

    Full stacktraces are multi-KB; the tail is the diagnostically useful
    bit (innermost frame + the exception line). Keeping a short slice
    means the column is safe to ship in a CSV / HTML report.
    """
    text = _to_str(value)
    if not text:
        return ""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    return "\n".join(lines)


def _span_exception_events(span: Mapping[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw in _coerce_list(span.get("events")):
        event = _to_plain_data(raw)
        if isinstance(event, dict) and event.get("name") == "exception":
            events.append(event)
    return events


def _is_span_status_error(status: Mapping[str, Any]) -> bool:
    code = _to_str(status.get("code")).upper()
    return bool(code) and "ERROR" in code


def extract_span_errors(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one record per error span (status ERROR OR exception event).

    Each record carries enough context for a downstream report to label
    *where* the error happened (span name / type / langgraph node) and
    *what* it was (status message + exception type/message + trimmed
    stacktrace tail). The last exception event on a span is kept (covers
    LangGraph's retry loop, where the final attempt is the one that
    bubbled out).
    """
    errors: list[dict[str, Any]] = []
    for span in spans:
        if not isinstance(span, Mapping):
            continue
        status = _coerce_mapping(span.get("status"))
        status_is_error = _is_span_status_error(status)
        exception_events = _span_exception_events(span)
        if not status_is_error and not exception_events:
            continue
        last_event = exception_events[-1] if exception_events else {}
        attrs = _coerce_mapping(last_event.get("attributes"))
        errors.append({
            "span_name": _to_str(span.get("name")),
            "span_type": _span_type(span),
            "langgraph_node": _span_langgraph_node(span),
            "status_code": _to_str(status.get("code")),
            "status_message": _to_str(status.get("message")),
            "exception_type": _to_str(attrs.get("exception.type")),
            "exception_message": _to_str(attrs.get("exception.message")),
            "stacktrace_tail": _stacktrace_tail(attrs.get("exception.stacktrace")),
        })
    return errors


def summarize_span_errors(errors: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll up :func:`extract_span_errors` into the four record columns.

    Types are deduped (preserve first-seen order). Lists are JSON-encoded
    so the columns round-trip through CSV unchanged.
    """
    types_seen: list[str] = []
    for entry in errors:
        etype = _to_str(entry.get("exception_type")) or _to_str(entry.get("status_code"))
        if etype and etype not in types_seen:
            types_seen.append(etype)
    return {
        "trace_has_span_error": bool(errors),
        "span_error_count": len(errors),
        "span_error_types_json": json.dumps(types_seen, ensure_ascii=False),
        "span_errors_json": json.dumps(errors, ensure_ascii=False),
    }


def build_span_errors_dataframe(traces_df: pd.DataFrame) -> pd.DataFrame:
    """Per-trace span-error summary indexed by ``trace_id`` (backfill aid).

    Mirrors :func:`hg_ds_evals.preprocessing.latency.build_latency_dataframe`:
    one row per trace, ``trace_id`` as the index, columns are exactly
    :data:`SPAN_ERROR_COLUMNS`. Joinable into an existing checkpoint
    without re-running the eval.
    """
    rows: list[dict[str, Any]] = []
    for raw_row in traces_df.to_dict(orient="records"):
        canonical = normalize_mlflow_trace_row(raw_row)
        info = canonical.get("info") or {}
        spans = (canonical.get("data") or {}).get("spans") or []
        spans = _sort_spans_by_dependency_order(spans)
        summary = summarize_span_errors(extract_span_errors(spans))
        summary["trace_id"] = _to_str(info.get("trace_id"))
        rows.append(summary)
    df = pd.DataFrame.from_records(rows)
    if df.empty:
        return pd.DataFrame(columns=list(SPAN_ERROR_COLUMNS))
    return df.set_index("trace_id")[list(SPAN_ERROR_COLUMNS)]


# ── Assessments ────────────────────────────────────────────────────────

def _assessment_name(assessment: Mapping[str, Any]) -> str:
    return _to_str(assessment.get("assessment_name") or assessment.get("name"))


def _decode_assessment_payload(payload: Any) -> Any:
    """Decode one ``expectation`` or ``feedback`` payload to a Python value.

    Handles two MLflow shapes:
    - ``{"value": <python>}`` → returned as-is
    - ``{"serialized_value": {"value": "<json>", "serialization_format": "JSON"}}``
      → JSON-decoded
    """
    if not isinstance(payload, Mapping):
        return None
    if "value" in payload:
        return payload["value"]
    sv = payload.get("serialized_value")
    if isinstance(sv, Mapping) and "value" in sv:
        raw = sv.get("value")
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
        return raw
    return None


def _extract_assessments(assessments_raw: Iterable[Any]) -> dict[str, Any]:
    """Decode HUMAN assessments into ``{name: deserialized_value}``.

    On duplicate names, keep the most recent by ``last_update_time``
    (lexicographic on ISO strings).
    """
    by_name: dict[str, tuple[str, Any]] = {}
    for a in assessments_raw:
        a = _to_plain_data(a)
        if not isinstance(a, dict):
            continue
        name = _to_str(a.get("assessment_name"))
        if not name:
            continue
        ts = _to_str(a.get("last_update_time"))
        if "expectation" in a:
            value = _decode_assessment_payload(a.get("expectation"))
        elif "feedback" in a:
            value = _decode_assessment_payload(a.get("feedback"))
        else:
            value = None
        prev = by_name.get(name)
        if prev is None or ts >= prev[0]:
            by_name[name] = (ts, value)
    return {name: value for name, (_, value) in by_name.items()}


def _extract_expected_enums_weights(
    assessment: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Pull ``{enum_id: weight}`` out of a ``target_enums_to_relevance`` assessment.

    Accepts either shape that the two call sites pass in:

    - **SKKB path** (``parse_trace_skkb``) passes the *raw assessment
      dict*, with ``expectation.serialized_value.value`` (JSON string)
      or ``expectation.value`` (decoded) or top-level ``value``.
    - **MLflow path** (``parse_trace_mlflow``) passes the
      *already-decoded* mapping, because ``_extract_assessments``
      peeled the ``expectation`` envelope upstream. The whole input is
      then the ``{enum_id: weight}`` dict itself.

    We try the SKKB candidate slots first; if none yield a mapping,
    fall back to treating the input AS the decoded mapping — but only
    when its values look numeric (so we don't mis-coerce a stray
    assessment dict that happens to lack the SKKB-shaped keys).
    """
    expectation = assessment.get("expectation") or {}
    candidates: list[Any] = []
    if isinstance(expectation, Mapping):
        serialized_value = expectation.get("serialized_value") or {}
        if isinstance(serialized_value, Mapping):
            candidates.append(serialized_value.get("value"))
        candidates.append(expectation.get("value"))
    candidates.append(assessment.get("value"))

    for candidate in candidates:
        weights = _coerce_mapping_payload(candidate)
        if weights:
            return json.dumps(weights, ensure_ascii=False), weights

    # MLflow path: ``assessment`` is itself the decoded
    # ``{enum_id: weight}`` mapping. Require numeric leaf values so a
    # raw assessment dict that fell through above (nested ``expectation``
    # / ``feedback`` etc.) doesn't get mis-parsed as the weights.
    if (
        isinstance(assessment, Mapping)
        and assessment
        and all(
            isinstance(v, (int, float)) and not isinstance(v, bool)
            for v in assessment.values()
        )
    ):
        weights = dict(assessment)
        return json.dumps(weights, ensure_ascii=False), weights

    return "{}", {}


def _normalize_expected_tool_calls(value: Any) -> list[dict[str, Any]]:
    """Coerce ``expected_tool_calls`` to the canonical dict-list shape.

    Accepts: missing/None → ``[]``; legacy list of tool-name strings →
    promoted to ``[{"tool": name, "parameters": {}}]``; current dict-list
    payload returned as-is.
    """
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            out.append(item)
        elif isinstance(item, str):
            out.append({"tool": item, "parameters": {}})
    return out


def _normalize_scorers_to_run(value: Any) -> list[str]:
    """Resolve scorer-name aliases (e.g. ``tool_input_parameters`` → ``tool_parameter``)."""
    if not isinstance(value, list):
        return []
    return [_SCORER_NAME_ALIASES.get(str(s), str(s)) for s in value if s is not None]


# ── test_case_id resolution ────────────────────────────────────────────

def _normalize_test_case_id_candidate(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    stripped = value.strip()
    if not stripped:
        return ""
    match = _TEST_CASE_ID_RE.search(stripped)
    if match is not None:
        return match.group(0)
    return stripped if stripped.lower().startswith("test case ") else ""


def _extract_test_case_id_from_assessments(assessments: Any) -> str:
    for assessment in assessments if isinstance(assessments, list) else []:
        if not isinstance(assessment, Mapping):
            continue
        if _assessment_name(assessment) not in TRACE_TEST_CASE_KEYS:
            continue
        for node in _walk_dicts(_to_plain_data(assessment)):
            for key in ("value", *TRACE_TEST_CASE_KEYS):
                candidate = _normalize_test_case_id_candidate(node.get(key))
                if candidate:
                    return candidate
    return ""


def _extract_trace_native_test_case_id(trace: Mapping[str, Any]) -> str:
    info = trace.get("info") or {}
    trace_metadata = info.get("trace_metadata") or {}
    tags = info.get("tags") or {}

    for container in (info, trace_metadata, tags):
        if not isinstance(container, Mapping):
            continue
        for key in TRACE_TEST_CASE_KEYS:
            candidate = _normalize_test_case_id_candidate(container.get(key))
            if candidate:
                return candidate

    trace_inputs = _parse_json_like_payload(trace_metadata.get("mlflow.traceInputs"))
    for node in _walk_dicts(trace_inputs):
        for key in TRACE_TEST_CASE_KEYS:
            candidate = _normalize_test_case_id_candidate(node.get(key))
            if candidate:
                return candidate

    from_assessments = _extract_test_case_id_from_assessments(info.get("assessments"))
    if from_assessments:
        return from_assessments

    client_request_id = _normalize_test_case_id_candidate(info.get("client_request_id"))
    if client_request_id:
        return client_request_id
    return ""


def resolve_test_case_id(trace: dict[str, Any]) -> str:
    """Resolve the SKKB-flavored test_case_id, falling back to ``trace_id``.

    Looks at trace metadata, tags, trace inputs, then assessments for a
    legacy-style identifier (``"Test case 17"`` etc.). Last resort is the
    trace_id itself.
    """
    native = _extract_trace_native_test_case_id(trace)
    if native:
        return native
    return _to_str((trace.get("info") or {}).get("trace_id"))


def _resolve_test_case_id_mlflow(
    *,
    assessments: Mapping[str, Any],
    user_query: str,
    trace_id: str,
) -> str:
    """Eval-flavor resolution: ``eval_item_id`` first, SKKB regex fallback, then trace_id."""
    candidate = _to_str(assessments.get(AssessmentNames.EVAL_ITEM_ID))
    if candidate:
        return candidate
    if user_query:
        match = _TEST_CASE_ID_RE.search(user_query)
        if match:
            return match.group(0).strip().lower().replace(" ", "_")
    return trace_id


# ── User query ─────────────────────────────────────────────────────────

def _user_query_payload_from_langgraph_root(
    spans: list[dict[str, Any]] | None,
) -> str:
    """Return ``mlflow.spanInputs`` from the root-child ``LangGraph`` span.

    Empty string when ``spans`` is missing, when no root span (parent is
    None) exists, or when no ``LangGraph`` child carries inputs.
    """
    if not spans:
        return ""
    root = next((s for s in spans if s.get("parent_span_id") is None), None)
    if root is None:
        return ""
    root_id = root.get("span_id")
    for span in spans:
        if span.get("name") != "LangGraph":
            continue
        if span.get("parent_span_id") != root_id:
            continue
        raw = (span.get("attributes") or {}).get("mlflow.spanInputs", "")
        if raw:
            return raw
    return ""


def _extract_user_query(
    trace_metadata: Mapping[str, Any],
    spans: list[dict[str, Any]] | None = None,
) -> str:
    """Resolve the user-facing prompt across orchestrator vintages.

    NEW (trace_schema v3, GCAI-3581+): read the ``LangGraph`` span that
    is the immediate child of the root span and pull ``mlflow.spanInputs``.
    In these traces ``trace_metadata["mlflow.traceInputs"]`` is empty.

    OLD (legacy orchestrator): fall back to
    ``trace_metadata["mlflow.traceInputs"]`` — same payload shape.
    """
    raw = _user_query_payload_from_langgraph_root(spans)
    if not raw:
        raw = trace_metadata.get("mlflow.traceInputs", "")
    if not raw:
        return ""
    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return ""
    if not isinstance(payload, Mapping):
        return ""
    messages = payload.get("messages") or []
    if messages and isinstance(messages[0], list) and len(messages[0]) >= 2:
        return messages[0][1] or ""
    if messages and isinstance(messages[0], dict):
        return messages[0].get("content", "") or ""
    return ""


# ── Agent answer (STRICT — single source of truth) ─────────────────────

def extract_agent_answer_span(spans: list[dict[str, Any]]) -> str:
    """Return the agent's final answer from the canonical ``agent_answer`` span.

    Reads ``mlflow.spanOutputs.answer`` from the AGENT span emitted by
    the orchestrator post GCAI-2566 / GCAI-3581. **No fallbacks** —
    returns ``""`` when the span is absent, when ``mlflow.spanOutputs``
    cannot be JSON-decoded, or when ``answer`` is missing/blank.

    This is the only place the agent's final answer is extracted.
    Previous heuristic paths (longest stopped AI message,
    ``traceOutputs.messages`` tail) were removed in favor of the canonical
    signal — heuristics silently produced wrong answers on traces where
    a sub-agent reasoning blob exceeded the user-facing reply.
    """
    for span in spans:
        if span.get("name") != "agent_answer":
            continue
        if _span_type(span) != _AGENT:
            continue
        parsed = _parse_attr_json(span.get("attributes") or {}, "mlflow.spanOutputs")
        if isinstance(parsed, Mapping):
            answer = parsed.get("answer")
            if isinstance(answer, str) and answer.strip():
                return answer
    return ""


# ── Agent path / classification ────────────────────────────────────────

def _is_subagent_transfer_tool(span: Mapping[str, Any]) -> bool:
    """A TOOL span whose name is ``transfer-to-<slug>`` — post-#388
    supervisor's delegate marker, not a callable tool."""
    return (
        _span_type(span) == _TOOL
        and _to_str(span.get("name")).startswith(_TRANSFER_TOOL_PREFIX)
    )


def _transfer_tool_agent_slug(span: Mapping[str, Any]) -> str:
    """Read the canonical sub-agent slug out of a transfer-to-* TOOL span.

    Prefers ``SubagentToolResult.agent`` from the output payload (commit
    ``100f2a2b`` allowed kebab-case tool slugs, so the payload field is
    the source of truth); falls back to the suffix of the tool name
    when the payload is absent (covers older traces written before the
    payload field landed).
    """
    out = _parse_attr_json(span.get("attributes") or {}, "mlflow.spanOutputs")
    if isinstance(out, Mapping):
        content = out.get("content")
        if isinstance(content, str):
            try:
                payload = json.loads(content)
                if isinstance(payload, Mapping):
                    slug = _to_str(payload.get("agent")).strip()
                    if slug:
                        return slug
            except json.JSONDecodeError:
                pass
    name = _to_str(span.get("name"))
    if name.startswith(_TRANSFER_TOOL_PREFIX):
        return name[len(_TRANSFER_TOOL_PREFIX):]
    return ""


# Post-#388 supervisor identity inference. The orchestrator no longer
# emits a CHAIN span named after the supervisor; we recover its identity
# from the set of ``transfer-to-<slug>`` tools registered in the
# supervisor's LLM call (``mlflow.chat.tools`` attribute on the first
# CHAT_MODEL span of the supervisor's graph). The mapping below converts
# that set to the canonical supervisor name expected by the test set's
# ``expected_agent`` field — same name the legacy contract produced via
# the (now-removed) CHAIN span, so :class:`RoutingCorrectnessScorer`
# keeps matching without any test-set rewrite.
_SUPERVISOR_BY_DELEGATE: tuple[tuple[frozenset[str], str], ...] = (
    (frozenset({"daily_banking_agent"}), "main_agent"),
    (frozenset({HG_INVEST_AGENT_NAME}),  HG_INVEST_AGENT_NAME),
)


def _supervisor_from_delegate_registry(spans: list[dict[str, Any]]) -> str:
    """Identify the post-#388 supervisor by which delegate tools its LLM
    call has access to. Works even when no delegation fired (chit-chat)
    because the tool-registry is set at LLM-binding time, not when a
    delegate is actually called.

    Returns the supervisor's canonical slug, or ``""`` when no
    transfer-to-* tool is registered anywhere (signal absent — don't
    invent one).
    """
    delegates: set[str] = set()
    for span in spans:
        if _span_type(span) != _CHAT_MODEL:
            continue
        attrs = span.get("attributes") or {}
        tools = _parse_attr_json(attrs, "mlflow.chat.tools")
        if not isinstance(tools, list):
            continue
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function") or {}
            name = _to_str(fn.get("name"))
            if name.startswith(_TRANSFER_TOOL_PREFIX):
                delegates.add(name[len(_TRANSFER_TOOL_PREFIX):])
    if not delegates:
        return ""
    for delegate_set, supervisor in _SUPERVISOR_BY_DELEGATE:
        if delegates & delegate_set:
            return supervisor
    return ""


def _extract_actual_agent_path(spans: list[dict[str, Any]]) -> tuple[list[str], str]:
    """Return ``(agents_path, last_agent)``.

    Three-path dispatch (no version flag — auto-detected from the trace):

    1. **Post-#388 (ReAct supervisor) with delegation**: each sub-agent
       is invoked via a ``transfer-to-<slug>`` TOOL span. Walk those in
       start-time order and read the canonical agent name from
       :func:`_transfer_tool_agent_slug`.
    2. **Post-#388 supervisor-direct (chit-chat)**: no transfer-to-*
       fired, but the supervisor's LLM call still has delegate tools
       *registered*. Infer the supervisor's name from that registry via
       :func:`_supervisor_from_delegate_registry`. Returns
       ``[supervisor]`` so ``last_agent`` is data-derived rather than
       defaulted by the UI.
    3. **Pre-#388 (handoff supervisor)**: each sub-agent emits its own
       CHAIN span whose name is in :data:`KNOWN_AGENT_NAMES`. Unchanged
       behavior — older runs still parse exactly as before.
    """
    # 1) Post-#388 delegated path.
    new_path: list[str] = []
    new_seen: set[str] = set()
    for span in _sort_spans_by_dependency_order(spans):
        if not _is_subagent_transfer_tool(span):
            continue
        slug = _transfer_tool_agent_slug(span)
        if slug and slug not in new_seen:
            new_seen.add(slug)
            new_path.append(slug)
    if new_path:
        return new_path, new_path[-1]

    # 2) Post-#388 chit-chat: no delegate fired, but the supervisor's
    #    LLM has transfer-to-* tools registered.
    supervisor = _supervisor_from_delegate_registry(spans)
    if supervisor:
        return [supervisor], supervisor

    # 3) Legacy contract: CHAIN spans named after the known agents.
    seen: set[str] = set()
    path: list[str] = []
    for span in spans:
        if _span_type(span) != _CHAIN:
            continue
        name = _to_str(span.get("name"))
        if name in KNOWN_AGENT_NAMES and name not in seen:
            seen.add(name)
            path.append(name)
    return path, (path[-1] if path else NO_AGENT)


def _classify_architecture(agents_path: list[str]) -> str:
    if "main_agent" in agents_path:
        return "classic"
    if HG_INVEST_AGENT_NAME in agents_path:
        return "hg-invest"
    return "none"


def extract_agent_and_tool_calls(spans: list[dict[str, Any]]) -> dict[str, Any]:
    """Collect every ``langgraph_node`` and every TOOL span (SKKB flavor).

    Handles both orchestrator contracts:

    - **Legacy**: each sub-agent ran its own LangGraph subgraph, so
      ``langgraph_node`` on inner spans equalled the agent slug
      (``main_agent``, ``daily_banking_agent``) and surfaced in
      ``agents_called`` directly.
    - **Post-#388**: sub-agents are invoked as ``transfer-to-<slug>``
      TOOL spans; the inner ``langgraph_node`` values are generic
      (``llm``, ``tools``, ``tools_condition``). We surface the
      delegate's canonical slug (from ``SubagentToolResult.agent``) as
      an agent in ``agents_called`` so :func:`classify_query_scope`
      can still detect ``daily_banking_agent`` reached, and we omit
      the transfer-to-* TOOL span from ``tools_called`` because from
      the eval scorer's perspective it is an agent invocation, not a
      tool call.
    """
    agents_seen: list[str] = []
    seen_set: set[str] = set()
    tools: list[dict[str, Any]] = []

    for span in _sort_spans_by_dependency_order(spans):
        attrs = span.get("attributes") or {}
        metadata = _parse_attr_json(attrs, "metadata")
        if isinstance(metadata, dict):
            node = metadata.get("langgraph_node")
            if isinstance(node, str) and node and node not in seen_set:
                seen_set.add(node)
                agents_seen.append(node)
        if _span_type(span) == _TOOL:
            if _is_subagent_transfer_tool(span):
                slug = _transfer_tool_agent_slug(span)
                if slug and slug not in seen_set:
                    seen_set.add(slug)
                    agents_seen.append(slug)
                continue
            tools.append({
                "name": span.get("name"),
                "inputs": _parse_attr_json(attrs, "mlflow.spanInputs"),
            })

    # Post-#388 chit-chat: no delegate fired, no legacy CHAIN names.
    # Infer the supervisor from its delegate-tool registry so this case
    # surfaces as "main_agent" instead of an empty path.
    if not any(a in KNOWN_AGENT_NAMES for a in agents_seen):
        supervisor = _supervisor_from_delegate_registry(spans)
        if supervisor and supervisor not in seen_set:
            seen_set.add(supervisor)
            agents_seen.append(supervisor)

    agents_only = [node for node in agents_seen if node in KNOWN_AGENT_NAMES]
    return {
        "agents_called": agents_seen,
        "agents_only": agents_only,
        "last_agent": agents_only[-1] if agents_only else "",
        "tools_called": tools,
    }


def classify_query_scope(
    agents_called: list[str],
    tools_called: list[dict[str, Any]],
) -> str:
    tool_names = {tool.get("name", "") for tool in tools_called or []}
    agents = agents_called or []
    reached_dba = "daily_banking_agent" in agents
    reached_hg_invest = HG_INVEST_AGENT_NAME in agents

    if reached_dba:
        if not tool_names:
            return "dba_no_tools"
        if tool_names & KB_TOOL_NAMES:
            return "kb"
        if any(name.startswith(BANKING_DATA_TOOL_PREFIX) for name in tool_names):
            return "mock_tool"
        return "other_tools"

    if reached_hg_invest:
        if not tool_names:
            return "hg_invest_no_tools"
        if tool_names & KB_TOOL_NAMES:
            return "hg_invest_kb"
        if any(name.startswith(HG_INVEST_DATA_TOOL_PREFIX) for name in tool_names):
            return "hg_invest_mock_tool"
        return "hg_invest_other_tools"

    return "main_agent"


def _violation_knowledge_search_requires_subagent(
    agents_called: list[str],
    tool_names: set[str],
    query_scope: str,
) -> bool:
    del query_scope
    if "knowledge_search" not in tool_names:
        return False
    agents = agents_called or []
    return ("daily_banking_agent" not in agents) and (HG_INVEST_AGENT_NAME not in agents)


def _violation_rerank_requires_knowledge_search(
    agents_called: list[str],
    tool_names: set[str],
    query_scope: str,
) -> bool:
    del query_scope
    if "rerank" not in (agents_called or []):
        return False
    return "knowledge_search" not in tool_names


def _violation_rule1_has_no_tools(
    agents_called: list[str],
    tool_names: set[str],
    query_scope: str,
) -> bool:
    del agents_called
    if query_scope not in {"dba_no_tools", "hg_invest_no_tools"}:
        return False
    return bool(tool_names)


def _violation_empty_tools_means_main_only(
    agents_called: list[str],
    tool_names: set[str],
    query_scope: str,
) -> bool:
    if tool_names or query_scope != "main_agent":
        return False
    return set(agents_called or []) - {"main_agent"} != set()


TRACE_INVARIANT_RULES = {
    "knowledge_search_requires_subagent": _violation_knowledge_search_requires_subagent,
    "rerank_requires_knowledge_search": _violation_rerank_requires_knowledge_search,
    "rule1_scope_has_no_tools": _violation_rule1_has_no_tools,
    "empty_tools_imply_main_agent_only": _violation_empty_tools_means_main_only,
}


def check_trace_invariants(
    agents_called: list[str],
    tools_called: list[dict[str, Any]],
    query_scope: str,
) -> list[str]:
    tool_names = {tool.get("name", "") for tool in tools_called or []}
    return [
        rule_id
        for rule_id, fn in TRACE_INVARIANT_RULES.items()
        if fn(agents_called, tool_names, query_scope)
    ]


# ── KB pipeline: candidate documents ───────────────────────────────────

def _candidate_document_from_mapping(value: Mapping[str, Any]) -> _CandidateDocument | None:
    fragment = value.get("fragment")
    if isinstance(fragment, Mapping):
        enum_id = fragment.get("id")
        if not isinstance(enum_id, str) or not enum_id:
            return None
        description = (
            fragment.get("description")
            or fragment.get("summary")
            or value.get("description")
            or value.get("summary")
            or ""
        )
        group_name = value.get("group_name") or value.get("groupName") or ""
        return _CandidateDocument(
            enum_id=enum_id,
            description=description if isinstance(description, str)
            else json.dumps(description, ensure_ascii=False),
            group_name=group_name if isinstance(group_name, str) else "",
        )
    enum_id = value.get("id")
    description = value.get("description")
    if isinstance(enum_id, str) and enum_id and isinstance(description, str):
        group_name = value.get("group_name") or value.get("groupName") or ""
        return _CandidateDocument(
            enum_id=enum_id,
            description=description,
            group_name=group_name if isinstance(group_name, str) else "",
        )
    return None


def _append_candidate_document(
    target: list[_CandidateDocument],
    candidate: _CandidateDocument,
    *,
    deduplicate: bool = True,
) -> None:
    if not deduplicate or candidate.enum_id not in {item.enum_id for item in target}:
        target.append(candidate)


def _extract_candidate_documents_from_payload(
    payload: Any,
    *,
    deduplicate: bool = True,
) -> list[_CandidateDocument]:
    parsed = _parse_json_like_payload(payload)
    if parsed is None:
        parsed = payload
    candidates: list[_CandidateDocument] = []

    def visit(node: Any) -> None:
        parsed_node = _parse_json_like_payload(node)
        if parsed_node is not None:
            node = parsed_node
        if isinstance(node, Mapping):
            candidate = _candidate_document_from_mapping(node)
            if candidate is not None:
                _append_candidate_document(candidates, candidate, deduplicate=deduplicate)
                return
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(parsed)
    return candidates


def _extract_reranker_prompt_candidate_documents(
    reranker_user_prompt: str,
) -> list[_CandidateDocument]:
    candidates: list[_CandidateDocument] = []
    for match in RERANK_CANDIDATE_RE.finditer(reranker_user_prompt or ""):
        enum_id = match.group("enum_id").strip()
        if not enum_id:
            continue
        _append_candidate_document(
            candidates,
            _CandidateDocument(
                enum_id=enum_id,
                group_name=match.group("group_name").strip(),
                description=match.group("description").strip(),
            ),
        )
    return candidates


def _format_retrieved_candidates(
    candidates: list[_CandidateDocument],
) -> tuple[str, list[str]]:
    blocks = [
        f"{candidate.enum_id}: {candidate.description}"
        for candidate in candidates
        if candidate.enum_id
    ]
    ids = [candidate.enum_id for candidate in candidates if candidate.enum_id]
    return "\n\n".join(blocks), ids


# ── KB pipeline: vector-DB HTTP spans ──────────────────────────────────

def _is_vector_db_http_span(span: Mapping[str, Any]) -> bool:
    if "/admin/knowledge-base/query" in _to_str(span.get("name")):
        return True
    span_inputs = _span_inputs(span)
    if not isinstance(span_inputs, Mapping):
        return False
    url = span_inputs.get("url")
    return isinstance(url, str) and "/admin/knowledge-base/query" in url


def _http_body_json(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    body_json = _parse_json_like_payload(payload.get("body_json"))
    if isinstance(body_json, Mapping):
        return dict(body_json)
    body = _parse_json_like_payload(payload.get("body"))
    return dict(body) if isinstance(body, Mapping) else {}


def _vector_db_http_items(span: Mapping[str, Any]) -> list[Any]:
    span_outputs = _span_outputs(span)
    if not isinstance(span_outputs, Mapping):
        return []
    body_json = _parse_json_like_payload(span_outputs.get("body_json"))
    if isinstance(body_json, Mapping):
        items = body_json.get("items")
        return items if isinstance(items, list) else []
    if isinstance(body_json, list):
        return body_json
    body = _parse_json_like_payload(span_outputs.get("body"))
    if isinstance(body, Mapping):
        items = body.get("items")
        return items if isinstance(items, list) else []
    return body if isinstance(body, list) else []


def extract_raw_vector_db_hits(spans: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-query vector DB hits before merge/dedup/prune."""
    hit_candidates: list[_CandidateDocument] = []
    hits_by_query: dict[str, int] = {}
    query_limits: list[int] = []

    for span in spans:
        if not _is_vector_db_http_span(span):
            continue
        request_body = _http_body_json(_span_inputs(span))
        query = _to_str(request_body.get("query")) or f"span:{_to_str(span.get('span_id'))}"
        if query in hits_by_query:
            query = f"{query} [{_to_str(span.get('span_id'))}]"
        limit = request_body.get("limit")
        if isinstance(limit, int):
            query_limits.append(limit)
        candidates = _extract_candidate_documents_from_payload(
            _vector_db_http_items(span),
            deduplicate=False,
        )
        hits_by_query[query] = len(candidates)
        hit_candidates.extend(candidates)

    candidates_text, hit_ids = _format_retrieved_candidates(hit_candidates)
    return {
        "raw_vector_db_retrieved_candidates_text": candidates_text,
        "raw_vector_db_retrieved_enum_ids": hit_ids,
        "raw_vector_db_retrieved_enum_count": len(hit_ids),
        "raw_vector_db_query_count": len(hits_by_query),
        "raw_vector_db_retrieved_count_by_query": hits_by_query,
        "raw_vector_db_query_limits": query_limits,
    }


# ── KB pipeline: pre/post-prune and reranker ───────────────────────────

def extract_pre_prune_candidates(spans: list[dict[str, Any]]) -> tuple[str, list[str]]:
    for span in spans:
        if not _span_matches_node(span, {"retrieve"}):
            continue
        candidates = _extract_candidate_documents_from_payload(_span_outputs(span))
        if candidates:
            return _format_retrieved_candidates(candidates)
    return "", []


def extract_post_prune_candidates(spans: list[dict[str, Any]]) -> tuple[str, list[str]]:
    for span in spans:
        if not _span_matches_node(span, {"prune"}):
            continue
        candidates = _extract_candidate_documents_from_payload(_span_outputs(span))
        if candidates:
            return _format_retrieved_candidates(candidates)
    return "", []


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def extract_prune_counts(spans: list[dict[str, Any]]) -> dict[str, int | bool]:
    """Pruning counters from the prune-leaf span.

    Accepts either ``knowledge_prune`` (GCAI-3581+) or ``kb_prune``
    (legacy). First match wins.
    """
    empty = {
        "prune_counts_available": False,
        "prune_candidates_in": 0,
        "prune_candidates_out": 0,
        "prune_candidates_dropped": 0,
    }
    for span in spans:
        if span.get("name") not in _PRUNE_LEAF_SPAN_NAMES:
            continue
        inputs = _span_inputs(span) if isinstance(_span_inputs(span), Mapping) else {}
        outputs = _span_outputs(span) if isinstance(_span_outputs(span), Mapping) else {}
        return {
            "prune_counts_available": True,
            "prune_candidates_in": _int_or_zero(inputs.get("candidates_in")),
            "prune_candidates_out": _int_or_zero(outputs.get("candidates_out")),
            "prune_candidates_dropped": _int_or_zero(outputs.get("candidates_dropped")),
        }
    return empty


def extract_retrieved_candidates(
    spans: list[dict[str, Any]],
    reranker_user_prompt: str = "",
) -> tuple[str, list[str]]:
    """Candidate pool offered to the reranker.

    Prefers the reranker prompt (exact retrieve/prune → rerank handoff)
    when present; falls back to structured node outputs.
    """
    prompt_candidates = _extract_reranker_prompt_candidate_documents(reranker_user_prompt)
    if prompt_candidates:
        return _format_retrieved_candidates(prompt_candidates)

    for node_name in ("prune", "retrieve"):
        for span in spans:
            if not _span_matches_node(span, {node_name}):
                continue
            candidates = _extract_candidate_documents_from_payload(_span_outputs(span))
            if candidates:
                return _format_retrieved_candidates(candidates)

    for span in spans:
        if not _span_matches_node(span, {"LangGraph", "knowledge_search"}):
            continue
        candidates = _extract_candidate_documents_from_payload(_span_outputs(span))
        if candidates:
            return _format_retrieved_candidates(candidates)
    return "", []


def extract_kb_context(spans: list[dict[str, Any]]) -> str:
    for span in spans:
        if span.get("name") != "knowledge_search":
            continue
        parsed = _parse_attr_json(span.get("attributes") or {}, "mlflow.spanOutputs")
        if isinstance(parsed, dict):
            content = parsed.get("content")
            if isinstance(content, str) and content.strip():
                return content
    return ""


def extract_reranked_enum_ids(kb_context: str) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for match in ENUM_RE.findall(kb_context or ""):
        if match not in seen:
            seen.add(match)
            output.append(match)
    return output


def extract_kb_version(
    spans: list[dict[str, Any]],
    reranker_user_prompt: str = "",
) -> str:
    for node_name in ("prune", "retrieve"):
        for span in spans:
            if not _span_matches_node(span, {node_name}):
                continue
            attrs = span.get("attributes") or {}
            span_inputs = _parse_attr_json(attrs, "mlflow.spanInputs")
            if isinstance(span_inputs, dict):
                names = span_inputs.get("knowledge_group_names") or []
                if isinstance(names, list) and names and isinstance(names[0], str):
                    return names[0]
            span_outputs = _parse_attr_json(attrs, "mlflow.spanOutputs")
            for candidate in _extract_candidate_documents_from_payload(span_outputs):
                if candidate.group_name:
                    return candidate.group_name

    for candidate in _extract_reranker_prompt_candidate_documents(reranker_user_prompt):
        if candidate.group_name:
            return candidate.group_name

    for span in spans:
        if not _span_matches_node(span, {"LangGraph", "knowledge_search"}):
            continue
        span_outputs = _parse_attr_json(span.get("attributes") or {}, "mlflow.spanOutputs")
        for candidate in _extract_candidate_documents_from_payload(span_outputs):
            if candidate.group_name:
                return candidate.group_name
    return ""


def extract_reranker_info(
    spans: list[dict[str, Any]],
    children: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    empty = {
        "reranker_system_prompt": "",
        "reranker_user_prompt": "",
        "reranker_model": "",
        "reranker_token_usage": "",
        "reranker_raw_selected_ids": "[]",
        "_reranker_span_present": False,
        "_reranker_selected_ids_found": False,
    }

    rerank = next((span for span in spans if _span_matches_node(span, {"rerank"})), None)
    if rerank is None:
        return empty

    chat_databricks = _find_descendant_by_name(rerank, children, "ChatDatabricks")
    if chat_databricks is None:
        return {**empty, "_reranker_span_present": True}

    attrs = chat_databricks.get("attributes") or {}

    system_prompt = ""
    user_prompt = ""
    span_inputs = _parse_attr_json(attrs, "mlflow.spanInputs")
    messages = None
    if isinstance(span_inputs, list) and span_inputs and isinstance(span_inputs[0], list):
        messages = span_inputs[0]
    elif isinstance(span_inputs, list):
        messages = span_inputs
    if messages:
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role") or message.get("type")
            content = message.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            if role == "system":
                system_prompt = content
            elif role in ("user", "human"):
                user_prompt = content

    model_raw = attrs.get("mlflow.llm.model", "")
    try:
        model = json.loads(model_raw) if isinstance(model_raw, str) and model_raw.strip().startswith('"') else model_raw
    except json.JSONDecodeError:
        model = model_raw
    if not isinstance(model, str):
        model = str(model)

    token_usage_raw = attrs.get("mlflow.chat.tokenUsage", "")
    if not isinstance(token_usage_raw, str):
        token_usage_raw = json.dumps(token_usage_raw, ensure_ascii=False)

    selected_ids: list[str] = []
    selected_ids_found = False
    span_outputs = _parse_attr_json(attrs, "mlflow.spanOutputs")
    if span_outputs is not None:
        for node in _walk_dicts(span_outputs):
            tool_calls = node.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function") or {}
                if function.get("name") != "_RerankResponse":
                    continue
                args_raw = function.get("arguments", "")
                if not args_raw:
                    continue
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except json.JSONDecodeError:
                    continue
                if isinstance(args, dict) and isinstance(args.get("selected_ids"), list):
                    selected_ids = [v for v in args["selected_ids"] if isinstance(v, str)]
                    selected_ids_found = True
                    break
            if selected_ids_found:
                break

    return {
        "reranker_system_prompt": system_prompt,
        "reranker_user_prompt": user_prompt,
        "reranker_model": model,
        "reranker_token_usage": token_usage_raw,
        "reranker_raw_selected_ids": json.dumps(selected_ids, ensure_ascii=False),
        "_reranker_span_present": True,
        "_reranker_selected_ids_found": selected_ids_found,
    }


# ── KB pipeline: per-run construction ──────────────────────────────────

def _has_knowledge_search_evidence(spans: list[dict[str, Any]]) -> bool:
    if any(span.get("name") == "knowledge_search" for span in spans):
        return True
    if any(_is_vector_db_http_span(span) for span in spans):
        return True
    if any(_span_matches_node(span, {"retrieve", "prune", "rerank"}) for span in spans):
        return True
    return bool(extract_kb_context(spans))


def _knowledge_search_scope(
    root: dict[str, Any],
    spans: list[dict[str, Any]],
    children: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    descendant_scope = _collect_span_descendants(root, children)
    time_scope = _spans_in_time_window(root, spans)
    scoped = _merge_spans_by_id([*descendant_scope, *time_scope])
    return scoped if scoped else [root]


def _build_knowledge_search_run(
    run_index: int,
    root: dict[str, Any] | None,
    run_spans: list[dict[str, Any]],
) -> dict[str, Any]:
    run_children = _build_span_children(run_spans)
    reranker = extract_reranker_info(run_spans, run_children)
    reranker_span_present = bool(reranker.pop("_reranker_span_present", False))
    reranker_selected_ids_found = bool(reranker.pop("_reranker_selected_ids_found", False))

    raw_vector_db = extract_raw_vector_db_hits(run_spans)
    pre_prune_text, pre_prune_ids = extract_pre_prune_candidates(run_spans)
    post_prune_text, post_prune_ids = extract_post_prune_candidates(run_spans)
    candidate_text, candidate_ids = extract_retrieved_candidates(
        run_spans,
        reranker.get("reranker_user_prompt", ""),
    )
    if not post_prune_ids and candidate_ids:
        post_prune_text = candidate_text
        post_prune_ids = candidate_ids
    kb_context = extract_kb_context(run_spans)
    kb_context_enum_ids = extract_reranked_enum_ids(kb_context)

    try:
        reranker_raw_selected_ids = json.loads(reranker.get("reranker_raw_selected_ids") or "[]")
        if not isinstance(reranker_raw_selected_ids, list):
            reranker_raw_selected_ids = []
    except json.JSONDecodeError:
        reranker_raw_selected_ids = []

    if reranker_selected_ids_found:
        if kb_context_enum_ids:
            selected_set = set(reranker_raw_selected_ids)
            reranked_enum_ids = [enum_id for enum_id in kb_context_enum_ids if enum_id in selected_set]
        else:
            reranked_enum_ids = reranker_raw_selected_ids
    elif not reranker_span_present:
        reranked_enum_ids = kb_context_enum_ids
    else:
        reranked_enum_ids = []

    prune_counts = extract_prune_counts(run_spans)
    run = {
        "run_index": run_index,
        "knowledge_search_span_id": _to_str(root.get("span_id")) if root else "",
        "knowledge_search_start_time": root.get("start_time_unix_nano") if root else None,
        "knowledge_search_end_time": root.get("end_time_unix_nano") if root else None,
        "reranked_kb_context": kb_context,
        "reranked_enum_ids": reranked_enum_ids,
        **raw_vector_db,
        "pre_prune_candidates_text": pre_prune_text,
        "pre_prune_enum_ids": pre_prune_ids,
        "pre_prune_enum_count": len(pre_prune_ids),
        "post_prune_candidates_text": post_prune_text,
        "post_prune_enum_ids": post_prune_ids,
        "post_prune_enum_count": len(post_prune_ids),
        "reranker_span_present": reranker_span_present,
        "reranker_selected_ids_found": reranker_selected_ids_found,
        "reranker_raw_selected_ids": json.dumps(reranker_raw_selected_ids, ensure_ascii=False),
        **prune_counts,
    }
    run.update({key: value for key, value in reranker.items() if key != "reranker_raw_selected_ids"})
    return run


def extract_knowledge_search_runs(
    spans: list[dict[str, Any]],
    children: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    roots = [span for span in spans if span.get("name") == "knowledge_search"]
    runs: list[dict[str, Any]] = []
    for run_index, root in enumerate(roots, start=1):
        run_spans = _knowledge_search_scope(root, spans, children)
        if len(roots) == 1 and not any(
            _is_vector_db_http_span(span)
            or _span_matches_node(span, {"retrieve", "prune", "rerank"})
            for span in run_spans
        ):
            run_spans = spans
        runs.append(_build_knowledge_search_run(run_index, root, run_spans))

    if not runs and _has_knowledge_search_evidence(spans):
        runs.append(_build_knowledge_search_run(1, None, spans))
    return runs


def _select_final_knowledge_search_run(runs: list[dict[str, Any]]) -> dict[str, Any]:
    for run in reversed(runs):
        if run.get("reranked_kb_context") or run.get("post_prune_enum_ids"):
            return run
    return runs[-1] if runs else {}


# ── Eval flavor: tool calls + tool registry + model ────────────────────

def _extract_actual_tool_calls(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract every TOOL span as a scorer-ready dict (eval flavor).

    Each emitted dict carries the args under both ``arguments`` and
    ``parameters`` (same value) — the ``ToolParameterScorer`` reads
    ``arguments`` on the actual side, callers that mirror the expected
    side's ``parameters`` key still work.

    Skips ``transfer-to-<slug>`` TOOL spans (post-#388 supervisor's
    delegate marker) — those are surfaced via ``actual_agents_path``
    instead. From the eval-scoring contract's perspective a sub-agent
    invocation is not a tool call.
    """
    calls: list[dict[str, Any]] = []
    step = 0
    for span in spans:
        if _span_type(span) != _TOOL:
            continue
        if _is_subagent_transfer_tool(span):
            continue
        step += 1
        attrs = span.get("attributes") or {}
        params = _parse_attr_json(attrs, "mlflow.spanInputs")
        outputs_obj = _parse_attr_json(attrs, "mlflow.spanOutputs") or {}
        if not isinstance(outputs_obj, dict):
            outputs_obj = {}
        outputs_text = _to_str(outputs_obj.get("content"))
        outputs_preview = outputs_text[:500] + ("…" if len(outputs_text) > 500 else "")

        span_status = span.get("status") or {}
        span_code = _to_str(span_status.get("code"))
        span_message = _to_str(span_status.get("message"))
        tm_status = _to_str(outputs_obj.get("status"))
        is_error = (
            (span_code and "OK" not in span_code.upper() and "UNSET" not in span_code.upper())
            or (tm_status and tm_status.lower() != "success")
        )
        error_message = span_message or (outputs_text if is_error else "")

        raw_tcid = attrs.get("tool_call_id")
        if isinstance(raw_tcid, str):
            try:
                tool_call_id = _to_str(json.loads(raw_tcid))
            except json.JSONDecodeError:
                tool_call_id = raw_tcid
        else:
            tool_call_id = _to_str(raw_tcid)

        metadata = _parse_attr_json(attrs, "metadata") or {}
        owning_node = _to_str(
            metadata.get("langgraph_node") if isinstance(metadata, dict) else ""
        )

        args = params if isinstance(params, dict) else {}
        calls.append({
            "tool": _to_str(span.get("name")),
            "step": step,
            "arguments": args,
            "parameters": args,
            "tool_call_id": tool_call_id,
            "outputs_preview": outputs_preview,
            "outputs_text": outputs_text,
            "outputs_obj": outputs_obj,
            "error": bool(is_error),
            "error_message": error_message,
            "agent": owning_node,
            "span_status_code": span_code,
            "tool_message_status": tm_status,
        })
    return calls


def _extract_tool_registry(
    spans: list[dict[str, Any]],
) -> tuple[list[str], dict[str, str], dict[str, dict[str, str]]]:
    """Collect tools the LLM was given (from CHAT_MODEL ``mlflow.chat.tools``).

    Union by name across all CHAT_MODEL spans; description is from the
    first occurrence (stable across the run).
    """
    descriptions: dict[str, str] = {}
    by_node: dict[str, dict[str, str]] = {}
    for span in spans:
        if _span_type(span) != _CHAT_MODEL:
            continue
        attrs = span.get("attributes") or {}
        tools = _parse_attr_json(attrs, "mlflow.chat.tools") or []
        metadata = _parse_attr_json(attrs, "metadata") or {}
        node = _to_str(metadata.get("langgraph_node") if isinstance(metadata, dict) else "")
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function") or {}
            name = _to_str(fn.get("name"))
            if not name:
                continue
            desc = _to_str(fn.get("description"))
            descriptions.setdefault(name, desc)
            if node:
                by_node.setdefault(node, {}).setdefault(name, desc)
    return sorted(descriptions.keys()), descriptions, by_node


def _extract_model(spans: list[dict[str, Any]]) -> str:
    """Best-effort model name from the first CHAT_MODEL span."""
    for span in spans:
        if _span_type(span) != _CHAT_MODEL:
            continue
        attrs = span.get("attributes") or {}
        model = attrs.get("mlflow.llm.model")
        if isinstance(model, str) and model.strip():
            try:
                return _to_str(json.loads(model))
            except json.JSONDecodeError:
                return _to_str(model)
        ip = _parse_attr_json(attrs, "invocation_params") or {}
        if isinstance(ip, dict):
            m = _to_str(ip.get("model"))
            if m:
                return m
    return ""


# ── Prompt hashes (SKKB flavor) ────────────────────────────────────────
#
# TODO(peter-pr): The LangGraph-orchestrator branch in
# `_get_agent_chat_system_prompt` is a compensation patch for the same
# orchestrator regression that broke `agent_answer` emission (see
# project memory `project_orchestrator_agent_answer_broken.md`, PR #388).
# Once Peter's upstream PR lands and restores the prior span layout —
# named `main_agent` / `daily_banking_agent` spans, ChatDatabricks
# children of those, and `mlflow.spanInputs` shape
# `[[{role, content}, ...]]` — the LangGraph fallback path (and the
# dict-shape branch of `_chat_databricks_system_message`) becomes dead
# code and should be removed in favour of the original legacy-only
# implementation. Re-run the SKKB import smoke test on a post-fix trace
# to confirm the legacy path alone produces non-empty hashes before
# stripping the workaround.

# Spans whose ChatDatabricks descendant is a reranker / parser call, not
# an agent system prompt. Excluded from the LangGraph-format ancestry
# scan in `_get_agent_chat_system_prompt`.
_NON_AGENT_CHAT_ANCESTORS = frozenset({
    "knowledge_rerank", "rerank", "RunnableSequence",
})


def _chat_databricks_system_message(span: Mapping[str, Any]) -> str:
    """Return the first non-empty system-message content from a
    ChatDatabricks span's ``mlflow.spanInputs``, or ``""``.

    Handles two trace formats seen in this codebase:

    * Legacy: ``[[{role, content}, ...]]`` — pre-LangGraph orchestrator
      emitted spanInputs as a single-element list wrapping the message
      list. The original implementation only accepted this shape.
    * LangGraph orchestrator (current prod): ``{"messages": [...]}`` —
      same message dicts, wrapped under a ``messages`` key.

    Both shapes carry the same role/content message dicts, so once we
    locate the message list the inner loop is identical.
    """
    span_inputs = _parse_attr_json(span.get("attributes") or {}, "mlflow.spanInputs")
    messages: list[Any] | None = None
    if isinstance(span_inputs, dict):
        m = span_inputs.get("messages")
        if isinstance(m, list):
            messages = m
    elif isinstance(span_inputs, list) and span_inputs:
        if isinstance(span_inputs[0], list):
            messages = span_inputs[0]
        elif isinstance(span_inputs[0], dict):
            # Defensive: a bare list of message dicts. Not observed in
            # the wild but cheap to support and avoids a silent zero.
            messages = span_inputs
    if not messages:
        return ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "system" or message.get("type") == "system":
            content = message.get("content", "")
            if isinstance(content, str) and content.strip():
                return content
    return ""


def _ancestor_names(
    span: Mapping[str, Any],
    by_id: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Walk ``parent_span_id`` links from ``span`` up to the root, returning
    the chain of ancestor names (immediate parent first). Stops on cycle
    or missing parent — both are defensive; neither should occur in
    well-formed MLflow traces."""
    chain: list[str] = []
    seen: set[str] = set()
    cur = span
    while True:
        psid = _to_str(cur.get("parent_span_id"))
        if not psid or psid in seen:
            break
        seen.add(psid)
        parent = by_id.get(psid)
        if parent is None:
            break
        chain.append(_to_str(parent.get("name")))
        cur = parent
    return chain


def _get_agent_chat_system_prompt(
    spans: list[dict[str, Any]],
    children: dict[str, list[dict[str, Any]]],
    agent_name: str,
) -> str:
    """Return the system prompt for ``agent_name`` from a trace's spans.

    Supports two orchestrator shapes — auto-detected, no flag needed:

    1. **Legacy** — a span literally named ``agent_name`` exists. Walk
       its descendants (via the pre-built ``children`` map, keyed on
       ``parent_span_id``) and return the first ChatDatabricks's system
       message. This is what the test fixtures and pre-LangGraph
       production traces emit.

    2. **LangGraph orchestrator** (current prod) — agents are LangGraph
       subgraphs, not named spans, so the lookup above returns ``None``.
       In that case identify the agent by walking each ChatDatabricks's
       ancestor chain:

       * ``main_agent``         → ChatDatabricks whose ancestors contain
         NO ``transfer-to-*`` span (the supervisor's own LLM calls,
         including the post-handoff "final answer" call).
       * sub-agent (e.g.        → ChatDatabricks whose ancestors
         ``daily_banking_agent``) contain ``transfer-to-{agent_name}``.

       Reranker / parser ChatDatabricks calls (under ``knowledge_rerank``
       / ``rerank`` / ``RunnableSequence``) are excluded from both
       classifications — they aren't an agent prompt.

    The first matching system prompt wins. In observed traces the
    supervisor's multiple ChatDatabricks calls carry identical system
    prompts, so "first match" matches "all matches".
    """
    # ── 1. Legacy format ────────────────────────────────────────────────
    agent = next((span for span in spans if span.get("name") == agent_name), None)
    if agent is not None:
        stack = [agent]
        while stack:
            current = stack.pop()
            if current.get("name") == "ChatDatabricks":
                content = _chat_databricks_system_message(current)
                if content:
                    return content
            stack.extend(children.get(_to_str(current.get("span_id")), []))
        return ""

    # ── 2. LangGraph orchestrator format ────────────────────────────────
    by_id: dict[str, dict[str, Any]] = {
        _to_str(s.get("span_id")): s for s in spans if s.get("span_id")
    }
    transfer_marker = f"transfer-to-{agent_name}"
    want_supervisor = (agent_name == "main_agent")

    for span in spans:
        if span.get("name") != "ChatDatabricks":
            continue
        ancestors = _ancestor_names(span, by_id)
        if any(a in _NON_AGENT_CHAT_ANCESTORS for a in ancestors):
            continue
        has_any_transfer = any(a.startswith("transfer-to-") for a in ancestors)
        if want_supervisor:
            # Supervisor calls: top-level under LangGraph ← eval_item, with
            # no `transfer-to-*` between the ChatDatabricks and the root.
            if has_any_transfer:
                continue
        else:
            # Sub-agent: ancestry must include the specific transfer marker.
            if transfer_marker not in ancestors:
                continue
        content = _chat_databricks_system_message(span)
        if content:
            return content
    return ""


def extract_agent_system_prompts(
    spans: list[dict[str, Any]],
    children: dict[str, list[dict[str, Any]]],
) -> dict[str, str]:
    """Return the timestamp-stripped system prompts for the named agents.

    Empty string when the corresponding agent span or ChatDatabricks system
    message is missing. The stripping mirrors what the hash sees, so the
    hash of the returned text equals the column hash.
    """
    main_raw = _get_agent_chat_system_prompt(spans, children, "main_agent")
    dba_raw = _get_agent_chat_system_prompt(spans, children, "daily_banking_agent")
    return {
        "main_agent_system_prompt": _TS_TAIL_RE.sub("", main_raw).rstrip(),
        "daily_banking_agent_system_prompt": _TS_TAIL_RE.sub("", dba_raw).rstrip(),
    }


def extract_agent_prompt_hashes(
    spans: list[dict[str, Any]],
    children: dict[str, list[dict[str, Any]]],
) -> dict[str, str]:
    prompts = extract_agent_system_prompts(spans, children)
    return {
        "main_agent_prompt_hash": _md5_short(prompts["main_agent_system_prompt"]),
        "daily_banking_agent_prompt_hash": _md5_short(prompts["daily_banking_agent_system_prompt"]),
    }


def hash_tool_descriptions(tool_descriptions: dict[str, str]) -> str:
    """Stable hash of a tool-name → description mapping.

    Sorted-key JSON serialization so hash equality means semantic equality
    regardless of insertion order. Empty registry hashes the same as the
    empty string, so an absent / missing-span case stays distinguishable
    from a non-empty registry that happens to be uniform.
    """
    if not tool_descriptions:
        return _md5_short("")
    canonical = json.dumps(tool_descriptions, sort_keys=True, ensure_ascii=False)
    return _md5_short(canonical)


# ── KB pipeline problem-cause + dataframe-level helpers ────────────────

KB_PIPELINE_PROBLEM_CAUSES = (
    "none",
    "not_applicable",
    "retrieval_no_hits",
    "prune_dropped_all",
    "reranker_empty_selection",
)
_KB_SCOPES = frozenset({"kb", "hg_invest_kb"})


def classify_kb_pipeline_problem_cause(
    query_scope: str,
    raw_vector_db_retrieved_enum_count: int,
    pre_prune_enum_count: int,
    post_prune_enum_count: int,
    reranker_selected_empty: bool,
) -> str:
    """Tag the earliest funnel stage that failed for a single trace.

    Mutually-exclusive categories; earliest failing stage wins.
    """
    if query_scope not in _KB_SCOPES:
        return "not_applicable"
    if raw_vector_db_retrieved_enum_count == 0:
        return "retrieval_no_hits"
    if pre_prune_enum_count > 0 and post_prune_enum_count == 0:
        return "prune_dropped_all"
    if post_prune_enum_count > 0 and reranker_selected_empty:
        return "reranker_empty_selection"
    return "none"


def apply_kb_pipeline_problem_cause(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``kb_pipeline_problem_cause`` to a trace dataframe in place."""
    required = {
        "query_scope",
        "raw_vector_db_retrieved_enum_count",
        "pre_prune_enum_count",
        "post_prune_enum_count",
        "reranker_selected_empty",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"dataframe missing required columns: {sorted(missing)}")

    def _to_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _to_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return False
        return str(value).strip().lower() in {"true", "1", "1.0", "yes"}

    df["kb_pipeline_problem_cause"] = [
        classify_kb_pipeline_problem_cause(
            query_scope=_to_str(row.get("query_scope")),
            raw_vector_db_retrieved_enum_count=_to_int(row.get("raw_vector_db_retrieved_enum_count")),
            pre_prune_enum_count=_to_int(row.get("pre_prune_enum_count")),
            post_prune_enum_count=_to_int(row.get("post_prune_enum_count")),
            reranker_selected_empty=_to_bool(row.get("reranker_selected_empty")),
        )
        for _, row in df.iterrows()
    ]
    return df


def reapply_query_scope(df: pd.DataFrame) -> pd.DataFrame:
    """Refresh ``query_scope`` / ``trace_invariant_violations`` / ``agents_only`` /
    ``last_agent`` in place from existing ``agents_called`` and
    ``tools_called`` columns. Other columns are left untouched.
    """

    def _coerce(value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return []
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                try:
                    import ast
                    parsed = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    return []
            return parsed if isinstance(parsed, list) else []
        return []

    if "agents_called" not in df.columns or "tools_called" not in df.columns:
        raise KeyError("dataframe must contain 'agents_called' and 'tools_called' columns")

    agents_input = df["agents_called"].tolist()
    tools_input = df["tools_called"].tolist()
    serialize_as_string = any(isinstance(value, str) for value in agents_input)

    new_scopes: list[str] = []
    new_violations: list[Any] = []
    new_agents_only: list[Any] = []
    new_last_agent: list[str] = []
    for agents_raw, tools_raw in zip(agents_input, tools_input):
        agents = _coerce(agents_raw)
        tools = _coerce(tools_raw)
        scope = classify_query_scope(agents, tools)
        violations = check_trace_invariants(agents, tools, scope)
        agents_only = [node for node in agents if node in KNOWN_AGENT_NAMES]
        new_scopes.append(scope)
        new_violations.append(
            json.dumps(violations, ensure_ascii=False) if serialize_as_string else violations
        )
        new_agents_only.append(
            json.dumps(agents_only, ensure_ascii=False) if serialize_as_string else agents_only
        )
        new_last_agent.append(agents_only[-1] if agents_only else "")

    df["query_scope"] = new_scopes
    df["trace_invariant_violations"] = new_violations
    df["agents_only"] = new_agents_only
    df["last_agent"] = new_last_agent
    return df


# ── Per-trace builders ─────────────────────────────────────────────────

def parse_trace_skkb(test_case_id: str, trace: dict[str, Any]) -> dict[str, Any]:
    """Build one SKKB/CZKB record from a normalized trace.

    ``agent_response`` comes strictly from the ``agent_answer`` span —
    no fallbacks (see :func:`extract_agent_answer_span`).
    """
    info = trace.get("info") or {}
    data = trace.get("data") or {}
    spans = _sort_spans_by_dependency_order(data.get("spans") or [])
    children = _build_span_children(spans)
    span_error_summary = summarize_span_errors(extract_span_errors(spans))

    trace_metadata = info.get("trace_metadata") or {}
    user_query = _extract_user_query(trace_metadata, spans)
    agent_response = extract_agent_answer_span(spans)

    knowledge_search_runs = extract_knowledge_search_runs(spans, children)
    final_kb_run = _select_final_knowledge_search_run(knowledge_search_runs)
    reranker = {
        key: final_kb_run.get(key, "")
        for key in (
            "reranker_system_prompt",
            "reranker_user_prompt",
            "reranker_model",
            "reranker_token_usage",
            "reranker_raw_selected_ids",
        )
    }

    kb_context = _to_str(final_kb_run.get("reranked_kb_context"))
    kb_version = extract_kb_version(spans, reranker.get("reranker_user_prompt", ""))
    raw_vector_db = {
        key: final_kb_run.get(key, value)
        for key, value in extract_raw_vector_db_hits([]).items()
    }
    pre_prune_text = _to_str(final_kb_run.get("pre_prune_candidates_text"))
    pre_prune_ids = list(final_kb_run.get("pre_prune_enum_ids") or [])
    post_prune_text = _to_str(final_kb_run.get("post_prune_candidates_text"))
    post_prune_ids = list(final_kb_run.get("post_prune_enum_ids") or [])
    reranker_span_present = bool(final_kb_run.get("reranker_span_present", False))
    reranker_selected_ids_found = bool(final_kb_run.get("reranker_selected_ids_found", False))
    kb_context_enum_ids = extract_reranked_enum_ids(kb_context)

    try:
        reranker_raw_selected_ids = json.loads(
            _to_str(final_kb_run.get("reranker_raw_selected_ids")) or "[]"
        )
        if not isinstance(reranker_raw_selected_ids, list):
            reranker_raw_selected_ids = []
    except json.JSONDecodeError:
        reranker_raw_selected_ids = []

    if reranker_selected_ids_found:
        if kb_context_enum_ids:
            selected_set = set(reranker_raw_selected_ids)
            reranked_enum_ids = [eid for eid in kb_context_enum_ids if eid in selected_set]
        else:
            reranked_enum_ids = reranker_raw_selected_ids
    elif not reranker_span_present:
        reranked_enum_ids = kb_context_enum_ids
    else:
        reranked_enum_ids = []

    agent_hashes = extract_agent_prompt_hashes(spans, children)
    available_tools, tool_descriptions, _ = _extract_tool_registry(spans)
    tool_descriptions_hash = hash_tool_descriptions(tool_descriptions)
    trace_metadata_map = _coerce_mapping(info.get("trace_metadata"))
    source_run_id = _to_str(trace_metadata_map.get("mlflow.sourceRun"))
    calls = extract_agent_and_tool_calls(spans)
    query_scope = classify_query_scope(calls["agents_called"], calls["tools_called"])
    trace_invariant_violations = check_trace_invariants(
        calls["agents_called"], calls["tools_called"], query_scope,
    )
    prune_counts = {
        key: final_kb_run.get(key, value)
        for key, value in extract_prune_counts([]).items()
    }

    if query_scope == "kb":
        post_prune_id_set = set(post_prune_ids)
        selected_id_set = set(reranker_raw_selected_ids)
        reranker_valid_selected_ids = [
            eid for eid in reranker_raw_selected_ids if eid in post_prune_id_set
        ]
        reranker_invalid_selected_ids = [
            eid for eid in reranker_raw_selected_ids if eid not in post_prune_id_set
        ]
        reranker_unselected_context_ids = [
            eid for eid in kb_context_enum_ids if eid not in selected_id_set
        ]
        reranker_selection_violations: list[str] = []
        if not reranker_span_present:
            reranker_selection_violations.append("reranker_span_missing")
        elif not reranker_selected_ids_found:
            reranker_selection_violations.append("reranker_selected_ids_missing")
        if reranker_invalid_selected_ids:
            reranker_selection_violations.append("reranker_selected_unknown_ids")
        if reranker_selected_ids_found and reranker_raw_selected_ids == [] and kb_context_enum_ids:
            reranker_selection_violations.append("reranker_empty_but_context_nonempty")
        if reranker_selected_ids_found and reranker_raw_selected_ids and reranker_unselected_context_ids:
            reranker_selection_violations.append("reranker_context_contains_unselected_ids")
        if reranker_span_present and not reranker_selected_ids_found and kb_context_enum_ids:
            reranker_selection_violations.append("reranker_context_without_selected_ids")

        if not reranker_span_present:
            reranker_selection_status = "missing_rerank"
        elif not reranker_selected_ids_found:
            reranker_selection_status = "missing_selected_ids"
        elif reranker_raw_selected_ids == [] and kb_context_enum_ids:
            reranker_selection_status = "fallback_context_after_empty_selection"
        elif reranker_raw_selected_ids == []:
            reranker_selection_status = "empty_selection"
        elif reranker_invalid_selected_ids:
            reranker_selection_status = "invalid_ids"
        elif reranker_unselected_context_ids:
            reranker_selection_status = "fallback_context"
        else:
            reranker_selection_status = "ok"
    else:
        reranker_valid_selected_ids = []
        reranker_invalid_selected_ids = []
        reranker_unselected_context_ids = []
        reranker_selection_violations = []
        reranker_selection_status = "not_applicable"

    reranker_selected_empty = (
        query_scope == "kb"
        and reranker_selected_ids_found
        and reranker_raw_selected_ids == []
    )

    assessments = {
        _assessment_name(a): a
        for a in (info.get("assessments") or [])
        if isinstance(a, Mapping) and _assessment_name(a)
    }

    expected_response = (
        (assessments.get("expected_response") or {})
        .get("expectation", {})
        .get("value", "")
    )

    grading = assessments.get("answer_hg_chat_admin_gpt_grading") or {}
    expert_score = (grading.get("feedback") or {}).get("value")
    # NaN keeps the column float-typed in pandas — an all-None column would
    # land as object dtype, get inferred as NullType by Spark.createDataFrame,
    # and be dropped under overwriteSchema=true, breaking column-level policies.
    if expert_score is None:
        expert_score = float("nan")
    expert_rationale = grading.get("rationale", "") or ""

    ter_value_str, expected_enums_weights = _extract_expected_enums_weights(
        assessments.get("target_enums_to_relevance") or {}
    )

    er2 = assessments.get("enum_relevance_2") or {}
    enum_relevance_score = (er2.get("feedback") or {}).get("value")
    if enum_relevance_score is None:
        enum_relevance_score = float("nan")

    record = {
        "test_case_id": test_case_id,
        "trace_id": info.get("trace_id"),
        "request_time": info.get("request_time"),
        "execution_duration_ms": info.get("execution_duration_ms"),
        "user_query": user_query,
        "knowledge_search_run_count": len(knowledge_search_runs),
        "knowledge_search_final_run_index": final_kb_run.get("run_index", 0),
        "knowledge_search_runs": knowledge_search_runs,
        "reranked_kb_context": kb_context,
        "kb_version": kb_version,
        "reranked_enum_ids": reranked_enum_ids,
        **raw_vector_db,
        "pre_prune_candidates_text": pre_prune_text,
        "pre_prune_enum_ids": pre_prune_ids,
        "pre_prune_enum_count": len(pre_prune_ids),
        "post_prune_candidates_text": post_prune_text,
        "post_prune_enum_ids": post_prune_ids,
        "post_prune_enum_count": len(post_prune_ids),
        **prune_counts,
        "agent_response": agent_response,
        "expected_response": expected_response,
        "expected_enums": sorted(expected_enums_weights.keys()),
        "expected_enums_weights": ter_value_str,
        "expert_score": expert_score,
        "expert_rationale": expert_rationale,
        "enum_relevance_score": enum_relevance_score,
        "agents_called": calls["agents_called"],
        "agents_only": calls["agents_only"],
        "last_agent": calls["last_agent"],
        "tools_called": calls["tools_called"],
        "query_scope": query_scope,
        "trace_invariant_violations": trace_invariant_violations,
        "reranker_selected_empty": reranker_selected_empty,
        "reranker_raw_selected_ids": json.dumps(reranker_raw_selected_ids, ensure_ascii=False),
        "reranker_valid_selected_ids": json.dumps(reranker_valid_selected_ids, ensure_ascii=False),
        "reranker_invalid_selected_ids": json.dumps(reranker_invalid_selected_ids, ensure_ascii=False),
        "reranker_unselected_context_ids": json.dumps(reranker_unselected_context_ids, ensure_ascii=False),
        "reranker_selection_status": reranker_selection_status,
        "reranker_selection_violations": json.dumps(reranker_selection_violations, ensure_ascii=False),
        **span_error_summary,
    }
    record.update({key: value for key, value in reranker.items() if key != "reranker_raw_selected_ids"})
    record.update(agent_hashes)
    record["available_tools"] = available_tools
    record["tool_descriptions"] = tool_descriptions
    record["tool_descriptions_hash"] = tool_descriptions_hash
    record["source_run_id"] = source_run_id
    return record


def parse_trace_mlflow(
    trace: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    """Build one eval-scoring record from a normalized trace.

    ``actual_response`` comes strictly from the ``agent_answer`` span —
    no fallbacks (see :func:`extract_agent_answer_span`).
    """
    info = trace.get("info") or {}
    data = trace.get("data") or {}
    spans = data.get("spans") or []
    span_error_summary = summarize_span_errors(extract_span_errors(spans))

    trace_id = _to_str(info.get("trace_id"))
    request_time = _to_str(info.get("request_time"))
    duration_ms = info.get("execution_duration_ms")
    state = _to_str(info.get("state"))
    trace_metadata = _coerce_mapping(info.get("trace_metadata"))
    tags = _coerce_mapping(info.get("tags"))

    user_query = _extract_user_query(trace_metadata, spans)
    assessments_raw = info.get("assessments") or []
    assessments = _extract_assessments(assessments_raw)

    test_case_id = _resolve_test_case_id_mlflow(
        assessments=assessments,
        user_query=user_query,
        trace_id=trace_id,
    )

    expected_agent = _to_str(assessments.get(AssessmentNames.EXPECTED_AGENT, ""))
    expected_tool_calls_raw = (
        assessments.get(AssessmentNames.EXPECTED_TOOL_CALLS)
        if AssessmentNames.EXPECTED_TOOL_CALLS in assessments
        else assessments.get(_LEGACY_ASSESSMENT_NAMES[AssessmentNames.EXPECTED_TOOL_CALLS])
    )
    expected_tool_calls = _normalize_expected_tool_calls(expected_tool_calls_raw)
    expected_response = _to_str(assessments.get(AssessmentNames.EXPECTED_RESPONSE, ""))
    eval_domain = _to_str(assessments.get(AssessmentNames.EVAL_DOMAIN, ""))
    eval_persona = _to_str(assessments.get(AssessmentNames.EVAL_PERSONA, ""))
    guidelines = assessments.get(AssessmentNames.GUIDELINES) or []
    scorers_to_run = _normalize_scorers_to_run(assessments.get(AssessmentNames.SCORERS))

    # Ground-truth KB-retrieval ENUMs (KB / KB&API cases only). Same
    # extraction helper used by parse_trace_skkb so the shape matches.
    # Returns ("{}", {}) when the assessment is absent — non-KB cases.
    ter_value_str, expected_enums_weights = _extract_expected_enums_weights(
        assessments.get(AssessmentNames.TARGET_ENUMS_TO_RELEVANCE) or {}
    )
    expected_enums = sorted(expected_enums_weights.keys())

    actual_agents_path, actual_agent = _extract_actual_agent_path(spans)
    actual_tool_calls = _extract_actual_tool_calls(spans)
    actual_response = extract_agent_answer_span(spans)

    available_tools, tool_descriptions, registry_by_node = _extract_tool_registry(spans)
    tool_descriptions_hash = hash_tool_descriptions(tool_descriptions)
    children = _build_span_children(spans)
    agent_hashes = extract_agent_prompt_hashes(spans, children)

    token_usage = _decode_json(trace_metadata.get("mlflow.trace.tokenUsage")) or {}
    git_branch = _to_str(trace_metadata.get("mlflow.source.git.branch"))
    git_commit = _to_str(trace_metadata.get("mlflow.source.git.commit"))
    source_run_id = _to_str(trace_metadata.get("mlflow.sourceRun"))
    mlflow_user = _to_str(trace_metadata.get("mlflow.user"))
    session_id = _to_str(trace_metadata.get("mlflow.trace.session"))
    architecture = _classify_architecture(actual_agents_path)

    record = {
        "trace_id": trace_id,
        "test_case_id": test_case_id,
        "session_id": session_id,
        "request_time": request_time,
        "execution_duration_ms": duration_ms,
        "state": state,
        "user_query": user_query,
        "eval_domain": eval_domain,
        "eval_persona": eval_persona,
        "expected_agent": expected_agent,
        "expected_tool_calls": expected_tool_calls,
        "expected_response": expected_response,
        "guidelines": guidelines,
        "scorers_to_run": scorers_to_run,
        "expected_enums": expected_enums,
        "expected_enums_weights": ter_value_str,
        "actual_agent": actual_agent,
        "actual_agents_path": actual_agents_path,
        "actual_tool_calls": actual_tool_calls,
        "actual_response": actual_response,
        "available_tools": available_tools,
        "tool_descriptions": tool_descriptions,
        "tool_descriptions_hash": tool_descriptions_hash,
        "tool_registry_by_node": {
            node: list(tools.keys()) for node, tools in registry_by_node.items()
        },
        **agent_hashes,
        "architecture": architecture,
        "model": _extract_model(spans),
        "token_usage": token_usage,
        "git_branch": git_branch,
        "git_commit": git_commit,
        "source_run_id": source_run_id,
        "mlflow_user": mlflow_user,
        "tags": tags,
        "assessments_raw": [_to_plain_data(a) for a in assessments_raw],
        **span_error_summary,
    }
    return record, registry_by_node


# Back-compat alias used by skkb/czkb notebooks and the existing test.
parse_trace = parse_trace_skkb


# ── Public entry points ────────────────────────────────────────────────

def build_skkb_dataframe_from_mlflow_search_traces(
    traces_df: pd.DataFrame,
) -> SKKBParseResult:
    """Build the SKKB/CZKB flat table from ``mlflow.search_traces`` pandas output."""
    records: list[dict[str, Any]] = []
    parse_errors: list[tuple[str, str]] = []
    unmapped_trace_ids: list[str] = []

    for row in traces_df.to_dict(orient="records"):
        trace_id = _to_str(row.get("trace_id"))
        try:
            canonical = normalize_mlflow_trace_row(row)
            test_case_id = resolve_test_case_id(canonical)
            if test_case_id == trace_id:
                unmapped_trace_ids.append(trace_id)
            records.append(parse_trace_skkb(test_case_id, canonical))
        except Exception as exc:
            parse_errors.append((trace_id, repr(exc)))

    return SKKBParseResult(
        dataframe=pd.DataFrame.from_records(records),
        parse_errors=parse_errors,
        unmapped_trace_ids=unmapped_trace_ids,
    )


def build_dataframe_from_mlflow_traces(traces_df: pd.DataFrame) -> MlflowParseResult:
    """Build the canonical eval-scoring DataFrame.

    Accepts either the JSONL ``info``/``data`` shape or the live
    ``mlflow.search_traces`` flat shape.
    """
    records: list[dict[str, Any]] = []
    parse_errors: list[ParseError] = []
    untagged: list[str] = []
    run_registry: dict[str, dict[str, str]] = {}

    for raw_row in traces_df.to_dict(orient="records"):
        trace_id_raw = (
            raw_row.get("trace_id")
            or (raw_row.get("info") or {}).get("trace_id")
            or ""
        )
        trace_id = _to_str(trace_id_raw)
        try:
            canonical = normalize_mlflow_trace_row(raw_row)
            record, registry_by_node = parse_trace_mlflow(canonical)
            if record.get("test_case_id") in (trace_id, "", None):
                untagged.append(trace_id)
            records.append(record)
            for node, tools in registry_by_node.items():
                run_registry.setdefault(node, {}).update(tools)
        except Exception as exc:
            parse_errors.append(ParseError(trace_id=trace_id, error=repr(exc)))

    return MlflowParseResult(
        dataframe=pd.DataFrame.from_records(records),
        parse_errors=parse_errors,
        untagged_trace_ids=untagged,
        run_tool_registry=run_registry,
    )


def collect_run_prompts(traces_df: pd.DataFrame) -> dict[str, Any]:
    """Pull run-level system prompts + tool descriptions out of raw traces.

    Walks every row of the mlflow.search_traces output. For each agent's
    system prompt, returns the first non-empty value encountered. For tool
    descriptions, returns the union by tool name (description from the
    first row that defined it — the registry is stable within a run).

    Output shape matches :func:`write_prompt_sidecar`'s JSON payload.
    Per-trace uniformity is verified separately via the ``*_prompt_hash``
    and ``tool_descriptions_hash`` columns on the parsed dataframe.
    """
    main_prompt = ""
    dba_prompt = ""
    tool_descriptions: dict[str, str] = {}

    for raw_row in traces_df.to_dict(orient="records"):
        try:
            canonical = normalize_mlflow_trace_row(raw_row)
        except Exception:
            continue
        spans = canonical.get("data", {}).get("spans") or []
        if not main_prompt or not dba_prompt:
            children = _build_span_children(spans)
            prompts = extract_agent_system_prompts(spans, children)
            if not main_prompt:
                main_prompt = prompts["main_agent_system_prompt"]
            if not dba_prompt:
                dba_prompt = prompts["daily_banking_agent_system_prompt"]
        _, descriptions, _ = _extract_tool_registry(spans)
        for name, desc in descriptions.items():
            tool_descriptions.setdefault(name, desc)

    return {
        "main_agent_system_prompt": main_prompt,
        "daily_banking_agent_system_prompt": dba_prompt,
        "tool_descriptions": tool_descriptions,
    }


def write_prompt_sidecar(
    traces_df: pd.DataFrame,
    run_id: str,
    out_dir: Path | str,
) -> Path:
    """Write ``prompt_{run_id}.json`` next to the traces CSV.

    Sidecar carries the run-level system prompts + tool descriptions in
    full text. The CSV stores only the per-trace hashes; reports load
    this file to display the prompts and use the hashes to flag any
    cross-trace inconsistency.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"run_id": run_id, **collect_run_prompts(traces_df)}
    path = out_dir / f"prompt_{run_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return path
