"""Parse MLflow traces into a canonical wide DataFrame for evaluation scoring.

This is the **canonical** parser for evaluation traces produced by the
ai-orchestrator (`run_evals.py`). It targets the new evaluation surface
where each trace carries `eval_*` HUMAN assessments (`eval_item_id`,
`eval_domain`, `eval_persona`, `expected_agent`, `expected_tool_calls`,
`expected_response`, `guidelines`, `scorers`) and is intended to feed the
deterministic scorers in `ai-data-science/evals` (`ToolUsageScorer`,
`RoutingCorrectnessScorer`, `ToolParameterScorer`).

Compared to ``skkb_traces.py``:

- Uses **canonical column names** matching ``EvalFields``:
  ``actual_agent``, ``actual_tool_calls``, ``actual_response``,
  ``expected_agent``, ``expected_tool_calls``, ``expected_response``.
- Emits ``actual_tool_calls`` in **dict-form** (one dict per call with
  ``tool``, ``step``, ``parameters``, ``tool_call_id``, ``outputs_*``,
  ``error``, ``error_message``, ``agent``) — drop-in for the scorer's
  ``normalize_tool_list``.
- Pulls ``test_case_id`` from the new ``eval_item_id`` HUMAN assessment.
  Falls back to the SKKB-style ``\\btest case \\d+\\b`` regex over the
  user query for legacy compatibility.
- Extracts the per-trace **tool registry** (names + descriptions) from
  ``CHAT_MODEL`` span attributes (``mlflow.chat.tools``) — both as a flat
  list of names (``available_tools``) suitable for the scorer's
  ``available_tools`` arg, and as a name→description mapping
  (``tool_descriptions``) suitable for the run-config section of reports.
- Captures both span-level OTel status (``STATUS_CODE_OK`` / ``ERROR``)
  AND the LangChain ``ToolMessage.status`` (``"success"`` / ``"error"``)
  and combines them into a single ``error`` boolean per call.

Does **not** include SKKB-specific extraction (KB pipeline, reranker,
vector-DB hits). Those live in ``skkb_traces.py``, which is left
untouched.

Input shape: accepts either
  * the live ``mlflow.search_traces()`` DataFrame with flat columns
    (``trace_id``, ``trace_metadata``, ``assessments``, ``spans``, …), or
  * the JSONL load shape with two columns ``info`` and ``data``.

Output shape: see ``mlflow_traces.md`` for the full column dictionary.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import json
import re
from typing import Any

import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────

#: AI/Assistant message ``type`` values seen in ``mlflow.spanOutputs``.
_AI_TYPE_STRINGS = frozenset({"ai", "AIMessage", "AIMessageChunk"})

#: Test-case-id fallback regex (matches SKKB-style "Test case 17" only).
_TEST_CASE_ID_LEGACY_RE = re.compile(r"(?i)\btest case \d+\b")

#: Agent-level CHAIN spans we recognise when reconstructing the agent path.
#: This is intentionally explicit so that internal LangGraph nodes
#: (``llm``, ``tools``, ``tools_condition``, ``retrieve``, ``prune``,
#: ``rerank``, ``RunnableSequence``, ``LangGraph``, ``PydanticToolsParser``)
#: do not pollute ``actual_agents_path``.

#TODO: Add a strict check on these names
KNOWN_AGENT_NAMES = frozenset({
    "main_agent",
    "daily_banking_agent",
    "hg-invest-phase2",
})

#: Default value for "no agent routed" — chit-chat queries.
NO_AGENT = ""

#: Span-type literals (after JSON-decoding ``mlflow.spanType``).
_TOOL = "TOOL"
_CHAT_MODEL = "CHAT_MODEL"
_CHAIN = "CHAIN"


# ── Assessment names ───────────────────

class AssessmentNames:
    """Assessment names attached to each eval_item trace.

    These are the contract between the orchestrator's eval runner
    (``run_evals.py``) and downstream scoring. Stored on the trace via
    ``mlflow.update_current_trace`` calls inside the eval harness.
    """
    EVAL_ITEM_ID = "eval_item_id"
    EVAL_DOMAIN = "eval_domain"
    EVAL_PERSONA = "eval_persona"
    EXPECTED_AGENT = "expected_agent"
    EXPECTED_TOOL_CALLS = "expected_tool_calls"
    EXPECTED_RESPONSE = "expected_response"
    GUIDELINES = "guidelines"
    SCORERS = "scorers"


# ── Backwards-compat aliases ──────────────────────────────────────────
#
# Assessment names and scorer names have changed over the lifetime of
# the eval pipeline. Rather than make the parser branch on run vintage,
# we read the new canonical name first and fall back to the legacy
# name(s) when missing. Each map is "legacy → canonical".
#
# Concrete changes recorded so far:
#  - ``expected_tools`` (pre-GCAI-2566) → ``expected_tool_calls``.
#    The legacy payload was a flat list of tool name strings
#    (``["tool_a"]``); the new payload is a list of dicts with
#    ``step``/``tool``/``parameters``/``reason``. ``_normalize_expected_tool_calls``
#    below promotes the legacy shape so downstream scorers
#    (``ToolUsageScorer``, ``ToolParameterScorer``) receive the same
#    structure regardless of vintage.
#  - ``tool_input_parameters`` (current eval-team alias) → ``tool_parameter``
#    (canonical scorer key — ``ToolParameterScorer.name`` /
#    ``EvalFields.TOOL_PARAMETER``). We normalize the value inside
#    ``scorers_to_run`` so the downstream SCORER_REGISTRY lookup works
#    without per-notebook patching.
_LEGACY_ASSESSMENT_NAMES: dict[str, str] = {
    AssessmentNames.EXPECTED_TOOL_CALLS: "expected_tools",
}

_SCORER_NAME_ALIASES: dict[str, str] = {
    "tool_input_parameters": "tool_parameter",
}


def _normalize_expected_tool_calls(value: Any) -> list[dict[str, Any]]:
    """Coerce ``expected_tool_calls`` to the canonical dict-list shape.

    Accepts three shapes seen across run vintages:

    1. ``None`` / missing → ``[]``
    2. Legacy ``expected_tools`` payload: a list of tool name strings,
       e.g. ``["george-gcg-product_getLoans"]``. Promoted to
       ``[{"tool": name, "parameters": {}}]`` so downstream scorers see
       the same shape as the new format.
    3. New ``expected_tool_calls`` payload: a list of dicts with
       ``step``/``tool``/``parameters``/``reason``. Returned as-is.

    Any other shape is coerced to ``[]`` to avoid surprising downstream
    code that assumes ``list[dict]``.
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
    """Resolve scorer-name aliases to canonical keys.

    The eval team sometimes ships the same scorer under a different
    alias (e.g. ``tool_input_parameters`` for what the scorer code
    knows as ``tool_parameter``). We rewrite the names here so every
    downstream consumer can key off a single canonical set.
    """
    if not isinstance(value, list):
        return []
    return [_SCORER_NAME_ALIASES.get(str(s), str(s)) for s in value if s is not None]


# ── Result shape ───────────────────────────────────────────────────────

@dataclass
class ParseError:
    trace_id: str
    error: str


@dataclass
class MlflowParseResult:
    """Output of :func:`build_dataframe_from_mlflow_traces`.

    Attributes
    ----------
    dataframe
        One row per trace, canonical wide schema. See ``mlflow_traces.md``.
    parse_errors
        Per-row exceptions captured so a single broken trace does not
        crash the batch.
    untagged_trace_ids
        Traces whose ``test_case_id`` could not be resolved (no
        ``eval_item_id`` assessment, no SKKB regex match). Falls back to
        the trace_id itself; listed here so callers can audit.
    run_tool_registry
        Cross-trace aggregate registry: ``{node: {tool: description}}``.
        Computed once from the union of all CHAT_MODEL spans across all
        traces. Useful for the run-config section of the report.
    """

    dataframe: pd.DataFrame
    parse_errors: list[ParseError] = field(default_factory=list)
    untagged_trace_ids: list[str] = field(default_factory=list)
    run_tool_registry: dict[str, dict[str, str]] = field(default_factory=dict)


# ── Public entry point ─────────────────────────────────────────────────

def build_dataframe_from_mlflow_traces(
    traces_df: pd.DataFrame,
) -> MlflowParseResult:
    """Build the canonical evaluation DataFrame from MLflow traces.

    Parameters
    ----------
    traces_df
        DataFrame with one row per trace. Two shapes accepted:
        - JSONL shape: columns ``info`` and ``data``.
        - ``mlflow.search_traces()`` shape: flat columns
          (``trace_id``, ``trace_metadata``, ``assessments``, ``spans`` …).

    Returns
    -------
    MlflowParseResult
        Wide DataFrame plus diagnostics.
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
            canonical = _normalize_input_row(raw_row)
            record, registry_by_node = parse_trace(canonical)
            if record.get("test_case_id") in (trace_id, "", None):
                untagged.append(trace_id)
            records.append(record)
            for node, tools in registry_by_node.items():
                run_registry.setdefault(node, {}).update(tools)
        except Exception as exc:  # pragma: no cover - defensive
            parse_errors.append(ParseError(trace_id=trace_id, error=repr(exc)))

    return MlflowParseResult(
        dataframe=pd.DataFrame.from_records(records),
        parse_errors=parse_errors,
        untagged_trace_ids=untagged,
        run_tool_registry=run_registry,
    )


# ── Per-trace parsing ──────────────────────────────────────────────────

def parse_trace(trace: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    """Parse a single trace into a canonical record plus its tool registry.

    Returns
    -------
    record
        Flat dict — one row in the output DataFrame.
    registry_by_node
        ``{langgraph_node: {tool_name: description}}`` for this trace.
        Returned separately so the caller can union across traces for
        the run-level registry.
    """
    info = trace.get("info") or {}
    data = trace.get("data") or {}
    spans = _normalize_spans(data.get("spans") or [])

    trace_id = _to_str(info.get("trace_id"))
    request_time = _to_str(info.get("request_time"))
    duration_ms = info.get("execution_duration_ms")
    state = _to_str(info.get("state"))
    trace_metadata = _coerce_mapping(info.get("trace_metadata"))
    tags = _coerce_mapping(info.get("tags"))

    user_query = _extract_user_query(trace_metadata)
    assessments_raw = info.get("assessments") or []
    assessments = _extract_assessments(assessments_raw)

    test_case_id = _resolve_test_case_id(
        assessments=assessments,
        user_query=user_query,
        trace_id=trace_id,
    )

    expected_agent = _to_str(assessments.get(AssessmentNames.EXPECTED_AGENT, ""))
    # NEW ``expected_tool_calls`` first, fall back to legacy ``expected_tools``;
    # the helper unifies the legacy string-list and the new dict-list shapes.
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
    # Rewrite scorer-name aliases (e.g. ``tool_input_parameters`` →
    # ``tool_parameter``) so the downstream registry lookup is stable.
    scorers_to_run = _normalize_scorers_to_run(assessments.get(AssessmentNames.SCORERS))

    # Actual outputs from spans
    actual_agents_path, actual_agent = _extract_actual_agent_path(spans)
    actual_tool_calls = _extract_actual_tool_calls(spans)
    actual_response = _extract_actual_response(spans, trace_metadata)

    # Tool registry — what the LLM had on offer this turn
    available_tools, tool_descriptions, registry_by_node = _extract_tool_registry(spans)

    # Provenance / cost
    token_usage = _decode_json(trace_metadata.get("mlflow.trace.tokenUsage")) or {}
    git_branch = _to_str(trace_metadata.get("mlflow.source.git.branch"))
    git_commit = _to_str(trace_metadata.get("mlflow.source.git.commit"))
    source_run_id = _to_str(trace_metadata.get("mlflow.sourceRun"))
    mlflow_user = _to_str(trace_metadata.get("mlflow.user"))
    session_id = _to_str(trace_metadata.get("mlflow.trace.session"))

    # Detect supervisor split (gate/ethics/language/route/decide/synthesis)
    # vs the classic main_agent monolith. Useful for cross-run comparison.
    architecture = _classify_architecture(actual_agents_path)

    record = {
        # ─── identity ─────────────────────────────────────────────────
        "trace_id": trace_id,
        "test_case_id": test_case_id,
        "session_id": session_id,
        "request_time": request_time,
        "execution_duration_ms": duration_ms,
        "state": state,
        # ─── eval expectations (HUMAN assessments) ────────────────────
        "user_query": user_query,
        "eval_domain": eval_domain,
        "eval_persona": eval_persona,
        "expected_agent": expected_agent,
        "expected_tool_calls": expected_tool_calls,
        "expected_response": expected_response,
        "guidelines": guidelines,
        "scorers_to_run": scorers_to_run,
        # ─── actuals from spans ───────────────────────────────────────
        "actual_agent": actual_agent,
        "actual_agents_path": actual_agents_path,
        "actual_tool_calls": actual_tool_calls,
        "actual_response": actual_response,
        # ─── per-trace tool surface (LLM-visible registry) ────────────
        "available_tools": available_tools,
        "tool_descriptions": tool_descriptions,
        "tool_registry_by_node": {
            node: list(tools.keys()) for node, tools in registry_by_node.items()
        },
        # ─── provenance / cost ────────────────────────────────────────
        "architecture": architecture,
        "model": _extract_model(spans),
        "token_usage": token_usage,
        "git_branch": git_branch,
        "git_commit": git_commit,
        "source_run_id": source_run_id,
        "mlflow_user": mlflow_user,
        # ─── raw passthroughs (kept for debugging / iteration 2) ──────
        "tags": tags,
        "assessments_raw": [_to_plain_data(a) for a in assessments_raw],
    }
    return record, registry_by_node


# ── Input normalization ────────────────────────────────────────────────

def _normalize_input_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Accept either JSONL (``info``/``data``) or live ``search_traces`` shape."""
    if "info" in row and "data" in row:
        return {
            "info": _coerce_mapping(row.get("info")),
            "data": _coerce_mapping(row.get("data")),
        }
    # Flat search_traces shape — reassemble.
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
            "spans": _coerce_list(row.get("spans")),
        },
    }


def _normalize_spans(raw_spans: Iterable[Any]) -> list[dict[str, Any]]:
    return [_normalize_span(s) for s in raw_spans if isinstance(_to_plain_data(s), dict)]


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


# ── Assessment extraction ──────────────────────────────────────────────

def _extract_assessments(assessments_raw: Iterable[Any]) -> dict[str, Any]:
    """Decode HUMAN assessments into ``{name: deserialized_value}``.

    For CODE assessments with the same name, we keep the **most recent**
    one by ``last_update_time`` (lexicographic on ISO strings). HUMAN
    assessments are written exactly once per eval run by the orchestrator,
    so dedup matters mainly for re-graded runs.

    Recognises both shapes the orchestrator emits:
    - ``expectation = {"value": <python value>}`` for plain types
    - ``expectation = {"serialized_value": {"value": "<json string>",
       "serialization_format": "JSON"}}`` for lists / dicts
    """
    by_name: dict[str, tuple[str, Any]] = {}  # name → (last_update_time, value)
    for a in assessments_raw:
        a = _to_plain_data(a)
        if not isinstance(a, dict):
            continue
        name = _to_str(a.get("assessment_name"))
        if not name:
            continue
        ts = _to_str(a.get("last_update_time"))
        # decode value from expectation OR feedback
        value = None
        if "expectation" in a:
            value = _decode_assessment_payload(a.get("expectation"))
        elif "feedback" in a:
            value = _decode_assessment_payload(a.get("feedback"))
        prev = by_name.get(name)
        if prev is None or ts >= prev[0]:
            by_name[name] = (ts, value)
    return {name: value for name, (_, value) in by_name.items()}


def _decode_assessment_payload(payload: Any) -> Any:
    """Decode one ``expectation`` or ``feedback`` payload to a Python value.

    Handles the two MLflow shapes:
    - ``{"value": <python>}`` → returned as-is
    - ``{"serialized_value": {"value": "<json>", "serialization_format":
       "JSON"}}`` → JSON-decoded
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


# ── test_case_id resolution ────────────────────────────────────────────

def _resolve_test_case_id(
    *,
    assessments: Mapping[str, Any],
    user_query: str,
    trace_id: str,
) -> str:
    """Resolve a stable per-row id with three-tier fallback.

    1. ``eval_item_id`` HUMAN assessment (preferred — set by run_evals.py).
    2. SKKB-style "Test case N" regex over the user query (legacy nightly
       SKKB traces).
    3. ``trace_id`` itself (worst case, signals the row is untagged).
    """
    candidate = _to_str(assessments.get(AssessmentNames.EVAL_ITEM_ID))
    if candidate:
        return candidate
    if user_query:
        m = _TEST_CASE_ID_LEGACY_RE.search(user_query)
        if m:
            return m.group(0).strip().lower().replace(" ", "_")  # → "test_case_17"
    return trace_id


# ── User query / response extraction ───────────────────────────────────

def _extract_user_query(trace_metadata: Mapping[str, Any]) -> str:
    """Pull the user-facing prompt from ``mlflow.traceInputs``.

    The orchestrator writes inputs as
    ``{"messages": [["human", "<query>"]]}``.
    """
    raw = trace_metadata.get("mlflow.traceInputs")
    payload = _decode_json(raw) if isinstance(raw, str) else raw
    if not isinstance(payload, Mapping):
        return ""
    messages = payload.get("messages") or []
    if not messages:
        return ""
    first = messages[0]
    if isinstance(first, list) and len(first) >= 2:
        return _to_str(first[1])
    if isinstance(first, Mapping):
        # AIMessage-style shape
        return _to_str(first.get("content"))
    return ""


def _extract_actual_response(
    spans: list[dict[str, Any]],
    trace_metadata: Mapping[str, Any],
) -> str:
    """Pick the final assistant-facing response text.

    Strategy (first non-empty wins):

    1. **NEW orchestrator (post GCAI-2566 / GCAI-3581):** look for a span
       named ``agent_answer`` with ``mlflow.spanType == "AGENT"``. Its
       ``mlflow.spanOutputs`` carries ``{"question": ..., "answer": ...}``
       — the cleanest single source of the final response.
    2. **Generic fallback (works on both vintages):** scan
       ``mlflow.spanOutputs`` of every span, look for AI-typed messages
       with ``finish_reason == "stop"`` (or absent), and pick the longest
       matching ``content``. Mirrors the SKKB ``extract_final_ai_content``
       heuristic.
    3. **Last-resort fallback:** last AI message in
       ``mlflow.traceOutputs.messages``.
    """
    answer = _extract_answer_from_agent_answer_span(spans)
    if answer:
        return answer

    best = ""
    for span in spans:
        attrs = span.get("attributes") or {}
        parsed = _parse_attr_json(attrs, "mlflow.spanOutputs")
        if parsed is None:
            continue
        for node in _walk_dicts(parsed):
            node_type = node.get("type")
            if node_type not in _AI_TYPE_STRINGS and str(node_type).lower() != "ai":
                continue
            content = node.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            response_metadata = node.get("response_metadata") or {}
            finish_reason = response_metadata.get("finish_reason")
            if finish_reason is not None and finish_reason != "stop":
                continue
            if len(content) > len(best):
                best = content
    if best:
        return best
    # Last-resort fallback: last AI message in mlflow.traceOutputs.messages.
    raw_out = trace_metadata.get("mlflow.traceOutputs")
    payload = _decode_json(raw_out) if isinstance(raw_out, str) else raw_out
    if isinstance(payload, Mapping):
        for msg in reversed(payload.get("messages") or []):
            if isinstance(msg, Mapping) and msg.get("type") in _AI_TYPE_STRINGS:
                return _to_str(msg.get("content"))
    return ""


def _extract_answer_from_agent_answer_span(spans: list[dict[str, Any]]) -> str:
    """Read the ``answer`` field from the NEW ``agent_answer`` AGENT span.

    Returns ``""`` when the span is absent, when ``mlflow.spanOutputs``
    can't be JSON-decoded, or when the ``answer`` key is missing/blank.
    Falls back silently — callers should chain to the generic scan.
    """
    for span in spans:
        if span.get("name") != "agent_answer":
            continue
        if _span_type(span) != "AGENT":
            continue
        attrs = span.get("attributes") or {}
        parsed = _parse_attr_json(attrs, "mlflow.spanOutputs")
        if isinstance(parsed, Mapping):
            answer = parsed.get("answer")
            if isinstance(answer, str) and answer.strip():
                return answer
    return ""


# ── Agent path extraction ──────────────────────────────────────────────

def _extract_actual_agent_path(spans: list[dict[str, Any]]) -> tuple[list[str], str]:
    """Return ``(actual_agents_path, actual_agent)``.

    ``actual_agents_path`` is the ordered de-duplicated sequence of CHAIN
    spans whose ``name`` is in :data:`KNOWN_AGENT_NAMES`. ``actual_agent``
    is the last entry, or ``NO_AGENT`` ("") if the trace never reached a
    sub-agent (chit-chat / refusal / supervisor-only).

    We use CHAIN span ``name`` rather than ``langgraph_node`` because
    CHAIN spans have the agent's own name (``daily_banking_agent``)
    whereas ``langgraph_node`` for inner spans is ``llm`` / ``tools`` /
    ``rerank`` etc.
    """
    seen: set[str] = set()
    path: list[str] = []
    for span in spans:
        if _span_type(span) != _CHAIN:
            continue
        name = _to_str(span.get("name"))
        if name in KNOWN_AGENT_NAMES and name not in seen:
            seen.add(name)
            path.append(name)
    last = path[-1] if path else NO_AGENT
    return path, last


def _classify_architecture(agents_path: list[str]) -> str:
    """Coarse architecture tag for cross-run comparison.

    - ``"classic"``: traverses ``main_agent``
    - ``"hg-invest"``: traverses ``hg-invest-phase2``
    - ``"none"``: never reached a known agent (chit-chat / refusal)
    """
    if "main_agent" in agents_path:
        return "classic"
    if "hg-invest-phase2" in agents_path:
        return "hg-invest"
    return "none"


# ── Tool calls extraction ──────────────────────────────────────────────

def _extract_actual_tool_calls(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract every TOOL span as a scorer-ready dict.

    Spans are visited in input order (the JSONL preserves dependency
    order). Each emitted dict has:

    - ``tool``: span name (e.g. ``"george-gcg-product_getLoans"``)
    - ``step``: 1-based positional index
    - ``arguments``: parsed ``mlflow.spanInputs`` (the args the LLM
      passed). Canonical key on the **actual** side — the
      ``ToolParameterScorer`` reads it under this name. Mirror of
      ``expected_tool_calls[i].parameters`` on the reference side; the
      asymmetric naming ("expected parameters" vs "actual arguments") is
      intentional in the scorer contract.
    - ``parameters``: alias of ``arguments``, kept so callers that read
      ``parameters`` symmetrically with the expected side still work.
      Both keys carry the same value — the args the LLM actually passed.
    - ``tool_call_id``: ``attributes.tool_call_id`` (links back to the LLM
      ``tool_calls`` block in the upstream CHAT_MODEL span)
    - ``outputs_preview``: first 500 chars of the ToolMessage ``content``
    - ``outputs_text``: full ToolMessage ``content`` string
    - ``outputs_obj``: parsed ToolMessage payload (dict) — useful for
      programmatic inspection in iteration 2
    - ``error``: True iff either the OTel span status reports non-OK OR
      the ToolMessage ``status`` reports non-success
    - ``error_message``: span ``status.message`` if present, else ToolMessage
      ``content`` when error
    - ``agent``: ``langgraph_node`` parent's owning agent (best-effort,
      currently the inner ``langgraph_node`` such as ``"tools"`` — useful
      diagnostic, can be cleaned up in iteration 2)
    - ``span_status_code`` / ``tool_message_status``: raw status strings,
      kept for transparency
    """
    calls: list[dict[str, Any]] = []
    step = 0
    for span in spans:
        if _span_type(span) != _TOOL:
            continue
        step += 1
        attrs = span.get("attributes") or {}
        params = _parse_attr_json(attrs, "mlflow.spanInputs")
        outputs_obj = _parse_attr_json(attrs, "mlflow.spanOutputs") or {}
        if not isinstance(outputs_obj, dict):
            outputs_obj = {}
        outputs_text = _to_str(outputs_obj.get("content"))
        outputs_preview = outputs_text[:500] + ("…" if len(outputs_text) > 500 else "")

        # Status — two layers
        span_status = span.get("status") or {}
        span_code = _to_str(span_status.get("code"))
        span_message = _to_str(span_status.get("message"))
        tm_status = _to_str(outputs_obj.get("status"))  # ToolMessage.status
        is_error = (
            (span_code and "OK" not in span_code.upper() and "UNSET" not in span_code.upper())
            or (tm_status and tm_status.lower() != "success")
        )
        error_message = span_message or (outputs_text if is_error else "")

        # tool_call_id is JSON-encoded ("call_xyz") — strip quoting
        raw_tcid = attrs.get("tool_call_id")
        if isinstance(raw_tcid, str):
            try:
                tool_call_id = _to_str(json.loads(raw_tcid))
            except json.JSONDecodeError:
                tool_call_id = raw_tcid
        else:
            tool_call_id = _to_str(raw_tcid)

        # Best-effort owning langgraph_node (will be "tools" usually).
        metadata = _parse_attr_json(attrs, "metadata") or {}
        owning_node = _to_str(
            metadata.get("langgraph_node") if isinstance(metadata, dict) else ""
        )

        # ``ToolParameterScorer`` reads the args under ``arguments`` on
        # the actual side (asymmetric with the reference's ``parameters``).
        # Emit both keys carrying the same value so every consumer works:
        # the scorer reads ``arguments``, callers that mirror the expected
        # side's ``parameters`` still see the data.
        args = params if isinstance(params, dict) else {}
        calls.append({
            "tool": _to_str(span.get("name")),
            "step": step,
            "arguments": args,
            "parameters": args,  # alias — see comment above
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


# ── Tool registry extraction ───────────────────────────────────────────

def _extract_tool_registry(
    spans: list[dict[str, Any]],
) -> tuple[list[str], dict[str, str], dict[str, dict[str, str]]]:
    """Collect tools the LLM was given in this trace.

    Reads ``mlflow.chat.tools`` from every CHAT_MODEL span. The same tool
    can appear in multiple spans (one per LLM call); we union by name and
    keep the description from the first occurrence (descriptions are
    stable across the run — verified empirically).

    Returns
    -------
    available_tools
        Sorted list of unique tool names (input for the scorer's
        ``available_tools`` arg).
    tool_descriptions
        ``{tool_name: description}`` for the report.
    registry_by_node
        ``{langgraph_node: {tool_name: description}}`` — preserves which
        agent's LLM had access to which tools (e.g. ``main_agent`` only
        sees ``Router``, ``daily_banking_agent``'s ``llm`` sees the full
        MCP API surface).
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
    """Best-effort: model name from the first CHAT_MODEL span."""
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


# ── Generic span helpers (local, intentionally duplicated from skkb_traces
#    so the new module has no behavioural dependency on the old one) ────

def _span_type(span: Mapping[str, Any]) -> str:
    """Return the JSON-decoded ``mlflow.spanType`` (e.g. ``"TOOL"``)."""
    raw = (span.get("attributes") or {}).get("mlflow.spanType")
    if isinstance(raw, str) and raw.startswith('"'):
        try:
            return _to_str(json.loads(raw))
        except json.JSONDecodeError:
            return raw
    return _to_str(raw)


def _parse_attr_json(attrs: Mapping[str, Any], key: str) -> Any:
    """Parse a span-attribute value that may be a JSON string."""
    val = attrs.get(key)
    if isinstance(val, str):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return val
    return val


def _walk_dicts(obj: Any) -> Iterable[dict[str, Any]]:
    """Yield every dict reachable inside a nested dict/list structure."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk_dicts(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_dicts(item)


# ── Generic conversion helpers ─────────────────────────────────────────

def _decode_json(value: Any) -> Any:
    """Best-effort JSON decode; returns the original on failure."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _coerce_mapping(value: Any) -> dict[str, Any]:
    value = _to_plain_data(value)
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(loaded) if isinstance(loaded, Mapping) else {}
    return {}


def _coerce_list(value: Any) -> list[Any]:
    value = _to_plain_data(value)
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return []
        return loaded if isinstance(loaded, list) else []
    return []


def _to_plain_data(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {key: _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain_data(item) for item in value]
    if isinstance(value, tuple):
        return [_to_plain_data(item) for item in value]
    if isinstance(value, set):
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
        return {key: _to_plain_data(item) for key, item in value_dict.items()}
    return str(value)


def _to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)
