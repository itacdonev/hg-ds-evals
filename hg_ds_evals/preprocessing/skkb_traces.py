"""SKKB trace parsing helpers.

These functions turn live MLflow ``search_traces`` results into the same flat
record shape produced by ``experiments/skkb/notebooks/skkb_001_data_preparation.ipynb``.
The parser logic is intentionally aligned with the notebook contract so the
resulting DataFrame can feed the same downstream eval flow.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

import pandas as pd

AI_TYPE_STRINGS = {"ai", "AIMessage", "AIMessageChunk"}
ENUM_RE = re.compile(r"(?m)^([A-Z][A-Z0-9_@-]*(?:\s*\+\s*[A-Z][A-Z0-9_@-]*)?)(?=:)")
RERANK_CANDIDATE_RE = re.compile(
    r"(?ms)^- ID:\s*(?P<enum_id>[^|\n]+?)\s*\|\s*"
    r"Group:\s*(?P<group_name>[^|\n]+?)\s*\|\s*"
    r"Description:\s*(?P<description>.*?)(?=^- ID:\s|\Z)"
)
_TS_TAIL_RE = re.compile(r"Current date and time:.*$", re.DOTALL)
_TEST_CASE_ID_RE = re.compile(r"(?i)\btest case \d+\b")

KB_TOOL_NAMES = {"knowledge_search"}
BANKING_DATA_TOOL_PREFIX = "mock_banking_"
HG_INVEST_AGENT_NAME = "hg-invest-phase2"
# TODO: confirm exact prefix once hg-invest tools are wired in a future PR.
# Current placeholder mirrors BANKING_DATA_TOOL_PREFIX. No traces in the
# 2026-04-24 nightly run exercised these tools yet.
HG_INVEST_DATA_TOOL_PREFIX = "mock_invest_"

# `agents_called` collects every distinct `langgraph_node` value seen in the
# trace, which is the entire graph traversal — supervisor + sub-agents +
# internal nodes (`llm`, `tools`, `retrieve`, `prune`, `rerank`). Use
# KNOWN_AGENT_NAMES to filter that down to the actual agents.
KNOWN_AGENT_NAMES = frozenset({"main_agent", "daily_banking_agent", HG_INVEST_AGENT_NAME})
TRACE_TEST_CASE_KEYS = (
    # NEW orchestrator (trace_schema v3 with the GCAI-3581 selector runtime):
    # the eval framework writes the test case id to ``info.tags`` as
    # ``eval.test_case_id`` and also into each span's
    # ``attributes.metadata["eval.test_case_id"]``.
    "eval.test_case_id",
    # Legacy orchestrator: the same id used to live in
    # ``info.assessments`` (HUMAN expectation ``test_case_id``) or in
    # ``mlflow.traceInputs`` payloads.
    "test_case_id",
    "testCaseId",
    "test_case",
    "dataset_example_id",
    "datasetExampleId",
)


@dataclass
class SKKBParseResult:
    """Flat SKKB parse result plus lightweight diagnostics."""

    dataframe: pd.DataFrame
    parse_errors: list[tuple[str, str]]
    unmapped_trace_ids: list[str]


@dataclass(frozen=True)
class _CandidateDocument:
    enum_id: str
    description: str
    group_name: str = ""


def build_skkb_dataframe_from_mlflow_search_traces(
    traces_df: pd.DataFrame,
) -> SKKBParseResult:
    """Build the SKKB flat table from ``mlflow.search_traces`` pandas output.

    Returns:
        Parse result with a flat DataFrame matching the notebook schema.
    """

    records: list[dict[str, Any]] = []
    parse_errors: list[tuple[str, str]] = []
    unmapped_trace_ids: list[str] = []

    for row in traces_df.to_dict(orient="records"):
        trace_id = _to_str(row.get("trace_id"))
        try:
            canonical_trace = normalize_mlflow_trace_row(row)
            test_case_id = resolve_test_case_id(canonical_trace)
            if test_case_id == trace_id:
                unmapped_trace_ids.append(trace_id)
            records.append(parse_trace(test_case_id, canonical_trace))
        except Exception as exc:
            parse_errors.append((trace_id, repr(exc)))

    return SKKBParseResult(
        dataframe=pd.DataFrame.from_records(records),
        parse_errors=parse_errors,
        unmapped_trace_ids=unmapped_trace_ids,
    )


def resolve_test_case_id(
    trace: dict[str, Any],
) -> str:
    """Resolve the original ``test_case_id`` for a canonical trace.

    The imported table must be sourced strictly from MLflow traces. When the
    run logged a native test-case identifier in trace metadata, tags, inputs,
    or assessments we preserve it; otherwise we fall back to ``trace_id``.
    """

    info = trace.get("info") or {}
    trace_native_test_case_id = _extract_trace_native_test_case_id(trace)
    if trace_native_test_case_id:
        return trace_native_test_case_id
    return _to_str(info.get("trace_id"))


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

    assessment_test_case_id = _extract_test_case_id_from_assessments(info.get("assessments"))
    if assessment_test_case_id:
        return assessment_test_case_id

    client_request_id = _normalize_test_case_id_candidate(info.get("client_request_id"))
    if client_request_id:
        return client_request_id

    return ""


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


def _assessment_name(assessment: Mapping[str, Any]) -> str:
    return _to_str(assessment.get("assessment_name") or assessment.get("name"))


def _normalize_test_case_id_candidate(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    stripped = value.strip()
    if not stripped:
        return ""

    exact_match = _TEST_CASE_ID_RE.search(stripped)
    if exact_match is not None:
        return exact_match.group(0)

    return stripped if stripped.lower().startswith("test case ") else ""


def _parse_json_like_payload(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return None


def _coerce_mapping_payload(value: Any) -> dict[str, Any]:
    parsed = _parse_json_like_payload(value)
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def normalize_mlflow_trace_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a ``search_traces`` pandas row into the notebook's trace shape."""

    trace_metadata = _coerce_mapping(row.get("trace_metadata"))
    tags = _coerce_mapping(row.get("tags"))
    assessments = [_to_plain_data(assessment) for assessment in _coerce_list(row.get("assessments"))]
    spans = [_normalize_span(span) for span in _coerce_list(row.get("spans")) if isinstance(_to_plain_data(span), dict)]

    return {
        "info": {
            "trace_id": row.get("trace_id"),
            "client_request_id": row.get("client_request_id"),
            "state": row.get("state"),
            "request_time": row.get("request_time"),
            "execution_duration_ms": row.get("execution_duration_ms") or row.get("execution_duration"),
            "trace_metadata": trace_metadata,
            "tags": tags,
            "assessments": assessments,
        },
        "data": {
            "spans": spans,
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
        return {
            key: _to_plain_data(item)
            for key, item in value_dict.items()
            if not key.startswith("_")
        }

    return value


def _walk_dicts(obj: Any) -> Iterable[dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _walk_dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_dicts(value)


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


def _span_langgraph_node(span: Mapping[str, Any]) -> str:
    metadata = _parse_attr_json(span.get("attributes") or {}, "metadata")
    if not isinstance(metadata, Mapping):
        return ""
    node = metadata.get("langgraph_node")
    return node if isinstance(node, str) else ""


def _span_matches_node(span: Mapping[str, Any], node_names: set[str]) -> bool:
    span_name = span.get("name")
    return span_name in node_names or _span_langgraph_node(span) in node_names


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
    value = span.get(key)
    try:
        return int(value)
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


def _parse_attr_json(attrs: dict[str, Any], key: str) -> Any:
    raw = attrs.get(key, "")
    if not raw:
        return None
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    return raw


def _span_payload(span: Mapping[str, Any], attr_key: str, direct_key: str) -> Any:
    parsed = _parse_attr_json(span.get("attributes") or {}, attr_key)
    if parsed is not None:
        return parsed
    return span.get(direct_key)


def _span_inputs(span: Mapping[str, Any]) -> Any:
    return _span_payload(span, "mlflow.spanInputs", "inputs")


def _span_outputs(span: Mapping[str, Any]) -> Any:
    return _span_payload(span, "mlflow.spanOutputs", "outputs")


def _md5_short(value: str) -> str:
    return hashlib.md5((value or "").encode("utf-8", errors="ignore")).hexdigest()[:10]


def _get_agent_chat_system_prompt(
    spans: list[dict[str, Any]],
    children: dict[str, list[dict[str, Any]]],
    agent_name: str,
) -> str:
    agent = next((span for span in spans if span.get("name") == agent_name), None)
    if agent is None:
        return ""

    stack = [agent]
    while stack:
        current = stack.pop()
        if current.get("name") == "ChatDatabricks":
            span_inputs = _parse_attr_json(current.get("attributes") or {}, "mlflow.spanInputs")
            if isinstance(span_inputs, list) and span_inputs and isinstance(span_inputs[0], list):
                for message in span_inputs[0]:
                    if not isinstance(message, dict):
                        continue
                    if message.get("role") == "system" or message.get("type") == "system":
                        content = message.get("content", "")
                        if isinstance(content, str) and content.strip():
                            return content
        stack.extend(children.get(_to_str(current.get("span_id")), []))
    return ""


def extract_agent_prompt_hashes(
    spans: list[dict[str, Any]],
    children: dict[str, list[dict[str, Any]]],
) -> dict[str, str]:
    main_raw = _get_agent_chat_system_prompt(spans, children, "main_agent")
    dba_raw = _get_agent_chat_system_prompt(spans, children, "daily_banking_agent")
    main_stripped = _TS_TAIL_RE.sub("", main_raw).rstrip()
    dba_stripped = _TS_TAIL_RE.sub("", dba_raw).rstrip()
    return {
        "main_agent_prompt_hash": _md5_short(main_stripped),
        "daily_banking_agent_prompt_hash": _md5_short(dba_stripped),
    }


def extract_final_ai_content(spans: list[dict[str, Any]]) -> str:
    best = ""
    for span in spans:
        parsed = _parse_attr_json(span.get("attributes") or {}, "mlflow.spanOutputs")
        if parsed is None:
            continue
        for node in _walk_dicts(parsed):
            node_type = node.get("type")
            if node_type not in AI_TYPE_STRINGS and str(node_type).lower() != "ai":
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
    return best


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


def _append_candidate_document(
    target: list[_CandidateDocument],
    candidate: _CandidateDocument,
    *,
    deduplicate: bool = True,
) -> None:
    if not deduplicate or candidate.enum_id not in {item.enum_id for item in target}:
        target.append(candidate)


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
            description=description if isinstance(description, str) else json.dumps(description, ensure_ascii=False),
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


def _extract_reranker_prompt_candidate_documents(reranker_user_prompt: str) -> list[_CandidateDocument]:
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


def _format_retrieved_candidates(candidates: list[_CandidateDocument]) -> tuple[str, list[str]]:
    blocks = [
        f"{candidate.enum_id}: {candidate.description}"
        for candidate in candidates
        if candidate.enum_id
    ]
    return ("\n\n".join(blocks), [candidate.enum_id for candidate in candidates if candidate.enum_id])


def _is_vector_db_http_span(span: Mapping[str, Any]) -> bool:
    name = _to_str(span.get("name"))
    if "/admin/knowledge-base/query" in name:
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
    """Extract per-query vector DB hits before merge/dedup/prune."""

    hit_candidates: list[_CandidateDocument] = []
    hits_by_query: dict[str, int] = {}
    query_limits: list[int] = []

    for span in spans:
        if not _is_vector_db_http_span(span):
            continue

        span_inputs = _span_inputs(span)
        request_body = _http_body_json(span_inputs)
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


def extract_pre_prune_candidates(spans: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Extract merged retrieve candidates before the prune node."""

    for span in spans:
        if not _span_matches_node(span, {"retrieve"}):
            continue
        candidates = _extract_candidate_documents_from_payload(_span_outputs(span))
        if candidates:
            return _format_retrieved_candidates(candidates)

    return ("", [])


def extract_post_prune_candidates(spans: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Extract candidates after pruning and before reranking."""

    for span in spans:
        if not _span_matches_node(span, {"prune"}):
            continue
        candidates = _extract_candidate_documents_from_payload(_span_outputs(span))
        if candidates:
            return _format_retrieved_candidates(candidates)

    return ("", [])


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


_PRUNE_LEAF_SPAN_NAMES = (
    "knowledge_prune",  # NEW orchestrator (GCAI-3581+)
    "kb_prune",         # legacy orchestrator
)


def extract_prune_counts(spans: list[dict[str, Any]]) -> dict[str, int | bool]:
    """Extract explicit pruning counters from the prune-leaf span.

    The leaf span name changed in the GCAI-3581 selector runtime:
    legacy traces emitted ``kb_prune``; newer traces emit
    ``knowledge_prune``. We accept either — first match wins — so the
    same parser works on both run vintages.
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
        span_inputs = _span_inputs(span)
        span_outputs = _span_outputs(span)
        inputs = span_inputs if isinstance(span_inputs, Mapping) else {}
        outputs = span_outputs if isinstance(span_outputs, Mapping) else {}
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
    """Extract the candidate pool available to the reranker.

    Direct MLflow trace rows do not always expose node state in the same shape
    as JSON exports. The reranker prompt is the exact retrieve/prune -> rerank
    handoff, so prefer it when present; otherwise fall back to structured node
    outputs.
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

    return ("", [])


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
                    selected_ids = [value for value in args["selected_ids"] if isinstance(value, str)]
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


def extract_agent_and_tool_calls(spans: list[dict[str, Any]]) -> dict[str, Any]:
    agents_seen: list[str] = []
    seen_set: set[str] = set()
    tools: list[dict[str, Any]] = []

    for span in spans:
        attrs = span.get("attributes") or {}

        metadata = _parse_attr_json(attrs, "metadata")
        if isinstance(metadata, dict):
            node = metadata.get("langgraph_node")
            if isinstance(node, str) and node and node not in seen_set:
                seen_set.add(node)
                agents_seen.append(node)

        span_type_raw = attrs.get("mlflow.spanType", "")
        if isinstance(span_type_raw, str) and span_type_raw.startswith('"'):
            try:
                span_type = json.loads(span_type_raw)
            except json.JSONDecodeError:
                span_type = span_type_raw
        else:
            span_type = span_type_raw

        if span_type == "TOOL":
            tools.append(
                {
                    "name": span.get("name"),
                    "inputs": _parse_attr_json(attrs, "mlflow.spanInputs"),
                }
            )

    agents_only = [node for node in agents_seen if node in KNOWN_AGENT_NAMES]
    last_agent = agents_only[-1] if agents_only else ""

    return {
        "agents_called": agents_seen,
        "agents_only": agents_only,
        "last_agent": last_agent,
        "tools_called": tools,
    }


def classify_query_scope(agents_called: list[str], tools_called: list[dict[str, Any]]) -> str:
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

    Pure function over existing parser columns — no LLM, no I/O. Categories
    are mutually exclusive; earliest failing stage wins:

    - ``not_applicable``           — query_scope didn't exercise the KB pipeline
    - ``retrieval_no_hits``        — vector DB returned 0 candidates
    - ``prune_dropped_all``        — pre_prune > 0 but post_prune == 0
    - ``reranker_empty_selection`` — reranker ran on a non-empty pool but selected nothing
    - ``none``                     — full funnel produced output
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
    """Recompute trace-derived columns in place from existing trace data.

    Reads ``agents_called`` and ``tools_called`` from the frame (parsing
    JSON-serialized strings when needed) and overwrites:

    - ``query_scope``                    — current ``classify_query_scope``
    - ``trace_invariant_violations``     — current ``check_trace_invariants``
    - ``agents_only``                    — ``agents_called`` filtered to KNOWN_AGENT_NAMES
    - ``last_agent``                     — last entry of ``agents_only`` (or "")

    All other columns are left untouched. Use this to refresh trace
    classifications after a parser change without re-running expensive
    upstream LLM-judge calls.
    """

    def _coerce_list(value: Any) -> list[Any]:
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
    serialize_lists_as_string = any(isinstance(value, str) for value in agents_input)

    new_scopes: list[str] = []
    new_violations: list[Any] = []
    new_agents_only: list[Any] = []
    new_last_agent: list[str] = []
    for agents_raw, tools_raw in zip(agents_input, tools_input):
        agents = _coerce_list(agents_raw)
        tools = _coerce_list(tools_raw)
        scope = classify_query_scope(agents, tools)
        violations = check_trace_invariants(agents, tools, scope)
        agents_only = [node for node in agents if node in KNOWN_AGENT_NAMES]
        new_scopes.append(scope)
        new_violations.append(
            json.dumps(violations, ensure_ascii=False) if serialize_lists_as_string else violations
        )
        new_agents_only.append(
            json.dumps(agents_only, ensure_ascii=False) if serialize_lists_as_string else agents_only
        )
        new_last_agent.append(agents_only[-1] if agents_only else "")

    df["query_scope"] = new_scopes
    df["trace_invariant_violations"] = new_violations
    df["agents_only"] = new_agents_only
    df["last_agent"] = new_last_agent
    return df


def _extract_expected_enums_weights(assessment: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
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

    return "{}", {}


def parse_trace(test_case_id: str, trace: dict[str, Any]) -> dict[str, Any]:
    info = trace.get("info") or {}
    data = trace.get("data") or {}
    spans = _sort_spans_by_dependency_order(data.get("spans") or [])
    children = _build_span_children(spans)

    trace_metadata = info.get("trace_metadata") or {}
    user_query = _extract_user_query(trace_metadata, spans)

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
    agent_response = extract_final_ai_content(spans)
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
        reranker_raw_selected_ids = json.loads(_to_str(final_kb_run.get("reranker_raw_selected_ids")) or "[]")
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

    agent_hashes = extract_agent_prompt_hashes(spans, children)
    calls = extract_agent_and_tool_calls(spans)
    query_scope = classify_query_scope(calls["agents_called"], calls["tools_called"])
    trace_invariant_violations = check_trace_invariants(
        calls["agents_called"],
        calls["tools_called"],
        query_scope,
    )
    prune_counts = {
        key: final_kb_run.get(key, value)
        for key, value in extract_prune_counts([]).items()
    }

    if query_scope == "kb":
        post_prune_id_set = set(post_prune_ids)
        selected_id_set = set(reranker_raw_selected_ids)
        reranker_valid_selected_ids = [
            enum_id for enum_id in reranker_raw_selected_ids if enum_id in post_prune_id_set
        ]
        reranker_invalid_selected_ids = [
            enum_id for enum_id in reranker_raw_selected_ids if enum_id not in post_prune_id_set
        ]
        reranker_unselected_context_ids = [
            enum_id for enum_id in kb_context_enum_ids if enum_id not in selected_id_set
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
        _assessment_name(assessment): assessment
        for assessment in (info.get("assessments") or [])
        if isinstance(assessment, Mapping) and _assessment_name(assessment)
    }

    expected_response = (
        (assessments.get("expected_response") or {})
        .get("expectation", {})
        .get("value", "")
    )

    grading = assessments.get("answer_hg_chat_admin_gpt_grading") or {}
    expert_score = (grading.get("feedback") or {}).get("value")
    # When this scorer wasn't run, value is None. Use NaN so the column stays
    # float-typed in pandas — otherwise an all-None column lands as object dtype
    # and Spark infers NullType on createDataFrame, which gets dropped under
    # overwriteSchema=true and breaks any policy bound to the column.
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
    }
    record.update({key: value for key, value in reranker.items() if key != "reranker_raw_selected_ids"})
    record.update(agent_hashes)
    return record


def _extract_user_query(
    trace_metadata: Mapping[str, Any],
    spans: list[dict[str, Any]] | None = None,
) -> str:
    """Resolve the user-facing prompt across orchestrator vintages.

    Strategy (first-non-empty wins):

    1. **NEW (trace_schema v3, GCAI-3581+):** read the ``LangGraph`` span
       that is the immediate child of the root span and pull
       ``mlflow.spanInputs`` from its attributes. In these traces
       ``trace_metadata["mlflow.traceInputs"]`` is the empty string and
       the user query only survives at the span level.
    2. **OLD (legacy orchestrator):** fall back to
       ``trace_metadata["mlflow.traceInputs"]``, which carried the same
       ``{"messages": [["human", "<query>"]]}`` payload.

    The payload shape is identical in both cases, so the messages
    parser is shared.
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


def _user_query_payload_from_langgraph_root(
    spans: list[dict[str, Any]] | None,
) -> str:
    """Return ``mlflow.spanInputs`` from the root-child ``LangGraph`` span.

    Returns an empty string when ``spans`` is unavailable, when the trace
    has no root span (no span with ``parent_span_id is None``), or when
    no ``LangGraph`` child of the root carries ``mlflow.spanInputs``.
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
        attrs = span.get("attributes") or {}
        raw = attrs.get("mlflow.spanInputs", "")
        if raw:
            return raw
    return ""


def _to_str(value: Any) -> str:
    return "" if value is None else str(value)
