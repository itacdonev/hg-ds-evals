import hashlib
import json
from pathlib import Path

import pandas as pd

from hg_ds_evals.preprocessing.skkb_traces import build_skkb_dataframe_from_mlflow_search_traces
from hg_ds_evals.preprocessing.traces import (
    build_dataframe_from_mlflow_traces,
    collect_run_prompts,
    extract_agent_system_prompts,
    hash_tool_descriptions,
    write_prompt_sidecar,
)


_EMPTY_PROMPT_HASH = hashlib.md5(b"").hexdigest()[:10]


def _chat_tools_attr(tool_descriptions: dict[str, str]) -> dict[str, object]:
    """Encode a ChatDatabricks tool registry under the attribute name the parser reads."""
    return {
        "mlflow.chat.tools": json.dumps(
            [
                {"type": "function", "function": {"name": name, "description": desc}}
                for name, desc in tool_descriptions.items()
            ],
            ensure_ascii=False,
        )
    }


def _agent_chat_spans(
    *,
    agent_name: str,
    agent_span_id: str,
    system_prompt: str,
    tool_descriptions: dict[str, str] | None = None,
    chat_span_id: str | None = None,
) -> list[dict[str, object]]:
    """Build the agent → ChatDatabricks span pair the prompt extractor walks."""
    chat_id = chat_span_id or f"{agent_span_id}-chat"
    extra = _chat_tools_attr(tool_descriptions) if tool_descriptions else None
    return [
        _span(agent_name, span_id=agent_span_id, span_type="AGENT", node=agent_name),
        _span(
            "ChatDatabricks",
            span_id=chat_id,
            parent_span_id=agent_span_id,
            span_type="CHAT_MODEL",
            node=agent_name,
            inputs=[[{"role": "system", "content": system_prompt}]],
            extra_attrs=extra,
        ),
    ]


def _span(
    name: str,
    *,
    span_id: str,
    parent_span_id: str = "",
    node: str | None = None,
    span_type: str = "CHAIN",
    inputs: object | None = None,
    outputs: object | None = None,
    start_time: int = 1,
    extra_attrs: dict[str, object] | None = None,
) -> dict[str, object]:
    attrs: dict[str, object] = {
        "mlflow.spanType": json.dumps(span_type),
        "metadata": json.dumps({"langgraph_node": node or name}),
    }
    if inputs is not None:
        attrs["mlflow.spanInputs"] = json.dumps(inputs, ensure_ascii=False)
    if outputs is not None:
        attrs["mlflow.spanOutputs"] = json.dumps(outputs, ensure_ascii=False)
    if extra_attrs:
        attrs.update(extra_attrs)

    return {
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "name": name,
        "start_time_unix_nano": start_time,
        "attributes": attrs,
    }


def test_build_skkb_dataframe_uses_test_case_id_from_assessment() -> None:
    traces_df = pd.DataFrame(
        [
            {
                "trace_id": "tr-791e58a12c89479187acb78e3e2a9726",
                "trace_metadata": {},
                "tags": {},
                "assessments": [
                    {
                        "name": "test_case_id",
                        "value": "Test case 2334",
                        "expectation": {"value": "Test case 2334"},
                    },
                    {
                        "name": "expected_response",
                        "expectation": {"value": "Expected answer"},
                    },
                ],
                "spans": [],
            }
        ]
    )

    result = build_skkb_dataframe_from_mlflow_search_traces(traces_df)

    assert result.parse_errors == []
    assert result.unmapped_trace_ids == []
    assert result.dataframe.loc[0, "test_case_id"] == "Test case 2334"
    assert result.dataframe.loc[0, "expected_response"] == "Expected answer"


def test_build_skkb_dataframe_accepts_direct_expectation_value_for_expected_enums() -> None:
    traces_df = pd.DataFrame(
        [
            {
                "trace_id": "tr-expected-enums",
                "trace_metadata": {},
                "tags": {},
                "assessments": [
                    {
                        "name": "target_enums_to_relevance",
                        "expectation": {"value": {"WRITE_MESSAGE": 4}},
                        "value": {"WRITE_MESSAGE": 4},
                    },
                ],
                "spans": [],
            }
        ]
    )

    result = build_skkb_dataframe_from_mlflow_search_traces(traces_df)

    assert result.parse_errors == []
    assert result.dataframe.loc[0, "expected_enums"] == ["WRITE_MESSAGE"]
    assert result.dataframe.loc[0, "expected_enums_weights"] == '{"WRITE_MESSAGE": 4}'


def test_retrieved_candidates_fall_back_to_reranker_prompt() -> None:
    user_prompt = (
        "User question: can I write my advisor?\n"
        "Expected facets: contact options\n\n"
        "Candidates:\n"
        "- ID: SAVING@ACCOUNT_MOVEMENTS | Group: KB_GAI_SK_SK_2026-04-20 | Description: Savings text\n"
        "- ID: WRITE_MESSAGE | Group: KB_GAI_SK_SK_2026-04-20 | Description: Write advisor text\n"
        "- ID: CALL_BRANCH_AUTHORISED | Group: KB_GAI_SK_SK_2026-04-20 | Description: Call branch text"
    )
    spans = [
        _span(
            "daily_banking_agent",
            span_id="daily",
            node="daily_banking_agent",
            start_time=2,
        ),
        _span(
            "knowledge_search",
            span_id="tool",
            node="tools",
            span_type="TOOL",
            outputs={
                "content": (
                    "[KB_GAI_SK_SK_2026-04-20]\n"
                    "WRITE_MESSAGE: Write advisor text\n"
                    "CALL_BRANCH_AUTHORISED: Call branch text"
                )
            },
            start_time=3,
        ),
        _span(
            "RunnableSequence",
            span_id="rerank",
            node="rerank",
            start_time=4,
        ),
        _span(
            "ChatDatabricks",
            span_id="rerank-chat",
            parent_span_id="rerank",
            node="rerank",
            inputs=[
                [
                    {"role": "system", "content": "selector"},
                    {"role": "user", "content": user_prompt},
                ]
            ],
            outputs={
                "generations": [
                    [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "_RerankResponse",
                                            "arguments": json.dumps(
                                                {
                                                    "selected_ids": [
                                                        "WRITE_MESSAGE",
                                                        "CALL_BRANCH_AUTHORISED",
                                                    ]
                                                }
                                            ),
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                ]
            },
            start_time=5,
            extra_attrs={
                "mlflow.llm.model": json.dumps("gpt-5-1"),
                "mlflow.chat.tokenUsage": "{}",
            },
        ),
    ]
    traces_df = pd.DataFrame(
        [
            {
                "trace_id": "tr-direct",
                "trace_metadata": {
                    "mlflow.traceInputs": json.dumps({"messages": [["human", "Question?"]]})
                },
                "tags": {},
                "assessments": [
                    {
                        "name": "test_case_id",
                        "expectation": {"value": "Test case 512"},
                        "value": "Test case 512",
                    }
                ],
                "spans": spans,
            }
        ]
    )

    result = build_skkb_dataframe_from_mlflow_search_traces(traces_df)

    assert result.parse_errors == []
    row = result.dataframe.iloc[0]
    assert row["kb_version"] == "KB_GAI_SK_SK_2026-04-20"
    assert row["post_prune_enum_ids"] == [
        "SAVING@ACCOUNT_MOVEMENTS",
        "WRITE_MESSAGE",
        "CALL_BRANCH_AUTHORISED",
    ]
    assert row["pre_prune_enum_ids"] == []
    assert "WRITE_MESSAGE: Write advisor text" in row["post_prune_candidates_text"]
    assert json.loads(row["reranker_valid_selected_ids"]) == [
        "WRITE_MESSAGE",
        "CALL_BRANCH_AUTHORISED",
    ]
    assert json.loads(row["reranker_invalid_selected_ids"]) == []
    assert row["reranker_selection_status"] == "ok"


def test_retrieved_candidates_use_langgraph_node_metadata_when_span_name_differs() -> None:
    spans = [
        _span(
            "RunnableCallable",
            span_id="retrieve",
            node="retrieve",
            outputs={
                "retrieved_documents": [
                    {
                        "group_name": "KB_GAI_SK_SK_2026-04-20",
                        "fragment": {
                            "id": "WRITE_MESSAGE",
                            "description": "Write advisor text",
                        },
                    }
                ]
            },
        )
    ]
    traces_df = pd.DataFrame(
        [
            {
                "trace_id": "tr-retrieve",
                "trace_metadata": {},
                "tags": {},
                "assessments": [],
                "spans": spans,
            }
        ]
    )

    result = build_skkb_dataframe_from_mlflow_search_traces(traces_df)

    assert result.parse_errors == []
    assert result.dataframe.loc[0, "pre_prune_enum_ids"] == ["WRITE_MESSAGE"]
    assert result.dataframe.loc[0, "post_prune_enum_ids"] == ["WRITE_MESSAGE"]
    assert result.dataframe.loc[0, "kb_version"] == "KB_GAI_SK_SK_2026-04-20"


def test_pre_prune_candidates_are_preserved_separately_from_post_prune_pool() -> None:
    user_prompt = (
        "User question: can I write my advisor?\n"
        "Expected facets: contact options\n\n"
        "Candidates:\n"
        "- ID: WRITE_MESSAGE | Group: KB_GAI_SK_SK_2026-04-20 | Description: Write advisor text\n"
        "- ID: CALL_BRANCH_AUTHORISED | Group: KB_GAI_SK_SK_2026-04-20 | Description: Call branch text"
    )
    spans = [
        _span(
            "RunnableCallable",
            span_id="retrieve",
            node="retrieve",
            outputs={
                "retrieved_documents": [
                    {
                        "group_name": "KB_GAI_SK_SK_2026-04-20",
                        "fragment": {"id": "RAW_A", "description": "Raw A"},
                    },
                    {
                        "group_name": "KB_GAI_SK_SK_2026-04-20",
                        "fragment": {"id": "RAW_B", "description": "Raw B"},
                    },
                    {
                        "group_name": "KB_GAI_SK_SK_2026-04-20",
                        "fragment": {"id": "WRITE_MESSAGE", "description": "Write advisor text"},
                    },
                ]
            },
            start_time=1,
        ),
        _span("RunnableSequence", span_id="rerank", node="rerank", start_time=2),
        _span(
            "ChatDatabricks",
            span_id="rerank-chat",
            parent_span_id="rerank",
            node="rerank",
            inputs=[
                [
                    {"role": "system", "content": "selector"},
                    {"role": "user", "content": user_prompt},
                ]
            ],
            outputs={
                "generations": [
                    [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "_RerankResponse",
                                            "arguments": json.dumps(
                                                {"selected_ids": ["WRITE_MESSAGE"]}
                                            ),
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                ]
            },
            start_time=3,
        ),
    ]
    traces_df = pd.DataFrame(
        [
            {
                "trace_id": "tr-raw",
                "trace_metadata": {},
                "tags": {},
                "assessments": [],
                "spans": spans,
            }
        ]
    )

    result = build_skkb_dataframe_from_mlflow_search_traces(traces_df)

    assert result.parse_errors == []
    row = result.dataframe.iloc[0]
    assert row["pre_prune_enum_ids"] == ["RAW_A", "RAW_B", "WRITE_MESSAGE"]
    assert row["pre_prune_enum_count"] == 3
    assert row["post_prune_enum_ids"] == ["WRITE_MESSAGE", "CALL_BRANCH_AUTHORISED"]
    assert row["post_prune_enum_count"] == 2


def test_vector_http_pre_prune_and_post_prune_stages_are_separate() -> None:
    group_name = "KB_HG_CZ_CZ_2026-01-26_15h45"

    def raw_item(enum_id: str) -> dict[str, object]:
        return {
            "groupName": group_name,
            "score": 0.01,
            "fragment": {
                "id": enum_id,
                "summary": f"{enum_id} summary",
                "description": f"{enum_id} description",
            },
        }

    def merged_item(enum_id: str, query: str) -> dict[str, object]:
        return {
            "item": {
                "group_name": group_name,
                "fragment": {
                    "id": enum_id,
                    "description": f"{enum_id} description",
                },
                "score": 0.01,
            },
            "provenance": [{"query": query, "rank": 1, "raw_score": 0.01}],
        }

    spans = [
        _span(
            "HTTP POST /admin/knowledge-base/query",
            span_id="http-1",
            inputs={
                "url": "http://localhost:8080/admin/knowledge-base/query",
                "body_json": {"query": "q1", "limit": 2},
            },
            outputs={"body_json": {"items": [raw_item("RAW_A"), raw_item("RAW_B")]}},
            start_time=1,
        ),
        _span(
            "HTTP POST /admin/knowledge-base/query",
            span_id="http-2",
            inputs={
                "url": "http://localhost:8080/admin/knowledge-base/query",
                "body_json": {"query": "q2", "limit": 2},
            },
            outputs={"body_json": {"items": [raw_item("RAW_B"), raw_item("RAW_C")]}},
            start_time=2,
        ),
        _span(
            "retrieve",
            span_id="retrieve",
            node="retrieve",
            outputs={
                "retrieved_candidates": [
                    merged_item("RAW_A", "q1"),
                    merged_item("RAW_B", "q1"),
                    merged_item("RAW_C", "q2"),
                ]
            },
            start_time=3,
        ),
        _span(
            "prune",
            span_id="prune",
            node="prune",
            outputs={"retrieved_candidates": [merged_item("RAW_B", "q1"), merged_item("RAW_C", "q2")]},
            start_time=4,
        ),
        _span(
            "kb_prune",
            span_id="kb-prune",
            inputs={"candidates_in": 3},
            outputs={"candidates_out": 2, "candidates_dropped": 1},
            start_time=5,
        ),
    ]
    traces_df = pd.DataFrame(
        [
            {
                "trace_id": "tr-vector",
                "trace_metadata": {},
                "tags": {},
                "assessments": [],
                "spans": spans,
            }
        ]
    )

    result = build_skkb_dataframe_from_mlflow_search_traces(traces_df)

    assert result.parse_errors == []
    row = result.dataframe.iloc[0]
    assert row["raw_vector_db_query_count"] == 2
    assert row["raw_vector_db_retrieved_enum_count"] == 4
    assert row["raw_vector_db_retrieved_enum_ids"] == ["RAW_A", "RAW_B", "RAW_B", "RAW_C"]
    assert row["raw_vector_db_retrieved_count_by_query"] == {"q1": 2, "q2": 2}
    assert row["pre_prune_enum_ids"] == ["RAW_A", "RAW_B", "RAW_C"]
    assert row["pre_prune_enum_count"] == 3
    assert row["post_prune_enum_ids"] == ["RAW_B", "RAW_C"]
    assert row["post_prune_enum_count"] == 2
    assert row["prune_counts_available"]
    assert row["prune_candidates_in"] == 3
    assert row["prune_candidates_out"] == 2
    assert row["prune_candidates_dropped"] == 1


def test_multiple_knowledge_search_runs_keep_final_run_as_flat_columns() -> None:
    def raw_item(enum_id: str) -> dict[str, object]:
        return {
            "groupName": "KB_TEST",
            "fragment": {"id": enum_id, "description": f"{enum_id} raw"},
        }

    def candidate(enum_id: str) -> dict[str, object]:
        return {
            "item": {
                "group_name": "KB_TEST",
                "fragment": {"id": enum_id, "description": f"{enum_id} merged"},
            }
        }

    spans = [
        _span(
            "knowledge_search",
            span_id="ks-1",
            span_type="TOOL",
            outputs={"content": "[KB_TEST]\nOLD_FINAL: Old final"},
            start_time=1,
        ),
        _span(
            "HTTP POST /admin/knowledge-base/query",
            span_id="http-1",
            parent_span_id="ks-1",
            inputs={
                "url": "http://localhost:8080/admin/knowledge-base/query",
                "body_json": {"query": "old query", "limit": 1},
            },
            outputs={"body_json": {"items": [raw_item("OLD_RAW")]}},
            start_time=2,
        ),
        _span(
            "retrieve",
            span_id="retrieve-1",
            parent_span_id="ks-1",
            node="retrieve",
            outputs={"retrieved_candidates": [candidate("OLD_PRE")]},
            start_time=3,
        ),
        _span(
            "prune",
            span_id="prune-1",
            parent_span_id="ks-1",
            node="prune",
            outputs={"retrieved_candidates": [candidate("OLD_POST")]},
            start_time=4,
        ),
        _span(
            "knowledge_search",
            span_id="ks-2",
            span_type="TOOL",
            outputs={"content": "[KB_TEST]\nNEW_FINAL: New final"},
            start_time=5,
        ),
        _span(
            "HTTP POST /admin/knowledge-base/query",
            span_id="http-2",
            parent_span_id="ks-2",
            inputs={
                "url": "http://localhost:8080/admin/knowledge-base/query",
                "body_json": {"query": "new query", "limit": 1},
            },
            outputs={"body_json": {"items": [raw_item("NEW_RAW")]}},
            start_time=6,
        ),
        _span(
            "retrieve",
            span_id="retrieve-2",
            parent_span_id="ks-2",
            node="retrieve",
            outputs={"retrieved_candidates": [candidate("NEW_PRE")]},
            start_time=7,
        ),
        _span(
            "prune",
            span_id="prune-2",
            parent_span_id="ks-2",
            node="prune",
            outputs={"retrieved_candidates": [candidate("NEW_POST")]},
            start_time=8,
        ),
    ]
    traces_df = pd.DataFrame(
        [
            {
                "trace_id": "tr-multi-kb",
                "trace_metadata": {},
                "tags": {},
                "assessments": [],
                "spans": spans,
            }
        ]
    )

    result = build_skkb_dataframe_from_mlflow_search_traces(traces_df)

    assert result.parse_errors == []
    row = result.dataframe.iloc[0]
    assert row["knowledge_search_run_count"] == 2
    assert row["knowledge_search_final_run_index"] == 2
    assert row["reranked_enum_ids"] == ["NEW_FINAL"]
    assert row["raw_vector_db_retrieved_enum_ids"] == ["NEW_RAW"]
    assert row["pre_prune_enum_ids"] == ["NEW_PRE"]
    assert row["post_prune_enum_ids"] == ["NEW_POST"]

    runs = row["knowledge_search_runs"]
    assert runs[0]["raw_vector_db_retrieved_enum_ids"] == ["OLD_RAW"]
    assert runs[0]["post_prune_enum_ids"] == ["OLD_POST"]
    assert runs[1]["raw_vector_db_retrieved_enum_ids"] == ["NEW_RAW"]
    assert runs[1]["post_prune_enum_ids"] == ["NEW_POST"]


def test_agent_response_extracted_from_agent_answer_span() -> None:
    spans = [
        _span(
            "agent_answer",
            span_id="answer",
            span_type="AGENT",
            outputs={
                "question": "How do I write to my advisor?",
                "answer": "Use the WRITE_MESSAGE option in George.",
            },
        ),
    ]
    traces_df = pd.DataFrame(
        [
            {
                "trace_id": "tr-answer",
                "trace_metadata": {},
                "tags": {},
                "assessments": [],
                "spans": spans,
            }
        ]
    )

    result = build_skkb_dataframe_from_mlflow_search_traces(traces_df)

    assert result.parse_errors == []
    assert result.dataframe.loc[0, "agent_response"] == "Use the WRITE_MESSAGE option in George."


def test_agent_response_is_empty_when_agent_answer_span_absent() -> None:
    """Strict extraction: no heuristic fallback when the canonical span is missing."""
    spans = [
        _span(
            "ChatDatabricks",
            span_id="chat",
            outputs={
                "generations": [
                    [
                        {
                            "type": "ai",
                            "content": "Some long heuristic-bait reasoning blob.",
                            "response_metadata": {"finish_reason": "stop"},
                        }
                    ]
                ]
            },
        ),
    ]
    traces_df = pd.DataFrame(
        [
            {
                "trace_id": "tr-no-answer",
                "trace_metadata": {},
                "tags": {},
                "assessments": [],
                "spans": spans,
            }
        ]
    )

    result = build_skkb_dataframe_from_mlflow_search_traces(traces_df)

    assert result.parse_errors == []
    assert result.dataframe.loc[0, "agent_response"] == ""


def test_agent_response_is_empty_when_answer_field_blank() -> None:
    spans = [
        _span(
            "agent_answer",
            span_id="answer",
            span_type="AGENT",
            outputs={"question": "Hi?", "answer": "   "},
        ),
    ]
    traces_df = pd.DataFrame(
        [
            {
                "trace_id": "tr-blank-answer",
                "trace_metadata": {},
                "tags": {},
                "assessments": [],
                "spans": spans,
            }
        ]
    )

    result = build_skkb_dataframe_from_mlflow_search_traces(traces_df)

    assert result.parse_errors == []
    assert result.dataframe.loc[0, "agent_response"] == ""


def test_agent_response_ignores_agent_answer_span_with_wrong_type() -> None:
    """``agent_answer`` name without ``AGENT`` span type must not match."""
    spans = [
        _span(
            "agent_answer",
            span_id="answer",
            span_type="CHAIN",  # wrong type — strict extractor must skip
            outputs={"answer": "Should not be picked up."},
        ),
    ]
    traces_df = pd.DataFrame(
        [
            {
                "trace_id": "tr-wrong-type",
                "trace_metadata": {},
                "tags": {},
                "assessments": [],
                "spans": spans,
            }
        ]
    )

    result = build_skkb_dataframe_from_mlflow_search_traces(traces_df)

    assert result.parse_errors == []
    assert result.dataframe.loc[0, "agent_response"] == ""


# ── Prompt-extraction + tool-registry helpers ──────────────────────────


def test_hash_tool_descriptions_is_order_independent() -> None:
    a = {"alpha": "first", "beta": "second"}
    b = {"beta": "second", "alpha": "first"}
    assert hash_tool_descriptions(a) == hash_tool_descriptions(b)
    assert hash_tool_descriptions({}) == _EMPTY_PROMPT_HASH


def test_extract_agent_system_prompts_strips_timestamp_tail() -> None:
    body = "You are George.\nFollow the rules.\n"
    spans = _agent_chat_spans(
        agent_name="main_agent",
        agent_span_id="m1",
        system_prompt=body + "Current date and time: 2026-05-21 12:00:00",
    )
    children: dict[str, list[dict[str, object]]] = {}
    for span in spans:
        children.setdefault(span["parent_span_id"], []).append(span)

    prompts = extract_agent_system_prompts(spans, children)
    assert prompts["main_agent_system_prompt"] == body.rstrip()
    assert prompts["daily_banking_agent_system_prompt"] == ""


# ── parse_trace_skkb column surface ─────────────────────────────────────


def test_parse_trace_skkb_emits_prompt_and_tool_descriptor_columns() -> None:
    main_prompt = "You are the supervisor."
    dba_prompt = "You handle daily banking."
    tools = {"transfer-to-daily_banking_agent": "Transfer to DBA.", "knowledge_search": "Search KB."}
    spans = [
        *_agent_chat_spans(
            agent_name="main_agent",
            agent_span_id="m1",
            system_prompt=main_prompt,
            tool_descriptions=tools,
        ),
        *_agent_chat_spans(
            agent_name="daily_banking_agent",
            agent_span_id="d1",
            system_prompt=dba_prompt,
        ),
    ]
    traces_df = pd.DataFrame(
        [
            {
                "trace_id": "tr-skkb-1",
                "trace_metadata": {"mlflow.sourceRun": "run-abc-123"},
                "tags": {},
                "assessments": [],
                "spans": spans,
            }
        ]
    )

    result = build_skkb_dataframe_from_mlflow_search_traces(traces_df)
    row = result.dataframe.iloc[0]

    assert result.parse_errors == []
    assert row["main_agent_prompt_hash"] != _EMPTY_PROMPT_HASH
    assert row["daily_banking_agent_prompt_hash"] != _EMPTY_PROMPT_HASH
    assert row["tool_descriptions"] == tools
    assert row["tool_descriptions_hash"] == hash_tool_descriptions(tools)
    assert row["source_run_id"] == "run-abc-123"
    assert sorted(row["available_tools"]) == sorted(tools.keys())


def test_parse_trace_skkb_missing_prompts_hash_to_empty() -> None:
    traces_df = pd.DataFrame(
        [
            {
                "trace_id": "tr-skkb-empty",
                "trace_metadata": {},
                "tags": {},
                "assessments": [],
                "spans": [],
            }
        ]
    )
    result = build_skkb_dataframe_from_mlflow_search_traces(traces_df)
    row = result.dataframe.iloc[0]
    assert row["main_agent_prompt_hash"] == _EMPTY_PROMPT_HASH
    assert row["daily_banking_agent_prompt_hash"] == _EMPTY_PROMPT_HASH
    assert row["tool_descriptions_hash"] == _EMPTY_PROMPT_HASH
    assert row["source_run_id"] == ""


# ── parse_trace_mlflow column surface ───────────────────────────────────


def _mlflow_trace_row(
    *,
    trace_id: str,
    spans: list[dict[str, object]],
    source_run_id: str = "",
) -> dict[str, object]:
    """Build the JSONL-shape input that build_dataframe_from_mlflow_traces expects."""
    return {
        "info": {
            "trace_id": trace_id,
            "request_time": "0",
            "execution_duration_ms": 0,
            "state": "OK",
            "trace_metadata": {"mlflow.sourceRun": source_run_id} if source_run_id else {},
            "tags": {},
            "assessments": [],
        },
        "data": {"spans": spans},
    }


def test_parse_trace_mlflow_extracts_expected_enums_from_serialized_expectation() -> None:
    """Regression: previously, _extract_expected_enums_weights only knew
    the raw-assessment shape used by parse_trace_skkb. In parse_trace_mlflow
    ``_extract_assessments`` already decodes the expectation envelope, so the
    helper received the bare ``{enum_id: weight}`` mapping and fell through
    to ({}, {}). Every MLflow-parsed row was emitting expected_enums=[].
    """
    spans = _agent_chat_spans(
        agent_name="main_agent",
        agent_span_id="m1",
        system_prompt="prompt",
        tool_descriptions={},
    )
    row = _mlflow_trace_row(trace_id="tr-enums-1", spans=spans)
    row["info"]["assessments"] = [
        {
            "assessment_name": "target_enums_to_relevance",
            "last_update_time": "2026-05-22T13:04:51.613Z",
            "expectation": {
                "serialized_value": {
                    "serialization_format": "JSON_FORMAT",
                    "value": '{"CALL_PHONE_AUTHORISED_V2": 4, "FRAUD@ABOUT": 4, "LOCK_CARD_PERM": 4}',
                }
            },
        },
    ]
    traces_df = pd.DataFrame([row])

    result = build_dataframe_from_mlflow_traces(traces_df)
    out = result.dataframe.iloc[0]
    assert out["expected_enums"] == [
        "CALL_PHONE_AUTHORISED_V2",
        "FRAUD@ABOUT",
        "LOCK_CARD_PERM",
    ]
    assert json.loads(out["expected_enums_weights"]) == {
        "CALL_PHONE_AUTHORISED_V2": 4,
        "FRAUD@ABOUT": 4,
        "LOCK_CARD_PERM": 4,
    }


def test_parse_trace_mlflow_emits_prompt_and_tool_hashes() -> None:
    main_prompt = "You are the API supervisor."
    tools = {"george-gcg-product_getLoans": "Fetch loans.", "transfer-to-daily_banking_agent": "Transfer."}
    spans = _agent_chat_spans(
        agent_name="main_agent",
        agent_span_id="m1",
        system_prompt=main_prompt,
        tool_descriptions=tools,
    )
    traces_df = pd.DataFrame([_mlflow_trace_row(trace_id="tr-api-1", spans=spans, source_run_id="run-xyz")])

    result = build_dataframe_from_mlflow_traces(traces_df)
    row = result.dataframe.iloc[0]
    assert row["main_agent_prompt_hash"] != _EMPTY_PROMPT_HASH
    assert row["daily_banking_agent_prompt_hash"] == _EMPTY_PROMPT_HASH
    assert row["tool_descriptions_hash"] == hash_tool_descriptions(tools)
    assert row["source_run_id"] == "run-xyz"


# ── Sidecar writer ──────────────────────────────────────────────────────


def test_collect_run_prompts_picks_first_nonempty_and_unions_tools() -> None:
    # Trace 1: empty (no spans). Trace 2: main prompt + tool_a. Trace 3:
    # DBA prompt + a different description for tool_a + tool_b.
    # First-seen-wins on tool descriptions keeps the union deterministic;
    # the registry is stable within a real run anyway.
    spans_2 = _agent_chat_spans(
        agent_name="main_agent",
        agent_span_id="m1",
        system_prompt="Main prompt.",
        tool_descriptions={"tool_a": "Description A"},
    )
    spans_3 = _agent_chat_spans(
        agent_name="daily_banking_agent",
        agent_span_id="d1",
        system_prompt="DBA prompt.",
        tool_descriptions={"tool_a": "DIFFERENT", "tool_b": "Description B"},
    )
    traces_df = pd.DataFrame(
        [
            _mlflow_trace_row(trace_id="tr-1", spans=[]),
            _mlflow_trace_row(trace_id="tr-2", spans=spans_2),
            _mlflow_trace_row(trace_id="tr-3", spans=spans_3),
        ]
    )

    payload = collect_run_prompts(traces_df)
    assert payload["main_agent_system_prompt"] == "Main prompt."
    assert payload["daily_banking_agent_system_prompt"] == "DBA prompt."
    assert payload["tool_descriptions"] == {"tool_a": "Description A", "tool_b": "Description B"}


def test_write_prompt_sidecar_writes_named_json(tmp_path: Path) -> None:
    spans = _agent_chat_spans(
        agent_name="main_agent",
        agent_span_id="m1",
        system_prompt="Supervisor.",
        tool_descriptions={"tool_x": "X desc"},
    )
    traces_df = pd.DataFrame([_mlflow_trace_row(trace_id="tr-1", spans=spans)])

    out = write_prompt_sidecar(traces_df, "run-abcd1234", tmp_path)
    assert out == tmp_path / "prompt_run-abcd1234.json"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["run_id"] == "run-abcd1234"
    assert payload["main_agent_system_prompt"] == "Supervisor."
    assert payload["tool_descriptions"] == {"tool_x": "X desc"}
