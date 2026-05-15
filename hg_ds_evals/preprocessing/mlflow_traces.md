# `mlflow_traces.py` — column dictionary and extraction logic

> **Anchor doc.** Captures the parsing contract as of **2026-05-08**, branch
> `feat/GCAI-2566-evals-migration-pr12-cleanup`, sample run
> `traces_offline_smoke_pr12_kb_smoke_infer.jsonl` (100 traces).
>
> Purpose: when downstream eval tooling, scorer fields, or trace schema change,
> compare against this doc to know what the parser was originally extracting
> and why. Update this doc together with any change to `mlflow_traces.py`.

## 1. Scope

`mlflow_traces.py` parses MLflow traces produced by the ai-orchestrator's
`run_evals.py` harness into a wide pandas DataFrame, one row per trace, ready
to feed the deterministic scorers in `ai-data-science/evals`
(`ToolUsageScorer`, `RoutingCorrectnessScorer`, `ToolParameterScorer`).

It is **separate from** and does **not modify** `skkb_traces.py`. Why two
parsers right now:

- The SKKB nightly pipeline depends on the legacy column names and the KB
  pipeline columns (`reranked_kb_context`, `raw_vector_db_*`, `prune_*`,
  reranker columns, `expected_enums`). Freezing it avoids regression while we
  iterate on the API parser.
- The new evaluation contract introduced by `run_evals.py` adds eight HUMAN
  assessments per trace (`eval_item_id`, `eval_domain`, `eval_persona`,
  `expected_agent`, `expected_tool_calls`, `expected_response`, `guidelines`,
  `scorers`) — these are the source of truth for the new parser, and the
  legacy SKKB extraction never knew about them.

Once the API pipeline is stable and SKKB has been validated against
`mlflow_traces.py`, KB-specific extraction can be re-folded in and
`skkb_traces.py` deprecated.

## 2. Input

Either of:

1. **JSONL load shape** (one trace per line, the API notebook path):
   each row has columns `info` and `data`, matching the JSON written by
   `mlflow.search_traces().to_json(orient="records", lines=True)`.

2. **Live `mlflow.search_traces()` shape**: flat columns `trace_id`,
   `client_request_id`, `state`, `request_time`, `execution_duration_ms`,
   `trace_metadata`, `tags`, `assessments`, `spans`. The parser
   reassembles these into the canonical `info`/`data` shape via
   `_normalize_input_row`.

`build_dataframe_from_mlflow_traces(traces_df) → MlflowParseResult`.

## 3. Output

```python
@dataclass
class MlflowParseResult:
    dataframe: pd.DataFrame                          # one row per trace
    parse_errors: list[ParseError]                   # captured per-row failures
    untagged_trace_ids: list[str]                    # rows where test_case_id fell back to trace_id
    run_tool_registry: dict[str, dict[str, str]]     # cross-trace {node: {tool: description}}
```

The DataFrame columns are documented in §5. The `run_tool_registry` is the
union across all traces — useful for the run-config section of the report
where you want a single canonical "what tools did this run expose, with
descriptions".

## 4. Per-trace flow

```
parse_trace(trace)
├── _normalize_spans(data.spans)         # flatten span shape
├── _extract_assessments(info.assessments)  # decode HUMAN expectations
├── _resolve_test_case_id(...)           # eval_item_id → SKKB regex → trace_id
├── _extract_user_query(trace_metadata)
├── _extract_actual_agent_path(spans)    # CHAIN spans matching KNOWN_AGENT_NAMES
├── _extract_actual_tool_calls(spans)    # TOOL spans → dict-form list
├── _extract_actual_response(spans, trace_metadata)
├── _extract_tool_registry(spans)        # CHAT_MODEL.mlflow.chat.tools
└── _classify_architecture(actual_agents_path)
```

Every helper is local to `mlflow_traces.py` (intentionally duplicated
from `skkb_traces.py` rather than imported) so the new module has no
behavioural dependency on the legacy parser.

## 5. Column dictionary

All columns produced by the parser, grouped by what they describe.

### 5.1 Identity

| column | type | source | logic |
|---|---|---|---|
| `trace_id` | `str` | `info.trace_id` | unmodified |
| `test_case_id` | `str` | `eval_item_id` HUMAN assessment, fallback SKKB regex, fallback `trace_id` | see §6 |
| `session_id` | `str` | `trace_metadata["mlflow.trace.session"]` | run-scope UUID, **not stable across reruns**; do not use as test_case_id |
| `request_time` | `str` | `info.request_time` | ISO-8601 string |
| `execution_duration_ms` | `int \| None` | `info.execution_duration_ms` | unmodified |
| `state` | `str` | `info.state` | usually `"OK"` |

### 5.2 Eval expectations (HUMAN assessments)

All from `info.assessments` where `source.source_type == "HUMAN"`.
HUMAN assessments are written by `run_evals.py` once per row at eval time
and carry the test-case definition. Decoded via
`_decode_assessment_payload` (handles both `expectation.value` and
`expectation.serialized_value` shapes — see §7).

| column | type | assessment name | example |
|---|---|---|---|
| `user_query` | `str` | (not an assessment — from `mlflow.traceInputs`) | `"Show only loans I own."` |
| `eval_domain` | `str` | `eval_domain` | `"api"`, `"chit_chat"`, `"kb"` |
| `eval_persona` | `str` | `eval_persona` | `"karl_jan"`, `"ava"`, `"sophia"` |
| `expected_agent` | `str` | `expected_agent` | `"daily_banking_agent"` or `""` (chit_chat) |
| `expected_tool_calls` | `list[dict]` | `expected_tool_calls` | scorer-ready dict-form, see §8 |
| `expected_response` | `str` | `expected_response` | reference text, often empty |
| `guidelines` | `list[str]` | `guidelines` | LLM-judge inputs (future) |
| `scorers_to_run` | `list[str]` | `scorers` | e.g. `["agent_routing", "tool_usage"]` — drives per-row scorer selection |

### 5.3 Actuals (from spans)

| column | type | source | logic |
|---|---|---|---|
| `actual_agent` | `str` | last entry of `actual_agents_path` | empty string `""` if no sub-agent reached (chit-chat / refusal / supervisor-only) |
| `actual_agents_path` | `list[str]` | CHAIN spans whose `name ∈ KNOWN_AGENT_NAMES` | ordered de-duplicated; uses span `name` (not `langgraph_node`) so we get the actual agent label |
| `actual_tool_calls` | `list[dict]` | TOOL spans, in input order | scorer-ready dict-form; each dict has `tool`, `step`, `parameters`, `tool_call_id`, `outputs_*`, `error`, `error_message`, `agent`, `span_status_code`, `tool_message_status` — see §9 |
| `actual_response` | `str` | longest AI-typed `content` across all `mlflow.spanOutputs`; fallback to last AI message in `mlflow.traceOutputs.messages` | mirrors `skkb_traces.extract_final_ai_content` so behaviour is consistent |

### 5.4 Tool surface (per-trace LLM registry)

| column | type | source | logic |
|---|---|---|---|
| `available_tools` | `list[str]` | union of `mlflow.chat.tools[].function.name` across all CHAT_MODEL spans | sorted, unique. Pass to `ToolUsageScorer(available_tools=...)` to enable misuse-vs-hallucination classification |
| `tool_descriptions` | `dict[str,str]` | `function.description` from same spans | first-seen wins (descriptions are stable within a run) |
| `tool_registry_by_node` | `dict[str, list[str]]` | grouped by `metadata.langgraph_node` | shows which agent's LLM had which tools (e.g. `main_agent → ["Router"]`, `llm → [<MCP API surface>]`, `rerank → ["_RerankResponse"]`) |

### 5.5 Provenance / cost

| column | type | source |
|---|---|---|
| `architecture` | `str` | derived: `"classic"` if `main_agent ∈ path`, `"hg-invest"` if hg-invest in path, else `"none"` |
| `model` | `str` | first CHAT_MODEL span's `mlflow.llm.model` (or `invocation_params.model`) |
| `token_usage` | `dict` | `trace_metadata["mlflow.trace.tokenUsage"]` (JSON-decoded) |
| `git_branch` | `str` | `trace_metadata["mlflow.source.git.branch"]` |
| `git_commit` | `str` | `trace_metadata["mlflow.source.git.commit"]` |
| `source_run_id` | `str` | `trace_metadata["mlflow.sourceRun"]` |
| `mlflow_user` | `str` | `trace_metadata["mlflow.user"]` |

### 5.6 Raw passthroughs (kept for iteration 2)

| column | type | purpose |
|---|---|---|
| `tags` | `dict` | full `info.tags` |
| `assessments_raw` | `list[dict]` | full assessment list, plain-data form, for re-decoding |

## 6. `test_case_id` resolution

Three-tier fallback inside `_resolve_test_case_id`:

1. **`eval_item_id` HUMAN assessment** — preferred. The orchestrator's
   `run_evals.py` writes one per row with the source dataset's row id
   (e.g. `"smoke-100"`). 100/100 unique on the current sample run.
2. **SKKB legacy regex** `\btest case \d+\b` over `user_query`. Only
   matches SKKB-style test prompts ("Test case 17 — …"). Normalised to
   `"test_case_17"`.
3. **`trace_id`** — last resort. Whenever this happens the trace is also
   appended to `MlflowParseResult.untagged_trace_ids` so callers can audit
   coverage. On the current sample run this is empty.

`session_id` (`mlflow.trace.session`) is **not** used. It is freshly
generated each run, so it cannot identify a stable test case across reruns.

## 7. Assessment decoding

MLflow stores HUMAN expectation values in two shapes depending on type:

| Python type | shape on the wire |
|---|---|
| `str`, `int`, `bool`, `None`, `dict` (plain) | `{"expectation": {"value": <python>}}` |
| `list`, `dict` (richer), anything serialized | `{"expectation": {"serialized_value": {"value": "<json string>", "serialization_format": "JSON"}}}` |

`_decode_assessment_payload` handles both, returning the Python value.
On JSON decode failure it returns the raw string rather than raising —
the eval team has changed the payload shape between runs before, so the
parser stays permissive.

CODE assessments (added by post-hoc scorers) use the same shape but live
under `feedback` instead of `expectation`. The dedup-by-`last_update_time`
logic in `_extract_assessments` keeps the most recent one per name —
relevant if a trace has been re-graded, otherwise a no-op.

## 8. `expected_tool_calls` dict shape

Comes through unmodified from the assessment (the orchestrator already
emits the canonical scorer dict-form):

```json
[
  {
    "step": 1,
    "tool": "george-gcg-product_getLoans",
    "parameters": {"ownedByCurrentUser": true},
    "reason": "Read loans and filter to those owned by the current user."
  }
]
```

- `step` is 1-based positional (used by `ToolUsageScorer` mode `"sequence"`).
- `tool` is the **sanitized** tool name (dots replaced by underscores by
  `ToolSpecification.sanitized_name` upstream — matches the TOOL span
  `name` byte-for-byte).
- `parameters` is the args dict the LLM should pass — what
  `ToolParameterScorer` compares against `actual_tool_calls[i].parameters`.
- `reason` is a free-text note for human reviewers; ignored by scorers.

For `eval_domain == "chit_chat"`, this is `[]` and `expected_agent` is
`""`. Per the user's confirmation: any tool call or sub-agent routing in
that case is a penalty (the supervisor should refuse / answer directly).

## 9. `actual_tool_calls` dict shape

Per TOOL span, in the order the spans appear:

```python
{
  "tool": "george-gcg-product_getLoans",        # span.name
  "step": 1,                                     # 1-based positional
  "arguments":  {"ownedByCurrentUser": True},    # canonical actual-side key (ToolParameterScorer reads this)
  "parameters": {"ownedByCurrentUser": True},    # alias of arguments — same value
  "tool_call_id": "call_m6saDGXZRNAA0OrJMWHiPnvP", # attributes.tool_call_id (unquoted)
  "outputs_preview": "{\"foundLoanCount\": 1, ...",  # first 500 chars of ToolMessage.content
  "outputs_text":    "{\"foundLoanCount\": 1, ... full payload ...",
  "outputs_obj":     {"content": "...", "status": "success", ...},  # full ToolMessage dict
  "error": False,
  "error_message": "",
  "agent": "tools",                              # owning langgraph_node (best-effort)
  "span_status_code": "STATUS_CODE_OK",          # OTel span status
  "tool_message_status": "success",              # LangChain ToolMessage status
}
```

### 9.0 Why both `arguments` and `parameters`?

`ToolParameterScorer` reads the **expected** side under `parameters`
(``expected_tool_calls[i]["parameters"]``) and the **actual** side under
`arguments` (``actual_tool_calls[i]["arguments"]``). The asymmetry is
intentional in the scorer's contract:

> "expected parameters" → what the test case asserts
> "actual arguments"    → what the LLM actually passed

If the actual side only carried `parameters`, the scorer would see
``arguments=None`` on every call → treat every call as a no-arg call →
score 0.0 even when the LLM passed the right args. We emit both keys
with the same value: the scorer reads `arguments`, and callers that
expect symmetry with the reference side (``row["actual_tool_calls"][i]["parameters"]``)
still get the data.

### 9.1 Status — two layers, combined into `error: bool`

| layer | location | values seen (this run) |
|---|---|---|
| OTel span status | `span.status.code` | `STATUS_CODE_OK` × 142, `STATUS_CODE_ERROR` (would set `error=True`), `STATUS_CODE_UNSET` (treated as OK) |
| ToolMessage status | `mlflow.spanOutputs.status` | `"success"` × 142, `"error"` (would set `error=True`) |

`error` is `True` iff *either* layer reports a failure. `error_message`
is filled from `span.status.message` when present, otherwise from the
ToolMessage `content` when there's an error. For the
`ignore_failed_calls=True` flag on `ToolUsageScorer`, this is the field
the scorer reads.

### 9.2 Output payload — full vs preview

Per request, both are kept right now so the report can show the full
payload without re-parsing the trace:

- `outputs_preview` — first 500 chars of the ToolMessage `content` string
  (truncated with `…`). Cheap to display in tables.
- `outputs_text` — full `content` string. Sometimes 10s of KB (banking
  list responses).
- `outputs_obj` — full ToolMessage dict (`content`, `status`,
  `tool_call_id`, `name`, `type`, `id`, `additional_kwargs`, `artifact`,
  `response_metadata`).

In iteration 2 we can drop `outputs_text`/`outputs_obj` from the
DataFrame and store them in a sidecar table keyed by `tool_call_id` to
keep the main DataFrame slim — for now we keep everything inline so you
have one place to look.

### 9.3 `agent` field

Currently filled with the inner `langgraph_node` of the TOOL span,
which in LangGraph is always `"tools"` (the ToolNode running on behalf
of whichever agent owns the graph). The actually-owning agent
(`daily_banking_agent`) can be derived in iteration 2 by walking up the
parent chain or by correlating against `actual_agents_path`. Kept as a
diagnostic for now.

## 10. Tool registry extraction

`_extract_tool_registry` reads `mlflow.chat.tools` from every CHAT_MODEL
span and groups by `metadata.langgraph_node`:

```python
{
  "main_agent": {"Router": "<empty desc>"},
  "llm": {
    "george-gcg-product_getAccounts": "Summary:\nList bank accounts ...",
    "george-gcg-product_getCards":    "Summary:\nList bank cards ...",
    "george-gcg-product_getCardLimits": "Summary:\nRetrieve current spending ...",
    "george-gcg-product_getProducts": "Summary:\nList banking products ...",
    "george-gcg-product_getLoans":    "Summary:\nList loans ...",
    "get_transactions":               "Fetch transactions from upstream ...",
    "create_transaction_analysis":    "Create a transaction analysis ...",
    "knowledge_search":               "Search the knowledge base ..."
  },
  "rerank": {"_RerankResponse": "Structured LLM output schema for the reranking step."}
}
```

Stability check: across all 100 traces in the sample run, every node
sees exactly **one** tool-set signature, and every tool sees exactly
**one** description. So the registry is run-stable, and the per-trace
column is the same as the run-aggregate. We still emit it per trace
because that does **not** hold across runs (registry can change between
branches/commits) and we want each row to be self-describing.

The same data is exposed two ways:

- **Per trace**: `available_tools` (flat sorted list of names) and
  `tool_descriptions` (name→desc) and `tool_registry_by_node`.
- **Per run**: `MlflowParseResult.run_tool_registry` (union across all
  traces, same shape as `tool_registry_by_node`).

The flat per-trace `available_tools` is the input for the scorer's
`available_tools` arg — it lets `ToolUsageScorer` distinguish *misused*
tools (call to a tool that exists on the registry but wasn't expected)
from *hallucinated* tools (call to a tool that doesn't exist at all).

## 11. Architecture classification

`_classify_architecture(actual_agents_path) → str`:

- `"classic"` — `main_agent ∈ path`. The current default for SK and
  pr12-kb-smoke runs.
- `"hg-invest"` — `hg-invest-phase2 ∈ path`. Investment domain (no
  traces yet on this run).
- `"none"` — neither known agent reached. Chit-chat refusals or other
  supervisor-handled flows.

Coarse on purpose. If/when the supervisor split lands
(`gate/ethics/language/route/decide/synthesis`) we add a case here
without changing the column.

## 12. Edge cases / failure modes

| input | parser behaviour |
|---|---|
| Trace with 0 spans | `actual_agent=""`, `actual_agents_path=[]`, `actual_tool_calls=[]`, `available_tools=[]`, `actual_response=""` |
| Trace with no `eval_item_id` assessment but matching SKKB regex | `test_case_id="test_case_N"` and *not* added to `untagged_trace_ids` |
| Trace with no `eval_item_id` assessment and no regex match | `test_case_id=trace_id`, *added* to `untagged_trace_ids` |
| Assessment value JSON-decode fails | helper returns the raw string, parser proceeds |
| Tool span with empty `mlflow.spanInputs` | `parameters={}`, error remains `False` |
| Tool span with `STATUS_CODE_ERROR` and ToolMessage `status="error"` | `error=True`, `error_message` populated |
| Per-row exception | captured into `MlflowParseResult.parse_errors`; row dropped from output DataFrame, batch continues |

## 13. Verified against (this snapshot)

```
file: experiments/api/input/traces_offline_smoke_pr12_kb_smoke_infer.jsonl
date: 2026-05-08
size: 100 traces, 57 MB
branch: feat/GCAI-2566-evals-migration-pr12-cleanup
commit: 2bce9eabd3fdeb1a7f7c157261a172e681cf2b96
source: ai-orchestrator/apps/orchestrator/src/orchestrator/infrastructure/cli/run_evals.py
```

Result: `rows=100`, `parse_errors=0`, `untagged_trace_ids=0`. All 100
test cases have unique `eval_item_id` (`smoke-001` … `smoke-100`).
Tool-call distribution: 47×1-call, 40×2-call, 5×3-call, 8×0-call (chit-chat).
All 142 TOOL spans `STATUS_CODE_OK` and ToolMessage `status="success"`.

## 14. Iteration 2 candidates

Tracked here so they don't get lost:

- Drop `outputs_text` and `outputs_obj` from the DataFrame, move to a
  sidecar table keyed by `tool_call_id`. Keeps the main DataFrame
  serializable to CSV without surprises.
- Resolve `actual_tool_calls[].agent` to the *owning* agent
  (`daily_banking_agent`) instead of the ToolNode (`tools`).
- Fold KB-pipeline extraction back in (reranker, vector-DB hits, prune
  counts) once the colleague's branch lands and the
  `/admin/knowledge-base/query` HTTP spans are restored. At that point
  we can deprecate `skkb_traces.py`.
- Add a structured-equality helper for `ToolParameterScorer` argument
  comparison (numeric tolerance, key normalisation).
- Cross-check `expected_tool_calls[i].step` against `actual_tool_calls[i].step`
  and emit a `step_alignment` column for the report.
- Surface the `tool_call_id` linkage from the upstream CHAT_MODEL
  `tool_calls` block to the TOOL span (currently we only read it on the
  TOOL span).
