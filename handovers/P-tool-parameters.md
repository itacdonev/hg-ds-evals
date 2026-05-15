# Handover — `ToolParameterScorer` (GCAI-782) + `ToolUsageScorer` rework

> **Purpose of this doc:** everything I (the assistant in a future session) need to pick up this work cold. Read top-to-bottom before writing any code. The implementation is functionally complete; the user is in **review mode**.

---

## 1. Where things are

**Branch:** `feat/GCAI-782-tool-parameters` (in `ai-data-science`).

**Working tree state at handover (uncommitted):**

```
M  evals/docs/scorers/deterministic/tool_parameter.md
M  evals/docs/scorers/deterministic/tool_usage.md
M  evals/nbs/tool_parameter_demo.ipynb
MM evals/nbs/tool_usage_demo.ipynb
M  evals/src/ai_data_science/evals/config.py
AM evals/src/ai_data_science/evals/scorers/deterministic/_tool_inputs.py
M  evals/src/ai_data_science/evals/scorers/deterministic/tool_parameter.py
M  evals/src/ai_data_science/evals/scorers/deterministic/tool_usage.py
M  evals/tests/test_scorers/fixtures/tool_parameter_cases.json
M  evals/tests/test_scorers/fixtures/tool_usage_cases.json
M  evals/tests/test_scorers/test_tool_parameter.py
M  evals/tests/test_scorers/test_tool_usage.py
```

**Already committed on this branch (local, not pushed):**

```
b6415a0 feat: GCAI-3582 Add ignore_failed_calls to ToolUsageScorer
ecf951d feat: GCAI-782 tool_parameter: retry-recovery docs & tests
8d9bf2b feat: GCAI-782 Add unit test, docs and demo notebook
278fc09 feat: GCAI-782 Add scoring logic
4ce02c9 feat: GCAI-782 Add ToolParameterScorer and register field
```

> The uncommitted changes correspond to **two large refactors landed after `b6415a0`:**
> 1. **Field consolidation** — collapsed 4 dataset fields (`expected_tools`, `expected_tool_sequence`, `tools_called`, `actual_tool_sequence`) down to **2 polymorphic fields** (`expected_tool_calls`, `actual_tool_calls`).
> 2. **Status taxonomy** — every data-related condition now produces a `ScorerResult` with `metadata["status"]`. Data-side raises are gone; only programmer/config errors still raise.
>
> **Both refactors are NOT yet committed.** The user said they would review one more time end-to-end before commit. **Do not commit without their explicit go-ahead.**

**Test status:** `648 passing, 0 failing` (last run). All 648 tests pass with PYTHONPATH set as below.

---

## 2. Project layout

Three repositories matter:

| Repo | Path | What's in it |
|---|---|---|
| **ai-data-science** | `/Users/SG7CB/Developer/ai-data-science/` | Where the scorers live (`evals/` subdirectory). This is the active branch. |
| **ai-orchestrator** | `/Users/SG7CB/Developer/ai-orchestrator/` | Reference for how tools are defined/called in production. We **read** from it; we do **not** write to it on this branch. |
| **hg-ds-evals** | `/Users/SG7CB/Developer/hg-ds-evals/` | Holds the smoke-100 trace data we used to verify shapes. Also where this handover lives. |
| **personal** | `/Users/SG7CB/Developer/personal/` | The user's notes — plan doc, tasks file, related references. |

### Key files (all paths absolute)

**The scorers (the actual implementation):**
- `/Users/SG7CB/Developer/ai-data-science/evals/src/ai_data_science/evals/scorers/deterministic/_tool_inputs.py` (149 lines) — shared normalizer + `DataError` + `STATUS_*` constants
- `/Users/SG7CB/Developer/ai-data-science/evals/src/ai_data_science/evals/scorers/deterministic/tool_parameter.py` (398 lines) — the new scorer
- `/Users/SG7CB/Developer/ai-data-science/evals/src/ai_data_science/evals/scorers/deterministic/tool_usage.py` (682 lines) — rewritten (was ~1149 before)
- `/Users/SG7CB/Developer/ai-data-science/evals/src/ai_data_science/evals/config.py` — `EvalFields` (added `EXPECTED_TOOL_CALLS`, `ACTUAL_TOOL_CALLS`, `TOOL_PARAMETER`; removed the four old field constants)

**Tests:**
- `/Users/SG7CB/Developer/ai-data-science/evals/tests/test_scorers/test_tool_parameter.py` (539 lines)
- `/Users/SG7CB/Developer/ai-data-science/evals/tests/test_scorers/test_tool_usage.py` (1072 lines)
- `/Users/SG7CB/Developer/ai-data-science/evals/tests/test_scorers/fixtures/tool_parameter_cases.json` (~18 cases)
- `/Users/SG7CB/Developer/ai-data-science/evals/tests/test_scorers/fixtures/tool_usage_cases.json` (~27 cases)

**Docs:**
- `/Users/SG7CB/Developer/ai-data-science/evals/docs/scorers/deterministic/tool_parameter.md`
- `/Users/SG7CB/Developer/ai-data-science/evals/docs/scorers/deterministic/tool_usage.md`

**Notebooks:**
- `/Users/SG7CB/Developer/ai-data-science/evals/nbs/tool_parameter_demo.ipynb` (28 cells, all executed cleanly with cached outputs)
- `/Users/SG7CB/Developer/ai-data-science/evals/nbs/tool_usage_demo.ipynb` (48 cells, executes cleanly)

**Planning + reference docs (outside the repo):**
- `/Users/SG7CB/Developer/personal/docs/jira-tasks/GCAI-782_tool_parameters_scorer_plan.md` — the live plan / spec doc. Section structure: §1 trace data shape, §2 dataset format, §3 scoring model, §4 matching algo, §5 extras + retry-recovery, §6 validator, §7 aggregation/reporting, §8 scorer interface, §9 files to add, §10 test cases, §11 deliverables, §12 roadmap, §13 locked decisions, §14 JIRA copy/paste. Has been kept in sync with code throughout.
- `/Users/SG7CB/Developer/personal/tasks.md` — follow-up task tracking. Sections: "ToolUsageScorer follow-ups" (5 items, including the runtime-parallelism v2 spec) and "SKKB report follow-ups" (unrelated — different effort).
- `/Users/SG7CB/Developer/personal/evals/Formatters, Schemas, and Tool Names — A Plain-English Tour.pdf` — colleague's MockMeister explainer (relevant for understanding tool aliasing, but mostly orthogonal to this scorer work).
- `/Users/SG7CB/Developer/personal/docs/Ky_tool_args.html` — Kyrylo's draft `tool_input_parameters` scorer (orchestrator-side reference implementation; we cherry-picked ideas from it but did not copy).

**Real trace data (used to verify shapes during the work):**
- `/Users/SG7CB/Developer/hg-ds-evals/experiments/api/input/traces_offline_smoke_pr12_kb_smoke_infer.csv` (185MB) — 100 trace rows from a smoke run. Format: CSV with columns `trace_id, trace, ..., spans, assessments`. The `spans` column is Python-`repr` (not JSON), so use `ast.literal_eval` to parse. TOOL spans carry args under `attributes["mlflow.spanInputs"]` as a **JSON string** (parse it). Same-tool-multi-call rate: 1/100. Confirmed: tool args are flat dicts, no nesting.
- `/Users/SG7CB/Downloads/smoke_100_tools.json` — the dataset definition that produced those traces. Currently only has `input` + `expected_tools` (flat list of names). The user will migrate this to the new `expected_tool_calls` / `actual_tool_calls` shape going forward.

---

## 3. The locked design (summary)

### 3.1 Two scorers

**`ToolUsageScorer`** — checks **which** tools were called.
- Scale: `BinaryScale`.
- Modes: `exact`, `subset`, `superset`, `intersection`, `sequence`. Multi-mode + `"all"` supported.
- Constructor: `mode`, `order_sensitive`, `case_sensitive`, `available_tools`, `ignore_failed_calls`.

**`ToolParameterScorer`** — checks **what arguments** were passed.
- Scale: `NumericScale([0.0, 1.0])` (partial credit).
- No modes; one fixed scoring algorithm.
- Constructor: `case_sensitive`, `value_coercion` (`"string"` | `"strict"`).

### 3.2 Two unified fields (after consolidation)

Both scorers read the **same two fields**, both polymorphic:

| Field | Accepts | What's in dict-form |
|---|---|---|
| `expected_tool_calls` | `list[str]` (flat) **or** `list[dict]` (rich) | `{tool, parameters?, depends_on?, step?, reason?}` |
| `actual_tool_calls` | `list[str]` (flat) **or** `list[dict]` (rich) | `{tool, arguments?, step?, error?}` |

- Mixing strings and dicts in one list → soft fail with `status="testcase_invalid_expected"` (or `"agent_invalid_actual"`).
- For `tool_parameter`, expected **must** be dict-form (needs `parameters` per entry); actual can be either.
- For `tool_usage`, both forms work for any mode.

### 3.3 Status taxonomy

Every `ScorerResult` carries `metadata["status"]`. The 8 possible values:

| `status` | Trigger | Score | Counts toward agent metric? |
|---|---|---|---|
| `"ok"` | Scored normally | normal | ✅ |
| `"agent_no_calls"` | Agent didn't invoke any tools (or all filtered by `ignore_failed_calls`) | `scale.min` | ✅ |
| `"agent_invalid_actual"` | `actual_tool_calls` malformed (non-list, mixed forms, malformed entry, bad `arguments`/`step`/`error` types) | `scale.min` | ✅ |
| `"testcase_missing_reference"` | `expected_tool_calls` is `None` or missing | `scale.min` | ❌ |
| `"testcase_invalid_expected"` | `expected_tool_calls` malformed (non-list, mixed forms, blank tool, etc.) | `scale.min` | ❌ |
| `"testcase_invalid_parameters"` | (tool_parameter only) `parameters` malformed or missing on an entry | `scale.min` | ❌ |
| `"testcase_no_scorable_entries"` | (tool_parameter only) all entries `parameters: {}` or list empty | `scale.min` | ❌ |
| `"testcase_bad_dependency"` | (tool_usage sequence mode) self / circular / dangling / non-string `depends_on` | `scale.min` | ❌ |

**Aggregation rule:** `status == "ok"` or starts with `"agent_"` → counts toward agent score. Starts with `"testcase_"` → broken test case, exclude from agent metrics, count separately as a "test-set defect."

### 3.4 What still raises (programmer/config errors only)

- Wrong dimension scale → `TypeError` (e.g. `BinaryScale` passed to `ToolParameterScorer`).
- Bad `mode` constructor value → `ValueError` / `TypeError`.
- Bad `value_coercion` constructor value → `ValueError`.
- Bad `available_tools` shape (bare string, empty, non-string elements) → `TypeError` / `ValueError`.

These halt the run because they affect every row identically; the right diagnostic is a stack trace, not a metadata field.

### 3.5 Other locked decisions

- **Field name parallelism:** `expected_tool_calls` / `actual_tool_calls` (parallel naming, decided over `expected_tools` / `actual_tools` for clarity).
- **Hard cut, no aliases:** old field names are gone. No deprecation period. The user confirmed nothing in production uses them yet.
- **Best-match-without-replacement** for tool_parameter's matching algorithm. Documented in plan §4 with a worked example using `get_account_detail` per account.
- **Partial credit, two scores:** `value_score` (correct values / expected keys) is the headline. `key_score` (matched keys / expected keys) is in metadata. The gap (`key_score - value_score`) reveals the "right key, wrong value" rate.
- **Rationale format:** `"<correct>/<total> values correct, <matched>/<total> keys present; <N> wrong value(s), <M> missing key(s); <X> extra invocation(s)."`
- **`step` is optional** on both expected and actual entries. On actual, when present on every entry, used to sort the flat list; otherwise list order is used.
- **`parameters: {}` is the explicit per-entry skip** for tool_parameter. Missing `parameters` key on an opted-in item produces `status="testcase_invalid_parameters"`.
- **String coercion for values:** default `value_coercion="string"` does `str(actual).casefold() == str(expected).casefold()`. Handles `100` ↔ `"100"`, `True` ↔ `"true"`, `"OUTGOING"` ↔ `"outgoing"`. Use `"strict"` for exact-case enums.
- **`ignore_failed_calls` is opt-in (default `False`).** When `True`, `actual_tool_calls` dict-form entries with `error: True` are filtered before counting. Lets `tool_usage` exact mode tolerate retry-after-error scenarios. Has no effect on flat-form actual (no per-entry error info).
- **`depends_on` is single-parent only** today (`str`, not `list[str]`). Multi-parent would need DAG cycle detection — see follow-up task §3 in `personal/tasks.md`.
- **Per-item opt-in via `scorers` field** — `tool_parameter` only runs on items whose `scorers` list includes `"tool_parameter"`. Honest aggregation: `mean(value_score)` is over rows where the scorer ran, not all dataset rows.

---

## 4. Code architecture

```
evals/src/ai_data_science/evals/scorers/deterministic/
├── _tool_inputs.py       ← shared normalizer + DataError + STATUS_* constants
├── tool_usage.py         ← consumes _tool_inputs
├── tool_parameter.py     ← consumes _tool_inputs
└── ...
```

### `_tool_inputs.py` exports

```python
# Status constants (eight values — see §3.3 above)
STATUS_OK
STATUS_AGENT_NO_CALLS
STATUS_AGENT_INVALID_ACTUAL
STATUS_TESTCASE_MISSING_REFERENCE
STATUS_TESTCASE_INVALID_EXPECTED
STATUS_TESTCASE_INVALID_PARAMETERS
STATUS_TESTCASE_NO_SCORABLE_ENTRIES
STATUS_TESTCASE_BAD_DEPENDENCY

# Internal exception
class DataError(ValueError):
    """Carries a `status` attribute. Subclass of ValueError so legacy
    catch-all handlers still work, but scorers catch DataError specifically."""

# Public function
def normalize_tool_list(raw, *, field_name, case_sensitive) -> list[dict]:
    """Detect flat-form vs dict-form, validate base shape, return canonical
    list-of-dicts. Each returned entry has at least a 'tool' key (normalized).
    Raises ValueError on shape problems (callers catch and tag with status)."""
```

### Score-time flow (both scorers, identical structure)

```python
def score(self, dimension, **kwargs):
    if not isinstance(dimension.scale, ExpectedScale):
        raise TypeError(...)                                  # programmer error → halt

    try: missing = self.validate_inputs(**kwargs)
    except ValueError as e:
        return self._data_error(dim, STATUS_TESTCASE_MISSING_REFERENCE, str(e))

    try: expected_canonical = normalize_tool_list(
            kwargs[EvalFields.EXPECTED_TOOL_CALLS], ...)
    except ValueError as e:
        return self._data_error(dim, STATUS_TESTCASE_INVALID_EXPECTED, str(e))

    # scorer-specific: extract scorable entries / validate sequence integrity
    try: scorable = self._extract_scorable(expected_canonical)         # tool_parameter
    # OR
    try: self._validate_sequence_integrity(expected_canonical)         # tool_usage (sequence mode)
    except DataError as e:
        return self._data_error(dim, e.status, str(e))

    if missing:
        return self._data_error(dim, STATUS_AGENT_NO_CALLS, "...")

    try: actual_canonical = normalize_tool_list(
            kwargs[EvalFields.ACTUAL_TOOL_CALLS], ...)
    except ValueError as e:
        return self._data_error(dim, STATUS_AGENT_INVALID_ACTUAL, str(e))

    # ... extraction, scoring, build_result with status="ok" ...
```

### `tool_parameter.py` — internal methods

- `_extract_scorable(canonical) -> list[dict]` — validates per-entry `parameters`, raises `DataError(STATUS_TESTCASE_INVALID_PARAMETERS, ...)` or `DataError(STATUS_TESTCASE_NO_SCORABLE_ENTRIES, ...)`.
- `_extract_actual(canonical) -> list[dict]` — validates per-entry `arguments` (must be dict if present), raises `DataError(STATUS_AGENT_INVALID_ACTUAL, ...)`.
- `_assign(scorable, actual) -> (per_entry, extras)` — best-match-without-replacement matching, scoped per tool name.
- `_compare_one(expected, actual) -> (matched, wrong, missing)` — per-key comparison with three states.
- `_values_match(actual, expected) -> bool` — equality first, then casefold-string-compare under string coercion.
- `_build_result(...)` — assembles totals, computes `key_score`/`value_score`, maps to scale, builds rationale, returns ScorerResult with `status="ok"`.
- `_data_error(dim, status, rationale)` — soft-fail builder.

### `tool_usage.py` — internal methods

- `_extract_actual_names(canonical) -> (list[str], int)` — flat-list extraction with optional `step`-sort and `error`-filtering. Raises `DataError(STATUS_AGENT_INVALID_ACTUAL, ...)` on bad `step`/`error` types.
- `_validate_sequence_integrity(canonical)` — raises `DataError(STATUS_TESTCASE_BAD_DEPENDENCY, ...)` for self/circular/dangling/non-string `depends_on`.
- `_score_set_mode(...)` / `_score_sequence(...)` — per-mode scorers.
- `_classify_tools(...)` — correct/incorrect/hallucinated classification when `available_tools` set.
- `_aggregate_results(...)` — assembles ScorerResult with `status="ok"` plus per-mode metadata.
- `_no_tools_called_result(...)` — builds the `agent_no_calls` ScorerResult with full echoed metadata.
- `_data_error(dim, status, rationale)` — soft-fail builder for shape errors.
- `_with_failed_suffix(rationale, count)` — appends `"(N failed call(s) skipped)"` to rationale when `ignore_failed_calls` filtered any entries.

---

## 5. Dataset authoring guidance (what the user's colleagues need to know)

Two examples — simplest and most comprehensive:

```jsonc
// Simplest — tool_usage only, no args, no ordering
"expected_tool_calls": [
  "get_accounts",
  "get_transactions"
]
```

```jsonc
// Most comprehensive — works for both tool_usage AND tool_parameter
"expected_tool_calls": [
  {
    "step": 1,
    "tool": "get_accounts",
    "parameters": {
      "type": "SAVING",
      "ownedByCurrentUser": true
    },
    "reason": "Resolve which accounts to inspect"
  },
  {
    "step": 2,
    "tool": "get_transactions",
    "parameters": {
      "date_from": "2026-01-01",
      "date_to":   "2026-01-31",
      "direction": "OUTGOING"
    },
    "depends_on": "get_accounts",
    "reason": "Fetch this month's outgoing transactions"
  }
]
```

**Per-entry key reference:**

| Key | Required | Read by | Notes |
|---|---|---|---|
| `tool` | yes | both | The only required key |
| `parameters` | required when item opts into `tool_parameter` (use `{}` to skip) | `tool_parameter` | Subset semantics — list only the keys you want to assert |
| `depends_on` | optional | `tool_usage` (sequence mode only) | Single tool name; expresses causal ordering |
| `step` | optional | neither (informational on expected; sort-key on actual when all entries have it) | Useful for documenting batches |
| `reason` | optional | neither (informational) | Free-form annotation |

**Authoring gotchas:**
- Don't assert on persona-specific opaque IDs (`accountId="408C..."`, etc.) — they regenerate. Stick to structural args (`type`, `direction`, `date_from`, `ownedByCurrentUser`, `search`).
- `expected_tool_calls = []` with `tool_usage` = "agent should call no tools" (valid assertion).
- `expected_tool_calls = []` with `tool_parameter` opted in = `status="testcase_no_scorable_entries"` (broken test).

---

## 6. Open follow-ups (ordered by priority)

### Immediately pending (this session)

- [ ] **User end-to-end review** of the uncommitted changes (consolidation + status taxonomy). The user said they would review one more time before commit. Wait for their go-ahead.
- [ ] After review: stage and commit. Probable commit messages:
  - `refactor: GCAI-782 consolidate tool-call scorer fields to expected_tool_calls/actual_tool_calls`
  - `refactor: GCAI-782 soft-fail data errors via metadata.status taxonomy`
- [ ] PR description / handoff to colleagues. The user wants colleagues to review the **whole** PR, not partial.

### Tracked in `/Users/SG7CB/Developer/personal/tasks.md` (separate branches, future work)

§1–§5 under "ToolUsageScorer follow-ups":
1. Add tests for parallel-sibling pattern in `expected_tool_calls` (no test today pins this).
2. Document parallel-sibling pattern in `tool_usage.md`.
3. (Larger) Multi-parent `depends_on` — schema change + DAG cycle detector.
4. (Smaller) Position tracking for repeated tool calls (`_build_position_index` first-occurrence-only limitation).
5. (**Larger, likely the next big chunk**) **Real runtime-parallelism semantics** — use `step` on actual entries to verify `depends_on` via temporal partial order instead of position-in-list. Discussed at length in this session. The user explicitly wants this on a separate branch when ready. Requires empirical pass against real LangGraph traces first to validate that span-level `step` aligns with parallelism intent.

---

## 7. How to set up the dev environment in a fresh session

**Python interpreter:** `/opt/homebrew/opt/python@3.13/bin/python3.13`

**`pytest` location:** `~/Library/Python/3.13/bin/pytest` (Python 3.13 user-site install; the project's `.venv` is empty due to a network/cert issue we hit early on).

**Run all tests:**
```bash
cd /Users/SG7CB/Developer/ai-data-science
PYTHONPATH=evals/src:core/src ~/Library/Python/3.13/bin/pytest evals/tests/
# expect: 648 passed
```

**Run scorer tests only:**
```bash
PYTHONPATH=evals/src:core/src ~/Library/Python/3.13/bin/pytest evals/tests/test_scorers/
# expect: 571 passed
```

**Re-execute notebooks:**
```bash
cd /Users/SG7CB/Developer/ai-data-science/evals/nbs
/opt/homebrew/opt/python@3.13/bin/python3.13 -c "
import nbformat
from nbclient import NotebookClient
for n in ['tool_usage_demo.ipynb', 'tool_parameter_demo.ipynb']:
    nb = nbformat.read(n, as_version=4)
    NotebookClient(nb, kernel_name='python3', timeout=60).execute()
    nbformat.write(nb, n)
"
```

**Quick sanity-check the new soft-fail behavior:**
```bash
PYTHONPATH=evals/src:core/src /opt/homebrew/opt/python@3.13/bin/python3.13 <<'EOF'
from ai_data_science.evals.dimension import Dimension
from ai_data_science.evals.scales import BinaryScale
from ai_data_science.evals.scorers.deterministic.tool_usage import ToolUsageScorer
from ai_data_science.evals.types import ScoreLevel
dim = Dimension(id="t", name="T", description="T", scale=BinaryScale(),
                score_levels=(ScoreLevel(value=0, label="z", description="z"),
                              ScoreLevel(value=1, label="f", description="f")))
# Reference missing — used to raise, now soft-fails
r = ToolUsageScorer().score(dim, actual_tool_calls=["a"])
print(f"value={r.value} status={r.metadata['status']}")
# expect: value=0 status=testcase_missing_reference
EOF
```

---

## 8. Useful trace-data parsing snippet

If you need to look at the smoke-100 traces again (the CSV is 185MB, parse selectively):

```python
import csv, ast, json, sys
csv.field_size_limit(sys.maxsize)

with open('/Users/SG7CB/Developer/hg-ds-evals/experiments/api/input/traces_offline_smoke_pr12_kb_smoke_infer.csv') as f:
    r = csv.DictReader(f)
    for row in r:
        spans = ast.literal_eval(row['spans'])  # NOT json — Python repr
        for s in spans:
            attrs = s.get('attributes', {})
            if attrs.get('mlflow.spanType', '').strip('"') == 'TOOL':
                tool_name = s['name']
                # spanInputs is a JSON string within the dict
                args = json.loads(attrs['mlflow.spanInputs']) if attrs.get('mlflow.spanInputs') else {}
                # spanOutputs: JSON string with the tool's response
                # status: s['status']['code'] — 'STATUS_CODE_OK' on success
```

Key trace observations from this work:
- TOOL span args are **flat dicts**. No nesting. (Confirmed against 100 traces.)
- Same-tool-multi-call rate: **1 in 100** in this dataset. The one example was actually a tool-usage regression (over-call), not a legitimate multi-call.
- No `error` field is currently populated by the trace adapter — that's an upstream change still pending. Until then, `ignore_failed_calls=True` is a no-op on real data.

---

## 9. Conversation context the next session needs

A quick chronological summary so you understand the *why* behind decisions:

1. **Plan phase:** designed `ToolParameterScorer` against ToolUsageScorer's structure. Reviewed Kyrylo's draft `tool_input_parameters` (orchestrator-side, MLflow-Trace-coupled) — adopted shape ideas, rejected coupling. Plan doc at `/Users/SG7CB/Developer/personal/docs/jira-tasks/GCAI-782_tool_parameters_scorer_plan.md`.
2. **Implementation phase (chunked):** config + skeleton → validation → scoring → tests → fixtures → docs → notebook. ~7 chunks landed across commits `4ce02c9`–`8d9bf2b`.
3. **Retry-after-error chunk** (commit `b6415a0`): added `ignore_failed_calls` flag to `ToolUsageScorer` after the user noticed that retry scenarios would false-fail in `exact` mode. Includes the `error: bool` field convention on `actual_tool_calls`.
4. **Field consolidation (uncommitted):** the user pushed back on the four-field design as confusing for dataset authors. Collapsed `expected_tools`/`expected_tool_sequence`/`tools_called`/`actual_tool_sequence` to `expected_tool_calls`/`actual_tool_calls`, both polymorphic.
5. **Status taxonomy (uncommitted):** the user asked "if `parameters: {}` causes ValueError, won't this crash my batch run?" Yes. Refactored both scorers to never raise on data errors — return ScorerResult with `metadata.status` instead. Programmer/config errors still raise.

The user's collaboration pattern:
- They prefer **chunked PRs/commits** they can review individually.
- They want me to **propose before coding** when the change is non-trivial; for mechanical work, they're fine with me executing then summarizing.
- They like **clear matrices/tables** for design comparisons and decision trees.
- They are **critical of bloat** — keep code lean, no redundant abstractions, helpers only when called from 2+ places.
- They will catch **inconsistencies** between code, tests, docs, and notebooks. Always update all four when behavior changes.

---

## 10. What to **NOT** do without checking with the user first

- **Do not commit the uncommitted changes.** The user is in review mode. Wait for explicit approval.
- **Do not push the branch.** The user has not asked for that.
- **Do not start the parallelism v2 work** (tasks §3 / §5 in `personal/tasks.md`). That's a separate branch the user will create and tell you about.
- **Do not modify `ai-orchestrator` files.** This branch is `ai-data-science` only.
- **Do not change the field names** to anything other than `expected_tool_calls` / `actual_tool_calls`. The user explicitly chose parallel naming after a brief debate.
- **Do not introduce backward-compat aliases** for the old field names. The user said hard cut, no deprecation.
- **Do not "fix" the soft-fail-everywhere pattern back to raising.** The user explicitly chose this design after deliberation in this session.

---

## 11. If something is unclear, look here first

1. **Plan doc** (`/Users/SG7CB/Developer/personal/docs/jira-tasks/GCAI-782_tool_parameters_scorer_plan.md`) — has the canonical spec for every locked decision. §13 is the locked-decisions list.
2. **The notebooks** (`evals/nbs/*.ipynb`) — running examples of every supported scenario with cached outputs.
3. **The tests** (`evals/tests/test_scorers/test_*.py`) — the most precise documentation of behavior; if behavior is ambiguous, the test is the source of truth.
4. **`personal/tasks.md`** for known follow-ups + their relationships to each other.
5. **This handover** for the high-level picture.

---

**Last verified:** all 648 tests passing, both notebooks executing cleanly with cached outputs, status taxonomy live in both scorers. Branch state matches §1 above. Implementation is ready for the user's final review before commit.
