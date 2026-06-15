# Refactoring hg-ds-evals to sit on top of ai-data-science-evals

A plain-English plan for cleaning up `hg-ds-evals` by leaning on a new shared library called `ai-data-science-evals`. We **keep the Hey-George-specific bits**, **delete the duplicates** (upstream now has them), and **fix the seams** between the two.

---

## 1. The end-state — what you'll have when this is done

### The two repos, side by side

```
┌──────────────────────────────────────────────────────────────────────┐
│  ai-data-science-evals    (shared library — generic building blocks) │
│                                                                      │
│   ScoreLevel ─────────  one score on a scale (value, label, desc)    │
│   Scale       ─────────  Binary / Discrete / Numeric                 │
│   Dimension   ─────────  one thing to evaluate (id, name, scale)     │
│   ScorerResult─────────  the answer for one dimension on one row     │
│   EvalFields  ─────────  canonical column names                      │
│                          (expected_output, actual_agent, …)          │
│                                                                      │
│   BaseScorer  ─────────  parent class for any scorer                 │
│   ├── DeterministicScorer (no LLM needed)                            │
│   │     ├── ExactMatchScorer                                         │
│   │     ├── ToolUsageScorer                                          │
│   │     └── ToolParameterScorer                                      │
│   └── LLMJudgeScorer       (async, calls an LLM)                     │
│                                                                      │
│   JudgeClient (Protocol)  ─── what any LLM client must provide       │
│                               (one async method: acomplete(...))     │
└──────────────────────────────────────────────────────────────────────┘
                                  ▲
                                  │   imports from upstream
                                  │
┌──────────────────────────────────────────────────────────────────────┐
│  hg-ds-evals   (the Hey-George layer — domain logic, glue, runner)   │
│                                                                      │
│   Rubric             ─── composes upstream Dimensions + adds         │
│                          prompt sections (system_context,            │
│                          judge_instructions, root_cause_categories…) │
│   YAML loader        ─── turns experiment YAML into a Rubric         │
│   PromptBuilder      ─── renders system + user prompts (Jinja)       │
│   AzureDatabricksJudgeClient                                         │
│                      ─── concrete LLM client; implements upstream's  │
│                          JudgeClient interface                       │
│   async_run_evals    ─── batch runner: concurrency, TPM throttling,  │
│                          checkpoint resume                           │
│   Trace preprocessing─── MLflow OTel traces → eval DataFrame         │
│   HeyGeorgeToolParameterScorer                                       │
│                      ─── subclass of upstream's ToolParameterScorer  │
│                          with banking-API rules                      │
└──────────────────────────────────────────────────────────────────────┘
```

### How a run flows, end to end

```
   experiments/<eval_type>/configs/*.yaml
                │
                ▼
   load_experiment_config()
                │
                ▼
   ┌─────────────────────────────────────────────────────────┐
   │ Rubric                                                  │
   │   dimensions: [upstream Dimension, …]                   │
   │   input_fields, output_schema                           │
   │   judge_instructions, system_context, …                 │
   └─────────────────────────────────────────────────────────┘
                │
                ▼
   PromptBuilder  ──►  system prompt + user prompt template
                │
                ▼
   async_run_evals(df, system_prompt, judge_client, config)
                │
                │  for each batch of rows:
                ▼
   AzureDatabricksJudgeClient.acomplete(system, user, config)
                │
                │  returns raw text (JSON)
                ▼
   parse_to_scorer_results(text, rubric)
                │
                │  →  list[ScorerResult]   (one per dimension)
                ▼
   Checkpoint CSV (one row appended)
                │
                ▼
   experiments/*_report.py  ──►  HTML reports
```

The key change: today the runner mostly passes dicts of strings around. After the refactor, the runner emits **`ScorerResult` objects** internally and only flattens to CSV at the very end.

---

## 2. What stays, what goes, what changes — the big picture

### KEEP from hg-ds-evals (the good parts — these are genuinely Hey George)

| What | Why we keep it |
|------|----------------|
| `Rubric` (in [hg_ds_evals/rubrics/base.py](hg_ds_evals/rubrics/base.py)) | Upstream only has individual `Dimension`s. The idea of *composing* dimensions and bundling extra prompt sections (`judge_instructions`, `root_cause_categories`, `system_context`, etc.) is HG-specific and not duplicated anywhere. |
| `Rubric.override_from_yaml` and the rest of [rubrics/loader.py](hg_ds_evals/rubrics/loader.py) | HG-specific YAML shape; upstream has no concept of "load a rubric from YAML." |
| `PromptBuilder` + the Jinja templates ([prompts/builder.py](hg_ds_evals/prompts/builder.py)) | Upstream has no prompt builder. The Jinja-template + per-experiment overrides have already paid off. |
| The async batch runner ([evals/evaluator.py:105-286](hg_ds_evals/evals/evaluator.py)) | TPM throttling, checkpoint resume, concurrent semaphore — upstream has none of this. |
| The Azure/Databricks client ([llm/api_calls.py](hg_ds_evals/llm/api_calls.py), [llm/api_client.py](hg_ds_evals/llm/api_client.py)) | Concrete implementation with Databricks OAuth refresh + tenacity retry. We'll just rename it and wrap it in upstream's `JudgeClient` shape. |
| All of [preprocessing/traces.py](hg_ds_evals/preprocessing/traces.py) | MLflow trace parsing. Upstream has nothing comparable. |
| [preprocessing/latency.py](hg_ds_evals/preprocessing/latency.py), [preprocessing/fallback.py](hg_ds_evals/preprocessing/fallback.py), [transformers/*](hg_ds_evals/transformers) | HG-specific PySpark pipeline pieces. |
| [evals/api_utils.py](hg_ds_evals/evals/api_utils.py) banking-API parameter rules | HG-specific; we just plug them *into* upstream's `ToolParameterScorer` instead of running a parallel rule engine. |
| HTML report scripts (`experiments/api_report.py`, `experiments/kb_report.py`) | They read from the checkpoint CSV — and we keep the CSV format unchanged. |

### DROP from hg-ds-evals (upstream now provides these)

| What we delete | Replace with |
|----------------|--------------|
| `ScoreLevel`, `Dimension` in [hg_ds_evals/core/types.py](hg_ds_evals/core/types.py) | `from ai_data_science.evals import ScoreLevel, Dimension` |
| The implicit "scale = tuple of ScoreLevels" pattern | Upstream's real `Scale` class hierarchy (`DiscreteScale(("bad","partial","good"))`, etc.) |
| Hard-coded column names like `"expected_output"`, `"actual_agent"` | `EvalFields.EXPECTED_OUTPUT`, `EvalFields.ACTUAL_AGENT` (upstream's `StrEnum`) |
| Re-export shims [preprocessing/skkb_traces.py](hg_ds_evals/preprocessing/skkb_traces.py) and [preprocessing/mlflow_traces.py](hg_ds_evals/preprocessing/mlflow_traces.py) | Just import from `preprocessing/traces.py` directly |
| Maybe `OutputField` / `OutputSchema` — barely used today | The Rubric already carries the prompt-rendering info; verify before deleting |

### CHANGE so the two repos fit together (the actual refactor)

Six concrete tasks. Details and order below.

---

## 3. The plan — six tasks in plain English

Each one is **independent enough to ship in its own PR**, but they have an order (later ones depend on earlier ones). Let's go through them in the recommended sequence.

> **Before you start any of these:** add safety-net tests. Right now `tests/` only covers `api_utils.py` and a slice of `traces.py`. Nothing covers `Rubric`, `PromptBuilder`, the YAML loader, `parse_single_row_response`, or `async_run_evals`. Add a small "characterization test" for each of those — basically a test that captures *current behaviour* — before you change anything. Even one assertion per file is enough as a tripwire.

---

### Task 1 — Adopt upstream's type system

**What changes:** Stop defining `Dimension` and `ScoreLevel` in hg-ds-evals. Import them from `ai_data_science.evals` instead.

**Why this is first:** Everything else depends on the two repos using the same type for `Dimension`.

**The catch:** Upstream's `ScoreLevel` uses a field called `value` (a `float`), while hg-ds-evals uses `score` (an `int`). So every place that constructs a `ScoreLevel(score=0, ...)` needs to become `ScoreLevel(value=0, ...)`. Same goes for code that reads `.score` off a `ScoreLevel`. About 30 sites, mostly in `rubrics/dimensions/catalog.py`.

**Files touched:**
- [hg_ds_evals/core/types.py](hg_ds_evals/core/types.py) — becomes a tiny shim that just re-exports upstream's symbols.
- Anywhere that does `from hg_ds_evals.core.types import ...`.

**Sketch — before:**
```python
# hg_ds_evals/core/types.py
@dataclass(frozen=True)
class ScoreLevel:
    score: int            # ← named "score", int only
    label: str
    description: str
```

**Sketch — after:**
```python
# hg_ds_evals/core/types.py — shim
from ai_data_science.evals.types import ScoreLevel, ScorerResult
from ai_data_science.evals.dimension import Dimension
from ai_data_science.evals.scales import DiscreteScale, BinaryScale, NumericScale

# Hey-George-specific types stay here:
@dataclass(frozen=True)
class InputField: ...
# (and OutputField/OutputSchema — see Task 6)
```

**Risk:** Medium. The field rename is mechanical but easy to miss one site. Mitigation: add a `mypy` or `ruff` check that runs as part of the commit; do the rename in one commit (don't leave an intermediate state).

**Cost:** Small (a day or two).

---

### Task 2 — Migrate the dimension catalog to use a real `Scale`

**What changes:** Today, an hg-ds-evals `Dimension` carries a `scale = (ScoreLevel(0, "bad", …), ScoreLevel(1, "partial", …), ScoreLevel(2, "good", …))`. The "scale" is implicit — just a tuple of levels.

Upstream's `Dimension` makes the scale **explicit**: there's a separate `scale: DiscreteScale(("bad","partial","good"))` object **and** a `score_levels` tuple that explains what each level means. Upstream's `Dimension` then validates at construction time that the two match — if you forget a level, you get a clear error.

**Why this matters for you (DS):** Today, if a YAML override changes the scale to a 5-point scale but forgets to declare the labels, you find out at LLM-response-parse time, often with a confusing error. After this change, you'd find out at the line where the `Dimension` is constructed.

**Files touched:**
- [hg_ds_evals/rubrics/dimensions/catalog.py](hg_ds_evals/rubrics/dimensions/catalog.py) — every `Dimension(...)` literal gets one small extra kwarg.
- [hg_ds_evals/rubrics/fallback.py](hg_ds_evals/rubrics/fallback.py), [hg_ds_evals/rubrics/output_prompt.py](hg_ds_evals/rubrics/output_prompt.py) — same shape change.
- [hg_ds_evals/rubrics/base.py:155-175](hg_ds_evals/rubrics/base.py) — the YAML scale parser needs to return both pieces.

**Sketch — before:**
```python
STANDARD_SCALE_3PT = (
    ScoreLevel(0, "bad",     "..."),
    ScoreLevel(1, "partial", "..."),
    ScoreLevel(2, "good",    "..."),
)

USER_QUERY_CLARITY = Dimension(
    id="user_query_clarity",
    name="...",
    description="...",
    scale=STANDARD_SCALE_3PT,
    weight=1.0,
)
```

**Sketch — after:**
```python
STANDARD_3PT_SCALE  = DiscreteScale(("bad", "partial", "good"))
STANDARD_3PT_LEVELS = (
    ScoreLevel(value=0, label="bad",     description="..."),
    ScoreLevel(value=1, label="partial", description="..."),
    ScoreLevel(value=2, label="good",    description="..."),
)

USER_QUERY_CLARITY = Dimension(
    id="user_query_clarity",
    name="...",
    description="...",
    scale=STANDARD_3PT_SCALE,         # NEW: the scale type
    score_levels=STANDARD_3PT_LEVELS, # what each level means
    weight=1.0,
)
```

**Risk:** Medium — if any YAML config in `experiments/*/configs/` defines a custom `scale:`, that YAML needs to be updated too. Audit first: `grep -rn "scale:" experiments/*/configs/`.

**Cost:** Medium (one to two days — mostly mechanical).

---

### Task 3 — Wrap the LLM client as a `JudgeClient`

**What changes:** Today, `async_call_llm_for_evaluation` in [llm/api_calls.py](hg_ds_evals/llm/api_calls.py) does too much: it knows about `row_dict`s, system prompts, the semaphore, the config dict, and the actual LLM call all at once.

Upstream defines a small interface called `JudgeClient` with **one method**:

```python
class JudgeClient(Protocol):
    async def acomplete(*, system: str, user: str, config: JudgeConfig) -> JudgeResponse: ...
```

We make our Azure/Databricks client implement that interface. The retry logic, OAuth refresh, and Databricks-specific bits all stay — we just expose them through a cleaner door.

**Why this matters:** Future upstream scorers (the `LLMJudgeScorer` family — upstream has the ABC ready, with concrete implementations coming) will speak `JudgeClient`. Once our client implements the interface, we can pick those up for free.

**Files touched:**
- [hg_ds_evals/llm/api_calls.py](hg_ds_evals/llm/api_calls.py) — the body becomes a method on a new class.
- [hg_ds_evals/llm/api_client.py](hg_ds_evals/llm/api_client.py) — `get_api_client()` becomes a private helper.
- New file: `hg_ds_evals/llm/judge_client.py` with `AzureDatabricksJudgeClient`.

**Sketch — after:**
```python
from ai_data_science.evals.scorers.llm_judge.client import JudgeClient, JudgeResponse
from ai_data_science.evals.scorers.llm_judge.config import JudgeConfig

class AzureDatabricksJudgeClient:           # implements JudgeClient Protocol
    def __init__(self, *, provider, model_deployment_name, ...): ...

    async def acomplete(self, *, system, user, config):
        # all the existing retry / OAuth-refresh / param-shaping code goes here
        resp = await self._call_with_refresh(...)
        return JudgeResponse(
            text=resp.output_text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )
```

**Risk:** Low. The hard logic (retry, OAuth) doesn't change — we're just moving it inside a class.

**Cost:** Small (a day).

---

### Task 4 — Use upstream's canonical column names in the trace builder

**What changes:** Today [preprocessing/traces.py](hg_ds_evals/preprocessing/traces.py) writes column names as bare strings: `"expected_output"`, `"actual_agent"`, `"expected_tool_calls"`, etc. Upstream defines these as an `EvalFields` enum.

**Why this matters:** If upstream ever renames a column, you'd silently get a missing column instead of a Python error. The docstring of `traces.py` already promises it matches upstream's contract — this just enforces that promise.

**Files touched:**
- [hg_ds_evals/preprocessing/traces.py](hg_ds_evals/preprocessing/traces.py) — ~30 string sites become enum references.
- Possibly [hg_ds_evals/evals/api_utils.py](hg_ds_evals/evals/api_utils.py) too.

**Sketch:**
```python
# Before:
out_df["expected_output"] = ...
out_df["actual_agent"]    = ...

# After:
from ai_data_science.evals.config import EvalFields
out_df[EvalFields.EXPECTED_OUTPUT] = ...
out_df[EvalFields.ACTUAL_AGENT]    = ...
```

**Risk:** Low. `tests/test_skkb_traces.py` exercises the resulting columns, so any typo would fail the test.

**Cost:** Small (half a day).

---

### Task 5 — Plug banking-API rules into upstream's `ToolParameterScorer`

**What changes:** Today [evals/api_utils.py](hg_ds_evals/evals/api_utils.py) has its own standalone function `compute_tool_parameter_equivalence(...)` that compares expected vs actual tool calls, with HG-specific normalisation (banking-API defaults, `analyze_transactions` overrides, alias handling).

Upstream has a `ToolParameterScorer` that does the same generic comparison. We make a Hey-George subclass that injects the banking normalisation.

**Why this matters:** Right now, API smoke evals live in a parallel universe — they're scored by a function that doesn't go through the same `Scorer` interface as judge evals. After this, an API eval row produces a `ScorerResult` just like a judge eval row, and reports can treat them uniformly.

**Files touched:**
- [hg_ds_evals/evals/api_utils.py](hg_ds_evals/evals/api_utils.py) — `compute_tool_parameter_equivalence` becomes `HeyGeorgeToolParameterScorer._apply_banking_defaults`.
- [tests/test_api_utils.py](tests/test_api_utils.py) — instantiate the scorer instead of calling the function directly.

**Sketch:**
```python
from ai_data_science.evals.scorers.deterministic import ToolParameterScorer

class HeyGeorgeToolParameterScorer(ToolParameterScorer):
    """Adds Hey-George banking-API rules to the generic tool-parameter scorer."""

    def _normalize(self, *, tool_name: str, params: dict) -> dict:
        return _apply_banking_defaults(tool_name, params)
```

**Risk:** Low — your existing `test_api_utils.py` tests act as the safety net. If upstream's `ToolParameterScorer` doesn't expose a clean override hook, fall back to a thin wrapper (call upstream's `.score()` after pre-normalising) — same effect.

**Cost:** Medium (one to two days).

**Depends on:** Task 1.

---

### Task 6 — Make the runner emit `ScorerResult[]` internally

**What changes:** Today the runner builds a row dict like `{topic_relevance_score: 2, topic_relevance_reasoning: "...", clarity_score: 1, ...}` and appends it straight to the checkpoint CSV. The "scoring" and "writing CSV" steps are tangled together.

After this change, the runner internally builds `[ScorerResult(name="topic_relevance", value=2, rationale="..."), ScorerResult(name="clarity", value=1, ...), ...]` and **then** a small `CheckpointWriter` flattens it to today's CSV layout.

**Why this matters:**
- The CSV format stays **identical** — your report scripts (`api_report.py`, `kb_report.py`) keep working unchanged.
- In memory, the data has the same shape as upstream's scorer contract, which makes it easy to compare/aggregate/feed elsewhere later.
- Errors become typed (`ScorerResult.metadata` carries the parse-error reason explicitly).

**Files touched:**
- [hg_ds_evals/evals/parsers.py](hg_ds_evals/evals/parsers.py) — split `parse_single_row_response` into "parse JSON" + "convert to ScorerResult list".
- [hg_ds_evals/evals/evaluator.py:188-256](hg_ds_evals/evals/evaluator.py) — the inner loop produces `ScorerResult`s; a `CheckpointWriter` does the CSV writing.
- [hg_ds_evals/common/utils.py](hg_ds_evals/common/utils.py) — `update_checkpoint_df` either goes away or becomes a method on the writer.

**This is the riskiest task on the list.** The checkpoint CSV is the durability story — if you can't resume, nothing matters. **Before you start:** write a "golden-file test" that runs a small eval (3-5 rows), kills it mid-batch, restarts it, and asserts the final CSV is byte-identical to one produced by a single uninterrupted run. Keep that test as a tripwire forever.

**Cost:** Medium (two to three days).

**Depends on:** Task 1.

---

## 4. Recommended order to do them in

```
   1. Add safety-net tests (Rubric, PromptBuilder, parsers, runner smoke test)
        │
        ▼
   2. Task 1 — adopt upstream Dimension/ScoreLevel
        │
        ▼
   3. Task 2 — migrate dimension catalog to real Scale
        │
        ▼
   4. Task 3 — wrap the LLM client as JudgeClient
        │
        ▼
   5. Task 4 — use upstream's column-name enum  ◄── independent, can move earlier
        │
        ▼
   6. Task 5 — plug banking rules into upstream's ToolParameterScorer
        │
        ▼
   7. Task 6 — runner emits ScorerResult internally  ◄── do last; highest risk
        │
        ▼
   8. Cleanup — delete OutputField/OutputSchema if unused; delete re-export shims;
                consider a typed ExperimentConfig dataclass instead of dict-of-anything
```

Tasks 4 and 5 are independent of each other; either can move earlier in the sequence if convenient.

---

## 5. Things I explicitly left out (and why)

These came up while I was reading the code but **I'm not proposing them now** — including them would balloon the scope.

- **Splitting `preprocessing/traces.py` (2,636 lines, 98 functions).** Real long-function smell, but unrelated to the upstream-adoption goal. Worth its own refactoring pass later — start by extracting the orchestrator functions `build_skkb_dataframe_from_mlflow_search_traces` and `build_dataframe_from_mlflow_traces` into smaller pieces. Don't bundle with this work.
- **Switching to upstream's per-Dimension `LLMJudgeScorer` (one async call per dimension).** Today Hey George makes *one* LLM call per row that returns scores for *all* dimensions — that's the cost model. Upstream's per-dimension scorer pattern would multiply your LLM calls by `len(dimensions)`. **We adopt the return type (`ScorerResult[]`), not the call shape.**
- **Replacing `Rubric` with a flat `list[Dimension]`** to match upstream's "atomic dimension" philosophy. The Rubric class carries non-dimension state (`judge_instructions`, `system_context`, `root_cause_categories`, `domain_specific_guidance`, `final_reminders`, `pass_threshold`) that maps to prompt-building responsibilities — these are real Hey George needs and upstream has nothing equivalent. Keep Rubric.
- **Moving `common/mlflow_otel.py` (1,088 LOC) upstream.** It was vendored from `ai-data-science` and *should* eventually live there, but upstream's `data-engineering/src/` is still empty (only test fixtures exist). Track this as a separate, longer-term cleanup — gated on upstream getting a public API for OTel materialisation.
- **Renaming `evals/api_utils.py` → `evals/api_scorers.py`** to match its new role after Task 5. Pure rename, no behaviour change; do it only if/when you touch the file anyway.

---

## 6. Side-observations (not part of the refactor, but worth knowing)

These are things I noticed while reading. They're not refactoring tasks — they're things you might want to fix separately.

1. **`evaluator.py:139-144`** — checkpoint CSV is read with `on_bad_lines="warn"`. If a checkpoint has a corrupt line from a mid-write crash, the runner silently skips rows that were meant to be filtered out, causing the LLM to be re-called on them. Worth changing to `on_bad_lines="error"` plus a one-pass repair tool.
2. **`evaluator.py:159-184`** — `asyncio.gather(*tasks)` raises on the first exception and cancels in-flight siblings, so a single transient error past tenacity's retries kills the whole batch (you lose responses that would have succeeded). Consider `return_exceptions=True` + per-row error capture.
3. **`evaluator.py:194-256`** — when the parse-retry loop exhausts `max_parse_retries`, the *last* failed result is what gets checkpointed, but `input_tokens` / `output_tokens` are accumulated across retries. The metrics for that row will look like it consumed N× the tokens. Verify the accounting before/after.
4. **`common/utils.py:22-80`** — large blocks of commented-out token-counting code. If genuinely abandoned, delete; if still planned, restore from git history when you need it.

---

## 7. What I need from you to start

> Reply with the numbers (1-6) you'd like me to apply, in the order you'd like them. I'll do them one at a time, run the tests in between, and check in after each.
>
> If you want the safest path, start with **"add safety-net tests, then Task 1"** and we'll go from there.
