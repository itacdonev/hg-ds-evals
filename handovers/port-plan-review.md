# Review of `llm_judge_scorers_plan.md` vs hg-ds-evals findings

Comparison of [the existing port plan](/Users/SG7CB/Developer/personal/docs/jira-tasks/LLM%20as%20a%20judge/llm_judge_scorers_plan.md) against what's actually in hg-ds-evals today.

---

## TL;DR

- **The plan is largely right.** Architecture, override audit, panel design, column semantics, scorer-per-dimension contract — all sound. The 11-story breakdown in the JIRA appendix is well-scoped.
- **One real gap**: the plan defers MLflow trace ingestion to v2, but your restated goal puts it in v1. The trace-parsing code in hg-ds-evals (`preprocessing/traces.py` + `common/mlflow_otel.py`, ~3,700 LOC together) needs a temporary home until `ai-data-science-data-engineering` catches up.
- **Three smaller decisions** are still open (see "Questions" at the bottom).

---

## Visual: what moves where

```
hg-ds-evals  (source)                  →   ai-data-science  (destination)
─────────────────────                       ──────────────────────────────

rubrics/                                ─►  evals/scorers/llm_judge/
  dimensions/catalog.py                     instances/ (one file per dimension:
                                              query_clarity.py, etc.)
  base.py (Rubric class)                ─►  evals/scorers/llm_judge/panel.py
                                            (Rubric → JudgePanel; YAML picks
                                             a subset of dims + output fields)
  loader.py (YAML loader)               ─►  evals/runner/config/loader.py

prompts/                                ─►  evals/scorers/llm_judge/
  builder.py                                prompt.py + templates/

llm/                                    ─►  evals/runner/api/
  api_client.py, api_calls.py               client.py, calls.py
                                            (concrete impls of JudgeClient)

evals/                                  ─►  evals/runner/execution/
  evaluator.py, run_evals.py                evaluator.py + entry point
  parsers.py                            ─►  evals/scorers/llm_judge/parser.py

preprocessing/                          ─►  evals/runner/_data_tmp/  ◄── NEW
  traces.py  (2,636 LOC)                    (transitional — moves to
                                             data-engineering when ready)
common/
  mlflow_otel.py  (1,088 LOC)           ─►  evals/runner/_data_tmp/  ◄── NEW

common/utils.py (checkpoint helpers)    ─►  evals/runner/checkpoint/

evals/api_utils.py                      ─►  ?  ◄── decision needed (Q3)

────────── stays in hg-ds-evals ──────────────────────────────────────────
preprocessing/latency.py, preprocessing/fallback.py
transformers/, common/base_transformers.py
rubrics/fallback.py, rubrics/output_prompt.py
experiments/*
```

---

## What the plan gets right (no changes needed)

- **One workspace package, two install profiles** (`evals` lean + `evals[runner]` heavy). The dep-inversion via `JudgeClient` Protocol is clean — `ai-orchestrator` consumers don't pull `pyspark`/`mlflow`/`openai`.
- **Scorer-per-dimension** with canonical dimension owned by each scorer class, plus override-audit on `__init__`. This is the right shape — the alternative (one LLM call per dim) would N-multiply your costs.
- **`JudgePanel`** = one LLM call, many dimensions, plus optional `OutputField`s. Matches today's hg-ds-evals cost model exactly.
- **`OutputField` classes** for diagnostic non-rubric fields (case_scope, hallucinated_claims, etc.). Cleaner than the current YAML approach where these live in YAML text.
- **Column-semantics enforcement** (Story 10): `id_columns / eval_columns / passthrough_columns` rules with no overlap. Eliminates the current mess where ~15 columns are duplicated across two lists.
- **Override audit with fingerprints + `--strict-canonical` mode** for BoM safety.
- **11-story breakdown** in the appendix is well-scoped — each is shippable.

---

## What's missing or should change

### Gap 1 — Trace ingestion belongs in v1, not v2

The plan §6.4 says *"Inline `mlflow.search_traces()` ingestion — v2; for v1, expect a pre-materialized table."*

Your restated goal: *"a user can run an evaluation on an inference dataset providing only the mlflow run ID."* That means v1 has to do the ingestion + trace parsing itself, since `ai-data-science-data-engineering` hasn't got there yet.

**Suggested change to the plan:**

- Move MLflow trace ingestion **into v1**.
- Add a new Story (call it 11.5 or expand Story 11): *"Port trace-parsing into `evals/runner/_data_tmp/`."*
- The `_data_tmp/` name signals "transitional — moves to `ai-data-science-data-engineering` when that package matures." Single public entrypoint, something like `load_eval_dataframe(mlflow_run_id, config) -> pd.DataFrame`.
- Update plan §6.2 ("What we port from hg-ds-evals near-verbatim") to add:
  - `hg_ds_evals/preprocessing/traces.py` (2,636 LOC) → `evals/runner/_data_tmp/traces.py`
  - `hg_ds_evals/common/mlflow_otel.py` (1,088 LOC) → `evals/runner/_data_tmp/mlflow_otel.py`

### Gap 2 — Where exactly does the transitional code live?

The plan doesn't name a location for "transitional, will-move-later" code. Three options — see **Q2** at the bottom.

### Gap 3 — `evals/api_utils.py` is not mentioned

The plan doesn't address the 484-LOC `evals/api_utils.py` with Hey-George-specific banking-API parameter rules (`analyze_transactions` defaults, alias handling, transaction-collection-id overrides). This is *deterministic* scoring logic, not LLM-judge — but it's coupled to the API-eval experiment that the plan otherwise ignores. See **Q3**.

### Gap 4 — `pass_threshold` semantics still open

This is open question #4 in the plan itself. The current `Rubric.pass_threshold` (weighted-average gate, default 1.5 on 0–2) doesn't get much real use in the codebase. In a strict-canonical world, BoM gating happens via fingerprints, not thresholds. See **Q4**.

---

## Small design choices worth confirming (not gaps)

These are minor but worth a sentence each:

1. **`NumericScale([0,1,2])` vs `DiscreteScale(("unclear","ambiguous","clear"))`** in the plan §3.2 example. `DiscreteScale` carries the labels as part of the scale (label-to-value mapping in one place). Suggest using `DiscreteScale` for ordinal labels like bad/partial/good; reserve `NumericScale` for genuinely numeric ranges (e.g. a 0–10 continuous score).

2. **Jinja templates in the wheel** (Story 3). Confirm experiments can still override per-experiment (today's hg-ds-evals lets you drop a `system.md.j2` next to the YAML). The simplest design: keep the defaults embedded as Python strings, with a YAML-level `paths.system_template_path` that loads a `.j2` from disk when set — mirrors today's behaviour exactly. The `templates/` folder in the plan is then just the embedded defaults' source-of-truth.

3. **Checkpoint format = CSV** (carries over from hg-ds-evals). Fine. The Story 10 column union should drive the CSV column order so existing report scripts (`api_report.py`, `kb_report.py`) stay readable without changes. Future Delta sink is v2+.

---

## Things in hg-ds-evals that should NOT move

(Confirming the boundary so nothing slips through.)

| File / folder | Why it stays |
|---|---|
| `hg_ds_evals/preprocessing/latency.py` | HG-specific latency bucketing — banking-pipeline-shaped |
| `hg_ds_evals/preprocessing/fallback.py` | HG fallback-experiment preprocessing |
| `hg_ds_evals/transformers/*`, `common/base_transformers.py` | HG PySpark transformers |
| `hg_ds_evals/rubrics/fallback.py`, `rubrics/output_prompt.py` | HG-specific rubric instances. (The skkb scorers DO move via Story 5; the fallback rubric stays for now.) |
| `experiments/*` | All HG experiment artifacts |

The `rubrics/dimensions/catalog.py` file is interesting: it has both generic-shaped dimensions (`USER_QUERY_CLARITY`) and HG-specific ones. Story 5 only ports the seven skkb dimensions — but if other dimensions in the catalog are reused across HG experiments, those stay in hg-ds-evals as imports from upstream + HG-only ones.

---

## Resolutions (from chat)

| Question | Decision |
|---|---|
| **Q1 — v1 input shape** | **Accept both.** v1 YAML can specify either an `mlflow_run_id` (runner does parsing via `_data_tmp/`) or a pre-materialized table name. Keeps existing skkb YAML working while enabling the run-ID-only workflow. |
| **Q2 — Temp code location** | **`evals/runner/_data_tmp/`** (ships under `[runner]` extras; underscore prefix signals "internal, transitional"; moves to `ai-data-science-data-engineering` once that package is ready). |
| **Q3 — `evals/api_utils.py`** | **Port, but stage it for adjustment.** It's tied to `ToolParameterScorer` and will be refactored during/after the port. Recommend: land it in a parallel temp location (e.g. `evals/runner/_scorers_tmp/api_utils.py`) with an "expect-to-change" tag; the real integration with `ToolParameterScorer` is a follow-up. |
| **Q4 — `pass_threshold`** | **Keep on `JudgePanel` for v1.** Mirrors today's hg-ds-evals; revisit later. |

---

## Concrete edits to the JIRA stories

Based on the resolutions above:

### Edit existing stories

- **Story 11** (Async runner + skkb e2e) — change *"v1: just read a Spark table by name from MLflow-registered datasets"* to *"v1: accept either an MLflow run ID (uses `_data_tmp/` parser) or a pre-materialized table name."*
- **Story 5** (Port the seven skkb scorers) — already correct; just be explicit that source-of-truth is a specific git SHA of the current skkb YAML (so iterative description edits are pinned).
- **Story 6** (OutputField base + 17 skkb output fields) — acceptance should include a diff of the LLM prompt produced by the canonical `OutputField` classes against the prompt produced by the current skkb YAML's raw-text `output_schema:`, to confirm semantic equivalence.
- **Story 10** (Column-semantics enforcement) — no change needed; the loader rule that `passthrough ∩ (id ∪ eval) == ∅` will already catch the current skkb YAML's ~15 duplicated columns and force a cleanup.
- **§6.4 (out-of-scope list)** — remove *"Inline `mlflow.search_traces()` ingestion (v2 — for v1, expect a pre-materialized table)"* since this is now in v1.

### Add three new stories

- **Story 12 — Port trace parsing into `evals/runner/_data_tmp/`**
  - **Description.** Copy `hg_ds_evals/preprocessing/traces.py` (2,636 LOC) and `hg_ds_evals/common/mlflow_otel.py` (1,088 LOC) into `evals/runner/_data_tmp/`. Don't refactor — `git mv`-equivalent. Mark with a `README.md` in the folder explaining "transitional code, slated to move to `ai-data-science-data-engineering`."
  - **Acceptance.**
    - A single `load_eval_dataframe(mlflow_run_id | table_name, config) -> pd.DataFrame` entrypoint at `evals/runner/_data_tmp/__init__.py`.
    - Output DataFrame columns match the `EvalFields` enum (`expected_output`, `actual_agent`, etc.).
    - The existing hg-ds-evals tests for trace parsing (`tests/test_skkb_traces.py`) port over and pass against the new location.
    - `README.md` documents the planned migration path.

- **Story 13 — Port `api_utils.py` into `evals/runner/_scorers_tmp/`** *(staged — expect refactoring)*
  - **Description.** Copy `hg_ds_evals/evals/api_utils.py` (484 LOC) into `evals/runner/_scorers_tmp/api_utils.py`. Don't redesign the integration with upstream's `ToolParameterScorer` yet — flag it as transitional. Tests port across with it (`tests/test_api_utils.py`).
  - **Why staged.** The banking-API rules (analyze_transactions defaults, parameter aliases, transaction-collection-id overrides) are tied to `ToolParameterScorer` and will likely need restructuring once the new scorer surface is in place. v1 keeps behaviour identical; v2 designs the proper integration.
  - **Acceptance.**
    - `compute_tool_parameter_equivalence` and the `_relax_*` helpers importable from `ai_data_science.evals.runner._scorers_tmp.api_utils`.
    - `tests/test_api_utils.py` passes against the new location.
    - `README.md` in `_scorers_tmp/` documents the planned refactor.

- **Story 14 — Cleanup follow-up** *(v2 — placeholder)*
  - Refactor `_scorers_tmp/api_utils.py` into a proper subclass / normalisation hook on `ToolParameterScorer`.
  - Move `_data_tmp/` contents into `ai-data-science-data-engineering` once that package exposes a public API.
  - Both folders' `README.md` files point at this ticket so the transitional code never becomes permanent by accident.

### Updated story sequence

```
Story 1  — LLMJudgeScorer base + registry            (foundation)
Story 2  — Azure + DBX async client                  (transport)
Story 3  — Prompt builder + parser                   (per-call mechanics)
Story 4  — JudgePanel + override audit               (composition)
Story 5  — Port the seven skkb scorers               (canonical content)
Story 6  — OutputField base + 17 skkb fields         (canonical content)
Story 7  — Publish ai-data-science-evals to JFrog    (infra; can land anytime)
Story 8  — evals/runner/ subpackage skeleton         (infra; before Stories 9-13)
Story 9  — YAML loader + override resolver           (config layer)
Story 10 — Dataset column-semantics enforcement      (config layer)
Story 11 — Async runner + checkpoint + skkb e2e      (the e2e demo)
Story 12 — Port trace parsing into _data_tmp/        ◄── NEW
Story 13 — Port api_utils.py into _scorers_tmp/      ◄── NEW (staged)
Story 14 — v2 cleanup placeholder                    ◄── NEW (deferred)
```

Story 12 should land **before or alongside** Story 11 (the e2e demo needs it). Story 13 can land independently any time after Story 8 (the runner skeleton).

---

## Critique check — the nine prior findings

Status of each of the nine critical findings raised previously, **after** my Story 12/13/14 additions above. Verified against the code where I could.

| # | Critique | Still present? | Impact | Verified by |
|---|---|---|---|---|
| 1 | "Port near-verbatim" misleads | **Yes** | High | Plan §6.2 still says "near-verbatim"; the listed files have shape mismatches |
| 2 | LLM scorer contract vs `BaseScorer` dimension arg | **Yes** | Medium | Upstream `LLMJudgeScorer.ascore(_dimension=None, …)` has no rule for mismatch |
| 3 | skkb `output_schema.additional_instructions` cross-field rules dropped | **Yes** | High | skkb YAML lines 657-718 — confirmed real |
| 4 | Override audit conflicts with YAML weights | **Yes** | High | Plan §7.1 YAML supplies weights; §4.1 calls weights an override |
| 5 | Runner dep isolation incomplete | **Yes** | Medium | Confirmed `evals/pyproject.toml` has only `ai-data-science-core`, no extras; `pandas` not listed in §6 deps |
| 6 | Don't ad-hoc-recreate trace ingestion in `evals.runner` | **Partially** | Medium | My Story 12 mitigates with `_data_tmp/` but doesn't say to reuse upstream `MLflowTraceExtractor` |
| 7 | Panel missing-input semantics undefined | **Yes** | High | Plan defers to "field categories" but doesn't combine them across N scorers |
| 8 | Prompt parity acceptance too soft | **Partially** | Medium | My Story 6 edit mentions prompt diff but doesn't make it a hard test |
| 9 | Public API / exports / registry discovery unfinished | **Yes** | Medium | Confirmed `scorers/__init__.py` exports only deterministic scorers; registry imports `instances/` but instances must self-import each module |

### Impact breakdown — what each one actually means

**#1 — "Near-verbatim" is misleading.** The listed files (`api_calls.py`, `evaluator.py`, `parsers.py`) return shapes that don't match the new contract:
- `api_calls.py` returns OpenAI Responses objects → the new `JudgeClient.acomplete` must return `JudgeResponse(text, input_tokens, output_tokens)`.
- `parsers.py` produces a flat dict with `<dim>_score` / `<dim>_reasoning` keys → the new contract is `list[ScorerResult]` (one per dim) plus an output-field dict.
- `evaluator.py` takes a `config: dict` and threads it everywhere → the new contract is a typed `JudgePanel` + a small set of config dataclasses.
This isn't a copy job. The hg-ds-evals files are *references*, not portable code. The plan should rephrase §6.2: "Re-implement the runner using these files as behavioural reference." Otherwise the team will copy code that the rest of the spec disagrees with, then have to back it out.

**#2 — `ascore(dimension)` semantics.** Three valid interpretations of "what happens if a caller passes a different `Dimension` than the bound one":
- **Hard error** (recommended) — passing a different dimension is a bug at the call site.
- **Override at call time** — risky: the prompt was built around the bound dimension; a fresh dimension would render the prompt incoherent.
- **Silent ignore** — worst of all: caller thinks they're changing the dimension; nothing happens.
Pick one (I'd pick *hard error*) and specify it in Story 1 or Story 3 acceptance.

**#3 — Cross-field rules.** Confirmed: skkb YAML lines 657-718 contain ~60 lines of cross-field rules (`Evaluation order`, `Selection relevance`, `Case-scope usage`, `Scope-aware scoring`, `Team separation in suggestions`, `Reference-defect rule`, `At-least-one-action rule`). These are panel-level instructions that govern how the LLM combines per-dimension judgments. They are not properties of individual `OutputField`s. The plan's v1 YAML §7.1 omits them; if Story 6 just lifts each output field's description, these rules vanish from the prompt and the LLM judges differently. Need a new home: most naturally, a `JudgePanel.cross_field_rules: str` field that renders into the system prompt alongside `critical_evaluation_rules` and `final_reminders`.

**#4 — Weights vs strict-canonical.** Plan §7.1 has:
```yaml
scorers:
  - id: selection_semantic_relevance
    weight: 1.5
  - id: answer_groundedness
    weight: 2.0
```
Plan §4.1 calls weight a (soft) override. Plan §4.3 says `--strict-canonical` refuses to run with any override. So this YAML would be rejected. Three ways out:
- **Canonical weights live in code.** Each scorer class has a default weight matching its skkb usage; YAML omits weights in strict mode.
- **Weights are panel-level, not scorer-level.** Move `weight` off `Dimension` entirely; put a `weights: dict[scorer_id, float]` on the panel. Audit the *panel composition* separately from the scorers.
- **"Pack" abstraction.** A named `skkb_baseline` panel pack is itself canonical (bundles scorers + weights + cross-field rules); YAML just selects the pack.
The third option is probably the cleanest. Pack name → canonical bundle of (scorers, weights, output fields, cross-field rules). Strict mode just checks the pack fingerprint.

**#5 — Dep isolation.** Verified: today's `evals/pyproject.toml` declares only `ai-data-science-core`. Adding the runner means:
- New `[project.optional-dependencies] runner = [...]`. Plan lists `openai, tenacity, pyspark, mlflow, databricks-sdk, pyyaml` — **add `pandas`** (checkpoint DataFrames; very hard to avoid).
- Lean-install consumers must not trigger a runner import on `import ai_data_science.evals`. That means **no top-level `from .runner import …` in any `__init__.py`** along the lean-package import path.
- Story 8 needs to add a CI check: `pip install ai-data-science-evals && python -c "import ai_data_science.evals; from ai_data_science.evals.scorers.llm_judge import LLMJudgeScorer"` in a clean venv with no `[runner]` extras must succeed. Today's plan asserts this conceptually but doesn't pin it to a CI job.
- mypy: today the root typecheck walks all `src/`. If `runner/` imports `pyspark`, mypy in dev needs the extras installed. Either install `[runner]` in dev (simplest) or add a `runner/`-scoped mypy override (`[[tool.mypy.overrides]] module = "ai_data_science.evals.runner.*"; ignore_missing_imports = true`). Pick one and put it in Story 8.

**#6 — Don't ad-hoc-recreate trace ingestion.** Upstream's `tools/trace-extractor/src/trace_extractor/extractor.py` already does the `client.search_traces(locations=…)` + JSON-serialize-spans dance (verified). My Story 12 puts `traces.py` + `mlflow_otel.py` into `_data_tmp/` near-verbatim, which silently duplicates that. Refinement:
- **Story 12 explicitly delegates MLflow trace fetching to `MLflowTraceExtractor`** (a dependency, not a copy). Only the OTel-table materialization + the eval-DataFrame-building (the "build_dataframe_from_mlflow_traces" pipeline that produces `expected_*` / `actual_*` columns) goes into `_data_tmp/`.
- That cleaves the work into a re-usable layer (upstream extractor → flat span JSON) and a transitional layer (span JSON → eval DataFrame). Only the transitional layer moves later.

**#7 — Panel missing-input semantics.** Today's `BaseScorer.validate_inputs` raises on missing reference fields and returns a list for missing prediction fields. In a panel:
- Row has `user_query` but no `agent_response`. The query-quality scorer can still run. The answer-quality scorers should *not* trust the LLM's score on a missing field.
- Three reasonable resolutions:
  1. **Pre-filter at panel level.** If *any* bound scorer's reference fields are missing → fail the row entirely (raise). If *some* scorer's prediction fields are missing → call the LLM but short-circuit those scorers to `scale.min`.
  2. **Per-scorer shortcuts after the call.** Make one LLM call, then per-scorer check fields and overwrite to `scale.min` if its prediction fields are missing.
  3. **Two-pass.** Compute the "viable scorers" subset first, build a reduced panel, then call the LLM with only those dimensions in the prompt.
- Option 1 is the simplest for v1; option 3 is the most token-efficient but adds complexity. Specify in Story 4 acceptance.

**#8 — Prompt parity.** "±0.1 mean score" is a behavioural check; what's missing is a structural check. Strengthen Story 6 acceptance to include a **snapshot test of the rendered system prompt**: take the current hg-ds-evals skkb YAML, render its system prompt; take the new v1 YAML, render its system prompt; diff section-by-section (persona, per-dim blocks, output JSON keys, cross-field rules, final reminders, user-field order). The diff should be exact-equal modulo whitespace.

**#9 — Public API & registry discovery.** Two concrete items for Story 5:
- Extend `evals/src/ai_data_science/evals/scorers/__init__.py` to also re-export the LLM-judge scorers (and `LLMJudgeScorer`, `JudgePanel`, `JudgeClient`). Lock these as stable import paths.
- The registry's `_ensure_instances_loaded()` (in `scorers/llm_judge/registry.py`) imports the `instances` package — that package's `__init__.py` must explicitly import every concrete scorer module so the `@register` decorators run. Story 5 should specify this and add a test: `list_scorers()` after fresh import returns exactly the seven skkb scorers.

---

## Other similar concerns worth flagging now

The pattern these critiques share: the plan describes *target behaviour* but not *contractual edges* (what happens at the seams, error paths, ambiguous inputs). Same lens applied to other parts of the design surfaces a few more:

**A. Token accounting shape.** `JudgeResponse` carries `input_tokens`/`output_tokens`. A panel produces N `ScorerResult`s from one LLM call. Where do the token counts go?
- On every `ScorerResult.metadata["llm_usage"]`? Duplicates the same number N times.
- Only on the panel-level metadata? Then individual `ScorerResult`s lose the cost trail.
- Both? Decide once, document. Today's hg-ds-evals tracks at the row level; the natural answer is panel-level. Specify in Story 4.

**B. Panel-level vs row-level error handling.** What happens if the LLM returns malformed JSON? Three options:
- All scorers get an error `ScorerResult(value=scale.min, rationale="parse_error")`.
- The panel emits one `PanelResult(error=…)` and individual scorer results are absent.
- Retry the LLM call once (today's behaviour) then fall back.
Decide in Story 3 (parser) / Story 4 (panel).

**C. Canonical fingerprint composition.** Story 5 mentions a checked-in `canonical_fingerprints.json`. The fingerprint hash must include: id, name, description, scale type, scale levels, every score_level (value+label+description), prompt_block, **and weight if weights are scorer-canonical**. If weights aren't in the hash but they're enforced under strict-canonical, the audit story has a hole. Tied to #4 above — decide weights first, then specify the fingerprint composition.

**D. Determinism in CI.** Story 11's "real DBX cluster, ±0.1 mean" is good for the milestone gate but bad for routine CI (slow, costly, nondeterministic). Add a **hermetic stub-LLM test** in Story 11 acceptance: a fixed-response `JudgeClient`, a 3-row fixture DataFrame, asserts the full result shape including override audit and run metadata.

**E. YAML schema validation.** Story 9 loader. With two input modes (run_id, table_name), output_fields, optional scorer weights, optional `cross_field_rules`, etc., the YAML surface grows. Use a JSON schema or a pydantic model — give errors at load time pointing to YAML line numbers. Plan doesn't say.

**F. `pass_threshold` and weights interact.** Per Q4 we kept `pass_threshold`. It only makes sense as a weighted aggregate; that requires weights to be unambiguous (see #4). Lock the resolution path: **decide #4 first, then `pass_threshold` falls out of it.**

**G. Async client shape and lifecycle.** Plan §6.2 says port `api_client.py` → `runner/api/client.py` and references a `get_api_client()` factory. But the factory currently constructs a *raw* OpenAI async client. The new `JudgeClient` is an instance wrapping that client. Story 2 should specify whether the public surface is `AzureDatabricksJudgeClient(...)` constructed by the caller, or a `make_judge_client(config) -> JudgeClient` factory consumed by the runner. Pick one — the latter is simpler for runner usage; the former is simpler for unit tests.

**H. `core` dep.** Story 7's lean install pulls `ai-data-science-core` + `jinja2`. I verified `core` is essentially empty (just `__init__.py`). Either `core` needs content this work depends on (Dimension/Scale could live there long-term), or the dep is currently a no-op and can stay so. Worth a line: "Story 0: confirm what belongs in core vs evals" — small but pre-requisite.

---

## Resolutions for the critiques (from chat)

| # | Decision | What it implies |
|---|---|---|
| **#4 — weights** | **Pack abstraction.** Named "panel packs" (e.g. `skkb_baseline`) bundle scorers + weights + cross-field rules + output fields. The *pack* is canonical and fingerprinted; YAML picks a pack by name. | New top-level concept: `JudgePanelPack`. New file: `scorers/llm_judge/packs/skkb_baseline.py` (or similar). YAML §7.1 shrinks further — `panel:` becomes mostly `pack: skkb_baseline` plus per-experiment additions. |
| **#3 — cross-field rules** | **Live on the pack** (via `JudgePanel.cross_field_rules: str`). They're panel-level instructions, not per-field. | The 60-line `additional_instructions` block from skkb YAML lines 657-718 moves into the skkb pack as a `cross_field_rules` string. Renders into the system prompt alongside `critical_evaluation_rules` and `final_reminders`. |
| **#7 — missing inputs** | **Pre-filter at panel level.** If any bound scorer's reference fields are missing → fail the row (raise). If any scorer's prediction fields are missing → call the LLM but short-circuit those scorers to `scale.min`. | `JudgePanel.ascore_row(**kwargs)` does the field check before calling the LLM. Per-scorer prediction-missing shortcuts apply after the call (or override the LLM's score for that specific scorer). |
| **#2 — dimension mismatch at `ascore`** | **Hard error.** Passing a Dimension that isn't the bound one raises. | Story 1 acceptance: `LLMJudgeScorer.ascore(dimension=other_dim)` raises `ValueError` with a message comparing the two by id. Override is *only* at construction time, never at call time. |

### Pack — recommended shape

Concretely, what a pack looks like (sketch; expand in the plan):

```python
@dataclass(frozen=True)
class JudgePanelPack:
    """Named, canonical bundle for one evaluation scenario."""

    id: str                                          # e.g. "skkb_baseline"
    version: str                                     # SemVer; bumps with any field change
    description: str

    # Scorer composition + per-scorer weight (canonical at the pack level)
    scorers: tuple[ScorerSpec, ...]                  # ScorerSpec = (scorer_id, weight)

    # Optional diagnostic fields
    output_fields: tuple[str, ...] = ()              # OutputField ids

    # Pack-level prompt sections (these survive into the system prompt)
    cross_field_rules: str = ""                      # ← skkb additional_instructions land here
    critical_evaluation_rules: str = ""
    domain_specific_guidance: str = ""
    final_reminders: str = ""
    persona: str = ""

    # Aggregation
    pass_threshold: float = 1.5                      # weighted avg
```

A **pack fingerprint** is `sha256(canonical_serialisation(pack))`. Strict-canonical mode just verifies `pack.fingerprint == checked_in_fingerprint`. Per-scorer fingerprints (for individual scorer drift) still exist; the pack fingerprint composes them.

The v1 YAML §7.1 becomes:

```yaml
panel:
  pack: skkb_baseline                    # picks scorers, weights, cross-field rules
  # nothing else; everything else lives in code under the pack
```

Per-experiment overrides (rare — must be explicitly opted in, breaks strict mode) would look like:

```yaml
panel:
  pack: skkb_baseline
  overrides:
    weights:
      answer_groundedness: 1.5          # was 2.0 → marks pack as overridden
    cross_field_rules_append: |
      ## Experiment-specific rules
      …
```

### Defaults I'm recommending for the remaining open items

Rather than asking more questions, here are my proposed defaults for items A, B, D, E, G, H from the "Other concerns" list. Push back on any of these and we'll revisit.

| Item | Recommendation |
|---|---|
| **A — Token accounting shape** | Put `input_tokens` / `output_tokens` on **panel-level metadata only**, not duplicated on each `ScorerResult`. Per-scorer `metadata["llm_usage"]` gets a single shared dict reference if downstream really needs per-scorer access. Today's hg-ds-evals already aggregates at the row level — this matches. |
| **B — Malformed-JSON error path** | One retry of the LLM call (today's `max_parse_retries` behaviour); if still malformed, emit a `PanelResult` with `error="parse_error"` and every scorer in the panel gets `ScorerResult(value=scale.min, rationale="upstream parse error", metadata={"parse_error": ...})`. Don't silently emit partial results. |
| **D — Determinism in CI** | Story 11 acceptance gets a second clause: a **hermetic stub-LLM test** with a fixed-response `JudgeClient`, a 3-row fixture DataFrame, asserts the full result shape including override audit. This runs in every CI build. The real-DBX run is a separate milestone-gate test. |
| **E — YAML schema validation** | Pydantic v2 models for the YAML schema (`ExperimentConfig`, `ModelConfig`, `DatasetConfig`, `PanelConfig`). Errors at load time point at YAML line numbers via `pydantic.ValidationError`. Add `pydantic` to the `[runner]` extras. |
| **G — Async client shape** | Public surface is the `AzureDatabricksJudgeClient` class (constructed by callers and runner alike). No factory function. Tests construct it with a stub HTTP transport; runner constructs it from `ModelConfig`. Keep it boring. |
| **H — `core` dep** | Today `core` is empty. Don't move anything there speculatively. Keep evals' lean install depending on `core` for the future shared bits, but the v1 work doesn't put anything in `core` itself. |

---

## Concrete edits to the plan (consolidated)

With the four resolutions + defaults above, here's the full list of edits to `llm_judge_scorers_plan.md`:

### Rewrites

- **§6.2** — replace "*What we port from `hg-ds-evals`, near-verbatim*" with "*What we re-implement using `hg-ds-evals` as behavioural reference*". List the same files but make clear they're references, not portable code. `parsers.py` in particular is rewritten — the new contract is `list[ScorerResult]` plus an output-field dict, not a flat row dict.
- **§6.4 (out-of-scope)** — remove "*Inline `mlflow.search_traces()` ingestion (v2 — for v1, expect a pre-materialized table)*". v1 accepts both run ID and table name per Q1.
- **§7.1 (v1 YAML)** — shrink further per the pack abstraction. The 80-line YAML drops to ~30 lines (`experiment`, `model`, `dataset`, `paths`, `panel: { pack: skkb_baseline }`).

### New sections

- **§3.5 (new) — Pack abstraction.** Defines `JudgePanelPack`, the pack registry, and how YAML resolves to a pack. Covers fingerprint composition.
- **§4.5 (new) — Missing-input handling.** Documents the pre-filter rule from #7.
- **§5.4 (new) — Cross-field rules.** Documents that cross-field rules live on the pack as `cross_field_rules`, not on `OutputField`.

### Story changes

- **Story 1** — add acceptance: `ascore(dimension=other)` raises `ValueError`.
- **Story 4 (JudgePanel)** — add acceptance: pre-filter missing inputs (raise on reference; short-circuit on prediction). Token accounting on panel-level metadata only. Malformed-JSON retry-then-error path.
- **Story 5** — add a "skkb_baseline pack" alongside the seven scorer ports. Include cross-field rules. Checked-in `canonical_fingerprints.json` includes the *pack* fingerprint, not just per-scorer.
- **Story 6** — strengthen acceptance: snapshot test of the rendered system prompt (current hg-ds-evals skkb prompt vs new pack-driven prompt, section-by-section diff modulo whitespace).
- **Story 7** — explicit CI smoke: lean install + `python -c "import ai_data_science.evals; from ai_data_science.evals.scorers.llm_judge import LLMJudgeScorer, JudgePanel"`.
- **Story 8** — declare `[runner]` extras including `pandas` and `pydantic`. Document the lazy-import rule (no top-level `from .runner import …` in any lean-package `__init__.py`). Decide mypy policy: either install `[runner]` in dev (recommended) or scope-exempt the runner module.
- **Story 9 (YAML loader)** — uses pydantic models; pack resolution; line-numbered errors.
- **Story 11** — add hermetic stub-LLM test alongside the real-DBX milestone gate.
- **Story 12 (new — added in previous iteration)** — refine: **delegate MLflow trace fetching to upstream's `MLflowTraceExtractor`** (a dep, not a copy). Only the OTel-table-materialisation + eval-DataFrame-build code from hg-ds-evals lands in `_data_tmp/`.
- **Story 13 (new — added in previous iteration)** — unchanged: lift-and-shift `api_utils.py` to `_scorers_tmp/`.
- **Story 14 (new — added in previous iteration)** — unchanged: v2 cleanup placeholder.
- **`scorers/__init__.py`** — extended to export `LLMJudgeScorer`, `JudgePanel`, `JudgeClient`, and the seven LLM-judge scorer classes. Stable import paths locked.

---

## Next steps

1. **Confirm the resolutions above** (or push back on the recommended defaults for A/B/D/E/G/H).
2. **Update the plan**: with everything settled, I can rewrite `llm_judge_scorers_plan.md` against the resolutions — preserve the existing structure but apply all the edits in one pass. Or you can do this and use this review as the change-list.
3. **Pre-kickoff sanity check** worth doing once the plan is updated: take the *current* skkb YAML, the *new* v1 YAML, and the *new* pack — render the three system prompts side-by-side and verify the pack faithfully captures the current prompt's content. This is a one-hour exercise that will save weeks if a section was missed.

