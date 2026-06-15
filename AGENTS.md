# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What this repo is

`hg-ds-evals` is a **reference-free LLM-as-Judge evaluation framework** for the Hey George conversational AI. There are no ground-truth labels — the judge LLM scores production traces against rubric dimensions defined in YAML experiment configs. The library is designed to run on Databricks (Spark + Azure OpenAI / Databricks model serving) but every report/notebook in `experiments/` also has a "local" variant.

Top-level: `README.md` has the canonical architecture diagram and walkthrough — read it before designing any change to the eval pipeline.

## Environment & commands

`.venv/` (Python 3.10+) at the repo root is the canonical local interpreter — invoke it as `/Users/SG7CB/Developer/hg-ds-evals/.venv/bin/python` (or `.venv/bin/python` from the repo root). Notebooks under `experiments/*/notebooks/` reach it as `../../../.venv/bin/python`. `requirements.txt` is intentionally minimal (pandas + pytest); heavyweight deps (`pyspark`, `ds_common`, `tenacity`, `openai`, `anthropic`, `sentence-transformers`) come from the Databricks runtime or are `pip install`-ed ad-hoc into `.venv`.

```bash
# Run the full unit-test suite (only one file today — span/trace parsing)
.venv/bin/python -m pytest tests/ -x -q

# Run a single test
.venv/bin/python -m pytest tests/test_skkb_traces.py::test_build_skkb_dataframe_uses_test_case_id_from_assessment -v

# Regenerate an experiment HTML report (CZKB example — see "Report filename" memory)
cd experiments/czkb/notebooks
../../../.venv/bin/python czkb_report.py \
    --checkpoint checkpoints/<exp_002_checkpoint.csv> \
    --output reports/czkb_baseline_test_002.html \
    --include-latency

# Regenerate the API smoke report
cd experiments/api/notebooks
../../../.venv/bin/python api_report.py \
    --input traces/<enriched_traces.csv> \
    --output reports/api_smoke_report.html
```

`databricks.yml` defines the asset bundle target for deployment to `adb-3174992876438447.7.azuredatabricks.net`; the package is installed on Databricks via the editable `ds_common` workspace path declared in `pyproject.toml`.

## Architecture — the bits that span files

### Pipeline shape (defined by `evals/run_evals.py::run_experiment`)

`run_experiment(yaml_path)` is the single entry point. It auto-detects the YAML format:

- **Experiment YAML** (has `rubric.base`) → system prompt is built programmatically from the rubric definition via `PromptBuilder`.
- **Legacy YAML** (has `paths.system_prompt_path`) → reads a static `.md` prompt file. Treat this as deprecated — prefer migrating to the experiment format.

The 6-step pipeline: load config → load Spark table → sample + checkpoint resolution → load checkpoint (resumability) → build Azure/Databricks async client → `async_run_evals()` with TPM-aware concurrency. **The eval loop saves to the checkpoint CSV row-by-row**, so an interrupted run resumes exactly where it stopped on the next invocation.

### Rubrics are composed, not edited

`rubrics/base.py` defines immutable dataclasses (`Rubric`, `Dimension`, `ScoreLevel`, `OutputField`, `OutputSchema`). Operations like `add_dimension`, `with_weight` return new instances. `rubrics/dimensions/catalog.py` is the shared dimension library; `rubrics/fallback.py` exports `FALLBACK_RUBRIC` / `FALLBACK_FULL_RUBRIC` used as `base:` references in experiment YAMLs. `rubrics/loader.py` parses YAML → `Rubric`; any section in the YAML (`dimensions`, `input_fields`, `output_schema`) is an optional override — omit it to inherit base defaults.

### Prompts are Jinja2 templates fed by Rubric fields

`PromptBuilder` (in `prompts/builder.py`) renders system + user prompts from a `Rubric` using Jinja2. Embedded defaults live in the code; per-experiment overrides go in `experiments/<eval_type>/configs/system.md.j2` (and optionally `user.md.j2`), referenced via `paths.system_template_path` / `paths.user_template_path` in the YAML. **Adding a dimension to a rubric automatically extends the rendered prompt** — do not duplicate the dimension into the template.

### Trace preprocessing — single source of truth lives in `preprocessing/traces.py`

The unified module replaces the legacy `skkb_traces` / `mlflow_traces` files (which still exist as thin re-export shims). Two public entry points:

- `build_skkb_dataframe_from_mlflow_search_traces(...)` → SKKB / CZKB KB-pipeline rows (retrieval, prune, reranker, query-scope, invariants, prompt-hash).
- `build_dataframe_from_mlflow_traces(...)` → canonical eval-scoring rows with `expected_*` / `actual_*` columns matching the `ai-data-science` scorer contract.

**Agent final answer**: comes **only** from the `agent_answer` AGENT span (`mlflow.spanOutputs.answer`). Missing span ⇒ empty string. The old "longest stopped AI message" heuristic was deliberately removed — it silently picked sub-agent reasoning. Coverage is now honest. There is a known orchestrator regression that broke `agent_answer` emission (PR #388); fix is pending on Peter's PR — **do not patch `traces.py` to compensate**.

**Span-level errors**: `info.state == "OK"` is unreliable. To detect failed spans, parse `status.code` and look for `events[]` where `name == "exception"`. Helpers `_span_exception_events`, `_is_span_status_error`, and `extract_span_errors` in `preprocessing/traces.py` do this correctly — reuse them rather than re-reading `info.state`.

### Async LLM calls

`llm/api_calls.py` handles Azure OpenAI and Databricks model-serving calls. Per-model param shaping is in `build_api_params` — e.g., `temperature` for gpt-4o/4.1, `reasoning.effort` for gpt-5/gpt-5-nano. Concurrency is bounded by `asyncio.Semaphore(max_concurrent_calls)`; after each batch, the engine measures token usage and sleeps to respect the model's TPM limit (sourced from `ds_common.llm_models_config`, overridable via `model.tpm_limit` in the YAML). Retries are `tenacity` exponential backoff on `RateLimitError` and `APITimeoutError`.

### Experiment layout convention

```
experiments/<eval_type>/
    configs/                       # YAML configs + optional .j2 template overrides
    notebooks/                     # Databricks notebooks + .py report generators
        <type>_report.py           # Standalone HTML report from a checkpoint CSV
        <type>_001_baseline.ipynb  # DBX run
        <type>_001_baseline_local.ipynb  # Local re-run from pickle
        checkpoints/               # Generated by runs — gitignored
        reports/                   # Generated HTML — gitignored
```

The YAML filename stem **is** the experiment id — it drives checkpoint filenames (`evals_<stem>_<reasoning_effort>_<...>.csv`) and result-table naming. Use `<eval_type>_exp_<NNN>_<short_description>.yaml`.

## Critical conventions for this codebase

- **Never overwrite the user's input data files.** No writes to enriched CSVs / pickles / checkpoints under `experiments/*/input/` or `experiments/*/notebooks/checkpoints/`. The user runs the data pipeline; assistant edits to those files break reproducibility.
- **Report HTML files are overwritten in place — no `_v2` / `_v3` suffixes.** Specifically: `experiments/api/notebooks/reports/api_smoke_report.html` and `experiments/czkb/notebooks/reports/czkb_baseline_test_002.html` are the canonical paths.
- **`ds_common` is an editable install from a Databricks workspace path** (`/Workspace/Repos/Shared_HeyGeorge/hey-george-ds/ds_common`). Locally it's importable via the `.venv`. Constants like `HGCol`, `HGTbl`, `print_emoji` come from `ds_common.config.config`.
- **`input/` (repo root) and `experiments/**/input/`, `**/reports/`, `**/checkpoints/`, `*.csv|jsonl|pkl|pickle|parquet`** are all gitignored — never check them in.
- **Notebooks (`.ipynb`)** are committed; report scripts (`*_report.py`) and `backfill_*.py` in notebook dirs are also committed and intended to be run from their notebook directory.
