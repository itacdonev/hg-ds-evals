#!/usr/bin/env python
"""Run the API trace evaluation flow from a terminal or notebook.

This script is the CLI version of the two local API notebooks:

1. ``experiments/api/notebooks/import_traces_local.ipynb``
2. ``experiments/api/notebooks/api_001_baseline_local.ipynb``

For every requested MLflow run it:

* downloads and parses traces,
* runs the deterministic API scorers,
* writes ``enriched_traces_<run_id>.csv`` for the reports,
* optionally runs the LLM language judge,
* writes one HTML report per run,
* and, when more than one run is supplied, writes a multi-run comparison.

The committed YAML config is never edited. Per-run YAML values are changed in
memory so each judge run points at the right MLflow run and input table name.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import os
import re
import shutil
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


DEFAULT_API_YAML = "api_exp_001_language"
DEFAULT_DATABRICKS_PROFILE = "adb-uat"
DEFAULT_INPUT_TRACES_DIR = "~/Developer/input_traces"
DEFAULT_EVALS_SRC = "/Users/SG7CB/Developer/ai-data-science/evals/src"
SUPERVISOR_AGENT = "main_agent"


@dataclass(frozen=True)
class RunnerPaths:
    repo_root: Path
    api_dir: Path
    api_notebooks_dir: Path
    api_configs_dir: Path
    input_traces_dir: Path
    reports_dir: Path


@dataclass
class PreparedRun:
    run_id: str
    enriched_path: Path
    judge_checkpoint_path: Path | None = None
    report_path: Path | None = None


def _find_repo_root(start: Path | None = None) -> Path:
    start_path = (start or Path(__file__).resolve()).resolve()
    candidates = [start_path] + list(start_path.parents)
    for candidate in candidates:
        if (candidate / "hg_ds_evals").is_dir() and (candidate / "experiments").is_dir():
            return candidate
    raise RuntimeError(f"Could not find repo root from {start_path}")


def _build_paths(
    *,
    input_traces_dir: str | Path = DEFAULT_INPUT_TRACES_DIR,
    reports_dir: str | Path | None = None,
) -> RunnerPaths:
    repo_root = _find_repo_root()
    api_dir = repo_root / "experiments" / "api"
    api_notebooks_dir = api_dir / "notebooks"
    api_configs_dir = api_dir / "configs"
    resolved_input_dir = Path(input_traces_dir).expanduser().resolve()
    resolved_reports_dir = (
        Path(reports_dir).expanduser().resolve()
        if reports_dir
        else api_notebooks_dir / "reports"
    )
    return RunnerPaths(
        repo_root=repo_root,
        api_dir=api_dir,
        api_notebooks_dir=api_notebooks_dir,
        api_configs_dir=api_configs_dir,
        input_traces_dir=resolved_input_dir,
        reports_dir=resolved_reports_dir,
    )


def _ensure_sys_path(paths: RunnerPaths, evals_src: str | Path | None) -> None:
    for path in (paths.repo_root, paths.api_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    if not evals_src:
        return
    evals_path = Path(evals_src).expanduser()
    if evals_path.is_dir() and str(evals_path) not in sys.path:
        sys.path.insert(0, str(evals_path))


def _split_cli_values(values: Sequence[Any] | None) -> list[str]:
    """Turn repeated CLI args and comma-separated values into a clean list."""
    if not values:
        return []

    flattened: list[str] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            flattened.extend(_split_cli_values(value))
            continue
        for piece in str(value).split(","):
            clean = piece.strip()
            if clean:
                flattened.append(clean)
    return flattened


def _as_list(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        return _split_cli_values([value])
    return _split_cli_values(list(value))


def _resolve_baselines(run_ids: list[str], baseline_ids: list[str] | None) -> list[str]:
    if not baseline_ids:
        return list(run_ids)
    if len(baseline_ids) == 1:
        return [baseline_ids[0] for _ in run_ids]
    if len(baseline_ids) == len(run_ids):
        return list(baseline_ids)
    raise ValueError(
        "RUN_BASELINE must be omitted, a single run id, or the same length as RUN_ID."
    )


def _sanitize_filename_piece(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return cleaned.strip("._") or "report"


def _short_run_id(run_id: str) -> str:
    return _sanitize_filename_piece(run_id)[:10]


def _single_report_path(
    reports_dir: Path,
    report_name: str,
    run_id: str,
    baseline_id: str,
) -> Path:
    name = _sanitize_filename_piece(report_name)
    return reports_dir / f"{name}_{_short_run_id(run_id)}_vs_{_short_run_id(baseline_id)}.html"


def _multi_report_path(reports_dir: Path, report_name: str) -> Path:
    return reports_dir / f"api_multi_report_{_sanitize_filename_piece(report_name)}.html"


def _checkpoint_suffix_for_run(
    checkpoint_suffix: str | None,
    report_name: str,
    run_id: str,
    run_index: int,
    *,
    unique_default: bool = False,
) -> str:
    if checkpoint_suffix is not None:
        raw_suffix = checkpoint_suffix
    elif unique_default:
        raw_suffix = f"{report_name}_{_short_run_id(run_id)}"
    else:
        raw_suffix = report_name
    return (
        raw_suffix.replace("{run_id}", _sanitize_filename_piece(run_id))
        .replace("{short_run_id}", _short_run_id(run_id))
        .replace("{run_index}", str(run_index))
    )


def _api_config_path(paths: RunnerPaths, yaml_file_name: str) -> Path:
    name = yaml_file_name if yaml_file_name.endswith(".yaml") else f"{yaml_file_name}.yaml"
    path = paths.api_configs_dir / name
    if not path.exists():
        raise FileNotFoundError(f"YAML config file does not exist: {path}")
    return path


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML did not load as a dictionary: {path}")
    return payload


def _resolve_relative_to_notebook_dir(path_value: str | Path, paths: RunnerPaths) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else paths.api_notebooks_dir / path


def _resolve_template_path(raw_path: str | None, config_dir: Path) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    candidates = [path] if path.is_absolute() else [config_dir / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _configure_mlflow_auth(profile: str, *, on_databricks: bool = False) -> None:
    import mlflow

    if on_databricks:
        print("Running on Databricks: using attached workspace credentials.")
        return

    for var in (
        "DATABRICKS_AUTH_TYPE",
        "DATABRICKS_METADATA_SERVICE_URL",
        "DATABRICKS_HOST",
        "DATABRICKS_CLUSTER_ID",
    ):
        os.environ.pop(var, None)
    os.environ["DATABRICKS_CONFIG_PROFILE"] = profile

    from databricks.sdk import WorkspaceClient

    workspace = WorkspaceClient()
    print(f"Authenticated as: {workspace.current_user.me().user_name}")
    print(f"Workspace host:   {workspace.config.host}")

    mlflow.set_tracking_uri("databricks")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")


def _fetch_traces(
    *,
    experiment_id: str,
    run_id: str,
    max_results: int | None,
) -> pd.DataFrame:
    import mlflow

    logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
    traces_df = mlflow.search_traces(
        locations=[experiment_id],
        run_id=run_id,
        max_results=max_results,
        order_by=["timestamp_ms DESC"],
    )
    print(f"Fetched {len(traces_df):,} trace rows for run {run_id}.")
    return traces_df


def _merge_kb_diagnostics(trace_level_df: pd.DataFrame, traces_df: pd.DataFrame) -> pd.DataFrame:
    from hg_ds_evals.preprocessing.traces import build_skkb_dataframe_from_mlflow_search_traces

    skkb_df = build_skkb_dataframe_from_mlflow_search_traces(traces_df).dataframe
    if "trace_id" not in skkb_df.columns or "trace_id" not in trace_level_df.columns:
        return trace_level_df
    kb_columns = [
        "query_scope",
        "knowledge_search_run_count",
        "kb_version",
        "pre_prune_enum_ids",
        "pre_prune_enum_count",
        "post_prune_enum_ids",
        "post_prune_enum_count",
        "reranked_enum_ids",
        "reranker_raw_selected_ids",
        "reranker_selection_status",
        "reranker_selection_violations",
    ]
    keep = ["trace_id"] + [column for column in kb_columns if column in skkb_df.columns]
    if len(keep) == 1:
        return trace_level_df
    merged = trace_level_df.merge(
        skkb_df[keep],
        on="trace_id",
        how="left",
        validate="one_to_one",
    )
    n_kb = int((merged.get("query_scope") == "kb").sum()) if "query_scope" in merged else 0
    print(f"Merged {len(keep) - 1} KB diagnostic columns; query_scope='kb' rows: {n_kb}")
    return merged


def _is_kb_knowledge_search_only(row: pd.Series) -> bool:
    if row.get("eval_domain") != "kb":
        return False
    expected = row.get("expected_tool_calls")
    if not isinstance(expected, list) or not expected:
        return False
    return all(
        isinstance(entry, dict)
        and (entry.get("tool") or entry.get("name") or "").lower() == "knowledge_search"
        for entry in expected
    )


def _has_no_expected_tool_calls(row: pd.Series) -> bool:
    expected = row.get("expected_tool_calls")
    if not (isinstance(expected, list) and len(expected) == 0):
        return False
    scorers = row.get("scorers_to_run")
    return isinstance(scorers, list) and bool(scorers)


def _augment_scorers_to_run(df: pd.DataFrame) -> pd.DataFrame:
    if "scorers_to_run" not in df.columns:
        raise RuntimeError("Parsed traces do not have a 'scorers_to_run' column.")

    with_scorers = df["scorers_to_run"].apply(lambda xs: isinstance(xs, list) and bool(xs)).sum()
    if with_scorers == 0:
        raise RuntimeError(
            "scorers_to_run is empty for all rows. The traces likely do not have "
            "the HUMAN scorer assessments attached."
        )
    if with_scorers < len(df) * 0.5:
        print(
            f"WARNING: only {with_scorers}/{len(df)} rows have non-empty "
            "scorers_to_run before augmentation."
        )

    def _row_scorers(row: pd.Series) -> list[str]:
        scorers = list(row.get("scorers_to_run") or [])
        if _is_kb_knowledge_search_only(row) or _has_no_expected_tool_calls(row):
            return [scorer for scorer in scorers if scorer != "tool_parameter"]
        return scorers

    out = df.copy()
    out["scorers_to_run"] = out.apply(_row_scorers, axis=1)

    no_tool_ids = out.loc[out.apply(_has_no_expected_tool_calls, axis=1), "test_case_id"].tolist()
    print(f"Rows with empty expected_tool_calls (tool_parameter dropped): {len(no_tool_ids)}")
    if no_tool_ids:
        preview = ", ".join(map(str, no_tool_ids[:10]))
        suffix = " ..." if len(no_tool_ids) > 10 else ""
        print(f"  ids: {preview}{suffix}")

    exploded = out["scorers_to_run"].explode()
    empty_rows = int(out["scorers_to_run"].apply(lambda xs: not xs).sum())
    scorer_counts = exploded.dropna().value_counts().to_dict()
    print(f"After augmentation, scorers_to_run distribution across {len(out)} rows:")
    print(f"  rows with NO scorers   : {empty_rows}")
    for scorer, count in sorted(scorer_counts.items()):
        print(f"  {scorer:>16s}: {count}")
    return out


def _canon_agent(name: object) -> str:
    return str(name).strip() or SUPERVISOR_AGENT


def _build_scorer_registry(
    *,
    apply_parameter_equivalence: bool,
) -> tuple[dict[str, tuple[object, object, Callable[[pd.Series], dict[str, Any]]]], type]:
    from ai_data_science.evals.dimension import Dimension
    from ai_data_science.evals.scales import BinaryScale, NumericScale
    from ai_data_science.evals.scorers.deterministic import (
        RoutingCorrectnessScorer,
        ToolParameterScorer,
        ToolUsageScorer,
    )
    from ai_data_science.evals.types import ScoreLevel
    from hg_ds_evals.evals.api_utils import tool_parameter_kwargs_from_row
    from hg_ds_evals.preprocessing.mlflow_traces import KNOWN_AGENT_NAMES

    binary_dim = Dimension(
        id="dim_binary",
        name="Binary",
        description="0 = wrong, 1 = correct",
        scale=BinaryScale(),
        score_levels=(
            ScoreLevel(value=0, label="fail", description="Incorrect"),
            ScoreLevel(value=1, label="pass", description="Correct"),
        ),
    )
    tool_parameter_dim = Dimension(
        id="dim_tool_params",
        name="Tool Parameters",
        description="Did the agent pass the right arguments to the tools it called?",
        scale=NumericScale([0.0, 1.0]),
        score_levels=(
            ScoreLevel(value=0.0, label="fail", description="No expected params correct"),
            ScoreLevel(value=1.0, label="pass", description="All expected params correct"),
        ),
    )

    registry = {
        "agent_routing": (
            RoutingCorrectnessScorer(available_agents=tuple(KNOWN_AGENT_NAMES)),
            binary_dim,
            lambda row: {
                "expected_agent": _canon_agent(row.get("expected_agent")),
                "actual_agent": _canon_agent(row.get("actual_agent")),
            },
        ),
        "tool_usage": (
            ToolUsageScorer(
                mode="exact",
                order_sensitive=True,
                case_sensitive=False,
                ignore_failed_calls=False,
            ),
            binary_dim,
            lambda row: {
                "expected_tool_calls": row.get("expected_tool_calls") or [],
                "actual_tool_calls": row.get("actual_tool_calls") or [],
                "available_tools": row.get("available_tools") or [],
            },
        ),
        "tool_parameter": (
            ToolParameterScorer(case_sensitive=False, value_coercion="string"),
            tool_parameter_dim,
            lambda row: tool_parameter_kwargs_from_row(
                row,
                equivalence_enabled=apply_parameter_equivalence,
            ),
        ),
    }
    return registry, NumericScale


_METADATA_FLATTENERS: dict[str, dict[str, Callable[[dict[str, Any]], Any]]] = {
    "tool_usage": {
        "correct": lambda m: (m.get("tool_classification") or {}).get("correct", []),
        "incorrect": lambda m: (m.get("tool_classification") or {}).get("incorrect", []),
        "hallucinated": lambda m: (m.get("tool_classification") or {}).get("hallucinated", []),
        "missing_expected": lambda m: (m.get("tool_classification") or {}).get("missing_expected", []),
    },
    "tool_parameter": {
        "key_score": lambda m: m.get("key_score"),
        "value_score": lambda m: m.get("value_score"),
        "expected_keys": lambda m: (m.get("totals") or {}).get("expected_keys"),
        "matched_keys": lambda m: (m.get("totals") or {}).get("matched_keys"),
        "correct_values": lambda m: (m.get("totals") or {}).get("correct_values"),
        "wrong_values": lambda m: (m.get("totals") or {}).get("wrong_values"),
        "missing_keys": lambda m: (m.get("totals") or {}).get("missing_keys"),
        "extra_by_tool": lambda m: m.get("extra_invocation_count_by_tool") or {},
        "per_entry": lambda m: m.get("per_entry_results") or [],
    },
}


def _serialize_meta_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _score_one_row(
    row: pd.Series,
    scorer_name: str,
    registry: Mapping[str, tuple[object, object, Callable[[pd.Series], dict[str, Any]]]],
) -> tuple[Any, str, str, dict[str, Any]]:
    entry = registry.get(scorer_name)
    if entry is None:
        return (None, f"unknown scorer: {scorer_name!r}", "unknown_scorer", {})
    scorer, dimension, build_kwargs = entry
    try:
        result = scorer.score(dimension, **build_kwargs(row))
    except Exception as exc:
        return (None, f"scorer raised: {exc!r}", "scorer_exception", {})
    meta = result.metadata or {}
    return (result.value, result.rationale or "", meta.get("status", "ok"), dict(meta))


def _score_all_rows(
    df: pd.DataFrame,
    *,
    apply_parameter_equivalence: bool,
) -> pd.DataFrame:
    from hg_ds_evals.evals.api_utils import compute_tool_parameter_equivalence

    registry, numeric_scale_cls = _build_scorer_registry(
        apply_parameter_equivalence=apply_parameter_equivalence,
    )
    out = df.copy()
    relax_fired: Counter[str] = Counter()

    def _compute_relaxation(row: pd.Series) -> dict[str, Any]:
        result = compute_tool_parameter_equivalence(
            row.get("expected_tool_calls") or [],
            row.get("actual_tool_calls") or [],
            eval_persona=row.get("eval_persona"),
            personas_without_product_filter=set(),
        )
        relax_fired.update(result.get("rule_counts") or {})
        return result

    if apply_parameter_equivalence:
        relax_diags = out.apply(_compute_relaxation, axis=1)
        out["_relaxation"] = relax_diags
        out["tool_parameter_expected_excused"] = relax_diags.apply(
            lambda diag: diag["expected_excused"]
        )
        out["tool_parameter_actual_excused"] = relax_diags.apply(
            lambda diag: diag["actual_excused"]
        )
    else:
        out["tool_parameter_expected_excused"] = [
            [[] for _ in (row.get("expected_tool_calls") or [])]
            for _, row in out.iterrows()
        ]
        out["tool_parameter_actual_excused"] = [
            [[] for _ in (row.get("actual_tool_calls") or [])]
            for _, row in out.iterrows()
        ]

    all_scorers = sorted({s for scorers in out["scorers_to_run"] for s in (scorers or [])})
    for scorer_name in all_scorers:
        out[f"{scorer_name}_score"] = None
        out[f"{scorer_name}_rationale"] = ""
        out[f"{scorer_name}_status"] = ""
        for suffix in _METADATA_FLATTENERS.get(scorer_name, {}):
            out[f"{scorer_name}_{suffix}"] = None

    for index, row in out.iterrows():
        for scorer_name in row.get("scorers_to_run") or []:
            value, rationale, status, meta = _score_one_row(row, scorer_name, registry)
            out.at[index, f"{scorer_name}_score"] = value
            out.at[index, f"{scorer_name}_rationale"] = rationale
            out.at[index, f"{scorer_name}_status"] = status
            for suffix, flattener in _METADATA_FLATTENERS.get(scorer_name, {}).items():
                out.at[index, f"{scorer_name}_{suffix}"] = _serialize_meta_value(
                    flattener(meta)
                )

    if apply_parameter_equivalence and relax_fired:
        print("Parameter equivalence summary:")
        for rule, count in relax_fired.most_common():
            print(f"  {count:4d}  {rule}")
    elif apply_parameter_equivalence:
        print("Parameter equivalence: enabled, but no rules fired on this run.")

    print("Per-scorer summary:")
    for scorer_name in all_scorers:
        score_col = f"{scorer_name}_score"
        status_col = f"{scorer_name}_status"
        scored = int(out[score_col].notna().sum())
        statuses = out[status_col].value_counts().to_dict()
        entry = registry.get(scorer_name)
        if entry is not None and isinstance(entry[1].scale, numeric_scale_cls):
            values = pd.to_numeric(out[score_col], errors="coerce")
            print(
                f"  {scorer_name:>16s}: scored={scored:3d}  mean={values.mean():.3f}  "
                f"pass(1.0)={(values == 1.0).sum():3d}  "
                f"partial={((values > 0) & (values < 1)).sum():3d}  "
                f"fail(0.0)={(values == 0.0).sum():3d}  statuses={statuses}"
            )
        else:
            print(
                f"  {scorer_name:>16s}: scored={scored:3d}  "
                f"correct={(out[score_col] == 1).sum():3d}  "
                f"wrong={(out[score_col] == 0).sum():3d}  statuses={statuses}"
            )

    return out


def _add_latency_seconds(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "execution_duration_ms" in out.columns:
        duration_ms = pd.to_numeric(out["execution_duration_ms"], errors="coerce")
        out["execution_duration_s"] = duration_ms / 1000
        out["execution_duration_min"] = duration_ms / (1000 * 60)
    return out


def _write_scored_csvs(df: pd.DataFrame, run_id: str, paths: RunnerPaths) -> Path:
    paths.input_traces_dir.mkdir(parents=True, exist_ok=True)
    csv_df = df.drop(columns=["_relaxation"], errors="ignore")
    scored_path = paths.input_traces_dir / f"scored_traces_{run_id}.csv"
    enriched_path = paths.input_traces_dir / f"enriched_traces_{run_id}.csv"
    csv_df.to_csv(scored_path, index=False)
    csv_df.to_csv(enriched_path, index=False)
    print(f"Wrote scored CSV:   {scored_path}")
    print(f"Wrote enriched CSV: {enriched_path}")
    return enriched_path


def _prepare_deterministic_run(
    *,
    run_id: str,
    experiment_id: str,
    paths: RunnerPaths,
    max_results: int | None,
    apply_parameter_equivalence: bool,
) -> PreparedRun:
    from hg_ds_evals.preprocessing.mlflow_traces import build_dataframe_from_mlflow_traces
    from hg_ds_evals.preprocessing.traces import write_prompt_sidecar

    print("\n" + "=" * 80)
    print(f"DETERMINISTIC SCORING: {run_id}")
    print("=" * 80)

    traces_df = _fetch_traces(
        experiment_id=experiment_id,
        run_id=run_id,
        max_results=max_results,
    )
    paths.input_traces_dir.mkdir(parents=True, exist_ok=True)
    raw_path = paths.input_traces_dir / f"raw_traces_{run_id}.csv"
    traces_df.to_csv(raw_path, index=False)
    print(f"Wrote raw traces: {raw_path}")

    parse_result = build_dataframe_from_mlflow_traces(traces_df)
    trace_level_df = parse_result.dataframe
    print(f"Parsed rows:  {len(trace_level_df):,}")
    print(f"Parse errors: {len(parse_result.parse_errors):,}")
    for parse_error in parse_result.parse_errors[:5]:
        print(f"  parse error {parse_error.trace_id}: {parse_error.error}")

    trace_level_df = _merge_kb_diagnostics(trace_level_df, traces_df)
    sidecar_path = write_prompt_sidecar(traces_df, run_id, paths.input_traces_dir)
    print(f"Wrote prompt sidecar: {sidecar_path}")

    trace_level_df = _augment_scorers_to_run(trace_level_df)
    trace_level_df = _score_all_rows(
        trace_level_df,
        apply_parameter_equivalence=apply_parameter_equivalence,
    )
    trace_level_df = _add_latency_seconds(trace_level_df)
    enriched_path = _write_scored_csvs(trace_level_df, run_id, paths)
    return PreparedRun(run_id=run_id, enriched_path=enriched_path)


def _move_existing_checkpoint(checkpoint_path: Path) -> Path | None:
    if not checkpoint_path.exists():
        return None
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    backup_path = checkpoint_path.with_suffix(f".csv.bak.{timestamp}")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(checkpoint_path), str(backup_path))
    return backup_path


def _run_async_from_sync(coro):
    """Run an async job from scripts and from notebooks.

    Plain ``asyncio.run`` fails when a notebook already owns the event loop.
    In that case we run the coroutine in a short-lived thread with its own
    event loop, so the public ``run_api_evaluation`` function stays easy to
    call from both places.
    """
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop and running_loop.is_running():
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(lambda: asyncio.run(coro)).result()
    return asyncio.run(coro)


async def _run_judge_for_run_async(
    *,
    run_id: str,
    enriched_path: Path,
    yaml_path: Path,
    paths: RunnerPaths,
    checkpoint_suffix: str,
    databricks_profile: str,
    delete_previous_checkpoint: bool,
    purge_error_rows: bool,
) -> Path:
    from hg_ds_evals.common.utils import (
        filter_df_with_checkpoints,
        load_checkpoint,
        prepare_eval_sample,
    )
    from hg_ds_evals.evals.evaluator import async_run_evals
    from hg_ds_evals.llm.api_client import get_api_client
    from hg_ds_evals.prompts.builder import PromptBuilder
    from hg_ds_evals.rubrics.loader import build_rubric_from_config

    print("\n" + "=" * 80)
    print(f"LLM JUDGE: {run_id}")
    print("=" * 80)

    config = _load_yaml(yaml_path)
    dataset = config.setdefault("dataset", {})
    dataset["mlflow_run_id"] = run_id
    dataset["input_dataset"] = f"dts_eval_ts100_exp_001_{run_id}"

    rubric = build_rubric_from_config(config)
    config_dir = yaml_path.parent
    paths_config = config.get("paths", {})
    system_template_path = _resolve_template_path(
        paths_config.get("system_template_path"),
        config_dir,
    )
    user_template_path = _resolve_template_path(
        paths_config.get("user_template_path"),
        config_dir,
    )
    builder = PromptBuilder(
        rubric=rubric,
        system_template_path=system_template_path,
        user_template_path=user_template_path,
    )
    system_prompt = builder.build_system_prompt()
    print(f"Rubric: {rubric.metadata.name} ({rubric.dimension_ids})")
    print(f"System template: {system_template_path or 'embedded default'}")
    print(f"User template:   {user_template_path or 'embedded default'}")

    if not enriched_path.exists():
        raise FileNotFoundError(f"Judge input CSV does not exist: {enriched_path}")
    df = pd.read_csv(enriched_path)
    print(f"Judge input CSV: {enriched_path} ({len(df):,} rows)")

    experiment_name = yaml_path.stem
    id_columns = config["dataset"].get("id_columns", [])
    num_rows = config["dataset"]["test_num_rows"]
    checkpoint_dir = _resolve_relative_to_notebook_dir(
        config["paths"].get("checkpoint_dir", "checkpoints"),
        paths,
    )

    df_sample, checkpoint_file_name = prepare_eval_sample(
        df=df,
        evals_name=experiment_name,
        reasoning_effort=config["model"]["reasoning_effort"],
        suffix=checkpoint_suffix or None,
        num_rows=num_rows,
    )

    checkpoint_path = checkpoint_dir / checkpoint_file_name
    if delete_previous_checkpoint:
        backup_path = _move_existing_checkpoint(checkpoint_path)
        if backup_path:
            print(f"Moved previous checkpoint to backup: {backup_path}")
        else:
            print(f"No previous checkpoint to clear: {checkpoint_path}")

    cols = list(config["dataset"]["eval_columns"])
    if id_columns:
        cols = list(dict.fromkeys(list(id_columns) + cols))
    passthrough = config["dataset"].get("passthrough_columns", [])
    if passthrough:
        cols = list(dict.fromkeys(cols + list(passthrough)))

    missing_cols = [column for column in cols if column not in df_sample.columns]
    if missing_cols:
        print(f"WARN: missing judge input columns dropped: {missing_cols}")
        cols = [column for column in cols if column in df_sample.columns]
    df_eval = df_sample[cols].copy()

    checkpoint_df, loaded_checkpoint_path = load_checkpoint(
        checkpoint_file_name=checkpoint_file_name,
        checkpoint_dir=checkpoint_dir,
        purge_error_rows=purge_error_rows,
    )
    df_eval = filter_df_with_checkpoints(df_eval, checkpoint_df, id_cols=id_columns)
    print(f"Rows left for judge: {len(df_eval):,}")
    print(f"Checkpoint path:    {loaded_checkpoint_path}")

    if df_eval.empty:
        print("No judge rows to process.")
        return loaded_checkpoint_path

    client = get_api_client(
        model_deployment_name=config["model"]["model_deployment_name"],
        api_provider=config["model"]["api_provider"],
        databricks_endpoint_url=config["model"].get("databricks_endpoint_url"),
        databricks_base_url=config["model"].get("databricks_base_url"),
        databricks_workspace_host=config["model"].get("databricks_workspace_host"),
        databricks_profile=databricks_profile or None,
    )

    _, metrics = await async_run_evals(
        df=df_eval,
        system_prompt=system_prompt,
        client=client,
        config=config,
        checkpoint_path=loaded_checkpoint_path,
        user_prompt_builder=builder.build_user_prompt,
    )
    print(f"Judge complete. Metrics rows: {len(metrics or {})}")
    return loaded_checkpoint_path


def _run_judge_for_run(
    *,
    run_id: str,
    enriched_path: Path,
    yaml_path: Path,
    paths: RunnerPaths,
    checkpoint_suffix: str,
    databricks_profile: str,
    delete_previous_checkpoint: bool,
    purge_error_rows: bool,
) -> Path:
    return _run_async_from_sync(
        _run_judge_for_run_async(
            run_id=run_id,
            enriched_path=enriched_path,
            yaml_path=yaml_path,
            paths=paths,
            checkpoint_suffix=checkpoint_suffix,
            databricks_profile=databricks_profile,
            delete_previous_checkpoint=delete_previous_checkpoint,
            purge_error_rows=purge_error_rows,
        )
    )


def _load_report_modules(paths: RunnerPaths):
    # The API notebooks live under experiments/api/notebooks, but the shared
    # report builders are in experiments/.
    experiments_dir = paths.repo_root / "experiments"
    if str(experiments_dir) not in sys.path:
        sys.path.insert(0, str(experiments_dir))
    api_report = importlib.import_module("api_report")
    api_report_multi = importlib.import_module("api_report_multi")
    return api_report, api_report_multi


def _write_single_report(
    *,
    input_path: Path,
    baseline_path: Path | None,
    judge_path: Path | None,
    output_path: Path,
    paths: RunnerPaths,
) -> Path:
    api_report, _ = _load_report_modules(paths)

    baseline_lookup = None
    if baseline_path is not None:
        baseline_lookup = api_report.load_baseline(baseline_path)
        print(f"[baseline] loaded {len(baseline_lookup)} cases from {baseline_path.name}")

    df = pd.read_csv(input_path)
    df = api_report.enrich(df)
    if judge_path is not None:
        if not judge_path.exists():
            raise FileNotFoundError(f"Judge checkpoint does not exist: {judge_path}")
        judge_lookup = api_report.load_judge_lookup(judge_path)
        print(f"[judge] loaded {len(judge_lookup)} cases from {judge_path.name}")
        df = api_report.attach_judge_columns(df, judge_lookup)

    html_text = api_report.render_html(
        df,
        input_path=input_path,
        output_path=output_path,
        baseline=baseline_lookup,
        baseline_path=baseline_path,
        prompts_path=None,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    print(f"Wrote report: {output_path}")
    return output_path


def _write_multi_report(
    *,
    input_paths: Sequence[Path],
    output_path: Path,
    paths: RunnerPaths,
) -> Path:
    _, api_report_multi = _load_report_modules(paths)
    if len(input_paths) < 2:
        raise ValueError("Multi report needs at least two input CSVs.")

    runs = []
    for input_path in input_paths:
        run = api_report_multi.load_run(input_path)
        print(f"[multi] {run['label']} — {run['n']} cases ({input_path.name})")
        runs.append(run)

    html_text = api_report_multi.render_html(runs, output_path=output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    print(f"Wrote multi report: {output_path}")
    return output_path


def run_api_evaluation(
    *,
    run_ids: str | Sequence[str],
    report_name: str,
    run_baseline: str | Sequence[str] | None = None,
    apply_parameter_equivalence: bool = True,
    run_judge: bool = False,
    checkpoint_suffix: str | None = None,
    yaml_file_name: str = DEFAULT_API_YAML,
    delete_previous_checkpoint: bool = False,
    purge_error_rows: bool = True,
    mlflow_experiment_id: str | None = None,
    databricks_profile: str = DEFAULT_DATABRICKS_PROFILE,
    evals_src: str | Path | None = DEFAULT_EVALS_SRC,
    input_traces_dir: str | Path = DEFAULT_INPUT_TRACES_DIR,
    reports_dir: str | Path | None = None,
    max_results: int | None = None,
    on_databricks: bool = False,
) -> list[PreparedRun]:
    """Run the API eval pipeline.

    This is the notebook-friendly entry point. It returns one ``PreparedRun``
    per requested run id, with paths to the enriched CSV, judge checkpoint
    and single-run report.
    """
    run_id_list = _as_list(run_ids)
    if not run_id_list:
        raise ValueError("At least one RUN_ID is required.")
    if not report_name:
        raise ValueError("report_name is required.")

    baseline_ids = _resolve_baselines(run_id_list, _as_list(run_baseline) if run_baseline else None)
    paths = _build_paths(input_traces_dir=input_traces_dir, reports_dir=reports_dir)
    _ensure_sys_path(paths, evals_src)

    yaml_path = _api_config_path(paths, yaml_file_name)
    config_for_defaults = _load_yaml(yaml_path)
    experiment_id = mlflow_experiment_id or str(
        config_for_defaults.get("dataset", {}).get("mlflow_experiment_id", "")
    )
    if not experiment_id:
        raise ValueError(
            "No MLflow experiment id supplied and the YAML has no dataset.mlflow_experiment_id."
        )

    _configure_mlflow_auth(databricks_profile, on_databricks=on_databricks)

    prepared_by_run_id: dict[str, PreparedRun] = {}
    requested_set = set(run_id_list)
    baseline_set = set(baseline_ids)

    for run_id in run_id_list:
        prepared_by_run_id[run_id] = _prepare_deterministic_run(
            run_id=run_id,
            experiment_id=experiment_id,
            paths=paths,
            max_results=max_results,
            apply_parameter_equivalence=apply_parameter_equivalence,
        )

    for baseline_id in sorted(baseline_set - requested_set):
        baseline_path = paths.input_traces_dir / f"enriched_traces_{baseline_id}.csv"
        if baseline_path.exists():
            print(f"Using existing baseline enriched CSV: {baseline_path}")
            prepared_by_run_id[baseline_id] = PreparedRun(
                run_id=baseline_id,
                enriched_path=baseline_path,
            )
            continue
        print(
            f"Baseline CSV was missing, so the baseline run will be prepared too: {baseline_id}"
        )
        prepared_by_run_id[baseline_id] = _prepare_deterministic_run(
            run_id=baseline_id,
            experiment_id=experiment_id,
            paths=paths,
            max_results=max_results,
            apply_parameter_equivalence=apply_parameter_equivalence,
        )

    prepared_runs: list[PreparedRun] = []
    for index, run_id in enumerate(run_id_list, start=1):
        prepared = prepared_by_run_id[run_id]
        baseline_id = baseline_ids[index - 1]
        baseline_prepared = prepared_by_run_id[baseline_id]
        suffix = _checkpoint_suffix_for_run(
            checkpoint_suffix,
            report_name,
            run_id,
            index,
            unique_default=run_judge and len(run_id_list) > 1,
        )

        judge_path = None
        if run_judge:
            judge_path = _run_judge_for_run(
                run_id=run_id,
                enriched_path=prepared.enriched_path,
                yaml_path=yaml_path,
                paths=paths,
                checkpoint_suffix=suffix,
                databricks_profile=databricks_profile,
                delete_previous_checkpoint=delete_previous_checkpoint,
                purge_error_rows=purge_error_rows,
            )
            prepared.judge_checkpoint_path = judge_path

        report_path = _single_report_path(paths.reports_dir, report_name, run_id, baseline_id)
        prepared.report_path = _write_single_report(
            input_path=prepared.enriched_path,
            baseline_path=baseline_prepared.enriched_path,
            judge_path=judge_path,
            output_path=report_path,
            paths=paths,
        )
        prepared_runs.append(prepared)

    if len(run_id_list) > 1:
        _write_multi_report(
            input_paths=[prepared_by_run_id[run_id].enriched_path for run_id in run_id_list],
            output_path=_multi_report_path(paths.reports_dir, report_name),
            paths=paths,
        )

    return prepared_runs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run-id",
        "--RUN_ID",
        dest="run_ids",
        nargs="+",
        action="append",
        required=True,
        help="One or more MLflow run ids. You can pass space-separated or comma-separated values.",
    )
    parser.add_argument(
        "--run-baseline",
        "--RUN_BASELINE",
        dest="run_baseline",
        nargs="+",
        action="append",
        default=None,
        help="Optional baseline run id(s). Omit to compare each run with itself.",
    )
    parser.add_argument(
        "--report-name",
        "--report_name",
        dest="report_name",
        required=True,
        help="Base report name, for example 'pr_620'.",
    )
    parser.add_argument(
        "--apply-parameter-equivalence",
        "--APPLY_PARAMETER_EQUIVALENCE",
        dest="apply_parameter_equivalence",
        action="store_true",
        default=True,
        help="Apply the source-backed parameter equivalence rules before tool_parameter scoring.",
    )
    parser.add_argument(
        "--no-apply-parameter-equivalence",
        dest="apply_parameter_equivalence",
        action="store_false",
        help="Disable parameter equivalence and score raw tool parameters.",
    )
    parser.add_argument(
        "--run-judge",
        "--RUN_JUDGE",
        dest="run_judge",
        action="store_true",
        help="Run the LLM language judge after deterministic scoring.",
    )
    parser.add_argument(
        "--checkpoint-suffix",
        "--CHECKPOINT_SUFFIX",
        dest="checkpoint_suffix",
        default=None,
        help=(
            "Judge checkpoint suffix. Defaults to report-name for one run, "
            "or report-name_<short_run_id> for multiple judge runs. "
            "Supports {run_id}, {short_run_id}, and {run_index} placeholders."
        ),
    )
    parser.add_argument(
        "--yaml-file-name",
        "--YAML_FILE_NAME",
        dest="yaml_file_name",
        default=DEFAULT_API_YAML,
        help=f"YAML file stem/name in experiments/api/configs (default: {DEFAULT_API_YAML}).",
    )
    parser.add_argument(
        "--delete-previous-checkpoint",
        "--DELETE_PREVIOUS_CHECKPOINT",
        dest="delete_previous_checkpoint",
        action="store_true",
        help="Move the previous judge checkpoint to a timestamped backup before running.",
    )
    parser.add_argument(
        "--no-purge-error-rows",
        dest="purge_error_rows",
        action="store_false",
        default=True,
        help="Keep error=True rows in an existing judge checkpoint.",
    )
    parser.add_argument(
        "--mlflow-experiment-id",
        dest="mlflow_experiment_id",
        default=None,
        help="MLflow experiment id. Defaults to dataset.mlflow_experiment_id from the YAML.",
    )
    parser.add_argument(
        "--profile",
        "--dbx-profile",
        dest="databricks_profile",
        default=os.environ.get("DATABRICKS_CONFIG_PROFILE", DEFAULT_DATABRICKS_PROFILE),
        help="Databricks CLI profile for local auth.",
    )
    parser.add_argument(
        "--evals-src",
        dest="evals_src",
        default=os.environ.get("AI_DS_EVALS_SRC", DEFAULT_EVALS_SRC),
        help="Path to ai-data-science/evals/src.",
    )
    parser.add_argument(
        "--input-traces-dir",
        dest="input_traces_dir",
        default=DEFAULT_INPUT_TRACES_DIR,
        help=f"Directory for raw/scored/enriched traces (default: {DEFAULT_INPUT_TRACES_DIR}).",
    )
    parser.add_argument(
        "--reports-dir",
        dest="reports_dir",
        default=None,
        help="Output report directory. Defaults to experiments/api/notebooks/reports.",
    )
    parser.add_argument(
        "--max-results",
        dest="max_results",
        type=int,
        default=None,
        help="Optional cap on MLflow traces fetched per run.",
    )
    parser.add_argument(
        "--on-databricks",
        dest="on_databricks",
        action="store_true",
        help="Use attached Databricks credentials instead of local CLI profile auth.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    prepared_runs = run_api_evaluation(
        run_ids=_split_cli_values(args.run_ids),
        run_baseline=(
            _split_cli_values(args.run_baseline)
            if args.run_baseline is not None
            else None
        ),
        apply_parameter_equivalence=args.apply_parameter_equivalence,
        report_name=args.report_name,
        run_judge=args.run_judge,
        checkpoint_suffix=args.checkpoint_suffix,
        yaml_file_name=args.yaml_file_name,
        delete_previous_checkpoint=args.delete_previous_checkpoint,
        purge_error_rows=args.purge_error_rows,
        mlflow_experiment_id=args.mlflow_experiment_id,
        databricks_profile=args.databricks_profile,
        evals_src=args.evals_src,
        input_traces_dir=args.input_traces_dir,
        reports_dir=args.reports_dir,
        max_results=args.max_results,
        on_databricks=args.on_databricks,
    )

    print("\nDone.")
    for prepared in prepared_runs:
        print(f"  run {prepared.run_id}")
        print(f"    enriched: {prepared.enriched_path}")
        if prepared.judge_checkpoint_path:
            print(f"    judge:    {prepared.judge_checkpoint_path}")
        if prepared.report_path:
            print(f"    report:   {prepared.report_path}")


if __name__ == "__main__":
    main()
