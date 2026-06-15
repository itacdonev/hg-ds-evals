#!/usr/bin/env python
"""Run the deterministic scorers over one MLflow inference run.

Standalone CLI extracted from ``import_traces_local.ipynb``. Given an MLflow
experiment id + run id, it fetches the traces, parses them into the canonical
eval-scoring dataframe, runs the three deterministic scorers, writes a scored
CSV, and prints a per-scorer summary. Nothing else — no translations, no HTML
report, no Unity Catalog write.

The scorers themselves live in the sibling ``ai-data-science/evals`` package,
which is NOT installed in this repo's ``.venv``. Point the script at your local
checkout with ``--evals-src`` or the ``AI_DS_EVALS_SRC`` env var.

Example (local, authenticating via a Databricks CLI profile):

    python run_deterministic_scorers.py \
        --experiment-id 2374936353493891 \
        --run-id 5abe9d81bba14d91af922a1ca0a52f4b \
        --profile adb-uat \
        --evals-src /path/to/ai-data-science/evals/src \
        --output scored_traces.csv

On a Databricks cluster (uses the attached spark/MLflow, no profile needed):

    python run_deterministic_scorers.py \
        --experiment-id ... --run-id ... --on-databricks
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# ── Deterministic scorers we know how to wire here ────────────────────
# Keys match what the eval team writes in the HUMAN ``scorers`` assessment
# (the parser surfaces them per row in ``scorers_to_run``). Any other
# scorer name in a row — e.g. an LLM-judge dimension — is skipped.
SUPPORTED_SCORERS = ("agent_routing", "tool_usage", "tool_parameter")

# Empty expected_agent means "the supervisor should handle it directly"
# (chit_chat / ethical / refusal cases). RoutingCorrectnessScorer rejects
# empty reference fields, so we canonicalize empty -> main_agent on both
# sides: actual="main_agent" scores 1, any sub-agent scores 0.
SUPERVISOR_AGENT = "main_agent"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-id", required=True, help="MLflow experiment id holding the traces.")
    parser.add_argument("--run-id", required=True, help="MLflow run id whose traces to score.")
    parser.add_argument(
        "--profile",
        default=os.environ.get("DATABRICKS_CONFIG_PROFILE", "adb-uat"),
        help="Databricks CLI profile for auth when running locally "
        "(default: $DATABRICKS_CONFIG_PROFILE or 'adb-uat'). Ignored with --on-databricks.",
    )
    parser.add_argument(
        "--evals-src",
        default=os.environ.get("AI_DS_EVALS_SRC"),
        help="Path to the ai-data-science/evals 'src' dir (the deterministic scorers). "
        "Falls back to $AI_DS_EVALS_SRC. Not needed if ai_data_science is already importable.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: ./scored_traces_<run_id>.csv).",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="Cap the number of traces fetched (default: all).",
    )
    parser.add_argument(
        "--scorers",
        default=None,
        help="Comma-separated scorers to force on EVERY row, overriding each row's "
        f"scorers_to_run. Choose from: {', '.join(SUPPORTED_SCORERS)}. "
        "Default: run whatever each row lists in scorers_to_run.",
    )
    parser.add_argument(
        "--on-databricks",
        action="store_true",
        help="Run on a Databricks cluster using the attached spark/MLflow (skip profile auth).",
    )
    return parser.parse_args()


def bootstrap_sys_path(evals_src: str | None) -> Path:
    """Make ``hg_ds_evals`` and ``ai_data_science`` importable; return repo root."""
    repo_root = Path(__file__).resolve().parent
    while repo_root != repo_root.parent and not (repo_root / "hg_ds_evals").is_dir():
        repo_root = repo_root.parent
    if not (repo_root / "hg_ds_evals").is_dir():
        sys.exit("Could not locate the hg_ds_evals repo root from this script's location.")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    if evals_src:
        evals_path = Path(evals_src).expanduser()
        if not evals_path.is_dir():
            sys.exit(f"--evals-src path does not exist: {evals_path}")
        if str(evals_path) not in sys.path:
            sys.path.insert(0, str(evals_path))
    return repo_root


def configure_auth(profile: str, on_databricks: bool) -> None:
    """Point MLflow at the workspace tracking server (local) or no-op (DBX)."""
    import mlflow

    if on_databricks:
        print("Running on Databricks — using the attached spark/MLflow.")
        return

    # Strip env vars the Databricks VS Code extension injects; its stale loopback
    # OAuth URL hangs auth. Falling through to the .databrickscfg profile is robust.
    for var in ("DATABRICKS_AUTH_TYPE", "DATABRICKS_METADATA_SERVICE_URL", "DATABRICKS_HOST", "DATABRICKS_CLUSTER_ID"):
        os.environ.pop(var, None)
    os.environ["DATABRICKS_CONFIG_PROFILE"] = profile

    from databricks.sdk import WorkspaceClient

    workspace = WorkspaceClient()
    print(f"Authenticated as: {workspace.current_user.me().user_name}")
    print(f"Workspace host:   {workspace.config.host}")

    mlflow.set_tracking_uri("databricks")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")


def fetch_traces(experiment_id: str, run_id: str, max_results: int | None):
    import mlflow

    # Quiet the noisy "Connection pool is full" warnings emitted while MLflow
    # downloads trace artifacts from blob storage in parallel.
    logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

    traces_df = mlflow.search_traces(
        locations=[experiment_id],
        run_id=run_id,
        max_results=max_results,
        order_by=["timestamp_ms DESC"],
    )
    print(f"Fetched {len(traces_df):,} trace rows for run {run_id}.")
    return traces_df


def build_registry():
    """Build the scorer-key -> (scorer, dimension, kwargs-builder) registry."""
    from ai_data_science.evals.dimension import Dimension
    from ai_data_science.evals.scales import BinaryScale, NumericScale
    from ai_data_science.evals.types import ScoreLevel
    from ai_data_science.evals.scorers.deterministic import (
        RoutingCorrectnessScorer,
        ToolUsageScorer,
        ToolParameterScorer,
    )
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

    routing_scorer = RoutingCorrectnessScorer(available_agents=tuple(KNOWN_AGENT_NAMES))
    tool_usage_scorer = ToolUsageScorer(
        mode="exact",
        order_sensitive=True,
        case_sensitive=False,
        ignore_failed_calls=False,
    )
    tool_parameter_scorer = ToolParameterScorer(case_sensitive=False, value_coercion="string")

    def _canon_agent(name: object) -> str:
        return str(name).strip() or SUPERVISOR_AGENT

    registry = {
        "agent_routing": (
            routing_scorer,
            binary_dim,
            lambda row: {
                "expected_agent": _canon_agent(row.get("expected_agent")),
                "actual_agent": _canon_agent(row.get("actual_agent")),
            },
        ),
        "tool_usage": (
            tool_usage_scorer,
            binary_dim,
            lambda row: {
                "expected_tool_calls": row.get("expected_tool_calls") or [],
                "actual_tool_calls": row.get("actual_tool_calls") or [],
                "available_tools": row.get("available_tools") or [],
            },
        ),
        # Raw expected/actual lists — no runtime-equivalence relaxation.
        "tool_parameter": (
            tool_parameter_scorer,
            tool_parameter_dim,
            lambda row: {
                "expected_tool_calls": row.get("expected_tool_calls") or [],
                "actual_tool_calls": row.get("actual_tool_calls") or [],
            },
        ),
    }
    return registry, NumericScale


# Metadata fields we flatten into per-row CSV columns, by scorer name.
# Lists/dicts are JSON-serialized so they survive the CSV round-trip.
_METADATA_FLATTENERS = {
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
        "per_entry": lambda m: m.get("per_entry_results") or [],
    },
}


def _serialize_meta_value(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _score_row(row, scorer_name, registry):
    entry = registry.get(scorer_name)
    if entry is None:
        return (None, f"unknown scorer: {scorer_name!r}", "unknown_scorer", {})
    scorer, dimension, build_kwargs = entry
    try:
        result = scorer.score(dimension, **build_kwargs(row))
    except Exception as exc:  # config / programmer error — surface, don't swallow
        return (None, f"scorer raised: {exc!r}", "scorer_exception", {})
    meta = result.metadata or {}
    return (result.value, result.rationale or "", meta.get("status", "ok"), dict(meta))


def resolve_per_row_scorers(df, forced_scorers):
    """Return a Series of scorer-name lists to run per row.

    With ``--scorers`` set, every row gets that fixed list. Otherwise each row
    runs the supported subset of its own ``scorers_to_run``.
    """
    if forced_scorers:
        return df.index.to_series().apply(lambda _: list(forced_scorers))

    if "scorers_to_run" not in df.columns:
        sys.exit(
            "Parsed dataframe has no 'scorers_to_run' column — the traces carry no "
            "HUMAN 'scorers' assessment to drive deterministic scoring. Use --scorers "
            "to force a fixed set instead."
        )

    def _supported(row_scorers):
        if not isinstance(row_scorers, list):
            return []
        return [s for s in row_scorers if s in SUPPORTED_SCORERS]

    return df["scorers_to_run"].apply(_supported)


def score_dataframe(df, per_row_scorers, registry):
    out = df.copy()
    all_scorers = sorted({s for sl in per_row_scorers for s in sl})
    for scorer_name in all_scorers:
        out[f"{scorer_name}_score"] = None
        out[f"{scorer_name}_rationale"] = ""
        out[f"{scorer_name}_status"] = ""
        for suffix in _METADATA_FLATTENERS.get(scorer_name, {}):
            out[f"{scorer_name}_{suffix}"] = None

    for i in out.index:
        for scorer_name in per_row_scorers.loc[i]:
            value, rationale, status, meta = _score_row(out.loc[i], scorer_name, registry)
            out.at[i, f"{scorer_name}_score"] = value
            out.at[i, f"{scorer_name}_rationale"] = rationale
            out.at[i, f"{scorer_name}_status"] = status
            for suffix, fn in _METADATA_FLATTENERS.get(scorer_name, {}).items():
                out.at[i, f"{scorer_name}_{suffix}"] = _serialize_meta_value(fn(meta))
    return out, all_scorers


def print_summary(df, all_scorers, registry, numeric_scale_cls):
    import pandas as pd

    print("\nPer-scorer summary:")
    for scorer_name in all_scorers:
        score_col, status_col = f"{scorer_name}_score", f"{scorer_name}_status"
        scored = df[score_col].notna().sum()
        statuses = df[status_col].value_counts().to_dict()
        entry = registry.get(scorer_name)
        if entry is not None and isinstance(entry[1].scale, numeric_scale_cls):
            vals = pd.to_numeric(df[score_col], errors="coerce")
            mean = vals.mean()
            full_pass = (vals == 1.0).sum()
            partial = ((vals > 0) & (vals < 1)).sum()
            zero = (vals == 0.0).sum()
            print(
                f"  {scorer_name:>16s}: scored={scored:3d}  mean={mean:.3f}  "
                f"pass(1.0)={full_pass:3d}  partial={partial:3d}  fail(0.0)={zero:3d}  statuses={statuses}"
            )
        else:
            correct = (df[score_col] == 1).sum()
            wrong = (df[score_col] == 0).sum()
            print(
                f"  {scorer_name:>16s}: scored={scored:3d}  correct={correct:3d}  "
                f"wrong={wrong:3d}  statuses={statuses}"
            )


def main() -> None:
    args = parse_args()
    bootstrap_sys_path(args.evals_src)

    forced_scorers = None
    if args.scorers:
        forced_scorers = [s.strip() for s in args.scorers.split(",") if s.strip()]
        unsupported = [s for s in forced_scorers if s not in SUPPORTED_SCORERS]
        if unsupported:
            sys.exit(f"Unsupported scorer(s): {unsupported}. Choose from {list(SUPPORTED_SCORERS)}.")

    configure_auth(args.profile, args.on_databricks)

    from hg_ds_evals.preprocessing.mlflow_traces import build_dataframe_from_mlflow_traces

    traces_df = fetch_traces(args.experiment_id, args.run_id, args.max_results)
    parse_result = build_dataframe_from_mlflow_traces(traces_df)
    df = parse_result.dataframe
    print(f"Parsed {len(df):,} rows  (parse errors: {len(parse_result.parse_errors)})")
    for parse_error in parse_result.parse_errors[:5]:
        print(f"  parse error {parse_error.trace_id}: {parse_error.error}")

    per_row_scorers = resolve_per_row_scorers(df, forced_scorers)
    rows_with_scorers = (per_row_scorers.apply(len) > 0).sum()
    if rows_with_scorers == 0:
        sys.exit(
            "No row has a deterministic scorer to run. Either the traces carry no "
            "HUMAN 'scorers' assessment (so scorers_to_run is empty), or none of the "
            "listed scorers are deterministic. Pass --scorers to force a fixed set."
        )
    print(f"Rows with at least one deterministic scorer: {rows_with_scorers}/{len(df)}")

    registry, numeric_scale_cls = build_registry()
    scored_df, all_scorers = score_dataframe(df, per_row_scorers, registry)

    output_path = Path(args.output) if args.output else Path(f"scored_traces_{args.run_id}.csv")
    scored_df.to_csv(output_path, index=False)
    print(f"\nWrote scored CSV: {output_path.resolve()}")

    print_summary(scored_df, all_scorers, registry, numeric_scale_cls)


if __name__ == "__main__":
    main()
