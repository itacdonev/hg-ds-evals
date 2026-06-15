#!/usr/bin/env python
"""Run the deterministic scorers over an already-parsed traces table.

Self-contained companion to ``run_deterministic_scorers.py``. Use this when you
have ALREADY turned your MLflow traces into a flat table (CSV / Parquet / pickle)
and just want to score it.

Your parsed table must carry these columns (the canonical eval-scoring contract):

    agent_routing   -> expected_agent, actual_agent
    tool_usage      -> expected_tool_calls, actual_tool_calls, available_tools
    tool_parameter  -> expected_tool_calls, actual_tool_calls

  * expected_agent / actual_agent : strings (an empty expected_agent means
    "the supervisor handles it directly" — see SUPERVISOR_AGENT below).
  * expected_tool_calls / actual_tool_calls : lists of dicts shaped like
    ``{"tool": "analyze_transactions", "parameters": {...}}`` (expected) /
    ``{"tool": "...", "arguments": {...}}`` (actual). When read from CSV these
    arrive as strings and are tolerantly parsed (JSON first, then Python literal).
  * available_tools : list of tool-name strings (lets tool_usage tell a
    misused tool from a hallucinated one). Optional — defaults to [].
  * scorers_to_run : optional list naming which scorers to run per row. If your
    table doesn't have it, pass --scorers to force a fixed set on every row.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path


SUPPORTED_SCORERS = ("agent_routing", "tool_usage", "tool_parameter")
DEFAULT_AVAILABLE_AGENTS = ("main_agent", "daily_banking_agent", "hg-invest-phase2")
SUPERVISOR_AGENT = "main_agent"

# Columns that hold lists/dicts and therefore need tolerant deserialization
# when the table is read from CSV.
_LISTISH_COLUMNS = ("expected_tool_calls", "actual_tool_calls", "available_tools", "scorers_to_run")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Parsed traces table: .csv, .parquet, or .pkl/.pickle.")
    parser.add_argument(
        "--evals-src",
        default=os.environ.get("AI_DS_EVALS_SRC"),
        help="Path to the ai-data-science/evals 'src' dir. Falls back to $AI_DS_EVALS_SRC. "
        "Not needed if ai_data_science is already importable.",
    )
    parser.add_argument("--output", default=None, help="Output CSV path (default: ./scored_<input-stem>.csv).")
    parser.add_argument(
        "--scorers",
        default=None,
        help="Comma-separated scorers to force on EVERY row, overriding scorers_to_run. "
        f"Choose from: {', '.join(SUPPORTED_SCORERS)}. Required if the table has no scorers_to_run column.",
    )
    parser.add_argument(
        "--available-agents",
        default=None,
        help="Comma-separated agent registry for the routing scorer "
        f"(default: {','.join(DEFAULT_AVAILABLE_AGENTS)}).",
    )
    return parser.parse_args()


def bootstrap_evals(evals_src: str | None) -> None:
    if not evals_src:
        return
    evals_path = Path(evals_src).expanduser()
    if not evals_path.is_dir():
        sys.exit(f"--evals-src path does not exist: {evals_path}")
    if str(evals_path) not in sys.path:
        sys.path.insert(0, str(evals_path))


def _coerce_listish(value):
    """Tolerantly turn a cell into a list/dict: pass-through if already one,
    else parse a JSON or Python-literal string; None/NaN/'' -> None."""
    import pandas as pd

    if isinstance(value, (list, dict)):
        return value
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none"):
        return None
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None


def load_table(input_path: Path):
    import pandas as pd

    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(input_path)
    elif suffix == ".parquet":
        df = pd.read_parquet(input_path)
    elif suffix in (".pkl", ".pickle"):
        df = pd.read_pickle(input_path)
    else:
        sys.exit(f"Unsupported input type {suffix!r}; use .csv, .parquet, or .pkl/.pickle.")

    for column in _LISTISH_COLUMNS:
        if column in df.columns:
            df[column] = df[column].apply(_coerce_listish)
    return df


def build_registry(available_agents):
    """scorer-key -> (scorer, dimension, kwargs-builder). Raw lists, no relaxation."""
    from ai_data_science.evals.dimension import Dimension
    from ai_data_science.evals.scales import BinaryScale, NumericScale
    from ai_data_science.evals.types import ScoreLevel
    from ai_data_science.evals.scorers.deterministic import (
        RoutingCorrectnessScorer,
        ToolUsageScorer,
        ToolParameterScorer,
    )

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

    routing_scorer = RoutingCorrectnessScorer(available_agents=tuple(available_agents))
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
    if forced_scorers:
        return df.index.to_series().apply(lambda _: list(forced_scorers))

    if "scorers_to_run" not in df.columns:
        sys.exit(
            "Table has no 'scorers_to_run' column and --scorers was not given. "
            f"Pass --scorers with any of: {', '.join(SUPPORTED_SCORERS)}."
        )

    def _supported(row_scorers):
        if not isinstance(row_scorers, list):
            return []
        return [s for s in row_scorers if s in SUPPORTED_SCORERS]

    return df["scorers_to_run"].apply(_supported)


def require_columns(df, all_scorers) -> None:
    needed: set[str] = set()
    if "agent_routing" in all_scorers:
        needed |= {"expected_agent", "actual_agent"}
    if "tool_usage" in all_scorers:
        needed |= {"expected_tool_calls", "actual_tool_calls"}  # available_tools optional
    if "tool_parameter" in all_scorers:
        needed |= {"expected_tool_calls", "actual_tool_calls"}
    missing = sorted(needed - set(df.columns))
    if missing:
        sys.exit(f"Input table is missing required column(s) for the chosen scorers: {missing}")


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
            print(
                f"  {scorer_name:>16s}: scored={scored:3d}  mean={vals.mean():.3f}  "
                f"pass(1.0)={(vals == 1.0).sum():3d}  partial={((vals > 0) & (vals < 1)).sum():3d}  "
                f"fail(0.0)={(vals == 0.0).sum():3d}  statuses={statuses}"
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
    bootstrap_evals(args.evals_src)

    forced_scorers = None
    if args.scorers:
        forced_scorers = [s.strip() for s in args.scorers.split(",") if s.strip()]
        unsupported = [s for s in forced_scorers if s not in SUPPORTED_SCORERS]
        if unsupported:
            sys.exit(f"Unsupported scorer(s): {unsupported}. Choose from {list(SUPPORTED_SCORERS)}.")

    available_agents = (
        [a.strip() for a in args.available_agents.split(",") if a.strip()]
        if args.available_agents
        else list(DEFAULT_AVAILABLE_AGENTS)
    )

    input_path = Path(args.input).expanduser()
    if not input_path.is_file():
        sys.exit(f"--input file not found: {input_path}")
    df = load_table(input_path)
    print(f"Loaded {len(df):,} rows from {input_path}")

    per_row_scorers = resolve_per_row_scorers(df, forced_scorers)
    rows_with_scorers = (per_row_scorers.apply(len) > 0).sum()
    if rows_with_scorers == 0:
        sys.exit("No row has a deterministic scorer to run. Pass --scorers to force a fixed set.")
    print(f"Rows with at least one deterministic scorer: {rows_with_scorers}/{len(df)}")

    all_intended = sorted({s for sl in per_row_scorers for s in sl})
    require_columns(df, all_intended)

    registry, numeric_scale_cls = build_registry(available_agents)
    scored_df, all_scorers = score_dataframe(df, per_row_scorers, registry)

    output_path = Path(args.output) if args.output else Path(f"scored_{input_path.stem}.csv")
    scored_df.to_csv(output_path, index=False)
    print(f"\nWrote scored CSV: {output_path.resolve()}")

    print_summary(scored_df, all_scorers, registry, numeric_scale_cls)


if __name__ == "__main__":
    main()
