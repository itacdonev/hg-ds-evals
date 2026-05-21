"""Backfill span-level error columns into a CZKB checkpoint CSV.

Existing checkpoints predate span-error capture: ``info.state`` says
``OK`` whenever the agent returned an answer, even if a child span
crashed (KB endpoint ``ReadTimeout`` / ``ConnectTimeout``, asyncio
``CancelledError`` on a retry, etc.). The judge / scorer never had
those signals in scope, so they're missing from
``checkpoints/*.csv`` and the report can't surface where things broke.

Re-running the eval just to capture them would be wasteful — the spans
are already cached in the MLflow trace pickle written by
``czkb_001_import_traces_local.ipynb``.

Pipeline:
1. Load the cached pickle (``traces_<RUN_NAME>.pickle``).
2. Walk spans → :func:`extract_span_errors` per ``trace_id``.
3. Make a timestamped ``.bak`` of the target checkpoint.
4. Merge :data:`SPAN_ERROR_COLUMNS` into the checkpoint by ``trace_id``
   and save in place.

Usage::

    python backfill_span_errors.py \\
        --pickle ../input/traces_online_adhoc_quiet_hawk_score.pickle \\
        --checkpoint checkpoints/evals_czkb_exp_002_baseline_no_expected_enums_high_online_adhoc_quiet_hawk_score.csv

Re-running is safe: rows already carrying ``span_*`` columns are
dropped and re-added with the freshly parsed values, and the previous
checkpoint is preserved as ``.bak.<ts>``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE
while _REPO_ROOT != _REPO_ROOT.parent and not (_REPO_ROOT / "hg_ds_evals").is_dir():
    _REPO_ROOT = _REPO_ROOT.parent
if (_REPO_ROOT / "hg_ds_evals").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hg_ds_evals.preprocessing.traces import (  # noqa: E402
    SPAN_ERROR_COLUMNS,
    build_span_errors_dataframe,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument(
        "--pickle",
        type=Path,
        required=True,
        help="path to the cached mlflow.search_traces pickle (has trace_id + spans columns)",
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="path to the judge checkpoint CSV to patch in place",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="parse and report coverage but do not modify the checkpoint",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.pickle.is_file():
        raise FileNotFoundError(f"pickle not found: {args.pickle}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    print(f"loading traces pickle: {args.pickle}")
    traces_df = pd.read_pickle(args.pickle)
    if "trace_id" not in traces_df.columns or "spans" not in traces_df.columns:
        raise RuntimeError(
            f"pickle missing required columns; got {list(traces_df.columns)!r}"
        )
    print(f"  rows: {len(traces_df):,}")

    print("parsing spans -> span errors")
    err_df = build_span_errors_dataframe(traces_df)
    n_total = len(err_df)
    n_with_err = int(err_df["trace_has_span_error"].sum()) if n_total else 0
    print(f"  traces with span errors: {n_with_err:,} / {n_total:,}")

    print(f"\nloading checkpoint: {args.checkpoint}")
    ckp_df = pd.read_csv(args.checkpoint)
    print(f"  rows: {len(ckp_df):,}")
    if "trace_id" not in ckp_df.columns:
        raise RuntimeError("checkpoint has no trace_id column — cannot join")

    matched = int(ckp_df["trace_id"].isin(err_df.index).sum())
    print(f"  trace_id matches: {matched:,} / {len(ckp_df):,}")
    if matched == 0:
        raise RuntimeError(
            "no trace_ids overlap between pickle and checkpoint — wrong files?"
        )

    if args.dry_run:
        print("\n--dry-run set; not modifying checkpoint.")
        return

    backup_path = args.checkpoint.with_name(
        f"{args.checkpoint.name}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    shutil.copy2(args.checkpoint, backup_path)
    print(f"\nbackup written: {backup_path}")

    # Drop any pre-existing span-error columns so we never carry stale
    # values alongside fresh ones; the join below re-introduces them.
    pre_existing = [c for c in SPAN_ERROR_COLUMNS if c in ckp_df.columns]
    if pre_existing:
        print(f"replacing existing columns: {pre_existing}")
        ckp_df = ckp_df.drop(columns=pre_existing)

    merged = ckp_df.merge(err_df, how="left", left_on="trace_id", right_index=True)
    if len(merged) != len(ckp_df):
        raise RuntimeError(
            f"row count changed after merge: {len(ckp_df)} -> {len(merged)}"
        )

    # build_span_errors_dataframe always emits a value per trace, but if
    # the join misses anything (shouldn't happen given the assertion
    # above) keep the columns CSV-stable rather than NaN-typed.
    merged["trace_has_span_error"] = merged["trace_has_span_error"].fillna(False).astype(bool)
    merged["span_error_count"] = merged["span_error_count"].fillna(0).astype(int)
    merged["span_error_types_json"] = merged["span_error_types_json"].fillna("[]")
    merged["span_errors_json"] = merged["span_errors_json"].fillna("[]")

    merged.to_csv(args.checkpoint, index=False)
    size_kb = args.checkpoint.stat().st_size / 1024
    print(f"saved: {args.checkpoint}  ({size_kb:,.1f} KB)")
    n_flagged = int(merged["trace_has_span_error"].sum())
    print(f"rows with trace_has_span_error=True: {n_flagged:,} / {len(merged):,}")
    if n_flagged:
        # one-line per exception type, sorted by count desc
        from collections import Counter
        import json as _json

        counter: Counter[str] = Counter()
        for raw in merged.loc[merged["trace_has_span_error"], "span_error_types_json"]:
            try:
                for t in _json.loads(raw or "[]"):
                    counter[t] += 1
            except _json.JSONDecodeError:
                continue
        if counter:
            print("exception type breakdown (across affected traces):")
            for t, n in counter.most_common():
                print(f"  {t:<24} {n}")
    print(f"\nif anything looks off, restore with:\n  cp {backup_path} {args.checkpoint}")


if __name__ == "__main__":
    main()
