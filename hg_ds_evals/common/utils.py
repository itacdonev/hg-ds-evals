# HEY GEORGE - EVALS
# Utility functions for common operations.

# utils.py
#=============================================================

import csv
import os
import shutil
import sys
import yaml
import pandas as pd
from pathlib import Path
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from ds_common.config.config import (
    HGCol as C,
    print_emoji as pe
    )

# ── Token counting ──────────────────────────────────────────

# tiktoken ships with Databricks Runtime (openai dependency).
# All current Azure OpenAI models (gpt-4o, gpt-4.1, gpt-5,
# gpt-5-nano) use the o200k_base encoding.
# import tiktoken

# _DEFAULT_ENCODING = "o200k_base"


# def count_tokens(text: str, encoding_name: str = _DEFAULT_ENCODING) -> int:
#     """
#     Count the number of tokens in a text string using tiktoken.

#     Args:
#         text: The input string to tokenise.
#         encoding_name: tiktoken encoding name (default ``o200k_base``
#             which covers gpt-4o / gpt-4.1 / gpt-5 family).

#     Returns:
#         Number of tokens. Returns 0 for ``None`` / empty strings.
#     """
#     if not text:
#         return 0
#     enc = tiktoken.get_encoding(encoding_name)
#     return len(enc.encode(text))


# def count_tokens_udf(col_name: str,
#                      encoding_name: str = _DEFAULT_ENCODING,
#                      output_col: str | None = None):
#     """
#     Return a PySpark **Column** with the token count for every row.

#     Can be used in ``withColumn`` or ``select``:

#         >>> from hg_ds_evals.common.utils import count_tokens_udf
#         >>> df = df.withColumn("n_tokens", count_tokens_udf("my_text_col"))

#     Or with a custom output column helper:

#         >>> col_expr = count_tokens_udf("my_text_col")
#         >>> df = df.withColumn("n_tokens", col_expr)

#     Args:
#         col_name: Name of the Spark DataFrame column containing text.
#         encoding_name: tiktoken encoding name.
#         output_col: Unused — kept for backwards compat. The caller
#             chooses the output column name via ``withColumn``.

#     Returns:
#         A PySpark Column expression (IntegerType).
#     """
#     @F.udf(IntegerType())
#     def _count_tokens(text: str) -> int:
#         if not text:
#             return 0
#         enc = tiktoken.get_encoding(encoding_name)
#         return len(enc.encode(text))

#     return _count_tokens(F.col(col_name))


def load_yaml_config(filepath):
    """Load configuration from a YAML file."""
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def purge_checkpoint_error_rows(
    checkpoint_path: str | Path,
    *,
    backup: bool = True,
) -> dict:
    """Drop rows where ``error == True`` from a checkpoint CSV in place.

    Useful when an eval run failed mid-stream (e.g. OAuth token expiry)
    and left error rows in the checkpoint. Removing them lets a re-run
    write fresh attempts cleanly without accumulating duplicate error
    rows over repeated retry cycles.

    Args:
        checkpoint_path: Path to the checkpoint CSV.
        backup: When True (default), saves a timestamped ``.csv.bak.<ts>``
            sibling before rewriting.

    Returns:
        ``{"removed": int, "kept": int, "backup_path": Path | None}``.
        Returns counts of 0 when the file does not exist, has no
        ``error`` column, or has no error rows to remove.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return {"removed": 0, "kept": 0, "backup_path": None}

    csv.field_size_limit(sys.maxsize)
    df = pd.read_csv(checkpoint_path, quoting=csv.QUOTE_ALL, dtype=str, low_memory=False)
    if "error" not in df.columns:
        return {"removed": 0, "kept": len(df), "backup_path": None}

    error_mask = df["error"].astype(str).str.lower().isin({"true", "1", "1.0", "yes"})
    n_removed = int(error_mask.sum())
    if n_removed == 0:
        return {"removed": 0, "kept": len(df), "backup_path": None}

    backup_path = None
    if backup:
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        backup_path = checkpoint_path.with_suffix(f".csv.bak.{ts}")
        shutil.copy2(checkpoint_path, backup_path)

    df_clean = df[~error_mask].copy()
    df_clean.to_csv(checkpoint_path, index=False, quoting=csv.QUOTE_ALL)
    return {"removed": n_removed, "kept": len(df_clean), "backup_path": backup_path}


def load_checkpoint(
    checkpoint_file_name: str,
    checkpoint_dir: str | Path | None = None,
    *,
    purge_error_rows: bool = False,
):
    """
    Load already completed evals from checkpoint file.

    Args:
        checkpoint_file_name: Name of checkpoint CSV file
        checkpoint_dir: Directory containing checkpoints (None = no checkpointing)
        purge_error_rows: When True, drop existing ``error == True`` rows
            from the checkpoint file before loading (with timestamped
            backup). Lets a re-run cleanly retry test cases that failed
            for transient reasons (e.g. OAuth token expiry mid-run)
            without accumulating duplicate error rows over retry cycles.

    Returns:
        Tuple of (checkpoint_df, checkpoint_path)
        - If no checkpointing: (empty DataFrame, None)
        - If file exists: (loaded DataFrame, Path)
        - If file doesn't exist: (empty DataFrame, Path)
    """

    if checkpoint_dir is None:
        return pd.DataFrame(), None

    checkpoint_dir = Path(checkpoint_dir)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ckp path
    checkpoint_path = checkpoint_dir / checkpoint_file_name

    if purge_error_rows and checkpoint_path.exists():
        result = purge_checkpoint_error_rows(checkpoint_path)
        if result["removed"]:
            backup_name = result["backup_path"].name if result["backup_path"] else "none"
            print(
                f"🧹 Purged {result['removed']} error rows from checkpoint "
                f"(kept {result['kept']}); backup → {backup_name}"
            )

    if checkpoint_path.exists():
        print(f"✓ Loading existing checkpoint: {checkpoint_path}")
        checkpoint_df = pd.read_csv(
            checkpoint_path,
            quoting=csv.QUOTE_ALL,
            on_bad_lines="warn",
        )
        return checkpoint_df, checkpoint_path
    else:
        print(f"ℹ No checkpoint found, starting fresh: {checkpoint_path}")
        return pd.DataFrame(), checkpoint_path


def update_checkpoint_df(result_dict:dict, checkpoint_path: Path, columns: list[str] | None = None) -> None:
    """
    Appends a dictionary of results as a new row to a CSV checkpoint file.
    If the checkpoint file does not exist, a header row is written. 
    Otherwise, the dictionary is appended as a new row.

    Args:
        result_dict (dict): The dictionary containing results to be saved.
        checkpoint_path (Path): The file path (as a pathlib.Path object) to the checkpoint CSV file.
        columns: Optional canonical checkpoint header. When provided, the
            first row is written with this header even if the result is an
            error row with fewer keys.

    Returns:
        None
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_columns = list(dict.fromkeys(columns or []))
    row_df = pd.DataFrame([result_dict])
    desired_columns = list(dict.fromkeys(canonical_columns + list(row_df.columns)))
    write_header = not checkpoint_path.exists()

    if not write_header:
        with open(checkpoint_path, newline="") as f:
            try:
                header = next(csv.reader(f))
            except StopIteration:
                write_header = True
            else:
                missing_columns = [col for col in desired_columns if col not in header]
                if missing_columns:
                    upgraded_columns = list(dict.fromkeys(header + missing_columns))
                    try:
                        existing_df = pd.read_csv(checkpoint_path, quoting=csv.QUOTE_ALL)
                    except Exception:
                        # If an existing checkpoint is already malformed, avoid
                        # rewriting it here. The current row will still be
                        # aligned to the existing header to prevent making the
                        # file worse.
                        pass
                    else:
                        existing_df.reindex(columns=upgraded_columns).to_csv(
                            checkpoint_path,
                            mode='w',
                            header=True,
                            index=False,
                            quoting=csv.QUOTE_ALL,
                        )
                        header = upgraded_columns

                # Keep the checkpoint rectangular even when a failed/variant
                # result has fewer or extra keys than the first row that
                # established the header.
                row_df = row_df.reindex(columns=header)
    else:
        row_df = row_df.reindex(columns=desired_columns)

    row_df.to_csv(
        checkpoint_path,
        mode='a',
        header=write_header,
        index=False,
        quoting=csv.QUOTE_ALL,
    )


def filter_df_with_checkpoints(df, checkpoint_df, id_cols=None):
    """Check which rows we already have processed and remove them from df.
    
    Args:
        df: DataFrame to filter.
        checkpoint_df: DataFrame of already-processed rows.
        id_cols: List of column names used as row identifiers.
            Defaults to [SESSION_ID, FLOW_SEQUENCE, EVENT_ID] for
            backward compatibility.
    """
    if checkpoint_df.empty:
        return df
    if df.empty:
        return df

    if 'error' in checkpoint_df.columns:
        error_mask = checkpoint_df['error'].astype(str).str.lower().isin({'true', '1', '1.0', 'yes'})
        checkpoint_df = checkpoint_df[~error_mask].copy()
        if checkpoint_df.empty:
            print(f"{pe['info']} Checkpoint only contains errored rows; retrying all input rows")
            return df
    
    if id_cols is None:
        id_cols = [C.SESSION_ID, C.FLOW_SEQUENCE, C.EVENT_ID]

    missing_cols = [col for col in id_cols if col not in df.columns or col not in checkpoint_df.columns]
    if missing_cols:
        raise ValueError("Missing ID columns for checkpoint filtering.")
    
    df = df.copy()
    checkpoint_df = checkpoint_df.copy()

    df['_temp_key'] = df[id_cols].astype(str).apply(lambda row: '|'.join(row), axis=1)
    checkpoint_df['_temp_key'] = checkpoint_df[id_cols].astype(str).apply(lambda row: '|'.join(row), axis=1)

    df_filtered = df[~df['_temp_key'].isin(checkpoint_df['_temp_key'])].copy()
    df_filtered.drop(columns=['_temp_key'], inplace=True)

    print(f"{pe['info']} Filtered {len(df) - len(df_filtered)} rows from checkpoint")
    print(f"{pe['info']} Remaining rows to process: {len(df_filtered)}")
    return df_filtered


def prepare_eval_sample(
    df,
    evals_name:str = None,
    version: str = None,
    suffix: str = None,
    reasoning_effort: str = None,
    num_rows: int = None,
    file_prefix: str = "evals_"
):
    """
    Prepares a sample DataFrame and constructs a file name.

    The result filename is derived from the experiment name. When num_rows
    is set (test run), '_test' is appended to distinguish from production
    runs. No dates are included — the experiment ID is the unique key.

    Args:
        df: Input DataFrame or Spark DataFrame.
        version: Version string (e.g., "v1").
        suffix: Optional suffix for the file name.
        reasoning_effort: Reasoning effort string for the file name.
        num_rows: If set, limits the sample to this many rows and
            appends '_test' to the filename.
        file_prefix: Prefix for the file name.

    Returns:
        Tuple of (sample_df, file_name)
    """
    # File name construction
    file_name = file_prefix

    if evals_name:
        file_name += f"{evals_name}"
    
    if reasoning_effort:
        file_name += f"_{reasoning_effort}"

    if version:
        file_name += f"_{version}"
    
    if suffix:
        file_name += f"_{suffix}"

    if num_rows:
        file_name += "_test"

    file_name += ".csv"

    if num_rows:
        if isinstance(df, pd.DataFrame):
            sample_df = df.head(num_rows)
        elif isinstance(df, SparkDataFrame):
            sample_df = df.limit(num_rows)
        elif isinstance(df, list):
            sample_df = df[:num_rows]
        else:
            sample_df = df
    else:
        sample_df = df

    if isinstance(sample_df, SparkDataFrame):
        sample_df = sample_df.toPandas()
    elif isinstance(sample_df, list):
        sample_df = pd.DataFrame(sample_df)

    print(f"Sample size: {len(sample_df):_}")
    print(f"File name: {file_name}")

    return sample_df, file_name
