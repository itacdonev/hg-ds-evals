# HEY GEORGE - EVALS
# General functions for evaluation operations.

# evaluator.py
#=============================================================
import csv
import time
import asyncio
from typing import Any
import pandas as pd
from pathlib import Path
from hg_ds_evals.evals.parsers import parse_single_row_response
from hg_ds_evals.common.utils import update_checkpoint_df, load_checkpoint, filter_df_with_checkpoints
from hg_ds_evals.llm.api_calls import async_call_llm_for_evaluation
from ds_common.config.config import (
    HGCol as C,
    print_emoji as pe,
    llm_models_config,
    )

# Default ID columns (fallback-era convention)
_DEFAULT_ID_COLUMNS = [C.SESSION_ID, C.FLOW_SEQUENCE, C.EVENT_ID]
_CHECKPOINT_ERROR_COLUMNS = ["error", "error_type", "error_message", "raw_output_text"]


def _dedupe_columns(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def _checkpoint_columns_from_config(
    config: dict,
    id_columns: list[str],
    passthrough_columns: list[str],
) -> list[str]:
    """Build the full checkpoint header before the first row is written."""
    rubric = config.get("rubric", {})
    output_fields = [
        field["name"]
        for field in rubric.get("output_schema", {}).get("fields", [])
        if isinstance(field, dict) and field.get("name")
    ]
    rubric_score_fields = []
    for dim in rubric.get("dimensions", []):
        if isinstance(dim, dict) and dim.get("id"):
            rubric_score_fields.extend([f"{dim['id']}_score", f"{dim['id']}_reasoning"])

    return _dedupe_columns(
        list(id_columns)
        + list(passthrough_columns)
        + _CHECKPOINT_ERROR_COLUMNS
        + output_fields
        + rubric_score_fields
    )


def _is_json_parse_error(parsed_result: dict) -> bool:
    return (
        parsed_result.get("error") is True
        and parsed_result.get("error_type") == "JSONDecodeError"
    )


def _get_tpm_limit(config: dict, model_deployment_name: str) -> int | None:
    model_config = config.get("model", {})
    configured_limit = model_config.get("tpm_limit") or model_config.get("tpm")
    if configured_limit is not None:
        return int(configured_limit)

    return llm_models_config.get(model_deployment_name, {}).get("tpm")


async def timed_call(row_dict:dict, 
                     row_data:Any, 
                     client,
                     semaphore:asyncio.Semaphore, 
                     system_prompt:str,
                     config,
                     user_prompt_builder=None):
    """
    Asynchronously calls the LLM evaluation function with timing.

    Args:
        row_dict (dict): The dictionary containing data for the evaluation.
        row_data (Any): Additional data associated with the row.
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent evaluations.
        system_prompt (str): The system prompt to use for the evaluation.
        config: Experiment config dictionary.
        user_prompt_builder: Optional callable(row_dict) -> str for the user prompt.

    Returns:
        tuple: A tuple containing:
            - result (Any): The result from the LLM evaluation.
            - elapsed_time (float): The time taken to perform the evaluation, in seconds.
            - row_data (Any): The original row_data passed to the function.
    """
    start_time = time.time()
    result = await async_call_llm_for_evaluation(
        row_dict, client, semaphore, system_prompt, config,
        user_prompt_builder=user_prompt_builder,
    )
    elapsed_time = time.time() - start_time
    return result, elapsed_time, row_data


async def async_run_evals(df: pd.DataFrame, 
                          system_prompt,
                          client,
                          config: dict,
                          checkpoint_path: Path = None,
                          user_prompt_builder=None,
                          ):

    model_deployment_name = config["model"]["model_deployment_name"]
    max_concurrent_calls = config["model"]["max_concurrent_calls"]
    max_parse_retries = int(config["model"].get("max_parse_retries", 1) or 0)
    # Use passed checkpoint_path, or fall back to config
    if checkpoint_path is None:
        checkpoint_path = Path(config["paths"]["checkpoint_path"])
    else:
        checkpoint_path = Path(checkpoint_path)

    # ID columns used to track / de-duplicate rows across checkpoints
    id_columns = config.get("dataset", {}).get("id_columns", _DEFAULT_ID_COLUMNS)

    # Passthrough columns: preserved in checkpoint but NOT sent to the LLM
    passthrough_columns = config.get("dataset", {}).get("passthrough_columns", [])
    checkpoint_columns = _checkpoint_columns_from_config(config, id_columns, passthrough_columns)

    tpm_limit = _get_tpm_limit(config, model_deployment_name)

    semaphore = asyncio.Semaphore(max_concurrent_calls)

    if df.empty:
        print(f"{pe['info']} No rows to process")
        return None, None

    # ── Filter out already-checkpointed rows ─────────────────────
    if checkpoint_path is not None and checkpoint_path.exists():
        ckp_df = pd.read_csv(
            checkpoint_path,
            quoting=csv.QUOTE_ALL,
            on_bad_lines="warn",
        )
        df = filter_df_with_checkpoints(df, ckp_df, id_cols=id_columns)

    if df.empty:
        print(f"{pe['info']} No rows to process — all rows already checkpointed")
        return None, None

    results = []
    metrics = {}
    token_usage = []
    start_time = time.time()

    # Process in batches
    batch_size = max_concurrent_calls
    rows_list = list(df.iterrows())
    
    for batch_num, batch_start in enumerate(range(0, len(rows_list), batch_size), 1):
        batch_rows = rows_list[batch_start:batch_start + batch_size]
        batch_start_time = time.time()
        
        # Create tasks for this batch
        tasks = []
        row_data_list = []
        
        for idx, row in batch_rows:
            row_data = {'idx': idx, '_row_dict': row.to_dict()}
            for col in id_columns:
                row_data[col] = row[col]
            for col in passthrough_columns:
                row_data[col] = row.get(col)
            
            task = asyncio.create_task(
                timed_call(row.to_dict(), row_data, client, semaphore, system_prompt, config,
                           user_prompt_builder=user_prompt_builder)
            )
            tasks.append(task)
            row_data_list.append(row_data)
        
        print(f"{pe['info']} Processing batch {batch_num}: {len(tasks)} tasks")
        
        # Wait for all tasks in this batch to complete
        batch_results = await asyncio.gather(*tasks)
        
        # Process results and track tokens for this batch
        batch_tokens = 0
        for (result, elapsed_time, row_data) in batch_results:
            parse_retries = 0
            elapsed_total = elapsed_time
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0

            while True:
                result_dict = {'idx': row_data["idx"]}
                for col in id_columns:
                    result_dict[col] = row_data[col]
                for col in passthrough_columns:
                    result_dict[col] = row_data.get(col)

                if isinstance(result, dict) and 'error' in result:
                    result_dict['output_text'] = None
                    result_dict['error'] = True
                    result_dict['error_type'] = result.get('error_type', 'EvaluationError')
                    result_dict['error_message'] = result.get('error', '')
                else:
                    result_dict['output_text'] = result.output_text
                    result_dict['error'] = False

                    usage = result.usage
                    input_tokens += usage.input_tokens
                    output_tokens += usage.output_tokens
                    total_tokens += usage.total_tokens
                    batch_tokens += usage.total_tokens
                    token_usage.append((time.time(), usage.total_tokens))

                parsed_results = parse_single_row_response(
                    result_dict, id_columns=id_columns, passthrough_columns=passthrough_columns
                )

                if not _is_json_parse_error(parsed_results) or parse_retries >= max_parse_retries:
                    break

                parse_retries += 1
                print(
                    f"{pe['warning']} JSON parse failed for row {row_data['idx']}; "
                    f"retrying evaluation ({parse_retries}/{max_parse_retries})"
                )
                result, retry_elapsed, _ = await timed_call(
                    row_data["_row_dict"], row_data, client, semaphore,
                    system_prompt, config, user_prompt_builder=user_prompt_builder
                )
                elapsed_total += retry_elapsed

            if parsed_results.get("error") is True:
                metrics[row_data["idx"]] = {
                    'time_elapsed_sec': elapsed_total,
                    'error': True,
                    'error_type': parsed_results.get("error_type"),
                    'parse_retries': parse_retries,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': total_tokens,
                }
            else:
                metrics[row_data["idx"]] = {
                    'time_elapsed_sec': elapsed_total,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': total_tokens,
                    'parse_retries': parse_retries,
                }

            results.append(result_dict)
            update_checkpoint_df(parsed_results, checkpoint_path, columns=checkpoint_columns)

        # Calculate batch timing
        batch_elapsed = time.time() - batch_start_time
        
        # Calculate TPM metrics
        current_time = time.time()
        token_usage = [(ts, tok) for ts, tok in token_usage if current_time - ts < 60]
        tokens_last_minute = sum(tok for _, tok in token_usage)
        
        print(f"✅ Batch {batch_num} complete: {batch_tokens:,} tokens in {batch_elapsed:.1f}s | "
              f"Last 60s: {tokens_last_minute:,} tokens")
        
        # Wait if needed before next batch
        if tpm_limit and batch_start + batch_size < len(rows_list):
            # Calculate minimum time needed for this batch to respect TPM
            seconds_needed = (batch_tokens / tpm_limit) * 60
            wait_time = max(0, seconds_needed - batch_elapsed)
            
            if wait_time > 0:
                print(f"{pe['warning']} Waiting {wait_time:.1f}s to respect TPM limit ({batch_tokens:,} tokens)")
                await asyncio.sleep(wait_time)
    
    total_elapsed = time.time() - start_time
    total_tokens = sum(m.get('total_tokens', 0) for m in metrics.values())
    avg_tpm = int(total_tokens / total_elapsed * 60) if total_elapsed > 0 else 0
    
    print(f"{pe['done']} All batches complete: {len(results)} rows, {total_tokens:,} tokens, "
          f"{total_elapsed:.1f}s, avg TPM: {avg_tpm:,}")
    
    return results, metrics
