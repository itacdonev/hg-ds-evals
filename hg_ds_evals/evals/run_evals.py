"""
Evals Runner Script

Runs evaluations from a YAML configuration file. Supports two config formats:
  1. Experiment YAML (recommended) — single YAML with rubric, model, dataset, paths.
     System prompt is built programmatically from the rubric definition.
  2. Legacy runtime YAML — separate YAML for runtime config + static .md prompt files.

The format is auto-detected: if the YAML has a `rubric` section with `base`, 
the prompt is built from the rubric. Otherwise, it reads a static .md file.

Usage:
    results, metrics = await run_experiment("experiments/fallback_exp_001_baseline.yaml")
"""
from pathlib import Path

from pyspark.sql import SparkSession
import yaml
from ds_common.config.config import (
    HGTbl as T, 
    print_emoji as pe
    )
from hg_ds_evals.llm.api_client import get_api_client
from hg_ds_evals.prompts.builder import PromptBuilder
from hg_ds_evals.prompts.common import read_md_file
from hg_ds_evals.rubrics.loader import (
    load_experiment_config,
    build_rubric_from_config,
    get_experiment_name,
)
from hg_ds_evals.common.utils import (
    load_yaml_config,
    load_checkpoint,
    filter_df_with_checkpoints,
    prepare_eval_sample,
)
from hg_ds_evals.evals.evaluator import async_run_evals


# =============================================================================
# PROMPT RESOLUTION
# =============================================================================

from typing import Optional


def _resolve_template_path(
    raw_path: str | None,
    config_dir: Path,
    template_type: str,
) -> Optional[Path]:
    """Resolve an optional prompt template path from experiment config."""
    if not raw_path:
        return None

    path = Path(raw_path)
    candidates = (
        [path]
        if path.is_absolute()
        else [config_dir / path, Path.cwd() / path]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    print(
        f"⚠ {template_type} template path not found: {raw_path}. "
        "Using embedded default template."
    )
    return None


def _resolve_prompt_and_config(config_file: str) -> tuple[dict, str, str, Optional[PromptBuilder]]:
    """
    Load config and resolve the system prompt based on config format.
    
    Returns:
        Tuple of (config_dict, system_prompt, experiment_name, prompt_builder)
        prompt_builder is None for legacy configs.
    """
    config_path = Path(config_file)
    
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    
    is_experiment = "rubric" in raw and "base" in raw.get("rubric", {})
    
    if is_experiment:
        # Experiment YAML → build prompt from rubric
        config_eval = load_experiment_config(config_file)
        experiment_name = config_eval["experiment"]["name"]
        rubric = build_rubric_from_config(config_eval)
        config_dir = config_path.resolve().parent
        paths_config = config_eval.get("paths", {})

        system_template_path = _resolve_template_path(
            paths_config.get("system_template_path"),
            config_dir,
            "System",
        )
        user_template_path = _resolve_template_path(
            paths_config.get("user_template_path"),
            config_dir,
            "User",
        )
        
        builder = PromptBuilder(
            rubric=rubric,
            system_template_path=system_template_path,
            user_template_path=user_template_path,
        )
        system_prompt = builder.build_system_prompt()
        
        print(f"✓ Rubric: {rubric.metadata.name} (v{rubric.metadata.version})")
        print(f"  Dimensions: {rubric.dimension_ids}")
        print(f"  Input fields: {rubric.input_field_names}")
        print(f"  Pass threshold: {rubric.pass_threshold}")
        print(f"  System template: {system_template_path or 'embedded default'}")
        print(f"  User template: {user_template_path or 'embedded default'}")
        print(f"✓ System prompt built from rubric ({len(system_prompt)} chars)")
    else:
        # Legacy YAML → read static .md prompt file
        config_eval = load_yaml_config(filepath=config_file)
        experiment_name = config_path.stem
        builder = None
        system_prompt = read_md_file(
            file_path=config_eval["paths"]["system_prompt_path"]
        )
        print(f"⚠ Using legacy config (static .md prompt). Consider migrating to experiment YAML.")
        print(f"✓ System prompt loaded from {config_eval['paths']['system_prompt_path']}")
    
    return config_eval, system_prompt, experiment_name, builder


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def run_experiment(config_file: str, verbose: bool = True):
    """
    Run evaluations from a YAML configuration file.
    
    Auto-detects config format:
    - Experiment YAML (has `rubric.base`) → builds system prompt from rubric
    - Legacy YAML (has `paths.system_prompt_path`) → reads static .md file
    
    The experiment name (YAML filename stem) is used for checkpoint files
    and can be used for DBX result table naming.
    
    Args:
        config_file: Path to the YAML configuration file.
        
    Returns:
        Tuple of (results, metrics)
        
    Example:
        results, metrics = await run_experiment("experiments/fallback_exp_001_baseline.yaml")
    """
    # ── Step 1: Load config and resolve system prompt ────────────
    print("=" * 80)
    experiment_name = get_experiment_name(config_file)
    print(f"EXPERIMENT: {experiment_name}")
    print("=" * 80)
    
    print("\n[1/6] Loading configuration...")
    config_eval, system_prompt, experiment_name, prompt_builder = _resolve_prompt_and_config(config_file)
    
    # Build user-prompt callable from rubric (None → legacy fallback template)
    user_prompt_builder = prompt_builder.build_user_prompt if prompt_builder else None
    
    INPUT_TBL_NAME = config_eval["dataset"]["input_dataset"]
    INPUT_TBL_SCHEMA = config_eval["dataset"]["input_schema"]
    CHECKPOINT_DIR = config_eval["paths"].get("checkpoint_dir", 
                         config_eval["paths"].get("checkpoint_path", "checkpoints"))
    NUM_ROWS = config_eval["dataset"]["test_num_rows"]
    ID_COLUMNS = config_eval["dataset"].get("id_columns")

    # ── Step 2: Load input data ──────────────────────────────────
    try:
        spark = SparkSession.builder.getOrCreate()
    except Exception as e:
        print(f"✗ Error getting Spark session: {e}")
        return None, None
    
    print("\n[2/6] Loading input data...")
    input_data = spark.read.table(f"{T.DBX_CATALOG}.{INPUT_TBL_SCHEMA}.{INPUT_TBL_NAME}")
    df = input_data
    print(f"✓ Data loaded with {df.count()} rows")
    # if verbose:
    #     print("  Sample rows:")
    #     df.show(5, truncate=False)
    
    # ── Step 3: Prepare eval sample ──────────────────────────────
    print("\n[3/6] Preparing eval sample...")
    
    # Experiment YAML uses experiment_name directly; legacy uses output_file_name config
    output_file_cfg = config_eval.get("paths", {}).get("output_file_name")
    if output_file_cfg:
        df_sample, file_name_eval = prepare_eval_sample(
            df=df,
            evals_name=output_file_cfg.get("evals_name", experiment_name),
            test_date=output_file_cfg.get("test_date"),
            version=output_file_cfg.get("version"),
            suffix=output_file_cfg.get("suffix"),
            reasoning_effort=config_eval["model"]["reasoning_effort"],
            num_rows=NUM_ROWS,
            file_prefix=output_file_cfg.get("file_prefix", "evals_"),
        )
    else:
        df_sample, file_name_eval = prepare_eval_sample(
            df=df,
            evals_name=experiment_name,
            reasoning_effort=config_eval["model"]["reasoning_effort"],
            num_rows=NUM_ROWS,
        )
    
    cols = config_eval["dataset"]["eval_columns"]
    # Ensure id_columns are always present in the eval DataFrame
    if ID_COLUMNS:
        cols = list(dict.fromkeys(ID_COLUMNS + cols))  # deduplicated, id cols first
    # Include passthrough columns (columns that should be perserved in the final checkpoint table)
    passthrough_cols = config_eval["dataset"].get("passthrough_columns", [])
    if passthrough_cols:
        cols = list(dict.fromkeys(cols + passthrough_cols))
    df_eval = df_sample[cols].copy()
    print(f"✓ Created sample with {len(df_eval)} rows")
    print(f"  Checkpoint file: {file_name_eval}")
    if verbose:
        print("  Sample rows:")
        print(df_eval.head(5))
    
    # ── Step 4: Load checkpoint ──────────────────────────────────
    print("\n[4/6] Loading checkpoints...")
    ckp_df, ckp_path = load_checkpoint(
        checkpoint_file_name=file_name_eval,
        checkpoint_dir=CHECKPOINT_DIR
    )
    print(f"  Checkpoint path: {ckp_path}")
    print(f"  Checkpoint rows: {len(ckp_df)}")
    
    df_eval = filter_df_with_checkpoints(df_eval, ckp_df, id_cols=ID_COLUMNS)
    print(f"✓ Filtered eval dataset to {len(df_eval)} remaining rows")
    
    # ── Step 5: Setup API client ─────────────────────────────────
    print("\n[5/6] Setting up API client...")
    client = get_api_client(
        model_deployment_name=config_eval["model"]["model_deployment_name"],
        api_provider=config_eval["model"]["api_provider"],
        databricks_endpoint_url=config_eval["model"].get("databricks_endpoint_url"),
        databricks_base_url=config_eval["model"].get("databricks_base_url"),
        databricks_workspace_host=config_eval["model"].get("databricks_workspace_host"),
    )
    print("✓ API client ready")
    
    # ── Step 6: Run evals ────────────────────────────────────────
    print("\n[6/6] Running evaluations...")
    print(f"  Model: {config_eval['model']['model_deployment_name']}")
    print(f"  Max concurrent calls: {config_eval['model']['max_concurrent_calls']}")
    print(f"  Reasoning effort: {config_eval['model']['reasoning_effort']}")
    
    results, metrics = await async_run_evals(
        df=df_eval,
        system_prompt=system_prompt,
        client=client,
        config=config_eval,
        checkpoint_path=ckp_path,
        user_prompt_builder=user_prompt_builder,
    )

    if results is None:
        print(f"\n{pe['info']} No rows to process - all rows have been evaluated or filtered out.")
        return None, None
    
    print("\n" + "=" * 80)
    print(f"EXPERIMENT COMPLETED: {experiment_name}")
    print("=" * 80)
    print(f"✓ Processed {len(results)} evaluations")
    print(f"✓ Results saved to: {ckp_path}")
    print(f"✓ Experiment name: {experiment_name}")
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    return results, metrics
