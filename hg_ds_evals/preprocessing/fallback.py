"""
Fallback-specific preprocessing steps.

Extracted from run_evals.py so the eval runner stays use-case agnostic.
Run these steps *before* calling `run_experiment()`.

Usage:
    from hg_ds_evals.preprocessing.fallback import preprocess_fallback_data
    df = preprocess_fallback_data(spark, input_schema, input_table)
"""

from pyspark.ml import Pipeline
from pyspark.sql import DataFrame, SparkSession
from ds_common.config.config import HGCol as C, HGTbl as T
from hg_ds_evals.transformers.events import FilterNotNullEnums


def preprocess_fallback_data(
    spark: SparkSession,
    input_schema: str,
    input_table: str,
    rename_legacy_columns: bool = True,
) -> DataFrame:
    """
    Load and preprocess fallback eval data.

    Steps:
        1. Read from ``{DBX_CATALOG}.{input_schema}.{input_table}``
        2. (Optional) Rename legacy ENUM columns to the canonical names
        3. Filter rows where both Phase-I and Phase-II ENUMs are non-null

    Args:
        spark: Active SparkSession.
        input_schema: Schema (database) name inside the DBX catalog.
        input_table: Table name to read.
        rename_legacy_columns: If True, rename ``ENUM_top_50`` → ``ENUM_PHASE_I``
            and ``ENUM_final`` → ``ENUM_PHASE_II``.  Set to False once the
            upstream table is fixed.

    Returns:
        Filtered Spark DataFrame ready for eval sampling.
    """
    input_data = spark.read.table(f"{T.DBX_CATALOG}.{input_schema}.{input_table}")

    # TODO: Remove once the input table is updated with correct column names
    if rename_legacy_columns:
        input_data = input_data.withColumnsRenamed({
            "ENUM_top_50": C.ENUM_PHASE_I,
            "ENUM_final": C.ENUM_PHASE_II,
        })

    filter_notnull_enums = FilterNotNullEnums()
    dp_pipeline = Pipeline(stages=[filter_notnull_enums])
    df = dp_pipeline.fit(input_data).transform(input_data)

    return df
