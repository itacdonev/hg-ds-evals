# Preprocessing

Use-case-specific data preprocessing steps that should be run **before** calling
`run_experiment()`.  These were originally embedded in `run_evals.py` and have
been extracted so the eval runner stays generic.

## Fallback preprocessing example

```python
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from ds_common.config.config import HGCol as C, HGTbl as T
from hg_ds_evals.transformers.events import FilterNotNullEnums

spark = SparkSession.builder.getOrCreate()

# Load raw data
input_data = spark.read.table(f"{T.DBX_CATALOG}.{INPUT_TBL_SCHEMA}.{INPUT_TBL_NAME}")

# Rename legacy columns (remove once input table is fixed)
input_data = input_data.withColumnsRenamed({
    "ENUM_top_50": C.ENUM_PHASE_I,
    "ENUM_final": C.ENUM_PHASE_II,
})

# Filter rows with non-null ENUMs
filter_notnull_enums = FilterNotNullEnums()
dp_pipeline = Pipeline(stages=[filter_notnull_enums])
df = dp_pipeline.fit(input_data).transform(input_data)
```

After preprocessing, pass `df` (as a Spark DataFrame) to whatever sampling /
eval logic you need.
