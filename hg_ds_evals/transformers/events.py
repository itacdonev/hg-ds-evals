# HEY GEORGE - EVALS
# Functions and classes for transforming and processing events data.
# The table base for these transformations should be silver_heygeorge_events table.

# events.py
#=============================================================

from pyspark.ml import Transformer
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from ds_common.config.config import HGCol as C


class FilterNotNullEnums(Transformer):
    """
    Filter ENUMs from events data so that only events with both RAG_01 (Phase I) 
    and RAG_04 (Phase II) ENUM(s) are retained.
    """
    def _transform(self, df:DataFrame) -> DataFrame:
        return df.filter(F.col(C.ENUM_PHASE_I).isNotNull() &
                         F.col(C.ENUM_PHASE_II).isNotNull())