from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window as W

from ds_common.config.config import HGCol as C
from hg_ds_evals.common.base_transformers import BaseTransformer


class SessionHasCarousel(BaseTransformer):
    """
    Transformer to flag if a session contains a carousel interaction.
    The carousel interaction is identified by the presence of the 'carousel' 
    text in the message type column.
    
    Returns:
        - boolean colulmn session_has_carousel
    """

    def _transform(self, df: DataFrame) -> DataFrame:
         return df.withColumn(
              "session_has_carousel",
              F.max(
                   F.when(F.col(C.MESSAGE_TYPE) == "carousel", 1).otherwise(0)
                ).over(W.partitionBy(C.SESSION_ID)))

class SessionHasQuickReply(BaseTransformer):
    """
    Transformer to flag if a session contains a quick reply interaction.
    The quick reply interaction is identified by the presence of the 'quick_reply' 
    text in the message type column.
    
    Returns:
        - boolean colulmn session_has_quick_reply
    """

    def _transform(self, df: DataFrame) -> DataFrame:
         return df.withColumn(
              "session_has_quick_reply",
              F.max(
                   F.when(F.col(C.MESSAGE_TYPE) == "quick_reply", 1).otherwise(0)
                ).over(W.partitionBy(C.SESSION_ID)))

class SessionHasKBX(BaseTransformer):
    """
    Transformer to flag if a session contains a KBX interaction.
    The KBX interaction is identified by the presence of the 'RAG_01'
    and 'RAG_04' event in the event_key column.
    
    Returns:
        - boolean colulmn session_has_kbx
    """

    def _transform(self, df: DataFrame) -> DataFrame:
         return df.withColumn(
              "session_has_kbx",
              F.max(
                   F.when(F.col(C.EVENT_KEY).isin(["RAG_01","RAG_04"]), 1).otherwise(0)
                ).over(W.partitionBy(C.SESSION_ID)))

class SessionComplexityFeature(BaseTransformer):
    """
    Transformer to create a session complexity feature based on the presence of
    quick reply, KBX, and carousel interactions within a session.
    The session complexity is categorized as follows:
        - 0: No complex interactions (no quick reply, no KBX, no carousel)
        - 1: Simple complex interactions (either quick reply or carousel present)
        - 2: Advanced complex interactions (KBX present, regardless of others)
        
    Returns:
        - integer column session_complex_flag
    """
    
    def _transform(self, df: DataFrame) -> DataFrame:
        return df.withColumn(
            "session_complex_flag",
            F.when(
                (F.col("session_has_quick_reply") == 0)
                & (F.col("session_has_kbx") == 0)
                & (F.col("session_has_carousel") == 0),
                0
            )
            .when(
                (F.col("session_has_quick_reply") == 1)
                | (F.col("session_has_carousel") == 1),
                1
            )
            .otherwise(2)
        )

class SessionIsEmpty(BaseTransformer):
    """Transformer to flag if a session is empty.
    A session is considered empty if there are no messages sent or received.
    Returns:
        - boolean column empty_session_flag
    """
    
    def _transform(self, df: DataFrame) -> DataFrame:
        return df.withColumn(
            "empty_session_flag",
            F.when(
                (F.col("cnt_msg_sent") > 0) & (F.col("cnt_msg_received") > 0),
                0
            ).otherwise(1)
        )