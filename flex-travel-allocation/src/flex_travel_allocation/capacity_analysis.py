"""Train-filling (capacity) analysis: sales vs activation-allocation demand by hour.

The allocation gives the *actual* pax per hour. Capacity planned from *sales* assumes
every paid day is travelled, so the potential demand per hour is the allocated demand
divided by each product's utilisation (used / paid-for days). Comparing the two shows
how much capacity a sales-based plan over-provisions and how full trains actually run.
"""

from __future__ import annotations

import pyspark.sql.functions as sf
from pyspark.sql import DataFrame


def train_filling_by_hour(travels: DataFrame, product_utilisation: DataFrame) -> DataFrame:
    return (
        travels.groupBy("travel_hour", "article_description_nl")
        .agg(sf.sum("pax").alias("allocation_pax"))
        .join(sf.broadcast(product_utilisation), on="article_description_nl", how="inner")
        .withColumn("_sales_potential_pax", sf.col("allocation_pax") / sf.col("utilisation"))
        .groupBy("travel_hour")
        .agg(
            sf.sum("allocation_pax").alias("allocation_pax"),
            sf.round(sf.sum("_sales_potential_pax")).alias("sales_potential_pax"),
        )
        .orderBy("travel_hour")
    )
