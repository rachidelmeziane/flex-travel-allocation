# Apply fitted model parameters to units as Spark column expressions (no UDF, no scipy).
from __future__ import annotations

import pyspark.sql.functions as sf
from pyspark.sql import DataFrame

from flex_travel_allocation.models.parameters import SCORING_COLUMNS


def score_units(units: DataFrame, parameters: DataFrame) -> DataFrame:
    params = parameters.select(*SCORING_COLUMNS)
    return (
        units.join(sf.broadcast(params), on="article_description_nl", how="left")
        .withColumn("expected_trips", sf.col("nb_rate_per_day") * sf.col("exposure_days"))
        .withColumn("weeks_since_sale", sf.datediff(sf.col("as_of"), sf.col("sales_date")) / sf.lit(7.0))
        .withColumn(
            "p_activated_by_now",
            sf.col("p_inf") * (sf.lit(1.0) - sf.exp(-sf.col("lambda_week") * sf.col("weeks_since_sale"))),
        )
        .withColumn("expected_breakage_value", (sf.lit(1.0) - sf.col("p_inf")) * sf.col("net_sales_euro"))
    )
