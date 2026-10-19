# Unit-level (per subscription sold) table for model estimation and scoring, built in Spark.
from __future__ import annotations

import datetime as dt

import pyspark.sql.functions as sf
from pyspark.sql import DataFrame

from flex_travel_allocation.allocation import join_sales_to_activations


def build_unit_table(
    sales: DataFrame,
    articles: DataFrame,
    activations: DataFrame,
    as_of: dt.date,
) -> DataFrame:
    # as_of censors not-yet-activated units and caps Stage-2 exposure.
    valid = join_sales_to_activations(sales=sales, articles=articles, activations=activations)
    per_ticket = valid.groupBy("ticket_number").agg(
        sf.min("activation_date").alias("first_activation_date"),
        sf.count(sf.lit(1)).alias("n_activations"),
    )
    as_of_col = sf.lit(as_of.isoformat()).cast("date")
    return (
        sales.join(per_ticket, on="ticket_number", how="left")
        .withColumn("activated", sf.col("first_activation_date").isNotNull())
        .withColumn("n_activations", sf.coalesce(sf.col("n_activations"), sf.lit(0)))
        .withColumn("as_of", as_of_col)
        .withColumn("lag_days", sf.datediff(sf.col("first_activation_date"), sf.col("sales_date")))
        .withColumn("censor_days", sf.greatest(sf.datediff(as_of_col, sf.col("sales_date")), sf.lit(0)))
        .withColumn("event", sf.col("activated").cast("int"))
        .withColumn(
            "time",
            sf.greatest(
                sf.when(sf.col("activated"), sf.col("lag_days")).otherwise(sf.col("censor_days")).cast("double"),
                sf.lit(0.0),
            ),
        )
        .withColumn(
            "exposure_days",
            sf.greatest(
                sf.datediff(sf.least(sf.col("validity_end_date"), as_of_col), sf.col("first_activation_date")),
                sf.lit(1),
            ).cast("double"),
        )
        .withColumn("travelled", (sf.col("n_activations") > 0).cast("int"))
    )
