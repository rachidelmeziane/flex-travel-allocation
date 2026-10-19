"""Revenue risk: sales (upfront) vs activation-based (usage) revenue.

Compares the guaranteed sales revenue with revenue recognised per actual activation,
quantifying subscription under-utilisation and the exposure to unpaid activations if
the business moved from charging at sale to charging at activation.

Assumption: a subscription day is priced uniformly, so the value of one activation is
net_sales_euro / maximum_activations_allowed.
"""

from __future__ import annotations

import pyspark.sql.functions as sf
from pyspark.sql import DataFrame

from flex_travel_allocation.allocation import join_sales_to_activations


def revenue_risk_by_product(
    sales: DataFrame,
    articles: DataFrame,
    activations: DataFrame,
    non_payment_rate: float,
) -> DataFrame:
    used_per_ticket = (
        join_sales_to_activations(sales=sales, articles=articles, activations=activations)
        .groupBy("ticket_number")
        .agg(sf.count(sf.lit(1)).alias("used_activations"))
    )
    per_ticket = (
        sales.join(sf.broadcast(articles), on="article_description_nl", how="inner")
        .join(used_per_ticket, on="ticket_number", how="left")
        .fillna(0, subset=["used_activations"])
        .withColumn(
            "_price_per_activation",
            sf.col("net_sales_euro") / sf.col("maximum_activations_allowed"),
        )
        .withColumn(
            "_activation_revenue", sf.col("_price_per_activation") * sf.col("used_activations")
        )
    )
    return (
        per_ticket.groupBy(
            "article_code_int", "article_description_nl", "maximum_activations_allowed"
        )
        .agg(
            sf.count(sf.lit(1)).alias("subscriptions"),
            sf.round(sf.sum("net_sales_euro"), 2).alias("sales_revenue"),
            sf.sum("used_activations").alias("used_activations"),
            sf.round(sf.sum("_activation_revenue"), 2).alias("activation_revenue"),
        )
        .withColumn(
            "capacity_activations", sf.col("subscriptions") * sf.col("maximum_activations_allowed")
        )
        .withColumn("utilisation", sf.col("used_activations") / sf.col("capacity_activations"))
        .withColumn(
            "paid_activation_revenue",
            sf.round(sf.col("activation_revenue") * sf.lit(1 - non_payment_rate), 2),
        )
        .withColumn(
            "underuse_loss", sf.round(sf.col("sales_revenue") - sf.col("activation_revenue"), 2)
        )
        .withColumn(
            "non_payment_loss", sf.round(sf.col("activation_revenue") * sf.lit(non_payment_rate), 2)
        )
        .withColumn(
            "total_loss", sf.round(sf.col("sales_revenue") - sf.col("paid_activation_revenue"), 2)
        )
        .withColumn("loss_pct", sf.col("total_loss") / sf.col("sales_revenue"))
        .orderBy("article_code_int")
    )
