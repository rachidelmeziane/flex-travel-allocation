"""Allocate Flex sales to travels while respecting validity periods and activation limits.

Pipeline: match each sale to its activations, keep only activations inside the
subscription validity window and within the product activation limit, then expand
every retained activation into a there-travel and a back-travel (each one pax).
"""

from __future__ import annotations

from functools import partial

import pyspark.sql.functions as sf
from pyspark.sql import DataFrame

from conf.config import TravelDirection
from flex_travel_allocation.ticket_matching import add_activation_ticket_key
from flex_travel_allocation.travel_timing import add_back_travel, add_there_travel


def _select_valid_activations(activated_sales: DataFrame) -> DataFrame:
    return activated_sales.filter(
        (sf.col("activation_date") >= sf.col("validity_start_date"))
        & (sf.col("activation_date") <= sf.col("validity_end_date"))
        & (sf.col("activation_sequence") <= sf.col("maximum_activations_allowed"))
    )


def join_sales_to_activations(
    sales: DataFrame, articles: DataFrame, activations: DataFrame
) -> DataFrame:
    return (
        sales.transform(add_activation_ticket_key)
        .join(sf.broadcast(articles), on="article_description_nl", how="inner")
        .join(activations, on="ticket_number_activations", how="inner")
        .transform(_select_valid_activations)
    )


def _to_travel_records(travels: DataFrame) -> DataFrame:
    shared_columns = ["ticket_number", "article_code_int", "article_description_nl"]
    there = travels.select(
        *shared_columns,
        sf.lit(TravelDirection.THERE.value).cast("byte").alias("travel_direction"),
        sf.col("there_travel_date").alias("travel_date"),
        sf.col("there_travel_hour").alias("travel_hour"),
        sf.col("there_travel_timestamp").alias("travel_timestamp"),
        sf.lit(1).cast("byte").alias("pax"),
    )
    back = travels.select(
        *shared_columns,
        sf.lit(TravelDirection.BACK.value).cast("byte").alias("travel_direction"),
        sf.col("back_travel_date").alias("travel_date"),
        sf.col("back_travel_hour").alias("travel_hour"),
        sf.col("back_travel_timestamp").alias("travel_timestamp"),
        sf.lit(1).cast("byte").alias("pax"),
    )
    return there.unionByName(back)


def build_travels(
    activated_activations: DataFrame,
    minutes_distribution: DataFrame,
    hours_distribution: DataFrame,
    seed: int,
) -> DataFrame:
    return (
        activated_activations.transform(
            partial(add_there_travel, minutes_distribution=minutes_distribution, seed=seed)
        )
        .transform(partial(add_back_travel, hours_distribution=hours_distribution, seed=seed + 1))
        .transform(_to_travel_records)
    )


def allocate_sales_to_travels(
    sales: DataFrame,
    articles: DataFrame,
    activations: DataFrame,
    minutes_distribution: DataFrame,
    hours_distribution: DataFrame,
    seed: int,
) -> DataFrame:
    return join_sales_to_activations(
        sales=sales, articles=articles, activations=activations
    ).transform(
        partial(
            build_travels,
            minutes_distribution=minutes_distribution,
            hours_distribution=hours_distribution,
            seed=seed,
        )
    )
