"""Aggregations for the presentation: travel distributions by product, date and hour.

Logic lives here so the analysis notebook only orchestrates and plots.
"""

from __future__ import annotations

import pyspark.sql.functions as sf
from pyspark.sql import DataFrame


def travels_by_product(travels: DataFrame) -> DataFrame:
    return (
        travels.groupBy("article_description_nl")
        .agg(sf.sum("pax").alias("pax"))
        .orderBy("article_description_nl")
    )


def travels_by_date(travels: DataFrame) -> DataFrame:
    return travels.groupBy("travel_date").agg(sf.sum("pax").alias("pax")).orderBy("travel_date")


def travels_by_hour(travels: DataFrame) -> DataFrame:
    return travels.groupBy("travel_hour").agg(sf.sum("pax").alias("pax")).orderBy("travel_hour")


def travels_by_hour_and_direction(travels: DataFrame) -> DataFrame:
    return (
        travels.groupBy("travel_hour", "travel_direction")
        .agg(sf.sum("pax").alias("pax"))
        .orderBy("travel_hour", "travel_direction")
    )


def travels_by_product_and_hour(travels: DataFrame) -> DataFrame:
    return (
        travels.groupBy("article_description_nl", "travel_hour")
        .agg(sf.sum("pax").alias("pax"))
        .orderBy("article_description_nl", "travel_hour")
    )
