"""Estimate travel times from activations using the survey probability distributions.

Each activation produces a there-travel and a back-travel. The there-travel time is
sampled by adding ``minutes_to_add`` to the activation timestamp; the back-travel hour
is sampled by adding ``hours_between_there_and_back`` to the there-travel. Sampling is
weighted by the survey probabilities, deterministic for a given seed, and implemented
with native Spark operations (a cumulative-probability range join) rather than a UDF.
"""

from __future__ import annotations

from functools import partial

import pyspark.sql.functions as sf
from pyspark.sql import DataFrame, Window

from conf.config import LOCAL_TIMEZONE

_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3600


def _add_cumulative_bounds(distribution: DataFrame, value_column: str) -> DataFrame:
    ordered_by_value = Window.orderBy(value_column)
    whole_distribution = Window.partitionBy()
    return (
        distribution.withColumn(
            "_normalised_probability",
            sf.col("probability") / sf.sum("probability").over(whole_distribution),
        )
        .withColumn("_cumulative_upper", sf.sum("_normalised_probability").over(ordered_by_value))
        .withColumn(
            "_cumulative_lower", sf.col("_cumulative_upper") - sf.col("_normalised_probability")
        )
        .withColumn(
            "_cumulative_upper",
            sf.when(
                sf.col("_cumulative_upper")
                == sf.max("_cumulative_upper").over(whole_distribution),
                sf.lit(1.0),
            ).otherwise(sf.col("_cumulative_upper")),
        )
        .select(value_column, "_cumulative_lower", "_cumulative_upper")
    )


def sample_from_distribution(
    events: DataFrame,
    distribution: DataFrame,
    value_column: str,
    draw_column: str,
    seed: int,
) -> DataFrame:
    bounds = _add_cumulative_bounds(distribution=distribution, value_column=value_column)
    return (
        events.withColumn(draw_column, sf.rand(seed))
        .join(
            sf.broadcast(bounds),
            on=(sf.col(draw_column) >= sf.col("_cumulative_lower"))
            & (sf.col(draw_column) < sf.col("_cumulative_upper")),
            how="inner",
        )
        .drop(draw_column, "_cumulative_lower", "_cumulative_upper")
    )


def add_there_travel(
    activations: DataFrame, minutes_distribution: DataFrame, seed: int
) -> DataFrame:
    return (
        activations.transform(
            partial(
                sample_from_distribution,
                distribution=minutes_distribution,
                value_column="minutes_to_add",
                draw_column="_there_draw",
                seed=seed,
            )
        )
        .withColumn(
            "there_travel_timestamp",
            (
                sf.col("activation_timestamp_utc").cast("long")
                + sf.col("minutes_to_add") * sf.lit(_SECONDS_PER_MINUTE)
            ).cast("timestamp"),
        )
        .withColumn(
            "_there_travel_local", sf.from_utc_timestamp("there_travel_timestamp", LOCAL_TIMEZONE)
        )
        .withColumn("there_travel_date", sf.to_date("_there_travel_local"))
        .withColumn("there_travel_hour", sf.hour("_there_travel_local").cast("byte"))
        .drop("_there_travel_local", "minutes_to_add")
    )


def add_back_travel(
    there_travels: DataFrame, hours_distribution: DataFrame, seed: int
) -> DataFrame:
    return (
        there_travels.transform(
            partial(
                sample_from_distribution,
                distribution=hours_distribution,
                value_column="hours_between_there_and_back",
                draw_column="_back_draw",
                seed=seed,
            )
        )
        .withColumn(
            "back_travel_timestamp",
            (
                sf.col("there_travel_timestamp").cast("long")
                + sf.col("hours_between_there_and_back") * sf.lit(_SECONDS_PER_HOUR)
            ).cast("timestamp"),
        )
        .withColumn(
            "_back_travel_local", sf.from_utc_timestamp("back_travel_timestamp", LOCAL_TIMEZONE)
        )
        .withColumn("back_travel_date", sf.to_date("_back_travel_local"))
        .withColumn("back_travel_hour", sf.hour("_back_travel_local").cast("byte"))
        .drop("_back_travel_local", "hours_between_there_and_back")
    )
