"""Expected (probability-weighted) travel-hour distribution.

Instead of sampling one travel time per activation, expand every activation across the
survey probability buckets to obtain the exact expected pax per hour and direction.
This complements the sampled allocation and is the basis for uncertainty analysis.
"""

from __future__ import annotations

import pyspark.sql.functions as sf
from pyspark.sql import DataFrame, Window

from conf.config import LOCAL_TIMEZONE, TravelDirection

_HOURS_PER_DAY = 24
_SECONDS_PER_MINUTE = 60


def _normalise(distribution: DataFrame) -> DataFrame:
    whole_distribution = Window.partitionBy()
    return distribution.withColumn(
        "probability", sf.col("probability") / sf.sum("probability").over(whole_distribution)
    )


def _there_expected_by_hour(activations: DataFrame, minutes_distribution: DataFrame) -> DataFrame:
    return (
        activations.crossJoin(sf.broadcast(_normalise(minutes_distribution)))
        .withColumn(
            "_there_local",
            sf.from_utc_timestamp(
                (
                    sf.col("activation_timestamp_utc").cast("long")
                    + sf.col("minutes_to_add") * sf.lit(_SECONDS_PER_MINUTE)
                ).cast("timestamp"),
                LOCAL_TIMEZONE,
            ),
        )
        .withColumn("travel_hour", sf.hour("_there_local").cast("byte"))
        .groupBy("travel_hour")
        .agg(sf.sum("probability").alias("expected_pax"))
    )


def _back_expected_by_hour(there_expected: DataFrame, hours_distribution: DataFrame) -> DataFrame:
    return (
        there_expected.crossJoin(sf.broadcast(_normalise(hours_distribution)))
        .withColumn(
            "travel_hour",
            (
                (sf.col("travel_hour") + sf.col("hours_between_there_and_back"))
                % sf.lit(_HOURS_PER_DAY)
            ).cast("byte"),
        )
        .withColumn("_weight", sf.col("expected_pax") * sf.col("probability"))
        .groupBy("travel_hour")
        .agg(sf.sum("_weight").alias("expected_pax"))
    )


def expected_pax_by_hour(
    activations: DataFrame,
    minutes_distribution: DataFrame,
    hours_distribution: DataFrame,
) -> DataFrame:
    there_expected = _there_expected_by_hour(activations, minutes_distribution)
    # Back hour-of-day = (there hour + whole hours between) mod 24, so the expensive
    # activation-level expansion is only needed once (for the there-travel).
    back_expected = _back_expected_by_hour(there_expected, hours_distribution)
    there_records = there_expected.withColumn(
        "travel_direction", sf.lit(TravelDirection.THERE.value).cast("byte")
    )
    back_records = back_expected.withColumn(
        "travel_direction", sf.lit(TravelDirection.BACK.value).cast("byte")
    )
    return there_records.unionByName(back_records).select(
        "travel_hour", "travel_direction", "expected_pax"
    )
