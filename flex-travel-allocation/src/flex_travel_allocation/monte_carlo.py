"""Monte-Carlo confidence bands for the sampled travel-hour distribution.

Repeat the stochastic allocation across several seeds and summarise the pax per hour
and direction as a mean with a confidence band (mean +/- 1.96 standard deviations),
quantifying the uncertainty introduced by the probabilistic sampling.
"""

from __future__ import annotations

from functools import reduce

import pyspark.sql.functions as sf
from pyspark.sql import DataFrame

from conf.config import TravelDirection
from flex_travel_allocation.travel_timing import add_back_travel, add_there_travel

_CONFIDENCE_Z = 1.96


def _replicate_pax_by_hour(
    activations: DataFrame,
    minutes_distribution: DataFrame,
    hours_distribution: DataFrame,
    seed: int,
) -> DataFrame:
    travels = add_back_travel(
        add_there_travel(activations, minutes_distribution, seed),
        hours_distribution,
        seed + 1,
    )
    there = travels.groupBy("there_travel_hour").count().select(
        sf.col("there_travel_hour").alias("travel_hour"),
        sf.lit(TravelDirection.THERE.value).cast("byte").alias("travel_direction"),
        sf.col("count").alias("pax"),
    )
    back = travels.groupBy("back_travel_hour").count().select(
        sf.col("back_travel_hour").alias("travel_hour"),
        sf.lit(TravelDirection.BACK.value).cast("byte").alias("travel_direction"),
        sf.col("count").alias("pax"),
    )
    return there.unionByName(back)


def monte_carlo_pax_by_hour(
    activations: DataFrame,
    minutes_distribution: DataFrame,
    hours_distribution: DataFrame,
    seeds: list[int],
) -> DataFrame:
    replicates = [
        _replicate_pax_by_hour(activations, minutes_distribution, hours_distribution, seed)
        .withColumn("_replicate", sf.lit(index))
        for index, seed in enumerate(seeds)
    ]
    combined = reduce(lambda left, right: left.unionByName(right), replicates)

    # Fill hours a replicate never produced with zero so every band is over all replicates.
    groups = combined.select("travel_hour", "travel_direction").distinct()
    replicate_ids = combined.select("_replicate").distinct()
    filled = (
        groups.crossJoin(replicate_ids)
        .join(combined, on=["travel_hour", "travel_direction", "_replicate"], how="left")
        .fillna(0, subset=["pax"])
    )

    return (
        filled.groupBy("travel_hour", "travel_direction")
        .agg(
            sf.avg("pax").alias("mean_pax"),
            sf.coalesce(sf.stddev_pop("pax"), sf.lit(0.0)).alias("std_pax"),
            sf.min("pax").alias("min_pax"),
            sf.max("pax").alias("max_pax"),
        )
        .withColumn(
            "lower_pax",
            sf.greatest(sf.lit(0.0), sf.col("mean_pax") - sf.lit(_CONFIDENCE_Z) * sf.col("std_pax")),
        )
        .withColumn("upper_pax", sf.col("mean_pax") + sf.lit(_CONFIDENCE_Z) * sf.col("std_pax"))
        .select(
            "travel_hour", "travel_direction", "mean_pax", "std_pax",
            "lower_pax", "upper_pax", "min_pax", "max_pax",
        )
    )
