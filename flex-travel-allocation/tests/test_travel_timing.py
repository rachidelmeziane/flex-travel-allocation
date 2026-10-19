import datetime

from flex_travel_allocation.travel_timing import (
    add_back_travel,
    add_there_travel,
    sample_from_distribution,
)


def test_sample_from_distribution_returns_the_only_bucket_with_probability(spark):
    events = spark.range(20).toDF("event_id")
    distribution = spark.createDataFrame(
        [(15, 1.0), (30, 0.0)], "minutes_to_add int, probability double"
    )

    sampled = sample_from_distribution(
        events=events,
        distribution=distribution,
        value_column="minutes_to_add",
        draw_column="_draw",
        seed=42,
    )

    assert sampled.count() == 20
    assert [row["minutes_to_add"] for row in sampled.collect()] == [15] * 20


def test_sample_from_distribution_only_returns_known_values(spark):
    events = spark.range(50).toDF("event_id")
    distribution = spark.createDataFrame(
        [(5, 0.5), (60, 0.5)], "minutes_to_add int, probability double"
    )

    values = {
        row["minutes_to_add"]
        for row in sample_from_distribution(
            events=events,
            distribution=distribution,
            value_column="minutes_to_add",
            draw_column="_draw",
            seed=7,
        ).collect()
    }

    assert values.issubset({5, 60})


def test_add_there_travel_adds_minutes_and_reports_local_hour(spark):
    activations = spark.createDataFrame(
        [("2025-02-04 06:48:47",)], "activation_timestamp_utc string"
    ).selectExpr("to_timestamp(activation_timestamp_utc) as activation_timestamp_utc")
    minutes_distribution = spark.createDataFrame(
        [(15, 1.0)], "minutes_to_add int, probability double"
    )

    result = add_there_travel(
        activations=activations, minutes_distribution=minutes_distribution, seed=1
    ).first()

    assert result["there_travel_hour"] == 8
    assert result["there_travel_date"] == datetime.date(2025, 2, 4)


def test_add_back_travel_adds_hours_between_to_there_travel(spark):
    there_travels = spark.createDataFrame(
        [("2025-02-04 07:03:47",)], "there_travel_timestamp string"
    ).selectExpr("to_timestamp(there_travel_timestamp) as there_travel_timestamp")
    hours_distribution = spark.createDataFrame(
        [(9, 1.0)], "hours_between_there_and_back int, probability double"
    )

    result = add_back_travel(
        there_travels=there_travels, hours_distribution=hours_distribution, seed=1
    ).first()

    assert result["back_travel_hour"] == 17
