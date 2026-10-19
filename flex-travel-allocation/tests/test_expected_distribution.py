from conf.config import TravelDirection
from flex_travel_allocation.expected_distribution import expected_pax_by_hour


def _activations(spark, timestamps):
    return spark.createDataFrame(
        [(value,) for value in timestamps], "activation_timestamp_utc string"
    ).selectExpr("to_timestamp(activation_timestamp_utc) as activation_timestamp_utc")


def test_expected_pax_totals_two_per_activation(spark):
    activations = _activations(spark, ["2025-02-04 06:48:47", "2025-02-05 07:10:00"])
    minutes = spark.createDataFrame(
        [(0, 0.4), (30, 0.6)], "minutes_to_add int, probability double"
    )
    hours = spark.createDataFrame(
        [(2, 0.5), (9, 0.5)], "hours_between_there_and_back int, probability double"
    )

    total = (
        expected_pax_by_hour(activations, minutes, hours)
        .groupBy()
        .sum("expected_pax")
        .first()[0]
    )

    assert round(total) == 4  # 2 activations x (there + back)


def test_expected_pax_places_hours_by_direction(spark):
    # 06:48:47 UTC + 15 min = 07:03:47 UTC = 08:03 Europe/Brussels (winter). Back = 08 + 9 = 17.
    activations = _activations(spark, ["2025-02-04 06:48:47"])
    minutes = spark.createDataFrame([(15, 1.0)], "minutes_to_add int, probability double")
    hours = spark.createDataFrame(
        [(9, 1.0)], "hours_between_there_and_back int, probability double"
    )

    rows = {
        (row["travel_direction"], row["travel_hour"]): round(row["expected_pax"])
        for row in expected_pax_by_hour(activations, minutes, hours).collect()
    }

    assert rows[(TravelDirection.THERE.value, 8)] == 1
    assert rows[(TravelDirection.BACK.value, 17)] == 1
