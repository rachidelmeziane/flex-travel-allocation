from conf.config import TravelDirection
from flex_travel_allocation.monte_carlo import monte_carlo_pax_by_hour


def _activations(spark, timestamps):
    return spark.createDataFrame(
        [(value,) for value in timestamps], "activation_timestamp_utc string"
    ).selectExpr("to_timestamp(activation_timestamp_utc) as activation_timestamp_utc")


def test_single_bucket_gives_zero_variance_bands(spark):
    activations = _activations(spark, ["2025-02-04 06:48:47", "2025-02-04 06:50:00"])
    minutes = spark.createDataFrame([(15, 1.0)], "minutes_to_add int, probability double")
    hours = spark.createDataFrame([(9, 1.0)], "hours_between_there_and_back int, probability double")

    rows = {
        (row["travel_direction"], row["travel_hour"]): row
        for row in monte_carlo_pax_by_hour(activations, minutes, hours, seeds=[1, 2, 3]).collect()
    }

    there = rows[(TravelDirection.THERE.value, 8)]
    assert there["mean_pax"] == 2
    assert there["std_pax"] == 0
    assert there["lower_pax"] == 2 and there["upper_pax"] == 2
    assert rows[(TravelDirection.BACK.value, 17)]["mean_pax"] == 2


def test_mean_pax_totals_two_per_activation(spark):
    activations = _activations(spark, ["2025-02-04 06:48:47", "2025-02-05 07:10:00", "2025-02-06 08:00:00"])
    minutes = spark.createDataFrame([(0, 0.5), (30, 0.5)], "minutes_to_add int, probability double")
    hours = spark.createDataFrame([(2, 0.5), (9, 0.5)], "hours_between_there_and_back int, probability double")

    total_mean = (
        monte_carlo_pax_by_hour(activations, minutes, hours, seeds=[11, 22, 33, 44])
        .groupBy()
        .sum("mean_pax")
        .first()[0]
    )

    assert round(total_mean) == 6  # 3 activations x (there + back)
