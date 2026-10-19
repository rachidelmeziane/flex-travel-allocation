import math

from flex_travel_allocation.models.scoring import score_units


def test_score_units(spark):
    units = spark.createDataFrame(
        [("Flex Abonnement 6", 10.0, "2026-01-15", "2026-01-01", 100.0)],
        "article_description_nl string, exposure_days double, as_of string, sales_date string, net_sales_euro double",
    ).selectExpr(
        "article_description_nl", "exposure_days",
        "to_date(as_of) as as_of", "to_date(sales_date) as sales_date", "net_sales_euro",
    )
    parameters = spark.createDataFrame(
        [("Flex Abonnement 6", 0.9, 0.5, 0.2)],
        "article_description_nl string, p_inf double, lambda_week double, nb_rate_per_day double",
    )
    scored = score_units(units, parameters).collect()[0]
    assert abs(scored["expected_trips"] - 0.2 * 10.0) < 1e-9
    assert abs(scored["weeks_since_sale"] - 2.0) < 1e-9
    assert abs(scored["p_activated_by_now"] - 0.9 * (1 - math.exp(-1.0))) < 1e-6
    assert abs(scored["expected_breakage_value"] - (1 - 0.9) * 100.0) < 1e-9
