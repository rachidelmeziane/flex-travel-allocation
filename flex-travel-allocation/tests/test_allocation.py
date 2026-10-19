from conf.config import TravelDirection
from flex_travel_allocation.allocation import (
    allocate_sales_to_travels,
    join_sales_to_activations,
)

_ACTIVATION_TICKET = "2025" + "0" * 16 + "100"  # sale year + ticket_number padded to 19


def _articles(spark):
    return spark.createDataFrame(
        [(256, "Flex Abonnement 6", 6)],
        "article_code_int short, article_description_nl string, maximum_activations_allowed int",
    )


def _sales(spark):
    return spark.createDataFrame(
        [(100, "Flex Abonnement 6", "2025-01-01", "2025-01-01", "2025-01-31")],
        "ticket_number long, article_description_nl string, sales_date string, "
        "validity_start_date string, validity_end_date string",
    ).selectExpr(
        "ticket_number",
        "article_description_nl",
        "to_date(sales_date) as sales_date",
        "to_date(validity_start_date) as validity_start_date",
        "to_date(validity_end_date) as validity_end_date",
    )


def _activations(spark):
    return spark.createDataFrame(
        [
            (_ACTIVATION_TICKET, "2025-01-05", 1, "2025-01-05 06:00:00"),
            (_ACTIVATION_TICKET, "2025-01-10", 7, "2025-01-10 06:00:00"),
            (_ACTIVATION_TICKET, "2025-02-05", 2, "2025-02-05 06:00:00"),
        ],
        "ticket_number_activations string, activation_date string, "
        "activation_sequence int, activation_timestamp_utc string",
    ).selectExpr(
        "ticket_number_activations",
        "to_date(activation_date) as activation_date",
        "activation_sequence",
        "to_timestamp(activation_timestamp_utc) as activation_timestamp_utc",
    )


def _single_value_distribution(spark, value_column):
    return spark.createDataFrame([(1, 1.0)], f"{value_column} int, probability double")


def test_join_keeps_only_activations_within_validity_and_activation_limit(spark):
    result = join_sales_to_activations(
        sales=_sales(spark), articles=_articles(spark), activations=_activations(spark)
    )

    assert result.count() == 1
    assert result.first()["activation_sequence"] == 1


def test_allocate_produces_one_there_and_one_back_travel_per_valid_activation(spark):
    travels = allocate_sales_to_travels(
        sales=_sales(spark),
        articles=_articles(spark),
        activations=_activations(spark),
        minutes_distribution=_single_value_distribution(spark, "minutes_to_add"),
        hours_distribution=_single_value_distribution(spark, "hours_between_there_and_back"),
        seed=42,
    )

    directions = sorted(row["travel_direction"] for row in travels.collect())
    assert directions == [TravelDirection.THERE.value, TravelDirection.BACK.value]


def test_allocated_pax_equals_twice_the_valid_activations(spark):
    valid_activations = join_sales_to_activations(
        sales=_sales(spark), articles=_articles(spark), activations=_activations(spark)
    ).count()

    travels = allocate_sales_to_travels(
        sales=_sales(spark),
        articles=_articles(spark),
        activations=_activations(spark),
        minutes_distribution=_single_value_distribution(spark, "minutes_to_add"),
        hours_distribution=_single_value_distribution(spark, "hours_between_there_and_back"),
        seed=42,
    )

    total_pax = travels.groupBy().sum("pax").first()[0]
    assert total_pax == 2 * valid_activations
