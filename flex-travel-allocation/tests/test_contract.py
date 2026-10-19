from flex_travel_allocation.allocation import allocate_sales_to_travels

_ACTIVATION_TICKET = "2025" + "0" * 16 + "100"

_EXPECTED_SCHEMA = {
    "ticket_number": "bigint",
    "article_code_int": "smallint",
    "article_description_nl": "string",
    "travel_direction": "tinyint",
    "travel_date": "date",
    "travel_hour": "tinyint",
    "travel_timestamp": "timestamp",
    "pax": "tinyint",
}


def test_travels_output_has_the_expected_schema(spark):
    articles = spark.createDataFrame(
        [(256, "Flex Abonnement 6", 6)],
        "article_code_int short, article_description_nl string, maximum_activations_allowed int",
    )
    sales = spark.createDataFrame(
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
    activations = spark.createDataFrame(
        [(_ACTIVATION_TICKET, "2025-01-05", 1, "2025-01-05 06:00:00")],
        "ticket_number_activations string, activation_date string, "
        "activation_sequence int, activation_timestamp_utc string",
    ).selectExpr(
        "ticket_number_activations",
        "to_date(activation_date) as activation_date",
        "activation_sequence",
        "to_timestamp(activation_timestamp_utc) as activation_timestamp_utc",
    )
    minutes_distribution = spark.createDataFrame(
        [(0, 1.0)], "minutes_to_add int, probability double"
    )
    hours_distribution = spark.createDataFrame(
        [(2, 1.0)], "hours_between_there_and_back int, probability double"
    )

    travels = allocate_sales_to_travels(
        sales=sales,
        articles=articles,
        activations=activations,
        minutes_distribution=minutes_distribution,
        hours_distribution=hours_distribution,
        seed=42,
    )

    assert dict(travels.dtypes) == _EXPECTED_SCHEMA
