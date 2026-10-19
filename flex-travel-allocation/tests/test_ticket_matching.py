from flex_travel_allocation.ticket_matching import add_activation_ticket_key


def test_add_activation_ticket_key_prefixes_year_and_pads_to_23_characters(spark):
    sales = spark.createDataFrame(
        [(3533611089227163, "2025-01-01")],
        "ticket_number long, sales_date string",
    ).selectExpr("ticket_number", "to_date(sales_date) as sales_date")

    result = add_activation_ticket_key(sales).select("ticket_number_activations").first()[0]

    assert result == "20250003533611089227163"
    assert len(result) == 23
