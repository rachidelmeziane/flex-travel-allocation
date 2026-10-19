from flex_travel_allocation.revenue_risk import revenue_risk_by_product

_ACTIVATION_TICKET = "2025" + "0" * 16 + "100"


def _articles(spark):
    return spark.createDataFrame(
        [(256, "Flex Abonnement 6", 6)],
        "article_code_int short, article_description_nl string, maximum_activations_allowed int",
    )


def _sales(spark):
    return spark.createDataFrame(
        [(100, "Flex Abonnement 6", 60.0, "2025-01-01", "2025-01-01", "2025-01-31")],
        "ticket_number long, article_description_nl string, net_sales_euro double, "
        "sales_date string, validity_start_date string, validity_end_date string",
    ).selectExpr(
        "ticket_number",
        "article_description_nl",
        "net_sales_euro",
        "to_date(sales_date) as sales_date",
        "to_date(validity_start_date) as validity_start_date",
        "to_date(validity_end_date) as validity_end_date",
    )


def _activations(spark):
    return spark.createDataFrame(
        [
            (_ACTIVATION_TICKET, "2025-01-05", 1, "2025-01-05 06:00:00"),
            (_ACTIVATION_TICKET, "2025-01-06", 2, "2025-01-06 06:00:00"),
            (_ACTIVATION_TICKET, "2025-01-07", 3, "2025-01-07 06:00:00"),
            (_ACTIVATION_TICKET, "2025-02-10", 4, "2025-02-10 06:00:00"),  # outside validity
        ],
        "ticket_number_activations string, activation_date string, "
        "activation_sequence int, activation_timestamp_utc string",
    ).selectExpr(
        "ticket_number_activations",
        "to_date(activation_date) as activation_date",
        "activation_sequence",
        "to_timestamp(activation_timestamp_utc) as activation_timestamp_utc",
    )


def test_revenue_risk_quantifies_underuse_and_non_payment(spark):
    row = revenue_risk_by_product(
        sales=_sales(spark),
        articles=_articles(spark),
        activations=_activations(spark),
        non_payment_rate=0.1,
    ).first()

    assert row["used_activations"] == 3            # 4th activation is outside validity
    assert row["sales_revenue"] == 60.0
    assert row["activation_revenue"] == 30.0       # 3 x (60 / 6)
    assert row["utilisation"] == 0.5               # 3 of 6 days
    assert row["underuse_loss"] == 30.0
    assert row["non_payment_loss"] == 3.0          # 30 x 0.1
    assert row["total_loss"] == 33.0               # 60 - 30 x 0.9
    assert abs(row["loss_pct"] - 0.55) < 1e-9
