import datetime as dt

from flex_travel_allocation.models.data_prep import build_unit_table

_ACTIVATION_TICKET = "2025" + "0" * 16 + "100"  # sale year + ticket_number padded to 19


def _articles(spark):
    return spark.createDataFrame(
        [(256, "Flex Abonnement 6", 6)],
        "article_code_int short, article_description_nl string, maximum_activations_allowed int",
    )


def _sales(spark):
    return spark.createDataFrame(
        [(100, "Flex Abonnement 6", "2025-01-01", "2025-01-01", "2025-01-31", 90.0)],
        "ticket_number long, article_description_nl string, sales_date string, "
        "validity_start_date string, validity_end_date string, net_sales_euro double",
    ).selectExpr(
        "ticket_number", "article_description_nl",
        "to_date(sales_date) as sales_date",
        "to_date(validity_start_date) as validity_start_date",
        "to_date(validity_end_date) as validity_end_date",
        "net_sales_euro",
    )


def _activations(spark):
    return spark.createDataFrame(
        [
            (_ACTIVATION_TICKET, "2025-01-05", 1, "2025-01-05 06:00:00"),
            (_ACTIVATION_TICKET, "2025-01-10", 7, "2025-01-10 06:00:00"),  # over the 6-activation cap
            (_ACTIVATION_TICKET, "2025-02-05", 2, "2025-02-05 06:00:00"),  # outside the validity window
        ],
        "ticket_number_activations string, activation_date string, "
        "activation_sequence int, activation_timestamp_utc string",
    ).selectExpr(
        "ticket_number_activations",
        "to_date(activation_date) as activation_date",
        "activation_sequence",
        "to_timestamp(activation_timestamp_utc) as activation_timestamp_utc",
    )


def test_build_unit_table(spark):
    units = build_unit_table(
        sales=_sales(spark), articles=_articles(spark), activations=_activations(spark),
        as_of=dt.date(2025, 3, 1),
    ).collect()
    assert len(units) == 1
    row = units[0]
    assert row["event"] == 1 and row["travelled"] == 1
    assert row["n_activations"] == 1          # only the in-window, within-cap activation counts
    assert row["lag_days"] == 4               # 2025-01-05 minus 2025-01-01
    assert row["exposure_days"] == 26.0       # min(validity_end, as_of) - first_activation = 01-31 - 01-05
