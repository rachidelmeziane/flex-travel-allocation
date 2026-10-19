"""Reconcile the ticket-number format inconsistency between sales and activations.

Business assumption (verified on the provided data, 37 675/37 675 sales match):
the activation ticket number is the four-digit sale year followed by the sales
``ticket_number`` left-padded with zeros to 19 digits (23 characters in total),
e.g. sales ``3533611089227163`` -> activation ``20250003533611089227163``.
"""

from __future__ import annotations

import pyspark.sql.functions as sf
from pyspark.sql import DataFrame

_ACTIVATION_TICKET_LENGTH = 19


def add_activation_ticket_key(sales: DataFrame) -> DataFrame:
    return sales.withColumn(
        "ticket_number_activations",
        sf.concat(
            sf.year("sales_date").cast("string"),
            sf.lpad(sf.col("ticket_number").cast("string"), _ACTIVATION_TICKET_LENGTH, "0"),
        ),
    )
