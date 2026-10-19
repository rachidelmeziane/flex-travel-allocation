"""Simulate revenue under the sales vs activation model and quantify the risk."""

import os
import sys

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from pyspark.sql import SparkSession

from conf.config import Table
from conf.params import PARAMS
from flex_travel_allocation.data_handler import DataHandler
from flex_travel_allocation.ingestion import (
    clean_activations,
    clean_articles,
    clean_sales,
)
from flex_travel_allocation.revenue_risk import revenue_risk_by_product

spark = (
    SparkSession.builder.master(PARAMS["spark"]["master"])
    .appName("revenue-simulation")
    .config("spark.sql.shuffle.partitions", str(PARAMS["spark"]["shuffle_partitions"]))
    .config("spark.ui.enabled", str(PARAMS["spark"]["ui_enabled"]).lower())
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

dh = DataHandler.from_config(spark=spark, environment=PARAMS["environment"])
articles = dh.read_table(Table.ARTICLES).transform(clean_articles)
sales = dh.read_table(Table.SALES).transform(clean_sales)
activations = dh.read_table(Table.ACTIVATIONS).transform(clean_activations)

by_product = revenue_risk_by_product(
    sales=sales, articles=articles, activations=activations, non_payment_rate=PARAMS["revenue_risk"]["non_payment_rate"]
).toPandas()
by_product.to_csv("streamlit_app/artifacts/output_revenue_risk.csv", index=False)

total_sales = by_product["sales_revenue"].sum()
total_activation = by_product["activation_revenue"].sum()
total_used = int(by_product["used_activations"].sum())
total_capacity = int(by_product["capacity_activations"].sum())
underuse_total = total_sales - total_activation

lines = []
lines.append("REVENUE RISK: sales (upfront) vs activation (usage) model")
lines.append("=" * 70)
lines.append("")
lines.append("Per product (non-payment rate = 0%):")
header = "%-20s %6s %7s %14s %16s %6s" % (
    "product", "subs", "util%", "sales_rev EUR", "activation_rev", "loss%"
)
lines.append(header)
for row in by_product.itertuples():
    lines.append(
        "%-20s %6d %6.1f%% %14s %16s %5.1f%%"
        % (
            row.article_description_nl,
            row.subscriptions,
            row.utilisation * 100,
            format(row.sales_revenue, ",.0f"),
            format(row.activation_revenue, ",.0f"),
            row.loss_pct * 100,
        )
    )
lines.append("-" * 70)
lines.append(
    "%-20s %6d %6.1f%% %14s %16s %5.1f%%"
    % (
        "TOTAL",
        int(by_product["subscriptions"].sum()),
        (total_used / total_capacity) * 100,
        format(total_sales, ",.0f"),
        format(total_activation, ",.0f"),
        (underuse_total / total_sales) * 100,
    )
)
lines.append("")
lines.append(
    "Activations used: %s of %s paid-for days (%.1f%% utilisation)."
    % (format(total_used, ","), format(total_capacity, ","), total_used / total_capacity * 100)
)
lines.append("")
lines.append("Sensitivity to non-payment during activation (evasion / uncaptured):")
lines.append("%-14s %18s %16s %8s" % ("non_payment", "activation_rev_paid", "total_loss EUR", "loss%"))
for rate in [0.0, 0.03, 0.05, 0.10]:
    paid = total_activation * (1 - rate)
    total_loss = underuse_total + total_activation * rate
    lines.append(
        "%-13.0f%% %18s %16s %7.1f%%"
        % (rate * 100, format(paid, ",.0f"), format(total_loss, ",.0f"), total_loss / total_sales * 100)
    )

report = "\n".join(lines)
with open("revenue_simulation.txt", "w", encoding="utf-8") as handle:
    handle.write(report)
print(report)
print("SIM_DONE")
spark.stop()
