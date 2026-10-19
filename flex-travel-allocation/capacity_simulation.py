"""Train-filling analysis: sales-planned vs allocation (actual) demand by hour."""

import os
import sys

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
sys.path.insert(0, "src")
sys.path.insert(0, ".")

import pyspark.sql.functions as sf
from pyspark.sql import SparkSession

from conf.config import SEED, Table
from conf.params import PARAMS
from flex_travel_allocation.allocation import allocate_sales_to_travels
from flex_travel_allocation.capacity_analysis import train_filling_by_hour
from flex_travel_allocation.data_handler import DataHandler
from flex_travel_allocation.ingestion import (
    clean_activations,
    clean_articles,
    clean_hours_between_probabilities,
    clean_minutes_to_add_probabilities,
    clean_sales,
)
from flex_travel_allocation.revenue_risk import revenue_risk_by_product

spark = (
    SparkSession.builder.master(PARAMS["spark"]["master"])
    .appName("capacity-analysis")
    .config("spark.sql.shuffle.partitions", str(PARAMS["spark"]["shuffle_partitions"]))
    .config("spark.ui.enabled", str(PARAMS["spark"]["ui_enabled"]).lower())
    .config("spark.sql.session.timeZone", PARAMS["spark"]["session_timezone"])
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

dh = DataHandler.from_config(spark=spark, environment=PARAMS["environment"])
articles = dh.read_table(Table.ARTICLES).transform(clean_articles)
sales = dh.read_table(Table.SALES).transform(clean_sales)
activations = dh.read_table(Table.ACTIVATIONS).transform(clean_activations)
minutes = dh.read_table(Table.MINUTES_TO_ADD_PROBABILITIES).transform(clean_minutes_to_add_probabilities)
hours = dh.read_table(Table.HOURS_BETWEEN_PROBABILITIES).transform(clean_hours_between_probabilities)

travels = allocate_sales_to_travels(
    sales=sales, articles=articles, activations=activations,
    minutes_distribution=minutes, hours_distribution=hours, seed=SEED,
)
product_utilisation = revenue_risk_by_product(
    sales=sales, articles=articles, activations=activations, non_payment_rate=PARAMS["revenue_risk"]["non_payment_rate"]
).select("article_description_nl", "utilisation")

by_hour = train_filling_by_hour(travels, product_utilisation).toPandas()
by_hour.to_csv("streamlit_app/artifacts/output_capacity_by_hour.csv", index=False)

total_allocation = by_hour["allocation_pax"].sum()
total_sales = by_hour["sales_potential_pax"].sum()
capacity = by_hour["sales_potential_pax"].max()          # fleet sized to the sales peak hour
alloc_peak = by_hour["allocation_pax"].max()
peak_hour_alloc = int(by_hour.loc[by_hour["allocation_pax"].idxmax(), "travel_hour"])
peak_hour_sales = int(by_hour.loc[by_hour["sales_potential_pax"].idxmax(), "travel_hour"])
active = by_hour[by_hour["allocation_pax"] > 0]
avg_lf_alloc = (active["allocation_pax"] / capacity).mean()

lines = []
lines.append("TRAIN FILLING: sales-planned capacity vs allocation (actual) demand")
lines.append("=" * 74)
lines.append("")
lines.append("%4s %16s %18s %14s" % ("hour", "allocation_pax", "sales_potential_pax", "load_factor%"))
for row in by_hour.itertuples():
    lines.append(
        "%4d %16s %18s %13.0f%%"
        % (row.travel_hour, format(int(row.allocation_pax), ","),
           format(int(row.sales_potential_pax), ","), row.allocation_pax / capacity * 100)
    )
lines.append("-" * 74)
lines.append("")
lines.append("Total travels  : allocation %s vs sales-potential %s pax."
             % (format(int(total_allocation), ","), format(int(total_sales), ",")))
lines.append("Overall fill   : %.1f%% (allocation / sales-potential)."
             % (total_allocation / total_sales * 100))
lines.append("Peak hour      : allocation %s pax at %02dh; sales-plan %s pax at %02dh."
             % (format(int(alloc_peak), ","), peak_hour_alloc, format(int(capacity), ","), peak_hour_sales))
lines.append("Extra capacity if planning to sales (peak): +%.0f%% seats."
             % ((capacity / alloc_peak - 1) * 100))
lines.append("Avg load factor of active hours vs sales-planned capacity: %.1f%%."
             % (avg_lf_alloc * 100))
lines.append("Peak load factor (actual vs sales-planned capacity): %.1f%%."
             % (alloc_peak / capacity * 100))

report = "\n".join(lines)
with open("capacity_simulation.txt", "w", encoding="utf-8") as handle:
    handle.write(report)
print(report)
print("CAPACITY_DONE")
spark.stop()
