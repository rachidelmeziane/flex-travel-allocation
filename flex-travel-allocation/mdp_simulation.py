"""Per-product what-if MDP for the sale -> activation transition (run on the dummy data)."""

import os
import sys

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np
import pyspark.sql.functions as sf
from pyspark.sql import SparkSession

from conf.config import Table
from conf.params import PARAMS
from flex_travel_allocation.allocation import join_sales_to_activations
from flex_travel_allocation.data_handler import DataHandler
from flex_travel_allocation.ingestion import clean_activations, clean_articles, clean_sales
from flex_travel_allocation.mdp_what_if import estimate_hazards, evaluate_transition

UPLIFT = PARAMS["mdp_what_if"]["uplift"]
COST_FRACTION = PARAMS["mdp_what_if"]["cost_fraction"]

spark = (
    SparkSession.builder.master(PARAMS["spark"]["master"])
    .appName("mdp-what-if")
    .config("spark.sql.shuffle.partitions", str(PARAMS["spark"]["shuffle_partitions"]))
    .config("spark.ui.enabled", str(PARAMS["spark"]["ui_enabled"]).lower())
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

dh = DataHandler.from_config(spark=spark, environment=PARAMS["environment"])
articles = dh.read_table(Table.ARTICLES).transform(clean_articles)
sales = dh.read_table(Table.SALES).transform(clean_sales)
activations = dh.read_table(Table.ACTIVATIONS).transform(clean_activations)

used_per_ticket = (
    join_sales_to_activations(sales=sales, articles=articles, activations=activations)
    .groupBy("ticket_number")
    .agg(sf.count(sf.lit(1)).alias("used"))
)
ticket = (
    sales.join(sf.broadcast(articles), on="article_description_nl", how="inner")
    .join(used_per_ticket, on="ticket_number", how="left")
    .fillna(0, subset=["used"])
)

histogram = (
    ticket.groupBy("article_code_int", "article_description_nl", "maximum_activations_allowed", "used")
    .count()
    .toPandas()
)
prices = (
    ticket.groupBy("article_code_int", "article_description_nl", "maximum_activations_allowed")
    .agg(
        sf.count(sf.lit(1)).alias("subscriptions"),
        sf.sum("net_sales_euro").alias("sales_revenue"),
        sf.avg("net_sales_euro").alias("avg_price"),
    )
    .orderBy("article_code_int")
    .toPandas()
)

records = []
lines = []
lines.append("WHAT-IF MDP: sale -> activation transition, per product")
lines.append("=" * 92)
lines.append(f"(promotion uplift = +{UPLIFT:.0%} activation prob, incentive cost = {COST_FRACTION:.0%} of per-day price)")
lines.append("")
lines.append(
    "%-20s %4s %6s %7s %10s %13s %13s %11s %11s"
    % ("product", "N", "subs", "util%", "per_day_EUR", "sale_rev", "activ_rev(sq)",
       "recover(+)", "promo<util%")
)
lines.append("-" * 92)

for _, product in prices.iterrows():
    code = product["article_code_int"]
    n = int(product["maximum_activations_allowed"])
    subs = int(product["subscriptions"])
    sale_revenue = float(product["sales_revenue"])
    per_day_price = float(product["avg_price"]) / n

    counts = np.zeros(n + 1)
    product_hist = histogram[histogram["article_code_int"] == code]
    for _, hrow in product_hist.iterrows():
        counts[int(hrow["used"])] = hrow["count"]

    hazards = estimate_hazards(counts)
    status_quo = evaluate_transition(hazards, per_day_price)
    promo_free = evaluate_transition(hazards, per_day_price, uplift=UPLIFT, incentive_cost=0.0)
    promo_cost = evaluate_transition(
        hazards, per_day_price, uplift=UPLIFT, incentive_cost=COST_FRACTION * per_day_price
    )

    e_sq = status_quo["expected_activations_status_quo"]
    activation_revenue_sq = per_day_price * e_sq * subs
    recover_cost_net = (promo_cost["value_optimal"] - promo_cost["value_status_quo"]) * subs
    recover_free = (promo_free["expected_activations_optimal"] - e_sq) * per_day_price * subs
    promote_share = promo_cost["promote_below_state"] / n

    records.append(
        {
            "product": product["article_description_nl"],
            "max_activations": n,
            "subscriptions": subs,
            "utilisation": e_sq / n,
            "per_day_price": round(per_day_price, 2),
            "sale_revenue": round(sale_revenue, 0),
            "activation_revenue_status_quo": round(activation_revenue_sq, 0),
            "transition_gap": round(sale_revenue - activation_revenue_sq, 0),
            "recoverable_promo_costed": round(recover_cost_net, 0),
            "recoverable_promo_free": round(recover_free, 0),
            "promote_below_utilisation": round(promote_share, 2),
            "breakeven_per_day_price": round(sale_revenue / (e_sq * subs), 2),
        }
    )
    lines.append(
        "%-20s %4d %6d %6.1f%% %10.2f %13s %13s %11s %10.0f%%"
        % (
            product["article_description_nl"], n, subs, e_sq / n * 100, per_day_price,
            format(sale_revenue, ",.0f"), format(activation_revenue_sq, ",.0f"),
            format(recover_cost_net, ",.0f"), promote_share * 100,
        )
    )

import pandas as pd

table = pd.DataFrame(records)
table.to_csv("streamlit_app/artifacts/output_mdp_what_if.csv", index=False)

lines.append("-" * 92)
lines.append(
    "%-20s %4s %6d %6.1f%% %10s %13s %13s %11s"
    % (
        "TOTAL", "", int(table["subscriptions"].sum()),
        table["utilisation"].mul(table["subscriptions"]).sum() / table["subscriptions"].sum() * 100,
        "", format(table["sale_revenue"].sum(), ",.0f"),
        format(table["activation_revenue_status_quo"].sum(), ",.0f"),
        format(table["recoverable_promo_costed"].sum(), ",.0f"),
    )
)
lines.append("")
lines.append("transition_gap = sale_revenue - activation_revenue (revenue lost by charging at activation).")
lines.append("recover(+) = extra revenue from the MDP-optimal costed promotion policy (net of incentive cost).")
lines.append("promo<util% = optimal policy promotes while a subscription's cumulative use is below this share of N.")

report = "\n".join(lines)
with open("mdp_simulation.txt", "w", encoding="utf-8") as handle:
    handle.write(report)
print(report)
print("MDP_DONE")
spark.stop()
