"""One-off: run the model on the dummy data and save small artifacts for the Streamlit demo."""

import json
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
from flex_travel_allocation.allocation import (
    allocate_sales_to_travels,
    join_sales_to_activations,
)
from flex_travel_allocation.analysis import (
    travels_by_date,
    travels_by_hour_and_direction,
    travels_by_product,
)
from flex_travel_allocation.data_handler import DataHandler
from flex_travel_allocation.ingestion import (
    clean_activations,
    clean_articles,
    clean_hours_between_probabilities,
    clean_minutes_to_add_probabilities,
    clean_sales,
)

ART = "streamlit_app/artifacts"
os.makedirs(ART, exist_ok=True)

spark = (
    SparkSession.builder.master(PARAMS["spark"]["master"])
    .appName("flex-precompute")
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

activated = join_sales_to_activations(sales=sales, articles=articles, activations=activations).cache()
travels = allocate_sales_to_travels(
    sales=sales, articles=articles, activations=activations,
    minutes_distribution=minutes, hours_distribution=hours, seed=SEED,
).cache()

# Example: a Flex 6 subscription with all 6 activations (compact input -> output story).
candidate = (
    activated.filter(sf.col("article_code_int") == 256)
    .groupBy("ticket_number").count()
    .filter(sf.col("count") == 6)
    .orderBy("ticket_number").limit(1).collect()
)
example = int(candidate[0]["ticket_number"]) if candidate else int(activated.select("ticket_number").first()["ticket_number"])
example_key = "2025" + str(example).zfill(19)

ex_sale = sales.filter(sf.col("ticket_number") == example)
example_product = ex_sale.select("article_description_nl").first()["article_description_nl"]

# --- input samples ---
articles.orderBy("article_code_int").toPandas().to_csv(f"{ART}/input_articles.csv", index=False)
minutes.orderBy("minutes_to_add").toPandas().to_csv(f"{ART}/input_minutes_probabilities.csv", index=False)
hours.orderBy("hours_between_there_and_back").toPandas().to_csv(f"{ART}/input_hours_probabilities.csv", index=False)

others = sales.filter(sf.col("ticket_number") != example).limit(9)
ex_sale.unionByName(others).toPandas().to_csv(f"{ART}/input_sales_sample.csv", index=False)

(
    activations.filter(sf.col("ticket_number_activations") == example_key)
    .orderBy("activation_sequence")
    .toPandas()
    .to_csv(f"{ART}/input_activations_sample.csv", index=False)
)

# --- output samples and aggregations ---
(
    travels.filter(sf.col("ticket_number") == example)
    .orderBy("travel_date", "travel_direction")
    .toPandas()
    .to_csv(f"{ART}/output_travels_sample.csv", index=False)
)

travels_by_product(travels).toPandas().to_csv(f"{ART}/output_by_product.csv", index=False)
travels_by_date(travels).toPandas().to_csv(f"{ART}/output_by_date.csv", index=False)

hour_direction = travels_by_hour_and_direction(travels).toPandas()
pivot = (
    hour_direction.pivot(index="travel_hour", columns="travel_direction", values="pax")
    .fillna(0).astype(int).rename(columns={1: "there", 2: "back"}).reset_index()
)
pivot.to_csv(f"{ART}/output_by_hour_direction.csv", index=False)

from flex_travel_allocation.expected_distribution import expected_pax_by_hour

expected = expected_pax_by_hour(
    activations=activated, minutes_distribution=minutes, hours_distribution=hours
).toPandas()
expected_pivot = (
    expected.pivot(index="travel_hour", columns="travel_direction", values="expected_pax")
    .fillna(0).round().astype(int).rename(columns={1: "there", 2: "back"}).reset_index()
)
expected_pivot.to_csv(f"{ART}/output_expected_by_hour_direction.csv", index=False)

from flex_travel_allocation.monte_carlo import monte_carlo_pax_by_hour

mc_seeds = [index * 1000 + 1 for index in range(PARAMS["monte_carlo"]["replicates"])]
monte_carlo = monte_carlo_pax_by_hour(
    activations=activated, minutes_distribution=minutes, hours_distribution=hours, seeds=mc_seeds
).toPandas()
monte_carlo.sort_values(["travel_direction", "travel_hour"]).to_csv(
    f"{ART}/output_monte_carlo_by_hour_direction.csv", index=False
)

from flex_travel_allocation.scenario_tables import capacity_by_hour_table, mdp_what_if_table, revenue_risk_table
from flex_travel_allocation.tradeoff_analysis import product_tradeoff

revenue_risk_table(sales=sales, articles=articles, activations=activations).to_csv(
    f"{ART}/output_revenue_risk.csv", index=False
)
mdp_table = mdp_what_if_table(sales=sales, articles=articles, activations=activations)
mdp_table.to_csv(
    f"{ART}/output_mdp_what_if.csv", index=False
)

product_tradeoff(mdp_table).to_csv(f"{ART}/output_tradeoff.csv", index=False)
capacity_by_hour_table(travels, sales, articles, activations).to_csv(
    f"{ART}/output_capacity_by_hour.csv", index=False
)

summary = {
    "input_sales": int(sales.count()),
    "input_activations": int(activations.count()),
    "output_travel_rows": int(travels.count()),
    "total_pax": int(travels.groupBy().sum("pax").first()[0]),
    "distinct_tickets": int(travels.select("ticket_number").distinct().count()),
    "example_ticket": example,
    "example_product": example_product,
    "seed": SEED,
    "monte_carlo_replicates": len(mc_seeds),
}
with open(f"{ART}/summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("PRECOMPUTE_DONE", summary)
spark.stop()
