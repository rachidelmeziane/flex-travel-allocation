# Databricks notebook source
# MAGIC %md
# MAGIC # Flex travel allocation - pipeline
# MAGIC Production pipeline: read inputs, allocate sales to travels, write the travels table.
# MAGIC Non-interactive and deterministic; contains orchestration only (no business logic).

# COMMAND ----------
# Setup cell
from pyspark.sql import SparkSession

from conf.config import SEED, ENVIRONMENT_NON_PRODUCTION, Table
from flex_travel_allocation.allocation import allocate_sales_to_travels
from flex_travel_allocation.data_handler import DataHandler
from flex_travel_allocation.ingestion import (
    clean_activations,
    clean_articles,
    clean_hours_between_probabilities,
    clean_minutes_to_add_probabilities,
    clean_sales,
)

spark = SparkSession.builder.getOrCreate()

try:
    environment = dbutils.widgets.get("environment")  # type: ignore[name-defined]
except Exception:
    environment = ENVIRONMENT_NON_PRODUCTION

data_handler = DataHandler.from_config(spark=spark, environment=environment)

# COMMAND ----------
# Read and clean inputs
articles = data_handler.read_table(Table.ARTICLES).transform(clean_articles)
sales = data_handler.read_table(Table.SALES).transform(clean_sales)
activations = data_handler.read_table(Table.ACTIVATIONS).transform(clean_activations)
minutes_distribution = data_handler.read_table(
    Table.MINUTES_TO_ADD_PROBABILITIES
).transform(clean_minutes_to_add_probabilities)
hours_distribution = data_handler.read_table(
    Table.HOURS_BETWEEN_PROBABILITIES
).transform(clean_hours_between_probabilities)

# COMMAND ----------
# Allocate sales to travels
travels = allocate_sales_to_travels(
    sales=sales,
    articles=articles,
    activations=activations,
    minutes_distribution=minutes_distribution,
    hours_distribution=hours_distribution,
    seed=SEED,
)

# COMMAND ----------
# Write results
data_handler.write_table(dataframe=travels, table_name=Table.TRAVELS)

# COMMAND ----------
# Score units from the fitted model parameters (Stage 1 + Stage 2); skips if not yet fitted.
import datetime as dt

from flex_travel_allocation.models.data_prep import build_unit_table
from flex_travel_allocation.models.scoring import score_units

try:
    parameters = data_handler.read_output_table(Table.MODEL_PARAMETERS)
    units = build_unit_table(sales=sales, articles=articles, activations=activations, as_of=dt.date.today())
    scored = score_units(units, parameters)
    data_handler.write_table(dataframe=scored, table_name=Table.UNIT_SCORES)
except Exception as error:  # noqa: BLE001 - scoring is optional until the model is first fitted
    print("Skipping scoring - MODEL_PARAMETERS unavailable:", error)
