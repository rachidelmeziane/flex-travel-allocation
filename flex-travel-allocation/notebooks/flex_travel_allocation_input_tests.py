# Databricks notebook source
# MAGIC %md
# MAGIC # Flex travel allocation - input tests
# MAGIC Validate input data quality before allocation. Issues are logged and collected;
# MAGIC the notebook raises if any error stops the pipeline.
# MAGIC 
# COMMAND ----------
# Setup cell
import logging

import pyspark.sql.functions as sf
from pyspark.sql import SparkSession

from conf.config import ENVIRONMENT_NON_PRODUCTION, ArticleCode, Table
from flex_travel_allocation.data_handler import DataHandler
from flex_travel_allocation.ingestion import (
    clean_activations,
    clean_articles,
    clean_sales,
)
from flex_travel_allocation.ticket_matching import add_activation_ticket_key

logger = logging.getLogger("flex_travel_allocation.input_tests")
spark = SparkSession.builder.getOrCreate()

try:
    environment = dbutils.widgets.get("environment")  # type: ignore[name-defined]
except Exception:
    environment = ENVIRONMENT_NON_PRODUCTION

data_handler = DataHandler.from_config(spark=spark, environment=environment)
errors_stopping_pipeline: list[str] = []

# COMMAND ----------
# Read inputs
articles = data_handler.read_table(Table.ARTICLES).transform(clean_articles)
sales = data_handler.read_table(Table.SALES).transform(clean_sales)
activations = data_handler.read_table(Table.ACTIVATIONS).transform(clean_activations)

# COMMAND ----------
# Not empty
for table_name, dataframe in {
    Table.ARTICLES: articles,
    Table.SALES: sales,
    Table.ACTIVATIONS: activations,
}.items():
    if dataframe.limit(1).count() == 0:
        message = f"Input table '{table_name}' is empty."
        logger.error(message)
        errors_stopping_pipeline.append(message)

# COMMAND ----------
# No duplicate sales on the ticket number (the sales join key)
duplicate_sales = sales.count() - sales.select("ticket_number").distinct().count()
if duplicate_sales > 0:
    message = f"Found {duplicate_sales} duplicate ticket_number values in sales."
    logger.error(message)
    errors_stopping_pipeline.append(message)

# COMMAND ----------
# No new article codes without allocation logic
known_article_codes = {code.value for code in ArticleCode}
unknown_article_codes = (
    articles.filter(~sf.col("article_code_int").isin(list(known_article_codes)))
    .select("article_code_int")
    .distinct()
    .count()
)
if unknown_article_codes > 0:
    message = f"Found {unknown_article_codes} article codes without allocation logic."
    logger.error(message)
    errors_stopping_pipeline.append(message)

# COMMAND ----------
# Every sale matches an activation ticket (completeness of the ticket-number reconciliation)
matched_sales = (
    sales.transform(add_activation_ticket_key)
    .join(
        activations.select("ticket_number_activations").distinct(),
        on="ticket_number_activations",
        how="left_semi",
    )
    .count()
)
unmatched_sales = sales.count() - matched_sales
if unmatched_sales > 0:
    message = f"{unmatched_sales} sales could not be matched to an activation ticket."
    logger.error(message)
    errors_stopping_pipeline.append(message)

# COMMAND ----------
# Raise if any check failed
if errors_stopping_pipeline:
    raise ValueError("Input validation failed:\n" + "\n".join(errors_stopping_pipeline))
