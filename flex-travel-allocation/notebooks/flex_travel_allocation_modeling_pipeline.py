# Databricks notebook source
# MAGIC %md
# MAGIC # Flex travel allocation - model estimation pipeline
# MAGIC Weekly job: fit Stage 1 (activation cure) and Stage 2 (travel intensity) models and write
# MAGIC the small MODEL_PARAMETERS table. The daily allocation pipeline scores from it, so scipy /
# MAGIC statsmodels stay off the hot path. The numerical fit runs on the driver over the small unit
# MAGIC frame (one row per subscription sold).

# COMMAND ----------
# Setup
import datetime as dt

import pyspark.sql.functions as sf
from pyspark.sql import SparkSession

from conf.config import ENVIRONMENT_NON_PRODUCTION, Table
from conf.params import PARAMS
from flex_travel_allocation.data_handler import DataHandler
from flex_travel_allocation.ingestion import clean_activations, clean_articles, clean_sales
from flex_travel_allocation.models.data_prep import build_unit_table
from flex_travel_allocation.models.estimators import estimate_parameters
from flex_travel_allocation.models.parameters import fit_version

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

# COMMAND ----------
# Build the unit table (Spark), then collect the small frame to the driver for fitting
as_of_setting = PARAMS.get("modeling", {}).get("as_of")
if as_of_setting:
    as_of = dt.date.fromisoformat(str(as_of_setting))
else:
    as_of = activations.select(sf.max("activation_date")).first()[0] + dt.timedelta(days=1)

units = build_unit_table(sales=sales, articles=articles, activations=activations, as_of=as_of)
unit_pdf = units.select(
    "article_description_nl", "event", "time", "lag_days", "n_activations", "exposure_days", "net_sales_euro"
).toPandas()

# COMMAND ----------
# Fit Stage 1 + Stage 2 and write the governed parameters table
parameters_pdf = estimate_parameters(unit_pdf, fit_version=fit_version(), as_of=as_of)
parameters = spark.createDataFrame(parameters_pdf)
data_handler.write_table(dataframe=parameters, table_name=Table.MODEL_PARAMETERS)
print("MODEL_PARAMETERS rows:", parameters.count(), "| version:", parameters_pdf["fit_version"].iloc[0])
