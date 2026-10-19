# Materialise the governed model tables locally as viewable CSV artifacts.
# Production writes flex_model_parameters / flex_unit_scores via DataHandler (Databricks);
# here we score the dummy data with Spark and export CSVs for the Streamlit demo, because a
# local parquet write needs winutils/HADOOP_HOME.
import datetime as dt
import os
import sys
import warnings

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
sys.path.insert(0, "src")
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import pyspark.sql.functions as sf
from pyspark.sql import SparkSession

from conf.config import ENVIRONMENT_NON_PRODUCTION, Table
from flex_travel_allocation.data_handler import DataHandler
from flex_travel_allocation.ingestion import clean_activations, clean_articles, clean_sales
from flex_travel_allocation.models.data_prep import build_unit_table
from flex_travel_allocation.models.estimators import estimate_parameters
from flex_travel_allocation.models.parameters import fit_version
from flex_travel_allocation.models.scoring import score_units

ART = "streamlit_app/artifacts"
os.makedirs(ART, exist_ok=True)

spark = (
    SparkSession.builder.master("local[*]")
    .appName("score-units")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.ui.enabled", "false")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.sql.session.timeZone", "UTC")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

dh = DataHandler.from_config(spark=spark, environment=ENVIRONMENT_NON_PRODUCTION)
articles = dh.read_table(Table.ARTICLES).transform(clean_articles)
sales = dh.read_table(Table.SALES).transform(clean_sales)
activations = dh.read_table(Table.ACTIVATIONS).transform(clean_activations)

as_of = activations.select(sf.max("activation_date")).first()[0] + dt.timedelta(days=1)
units = build_unit_table(sales=sales, articles=articles, activations=activations, as_of=as_of)

# Fit the Stage 1/2 parameters (the flex_model_parameters table) and export them.
params_pdf = estimate_parameters(
    units.select(
        "article_description_nl", "event", "time", "lag_days", "n_activations", "exposure_days", "net_sales_euro"
    ).toPandas(),
    fit_version=fit_version(),
    as_of=as_of,
)
params_pdf.to_csv(f"{ART}/modeling_parameters.csv", index=False)

# Score every unit (the flex_unit_scores table) and export a sample + a per-product summary.
scored = score_units(units, spark.createDataFrame(params_pdf))
score_columns = [
    sf.col("ticket_number").cast("string").alias("ticket_number"),
    "article_description_nl", "event", "exposure_days", "nb_rate_per_day", "expected_trips",
    "weeks_since_sale", "p_inf", "p_activated_by_now", "expected_breakage_value",
]
sample = (
    scored.select(*score_columns)
    .orderBy("article_description_nl", "ticket_number")
    .limit(400)
    .toPandas()
)
sample.to_csv(f"{ART}/modeling_unit_scores.csv", index=False)

by_product = (
    scored.groupBy("article_description_nl")
    .agg(
        sf.count(sf.lit(1)).alias("units"),
        sf.round(sf.avg("expected_trips"), 3).alias("avg_expected_trips"),
        sf.round(sf.avg("p_activated_by_now"), 4).alias("avg_p_activated"),
        sf.round(sf.sum("expected_breakage_value"), 0).alias("total_breakage_value_eur"),
    )
    .orderBy("article_description_nl")
    .toPandas()
)
by_product.to_csv(f"{ART}/modeling_unit_scores_by_product.csv", index=False)

print("flex_unit_scores rows:", scored.count())
print(by_product.to_string(index=False))
spark.stop()
print("SCORE_UNITS_DONE ->", ART)
