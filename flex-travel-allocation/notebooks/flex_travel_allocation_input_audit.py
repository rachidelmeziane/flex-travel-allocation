# Databricks notebook source
# MAGIC %md
# MAGIC # Flex travel allocation - input data audit
# MAGIC Describes and audits every raw input table before allocation. For each table it
# MAGIC reports a **data sample**, **column data types**, **null / distinct counts** and a
# MAGIC **statistical description**. Tables are read as delivered (no cleaning) so the audit
# MAGIC reflects the real inputs. Read-only: it never writes and never mutates the data.

# COMMAND ----------
# Setup
import os
import sys

import pyspark.sql.functions as sf
from pyspark.sql import DataFrame, SparkSession

from conf.config import ENVIRONMENT_NON_PRODUCTION, Table
from flex_travel_allocation.data_handler import DataHandler

# Databricks provides Spark; locally (e.g. Jupyter on Windows) the Python worker needs
# PYSPARK_PYTHON and a stable loopback address to connect back to the driver.
if "DATABRICKS_RUNTIME_VERSION" in os.environ:
    spark = SparkSession.builder.getOrCreate()
else:
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    spark = (
        SparkSession.builder.appName("flex-input-audit")
        .master("local[*]")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

try:
    environment = dbutils.widgets.get("environment")  # type: ignore[name-defined]
except Exception:
    environment = ENVIRONMENT_NON_PRODUCTION

data_handler = DataHandler.from_config(spark=spark, environment=environment)

SAMPLE_ROWS = 5

INPUT_TABLES = [
    Table.ARTICLES,
    Table.SALES,
    Table.ACTIVATIONS_ARTICLE,
    Table.ACTIVATIONS,
    Table.MINUTES_TO_ADD_PROBABILITIES,
    Table.HOURS_BETWEEN_PROBABILITIES,
]

# COMMAND ----------
# Audit helpers - use display() on Databricks, fall back to .show() locally


def render(dataframe: DataFrame, n: int = SAMPLE_ROWS) -> None:
    # Collect the (already small) result to pandas so real values render as a table
    # in Jupyter/Databricks; plain Spark display() only shows the schema repr.
    pdf = dataframe.limit(n).toPandas()
    try:
        display(pdf)  # type: ignore[name-defined]
    except NameError:
        print(pdf.to_string(index=False))


def schema_overview(dataframe: DataFrame) -> DataFrame:
    """One row per column: name, Spark data type and nullable flag."""
    rows = [(f.name, f.dataType.simpleString(), f.nullable) for f in dataframe.schema.fields]
    return spark.createDataFrame(rows, ["column", "data_type", "nullable"])


def null_and_distinct_counts(dataframe: DataFrame) -> DataFrame:
    """Per-column non-null, null and distinct counts."""
    total = dataframe.count()
    exprs = []
    for column in dataframe.columns:
        exprs.append(sf.count(sf.col(column)).alias(f"{column}__nonnull"))
        exprs.append(sf.sum(sf.col(column).isNull().cast("long")).alias(f"{column}__null"))
        exprs.append(sf.countDistinct(sf.col(column)).alias(f"{column}__distinct"))
    agg = dataframe.agg(*exprs).collect()[0].asDict()
    rows = [
        (column, total, agg[f"{column}__nonnull"], agg[f"{column}__null"], agg[f"{column}__distinct"])
        for column in dataframe.columns
    ]
    return spark.createDataFrame(rows, ["column", "rows", "non_null", "nulls", "distinct"])


def descriptive_statistics(dataframe: DataFrame) -> DataFrame:
    """One row per numeric column: count, nulls, mean, stddev, min, quartiles, median, max, skewness, kurtosis."""
    numeric_types = {"tinyint", "smallint", "int", "bigint", "float", "double", "decimal"}
    numeric_columns = [
        field.name
        for field in dataframe.schema.fields
        if field.dataType.simpleString().split("(")[0] in numeric_types
    ]
    metrics = ["count", "nulls", "mean", "stddev", "min", "p25", "median", "p75", "max", "skewness", "kurtosis"]
    schema = (
        "column string, count long, nulls long, mean double, stddev double, "
        "min double, p25 double, median double, p75 double, max double, "
        "skewness double, kurtosis double"
    )
    if not numeric_columns:
        return spark.createDataFrame([], schema)
    exprs = []
    for column in numeric_columns:
        quoted = f"`{column}`"
        exprs += [
            sf.count(sf.col(column)).alias(f"{column}~count"),
            sf.sum(sf.col(column).isNull().cast("long")).alias(f"{column}~nulls"),
            sf.mean(sf.col(column)).alias(f"{column}~mean"),
            sf.stddev(sf.col(column)).alias(f"{column}~stddev"),
            sf.min(sf.col(column)).cast("double").alias(f"{column}~min"),
            sf.expr(f"percentile_approx({quoted}, 0.25)").cast("double").alias(f"{column}~p25"),
            sf.expr(f"percentile_approx({quoted}, 0.5)").cast("double").alias(f"{column}~median"),
            sf.expr(f"percentile_approx({quoted}, 0.75)").cast("double").alias(f"{column}~p75"),
            sf.max(sf.col(column)).cast("double").alias(f"{column}~max"),
            sf.skewness(sf.col(column)).alias(f"{column}~skewness"),
            sf.kurtosis(sf.col(column)).alias(f"{column}~kurtosis"),
        ]
    values = dataframe.agg(*exprs).collect()[0].asDict()
    rows = []
    for column in numeric_columns:
        rows.append(
            (
                column,
                int(values[f"{column}~count"]) if values[f"{column}~count"] is not None else None,
                int(values[f"{column}~nulls"]) if values[f"{column}~nulls"] is not None else None,
                *[
                    float(values[f"{column}~{metric}"]) if values[f"{column}~{metric}"] is not None else None
                    for metric in metrics[2:]
                ],
            )
        )
    return spark.createDataFrame(rows, schema)


def non_numeric_statistics(dataframe: DataFrame) -> DataFrame:
    """One row per non-numeric column (text/date/boolean): count, nulls, distinct, min, max, mode."""
    numeric_types = {"tinyint", "smallint", "int", "bigint", "float", "double", "decimal"}
    categorical = [
        field.name
        for field in dataframe.schema.fields
        if field.dataType.simpleString().split("(")[0] not in numeric_types
    ]
    schema = (
        "column string, data_type string, count long, nulls long, distinct long, "
        "min string, max string, top string, top_freq long"
    )
    if not categorical:
        return spark.createDataFrame([], schema)
    types = {field.name: field.dataType.simpleString() for field in dataframe.schema.fields}
    exprs = []
    for column in categorical:
        exprs += [
            sf.count(sf.col(column)).alias(f"{column}~count"),
            sf.sum(sf.col(column).isNull().cast("long")).alias(f"{column}~nulls"),
            sf.countDistinct(sf.col(column)).alias(f"{column}~distinct"),
            sf.min(sf.col(column)).cast("string").alias(f"{column}~min"),
            sf.max(sf.col(column)).cast("string").alias(f"{column}~max"),
        ]
    values = dataframe.agg(*exprs).collect()[0].asDict()
    rows = []
    for column in categorical:
        mode = (
            dataframe.filter(sf.col(column).isNotNull())
            .groupBy(sf.col(column).cast("string").alias("value"))
            .count()
            .orderBy(sf.desc("count"), "value")
            .first()
        )
        top, top_freq = (mode["value"], int(mode["count"])) if mode is not None else (None, None)
        rows.append(
            (
                column,
                types[column],
                int(values[f"{column}~count"]) if values[f"{column}~count"] is not None else None,
                int(values[f"{column}~nulls"]) if values[f"{column}~nulls"] is not None else None,
                int(values[f"{column}~distinct"]) if values[f"{column}~distinct"] is not None else None,
                values[f"{column}~min"],
                values[f"{column}~max"],
                top,
                top_freq,
            )
        )
    return spark.createDataFrame(rows, schema)


def audit_table(table_name: str) -> None:
    dataframe = data_handler.read_table(table_name)
    print(f"TABLE: {table_name}  |  rows: {dataframe.count():,}  |  columns: {len(dataframe.columns)}")
    print("--- column data types ---")
    render(schema_overview(dataframe), n=len(dataframe.columns))
    print("--- data sample ---")
    render(dataframe, n=SAMPLE_ROWS)
    print("--- null / distinct counts ---")
    render(null_and_distinct_counts(dataframe), n=len(dataframe.columns))
    print("--- descriptive statistics (numeric) ---")
    render(descriptive_statistics(dataframe), n=len(dataframe.columns))
    print("--- descriptive statistics (non-numeric) ---")
    render(non_numeric_statistics(dataframe), n=len(dataframe.columns))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Articles (`article_code_flex`)
# MAGIC Product reference: article code -> Flex product, max activations and validity window.

# COMMAND ----------
audit_table(Table.ARTICLES)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Sales (`sales_flex_dummy`)
# MAGIC One row per Flex subscription sold: ticket number, article, net sales and travel dates.

# COMMAND ----------
audit_table(Table.SALES)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Activations per article (`activations_article_dummy`)
# MAGIC Links each activation ticket number to its article code.

# COMMAND ----------
audit_table(Table.ACTIVATIONS_ARTICLE)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Activations (`activations_flex_dummy`)
# MAGIC Individual activation (usage) events - the largest input table.

# COMMAND ----------
audit_table(Table.ACTIVATIONS)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Minutes-to-add probabilities (`minutes_to_add_probabilities`)
# MAGIC Weighted distribution of the minute offset added when sampling travel times.

# COMMAND ----------
audit_table(Table.MINUTES_TO_ADD_PROBABILITIES)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Hours-between probabilities (`hours_between_there_and_back_probabilities`)
# MAGIC Weighted distribution of the hours between the there and back travels.

# COMMAND ----------
audit_table(Table.HOURS_BETWEEN_PROBABILITIES)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Audit all inputs in one pass
# MAGIC Convenience loop over every input table (same output as the cells above).

# COMMAND ----------
for _table in INPUT_TABLES:
    audit_table(_table)
    print()
