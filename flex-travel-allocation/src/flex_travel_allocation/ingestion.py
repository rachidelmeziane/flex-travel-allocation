"""Clean raw input tables: select relevant columns, apply naming and cast types.

Kept separate from transformation logic (ingestion is separated from transformation).
"""

from __future__ import annotations

import pyspark.sql.functions as sf
from pyspark.sql import DataFrame
from pyspark.sql import types as st


def clean_articles(articles: DataFrame) -> DataFrame:
    return articles.select(
        sf.col("article_code_int").cast(st.ShortType()).alias("article_code_int"),
        sf.col("article_description_nl").cast(st.StringType()).alias("article_description_nl"),
        sf.col("maximum_activations_allowed")
        .cast(st.IntegerType())
        .alias("maximum_activations_allowed"),
    )


def clean_sales(sales: DataFrame) -> DataFrame:
    return sales.select(
        sf.col("ticket_number").cast(st.LongType()).alias("ticket_number"),
        sf.col("article_description_nl").cast(st.StringType()).alias("article_description_nl"),
        sf.col("net_sales_euro").cast(st.DoubleType()).alias("net_sales_euro"),
        sf.to_date("sales_date").alias("sales_date"),
        sf.to_date("travel_start_date").alias("validity_start_date"),
        sf.to_date("travel_end_date").alias("validity_end_date"),
    )


def clean_activations_article(activations_article: DataFrame) -> DataFrame:
    return activations_article.select(
        sf.col("ticket_number_activations")
        .cast(st.StringType())
        .alias("ticket_number_activations"),
        sf.col("article_code_int").cast(st.ShortType()).alias("article_code_int"),
    )


def clean_activations(activations: DataFrame) -> DataFrame:
    return activations.select(
        sf.col("ticket_number_activations")
        .cast(st.StringType())
        .alias("ticket_number_activations"),
        sf.to_date("activation_date").alias("activation_date"),
        sf.col("activation_sequence").cast(st.IntegerType()).alias("activation_sequence"),
        sf.col("activation_timestamp_utc")
        .cast(st.TimestampType())
        .alias("activation_timestamp_utc"),
    )


def clean_minutes_to_add_probabilities(minutes_probabilities: DataFrame) -> DataFrame:
    return minutes_probabilities.select(
        sf.col("minutes_to_add").cast(st.IntegerType()).alias("minutes_to_add"),
        sf.col("probability").cast(st.DoubleType()).alias("probability"),
    )


def clean_hours_between_probabilities(hours_probabilities: DataFrame) -> DataFrame:
    return hours_probabilities.select(
        sf.col("hours_between_there_and_back")
        .cast(st.IntegerType())
        .alias("hours_between_there_and_back"),
        sf.col("probability").cast(st.DoubleType()).alias("probability"),
    )
