# Databricks notebook source
"""Central configuration: environment-specific settings, table names and constants.

All variables and environment-specific settings live here so that notebooks and
functions never hardcode values. Values are passed as arguments where needed.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

# Local Windows dev: put %HADOOP_HOME%\bin on PATH so the JVM can load hadoop.dll
# (winutils.exe + native IO) for Spark file writes. Inert on Databricks/Linux.
if platform.system() == "Windows":
    _hadoop_bin = os.path.join(os.environ.get("HADOOP_HOME", ""), "bin")
    if os.path.isdir(_hadoop_bin) and _hadoop_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _hadoop_bin + os.pathsep + os.environ.get("PATH", "")
    # The hostname resolves to a Docker interface (host.docker.internal), so Python
    # workers time out connecting back to the driver. Pin Spark to loopback and make
    # workers use this interpreter. Inert on Databricks/Linux.
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

ENVIRONMENT_NON_PRODUCTION = "NPD"
ENVIRONMENT_PRODUCTION = "PRD"

# Seed for the weighted sampling of travel times so that allocations are deterministic.
SEED = 42

# Travel timestamps are reported in the local railway timezone (activation is stored in UTC).
LOCAL_TIMEZONE = "Europe/Brussels"


class Table:
    ARTICLES = "article_code_flex"
    SALES = "sales_flex"
    ACTIVATIONS_ARTICLE = "activations_article"
    ACTIVATIONS = "activations_flex"
    MINUTES_TO_ADD_PROBABILITIES = "minutes_to_add_probabilities"
    HOURS_BETWEEN_PROBABILITIES = "hours_between_there_and_back_probabilities"
    TRAVELS = "flex_travels"
    MODEL_PARAMETERS = "flex_model_parameters"
    UNIT_SCORES = "flex_unit_scores"


LOCAL_TABLE_FILES = {
    Table.ARTICLES: "article_code_flex.parquet",
    Table.SALES: "sales_flex_dummy.parquet",
    Table.ACTIVATIONS_ARTICLE: "activations_article_dummy.parquet",
    Table.ACTIVATIONS: "activations_flex_dummy.parquet",
    Table.MINUTES_TO_ADD_PROBABILITIES: "minutes_to_add_probabilities_dummy.parquet",
    Table.HOURS_BETWEEN_PROBABILITIES: "hours_between_there_and_back_probabilities_dummy.parquet",
}


# IntEnums map string categories to byte values for runtime and storage efficiency.
class ArticleCode(IntEnum):
    FLEX_6 = 256
    FLEX_10 = 257
    FLEX_80 = 258
    FLEX_120 = 259


class TravelDirection(IntEnum):
    THERE = 1
    BACK = 2


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_DIR = _REPOSITORY_ROOT.parent / "data"
_DEFAULT_OUTPUT_DIR = _REPOSITORY_ROOT / "output"


@dataclass(frozen=True)
class DataHandlerConfig:
    environment: str
    data_dir: Path
    output_dir: Path
    table_files: dict[str, str]


def get_data_handler_config(environment: str = ENVIRONMENT_NON_PRODUCTION) -> DataHandlerConfig:
    return DataHandlerConfig(
        environment=environment,
        data_dir=_DEFAULT_DATA_DIR,
        output_dir=_DEFAULT_OUTPUT_DIR,
        table_files=LOCAL_TABLE_FILES,
    )