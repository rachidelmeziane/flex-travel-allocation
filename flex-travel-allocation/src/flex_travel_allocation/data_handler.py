"""Central data access. All reads and writes are governed through the DataHandler.

Direct ``spark.read`` / ``df.write`` calls are not allowed elsewhere in the model.
Locally (NPD) tables are resolved to the provided dummy parquet files; the same
interface maps to Unity catalog tables in production (PRD).
"""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from conf.config import (
    ENVIRONMENT_NON_PRODUCTION,
    DataHandlerConfig,
    get_data_handler_config,
)


class DataHandler:
    def __init__(self, spark: SparkSession, config: DataHandlerConfig) -> None:
        self._spark = spark
        self._config = config

    @classmethod
    def from_config(
        cls,
        spark: SparkSession,
        environment: str = ENVIRONMENT_NON_PRODUCTION,
    ) -> "DataHandler":
        return cls(spark=spark, config=get_data_handler_config(environment=environment))

    def read_table(self, table_name: str) -> DataFrame:
        file_name = self._config.table_files.get(table_name)
        if file_name is None:
            raise KeyError(f"No physical location configured for table '{table_name}'.")
        return self._spark.read.parquet(str(self._config.data_dir / file_name))

    def write_table(self, dataframe: DataFrame, table_name: str, mode: str = "overwrite") -> None:
        output_path: Path = self._config.output_dir / table_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.write.mode(mode).parquet(str(output_path))

    def read_output_table(self, table_name: str) -> DataFrame:
        # Symmetric to write_table: read back a governed output table (e.g. model parameters).
        return self._spark.read.parquet(str(self._config.output_dir / table_name))
