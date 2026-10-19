# Build the unit-level (per subscription sold) modeling table from the raw inputs.
from __future__ import annotations

import numpy as np
import pandas as pd

from modeling.config import ModelingConfig


def build_unit_table(config: ModelingConfig) -> pd.DataFrame:
    sales = pd.read_parquet(config.data_dir / config.sales_file)
    acts = pd.read_parquet(config.data_dir / config.activations_file)

    sales["sale_date"] = pd.to_datetime(sales["sales_date"]).dt.normalize()
    sales["validity_start"] = pd.to_datetime(sales["travel_start_date"]).dt.normalize()
    sales["validity_end"] = pd.to_datetime(sales["travel_end_date"]).dt.normalize()
    # Reconcile the sales ticket to the 23-char activation key: year + lpad(ticket, 19).
    sales["ticket_key"] = (
        sales["sale_date"].dt.year.astype(str)
        + sales["ticket_number"].astype("int64").astype(str).str.zfill(19)
    )

    acts["activation_date"] = pd.to_datetime(acts["activation_date"]).dt.normalize()
    agg = (
        acts.groupby("ticket_number_activations")
        .agg(
            first_activation_date=("activation_date", "min"),
            last_activation_date=("activation_date", "max"),
            n_activations=("activation_sequence", "count"),
        )
        .reset_index()
        .rename(columns={"ticket_number_activations": "ticket_key"})
    )

    unit = sales.merge(agg, on="ticket_key", how="left")
    unit["activated"] = unit["first_activation_date"].notna()
    unit["n_activations"] = unit["n_activations"].fillna(0).astype(int)

    max_obs = acts["activation_date"].max()
    as_of = pd.Timestamp(config.as_of_date) if config.as_of_date is not None else (max_obs + pd.Timedelta(days=1))
    unit["as_of"] = as_of

    unit["lag_days"] = (unit["first_activation_date"] - unit["sale_date"]).dt.days
    unit["censor_days"] = (as_of - unit["sale_date"]).dt.days.clip(lower=0)
    unit["event"] = unit["activated"].astype(int)
    unit["time"] = np.where(unit["activated"], unit["lag_days"], unit["censor_days"]).astype(float)
    unit["time"] = unit["time"].clip(lower=0.0)

    # Stage-2 exposure: usable days after activation, capped at the validity end / as_of.
    horizon = np.minimum(unit["validity_end"].values, np.datetime64(as_of))
    exposure = (pd.to_datetime(horizon) - unit["first_activation_date"]).dt.days
    unit["exposure_days"] = exposure.fillna(0).clip(lower=1).astype(float)

    unit["travelled"] = (unit["n_activations"] > 0).astype(int)
    unit["product"] = unit["article_description_nl"].astype(str)
    unit["sale_month"] = unit["sale_date"].dt.month.astype(int)
    unit["price_tier"] = (
        pd.qcut(unit["net_sales_euro"], 4, labels=["q1", "q2", "q3", "q4"], duplicates="drop").astype(str)
    )
    return unit
