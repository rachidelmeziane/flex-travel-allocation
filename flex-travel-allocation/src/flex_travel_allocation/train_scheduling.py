"""Train scheduling optimisation: trains a sales plan vs an activation plan must run.

Per-hour trains = pax / (seats per train x target load factor), rounded up. Planning to
actual activation demand instead of sales removes the trains provisioned for days that are
paid for but never travelled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def trains_by_hour(
    capacity: pd.DataFrame,
    seats_per_train: int,
    target_load_factor: float = 1.0,
) -> pd.DataFrame:
    effective_seats = seats_per_train * target_load_factor
    trains = capacity[["travel_hour", "allocation_pax", "sales_potential_pax"]].copy()
    trains["trains_sales"] = np.ceil(trains["sales_potential_pax"] / effective_seats).astype(int)
    trains["trains_activation"] = np.ceil(trains["allocation_pax"] / effective_seats).astype(int)
    trains["trains_saved"] = trains["trains_sales"] - trains["trains_activation"]
    return trains


def train_optimisation_summary(trains: pd.DataFrame) -> dict:
    total_sales = int(trains["trains_sales"].sum())
    total_activation = int(trains["trains_activation"].sum())
    return {
        "trains_sales": total_sales,
        "trains_activation": total_activation,
        "trains_saved": total_sales - total_activation,
        "reduction_pct": (total_sales - total_activation) / total_sales if total_sales else 0.0,
        "peak_trains_sales": int(trains["trains_sales"].max()),
        "peak_trains_activation": int(trains["trains_activation"].max()),
    }
