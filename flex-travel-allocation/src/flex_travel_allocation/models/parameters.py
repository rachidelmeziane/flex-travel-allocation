# Schema + helpers for the MODEL_PARAMETERS governed table.
from __future__ import annotations

import datetime as dt

# Columns the estimation job writes and the scoring step relies on.
PARAMETER_COLUMNS = [
    "article_description_nl",
    "p_inf",
    "breakage",
    "lambda_week",
    "weibull_k",
    "weibull_eta_days",
    "weibull_median_lag_days",
    "nb_rate_per_day",
    "nb_alpha",
    "poisson_overdispersion",
    "activated_rate",
    "n",
    "fit_version",
    "fit_ts",
    "as_of",
]

# Minimal set the Spark scoring step joins on.
SCORING_COLUMNS = ["article_description_nl", "p_inf", "lambda_week", "nb_rate_per_day"]


def fit_version() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
