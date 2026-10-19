# Driver-side estimators for Stage 1 (activation cure) and Stage 2 (travel intensity).
# Pure pandas/numpy/scipy/statsmodels: fit on the small collected unit frame, return one
# parameter row per product. Kept Spark-free so it is fast to unit-test.
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import weibull_min
import statsmodels.api as sm

from flex_travel_allocation.models.parameters import PARAMETER_COLUMNS


def saturating(t, p_inf, lam):
    return p_inf * (1.0 - np.exp(-lam * np.asarray(t, float)))


def km_activation_curve(time, event, grid_days):
    # Kaplan-Meier estimate of the activation CDF F(t) = 1 - S(t) under right censoring.
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    event_times = np.unique(time[event == 1])
    surv = 1.0
    xs, ys = [0.0], [0.0]
    for ti in event_times:
        d = np.sum((time == ti) & (event == 1))
        at_risk = np.sum(time >= ti)
        if at_risk > 0:
            surv *= 1.0 - d / at_risk
        xs.append(float(ti)); ys.append(1.0 - surv)
    xs = np.asarray(xs); ys = np.asarray(ys)
    return np.interp(np.asarray(grid_days, float), xs, ys, left=0.0, right=ys[-1])


def _weibull_fit(lags):
    x = np.asarray(lags, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    x = np.where(x <= 0, 0.5, x)
    k, _loc, eta = weibull_min.fit(x, floc=0.0)
    return float(k), float(eta)


def fit_stage1_segment(g):
    obs = g["time"].values
    max_weeks = max(4, int(np.ceil(np.nanmax(obs) / 7.0)))
    weeks = np.arange(1, max_weeks + 1)
    F = km_activation_curve(g["time"].values, g["event"].values, weeks * 7.0)
    ceil0 = min(0.999, max(float(F.max()), 0.5))
    try:
        (p_inf, lam), _ = curve_fit(saturating, weeks, F, p0=[ceil0, 0.2], bounds=([0.0, 1e-4], [1.0, 5.0]), maxfev=20000)
    except Exception:
        p_inf, lam = float(F.max()), np.nan
    k, eta = _weibull_fit(g.loc[g["event"] == 1, "lag_days"].values)
    weib_median = eta * np.log(2.0) ** (1.0 / k) if (np.isfinite(eta) and np.isfinite(k) and k > 0) else np.nan
    return dict(
        p_inf=float(p_inf), breakage=1.0 - float(p_inf),
        lambda_week=float(lam) if np.isfinite(lam) else np.nan,
        weibull_k=float(k) if np.isfinite(k) else np.nan,
        weibull_eta_days=float(eta) if np.isfinite(eta) else np.nan,
        weibull_median_lag_days=float(weib_median) if np.isfinite(weib_median) else np.nan,
    )


def fit_stage2(unit_pdf):
    # Negative-binomial trips per active-day (log-exposure offset); returns per-product rates.
    act = unit_pdf[unit_pdf["event"] == 1].copy()
    offset = np.log(np.clip(act["exposure_days"].values.astype(float), 1.0, None))
    prod = pd.get_dummies(act["article_description_nl"].astype(str), drop_first=True).astype(float)
    xdf = prod.copy(); xdf.insert(0, "intercept", 1.0)
    cols = list(xdf.columns); X = xdf.values
    y = act["n_activations"].values.astype(float)
    poisson = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset).fit()
    overdispersion = float(poisson.pearson_chi2 / poisson.df_resid)
    try:
        nb = sm.NegativeBinomial(y, X, offset=offset).fit(disp=0, maxiter=200)
        params = np.asarray(nb.params[: X.shape[1]], float); alpha = float(nb.params[-1])
    except Exception:
        mu = poisson.predict()
        alpha = float(max(1e-6, np.mean(((y - mu) ** 2 - mu) / np.clip(mu ** 2, 1e-9, None))))
        nb = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha), offset=offset).fit()
        params = np.asarray(nb.params, float)
    rates = {}
    for pr in act["article_description_nl"].unique():
        row = {c: 0.0 for c in cols}; row["intercept"] = 1.0
        for c in cols:
            if c != "intercept" and c.endswith(pr):
                row[c] = 1.0
        rates[pr] = float(np.exp(float(sum(params[i] * row[c] for i, c in enumerate(cols)))))
    return rates, float(alpha), overdispersion


def estimate_parameters(unit_pdf, fit_version, as_of):
    # One parameter row per product; combines Stage 1 (cure/Weibull) and Stage 2 (NB) fits.
    rates, alpha, overdispersion = fit_stage2(unit_pdf)
    fit_ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for product, g in unit_pdf.groupby("article_description_nl"):
        s1 = fit_stage1_segment(g)
        row = dict(
            article_description_nl=str(product),
            nb_rate_per_day=float(rates.get(str(product), np.nan)),
            nb_alpha=alpha,
            poisson_overdispersion=overdispersion,
            activated_rate=float(g["event"].mean()),
            n=int(len(g)),
            fit_version=str(fit_version),
            fit_ts=fit_ts,
            as_of=str(as_of),
        )
        row.update(s1)
        rows.append(row)
    df = pd.DataFrame(rows)
    ordered = [c for c in PARAMETER_COLUMNS if c in df.columns]
    return df[ordered + [c for c in df.columns if c not in ordered]]
