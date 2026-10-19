# Stage 1: sale -> activation. Mixture cure model (logistic incidence + Weibull latency)
# plus a pragmatic saturating-curve substitute fitted on censoring-corrected cohort curves.
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy.special import expit
from scipy.stats import weibull_min


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
    x = np.where(x <= 0, 0.5, x)  # activation on the sale day -> half a day
    k, _loc, eta = weibull_min.fit(x, floc=0.0)
    return float(k), float(eta)


def fit_saturating_by_segment(unit, segment_col="product"):
    rows = []
    for seg, g in unit.groupby(segment_col):
        obs = g["time"].values
        max_weeks = max(4, int(np.ceil(np.nanmax(obs) / 7.0)))
        weeks = np.arange(1, max_weeks + 1)
        F = km_activation_curve(g["time"].values, g["event"].values, weeks * 7.0)
        ceil0 = min(0.999, max(float(F.max()), 0.5))
        try:
            (p_inf, lam), _ = curve_fit(
                saturating, weeks, F, p0=[ceil0, 0.2],
                bounds=([0.0, 1e-4], [1.0, 5.0]), maxfev=20000,
            )
        except Exception:
            p_inf, lam = float(F.max()), np.nan
        lags = g.loc[g["event"] == 1, "lag_days"].values
        k, eta = _weibull_fit(lags)
        weib_median = eta * np.log(2.0) ** (1.0 / k) if (np.isfinite(eta) and np.isfinite(k) and k > 0) else np.nan
        sat_median = np.log(2.0) / lam if np.isfinite(lam) and lam > 0 else np.nan
        rows.append(dict(
            segment=seg, n=int(len(g)), activated_rate=round(float(g["event"].mean()), 4),
            p_inf=round(float(p_inf), 4), breakage=round(1.0 - float(p_inf), 4),
            lam_per_week=round(float(lam), 4) if np.isfinite(lam) else np.nan,
            sat_median_lag_weeks=round(float(sat_median), 2) if np.isfinite(sat_median) else np.nan,
            weibull_k=round(float(k), 3) if np.isfinite(k) else np.nan,
            weibull_eta_days=round(float(eta), 2) if np.isfinite(eta) else np.nan,
            weibull_median_lag_days=round(float(weib_median), 2) if np.isfinite(weib_median) else np.nan,
        ))
    return pd.DataFrame(rows)


def cohort_comparison(unit, cohort_col="sale_month"):
    # Naive activation rate vs KM (censoring-corrected) activation at each cohort's own horizon.
    rows = []
    for cohort, g in unit.groupby(cohort_col):
        horizon = float(np.nanmax(g["time"].values))
        km = km_activation_curve(g["time"].values, g["event"].values, [horizon])[0]
        rows.append(dict(cohort=int(cohort), n=int(len(g)),
                         naive_rate=round(float(g["event"].mean()), 4),
                         corrected_rate=round(float(km), 4),
                         horizon_days=round(horizon, 1)))
    return pd.DataFrame(rows).sort_values("cohort")


def _design(unit, cols=("product",)):
    parts = [pd.Series(1.0, index=unit.index, name="intercept")]
    for c in cols:
        dummies = pd.get_dummies(unit[c].astype(str), prefix=c, drop_first=True).astype(float)
        parts.append(dummies)
    X = pd.concat(parts, axis=1)
    return X.values, list(X.columns)


def _cure_negll(params, X, t, e, eps=1e-9):
    p = X.shape[1]
    beta = params[:p]
    k = np.exp(params[p])
    gamma = params[p + 1:]
    eta = np.exp(np.clip(X @ gamma, -30, 30))
    pi = expit(np.clip(X @ beta, -30, 30))
    tt = np.clip(t, eps, None)
    zk = (tt / eta) ** k
    logf = np.log(k) - np.log(eta) + (k - 1.0) * (np.log(tt) - np.log(eta)) - zk
    like_event = pi * np.exp(logf)
    like_cens = (1.0 - pi) + pi * np.exp(-zk)
    ll = np.where(e == 1, np.log(np.clip(like_event, eps, None)), np.log(np.clip(like_cens, eps, None)))
    return -np.sum(ll)


def fit_mixture_cure(unit, cols=("product",)):
    # Joint MLE of P(activate ever)=logit^-1(Xb) and Weibull AFT latency (shared shape).
    X, names = _design(unit, cols)
    t = unit["time"].values.astype(float)
    e = unit["event"].values.astype(int)
    p = X.shape[1]
    beta0 = np.zeros(p); beta0[0] = 3.0
    gamma0 = np.zeros(p); gamma0[0] = np.log(max(1.0, float(np.mean(t[e == 1])) if e.sum() else 1.0))
    x0 = np.concatenate([beta0, [0.0], gamma0])
    bounds = [(-8, 8)] * p + [(np.log(0.2), np.log(5))] + [(-5, 12)] * p
    res = minimize(_cure_negll, x0, args=(X, t, e), method="L-BFGS-B", bounds=bounds)
    beta = res.x[:p]
    pi = expit(np.clip(X @ beta, -30, 30))
    out = unit[["product"]].copy(); out["pi"] = pi
    pinf = out.groupby("product")["pi"].mean().round(4)
    return dict(shape_k=round(float(np.exp(res.x[p])), 3), converged=bool(res.success),
                pinf_by_product=pinf)
