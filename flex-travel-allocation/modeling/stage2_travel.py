# Stage 2: travel after activation. Logistic (binary travelled) + Negative Binomial (trip count).
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit_travel_binary(unit):
    act = unit[unit["activated"]].copy()
    rate = float(act["travelled"].mean())
    if act["travelled"].nunique() < 2:
        return dict(model="degenerate", p_travel=rate,
                    note="every activated unit travelled at least once; logistic is not identifiable (P=1).")
    y = act["travelled"].values.astype(float)
    Xdf = pd.get_dummies(act["product"].astype(str), drop_first=True).astype(float)
    Xdf.insert(0, "intercept", 1.0)
    Xdf["tsa_days"] = (act["as_of"] - act["first_activation_date"]).dt.days.astype(float)
    res = sm.Logit(y, Xdf.values).fit(disp=0)
    p_hat = 1.0 / (1.0 + np.exp(-(Xdf.values @ res.params)))
    return dict(model="logit", p_travel=float(p_hat.mean()))


def fit_travel_count(unit):
    act = unit[unit["activated"]].copy()
    offset = np.log(act["exposure_days"].clip(lower=1).values.astype(float))
    Xdf = pd.get_dummies(act["product"].astype(str), drop_first=True).astype(float)
    Xdf.insert(0, "intercept", 1.0)
    cols = list(Xdf.columns)
    X = Xdf.values
    y = act["n_activations"].values.astype(float)

    poisson = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset).fit()
    overdispersion = float(poisson.pearson_chi2 / poisson.df_resid)
    method = "nb2"
    try:
        nb = sm.NegativeBinomial(y, X, offset=offset).fit(disp=0, maxiter=200)
        params = np.asarray(nb.params[: X.shape[1]], float)
        alpha = float(nb.params[-1])
    except Exception:
        mu = poisson.predict()
        alpha = float(max(1e-6, np.mean(((y - mu) ** 2 - mu) / np.clip(mu ** 2, 1e-9, None))))
        nb = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha), offset=offset).fit()
        params = np.asarray(nb.params, float)
        method = "nb2-glm"

    rows = []
    for pr in sorted(act["product"].unique()):
        row = {c: 0.0 for c in cols}
        row["intercept"] = 1.0
        for c in cols:
            if c != "intercept" and c.endswith(pr):
                row[c] = 1.0
        lp = float(sum(params[i] * row[c] for i, c in enumerate(cols)))
        g = act[act["product"] == pr]
        rate_emp = float(g["n_activations"].sum() / g["exposure_days"].clip(lower=1).sum())
        rows.append(dict(product=pr, trips_per_active_day_model=round(float(np.exp(lp)), 4),
                         trips_per_active_day_empirical=round(rate_emp, 4),
                         mean_trips=round(float(g["n_activations"].mean()), 2),
                         mean_exposure_days=round(float(g["exposure_days"].mean()), 1),
                         n=int(len(g))))
    return dict(method=method, dispersion_alpha=round(float(alpha), 4),
                poisson_overdispersion=round(overdispersion, 2),
                by_product=pd.DataFrame(rows).sort_values("product"))
