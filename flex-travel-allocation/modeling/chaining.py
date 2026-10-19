# Chain stage 1 and stage 2: expected value per sale, bootstrap CIs, and the insights.
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from modeling.stage1_cure import km_activation_curve, saturating


def expected_value_per_sale(p_inf, p_travel, price, margins):
    m_used = margins.used_fraction * price
    m_idle = margins.idle_fraction * price
    m_break = margins.breakage_fraction * price
    return p_inf * (p_travel * m_used + (1.0 - p_travel) * m_idle) + (1.0 - p_inf) * m_break


def value_by_segment(unit, p_travel_by_seg, config, segment_col="product"):
    m = config.margins
    rows = []
    for seg, g in unit.groupby(segment_col):
        p_inf = float(g["event"].mean())
        p_tr = float(p_travel_by_seg.get(seg, g["travelled"].mean()))
        price = float(g["net_sales_euro"].mean())
        ev = expected_value_per_sale(p_inf, p_tr, price, m)
        postpaid = config.postpaid_margin_fraction * price
        rows.append(dict(segment=seg, n=int(len(g)), price=round(price, 2),
                         p_inf=round(p_inf, 4), p_travel=round(p_tr, 4),
                         prepaid_expected_value=round(ev, 2),
                         postpaid_value=round(postpaid, 2),
                         delta_vs_postpaid=round(ev - postpaid, 2)))
    return pd.DataFrame(rows).sort_values("prepaid_expected_value", ascending=False)


def bootstrap_chain(unit, config, p_travel_overall, n_draws=None):
    n_draws = n_draws or config.bootstrap_draws
    rng = np.random.default_rng(config.seed)
    m = config.margins
    event = unit["event"].values; travelled = unit["travelled"].values; price = unit["net_sales_euro"].values
    ev_draws, post_draws = [], []
    for _ in range(n_draws):
        take = rng.integers(0, len(unit), len(unit))
        e = event[take]; tr = travelled[take]; pr = price[take]
        p_inf = e.mean()
        p_tr = tr[e == 1].mean() if (e == 1).any() else 0.0
        mean_price = pr.mean()
        ev_draws.append(expected_value_per_sale(p_inf, p_tr, mean_price, m))
        post_draws.append(config.postpaid_margin_fraction * mean_price)
    ev = np.array(ev_draws); post = np.array(post_draws); delta = ev - post

    def ci(a):
        return dict(lo=round(float(np.percentile(a, 2.5)), 2),
                    mean=round(float(np.mean(a)), 2),
                    hi=round(float(np.percentile(a, 97.5)), 2))

    return dict(n_draws=int(n_draws), prepaid_expected_value=ci(ev),
                postpaid_value=ci(post), delta=ci(delta))


def funnel_leak(value_df, config):
    m = config.margins
    rows = []
    for _, r in value_df.iterrows():
        price = r["price"]; p_inf = r["p_inf"]; p_tr = r["p_travel"]
        m_used = m.used_fraction * price; m_idle = m.idle_fraction * price; m_break = m.breakage_fraction * price
        m_post = config.postpaid_margin_fraction * price
        rows.append(dict(
            segment=r["segment"],
            leak_never_activated=round((1 - p_inf) * (m_post - m_break), 2),
            leak_activated_idle=round(p_inf * (1 - p_tr) * (m_post - m_idle), 2),
            leak_used_shortfall=round(p_inf * p_tr * (m_post - m_used), 2),
            total_gap_vs_postpaid=round(m_post - r["prepaid_expected_value"], 2),
        ))
    return pd.DataFrame(rows)


def breakeven_breakage(value_df, config):
    m = config.margins
    rows = []
    for _, r in value_df.iterrows():
        price = r["price"]; p_tr = r["p_travel"]
        a = p_tr * m.used_fraction * price + (1 - p_tr) * m.idle_fraction * price
        m_break = m.breakage_fraction * price
        m_post = config.postpaid_margin_fraction * price
        denom = a - m_break
        p_star = (m_post - m_break) / denom if denom != 0 else np.nan
        note = "viable for all p_inf in [0,1]" if (not np.isfinite(p_star) or p_star > 1 or p_star < 0) else "breakeven inside range"
        rows.append(dict(segment=r["segment"], actual_p_inf=r["p_inf"],
                         breakeven_p_inf=round(float(p_star), 4) if np.isfinite(p_star) else np.nan,
                         interpretation=note))
    return pd.DataFrame(rows)


def cash_timing(unit, config, weeks=26):
    grid = np.arange(0, weeks + 1) * 7.0
    F = km_activation_curve(unit["time"].values, unit["event"].values, grid)
    try:
        (p_inf, lam), _ = curve_fit(saturating, grid / 7.0, F,
                                    p0=[min(0.999, max(float(F.max()), 0.5)), 0.3],
                                    bounds=([0, 1e-4], [1, 5]), maxfev=20000)
    except Exception:
        p_inf, lam = float(F.max()), np.nan
    total_prepaid = float(unit["net_sales_euro"].sum())
    rows = [dict(weeks_since_sale=int(w), cum_activation=round(float(f), 4),
                 revenue_recognised_eur=round(total_prepaid * f, 0),
                 prepaid_liability_eur=round(total_prepaid * (1 - f), 0))
            for w, f in zip(range(weeks + 1), F)]
    return pd.DataFrame(rows), dict(p_inf=round(float(p_inf), 4),
                                    lam_per_week=round(float(lam), 4) if np.isfinite(lam) else None,
                                    total_prepaid_eur=round(total_prepaid, 0))


def product_reallocation(value_df):
    df = value_df[["segment", "n", "p_inf", "p_travel", "prepaid_expected_value"]].copy()
    df["rank_by_activation"] = df["p_inf"].rank(ascending=False, method="min").astype(int)
    df["rank_by_expected_value"] = df["prepaid_expected_value"].rank(ascending=False, method="min").astype(int)
    return df.sort_values("prepaid_expected_value", ascending=False)


def product_recommendation(value_df, seg1, intensity_df, config):
    # Per-product recommendation from the chained model (activation ceiling, usage, uplift).
    seg1_idx = seg1.set_index("segment")
    intensity = intensity_df.set_index("product")
    order = {"High": 0, "Medium": 1, "Low": 2}
    rows = []
    for _, r in value_df.iterrows():
        seg = r["segment"]
        p_inf = float(r["p_inf"]); breakage = 1.0 - p_inf
        delta = float(r["delta_vs_postpaid"]); postpaid = float(r["postpaid_value"])
        delta_pct = delta / postpaid if postpaid else float("nan")
        lag = float(seg1_idx.loc[seg, "weibull_median_lag_days"]) if seg in seg1_idx.index else float("nan")
        tpad = float(intensity.loc[seg, "trips_per_active_day_model"]) if seg in intensity.index else float("nan")
        priority = "High" if delta >= 100 else ("Medium" if delta >= 20 else "Low")
        usage = ("heavy per-day usage - watch service cost" if (tpad == tpad and tpad >= 0.30)
                 else "light per-day usage - high idle margin, low service cost")
        rec = ("Migrate to travel-allocation recognition: activation %.1f%% (breakage %.1f%%), median lag %.0fd; "
               "+EUR %.0f/sale (%+.0f%%). %s." % (p_inf * 100, breakage * 100, lag, delta, delta_pct * 100, usage))
        rows.append(dict(product=seg, priority=priority, p_inf=round(p_inf, 4), breakage=round(breakage, 4),
                         median_lag_days=round(lag, 1) if lag == lag else None,
                         trips_per_active_day=round(tpad, 3) if tpad == tpad else None,
                         prepaid_ev=round(float(r["prepaid_expected_value"]), 2),
                         postpaid=round(postpaid, 2), delta_per_sale=round(delta, 2),
                         delta_pct=round(float(delta_pct), 4) if delta_pct == delta_pct else None,
                         recommendation=rec))
    df = pd.DataFrame(rows)
    df["_r"] = df["priority"].map(order)
    return df.sort_values(["_r", "delta_per_sale"], ascending=[True, False]).drop(columns="_r")
