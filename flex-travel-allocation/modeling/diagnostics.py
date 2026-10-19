# Diagnostic: is the activation-lag distribution bimodal (gifting / channel stockpiling)?
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _normal_logpdf(x, mu, sd):
    sd = max(float(sd), 1e-6)
    return -0.5 * np.log(2 * np.pi * sd * sd) - 0.5 * ((x - mu) / sd) ** 2


def _em_two_component(x, iters=300):
    w = np.array([0.5, 0.5])
    mu = np.array([np.quantile(x, 0.25), np.quantile(x, 0.75)])
    sd = np.array([x.std() / 2 + 1e-3, x.std() / 2 + 1e-3])
    for _ in range(iters):
        r0 = w[0] * np.exp(_normal_logpdf(x, mu[0], sd[0]))
        r1 = w[1] * np.exp(_normal_logpdf(x, mu[1], sd[1]))
        denom = r0 + r1 + 1e-12
        g0 = r0 / denom; g1 = r1 / denom
        n0 = g0.sum(); n1 = g1.sum()
        w = np.array([n0, n1]) / len(x)
        mu = np.array([(g0 * x).sum() / max(n0, 1e-9), (g1 * x).sum() / max(n1, 1e-9)])
        sd = np.array([np.sqrt((g0 * (x - mu[0]) ** 2).sum() / max(n0, 1e-9)),
                       np.sqrt((g1 * (x - mu[1]) ** 2).sum() / max(n1, 1e-9))])
    ll = float(np.sum(np.log(w[0] * np.exp(_normal_logpdf(x, mu[0], sd[0]))
                             + w[1] * np.exp(_normal_logpdf(x, mu[1], sd[1])) + 1e-12)))
    return dict(w=w, mu=mu, sd=sd, loglik=ll)


def lag_bimodality(unit, out_png):
    act = unit[(unit["activated"]) & (unit["lag_days"] >= 0)]
    lag = act["lag_days"].values.astype(float)
    x = np.log1p(lag)
    n = len(x)
    ll1 = float(np.sum(_normal_logpdf(x, x.mean(), x.std())))
    two = _em_two_component(x)
    bic1 = -2 * ll1 + 2 * np.log(n)
    bic2 = -2 * two["loglik"] + 5 * np.log(n)
    bimodal = bool(bic2 + 10 < bic1)
    modes = np.sort(np.expm1(two["mu"]))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lag, bins=range(0, int(lag.max()) + 2), color="#2E6FD6", alpha=0.85, edgecolor="white")
    ax.set_xlabel("activation lag (days since sale)")
    ax.set_ylabel("subscriptions")
    ax.set_title("Activation-lag distribution among activated units")
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)

    return dict(n=int(n), bic_1comp=round(bic1, 1), bic_2comp=round(bic2, 1), bimodal=bimodal,
                component_modes_days=[round(float(m), 1) for m in modes],
                component_weights=[round(float(w), 3) for w in two["w"]])
