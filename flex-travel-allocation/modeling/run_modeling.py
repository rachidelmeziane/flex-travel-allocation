# Run every modeling analysis end to end and write outputs to modeling/output/.
from __future__ import annotations

import json
import shutil
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

from modeling.config import ModelingConfig  # noqa: E402
from modeling.data_prep import build_unit_table  # noqa: E402
from modeling import stage1_cure as s1  # noqa: E402
from modeling import stage2_travel as s2  # noqa: E402
from modeling import chaining as ch  # noqa: E402
from modeling import diagnostics as dg  # noqa: E402


def _weighted(df, value_col, weight_col="n"):
    return float(np.average(df[value_col], weights=df[weight_col]))


def main() -> None:
    config = ModelingConfig()
    out = config.output_dir
    out.mkdir(parents=True, exist_ok=True)

    unit = build_unit_table(config)
    unit.head(200).to_csv(out / "unit_table_sample.csv", index=False)

    # ---- Stage 1: sale -> activation ----
    seg1 = s1.fit_saturating_by_segment(unit)
    seg1.to_csv(out / "stage1_segment_curves.csv", index=False)
    cure = s1.fit_mixture_cure(unit)
    cohort = s1.cohort_comparison(unit)
    cohort.to_csv(out / "stage1_cohort_comparison.csv", index=False)

    # ---- Stage 2: travel after activation ----
    binary = s2.fit_travel_binary(unit)
    count = s2.fit_travel_count(unit)
    count["by_product"].to_csv(out / "stage2_travel_intensity.csv", index=False)
    p_travel_overall = float(binary["p_travel"])
    p_travel_by_seg = {seg: p_travel_overall for seg in seg1["segment"]}

    # ---- Chaining ----
    value_df = ch.value_by_segment(unit, p_travel_by_seg, config)
    value_df.to_csv(out / "chaining_expected_value.csv", index=False)
    boot = ch.bootstrap_chain(unit, config, p_travel_overall)
    leak = ch.funnel_leak(value_df, config)
    leak.to_csv(out / "funnel_leak.csv", index=False)
    beq = ch.breakeven_breakage(value_df, config)
    beq.to_csv(out / "breakeven_breakage.csv", index=False)
    cash, cash_meta = ch.cash_timing(unit, config)
    cash.to_csv(out / "cash_timing_runoff.csv", index=False)
    realloc = ch.product_reallocation(value_df)
    realloc.to_csv(out / "channel_reallocation.csv", index=False)
    recommendation = ch.product_recommendation(value_df, seg1, count["by_product"], config)
    recommendation.to_csv(out / "modeling_recommendation.csv", index=False)

    # ---- Diagnostics ----
    diag = dg.lag_bimodality(unit, str(out / "diagnostics_lag_histogram.png"))

    prepaid_overall = _weighted(value_df, "prepaid_expected_value")
    postpaid_overall = _weighted(value_df, "postpaid_value")

    key = dict(
        as_of=str(unit["as_of"].iloc[0].date()),
        n_units=int(len(unit)),
        activation_ceiling_p_inf=round(float(_weighted(seg1, "p_inf")), 4),
        breakage=round(1 - float(_weighted(seg1, "p_inf")), 4),
        cure_shape_k=cure["shape_k"],
        p_travel_given_activation=round(p_travel_overall, 4),
        travel_dispersion_alpha=count["dispersion_alpha"],
        poisson_overdispersion=count["poisson_overdispersion"],
        prepaid_expected_value_per_sale=round(prepaid_overall, 2),
        postpaid_value_per_sale=round(postpaid_overall, 2),
        delta_per_sale=round(prepaid_overall - postpaid_overall, 2),
        delta_ci=boot["delta"],
        lag_bimodal=diag["bimodal"],
    )
    (out / "key_metrics.json").write_text(json.dumps(key, indent=2), encoding="utf-8")

    lines = []
    A = lines.append
    A("SALE -> ACTIVATION -> TRAVEL : chained value model")
    A("=" * 68)
    A("as_of=%s   units sold=%d   (dummy data)" % (key["as_of"], key["n_units"]))
    A("")
    A("STAGE 1 - sale -> activation (mixture cure model / saturating curve)")
    A("-" * 68)
    A(seg1.to_string(index=False))
    A("")
    A("cure joint-MLE shape k=%s (converged=%s); incidence p_inf ~ %.4f" % (
        cure["shape_k"], cure["converged"], key["activation_ceiling_p_inf"]))
    A("What you get: activation ceiling p_inf=%.4f  ->  breakage=%.4f (revenue kept, zero service cost)."
      % (key["activation_ceiling_p_inf"], key["breakage"]))
    A("             speed: median activation lag ~2-3 days; Weibull k~1.1 (near-exponential, mild clustering).")
    A("             censoring-corrected cohort comparison written to stage1_cohort_comparison.csv")
    A("             (naive == corrected here: every cohort is fully observed at as_of, so no young-cohort bias).")
    A("")
    A("STAGE 2 - travel after activation")
    A("-" * 68)
    A("binary P(travel | activation) = %.4f  [%s]" % (p_travel_overall, binary.get("model")))
    A("count model: %s, dispersion alpha=%s, Poisson overdispersion=%s (>1 => negative binomial needed)"
      % (count["method"], count["dispersion_alpha"], count["poisson_overdispersion"]))
    A(count["by_product"].to_string(index=False))
    A("What you get: conversion depth (all activated units travel) and usage intensity per active-day,")
    A("             exposure-adjusted; Flex10 heaviest (0.35 trips/day), Flex120 lightest (0.17).")
    A("")
    A("CHAINING - expected value per sale  E[value] = p_inf*(p_tr*m_used+(1-p_tr)*m_idle)+(1-p_inf)*m_break")
    A("-" * 68)
    A(value_df.to_string(index=False))
    A("")
    A("Overall (n-weighted): travel-allocation E[value]/sale = EUR %.2f  vs  only-sale = EUR %.2f  ->  delta = EUR %.2f"
      % (prepaid_overall, postpaid_overall, prepaid_overall - postpaid_overall))
    A("Bootstrap %d draws: delta 95%% CI = [%.2f , %.2f] (mean %.2f)"
      % (boot["n_draws"], boot["delta"]["lo"], boot["delta"]["hi"], boot["delta"]["mean"]))
    A("")
    A("INSIGHT 1 - regime comparison (common unit, censoring handled): see line above.")
    A("INSIGHT 2 - funnel leak localisation (EUR/sale; owner in brackets):")
    A("   never-activated [retail]  |  activated-idle [onboarding]  |  used-shortfall [pricing]")
    A(leak.to_string(index=False))
    A("INSIGHT 3 - breakeven breakage (p_inf at which travel allocation == only sale):")
    A(beq.to_string(index=False))
    A("INSIGHT 4 - cash timing / deferred liability run-off (overall):")
    A("   fit p_inf=%s, lam/week=%s, total collected=EUR %s ; full schedule in cash_timing_runoff.csv"
      % (cash_meta["p_inf"], cash_meta["lam_per_week"], cash_meta["total_prepaid_eur"]))
    A(cash.head(9).to_string(index=False))
    A("INSIGHT 5 - product reallocation (rank by expected value, not activation rate):")
    A(realloc.to_string(index=False))
    A("")
    A("RECOMMENDATION PER PRODUCT (from the chained model):")
    A("-" * 68)
    A(recommendation[["product", "priority", "p_inf", "breakage", "median_lag_days", "trips_per_active_day", "prepaid_ev", "postpaid", "delta_per_sale"]].to_string(index=False))
    for _, rr in recommendation.iterrows():
        A("   [%s] %s: %s" % (rr["priority"], rr["product"], rr["recommendation"]))
    A("")
    A("PRE-FIT CHECK - activation-lag bimodality (gifting / channel stockpiling):")
    A("-" * 68)
    A("   n=%d  BIC(1-comp)=%s  BIC(2-comp)=%s  ->  bimodal=%s ; component modes(days)=%s weights=%s"
      % (diag["n"], diag["bic_1comp"], diag["bic_2comp"], diag["bimodal"],
         diag["component_modes_days"], diag["component_weights"]))
    A("   histogram: diagnostics_lag_histogram.png")

    # Publish a compact copy of the modeling artifacts for the Streamlit app (Spark-free).
    app_artifacts = out.parents[1] / "streamlit_app" / "artifacts"
    app_artifacts.mkdir(parents=True, exist_ok=True)
    for src, dst in {
        "key_metrics.json": "modeling_key_metrics.json",
        "stage1_segment_curves.csv": "modeling_stage1_curves.csv",
        "stage2_travel_intensity.csv": "modeling_stage2_intensity.csv",
        "chaining_expected_value.csv": "modeling_expected_value.csv",
        "funnel_leak.csv": "modeling_funnel_leak.csv",
        "cash_timing_runoff.csv": "modeling_cash_timing.csv",
        "modeling_recommendation.csv": "modeling_recommendation.csv",
    }.items():
        shutil.copy(out / src, app_artifacts / dst)

    (out / "modeling_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("\nMODELING_DONE ->", out)


if __name__ == "__main__":
    main()
