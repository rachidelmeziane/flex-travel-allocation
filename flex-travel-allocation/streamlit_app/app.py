"""Streamlit demo for the Flex travel allocation model.

Reads the small precomputed artifacts (see precompute_demo.py) so the app is fast
and needs no Spark. Shows the allocation output, then the capacity, revenue-risk /
what-if and trade-off analyses, and finally the per-product recommendation.
"""

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
ARTIFACTS = BASE_DIR / "artifacts"
ASSETS = BASE_DIR / "assets"

st.set_page_config(page_title="SNCB Flex travel allocation", page_icon="\U0001F686", layout="wide")


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / name)


@st.cache_data
def load_summary() -> dict:
    return json.loads((ARTIFACTS / "summary.json").read_text(encoding="utf-8"))


def as_id_string(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    # Ticket numbers are identifiers (up to 23 digits); keep them as text, not integers.
    frame = frame.copy()
    frame[column] = frame[column].astype(str)
    return frame


def render_logo() -> None:
    for name in ["sncb_logo.png", "sncb_logo.jpg", "sncb_logo.jpeg"]:
        path = ASSETS / name
        if path.exists():
            st.image(str(path), width=240)
            return
    svg = ASSETS / "sncb_logo.svg"
    if svg.exists():
        st.markdown(svg.read_text(encoding="utf-8"), unsafe_allow_html=True)


DIRECTION_LABELS = {1: "there", 2: "back"}

summary = load_summary()

header_left, header_right = st.columns([1, 4], vertical_alignment="center", gap="large")
with header_left:
    render_logo()
with header_right:
    st.title("Flex travel allocation")
    st.caption(
        "Allocating Flex subscription sales to individual travels (pax), "
        "with a travel date, hour and direction."
    )

st.divider()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Sales", f"{summary['input_sales']:,}")
col2.metric("Activations", f"{summary['input_activations']:,}")
col3.metric("Travel rows (pax)", f"{summary['output_travel_rows']:,}")
col4.metric("Products", "4")

with st.expander("How the allocation works"):
    st.markdown(
        "- Each **sale** is matched to its **activations** (reconciling the ticket-number "
        "format between the two datasets).\n"
        "- Only activations **inside the validity window** and **within the activation limit** "
        "are kept.\n"
        "- Every activation becomes a **there-travel** and a **back-travel** (one pax each). "
        "Travel times are sampled from the survey probability distributions "
        "(deterministic for a fixed seed).\n"
        f"- Result: `travel rows = 2 x activations` = **{summary['output_travel_rows']:,}**."
    )

st.header("Sample subscription: input \u2192 output")
st.write(
    f"Example ticket **{summary['example_ticket']}** "
    f"(**{summary['example_product']}**): a 6-day pass, fully activated."
)

sales_sample = as_id_string(load_csv("input_sales_sample.csv"), "ticket_number")
example_sale = sales_sample[sales_sample["ticket_number"] == str(summary["example_ticket"])]
activations_sample = as_id_string(
    load_csv("input_activations_sample.csv"), "ticket_number_activations"
)
travels_sample = load_csv("output_travels_sample.csv").copy()
travels_sample["direction"] = travels_sample["travel_direction"].map(DIRECTION_LABELS)

input_col, output_col = st.columns(2)
with input_col:
    st.subheader("Input")
    st.caption("The sale")
    st.dataframe(example_sale, hide_index=True, use_container_width=True)
    st.caption("Its activations (after validity / limit checks)")
    st.dataframe(activations_sample, hide_index=True, use_container_width=True)
with output_col:
    st.subheader("Output travels")
    st.caption("Each activation -> one there-travel + one back-travel")
    st.dataframe(
        travels_sample[["travel_date", "direction", "travel_hour", "travel_timestamp", "pax"]],
        hide_index=True,
        use_container_width=True,
    )

with st.expander("Survey probability distributions used for sampling"):
    prob_left, prob_right = st.columns(2)
    with prob_left:
        st.caption("Minutes to add (there-travel)")
        st.dataframe(load_csv("input_minutes_probabilities.csv"), hide_index=True)
    with prob_right:
        st.caption("Hours between there and back")
        st.dataframe(load_csv("input_hours_probabilities.csv"), hide_index=True)

st.header("Travel distributions")

st.subheader("Pax by product")
st.bar_chart(load_csv("output_by_product.csv"), x="article_description_nl", y="pax", color="#2e6fd6")

st.subheader("Pax by hour and direction")
st.caption("There-travels peak in the morning; back-travels peak in the evening.")
st.bar_chart(load_csv("output_by_hour_direction.csv").set_index("travel_hour")[["there", "back"]], stack=False)

st.subheader("Sampled vs expected distribution (probabilistic expansion)")
st.caption(
    "Expected pax per hour from expanding every activation across the survey probability "
    "buckets (deterministic, no sampling), compared with the sampled allocation."
)
sampled_hd = load_csv("output_by_hour_direction.csv")
expected_hd = load_csv("output_expected_by_hour_direction.csv")
direction_choice = st.radio("Direction", ["there", "back"], horizontal=True, key="expected_dir")
comparison = (
    sampled_hd[["travel_hour", direction_choice]]
    .rename(columns={direction_choice: "sampled"})
    .merge(
        expected_hd[["travel_hour", direction_choice]].rename(columns={direction_choice: "expected"}),
        on="travel_hour",
        how="outer",
    )
    .fillna(0)
    .sort_values("travel_hour")
    .set_index("travel_hour")
)
st.line_chart(comparison[["sampled", "expected"]])

st.subheader("Monte-Carlo confidence bands")
replicates = summary.get("monte_carlo_replicates", "several")
st.caption(
    f"Uncertainty from repeating the stochastic allocation across {replicates} seeds: "
    "mean pax per hour with a 95% band (mean \u00B1 1.96 sd)."
)
monte_carlo = load_csv("output_monte_carlo_by_hour_direction.csv")
mc_direction = st.radio("Direction", ["there", "back"], horizontal=True, key="mc_direction")
mc_value = 1 if mc_direction == "there" else 2
mc_rows = monte_carlo[monte_carlo["travel_direction"] == mc_value]
mc_band = alt.Chart(mc_rows).mark_area(opacity=0.3, color="#2e6fd6").encode(
    x=alt.X("travel_hour:O", title="hour of day"),
    y=alt.Y("lower_pax:Q", title="pax"),
    y2="upper_pax:Q",
)
mc_mean = alt.Chart(mc_rows).mark_line(color="#0b1f3a", point=True).encode(
    x="travel_hour:O", y="mean_pax:Q"
)
st.altair_chart(mc_band + mc_mean, use_container_width=True)

st.subheader("Pax by travel date")
st.line_chart(load_csv("output_by_date.csv"), x="travel_date", y="pax", color="#2e6fd6")

# ---------------------------------------------------------------------------
# Analyses, in order: capacity -> revenue risk / what-if -> trade-off -> recommendation
# ---------------------------------------------------------------------------

st.header("1. Train filling: capacity, sales vs allocation")
st.markdown(
    "The allocation gives the **actual** pax per hour. Capacity planned from **sales** assumes "
    "every paid day is travelled, so potential demand is the allocated demand divided by each "
    "product's utilisation. Sizing trains to sales over-provisions the peak."
)
capacity_by_hour = load_csv("output_capacity_by_hour.csv")
alloc_peak = int(capacity_by_hour["allocation_pax"].max())
sales_peak = int(capacity_by_hour["sales_potential_pax"].max())
total_alloc = int(capacity_by_hour["allocation_pax"].sum())
total_sales = int(capacity_by_hour["sales_potential_pax"].sum())
seats = st.slider("Seats offered per hour", alloc_peak, int(sales_peak * 1.1), sales_peak, 5000)
cap1, cap2, cap3, cap4 = st.columns(4)
cap1.metric("Travels (allocation)", f"{total_alloc:,}")
cap2.metric("Travels (sales-potential)", f"{total_sales:,}")
cap3.metric("Overall fill", f"{total_alloc / total_sales:.1%}")
cap4.metric("Extra peak seats if sizing to sales", f"+{sales_peak / alloc_peak - 1:.0%}")
st.caption("Pax per hour: actual allocation vs sales-planned (full utilisation).")
st.bar_chart(
    capacity_by_hour.set_index("travel_hour")[["allocation_pax", "sales_potential_pax"]],
    stack=False,
)
load_factor = pd.DataFrame(
    {
        "travel_hour": capacity_by_hour["travel_hour"],
        "allocation": capacity_by_hour["allocation_pax"] / seats,
        "sales_potential": capacity_by_hour["sales_potential_pax"] / seats,
    }
).set_index("travel_hour")
st.caption(f"Load factor vs {seats:,} seats/hour (1.0 = full).")
st.line_chart(load_factor[["allocation", "sales_potential"]])

st.subheader("Trains scheduled (sales vs activation)")
st.caption(
    "Trains per hour = pax / (seats per train x target load factor), rounded up. Planning to "
    "actual activation demand schedules fewer trains than planning to sales."
)
ts_col1, ts_col2 = st.columns(2)
seats_per_train = ts_col1.slider("Seats per train", 200, 800, 500, 50)
target_lf = ts_col2.slider("Target max load factor", 0.5, 1.0, 0.8, 0.05)
effective_seats = seats_per_train * target_lf
trains = capacity_by_hour[["travel_hour", "allocation_pax", "sales_potential_pax"]].copy()
trains["sales plan"] = np.ceil(trains["sales_potential_pax"] / effective_seats).astype(int)
trains["activation plan"] = np.ceil(trains["allocation_pax"] / effective_seats).astype(int)
trains_sales_total = int(trains["sales plan"].sum())
trains_activation_total = int(trains["activation plan"].sum())
trains_saved = trains_sales_total - trains_activation_total
tr1, tr2, tr3 = st.columns(3)
tr1.metric("Trains/day (sales plan)", f"{trains_sales_total:,}")
tr2.metric("Trains/day (activation plan)", f"{trains_activation_total:,}")
tr3.metric("Trains saved", f"{trains_saved:,} (-{trains_saved / trains_sales_total:.0%})")
st.caption("Trains per hour by planning basis.")
st.bar_chart(trains.set_index("travel_hour")[["sales plan", "activation plan"]], stack=False)

st.header("2. Revenue risk: sales vs activation")
st.markdown(
    "Two ways to recognise revenue:\n"
    "- **Sales (today):** the full price is collected at purchase, regardless of use.\n"
    "- **Activation (proposed):** revenue follows actual travel at price / max activations.\n\n"
    "Flex holders under-use their passes, so the activation model earns less; charging at "
    "activation also exposes revenue to non-payment (evasion / uncaptured taps)."
)
revenue_risk = load_csv("output_revenue_risk.csv")
non_payment = st.slider("Non-payment rate during activation (%)", 0, 15, 5, 1) / 100
total_sales_rev = revenue_risk["sales_revenue"].sum()
total_activation_rev = revenue_risk["activation_revenue"].sum()
underuse_loss = total_sales_rev - total_activation_rev
total_loss = underuse_loss + total_activation_rev * non_payment
rr1, rr2, rr3, rr4 = st.columns(4)
rr1.metric("Sales revenue", f"EUR {total_sales_rev:,.0f}")
rr2.metric("Activation revenue", f"EUR {total_activation_rev:,.0f}")
rr3.metric("Revenue at risk", f"EUR {total_loss:,.0f}")
rr4.metric("Loss vs sales", f"{total_loss / total_sales_rev:.1%}")
st.bar_chart(
    revenue_risk.set_index("article_description_nl")[["sales_revenue", "activation_revenue"]],
    stack=False,
)
st.dataframe(
    revenue_risk[
        ["article_description_nl", "subscriptions", "utilisation", "sales_revenue",
         "activation_revenue", "underuse_loss"]
    ],
    hide_index=True,
    use_container_width=True,
)

st.subheader("What-if: sale to activation (Markov Decision Process)")
st.markdown(
    "Each subscription is modelled as an MDP: the **state** is the number of paid days already "
    "activated; the **hazard** h[j] (from data) is the chance of activating the next day; the "
    "**reward** is the per-day price; and the operator **action** is to *promote* an extra "
    "activation (hazard +10 pts, cost = 25% of the per-day price). **Value iteration** finds the "
    "optimal promotion policy and the revenue it can recover."
)
mdp = load_csv("output_mdp_what_if.csv")
mdp1, mdp2, mdp3 = st.columns(3)
mdp1.metric("Sales revenue", f"EUR {mdp['sale_revenue'].sum():,.0f}")
mdp2.metric("Activation revenue", f"EUR {mdp['activation_revenue'].sum():,.0f}")
mdp3.metric("MDP-recoverable", f"EUR {mdp['recoverable_mdp'].sum():,.0f}")
st.bar_chart(mdp.set_index("product")[["transition_gap", "recoverable_mdp"]], stack=False)
st.dataframe(
    mdp[
        ["product", "utilisation", "per_day_price", "sale_revenue", "activation_revenue",
         "transition_gap", "recoverable_mdp", "promote_below_utilisation", "breakeven_per_day_price"]
    ],
    hide_index=True,
    use_container_width=True,
)

st.header("3. Trade-off: capacity vs revenue risk")
st.markdown(
    "Both effects grow with **under-use**: the products that expose the most **revenue** if "
    "moved to activation pricing are exactly those that free the most **train capacity** if "
    "planned to actual demand - and where MDP promotion can recover the most. Bubble size = "
    "MDP-recoverable revenue; colour = priority."
)
tradeoff = load_csv("output_tradeoff.csv")
scatter = alt.Chart(tradeoff).encode(
    x=alt.X("capacity_saving_pct:Q", title="Capacity freed if planning to actual", axis=alt.Axis(format="%")),
    y=alt.Y("revenue_at_risk_pct:Q", title="Revenue at risk if switching to activation", axis=alt.Axis(format="%")),
)
points = scatter.mark_circle().encode(
    size=alt.Size("recoverable_mdp:Q", title="MDP recoverable (EUR)"),
    color=alt.Color("priority:N", title="priority"),
)
labels = scatter.mark_text(dx=10, align="left").encode(text="product:N")
st.altair_chart(points + labels, use_container_width=True)
st.dataframe(
    tradeoff[
        ["product", "utilisation", "revenue_at_risk_pct", "capacity_saving_pct",
         "recoverable_mdp", "priority"]
    ],
    hide_index=True,
    use_container_width=True,
)

st.header("4. Recommendation per product")
priority_icon = {"High": "\U0001F534", "Medium": "\U0001F7E0", "Low": "\U0001F7E2"}
for row in tradeoff.itertuples():
    st.markdown(f"{priority_icon.get(row.priority, '')} **{row.product}** ({row.priority}) - {row.recommendation}")

st.divider()
st.header("5. Travel-allocation value model (sale \u2192 activation \u2192 travel)")
st.markdown(
    "A unit-level model of what a **sale** is worth once you account for whether it is "
    "**activated** and then **travelled** \u2013 the sale \u2192 activation \u2192 travel funnel. "
    "It reframes revenue from *cash collected* to *expected value per sale*."
)
try:
    mkey = json.loads((ARTIFACTS / "modeling_key_metrics.json").read_text(encoding="utf-8"))
    m_stage1 = load_csv("modeling_stage1_curves.csv")
    m_intensity = load_csv("modeling_stage2_intensity.csv")
    m_value = load_csv("modeling_expected_value.csv")
    m_cash = load_csv("modeling_cash_timing.csv")
    m_reco = load_csv("modeling_recommendation.csv")
    modeling_ok = True
except Exception as exc:  # noqa: BLE001
    modeling_ok = False
    st.info(f"Run `python -m modeling.run_modeling` to generate the modeling artifacts. ({exc})")

if modeling_ok:
    delta = mkey["delta_per_sale"]
    ci = mkey["delta_ci"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Activation ceiling p\u221e", f"{mkey['activation_ceiling_p_inf']:.1%}")
    c2.metric("Breakage", f"{mkey['breakage']:.1%}")
    c3.metric("Travel-allocation E[value] / sale", f"\u20ac{mkey['prepaid_expected_value_per_sale']:,.0f}")
    c4.metric("vs only-sale / sale", f"\u20ac{delta:+,.0f}", help=f"bootstrap 95% CI [{ci['lo']}, {ci['hi']}]")

    with st.expander("How the two stages are modelled", expanded=False):
        st.markdown(
            "- **Stage 1 \u2013 sale \u2192 activation (mixture cure model).** "
            "P(activate ever) = logit\u207b\u00b9(X\u03b2); timing among activators is Weibull (AFT). "
            "The ceiling **p\u221e** is the activation rate; **breakage = 1 \u2212 p\u221e** is revenue kept at zero service cost. "
            "A saturating curve p(t)=p\u221e(1\u2212e\u207b\u02e1\u1d57) recovers the same two numbers, censoring-corrected.\n"
            "- **Stage 2 \u2013 travel after activation.** Binary travel via logistic regression; trip **count** via a "
            "negative-binomial GLM with a log(exposure) offset \u2192 trips per active-day "
            f"(Poisson overdispersion = {mkey['poisson_overdispersion']} \u2192 negative binomial needed).\n"
            "- **Chaining (per unit sold):** "
            "E[value] = p\u221e\u00b7(P(travel)\u00b7m_used + (1\u2212P(travel))\u00b7m_idle) + (1\u2212p\u221e)\u00b7m_breakage, "
            "margins as fractions of price; uncertainty from a 1000-draw joint bootstrap."
        )

    st.subheader("Stage 1 \u2013 activation ceiling & speed by product")
    st.dataframe(
        m_stage1[["segment", "p_inf", "breakage", "weibull_median_lag_days", "weibull_k"]],
        hide_index=True, use_container_width=True,
    )

    st.subheader("Stage 2 \u2013 usage intensity (trips per active-day)")
    st.altair_chart(
        alt.Chart(m_intensity).mark_bar(color="#2E6FD6").encode(
            x=alt.X("product:N", sort="-y", title=None),
            y=alt.Y("trips_per_active_day_model:Q", title="trips / active-day"),
            tooltip=["product", "trips_per_active_day_model", "mean_trips", "mean_exposure_days"],
        ),
        use_container_width=True,
    )

    st.subheader("Chaining \u2013 expected value per sale: travel allocation vs only sale")
    vlong = m_value.melt(
        id_vars=["segment"], value_vars=["prepaid_expected_value", "postpaid_value"],
        var_name="regime", value_name="eur",
    )
    vlong["regime"] = vlong["regime"].map(
        {"prepaid_expected_value": "travel allocation", "postpaid_value": "only sale"}
    )
    st.altair_chart(
        alt.Chart(vlong).mark_bar().encode(
            x=alt.X("segment:N", title=None, sort="-y"),
            y=alt.Y("eur:Q", title="\u20ac / sale"),
            color=alt.Color("regime:N", scale=alt.Scale(range=["#2E6FD6", "#B9C6DA"]), title=None),
            xOffset="regime:N",
            tooltip=["segment", "regime", "eur"],
        ),
        use_container_width=True,
    )

    st.subheader("Cash timing \u2013 deferred liability run-off")
    cash_long = m_cash.melt(
        id_vars=["weeks_since_sale"], value_vars=["revenue_recognised_eur", "prepaid_liability_eur"],
        var_name="series", value_name="eur",
    )
    cash_long["series"] = cash_long["series"].map(
        {"revenue_recognised_eur": "revenue recognised", "prepaid_liability_eur": "deferred liability"}
    )
    st.altair_chart(
        alt.Chart(cash_long[cash_long["weeks_since_sale"] <= 8]).mark_line(point=True).encode(
            x=alt.X("weeks_since_sale:Q", title="weeks since sale"),
            y=alt.Y("eur:Q", title="\u20ac"),
            color=alt.Color("series:N", title=None),
        ),
        use_container_width=True,
    )

    st.subheader("Insights")
    st.markdown(
        f"- **Regime comparison (common unit).** Travel-allocation E[value] **\u20ac{mkey['prepaid_expected_value_per_sale']:,.0f}** "
        f"vs only-sale **\u20ac{mkey['postpaid_value_per_sale']:,.0f}** per sale "
        f"(**\u20ac{delta:+,.0f}**, 95% CI [{ci['lo']}, {ci['hi']}]).\n"
        f"- **Breakage is negligible** (p\u221e \u2248 {mkey['activation_ceiling_p_inf']:.1%}); activation is fast "
        "(median lag ~2\u20133 days), so almost no revenue is trapped in unactivated units.\n"
        "- **Funnel leak** sits in usage (pricing), not activation (retail) or onboarding \u2013 the migration is structurally sound.\n"
        f"- **Cash timing:** the deferred liability of \u20ac{m_cash['prepaid_liability_eur'].iloc[0]:,.0f} runs off within ~3 weeks.\n"
        f"- **Activation lag is bimodal** ({mkey['lag_bimodal']}): a same-day spike plus a ~3-day hump "
        "(possible gifting / channel stockpiling \u2013 worth a separate look)."
    )

    st.subheader("Recommendation per product")
    for row in m_reco.itertuples():
        st.markdown(f"{priority_icon.get(row.priority, '')} **{row.product}** ({row.priority}) \u2013 {row.recommendation}")
    st.subheader("Model output (data)")
    st.caption("Tables written by the modeling phase (mirrored from modeling/output/).")
    value_data = m_value.rename(
        columns={
            "prepaid_expected_value": "travel_allocation_eur",
            "postpaid_value": "only_sale_eur",
            "delta_vs_postpaid": "delta_eur",
        }
    )
    st.dataframe(value_data, hide_index=True, use_container_width=True)
    reco_data = m_reco.rename(columns={"prepaid_ev": "travel_allocation_eur", "postpaid": "only_sale_eur"})
    st.dataframe(reco_data, hide_index=True, use_container_width=True)

    st.markdown(
        "**Governed output table `flex_unit_scores`** – the scored per-subscription output the "
        "pipeline writes (expected trips, activation probability, breakage value)."
    )
    try:
        scores_by_product = load_csv("modeling_unit_scores_by_product.csv")
        st.dataframe(scores_by_product, hide_index=True, use_container_width=True)
        scores_sample = load_csv("modeling_unit_scores.csv")
        scores_sample["ticket_number"] = scores_sample["ticket_number"].astype(str)
        st.caption(f"Sample of {len(scores_sample):,} scored units (full table = one row per subscription).")
        st.dataframe(scores_sample, hide_index=True, use_container_width=True)
    except Exception:  # noqa: BLE001
        st.info("Run `python score_units_simulation.py` to materialise flex_unit_scores locally.")

    st.caption(
        "Productionised: the Stage 1 cure model and Stage 2 negative-binomial GLM are fitted "
        "weekly by the estimation pipeline (writes flex_model_parameters); the daily allocation "
        "pipeline then scores every unit in pure Spark (expected_trips, p_activated_by_now, "
        "expected_breakage_value) into flex_unit_scores."
    )

st.divider()
st.caption(
    f"Deterministic allocation (seed = {summary['seed']}). "
    "Figures precomputed from the dummy dataset via the PySpark model."
)
