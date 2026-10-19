"""Pandas scenario tables for the revenue-risk and MDP what-if analyses.

Shared by the precompute (demo artifacts) and the standalone report scripts so the
Spark estimation and the numpy MDP solve are defined once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyspark.sql.functions as sf
from pyspark.sql import DataFrame

from flex_travel_allocation.allocation import join_sales_to_activations
from flex_travel_allocation.capacity_analysis import train_filling_by_hour
from flex_travel_allocation.mdp_what_if import estimate_hazards, evaluate_transition
from flex_travel_allocation.revenue_risk import revenue_risk_by_product


def revenue_risk_table(
    sales: DataFrame,
    articles: DataFrame,
    activations: DataFrame,
    non_payment_rate: float = 0.0,
) -> pd.DataFrame:
    return revenue_risk_by_product(
        sales=sales, articles=articles, activations=activations, non_payment_rate=non_payment_rate
    ).toPandas()


def mdp_what_if_table(
    sales: DataFrame,
    articles: DataFrame,
    activations: DataFrame,
    uplift: float = 0.10,
    cost_fraction: float = 0.25,
) -> pd.DataFrame:
    used_per_ticket = (
        join_sales_to_activations(sales=sales, articles=articles, activations=activations)
        .groupBy("ticket_number")
        .agg(sf.count(sf.lit(1)).alias("used"))
    )
    ticket = (
        sales.join(sf.broadcast(articles), on="article_description_nl", how="inner")
        .join(used_per_ticket, on="ticket_number", how="left")
        .fillna(0, subset=["used"])
    )
    histogram = (
        ticket.groupBy(
            "article_code_int", "article_description_nl", "maximum_activations_allowed", "used"
        )
        .count()
        .toPandas()
    )
    prices = (
        ticket.groupBy("article_code_int", "article_description_nl", "maximum_activations_allowed")
        .agg(
            sf.count(sf.lit(1)).alias("subscriptions"),
            sf.sum("net_sales_euro").alias("sales_revenue"),
            sf.avg("net_sales_euro").alias("avg_price"),
        )
        .orderBy("article_code_int")
        .toPandas()
    )

    records = []
    for _, product in prices.iterrows():
        code = product["article_code_int"]
        n = int(product["maximum_activations_allowed"])
        subs = int(product["subscriptions"])
        sale_revenue = float(product["sales_revenue"])
        per_day_price = float(product["avg_price"]) / n

        counts = np.zeros(n + 1)
        for _, hrow in histogram[histogram["article_code_int"] == code].iterrows():
            counts[int(hrow["used"])] = hrow["count"]

        hazards = estimate_hazards(counts)
        status_quo = evaluate_transition(hazards, per_day_price)
        promo_cost = evaluate_transition(
            hazards, per_day_price, uplift=uplift, incentive_cost=cost_fraction * per_day_price
        )
        expected = status_quo["expected_activations_status_quo"]
        activation_revenue = per_day_price * expected * subs
        recoverable = (promo_cost["value_optimal"] - promo_cost["value_status_quo"]) * subs

        records.append(
            {
                "product": product["article_description_nl"],
                "max_activations": n,
                "subscriptions": subs,
                "utilisation": round(expected / n, 4),
                "per_day_price": round(per_day_price, 2),
                "sale_revenue": round(sale_revenue, 0),
                "activation_revenue": round(activation_revenue, 0),
                "transition_gap": round(sale_revenue - activation_revenue, 0),
                "recoverable_mdp": round(recoverable, 0),
                "promote_below_utilisation": round(promo_cost["promote_below_state"] / n, 2),
                "breakeven_per_day_price": round(sale_revenue / (expected * subs), 2),
            }
        )
    return pd.DataFrame(records)


def capacity_by_hour_table(
    travels: DataFrame,
    sales: DataFrame,
    articles: DataFrame,
    activations: DataFrame,
) -> pd.DataFrame:
    product_utilisation = revenue_risk_by_product(
        sales=sales, articles=articles, activations=activations, non_payment_rate=0.0
    ).select("article_description_nl", "utilisation")
    return train_filling_by_hour(travels, product_utilisation).toPandas()