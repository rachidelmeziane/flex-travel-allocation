"""Trade-off between train capacity and revenue risk, with a per-product recommendation.

Under-use drives both effects: a product moved to activation pricing loses the most
revenue exactly where planning capacity to actual demand frees the most train capacity,
and where MDP promotion can recover the most. This turns the per-product MDP table into
a ranked recommendation.
"""

from __future__ import annotations

import pandas as pd


def product_tradeoff(mdp: pd.DataFrame) -> pd.DataFrame:
    out = mdp.copy()
    out["revenue_at_risk_pct"] = out["transition_gap"] / out["sale_revenue"]
    out["capacity_saving_pct"] = 1.0 - out["utilisation"]
    out["mdp_recoverable_pct"] = out["recoverable_mdp"] / out["sale_revenue"]

    priorities = []
    recommendations = []
    for row in out.itertuples():
        if row.utilisation >= 0.90:
            priorities.append("Low")
            recommendations.append(
                "Keep the sales model (activation is ~revenue-neutral); capacity already "
                "well matched; no promotion needed."
            )
        elif row.utilisation >= 0.65:
            priorities.append("Medium")
            recommendations.append(
                f"Keep sales pricing; plan capacity to actual demand "
                f"(-{row.capacity_saving_pct:.0%} fleet); pilot targeted promotion "
                f"(recover EUR {row.recoverable_mdp:,.0f})."
            )
        else:
            priorities.append("High")
            recommendations.append(
                f"Keep sales pricing (activation would cut revenue "
                f"-{row.revenue_at_risk_pct:.0%}); plan capacity to actual "
                f"(-{row.capacity_saving_pct:.0%} fleet); prioritise promotion "
                f"(recover EUR {row.recoverable_mdp:,.0f})."
            )
    out["priority"] = priorities
    out["recommendation"] = recommendations
    return out[
        [
            "product", "utilisation", "revenue_at_risk_pct", "capacity_saving_pct",
            "recoverable_mdp", "mdp_recoverable_pct", "priority", "recommendation",
        ]
    ]
