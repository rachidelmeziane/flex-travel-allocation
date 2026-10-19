import pandas as pd

from flex_travel_allocation.tradeoff_analysis import product_tradeoff


def _mdp() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"product": "Flex 6", "utilisation": 0.99, "sale_revenue": 1000.0,
             "transition_gap": 10.0, "recoverable_mdp": 0.0},
            {"product": "Flex 80", "utilisation": 0.77, "sale_revenue": 1000.0,
             "transition_gap": 230.0, "recoverable_mdp": 70.0},
            {"product": "Flex 120", "utilisation": 0.50, "sale_revenue": 1000.0,
             "transition_gap": 500.0, "recoverable_mdp": 130.0},
        ]
    )


def test_tradeoff_derived_columns_and_priority():
    result = product_tradeoff(_mdp()).set_index("product")

    assert abs(result.loc["Flex 120", "capacity_saving_pct"] - 0.50) < 1e-9
    assert abs(result.loc["Flex 80", "revenue_at_risk_pct"] - 0.23) < 1e-9
    assert result.loc["Flex 6", "priority"] == "Low"
    assert result.loc["Flex 80", "priority"] == "Medium"
    assert result.loc["Flex 120", "priority"] == "High"
