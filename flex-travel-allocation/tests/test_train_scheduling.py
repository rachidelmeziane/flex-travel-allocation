import pandas as pd

from flex_travel_allocation.train_scheduling import train_optimisation_summary, trains_by_hour


def _capacity() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "travel_hour": [7, 8],
            "allocation_pax": [150, 90],
            "sales_potential_pax": [300, 210],
        }
    )


def test_trains_rounded_up_per_hour():
    trains = trains_by_hour(_capacity(), seats_per_train=100, target_load_factor=1.0).set_index("travel_hour")
    # hour 7: sales 300/100 = 3, activation 150/100 -> 2 (ceil 1.5); hour 8: sales 210 -> 3, activation 90 -> 1
    assert trains.loc[7, "trains_sales"] == 3
    assert trains.loc[7, "trains_activation"] == 2
    assert trains.loc[8, "trains_sales"] == 3
    assert trains.loc[8, "trains_activation"] == 1
    assert trains.loc[8, "trains_saved"] == 2


def test_optimisation_summary_totals_and_reduction():
    summary = train_optimisation_summary(trains_by_hour(_capacity(), seats_per_train=100))
    assert summary["trains_sales"] == 6
    assert summary["trains_activation"] == 3
    assert summary["trains_saved"] == 3
    assert abs(summary["reduction_pct"] - 0.5) < 1e-9
    assert summary["peak_trains_sales"] == 3
