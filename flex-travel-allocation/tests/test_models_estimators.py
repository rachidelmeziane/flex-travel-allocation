import numpy as np
import pandas as pd

from flex_travel_allocation.models.estimators import estimate_parameters
from flex_travel_allocation.models.parameters import PARAMETER_COLUMNS


def _synthetic_units():
    rng = np.random.default_rng(0)
    rows = []
    for product, exposure, rate in [("Product A", 30.0, 9), ("Product B", 300.0, 30)]:
        for _ in range(300):
            lag = float(rng.integers(0, 7))
            rows.append(
                dict(article_description_nl=product, event=1, time=lag, lag_days=lag,
                     n_activations=int(rng.poisson(rate)), exposure_days=exposure, net_sales_euro=100.0)
            )
    return pd.DataFrame(rows)


def test_estimate_parameters_schema_and_ranges():
    params = estimate_parameters(_synthetic_units(), fit_version="test", as_of="2026-01-01")
    for column in PARAMETER_COLUMNS:
        assert column in params.columns
    assert len(params) == 2
    assert params["p_inf"].between(0.0, 1.0).all()
    assert (params["nb_rate_per_day"] > 0).all()
    assert (params["n"] == 300).all()
    assert (params["fit_version"] == "test").all()


def test_stage2_intensity_orders_products():
    # Higher trip rate per active-day for the heavier-usage product.
    params = estimate_parameters(_synthetic_units(), fit_version="test", as_of="2026-01-01")
    by_product = params.set_index("article_description_nl")["nb_rate_per_day"]
    assert by_product["Product A"] > by_product["Product B"]
