import numpy as np

from flex_travel_allocation.mdp_what_if import estimate_hazards, evaluate_transition


def test_estimate_hazards_from_used_counts():
    # 20 subscriptions (max 2 days): 10 used 0 days, 0 used 1, 10 used 2.
    hazards = estimate_hazards([10, 0, 10])
    assert np.allclose(hazards, [0.5, 1.0])


def test_status_quo_value_equals_price_times_expected_activations():
    result = evaluate_transition(np.array([0.5, 0.5]), per_day_price=10.0)
    assert np.isclose(result["expected_activations_status_quo"], 0.75)
    assert np.isclose(result["value_status_quo"], 7.5)  # 10 * 0.75


def test_free_promotion_lifts_activations_and_value():
    result = evaluate_transition(
        np.array([0.5, 0.5]), per_day_price=10.0, uplift=0.5, incentive_cost=0.0
    )
    # uplift raises both hazards to 1.0 -> 2 activations, value 20.
    assert result["policy"] == [1, 1]
    assert np.isclose(result["expected_activations_optimal"], 2.0)
    assert np.isclose(result["value_optimal"], 20.0)


def test_costly_promotion_only_where_it_pays():
    # High cost makes promotion not worthwhile anywhere -> status-quo policy.
    result = evaluate_transition(
        np.array([0.5, 0.5]), per_day_price=10.0, uplift=0.1, incentive_cost=100.0
    )
    assert result["policy"] == [0, 0]
    assert np.isclose(result["value_optimal"], result["value_status_quo"])
