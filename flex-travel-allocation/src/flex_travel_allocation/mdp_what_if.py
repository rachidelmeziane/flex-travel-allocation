"""What-if analysis of the sale -> activation transition as a Markov Decision Process.

State j = number of paid days already activated (0..N). From each state the holder
activates the next day with hazard h[j] (estimated from data) or lapses (absorbing).
Each activation earns per_day_price. The operator action ``promote`` raises the hazard
by ``uplift`` at ``incentive_cost``; value iteration returns the optimal policy and the
expected activation-model revenue per subscription. The state space is acyclic, so a
single backward sweep is an exact value iteration.
"""

from __future__ import annotations

import numpy as np


def estimate_hazards(used_counts: np.ndarray) -> np.ndarray:
    # used_counts[m] = number of subscriptions that activated exactly m days (m = 0..N).
    counts = np.asarray(used_counts, dtype=float)
    survivors = np.cumsum(counts[::-1])[::-1]  # survivors[j] = subscriptions reaching >= j
    hazards = np.zeros(len(counts) - 1)
    for j in range(len(hazards)):
        hazards[j] = survivors[j + 1] / survivors[j] if survivors[j] > 0 else 0.0
    return hazards


def _expected_activations(hazards: np.ndarray, policy: np.ndarray, uplift: float) -> float:
    n = len(hazards)
    expected = np.zeros(n + 1)
    for j in range(n - 1, -1, -1):
        hazard = min(1.0, hazards[j] + uplift) if policy[j] else hazards[j]
        expected[j] = hazard * (1.0 + expected[j + 1])
    return float(expected[0])


def evaluate_transition(
    hazards: np.ndarray,
    per_day_price: float,
    uplift: float = 0.0,
    incentive_cost: float = 0.0,
    discount: float = 1.0,
) -> dict:
    n = len(hazards)
    value = np.zeros(n + 1)
    value_status_quo = np.zeros(n + 1)
    policy = np.zeros(n, dtype=int)
    for j in range(n - 1, -1, -1):
        promoted_hazard = min(1.0, hazards[j] + uplift)
        value_none = hazards[j] * (per_day_price + discount * value[j + 1])
        value_promote = (
            promoted_hazard * (per_day_price + discount * value[j + 1]) - incentive_cost
        )
        if value_promote > value_none:
            value[j] = value_promote
            policy[j] = 1
        else:
            value[j] = value_none
        value_status_quo[j] = hazards[j] * (
            per_day_price + discount * value_status_quo[j + 1]
        )
    promoted_states = np.flatnonzero(policy)
    return {
        "value_status_quo": float(value_status_quo[0]),
        "value_optimal": float(value[0]),
        "expected_activations_status_quo": _expected_activations(hazards, np.zeros(n, int), 0.0),
        "expected_activations_optimal": _expected_activations(hazards, policy, uplift),
        "policy": policy.tolist(),
        "promote_below_state": int(promoted_states.max() + 1) if promoted_states.size else 0,
    }
