"""Portfolio optimisers — the headline robust allocator + the two benchmarks.

All long-only, fully invested, with a per-asset cap. Weights are returned as a
numpy array aligned with the input `mu` ordering.

Headline
--------
robust_mean_variance solves

    max_w   muᵀw  -  kappa·‖diag(sigma_epi)·w‖₂  -  gamma·wᵀΣw
    s.t.    Σw = 1,  0 ≤ w ≤ w_max

The middle (second-order-cone) term is the worst-case loss of muᵀw over an
ellipsoidal uncertainty set around mu whose radius scales with the per-asset
epistemic forecast uncertainty. Uncertain forecasts are therefore trusted less.

Benchmarks
----------
markowitz : same constraints, no robustness term (the naive optimiser).
equal_weight : 1/N, the famously hard-to-beat floor.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np


def _solve(objective, n: int, w_max: float) -> np.ndarray:
    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1, w >= 0, w <= w_max]
    prob = cp.Problem(cp.Maximize(objective(w)), constraints)
    prob.solve()
    if w.value is None:
        raise RuntimeError(f"Optimiser failed (status={prob.status}).")
    # Clean tiny numerical violations and renormalise.
    weights = np.clip(w.value, 0.0, None)
    return weights / weights.sum()


def equal_weight(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def markowitz(mu, Sigma, gamma: float, w_max: float) -> np.ndarray:
    Sig = cp.psd_wrap(np.asarray(Sigma))
    mu = np.asarray(mu)
    return _solve(lambda w: mu @ w - gamma * cp.quad_form(w, Sig), len(mu), w_max)


def robust_mean_variance(mu, Sigma, sigma_epi, gamma: float, kappa: float, w_max: float) -> np.ndarray:
    Sig = cp.psd_wrap(np.asarray(Sigma))
    mu = np.asarray(mu)
    sigma_epi = np.asarray(sigma_epi)
    return _solve(
        lambda w: mu @ w
        - kappa * cp.norm(cp.multiply(sigma_epi, w), 2)
        - gamma * cp.quad_form(w, Sig),
        len(mu),
        w_max,
    )
