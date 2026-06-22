"""Tests for the optimisation layer — constraints + the robustness mechanism."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cvxpy")  # skip cleanly if cvxpy isn't installed (before optimizers import)

from src.optimization import optimizers as opt  # noqa: E402


def _spd(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    return A @ A.T / n + np.eye(n) * 1e-3


def test_equal_weight():
    w = opt.equal_weight(4)
    assert np.allclose(w, 0.25)


def test_markowitz_respects_constraints():
    n = 6
    mu = np.linspace(0.01, 0.06, n)
    w = opt.markowitz(mu, _spd(n), gamma=5.0, w_max=0.30)
    assert abs(w.sum() - 1) < 1e-6
    assert (w >= -1e-8).all()
    assert (w <= 0.30 + 1e-6).all()


def test_robustness_downweights_uncertain_asset():
    # Two assets, identical mu and identical (diagonal) risk; asset 0 has a much
    # larger epistemic uncertainty. The robust optimiser must tilt toward asset 1.
    mu = np.array([0.03, 0.03])
    Sigma = np.eye(2) * 0.004
    sigma_epi = np.array([0.05, 0.001])
    w = opt.robust_mean_variance(mu, Sigma, sigma_epi, gamma=1.0, kappa=5.0, w_max=1.0)
    assert w[1] > w[0]
    # With zero penalty the tie is symmetric (sanity check the mechanism is the cause).
    w0 = opt.robust_mean_variance(mu, Sigma, sigma_epi, gamma=1.0, kappa=0.0, w_max=1.0)
    assert abs(w0[0] - w0[1]) < 1e-3
