"""Tests for the bootstrap-ensemble uncertainty layer."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import get_forecaster
from src.models.uncertainty import (
    BootstrapEnsembleForecaster,
    calibration_report,
)


def _xy(n: int = 400):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = 0.5 * X["f1"] + rng.normal(scale=0.3, size=n)  # signal + noise
    return X, y


def test_ensemble_uncertainty_is_well_formed():
    X, y = _xy()
    ens = BootstrapEnsembleForecaster(lambda: get_forecaster("ridge"), n_boot=20).fit(X, y)
    u = ens.predict_uncertainty(X)
    assert set(u.columns) == {"mu", "sigma_epistemic", "sigma_aleatoric", "sigma_total"}
    assert (u["sigma_epistemic"] >= 0).all()
    # Total uncertainty must dominate its epistemic component.
    assert (u["sigma_total"] >= u["sigma_epistemic"] - 1e-9).all()
    # With genuine noise in the data, aleatoric must be picked up.
    assert ens.aleatoric_ > 0


def test_calibration_coverage_is_monotone():
    rng = np.random.default_rng(1)
    n = 5000
    mu = np.zeros(n)
    sigma = np.full(n, 0.3)
    fc = pd.DataFrame({
        "mu": mu,
        "sigma_total": sigma,
        "y_true": rng.normal(mu, sigma),  # perfectly specified -> good calibration
    })
    cal = calibration_report(fc, levels=(0.5, 0.8, 0.9)).set_index("nominal")
    # Higher nominal level -> wider interval -> more coverage.
    assert cal["empirical_coverage"].is_monotonic_increasing
    # Well-specified intervals should be close to nominal.
    assert abs(cal.loc[0.9, "empirical_coverage"] - 0.9) < 0.03
