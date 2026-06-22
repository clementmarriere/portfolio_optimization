"""Couche 2 — forecast uncertainty via bootstrap ensembling of the headline model.

Run:
    python -m src.models.uncertainty

Idea
----
Refit the headline forecaster on B bootstrap resamples of the training set. For
each asset at each rebalancing date:
  - mu              = ensemble mean prediction (the point forecast)
  - sigma_epistemic = std across the B models -> how much the *model* disagrees
                      with itself = reliability of mu. THIS is what the robust
                      optimiser uses to down-weight unreliable forecasts.
  - sigma_aleatoric = irreducible noise, estimated from out-of-bag residuals
  - sigma_total     = sqrt(epistemic^2 + aleatoric^2), used only to sanity-check
                      calibration (does a nominal X% interval cover X% of cases?)

Limitation (documented on purpose): aleatoric noise is estimated as a single
pooled scalar per training step, so sigma_total is homoscedastic across assets.
Epistemic uncertainty, the part the allocator actually consumes, is per-asset.

Outputs:
    data/processed/forecasts_uncertainty.parquet
    results/metrics/uncertainty_calibration.csv
    results/figures/uncertainty_calibration.png
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm

from src import config
from src.models import baselines, ml  # noqa: F401  (populate the registry)
from src.models.base import Forecaster, get_forecaster
from src.models.walkforward import TARGET

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("uncertainty")


class BootstrapEnsembleForecaster:
    """Wrap any base forecaster into a bootstrap ensemble with uncertainty."""

    def __init__(
        self,
        base_factory: Callable[[], Forecaster],
        n_boot: int = config.N_BOOTSTRAP,
        n_jobs: int = -1,
        random_state: int = 0,
    ) -> None:
        self.base_factory = base_factory
        self.n_boot = n_boot
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BootstrapEnsembleForecaster":
        n = len(X)
        seeds = np.random.default_rng(self.random_state).integers(0, 2**31 - 1, self.n_boot)

        def _fit_one(seed: int):
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, n, n)                       # bootstrap sample
            model = self.base_factory().fit(X.iloc[idx], y.iloc[idx])
            oob = np.setdiff1d(np.arange(n), np.unique(idx))  # ~37% held out
            resid = (
                y.iloc[oob].to_numpy() - model.predict(X.iloc[oob])
                if len(oob) else np.empty(0)
            )
            return model, resid

        results = Parallel(n_jobs=self.n_jobs)(delayed(_fit_one)(int(s)) for s in seeds)
        self.models_ = [m for m, _ in results]
        oob_resid = np.concatenate([r for _, r in results if len(r)])
        self.aleatoric_ = float(oob_resid.std()) if oob_resid.size else 0.0
        return self

    def predict_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([m.predict(X) for m in self.models_])  # (n, B)

    def predict_uncertainty(self, X: pd.DataFrame) -> pd.DataFrame:
        ens = self.predict_ensemble(X)
        mu = ens.mean(axis=1)
        epi = ens.std(axis=1, ddof=1)
        total = np.sqrt(epi**2 + self.aleatoric_**2)
        return pd.DataFrame({
            "mu": mu,
            "sigma_epistemic": epi,
            "sigma_aleatoric": self.aleatoric_,
            "sigma_total": total,
        })


def walk_forward_uncertainty(
    table: pd.DataFrame, feature_cols: list[str], min_train_months: int, n_boot: int,
) -> pd.DataFrame:
    """Expanding-window walk-forward producing mu + uncertainty per (date, ticker)."""
    df = table.reset_index()
    dates = np.sort(df["date"].unique())
    test_dates = dates[min_train_months:]
    factory = lambda: get_forecaster(config.HEADLINE_MODEL)  # noqa: E731

    out = []
    for i, t in enumerate(test_dates, 1):
        train = df[df["date"] < t]
        test = df[df["date"] == t]
        ens = BootstrapEnsembleForecaster(factory, n_boot=n_boot).fit(
            train[feature_cols], train[TARGET]
        )
        u = ens.predict_uncertainty(test[feature_cols])
        u.insert(0, "ticker", test["ticker"].to_numpy())
        u.insert(0, "date", test["date"].to_numpy())
        u["y_true"] = test[TARGET].to_numpy()
        out.append(u)
        if i % 25 == 0 or i == len(test_dates):
            logger.info("  %d/%d dates", i, len(test_dates))

    logger.info("Walk-forward done: %s -> %s",
                pd.Timestamp(test_dates[0]).date(), pd.Timestamp(test_dates[-1]).date())
    return pd.concat(out, ignore_index=True)


def calibration_report(fc: pd.DataFrame, levels) -> pd.DataFrame:
    """Empirical coverage of nominal central intervals built from sigma_total."""
    rows = []
    for lvl in levels:
        z = norm.ppf(0.5 + lvl / 2)
        lo = fc["mu"] - z * fc["sigma_total"]
        hi = fc["mu"] + z * fc["sigma_total"]
        rows.append({
            "nominal": lvl,
            "empirical_coverage": float(((fc["y_true"] >= lo) & (fc["y_true"] <= hi)).mean()),
            "mean_interval_width": float((2 * z * fc["sigma_total"]).mean()),
        })
    return pd.DataFrame(rows)


def _plot_calibration(cal: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="parfaitement calibré")
    ax.plot(cal["nominal"], cal["empirical_coverage"], "o-", label="observé")
    ax.set_xlabel("Couverture nominale")
    ax.set_ylabel("Couverture empirique")
    ax.set_title("Calibration des intervalles (sigma_total)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    table = pd.read_parquet(config.FEATURES_PATH)
    feature_cols = [c for c in table.columns if c != TARGET]
    logger.info("Bootstrap ensemble of '%s' (B=%d) over %d features",
                config.HEADLINE_MODEL, config.N_BOOTSTRAP, len(feature_cols))

    fc = walk_forward_uncertainty(table, feature_cols, config.TRAIN_MIN_MONTHS, config.N_BOOTSTRAP)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fc.to_parquet(config.FORECASTS_UNCERTAINTY_PATH)
    logger.info("Saved uncertainty forecasts -> %s", config.FORECASTS_UNCERTAINTY_PATH)

    cal = calibration_report(fc, config.CALIBRATION_LEVELS)
    cal.to_csv(config.CALIBRATION_PATH, index=False)
    _plot_calibration(cal, config.CALIBRATION_FIG_PATH)
    logger.info("Saved calibration -> %s (+ figure)", config.CALIBRATION_PATH)

    logger.info("\n%s", cal.round(4).to_string(index=False))
    logger.info(
        "Median epistemic sigma: %.4f | aleatoric (last step): %.4f",
        fc["sigma_epistemic"].median(), fc["sigma_aleatoric"].iloc[-1],
    )


if __name__ == "__main__":
    main()
