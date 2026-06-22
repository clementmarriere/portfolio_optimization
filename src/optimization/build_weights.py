"""Couche 3 — build allocation weights at every rebalancing date.

Run:
    python -m src.optimization.build_weights

For each rebalancing date it assembles the optimiser inputs (all causal):
    mu          : ensemble-mean forecast               (couche 1/2)
    sigma_epi   : per-asset epistemic uncertainty       (couche 2)
    Sigma       : monthly covariance from trailing daily returns up to the date
                  (Ledoit-Wolf for the headline, sample cov for naive Markowitz)
and solves the three strategies.

Output:
    data/processed/weights.parquet   tidy (date, ticker, strategy, weight)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config
from src.optimization import optimizers as opt
from src.optimization.covariance import ledoit_wolf_monthly, sample_monthly

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("optimize")


def build_weights() -> pd.DataFrame:
    fc = pd.read_parquet(config.FORECASTS_UNCERTAINTY_PATH)
    daily = pd.read_parquet(config.RETURNS_PATH)

    rows = []
    n_failed = 0
    for date, g in fc.groupby("date"):
        g = g.sort_values("ticker")
        tickers = g["ticker"].tolist()
        mu = g["mu"].to_numpy()
        sigma_epi = g["sigma_epistemic"].to_numpy()
        n = len(tickers)

        try:
            cov_lw = ledoit_wolf_monthly(daily, date, config.COV_LOOKBACK_DAYS, tickers)
            cov_s = sample_monthly(daily, date, config.COV_LOOKBACK_DAYS, tickers)
            weights = {
                "robust": opt.robust_mean_variance(
                    mu, cov_lw, sigma_epi, config.RISK_AVERSION,
                    config.UNCERTAINTY_PENALTY, config.MAX_WEIGHT),
                # ablation: same Ledoit-Wolf Sigma as robust, but no uncertainty
                # penalty -> isolates the shrinkage effect from the thesis term.
                "markowitz_lw": opt.markowitz(
                    mu, cov_lw, config.RISK_AVERSION, config.MAX_WEIGHT),
                "markowitz_naive": opt.markowitz(
                    mu, cov_s, config.RISK_AVERSION, config.MAX_WEIGHT),
                "equal_weight": opt.equal_weight(n),
            }
        except Exception as exc:  # numerical failure -> fall back to 1/N, logged
            n_failed += 1
            logger.warning("Optimiser fell back to 1/N on %s: %s", pd.Timestamp(date).date(), exc)
            weights = {s: opt.equal_weight(n) for s in config.STRATEGIES}

        for strategy, w in weights.items():
            for ticker, wi in zip(tickers, w):
                rows.append({"date": date, "ticker": ticker, "strategy": strategy, "weight": wi})

    out = pd.DataFrame(rows)
    logger.info("Built weights for %d dates x %d strategies (%d fallbacks)",
                fc["date"].nunique(), len(config.STRATEGIES), n_failed)
    return out


def _summary(weights: pd.DataFrame) -> pd.DataFrame:
    """Concentration diagnostics per strategy (effective #assets, max weight)."""
    rows = []
    for strategy, g in weights.groupby("strategy"):
        per_date = g.groupby("date")["weight"]
        herfindahl = g.groupby("date")["weight"].apply(lambda w: (w**2).sum())
        rows.append({
            "strategy": strategy,
            "avg_max_weight": float(per_date.max().mean()),
            "avg_eff_n_assets": float((1.0 / herfindahl).mean()),
            "avg_n_held": float(g[g["weight"] > 1e-4].groupby(g["date"]).size().mean()),
        })
    order = [config.HEADLINE_STRATEGY] + [s for s in config.STRATEGIES if s != config.HEADLINE_STRATEGY]
    return pd.DataFrame(rows).set_index("strategy").loc[order]


def main() -> None:
    weights = build_weights()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    weights.to_parquet(config.WEIGHTS_PATH)
    logger.info("Saved weights -> %s", config.WEIGHTS_PATH)
    logger.info("\n%s", _summary(weights).round(3).to_string())


if __name__ == "__main__":
    main()
