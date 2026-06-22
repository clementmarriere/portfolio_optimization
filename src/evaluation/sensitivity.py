"""Sensitivity of the robust strategy to the uncertainty penalty kappa.

Run:
    python -m src.evaluation.sensitivity

Re-solves the robust allocation across a grid of kappa values and backtests each,
to show the headline conclusion does not hinge on one hand-picked kappa. kappa=0
coincides with the markowitz_lw ablation (no uncertainty penalty), giving a clean
left anchor. Covariances are computed once and reused across the grid.

Outputs:
    results/metrics/sensitivity_kappa.csv
    results/figures/sensitivity_kappa.png
"""
from __future__ import annotations

import logging

import pandas as pd

from src import config
from src.evaluation.backtest import backtest_strategy, performance_metrics
from src.optimization import optimizers as opt
from src.optimization.covariance import ledoit_wolf_monthly

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sensitivity")


def _precompute(fc: pd.DataFrame, daily: pd.DataFrame) -> dict:
    """Per-date inputs (tickers, mu, sigma_epi, Ledoit-Wolf Sigma), computed once."""
    cache = {}
    for date, g in fc.groupby("date"):
        g = g.sort_values("ticker")
        tickers = g["ticker"].tolist()
        cache[date] = {
            "tickers": tickers,
            "mu": g["mu"].to_numpy(),
            "sigma_epi": g["sigma_epistemic"].to_numpy(),
            "cov_lw": ledoit_wolf_monthly(daily, date, config.COV_LOOKBACK_DAYS, tickers),
        }
    return cache


def main() -> None:
    fc = pd.read_parquet(config.FORECASTS_UNCERTAINTY_PATH)
    daily = pd.read_parquet(config.RETURNS_PATH)
    realised = fc.pivot(index="date", columns="ticker", values="y_true").sort_index()

    logger.info("Precomputing covariances for %d dates...", fc["date"].nunique())
    cache = _precompute(fc, daily)

    rows = []
    for kappa in config.KAPPA_GRID:
        weight_rows = []
        for date, c in cache.items():
            w = opt.robust_mean_variance(
                c["mu"], c["cov_lw"], c["sigma_epi"],
                config.RISK_AVERSION, kappa, config.MAX_WEIGHT,
            )
            for ticker, wi in zip(c["tickers"], w):
                weight_rows.append({"date": date, "ticker": ticker, "weight": wi})
        W = (pd.DataFrame(weight_rows)
             .pivot(index="date", columns="ticker", values="weight").sort_index())

        net, _, turn = backtest_strategy(W, realised, config.TRANSACTION_COST_BPS)
        m = performance_metrics(net, config.PERIODS_PER_YEAR, config.RISK_FREE_ANNUAL)
        m["kappa"] = kappa
        m["avg_turnover"] = float(turn.iloc[1:].mean())
        rows.append(m)
        logger.info("kappa=%.1f -> Sharpe %.3f | maxDD %.1f%%",
                    kappa, m["sharpe"], 100 * m["max_drawdown"])

    df = pd.DataFrame(rows).set_index("kappa")
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.SENSITIVITY_PATH)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df.index, df["sharpe"], "o-", color="tab:blue", label="Sharpe (net)")
    ax1.set_xlabel("kappa (pénalité d'incertitude)")
    ax1.set_ylabel("Sharpe", color="tab:blue")
    ax1.axvline(config.UNCERTAINTY_PENALTY, ls="--", color="grey", label="kappa retenu")
    ax2 = ax1.twinx()
    ax2.plot(df.index, 100 * df["max_drawdown"], "s--", color="tab:red", label="max DD %")
    ax2.set_ylabel("max drawdown (%)", color="tab:red")
    ax1.set_title("Sensibilité du robuste à kappa (kappa=0 = Markowitz-LW)")
    fig.tight_layout(); fig.savefig(config.SENSITIVITY_FIG_PATH, dpi=120); plt.close(fig)

    logger.info("Saved sensitivity -> %s (+ figure)", config.SENSITIVITY_PATH)
    logger.info("\n%s", df.round(4).to_string())


if __name__ == "__main__":
    main()
