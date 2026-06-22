"""Couche 4 — backtest: turn weights into a decision with measured impact.

Run:
    python -m src.evaluation.backtest

The payoff of the whole project. For each strategy we compute the realised,
net-of-cost monthly return stream and the risk-adjusted metrics, then compare
the headline (robust, uncertainty-aware) against the two naive benchmarks.

Mechanics
---------
- Period return at rebalance date t = sum_i w_{t,i} * realised_{t,i}, where
  realised is the month-end -> next-month-end return already stored as `y_true`.
- Turnover is drift-aware: last period's weights drift with realised returns
  before being compared to the new target weights. Cost = turnover * bps.

Outputs:
    data/processed/portfolio_returns.parquet   net monthly returns per strategy
    results/metrics/backtest_metrics.csv        Sharpe / max DD / turnover / ...
    results/figures/equity_curves.png, drawdowns.png
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest")


def backtest_strategy(W: pd.DataFrame, realised: pd.DataFrame, cost_bps: float):
    """Return (net_returns, gross_returns, turnover) series for one strategy.

    W and realised are date x ticker frames on the same grid.
    """
    W = W.reindex_like(realised).fillna(0.0)
    gross = (W * realised).sum(axis=1)

    turnover = pd.Series(0.0, index=W.index)
    prev_w = None
    prev_date = None
    for t in W.index:
        w = W.loc[t]
        if prev_w is None:
            turnover.loc[t] = w.abs().sum()  # initial purchase from cash
        else:
            r = realised.loc[prev_date]
            drifted = prev_w * (1 + r)
            drifted = drifted / drifted.sum()
            turnover.loc[t] = (w - drifted).abs().sum()
        prev_w, prev_date = w, t

    cost = turnover * (cost_bps / 1e4)
    net = gross - cost
    return net, gross, turnover


def performance_metrics(returns: pd.Series, ppy: int, rf_annual: float) -> dict:
    equity = (1 + returns).cumprod()
    years = len(returns) / ppy
    ann_return = equity.iloc[-1] ** (1 / years) - 1
    ann_vol = returns.std(ddof=1) * np.sqrt(ppy)
    excess = returns.mean() * ppy - rf_annual
    sharpe = excess / ann_vol if ann_vol > 0 else np.nan
    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()
    calmar = ann_return / abs(max_dd) if max_dd < 0 else np.nan
    return {
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
    }


def _plot(equity: pd.DataFrame, drawdown: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [config.HEADLINE_STRATEGY] + [s for s in equity.columns if s != config.HEADLINE_STRATEGY]

    fig, ax = plt.subplots(figsize=(9, 5))
    for s in order:
        ax.plot(equity.index, equity[s], label=s, lw=2 if s == config.HEADLINE_STRATEGY else 1.2)
    ax.set_title("Courbes d'équité (net de coûts)")
    ax.set_ylabel("Croissance de 1$"); ax.set_yscale("log"); ax.legend()
    fig.tight_layout(); fig.savefig(config.EQUITY_FIG_PATH, dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    for s in order:
        ax.fill_between(drawdown.index, drawdown[s], 0, alpha=0.3 if s == config.HEADLINE_STRATEGY else 0.15)
        ax.plot(drawdown.index, drawdown[s], label=s, lw=1.5 if s == config.HEADLINE_STRATEGY else 1.0)
    ax.set_title("Drawdowns"); ax.set_ylabel("Drawdown"); ax.legend()
    fig.tight_layout(); fig.savefig(config.DRAWDOWN_FIG_PATH, dpi=120); plt.close(fig)


def main() -> None:
    weights = pd.read_parquet(config.WEIGHTS_PATH)
    fc = pd.read_parquet(config.FORECASTS_UNCERTAINTY_PATH)
    realised = fc.pivot(index="date", columns="ticker", values="y_true").sort_index()

    net_returns, gross_returns, turnovers, metrics = {}, {}, {}, {}
    for strategy, g in weights.groupby("strategy"):
        W = g.pivot(index="date", columns="ticker", values="weight").sort_index()
        net, gross, turn = backtest_strategy(W, realised, config.TRANSACTION_COST_BPS)
        net_returns[strategy] = net
        gross_returns[strategy] = gross
        turnovers[strategy] = turn
        m = performance_metrics(net, config.PERIODS_PER_YEAR, config.RISK_FREE_ANNUAL)
        m["sharpe_gross"] = performance_metrics(gross, config.PERIODS_PER_YEAR, config.RISK_FREE_ANNUAL)["sharpe"]
        m["avg_turnover"] = float(turn.iloc[1:].mean())  # excl. initial purchase
        metrics[strategy] = m

    net_df = pd.DataFrame(net_returns)
    equity = (1 + net_df).cumprod()
    drawdown = equity / equity.cummax() - 1

    order = [config.HEADLINE_STRATEGY] + [s for s in config.STRATEGIES if s != config.HEADLINE_STRATEGY]
    metrics_df = pd.DataFrame(metrics).T.loc[order]

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    net_df.to_parquet(config.PORTFOLIO_RETURNS_PATH)
    metrics_df.to_csv(config.BACKTEST_METRICS_PATH)
    _plot(equity, drawdown)

    logger.info("Backtest window: %s -> %s (%d months)",
                net_df.index.min().date(), net_df.index.max().date(), len(net_df))
    logger.info("Saved metrics -> %s (+ equity/drawdown figures)", config.BACKTEST_METRICS_PATH)
    logger.info("\n%s", metrics_df.round(4).to_string())

    # The verdicts the README is built around.
    h = config.HEADLINE_STRATEGY
    for b in ("markowitz_lw", "markowitz_naive", "equal_weight"):
        if b in metrics_df.index:
            logger.info(
                "VERDICT  %-15s vs %-15s: Sharpe %.2f vs %.2f | maxDD %.1f%% vs %.1f%%",
                h, b, metrics_df.loc[h, "sharpe"], metrics_df.loc[b, "sharpe"],
                100 * metrics_df.loc[h, "max_drawdown"], 100 * metrics_df.loc[b, "max_drawdown"],
            )
    logger.info("(robust vs markowitz_lw = pure effect of the uncertainty layer)")


if __name__ == "__main__":
    main()
