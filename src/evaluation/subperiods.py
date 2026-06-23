"""Couche 4 (validation) — does the result hold across market regimes?

Run:
    python -m src.evaluation.subperiods

Slices the already-computed net monthly returns (portfolio_returns.parquet) into
economic regimes and recomputes the headline metrics, to check that the central
claim — the uncertainty-aware robust strategy beats the Markowitz family — is not
an artefact of one lucky window. No new modelling; pure out-of-the-box validation.

Output:
    results/metrics/subperiod_metrics.csv
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config
from src.evaluation.backtest import performance_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("subperiods")

# (label, start, end, is_crisis). Crisis windows are short -> annualised Sharpe is
# noisy, so for those we lean on cumulative return and max drawdown instead.
REGIMES = [
    ("Full 2013-2026", None, None, False),
    ("Bull 2013-2019", "2013-01-01", "2019-12-31", False),
    ("Post-2020", "2020-01-01", None, False),
    ("COVID crash 2020", "2020-02-01", "2020-04-30", True),
    ("Rate shock 2022", "2022-01-01", "2022-12-31", True),
]


def _cumulative_return(r: pd.Series) -> float:
    return float((1 + r).prod() - 1)


def main() -> None:
    net = pd.read_parquet(config.PORTFOLIO_RETURNS_PATH)
    order = [config.HEADLINE_STRATEGY] + [s for s in net.columns if s != config.HEADLINE_STRATEGY]

    rows = []
    for label, start, end, is_crisis in REGIMES:
        window = net.loc[start:end]
        if len(window) < 2:
            continue
        for strategy in order:
            r = window[strategy]
            m = performance_metrics(r, config.PERIODS_PER_YEAR, config.RISK_FREE_ANNUAL)
            rows.append({
                "regime": label,
                "months": len(window),
                "strategy": strategy,
                "cum_return": _cumulative_return(r),
                "sharpe": m["sharpe"] if not is_crisis else np.nan,  # unreliable on short windows
                # within-window drawdown is meaningless on a handful of points
                "max_drawdown": m["max_drawdown"] if len(window) >= 4 else np.nan,
            })

    table = pd.DataFrame(rows)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.METRICS_DIR / "subperiod_metrics.csv"
    table.to_csv(out_path, index=False)
    logger.info("Saved subperiod metrics -> %s", out_path)

    # Headline check: robust vs the Markowitz family, regime by regime.
    logger.info("\n%s", table.round(4).to_string(index=False))
    logger.info("\n--- Does robust beat the Markowitz family per regime? ---")
    for label, _, _, is_crisis in REGIMES:
        sub = table[table["regime"] == label].set_index("strategy")
        if sub.empty:
            continue
        # Longer windows -> Sharpe; crisis windows -> cumulative return (the real
        # loss). For both, higher is better (less-negative return = smaller loss).
        metric = "cum_return" if is_crisis else "sharpe"
        rob = sub.loc[config.HEADLINE_STRATEGY, metric]
        mkt = sub.loc[["markowitz_lw", "markowitz_naive"], metric]
        better = rob > mkt.max()
        verdict = "robust LEADS" if better else "robust trails"
        logger.info("  %-18s [%s] robust %.3f vs Markowitz max %.3f -> %s",
                    label, metric, rob, mkt.max(), verdict)


if __name__ == "__main__":
    main()
