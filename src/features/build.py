"""Features layer — turn the clean price panel into a model-ready feature table.

Run:
    python -m src.features.build

Output:
    data/processed/features.parquet
        Tidy table indexed by (date, ticker). Columns = predictive features
        computed from history up to `date`, plus the target `fwd_ret` = the
        simple return from `date` to the next month-end rebalancing date.

No-leakage contract
-------------------
- Every feature at row `date` uses prices/returns observed at or before `date`
  (rolling windows are causal, no centering, no forward fill into the future).
- The target is the ONLY forward-looking column; it is the realised return the
  optimiser will be judged against.
- Rows are sampled at month-end rebalancing dates only, so successive targets do
  not overlap (no autocorrelation inflation in train/test).
- The final rebalancing date has no "next" date -> its target is NaN and the row
  is dropped.
"""
from __future__ import annotations

import logging

import pandas as pd

from src import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("features")


def rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last available trading day of each calendar month."""
    by_month = pd.Series(index, index=index.to_period("M"))
    return pd.DatetimeIndex(by_month.groupby(level=0).last().values)


def compute_features(prices: pd.DataFrame, returns: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Causal, per-asset features as a dict of wide (date x ticker) frames."""
    feats: dict[str, pd.DataFrame] = {}

    # --- Momentum / trend (cumulative returns over lookback windows) ------- #
    feats["ret_1m"] = prices.pct_change(21)
    feats["ret_3m"] = prices.pct_change(63)
    feats["ret_6m"] = prices.pct_change(126)
    feats["ret_12m"] = prices.pct_change(252)
    # Classic 12-1 momentum: 12-month return excluding the most recent month.
    feats["mom_12_1"] = prices.shift(21) / prices.shift(252) - 1.0

    # --- Volatility / risk state (monthly-scaled) -------------------------- #
    feats["vol_1m"] = returns.rolling(21).std() * (21 ** 0.5)
    feats["vol_3m"] = returns.rolling(63).std() * (21 ** 0.5)
    # Only negative returns contribute; require >= 10 of them in the window,
    # otherwise rolling(63) would demand 63 non-NaN values and never trigger.
    feats["downside_vol_3m"] = (
        returns.where(returns < 0).rolling(63, min_periods=10).std() * (21 ** 0.5)
    )
    feats["mean_ret_1m"] = returns.rolling(21).mean()

    # --- Position relative to recent levels -------------------------------- #
    feats["ma_gap_200"] = prices / prices.rolling(200).mean() - 1.0
    feats["dd_from_high_1y"] = prices / prices.rolling(252).max() - 1.0

    return feats


def compute_target(prices: pd.DataFrame, rebal: pd.DatetimeIndex) -> pd.DataFrame:
    """Forward simple return from each rebalancing date to the next one."""
    px_rebal = prices.loc[rebal]
    fwd = px_rebal.shift(-1) / px_rebal - 1.0  # last row -> NaN (no next date)
    return fwd


def build_feature_table(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Assemble the tidy (date, ticker) feature+target table at rebalance dates."""
    rebal = rebalance_dates(prices.index)
    logger.info("%d rebalancing dates (%s -> %s)", len(rebal), rebal[0].date(), rebal[-1].date())

    feats = compute_features(prices, returns)
    target = compute_target(prices, rebal)

    # Sample every feature on the rebalancing grid, then stack to tidy form.
    wide = {name: df.loc[rebal] for name, df in feats.items()}
    wide["fwd_ret"] = target

    panel = pd.concat(wide, axis=1)                # cols: (feature, ticker)
    panel.columns.names = ["feature", "ticker"]
    tidy = panel.stack(level="ticker", future_stack=True)
    tidy.index.names = ["date", "ticker"]

    # Drop the warm-up period (incomplete 252d windows) and the final, target-less
    # rebalancing date. A row needs all features AND a target to be usable.
    before = len(tidy)
    tidy = tidy.dropna(how="any")
    logger.info("Feature table: %d rows (dropped %d incomplete)", len(tidy), before - len(tidy))
    if tidy.empty:
        raise RuntimeError(
            "Feature table is empty after dropna — a feature column is likely "
            "all-NaN at rebalancing dates. Inspect per-feature coverage."
        )
    return tidy.sort_index()


def main() -> None:
    prices = pd.read_parquet(config.PRICES_PATH)
    returns = pd.read_parquet(config.RETURNS_PATH)

    table = build_feature_table(prices, returns)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(config.FEATURES_PATH)
    logger.info("Saved features -> %s", config.FEATURES_PATH)
    logger.info(
        "Coverage: %s -> %s | %d assets | %d feature cols",
        table.index.get_level_values("date").min().date(),
        table.index.get_level_values("date").max().date(),
        table.index.get_level_values("ticker").nunique(),
        table.shape[1] - 1,
    )


if __name__ == "__main__":
    main()
