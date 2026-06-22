"""Tests for the features layer — focus on the no-leakage contract."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.build import (
    build_feature_table,
    compute_target,
    rebalance_dates,
)


def _synthetic(n_days: int = 600) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.bdate_range("2018-01-01", periods=n_days, name="date")
    rng = np.random.default_rng(0)
    # Two assets with mild upward drift.
    rets = pd.DataFrame(
        rng.normal(0.0004, 0.01, size=(n_days, 2)), index=idx, columns=["AAA", "BBB"]
    )
    prices = 100 * (1 + rets).cumprod()
    return prices, rets


def test_rebalance_dates_are_month_ends():
    prices, _ = _synthetic()
    rebal = rebalance_dates(prices.index)
    # One date per (year, month) present in the index.
    n_months = len({(d.year, d.month) for d in prices.index})
    assert len(rebal) == n_months
    assert rebal.is_monotonic_increasing


def test_target_matches_realised_forward_return():
    prices, _ = _synthetic()
    rebal = rebalance_dates(prices.index)
    target = compute_target(prices, rebal)
    # Target at rebal[i] must equal price ratio rebal[i] -> rebal[i+1].
    expected = prices.loc[rebal[1]] / prices.loc[rebal[0]] - 1.0
    pd.testing.assert_series_equal(
        target.iloc[0], expected, check_names=False
    )
    # Last rebalancing date has no future -> NaN target.
    assert target.iloc[-1].isna().all()


def test_feature_table_has_no_nans_and_drops_last_date():
    prices, rets = _synthetic()
    table = build_feature_table(prices, rets)
    assert not table.empty                      # guard against the all-NaN-column bug
    assert table.notna().all().all()            # warm-up + target-less rows dropped
    assert "fwd_ret" in table.columns
    # downside_vol_3m must actually survive sampling (regression test).
    assert table["downside_vol_3m"].notna().any()
    # The final (target-less) rebalancing date must not survive.
    rebal = rebalance_dates(prices.index)
    surviving_dates = table.index.get_level_values("date").unique()
    assert rebal[-1] not in surviving_dates
