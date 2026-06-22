"""Tests for the backtest metrics & turnover accounting."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.backtest import backtest_strategy, performance_metrics


def test_performance_metrics_known_values():
    # Constant +1%/month for 12 months.
    r = pd.Series([0.01] * 12)
    m = performance_metrics(r, ppy=12, rf_annual=0.0)
    assert abs(m["ann_return"] - (1.01**12 - 1)) < 1e-9
    assert m["ann_vol"] < 1e-9        # constant series -> ~zero vol (float-safe)
    assert abs(m["max_drawdown"]) < 1e-12   # monotonically rising -> no drawdown


def test_max_drawdown_sign_and_magnitude():
    # +10%, then -50%, then flat.
    r = pd.Series([0.10, -0.50, 0.0])
    m = performance_metrics(r, ppy=12, rf_annual=0.0)
    assert m["max_drawdown"] < 0
    assert abs(m["max_drawdown"] - (-0.50)) < 1e-9


def test_turnover_zero_when_weights_track_drift():
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    tickers = ["A", "B"]
    realised = pd.DataFrame(0.0, index=dates, columns=tickers)  # zero returns
    W = pd.DataFrame([[0.5, 0.5]] * 3, index=dates, columns=tickers)
    net, gross, turn = backtest_strategy(W, realised, cost_bps=10.0)
    # With zero returns and unchanged weights, post-initial turnover is zero.
    assert turn.iloc[0] == 1.0          # initial purchase
    assert turn.iloc[1:].abs().max() < 1e-12
    assert (net == gross).all() | (net.iloc[1:] == gross.iloc[1:]).all()
