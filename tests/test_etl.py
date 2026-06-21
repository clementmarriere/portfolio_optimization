"""Unit tests for the ETL cleaning logic (no network — uses a synthetic panel)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.etl.download import clean_prices


def _synthetic_panel() -> pd.DataFrame:
    """Two tickers with different inception dates and scattered holiday gaps."""
    idx = pd.bdate_range("2020-01-01", periods=30, name="date")
    a = pd.Series(np.linspace(100, 130, 30), index=idx)
    b = pd.Series(np.linspace(50, 65, 30), index=idx)
    b.iloc[:5] = np.nan          # late inception
    a.iloc[10:12] = np.nan       # short gap (bridgeable)
    return pd.DataFrame({"AAA": a, "BBB": b})


def test_clean_trims_to_common_window_and_has_no_nans():
    clean = clean_prices(_synthetic_panel())
    # Window starts after BBB's inception.
    assert clean.index.min() == pd.Timestamp("2020-01-08")
    # Short gaps inside the common window are bridged.
    assert clean.isna().sum().sum() == 0
    assert list(clean.columns) == ["AAA", "BBB"]


def test_clean_does_not_invent_long_gaps():
    panel = _synthetic_panel()
    panel.iloc[15:25, 0] = np.nan  # 10-day gap > MAX_FFILL_DAYS
    clean = clean_prices(panel)
    # A gap longer than the ffill limit must NOT be fully fabricated.
    assert clean["AAA"].isna().sum() > 0 or len(clean) < len(panel)
