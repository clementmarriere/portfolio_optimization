"""Single source of truth for the project: paths, dates, and the investable universe.

Design note (univers hybride)
------------------------------
The narrative frames the problem in terms of *asset classes* (equities, bonds,
real estate, commodities, ...). Each asset class is implemented with the proxy
ETF that offers the longest clean history on yfinance so the backtest can reach
back to 2007 (and through the 2008 crisis). Some proxies are US-listed for data
depth; European exposure is kept explicit (EZU, EFA). Currency at the data layer
is USD; CHF reporting is a downstream concern (see TODO in README).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

# Canonical artifacts produced by the ETL layer.
RAW_PRICES_PATH = RAW_DIR / "prices_raw.parquet"
PRICES_PATH = PROCESSED_DIR / "prices.parquet"
RETURNS_PATH = PROCESSED_DIR / "returns.parquet"
COVERAGE_PATH = PROCESSED_DIR / "coverage.csv"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"
FORECASTS_PATH = PROCESSED_DIR / "forecasts.parquet"
FORECAST_METRICS_PATH = METRICS_DIR / "forecast_metrics.csv"

# --------------------------------------------------------------------------- #
# Backtest window
# --------------------------------------------------------------------------- #
# Start in 2007 to capture the 2008 crisis (the strongest drawdown stress test).
START_DATE = "2007-01-01"
END_DATE = None  # None -> today, resolved at download time.

# Trading-currency of the price data. Conversion to CHF for reporting is a
# downstream step and intentionally NOT done here.
BASE_CURRENCY = "USD"

# --------------------------------------------------------------------------- #
# Forecast / rebalancing horizon
# --------------------------------------------------------------------------- #
# Monthly: forecast & rebalance at each month-end. Targets are non-overlapping
# (month-end to month-end), which keeps the train/test setup clean.
REBALANCE_FREQ = "ME"        # pandas month-end alias
HORIZON_DAYS = 21            # informational: ~trading days per month
MIN_HISTORY_DAYS = 252       # longest rolling window -> drop earlier rows

# --------------------------------------------------------------------------- #
# Forecasting layer (couche 1)
# --------------------------------------------------------------------------- #
# Headline model shown in the main pipeline; others stay available as annex.
HEADLINE_MODEL = "gbm"
# Months of history required before the first out-of-sample forecast
# (expanding-window walk-forward warm-up).
TRAIN_MIN_MONTHS = 60
# Baselines reported alongside the headline (the "annex" comparison).
ANNEX_MODELS = ["ridge", "moving_average", "historical_mean"]


# --------------------------------------------------------------------------- #
# Investable universe
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Asset:
    ticker: str
    name: str
    asset_class: str
    region: str


# ~17 instruments spanning equities, fixed income, real estate, commodities and
# cash. Chosen for diversification AND for inception dates compatible with a
# 2007 backtest start.
UNIVERSE: list[Asset] = [
    # --- Equities --------------------------------------------------------- #
    Asset("SPY", "S&P 500", "equity", "US"),
    Asset("IWM", "US Small Cap (Russell 2000)", "equity", "US"),
    Asset("EZU", "Eurozone Equities (MSCI EMU)", "equity", "Europe"),
    Asset("EFA", "Developed ex-US (MSCI EAFE)", "equity", "Developed-exUS"),
    Asset("EEM", "Emerging Markets (MSCI EM)", "equity", "Emerging"),
    # --- Fixed income ----------------------------------------------------- #
    Asset("AGG", "US Aggregate Bond", "bond_govt_ig", "US"),
    Asset("IEF", "US Treasury 7-10y", "bond_govt", "US"),
    Asset("TLT", "US Treasury 20y+", "bond_govt_long", "US"),
    Asset("LQD", "US Investment Grade Corp", "bond_corp_ig", "US"),
    Asset("HYG", "US High Yield Corp", "bond_corp_hy", "US"),
    Asset("TIP", "US Inflation-Linked (TIPS)", "bond_inflation", "US"),
    # --- Real estate ------------------------------------------------------ #
    Asset("VNQ", "US REITs", "real_estate", "US"),
    Asset("RWX", "International REITs ex-US", "real_estate", "Global-exUS"),
    # --- Commodities / gold ---------------------------------------------- #
    Asset("GLD", "Gold", "gold", "Global"),
    Asset("DBC", "Broad Commodities", "commodity", "Global"),
    # --- Cash / short duration ------------------------------------------- #
    Asset("SHY", "US Treasury 1-3y (cash proxy)", "cash", "US"),
]

TICKERS: list[str] = [a.ticker for a in UNIVERSE]


def asset_metadata() -> dict[str, Asset]:
    """Ticker -> Asset mapping, handy for joining metadata onto results."""
    return {a.ticker: a for a in UNIVERSE}
