# Notebooks — exploration only

Notebooks are for **exploration and visual sanity checks only**. No business
logic lives here: anything reusable belongs in `src/` and is called from a
notebook, never the other way around.

Suggested entries:
- `01_data_exploration.ipynb` — coverage, correlations, drawdown of each asset class
- `02_forecast_diagnostics.ipynb` — residuals, calibration of uncertainty bands
- `03_allocation_inspection.ipynb` — weights over time, turnover, regime behaviour
