"""Test the cumulative-return helper used in the regime analysis."""
from __future__ import annotations

import pandas as pd

from src.evaluation.subperiods import _cumulative_return


def test_cumulative_return_compounds():
    r = pd.Series([0.10, -0.10])           # +10% then -10% -> 0.99
    assert abs(_cumulative_return(r) - (-0.01)) < 1e-12


def test_cumulative_return_zero_on_flat():
    assert abs(_cumulative_return(pd.Series([0.0, 0.0, 0.0]))) < 1e-12
