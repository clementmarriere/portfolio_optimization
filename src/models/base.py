"""Common forecaster interface + registry.

Every model — headline or annex — implements the same `Forecaster` API so the
walk-forward engine and the optimiser never care which one is plugged in.
A forecaster maps a feature matrix (one row per asset at a rebalancing date) to
an expected forward return per asset.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import pandas as pd


class Forecaster(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Forecaster":
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...


# Registry: name -> zero-arg factory. Populated by the model modules.
_REGISTRY: dict[str, Callable[[], Forecaster]] = {}


def register(name: str, factory: Callable[[], Forecaster]) -> None:
    _REGISTRY[name] = factory


def get_forecaster(name: str) -> Forecaster:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def available_models() -> list[str]:
    return sorted(_REGISTRY)
