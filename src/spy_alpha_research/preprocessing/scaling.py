"""Causal time-series scalers (no full-sample StandardScaler)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ExpandingScaler(BaseEstimator, TransformerMixin):
    """Expanding mean/std using only information up to each time t.

    ``fit`` stores the training history. ``transform`` concatenates that history
    so OOS rows continue the expanding window causally.
    """

    def __init__(self, min_periods: int = 63) -> None:
        self.min_periods = min_periods
        self.mean_: pd.Series | None = None
        self.std_: pd.Series | None = None
        self.columns_: list[str] | None = None
        self._history: pd.DataFrame | None = None

    def fit(self, X: pd.DataFrame, y: object | None = None) -> "ExpandingScaler":
        X = X.sort_index()
        self.columns_ = list(X.columns)
        self._history = X.copy()
        exp_mean = X.expanding(min_periods=self.min_periods).mean()
        exp_std = X.expanding(min_periods=self.min_periods).std().replace(0.0, np.nan)
        self.mean_ = exp_mean.iloc[-1]
        self.std_ = exp_std.iloc[-1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.columns_ is None or self._history is None:
            raise RuntimeError("Scaler not fitted")
        X = X.reindex(columns=self.columns_).sort_index()
        hist = self._history
        prior = hist.loc[hist.index < X.index.min()] if len(X) else hist.iloc[0:0]
        combined = pd.concat([prior, X]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        mu = combined.expanding(min_periods=self.min_periods).mean()
        sd = combined.expanding(min_periods=self.min_periods).std().replace(0.0, np.nan)
        return ((combined - mu) / sd).reindex(X.index)

    def fit_transform(self, X: pd.DataFrame, y: object | None = None, **fit_params) -> pd.DataFrame:
        self.fit(X, y)
        mu = X.expanding(min_periods=self.min_periods).mean()
        sd = X.expanding(min_periods=self.min_periods).std().replace(0.0, np.nan)
        return (X - mu) / sd


class RollingScaler(BaseEstimator, TransformerMixin):
    """Rolling mean/std with a warm-start history buffer for OOS transforms.

    ``fit`` keeps the last ``window`` training rows. ``transform`` prepends rows
    strictly before the transform frame so early OOS dates have a full causal
    rolling window from train history.
    """

    def __init__(self, window: int = 252, min_periods: int | None = None) -> None:
        self.window = window
        self.min_periods = min_periods or max(window // 4, 1)
        self.mean_: pd.Series | None = None
        self.std_: pd.Series | None = None
        self.columns_: list[str] | None = None
        self._history: pd.DataFrame | None = None

    def fit(self, X: pd.DataFrame, y: object | None = None) -> "RollingScaler":
        X = X.sort_index()
        self.columns_ = list(X.columns)
        self._history = X.iloc[-self.window :].copy()
        roll_mean = X.rolling(self.window, min_periods=self.min_periods).mean()
        roll_std = X.rolling(self.window, min_periods=self.min_periods).std().replace(0.0, np.nan)
        self.mean_ = roll_mean.iloc[-1]
        self.std_ = roll_std.iloc[-1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.columns_ is None or self._history is None:
            raise RuntimeError("Scaler not fitted")
        X = X.reindex(columns=self.columns_).sort_index()
        hist = self._history
        prior = hist.loc[hist.index < X.index.min()] if len(X) else hist.iloc[0:0]
        combined = pd.concat([prior, X]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        mu = combined.rolling(self.window, min_periods=self.min_periods).mean()
        sd = combined.rolling(self.window, min_periods=self.min_periods).std().replace(0.0, np.nan)
        return ((combined - mu) / sd).reindex(X.index)

    def fit_transform(self, X: pd.DataFrame, y: object | None = None, **fit_params) -> pd.DataFrame:
        self.fit(X, y)
        mu = X.rolling(self.window, min_periods=self.min_periods).mean()
        sd = X.rolling(self.window, min_periods=self.min_periods).std().replace(0.0, np.nan)
        return (X - mu) / sd


def make_scaler(method: str = "expanding", min_periods: int = 63, rolling_window: int = 252):
    if method == "expanding":
        return ExpandingScaler(min_periods=min_periods)
    if method == "rolling":
        return RollingScaler(window=rolling_window, min_periods=min_periods)
    raise ValueError("method must be 'expanding' or 'rolling'")
