"""Newey–West inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def newey_west_variance(x: np.ndarray, lags: int = 5) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return np.nan
    x = x - x.mean()
    gamma0 = np.dot(x, x) / n
    var = gamma0
    max_lag = min(lags, n - 1)
    for lag in range(1, max_lag + 1):
        w = 1.0 - lag / (max_lag + 1)
        gamma = np.dot(x[lag:], x[:-lag]) / n
        var += 2.0 * w * gamma
    return float(var / n)


def newey_west_tstat(series: pd.Series, lags: int = 5) -> float:
    x = series.dropna().to_numpy(dtype=float)
    if len(x) < 2:
        return np.nan
    se = np.sqrt(newey_west_variance(x, lags=lags))
    if se == 0 or np.isnan(se):
        return np.nan
    return float(x.mean() / se)


def mean_with_nw(series: pd.Series, lags: int = 5) -> dict[str, float]:
    x = series.dropna()
    t = newey_west_tstat(x, lags=lags)
    df = max(len(x) - 1, 1)
    p = float(2 * stats.t.sf(abs(t), df=df)) if np.isfinite(t) else np.nan
    return {"mean": float(x.mean()) if len(x) else np.nan, "t_stat": t, "p_value": p}
