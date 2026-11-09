"""Return and target construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def price_series(ohlcv: pd.DataFrame, price_col: str = "Adj Close") -> pd.Series:
    if price_col not in ohlcv.columns:
        if price_col == "Adj Close" and "Close" in ohlcv.columns:
            price_col = "Close"
        else:
            raise KeyError(f"Missing price column {price_col}")
    return ohlcv[price_col].astype(float).rename("price")


def compute_returns(prices: pd.Series, method: str = "log") -> pd.Series:
    if method == "simple":
        return prices.pct_change().rename("ret")
    if method == "log":
        return np.log(prices).diff().rename("ret")
    raise ValueError("method must be 'log' or 'simple'")


def make_target(returns: pd.Series, horizon: int = 1) -> pd.Series:
    """Forward return used as regression label (no leakage into features)."""
    return returns.shift(-horizon).rename(f"target_h{horizon}")


def split_by_dates(
    index: pd.DatetimeIndex,
    train_end: str,
    test_start: str,
    end: str,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Legacy two-way split (train / combined post-train). Prefer split_three_way."""
    train_end_ts = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)
    end_ts = pd.Timestamp(end)
    train = index[(index <= train_end_ts)]
    test = index[(index >= test_start_ts) & (index <= end_ts)]
    return train, test


def split_three_way(
    index: pd.DatetimeIndex,
    *,
    train_end: str,
    selection_start: str,
    selection_end: str,
    final_start: str,
    end: str,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
    """Train / selection / final holdout with no overlap."""
    train_end_ts = pd.Timestamp(train_end)
    sel_start = pd.Timestamp(selection_start)
    sel_end = pd.Timestamp(selection_end)
    fin_start = pd.Timestamp(final_start)
    end_ts = pd.Timestamp(end)

    train = index[index <= train_end_ts]
    selection = index[(index >= sel_start) & (index <= sel_end)]
    final = index[(index >= fin_start) & (index <= end_ts)]

    if len(train) == 0 or len(selection) == 0 or len(final) == 0:
        raise ValueError("One of train/selection/final windows is empty")
    if train.max() >= selection.min() or selection.max() >= final.min():
        raise ValueError("Overlapping or mis-ordered three-way windows")
    return train, selection, final
