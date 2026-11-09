"""Causal SPY feature factory.

Timestamp contract
------------------
Every row indexed by calendar date ``t`` contains information available at the
**close of day t** only:

* ``ret_lag_k`` = return from close(t-k-1)→close(t-k)  (strictly past)
* ``mom_w``     = log(Close_t) - log(Close_{t-w})       (known at close t)
* ``vol_w``     = realized vol of returns through t
* ``hl_range``, ``volume_z_*`` use day-t OHLCV (known at close t)
* calendar fields use date t

The regression / IC **target** must be the *next* period return ``r_{t+1}``
(``make_target(..., horizon=1)``). Features at ``t`` never include ``r_{t+1}``.

This is a close-to-close research convention: signals formed at close t are
evaluated against the subsequent day's return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Documented asof convention for all columns produced by build_feature_frame
FEATURE_ASOF = "close_t"
TARGET_ASOF = "close_t_to_close_t_plus_horizon"


def build_feature_frame(
    ohlcv: pd.DataFrame,
    returns: pd.Series,
    *,
    return_lags: list[int] | None = None,
    momentum_windows: list[int] | None = None,
    vol_windows: list[int] | None = None,
    vol_of_vol_window: int = 63,
    volume_windows: list[int] | None = None,
    include_calendar: bool = True,
) -> pd.DataFrame:
    """Build features known at close of date t (see module docstring)."""
    return_lags = return_lags or [1, 2, 3, 5, 10, 21]
    momentum_windows = momentum_windows or [10, 21, 63, 126]
    vol_windows = vol_windows or [10, 21, 63]
    volume_windows = volume_windows or [10, 21]

    feats: dict[str, pd.Series] = {}
    ret = returns.astype(float)

    for lag in return_lags:
        # r_{t-lag}: strictly lagged, no same-day return in the feature
        feats[f"ret_lag_{lag}"] = ret.shift(lag)

    if "Close" in ohlcv.columns:
        log_px = np.log(ohlcv["Close"].astype(float))
    else:
        log_px = np.log(ret.add(1.0).cumprod())

    for w in momentum_windows:
        # Uses Close_t — valid when predicting r_{t+1} after the close
        feats[f"mom_{w}"] = log_px.diff(w)

    for w in vol_windows:
        feats[f"vol_{w}"] = ret.rolling(w).std() * np.sqrt(252)

    base_vol = feats.get("vol_21", ret.rolling(21).std() * np.sqrt(252))
    feats[f"vov_{vol_of_vol_window}"] = base_vol.rolling(vol_of_vol_window).std()

    if "High" in ohlcv.columns and "Low" in ohlcv.columns and "Close" in ohlcv.columns:
        feats["hl_range"] = (ohlcv["High"] - ohlcv["Low"]) / ohlcv["Close"].replace(0, np.nan)

    if "Volume" in ohlcv.columns:
        volu = ohlcv["Volume"].astype(float)
        for w in volume_windows:
            mu = volu.rolling(w).mean()
            sd = volu.rolling(w).std().replace(0.0, np.nan)
            feats[f"volume_z_{w}"] = (volu - mu) / sd

    if include_calendar:
        feats["dow"] = pd.Series(ret.index.dayofweek, index=ret.index, dtype=float)
        feats["month"] = pd.Series(ret.index.month, index=ret.index, dtype=float)

    frame = pd.DataFrame(feats, index=ret.index)
    frame.attrs["feature_asof"] = FEATURE_ASOF
    frame.attrs["target_asof"] = TARGET_ASOF
    return frame


def assert_feature_target_alignment(
    features: pd.DataFrame,
    target: pd.Series,
    returns: pd.Series | None = None,
) -> None:
    """Raise if the timestamp contract is violated for common leakage patterns."""
    if features.attrs.get("feature_asof") not in (None, FEATURE_ASOF):
        raise ValueError(f"Unexpected feature_asof={features.attrs.get('feature_asof')}")

    # Target must be forward-looking relative to contemporaneous return
    if returns is not None:
        both = pd.concat({"tgt": target, "ret": returns}, axis=1).dropna()
        if len(both) > 10 and np.allclose(both["tgt"], both["ret"], equal_nan=True):
            raise ValueError(
                "Target equals contemporaneous return — use make_target(returns, horizon=1) "
                "so labels are r_{t+1}, not r_t."
            )

    if "ret_lag_1" in features.columns and returns is not None:
        lag = features["ret_lag_1"]
        expected = returns.shift(1)
        mask = lag.notna() & expected.notna()
        if mask.any() and not np.allclose(lag[mask], expected[mask]):
            raise ValueError("ret_lag_1 does not equal returns.shift(1) — lag contract broken")
