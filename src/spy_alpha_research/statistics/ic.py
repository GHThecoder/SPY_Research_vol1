"""Time-series information coefficient helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .bootstrap_ts import block_bootstrap_ic
from .inference import mean_with_nw, newey_west_tstat


def series_ic(signal: pd.Series, forward_return: pd.Series) -> float:
    s, r = signal.align(forward_return, join="inner")
    mask = s.notna() & r.notna()
    if mask.sum() < 5:
        return np.nan
    return float(s[mask].corr(r[mask]))


def rolling_ic(
    signal: pd.Series,
    forward_return: pd.Series,
    window: int = 63,
) -> pd.Series:
    """Overlapping rolling IC — for stability plots only, not primary inference."""
    s, r = signal.align(forward_return, join="inner")
    return s.rolling(window).corr(r).rename("rolling_ic")


def _nw_lags_auto(n: int, configured: int | str = "auto") -> int:
    if configured == "auto" or configured is None:
        return max(1, int(np.floor(1.5 * (n ** (1.0 / 3.0)))))
    return int(configured)


def demeaned_product_series(signal: pd.Series, forward_return: pd.Series) -> pd.Series:
    """Daily contribution proxy: (s-μs)*(r-μr); HAC on this series is secondary inference."""
    df = pd.concat({"s": signal, "r": forward_return}, axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float, name="ic_contrib")
    s = df["s"] - df["s"].mean()
    r = df["r"] - df["r"].mean()
    return (s * r).rename("ic_contrib")


def ic_inference(
    signal: pd.Series,
    forward_return: pd.Series,
    *,
    n_bootstrap: int = 1000,
    block_size: int = 21,
    seed: int = 42,
    nw_lags: int | str = "auto",
    rolling_window: int = 63,
) -> dict[str, float]:
    """Primary: block-bootstrap IC. Secondary: NW on demeaned product series.

    Rolling IC is returned for plotting only (mean of rolling series) — do not
    use rolling-IC NW as a significance claim.
    """
    boot = block_bootstrap_ic(
        signal,
        forward_return,
        n_bootstrap=n_bootstrap,
        block_size=block_size,
        seed=seed,
    )
    contrib = demeaned_product_series(signal, forward_return)
    lags = _nw_lags_auto(len(contrib), nw_lags)
    nw = mean_with_nw(contrib, lags=lags) if len(contrib) else {
        "mean": np.nan,
        "t_stat": np.nan,
        "p_value": np.nan,
    }
    ric = rolling_ic(signal, forward_return, window=rolling_window).dropna()
    return {
        "ic": boot["ic"],
        "ci_low": boot["ci_low"],
        "ci_high": boot["ci_high"],
        "p_value": boot["p_value_one_sided"],  # primary (bootstrap one-sided)
        "p_value_bootstrap": boot["p_value_one_sided"],
        "nw_t_stat": nw["t_stat"],
        "nw_p_value": nw["p_value"],
        "nw_lags": float(lags),
        "n_obs": float(boot["n_obs"]),
        "rolling_ic_mean": float(ric.mean()) if len(ric) else np.nan,
    }


# Back-compat name used by older notebooks/reporting
def ic_with_nw(
    signal: pd.Series,
    forward_return: pd.Series,
    *,
    window: int = 63,
    nw_lags: int = 5,
    n_bootstrap: int = 1000,
    block_size: int = 21,
    seed: int = 42,
) -> dict[str, float]:
    return ic_inference(
        signal,
        forward_return,
        n_bootstrap=n_bootstrap,
        block_size=block_size,
        seed=seed,
        nw_lags=nw_lags,
        rolling_window=window,
    )


def ic_table(
    signals: dict[str, pd.Series],
    forward_return: pd.Series,
    *,
    window: int = 63,
    nw_lags: int | str = "auto",
    n_bootstrap: int = 1000,
    block_size: int = 21,
    seed: int = 42,
) -> pd.DataFrame:
    rows = []
    for name, sig in signals.items():
        rows.append(
            {
                "alpha": name,
                **ic_inference(
                    sig,
                    forward_return,
                    n_bootstrap=n_bootstrap,
                    block_size=block_size,
                    seed=seed,
                    nw_lags=nw_lags,
                    rolling_window=window,
                ),
            }
        )
    return pd.DataFrame(rows).set_index("alpha")
