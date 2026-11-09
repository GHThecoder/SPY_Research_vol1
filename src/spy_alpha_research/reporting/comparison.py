"""Build comparison tables across alphas and model baselines."""

from __future__ import annotations

import pandas as pd

from spy_alpha_research.backtesting import run_timing_backtest
from spy_alpha_research.statistics import ic_with_nw


def compare_signals(
    signals: dict[str, pd.Series],
    asset_returns: pd.Series,
    forward_return: pd.Series,
    *,
    signal_lag: int = 1,
    position: str = "tanh",
    costs_bps: float = 1.0,
    periods_per_year: int = 252,
    ic_window: int = 63,
    nw_lags: int = 5,
    return_type: str = "simple",
) -> pd.DataFrame:
    rows = []
    for name, sig in signals.items():
        ic = ic_with_nw(sig, forward_return, window=ic_window, nw_lags=nw_lags)
        bt = run_timing_backtest(
            sig,
            asset_returns,
            signal_lag=signal_lag,
            position=position,
            costs_bps=costs_bps,
            periods_per_year=periods_per_year,
            return_type=return_type,
        )
        rows.append({"signal": name, **ic, **bt.metrics})
    return pd.DataFrame(rows).set_index("signal")
