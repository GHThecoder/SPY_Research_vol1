"""Timing backtest benchmarks: SPY B&H, cash, volatility-matched SPY."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import TimingResult, run_timing_backtest, summarize


def buy_and_hold_returns(asset_returns: pd.Series) -> pd.Series:
    return asset_returns.fillna(0.0).rename("spy_bh")


def cash_returns(index: pd.Index) -> pd.Series:
    return pd.Series(0.0, index=index, name="cash")


def vol_matched_spy_scale(
    strategy_returns: pd.Series,
    spy_returns: pd.Series,
    *,
    fit_index: pd.Index,
    periods_per_year: int = 252,
) -> float:
    """Scale factor estimated on fit_index only so final holdout is not used."""
    s = strategy_returns.reindex(fit_index).dropna()
    b = spy_returns.reindex(fit_index).dropna()
    both = pd.concat({"s": s, "b": b}, axis=1).dropna()
    if len(both) < 20:
        return 1.0
    s_vol = both["s"].std() * np.sqrt(periods_per_year)
    b_vol = both["b"].std() * np.sqrt(periods_per_year)
    if not np.isfinite(b_vol) or b_vol == 0:
        return 1.0
    return float(s_vol / b_vol)


def vol_matched_spy_returns(
    strategy_returns: pd.Series,
    spy_returns: pd.Series,
    *,
    fit_index: pd.Index,
    eval_index: pd.Index,
    periods_per_year: int = 252,
) -> pd.Series:
    scale = vol_matched_spy_scale(
        strategy_returns, spy_returns, fit_index=fit_index, periods_per_year=periods_per_year
    )
    return (spy_returns.reindex(eval_index).fillna(0.0) * scale).rename("spy_vol_matched")


def benchmark_bundle(
    strategy_net: pd.Series,
    spy_returns: pd.Series,
    *,
    selection_index: pd.Index,
    eval_index: pd.Index,
    periods_per_year: int = 252,
) -> dict[str, pd.Series]:
    """Return named benchmark return series aligned to eval_index."""
    return {
        "spy_bh": buy_and_hold_returns(spy_returns.reindex(eval_index)),
        "cash": cash_returns(eval_index),
        "spy_vol_matched": vol_matched_spy_returns(
            strategy_net,
            spy_returns,
            fit_index=selection_index,
            eval_index=eval_index,
            periods_per_year=periods_per_year,
        ),
    }


def summarize_benchmarks(
    benchmarks: dict[str, pd.Series],
    *,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    rows = []
    empty_to = pd.Series(dtype=float)
    for name, r in benchmarks.items():
        m = summarize(r, empty_to, periods_per_year=periods_per_year)
        rows.append({"benchmark": name, **m})
    return pd.DataFrame(rows).set_index("benchmark")
