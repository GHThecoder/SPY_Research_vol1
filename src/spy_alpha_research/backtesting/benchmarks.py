"""Timing backtest benchmarks: SPY B&H, cash, volatility-matched SPY."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import summarize, to_simple_returns


def buy_and_hold_returns(
    asset_returns: pd.Series,
    *,
    return_type: str = "simple",
) -> pd.Series:
    simple = to_simple_returns(asset_returns, return_type=return_type)
    return simple.fillna(0.0).rename("spy_bh")


def cash_returns(index: pd.Index) -> pd.Series:
    return pd.Series(0.0, index=index, name="cash")


def vol_matched_spy_scale(
    strategy_returns_selection: pd.Series,
    spy_returns: pd.Series,
    *,
    selection_index: pd.Index,
    periods_per_year: int = 252,
    return_type: str = "simple",
    min_obs: int = 20,
) -> float:
    """Scale factor estimated on selection only — never on final holdout.

    Raises if strategy returns do not cover selection with enough overlap
    (avoids silently falling back to scale=1.0 / plain SPY).
    """
    spy_simple = to_simple_returns(spy_returns, return_type=return_type)
    s = strategy_returns_selection.reindex(selection_index).dropna()
    b = spy_simple.reindex(selection_index).dropna()
    both = pd.concat({"s": s, "b": b}, axis=1).dropna()
    if len(both) < min_obs:
        raise ValueError(
            "vol_matched scale requires strategy net returns on selection_index "
            f"(need>={min_obs} overlapping obs, got {len(both)}). "
            "Pass selection-window strategy returns, not final-only."
        )
    s_vol = both["s"].std() * np.sqrt(periods_per_year)
    b_vol = both["b"].std() * np.sqrt(periods_per_year)
    if not np.isfinite(b_vol) or b_vol == 0:
        raise ValueError("vol_matched scale: SPY selection vol is zero/non-finite")
    if not np.isfinite(s_vol):
        raise ValueError("vol_matched scale: strategy selection vol is non-finite")
    return float(s_vol / b_vol)


def vol_matched_spy_returns(
    strategy_returns_selection: pd.Series,
    spy_returns: pd.Series,
    *,
    selection_index: pd.Index,
    eval_index: pd.Index,
    periods_per_year: int = 252,
    return_type: str = "simple",
) -> pd.Series:
    scale = vol_matched_spy_scale(
        strategy_returns_selection,
        spy_returns,
        selection_index=selection_index,
        periods_per_year=periods_per_year,
        return_type=return_type,
    )
    spy_simple = to_simple_returns(spy_returns, return_type=return_type)
    return (spy_simple.reindex(eval_index).fillna(0.0) * scale).rename("spy_vol_matched")


def benchmark_bundle(
    strategy_net_selection: pd.Series,
    spy_returns: pd.Series,
    *,
    selection_index: pd.Index,
    eval_index: pd.Index,
    periods_per_year: int = 252,
    return_type: str = "simple",
) -> dict[str, pd.Series]:
    """Named benchmark simple-return series aligned to eval_index.

    ``strategy_net_selection`` must cover ``selection_index`` so vol-matching
    scale is estimated without peeking at the final window.
    """
    return {
        "spy_bh": buy_and_hold_returns(spy_returns.reindex(eval_index), return_type=return_type),
        "cash": cash_returns(eval_index),
        "spy_vol_matched": vol_matched_spy_returns(
            strategy_net_selection,
            spy_returns,
            selection_index=selection_index,
            eval_index=eval_index,
            periods_per_year=periods_per_year,
            return_type=return_type,
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
