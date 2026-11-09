"""SPY timing backtest engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TimingResult:
    positions: pd.Series
    traded_positions: pd.Series
    gross_returns: pd.Series
    net_returns: pd.Series
    costs: pd.Series
    turnover: pd.Series
    metrics: dict[str, float]


def signal_to_position(signal: pd.Series, method: str = "tanh") -> pd.Series:
    s = signal.astype(float)
    if method == "tanh":
        return np.tanh(s).rename("position")
    if method == "sign":
        return np.sign(s).fillna(0.0).rename("position")
    raise ValueError("position method must be 'tanh' or 'sign'")


def to_simple_returns(returns: pd.Series, return_type: str = "simple") -> pd.Series:
    """Convert asset returns to simple returns for linear PnL / cumprod wealth."""
    r = returns.astype(float)
    if return_type == "simple":
        return r
    if return_type == "log":
        return (np.exp(r) - 1.0).rename(getattr(returns, "name", None) or "ret_simple")
    raise ValueError("return_type must be 'simple' or 'log'")


def max_drawdown(returns: pd.Series) -> float:
    """`returns` must be simple period returns."""
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    dd = wealth / wealth.cummax() - 1.0
    return float(dd.min()) if len(dd) else np.nan


def summarize(returns: pd.Series, turnover: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
    """`returns` must be simple period returns (already converted if inputs were log)."""
    r = returns.dropna()
    if len(r) == 0:
        return {
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "hit_rate": np.nan,
            "avg_turnover": np.nan,
        }
    ann_return = float((1.0 + r).prod() ** (periods_per_year / len(r)) - 1.0)
    ann_vol = float(r.std() * np.sqrt(periods_per_year))
    sharpe = float(r.mean() / r.std() * np.sqrt(periods_per_year)) if r.std() > 0 else np.nan
    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(r),
        "hit_rate": float((r > 0).mean()),
        "avg_turnover": float(turnover.dropna().mean()),
    }


def run_timing_backtest(
    signal: pd.Series,
    asset_returns: pd.Series,
    *,
    signal_lag: int = 1,
    position: str = "tanh",
    costs_bps: float = 1.0,
    periods_per_year: int = 252,
    return_type: str = "simple",
) -> TimingResult:
    """Long/short timing with lagged positions and costs on traded turnover.

    Parameters
    ----------
    return_type:
        ``simple`` or ``log``. Log asset returns are converted to simple before
        PnL and wealth compounding: ``simple = exp(log) - 1``.
    """
    sig, ret_raw = signal.align(asset_returns, join="inner")
    ret = to_simple_returns(ret_raw, return_type=return_type)
    pos = signal_to_position(sig, method=position)
    traded = pos.shift(signal_lag).fillna(0.0).rename("traded")
    gross = (traded * ret.fillna(0.0)).rename("gross")
    # Costs must land on the same day the lagged position changes (not target pos).
    turnover = traded.diff().abs()
    if len(turnover):
        turnover.iloc[0] = abs(float(traded.iloc[0]))
    turnover = turnover.fillna(0.0)
    costs = (turnover * (costs_bps / 1e4)).rename("cost")
    net = (gross - costs).rename("net")
    metrics = summarize(net, turnover, periods_per_year=periods_per_year)
    return TimingResult(
        positions=pos,
        traded_positions=traded,
        gross_returns=gross,
        net_returns=net,
        costs=costs,
        turnover=turnover.rename("turnover"),
        metrics=metrics,
    )
