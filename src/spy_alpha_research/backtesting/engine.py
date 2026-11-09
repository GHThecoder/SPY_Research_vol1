"""SPY timing backtest engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TimingResult:
    positions: pd.Series
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


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    dd = wealth / wealth.cummax() - 1.0
    return float(dd.min()) if len(dd) else np.nan


def summarize(returns: pd.Series, turnover: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
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
) -> TimingResult:
    sig, ret = signal.align(asset_returns, join="inner")
    pos = signal_to_position(sig, method=position)
    traded = pos.shift(signal_lag).fillna(0.0)
    gross = (traded * ret.fillna(0.0)).rename("gross")
    turnover = pos.diff().abs().fillna(0.0)
    if len(turnover):
        turnover.iloc[0] = abs(float(pos.iloc[0]))
    costs = (turnover * (costs_bps / 1e4)).rename("cost")
    net = (gross - costs).rename("net")
    metrics = summarize(net, turnover, periods_per_year=periods_per_year)
    return TimingResult(
        positions=pos,
        gross_returns=gross,
        net_returns=net,
        costs=costs,
        turnover=turnover.rename("turnover"),
        metrics=metrics,
    )
