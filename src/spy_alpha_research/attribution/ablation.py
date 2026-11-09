"""Grouped feature ablation for equal-weight alpha blends."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from spy_alpha_research.alpha import equal_weight_blend, oriented_signal
from spy_alpha_research.backtesting import run_timing_backtest
from spy_alpha_research.statistics import series_ic


FEATURE_GROUPS = {
    "momentum": re.compile(r"^mom_"),
    "ret_lags": re.compile(r"^ret_lag_"),
    "vol": re.compile(r"^(vol_|vov_)"),
    "volume": re.compile(r"^volume_"),
    "calendar": re.compile(r"^(dow|month)$"),
    "range": re.compile(r"^hl_"),
}


def group_members(feature_names: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {g: [] for g in FEATURE_GROUPS}
    out["other"] = []
    for name in feature_names:
        matched = False
        for g, pat in FEATURE_GROUPS.items():
            if pat.search(name):
                out[g].append(name)
                matched = True
                break
        if not matched:
            out["other"].append(name)
    return {k: v for k, v in out.items() if v}


def grouped_ablation(
    features: pd.DataFrame,
    alpha_names: list[str],
    forward_return: pd.Series,
    asset_returns: pd.Series,
    train_index: pd.Index,
    eval_index: pd.Index,
    *,
    min_periods: int = 63,
    signal_lag: int = 1,
    position: str = "tanh",
    costs_bps: float = 1.0,
) -> pd.DataFrame:
    """Drop each feature group from the shortlist; report ΔIC and ΔSharpe on eval."""
    names = [n for n in alpha_names if n in features.columns]
    groups = group_members(names)

    def _metrics(use_names: list[str]) -> dict[str, float]:
        if not use_names:
            return {"ic": np.nan, "sharpe": np.nan, "n_alphas": 0.0}
        sigs = {
            n: oriented_signal(
                features[n],
                forward_return,
                train_index=train_index,
                min_periods=min_periods,
            )
            for n in use_names
        }
        blend = equal_weight_blend(sigs).reindex(eval_index)
        ic = series_ic(blend, forward_return.reindex(eval_index))
        bt = run_timing_backtest(
            blend,
            asset_returns.reindex(eval_index),
            signal_lag=signal_lag,
            position=position,
            costs_bps=costs_bps,
        )
        return {"ic": ic, "sharpe": bt.metrics["sharpe"], "n_alphas": float(len(use_names))}

    full = _metrics(names)
    rows = [{"ablation": "full", "dropped": "", **full, "delta_ic": 0.0, "delta_sharpe": 0.0}]
    for g, members in groups.items():
        remain = [n for n in names if n not in members]
        m = _metrics(remain)
        rows.append(
            {
                "ablation": f"drop_{g}",
                "dropped": ",".join(members),
                **m,
                "delta_ic": m["ic"] - full["ic"] if np.isfinite(m["ic"]) and np.isfinite(full["ic"]) else np.nan,
                "delta_sharpe": (
                    m["sharpe"] - full["sharpe"]
                    if np.isfinite(m["sharpe"]) and np.isfinite(full["sharpe"])
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).set_index("ablation")


def cnn_channel_ablation_delta(
    channel_losses: dict[str, float],
    full_loss: float,
) -> pd.Series:
    """ΔOOS loss when leaving one CNN channel out (higher => channel mattered)."""
    return pd.Series(
        {k: v - full_loss for k, v in channel_losses.items()},
        name="delta_oos_loss",
    ).sort_values(ascending=False)
