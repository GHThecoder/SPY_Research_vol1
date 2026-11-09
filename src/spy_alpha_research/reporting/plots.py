"""Plot helpers writing under reports/figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spy_alpha_research.data.config import resolve_path


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_acf_pacf(acf_v, pacf_v, out_path: str | Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].bar(range(len(acf_v)), acf_v)
    axes[0].set_title("ACF")
    axes[1].bar(range(len(pacf_v)), pacf_v)
    axes[1].set_title("PACF")
    fig.suptitle("SPY return ACF / PACF")
    return _save(fig, resolve_path(out_path))


def plot_importance(importance: pd.Series, out_path: str | Path, title: str = "Feature importance") -> Path:
    s = importance.abs().sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    s.plot(kind="barh", ax=ax)
    ax.set_title(title)
    return _save(fig, resolve_path(out_path))


def plot_equity_curves(curves: dict[str, pd.Series], out_path: str | Path, title: str = "Equity") -> Path:
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, r in curves.items():
        (1.0 + r.fillna(0.0)).cumprod().plot(ax=ax, label=name)
    ax.legend(fontsize=8)
    ax.set_title(title)
    return _save(fig, resolve_path(out_path))


def plot_ic_bars(ic_df: pd.DataFrame, out_path: str | Path) -> Path:
    """IC point estimates with bootstrap CI error bars (primary inference)."""
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(ic_df))
    ic = ic_df["ic"].to_numpy(dtype=float)
    ax.bar(x, ic, color="C0", alpha=0.85)
    if {"ci_low", "ci_high"}.issubset(ic_df.columns):
        yerr = np.vstack(
            [
                ic - ic_df["ci_low"].to_numpy(dtype=float),
                ic_df["ci_high"].to_numpy(dtype=float) - ic,
            ]
        )
        ax.errorbar(x, ic, yerr=np.abs(yerr), fmt="none", ecolor="black", capsize=3, lw=1)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(list(ic_df.index), rotation=45, ha="right")
    ax.set_title("IC by alpha (block-bootstrap 95% CI)")
    ax.set_ylabel("IC")
    return _save(fig, resolve_path(out_path))


def plot_comparison_table_bars(metrics: pd.DataFrame, col: str, out_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4))
    metrics[col].plot(kind="bar", ax=ax)
    ax.set_title(f"Comparison: {col}")
    plt.xticks(rotation=45, ha="right")
    return _save(fig, resolve_path(out_path))


def plot_rank_stability(agreement: pd.DataFrame, out_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(agreement.to_numpy(dtype=float), cmap="viridis", vmin=-1, vmax=1)
    ax.set_xticks(range(len(agreement.columns)))
    ax.set_yticks(range(len(agreement.index)))
    ax.set_xticklabels(list(agreement.columns), rotation=45, ha="right")
    ax.set_yticklabels(list(agreement.index))
    ax.set_title("Rank Spearman agreement")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, resolve_path(out_path))


def plot_ablation(ablation: pd.DataFrame, out_path: str | Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ablation["delta_ic"].plot(kind="barh", ax=axes[0], color="C0")
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_title("ΔIC vs full eqw")
    ablation["delta_sharpe"].plot(kind="barh", ax=axes[1], color="C1")
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_title("ΔSharpe vs full eqw")
    fig.suptitle("Grouped feature ablation (eval window)")
    plt.tight_layout()
    return _save(fig, resolve_path(out_path))
