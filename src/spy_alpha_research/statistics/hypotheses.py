"""Pre-registered hypothesis tests for selected alphas."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .bootstrap_ts import block_bootstrap_ic
from .ic import series_ic


def test_mean_ic_positive(
    signal: pd.Series,
    forward_return: pd.Series,
    *,
    n_bootstrap: int = 1000,
    block_size: int = 21,
    seed: int = 42,
    alpha: float = 0.05,
    **_ignored,
) -> dict:
    """H1: IC > 0 via moving-block bootstrap (primary)."""
    boot = block_bootstrap_ic(
        signal,
        forward_return,
        n_bootstrap=n_bootstrap,
        block_size=block_size,
        seed=seed,
    )
    p = boot["p_value_one_sided"]
    return {
        "hypothesis": "H1_mean_ic_gt_0",
        "ic": boot["ic"],
        "ci_low": boot["ci_low"],
        "ci_high": boot["ci_high"],
        "p_value_one_sided": p,
        "reject": bool(np.isfinite(p) and p < alpha),
        "method": "moving_block_bootstrap",
    }


def test_combo_ic_positive(
    combo: pd.Series,
    forward_return: pd.Series,
    **kwargs,
) -> dict:
    out = test_mean_ic_positive(combo, forward_return, **kwargs)
    out["hypothesis"] = "H2_eqw_mean_ic_gt_0"
    return out


def test_combo_vs_best(
    combo: pd.Series,
    singles: dict[str, pd.Series],
    forward_return: pd.Series,
    *,
    n_bootstrap: int = 1000,
    block_size: int = 21,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """H3: combo IC >= best single IC (paired block bootstrap on IC difference)."""
    best_name = max(singles, key=lambda k: series_ic(singles[k], forward_return) or -np.inf)
    best = singles[best_name]
    df = pd.concat({"c": combo, "b": best, "r": forward_return}, axis=1).dropna()
    if len(df) < 40:
        return {
            "hypothesis": "H3_combo_ge_best",
            "best_alpha": best_name,
            "combo_ic": np.nan,
            "best_ic": np.nan,
            "diff": np.nan,
            "p_value": np.nan,
            "fail_to_reject_h3": False,
            "method": "paired_block_bootstrap",
        }

    def _ic(a, b_):
        if np.std(a) == 0 or np.std(b_) == 0:
            return np.nan
        return float(np.corrcoef(a, b_)[0, 1])

    c = df["c"].to_numpy(dtype=float)
    b = df["b"].to_numpy(dtype=float)
    r = df["r"].to_numpy(dtype=float)
    combo_ic = _ic(c, r)
    best_ic = _ic(b, r)
    diff = combo_ic - best_ic
    rng = np.random.default_rng(seed)
    n = len(df)
    block = max(1, min(block_size, n))
    diffs = []
    from .bootstrap_ts import moving_block_indices

    for _ in range(n_bootstrap):
        idx = moving_block_indices(n, block, rng)
        d = _ic(c[idx], r[idx]) - _ic(b[idx], r[idx])
        if np.isfinite(d):
            diffs.append(d)
    # Under H3 combo >= best; evidence against if bootstrap diffs are mostly negative
    p_worse = float(np.mean(np.asarray(diffs) < 0)) if diffs else np.nan
    return {
        "hypothesis": "H3_combo_ge_best",
        "best_alpha": best_name,
        "combo_ic": combo_ic,
        "best_ic": best_ic,
        "diff": diff,
        "p_value": p_worse,
        "fail_to_reject_h3": bool(np.isfinite(p_worse) and p_worse <= (1 - alpha)),
        "method": "paired_block_bootstrap",
    }
