"""Pre-registered hypothesis tests for selected alphas."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .bootstrap_ts import block_bootstrap_ic, moving_block_indices
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
    """H1: IC > 0 via centered moving-block bootstrap (primary)."""
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
        "method": boot.get("method", "centered_moving_block_bootstrap"),
    }


def test_combo_ic_positive(
    combo: pd.Series,
    forward_return: pd.Series,
    **kwargs,
) -> dict:
    out = test_mean_ic_positive(combo, forward_return, **kwargs)
    out["hypothesis"] = "H2_eqw_mean_ic_gt_0"
    return out


def _best_single_name(
    singles: dict[str, pd.Series],
    forward_return: pd.Series,
    index: pd.Index | None,
) -> str:
    def _score(name: str) -> float:
        sig = singles[name]
        if index is not None:
            ic = series_ic(sig.reindex(index), forward_return.reindex(index))
        else:
            ic = series_ic(sig, forward_return)
        return ic if np.isfinite(ic) else -np.inf

    return max(singles, key=_score)


def test_combo_vs_best(
    combo: pd.Series,
    singles: dict[str, pd.Series],
    forward_return: pd.Series,
    *,
    selection_index: pd.Index | None = None,
    n_bootstrap: int = 1000,
    block_size: int = 21,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """H3: combo IC >= best single IC on the evaluation window.

    Best single is frozen on ``selection_index`` IC. If omitted, argmax uses the
    evaluation window and is labeled ``eval_peek_exploratory`` (selection-biased).
    """
    if not singles:
        raise ValueError("singles must be non-empty")

    best_source = "selection" if selection_index is not None else "eval_peek_exploratory"
    best_name = _best_single_name(singles, forward_return, selection_index)
    best = singles[best_name]
    df = pd.concat({"c": combo, "b": best, "r": forward_return}, axis=1).dropna()
    if len(df) < 40:
        return {
            "hypothesis": "H3_combo_ge_best",
            "best_alpha": best_name,
            "best_source": best_source,
            "combo_ic": np.nan,
            "best_ic": np.nan,
            "diff": np.nan,
            "p_value": np.nan,
            "fail_to_reject_h3": False,
            "method": "paired_centered_block_bootstrap",
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
    diff = combo_ic - best_ic  # H3: diff >= 0; evidence against if diff < 0
    rng = np.random.default_rng(seed)
    n = len(df)
    block = max(1, min(block_size, n))
    diffs = []
    for _ in range(n_bootstrap):
        idx = moving_block_indices(n, block, rng)
        d = _ic(c[idx], r[idx]) - _ic(b[idx], r[idx])
        if np.isfinite(d):
            diffs.append(d)
    diffs_arr = np.asarray(diffs, dtype=float)

    if len(diffs_arr) == 0 or not np.isfinite(diff):
        p_worse = np.nan
    else:
        # Centered lower-tail p for H1': diff < 0 (combo worse than best)
        # p = P*(d* <= 2·diff)
        p_worse = float((1 + np.sum(diffs_arr <= (2.0 * diff))) / (len(diffs_arr) + 1))

    return {
        "hypothesis": "H3_combo_ge_best",
        "best_alpha": best_name,
        "best_source": best_source,
        "combo_ic": combo_ic,
        "best_ic": best_ic,
        "diff": diff,
        "p_value": p_worse,
        # Fail to reject H3 when little evidence that combo is worse
        "fail_to_reject_h3": bool(np.isfinite(p_worse) and p_worse > alpha),
        "method": "paired_centered_block_bootstrap",
    }
