"""Time-series block bootstrap helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def moving_block_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample overlapping moving blocks with replacement until length n."""
    if n <= 0:
        return np.array([], dtype=int)
    block_size = max(1, min(block_size, n))
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n - block_size + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
    return idx


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def block_bootstrap_ic(
    signal: pd.Series,
    forward_return: pd.Series,
    *,
    n_bootstrap: int = 1000,
    block_size: int = 21,
    seed: int = 42,
) -> dict[str, float]:
    """Moving-block bootstrap CI + centered one-sided p for H1: IC > 0.

    Percentile CI uses raw bootstrap replicates ``IC*``.

    One-sided p-value uses the **recentered** (Hall–Wilson style) null:
    under H0: IC = 0, compare the observed ``IC`` to the centered replicates
    ``IC* - IC``. Equivalently ``p = P*(IC* >= 2·IC)`` (with +1 smoothing).
    This is not the naive ``P*(IC* <= 0)`` fraction of the percentile cloud.
    """
    df = pd.concat({"s": signal, "r": forward_return}, axis=1).dropna()
    if len(df) < max(block_size + 5, 20):
        return {
            "ic": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value_one_sided": np.nan,
            "n_obs": int(len(df)),
            "method": "centered_moving_block_bootstrap",
        }
    s = df["s"].to_numpy(dtype=float)
    r = df["r"].to_numpy(dtype=float)
    n = len(s)
    point = _pearson(s, r)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = moving_block_indices(n, block_size, rng)
        samples[i] = _pearson(s[idx], r[idx])
    samples = samples[np.isfinite(samples)]
    if not np.isfinite(point) or len(samples) == 0:
        p_one = np.nan
    else:
        # Centered test of H0: IC=0 vs H1: IC>0
        # p = P*(IC*_c >= IC) with IC*_c = IC* - IC  =>  P*(IC* >= 2·IC)
        p_one = float((1 + np.sum(samples >= (2.0 * point))) / (len(samples) + 1))
    return {
        "ic": point,
        "ci_low": float(np.percentile(samples, 2.5)) if len(samples) else np.nan,
        "ci_high": float(np.percentile(samples, 97.5)) if len(samples) else np.nan,
        "p_value_one_sided": p_one,
        "n_obs": n,
        "method": "centered_moving_block_bootstrap",
    }
