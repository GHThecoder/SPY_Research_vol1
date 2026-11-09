"""Purged expanding time-series CV."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    train_idx: np.ndarray
    test_idx: np.ndarray
    purge_idx: np.ndarray
    embargo_idx: np.ndarray


def make_walk_forward_folds(
    index: pd.DatetimeIndex | pd.Index,
    *,
    n_splits: int = 5,
    purge_periods: int = 5,
    embargo_periods: int = 5,
    min_train_frac: float = 0.4,
) -> list[WalkForwardFold]:
    """Expanding walk-forward folds with purge and embargo gaps."""
    n = len(index)
    min_train = max(int(n * min_train_frac), 50)
    test_len = max((n - min_train) // n_splits, 20)
    if min_train + test_len > n:
        raise ValueError("Index too short for requested CV settings")

    folds: list[WalkForwardFold] = []
    starts = np.linspace(min_train, n - test_len, num=n_splits, dtype=int)
    starts = np.unique(starts)
    for test_start in starts:
        test_end = min(test_start + test_len, n)
        purge_start = max(test_start - purge_periods, 0)
        train_idx = np.arange(0, purge_start)
        purge_idx = np.arange(purge_start, test_start)
        test_idx = np.arange(test_start, test_end)
        embargo_end = min(test_end + embargo_periods, n)
        embargo_idx = np.arange(test_end, embargo_end)
        forbidden = set(purge_idx.tolist()) | set(test_idx.tolist()) | set(embargo_idx.tolist())
        train_idx = np.array(sorted(set(train_idx.tolist()) - forbidden), dtype=int)
        folds.append(
            WalkForwardFold(
                train_idx=train_idx,
                test_idx=test_idx,
                purge_idx=purge_idx,
                embargo_idx=embargo_idx,
            )
        )
    return folds


class PurgedTimeSeriesSplit:
    def __init__(
        self,
        n_splits: int = 5,
        purge_periods: int = 5,
        embargo_periods: int = 5,
    ) -> None:
        self.n_splits = n_splits
        self.purge_periods = purge_periods
        self.embargo_periods = embargo_periods
        self.folds_: list[WalkForwardFold] | None = None

    def split(self, X: pd.DataFrame | pd.Series | pd.Index):
        index = X.index if hasattr(X, "index") else pd.Index(X)
        self.folds_ = make_walk_forward_folds(
            index,
            n_splits=self.n_splits,
            purge_periods=self.purge_periods,
            embargo_periods=self.embargo_periods,
        )
        for fold in self.folds_:
            yield fold.train_idx, fold.test_idx
