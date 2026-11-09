"""Cross-model and fold-wise ranking stability."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .consensus import importance_stability, rank_features


def pairwise_rank_spearman(importance_a: pd.Series, importance_b: pd.Series) -> float:
    a = rank_features(importance_a)
    b = rank_features(importance_b)
    both = pd.concat({"a": a, "b": b}, axis=1).dropna()
    if len(both) < 3:
        return np.nan
    return float(both["a"].corr(both["b"], method="spearman"))


def ranking_agreement_table(importances: dict[str, pd.Series]) -> pd.DataFrame:
    names = list(importances.keys())
    mat = pd.DataFrame(index=names, columns=names, dtype=float)
    for i, na in enumerate(names):
        for j, nb in enumerate(names):
            if i == j:
                mat.loc[na, nb] = 1.0
            elif i < j:
                v = pairwise_rank_spearman(importances[na], importances[nb])
                mat.loc[na, nb] = v
                mat.loc[nb, na] = v
    return mat


def fold_importance_stability(fold_importances: list[pd.Series]) -> pd.Series:
    return importance_stability(fold_importances)
