"""Cross-pipeline importance consensus and alpha shortlist."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from spy_alpha_research.data.config import resolve_path


def rank_features(importance: pd.Series) -> pd.Series:
    """Higher importance → rank 1."""
    return importance.abs().rank(ascending=False, method="min")


def borda_consensus(rank_tables: dict[str, pd.Series], top_k: int = 8) -> pd.DataFrame:
    """Borda count across pipelines (lower rank better)."""
    ranks = pd.DataFrame(rank_tables)
    # Missing features get worst rank
    worst = ranks.max().max() + 1
    ranks = ranks.fillna(worst)
    scores = ranks.sum(axis=1).sort_values()
    out = pd.DataFrame({"borda_score": scores, "mean_rank": ranks.mean(axis=1)})
    out["selected"] = False
    out.iloc[:top_k, out.columns.get_loc("selected")] = True
    return out


def intersect_top_k(rank_tables: dict[str, pd.Series], top_k: int = 8) -> list[str]:
    sets = []
    for s in rank_tables.values():
        sets.append(set(s.sort_values().head(top_k).index))
    if not sets:
        return []
    inter = set.intersection(*sets)
    # If empty, fall back to borda top_k
    if not inter:
        return list(borda_consensus(rank_tables, top_k=top_k).head(top_k).index)
    return sorted(inter)


def select_alpha_candidates(
    importance_map: dict[str, pd.Series],
    *,
    top_k: int = 8,
    method: str = "borda",
) -> list[str]:
    ranks = {k: rank_features(v) for k, v in importance_map.items()}
    if method == "intersect":
        return intersect_top_k(ranks, top_k=top_k)
    table = borda_consensus(ranks, top_k=top_k)
    return list(table.loc[table["selected"]].index)


def save_alpha_candidates(
    names: list[str],
    path: str | Path,
    *,
    meta: dict | None = None,
) -> Path:
    out = resolve_path(path) if not Path(path).is_absolute() else Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"alphas": names, "meta": meta or {}}
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return out


def load_alpha_candidates(path: str | Path) -> list[str]:
    p = resolve_path(path) if not Path(path).is_absolute() else Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return list(data.get("alphas", []))


def importance_stability(fold_importances: list[pd.Series]) -> pd.Series:
    """Mean Spearman correlation of ranks across fold pairs."""
    if len(fold_importances) < 2:
        return pd.Series(dtype=float)
    ranks = [rank_features(s) for s in fold_importances]
    mat = pd.DataFrame(ranks).T
    corr = mat.corr(method="spearman")
    # average off-diagonal
    vals = corr.values
    n = len(vals)
    mask = ~np.eye(n, dtype=bool)
    return pd.Series({"mean_spearman": float(vals[mask].mean())})
