"""BH-FDR correction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def bh_fdr(p_values: pd.Series | np.ndarray, alpha: float = 0.05) -> pd.DataFrame:
    p = pd.Series(p_values, dtype=float).dropna()
    m = len(p)
    if m == 0:
        return pd.DataFrame(columns=["p_value", "p_adjusted", "reject"])
    order = p.argsort()
    ranked = p.iloc[order]
    adj = ranked.values * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out = pd.DataFrame(
        {"p_value": ranked.values, "p_adjusted": adj, "reject": adj <= alpha},
        index=ranked.index,
    )
    return out.reindex(p.index)
