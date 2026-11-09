"""Equal-weight alpha combination."""

from __future__ import annotations

import pandas as pd


def equal_weight_blend(signals: dict[str, pd.Series]) -> pd.Series:
    """Simple equal average of aligned signals."""
    if not signals:
        raise ValueError("No signals to blend")
    frame = pd.DataFrame(signals)
    return frame.mean(axis=1).rename("eqw_alpha")
