import numpy as np
import pandas as pd

from spy_alpha_research.features import build_feature_frame
from spy_alpha_research.data import make_target, split_by_dates
from spy_alpha_research.validation import make_walk_forward_folds


def test_feature_target_lag_discipline(spy_like):
    ohlcv, rets = spy_like
    feats = build_feature_frame(ohlcv, rets)
    target = make_target(rets, horizon=1)
    lag = feats["ret_lag_1"].dropna()
    expected = rets.shift(1).reindex(lag.index)
    assert (lag - expected).abs().max() < 1e-12
    both = pd.concat([feats["ret_lag_1"], target], axis=1).dropna()
    assert not np.allclose(both.iloc[:, 0], both.iloc[:, 1])


def test_date_split_and_cap():
    idx = pd.bdate_range("2016-01-01", "2024-12-31")
    train, test = split_by_dates(idx, "2021-12-31", "2022-01-01", "2024-12-31")
    assert train.max() <= pd.Timestamp("2021-12-31")
    assert test.min() >= pd.Timestamp("2022-01-01")
    assert test.max() <= pd.Timestamp("2024-12-31")


def test_purged_cv_disjoint(spy_like):
    _, rets = spy_like
    folds = make_walk_forward_folds(rets.index, n_splits=5, purge_periods=5, embargo_periods=5)
    for fold in folds:
        assert set(fold.train_idx).isdisjoint(fold.test_idx)
        assert set(fold.train_idx).isdisjoint(fold.purge_idx)
