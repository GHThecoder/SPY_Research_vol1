import numpy as np
import pandas as pd

from spy_alpha_research.pipelines import (
    fit_random_forest,
    fit_pca_ridge,
    fit_pca_kernel_ridge,
    fit_cnn,
)
from spy_alpha_research.features import build_feature_frame, assert_feature_target_alignment
from spy_alpha_research.data import make_target


def _toy_xy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    X = pd.DataFrame(rng.normal(size=(n, 5)), index=idx, columns=list("abcde"))
    y = pd.Series(X["a"] * 0.3 + rng.normal(scale=0.5, size=n), index=idx)
    return X.iloc[:140], y.iloc[:140], X.iloc[140:], y.iloc[140:]


def test_tree_eval_predictions_not_train_index():
    Xtr, ytr, Xev, yev = _toy_xy()
    res = fit_random_forest(
        Xtr, ytr, Xev, yev, n_estimators=30, max_depth=3, min_samples_leaf=5, n_repeats=3
    )
    assert res.eval_predictions.index.min() >= Xev.index.min()
    assert res.eval_predictions.index.max() <= Xev.index.max()
    assert not res.eval_predictions.index.equals(res.train_predictions.index)


def test_pca_pipeline_has_scaler_and_kernel_perm():
    Xtr, ytr, Xev, yev = _toy_xy()
    ridge = fit_pca_ridge(Xtr, ytr, Xev, yev, n_components=3, scale=True, min_periods=20)
    assert "scaler" in ridge.pipeline.named_steps
    assert ridge.eval_predictions.index.min() >= Xev.index.min()

    kr = fit_pca_kernel_ridge(
        Xtr, ytr, Xev, yev, n_components=3, scale=True, min_periods=20, n_repeats=3
    )
    assert kr.feature_attribution.name == "kernel_perm_importance"
    assert set(kr.feature_attribution.index) == set(Xtr.columns)


def test_cnn_saliency_on_eval():
    Xtr, ytr, Xev, yev = _toy_xy(n=180)
    res = fit_cnn(
        Xtr.iloc[:, :3],
        ytr,
        Xev.iloc[:, :3],
        yev,
        lookback=16,
        conv_channels=[4, 4],
        epochs=2,
        batch_size=32,
        seed=0,
    )
    assert res.eval_predictions.index.min() >= Xev.index.min()
    assert set(res.channel_saliency.index) == set(Xtr.columns[:3])


def test_feature_timestamp_contract(spy_like):
    ohlcv, rets = spy_like
    feats = build_feature_frame(ohlcv, rets)
    target = make_target(rets, 1)
    assert feats.attrs["feature_asof"] == "close_t"
    assert_feature_target_alignment(feats, target, returns=rets)
