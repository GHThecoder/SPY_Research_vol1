"""PCA + linear / kernel regression pipelines.

Causal ExpandingScaler is the first Pipeline step by default. Models fit on
train only; predictions and attribution use the evaluation window.
Kernel models use permutation importance on eval (not feature–prediction corr).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from spy_alpha_research.preprocessing import ExpandingScaler


@dataclass
class PCAModelResult:
    name: str
    pipeline: Any
    feature_names: list[str]
    train_predictions: pd.Series
    eval_predictions: pd.Series
    feature_attribution: pd.Series
    eval_index: pd.Index


def _xy(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    df = X.join(y.rename("y"), how="inner").dropna()
    return df.drop(columns=["y"]), df["y"]


def _linear_feature_attribution(pca: PCA, coef: np.ndarray, feature_names: list[str]) -> pd.Series:
    loadings = pca.components_.T
    attr = loadings @ np.asarray(coef).ravel()
    return pd.Series(attr, index=feature_names, name="pca_linear_attr").sort_values(
        key=np.abs, ascending=False
    )


def _make_pipe(steps_tail: list[tuple[str, Any]], *, scale: bool, min_periods: int) -> Pipeline:
    steps: list[tuple[str, Any]] = []
    if scale:
        # Expanding burn-in produces NaNs for the first min_periods-1 rows; zero-fill those only.
        steps.append(("scaler", ExpandingScaler(min_periods=min_periods)))
        steps.append(("imputer", SimpleImputer(strategy="constant", fill_value=0.0)))
    steps.extend(steps_tail)
    return Pipeline(steps)


def _predict(pipe: Pipeline, X: pd.DataFrame, name: str) -> pd.Series:
    Xc = X.dropna()
    if Xc.empty:
        return pd.Series(dtype=float, name=name)
    return pd.Series(pipe.predict(Xc), index=Xc.index, name=name)


def _perm_attr(
    pipe: Pipeline,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    *,
    n_repeats: int,
    random_state: int,
    name: str,
) -> pd.Series:
    Xc, yc = _xy(X_eval, y_eval)
    perm = permutation_importance(
        pipe, Xc, yc, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
    )
    return pd.Series(perm.importances_mean, index=Xc.columns, name=name).sort_values(
        ascending=False
    )


def fit_pca_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    *,
    n_components: int = 8,
    alpha: float = 1.0,
    scale: bool = True,
    min_periods: int = 63,
    n_repeats: int = 10,
    random_state: int = 0,
) -> PCAModelResult:
    Xc, yc = _xy(X_train, y_train)
    n_comp = min(n_components, Xc.shape[1], max(Xc.shape[0] - 1, 1))
    pipe = _make_pipe(
        [
            ("pca", PCA(n_components=n_comp, random_state=random_state)),
            ("model", Ridge(alpha=alpha)),
        ],
        scale=scale,
        min_periods=min_periods,
    )
    pipe.fit(Xc, yc)
    pca = pipe.named_steps["pca"]
    coef = pipe.named_steps["model"].coef_
    attr = _linear_feature_attribution(pca, coef, list(Xc.columns))
    # Also expose eval permutation as a robustness check column name via attrs
    Xe, ye = _xy(X_eval, y_eval)
    return PCAModelResult(
        name="pca_ridge",
        pipeline=pipe,
        feature_names=list(Xc.columns),
        train_predictions=_predict(pipe, Xc, "pca_ridge_train"),
        eval_predictions=_predict(pipe, Xe, "pca_ridge_eval"),
        feature_attribution=attr,
        eval_index=Xe.index,
    )


def fit_pca_elasticnet(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    *,
    n_components: int = 8,
    alpha: float = 0.001,
    l1_ratio: float = 0.5,
    scale: bool = True,
    min_periods: int = 63,
    random_state: int = 0,
) -> PCAModelResult:
    Xc, yc = _xy(X_train, y_train)
    n_comp = min(n_components, Xc.shape[1], max(Xc.shape[0] - 1, 1))
    pipe = _make_pipe(
        [
            ("pca", PCA(n_components=n_comp, random_state=random_state)),
            ("model", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)),
        ],
        scale=scale,
        min_periods=min_periods,
    )
    pipe.fit(Xc, yc)
    attr = _linear_feature_attribution(
        pipe.named_steps["pca"], pipe.named_steps["model"].coef_, list(Xc.columns)
    )
    Xe, ye = _xy(X_eval, y_eval)
    return PCAModelResult(
        name="pca_elasticnet",
        pipeline=pipe,
        feature_names=list(Xc.columns),
        train_predictions=_predict(pipe, Xc, "pca_enet_train"),
        eval_predictions=_predict(pipe, Xe, "pca_enet_eval"),
        feature_attribution=attr,
        eval_index=Xe.index,
    )


def fit_pca_kernel_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    *,
    n_components: int = 8,
    alpha: float = 1.0,
    gamma: float | None = None,
    scale: bool = True,
    min_periods: int = 63,
    n_repeats: int = 10,
    random_state: int = 0,
) -> PCAModelResult:
    Xc, yc = _xy(X_train, y_train)
    n_comp = min(n_components, Xc.shape[1], max(Xc.shape[0] - 1, 1))
    g = gamma if gamma is not None else 1.0 / n_comp
    pipe = _make_pipe(
        [
            ("pca", PCA(n_components=n_comp, random_state=random_state)),
            ("model", KernelRidge(alpha=alpha, kernel="rbf", gamma=g)),
        ],
        scale=scale,
        min_periods=min_periods,
    )
    pipe.fit(Xc, yc)
    Xe, ye = _xy(X_eval, y_eval)
    attr = _perm_attr(
        pipe, Xe, ye, n_repeats=n_repeats, random_state=random_state, name="kernel_perm_importance"
    )
    return PCAModelResult(
        name="pca_kernel_ridge",
        pipeline=pipe,
        feature_names=list(Xc.columns),
        train_predictions=_predict(pipe, Xc, "kernel_train"),
        eval_predictions=_predict(pipe, Xe, "kernel_eval"),
        feature_attribution=attr,
        eval_index=Xe.index,
    )


def fit_pca_svr(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    *,
    n_components: int = 8,
    C: float = 1.0,
    epsilon: float = 0.01,
    gamma: str | float = "scale",
    scale: bool = True,
    min_periods: int = 63,
    n_repeats: int = 10,
    random_state: int = 0,
) -> PCAModelResult:
    Xc, yc = _xy(X_train, y_train)
    n_comp = min(n_components, Xc.shape[1], max(Xc.shape[0] - 1, 1))
    pipe = _make_pipe(
        [
            ("pca", PCA(n_components=n_comp, random_state=random_state)),
            ("model", SVR(C=C, epsilon=epsilon, gamma=gamma, kernel="rbf")),
        ],
        scale=scale,
        min_periods=min_periods,
    )
    pipe.fit(Xc, yc)
    Xe, ye = _xy(X_eval, y_eval)
    attr = _perm_attr(
        pipe, Xe, ye, n_repeats=n_repeats, random_state=random_state, name="svr_perm_importance"
    )
    return PCAModelResult(
        name="pca_svr",
        pipeline=pipe,
        feature_names=list(Xc.columns),
        train_predictions=_predict(pipe, Xc, "svr_train"),
        eval_predictions=_predict(pipe, Xe, "svr_eval"),
        feature_attribution=attr,
        eval_index=Xe.index,
    )
