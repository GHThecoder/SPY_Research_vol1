"""Tree-based regression pipelines for attribution.

Fit on a training window; predictions and permutation importance are computed on
an evaluation window (OOS / CV fold). In-sample scores are never used as the
primary attribution surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance


@dataclass
class TreePipelineResult:
    model_name: str
    model: Any
    feature_names: list[str]
    train_predictions: pd.Series
    eval_predictions: pd.Series
    importances: pd.Series
    perm_importances: pd.Series
    eval_index: pd.Index


def _xy(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    df = X.join(y.rename("y"), how="inner").dropna()
    return df.drop(columns=["y"]), df["y"]


def _predict_frame(model: Any, X: pd.DataFrame, name: str) -> pd.Series:
    Xc = X.dropna()
    if Xc.empty:
        return pd.Series(dtype=float, name=name)
    return pd.Series(model.predict(Xc), index=Xc.index, name=name)


def permutation_importances(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.Series:
    """Permutation importance on the provided (typically eval) frame."""
    Xc, yc = _xy(X, y)
    perm = permutation_importance(
        model, Xc, yc, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
    )
    return pd.Series(perm.importances_mean, index=Xc.columns).sort_values(ascending=False)


def fit_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    *,
    n_estimators: int = 200,
    max_depth: int = 6,
    min_samples_leaf: int = 20,
    random_state: int = 42,
    n_jobs: int = -1,
    n_repeats: int = 10,
) -> TreePipelineResult:
    """Fit RF on train; predict + permutation-importance on eval."""
    Xc, yc = _xy(X_train, y_train)
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(Xc, yc)

    perm = permutation_importances(
        model, X_eval, y_eval, n_repeats=n_repeats, random_state=random_state
    )
    perm.name = "rf_perm_importance"

    return TreePipelineResult(
        model_name="random_forest",
        model=model,
        feature_names=list(Xc.columns),
        train_predictions=_predict_frame(model, Xc, "rf_train_pred"),
        eval_predictions=_predict_frame(model, X_eval, "rf_eval_pred"),
        importances=perm,  # primary: eval permutation
        perm_importances=perm,
        eval_index=_xy(X_eval, y_eval)[0].index,
    )


def fit_hist_gbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    *,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    max_iter: int = 200,
    l2_regularization: float = 0.1,
    random_state: int = 42,
    n_repeats: int = 10,
) -> TreePipelineResult:
    """Fit HGBM on train; predict + permutation-importance on eval."""
    Xc, yc = _xy(X_train, y_train)
    model = HistGradientBoostingRegressor(
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=max_iter,
        l2_regularization=l2_regularization,
        random_state=random_state,
    )
    model.fit(Xc, yc)

    perm = permutation_importances(
        model, X_eval, y_eval, n_repeats=n_repeats, random_state=random_state
    )
    perm.name = "hgb_perm_importance"

    return TreePipelineResult(
        model_name="hist_gbm",
        model=model,
        feature_names=list(Xc.columns),
        train_predictions=_predict_frame(model, Xc, "hgb_train_pred"),
        eval_predictions=_predict_frame(model, X_eval, "hgb_eval_pred"),
        importances=perm,
        perm_importances=perm,
        eval_index=_xy(X_eval, y_eval)[0].index,
    )
