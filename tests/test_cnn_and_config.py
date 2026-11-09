import numpy as np
import pandas as pd

from spy_alpha_research.data import load_config, project_root
from spy_alpha_research.pipelines.cnn import build_window_tensor, fit_cnn


def test_configs_exist():
    root = project_root()
    exp = load_config(root / "configs" / "experiment.yaml")
    assert exp["dates"]["end"] == "2024-12-31"
    assert exp["dates"]["train_end"] == "2021-12-31"
    bt = load_config(root / "configs" / "backtest.yaml")
    assert bt["equal_weight"] is True


def test_cnn_forward_shape():
    idx = pd.bdate_range("2020-01-01", periods=160)
    rng = np.random.default_rng(0)
    panel = pd.DataFrame(rng.normal(size=(160, 3)), index=idx, columns=list("abc"))
    target = pd.Series(rng.normal(size=160), index=idx)
    X, y, ids = build_window_tensor(panel, target, lookback=16)
    assert X.ndim == 3
    assert X.shape[1] == 3
    assert X.shape[2] == 16
    assert len(y) == len(ids) == X.shape[0]

    result = fit_cnn(
        panel.iloc[:100],
        target.iloc[:100],
        panel.iloc[100:],
        target.iloc[100:],
        lookback=16,
        conv_channels=[8, 8],
        epochs=2,
        batch_size=32,
        seed=0,
    )
    assert len(result.eval_predictions) > 0
    assert set(result.channel_saliency.index) == set(panel.columns)
