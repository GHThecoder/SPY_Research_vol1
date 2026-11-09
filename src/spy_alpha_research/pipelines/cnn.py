"""PyTorch 1D CNN feature extractor with linear readout.

Train on a training panel only. Predictions and channel saliency are computed on
an evaluation panel (OOS / holdout), never reported as primary metrics from the
training windows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover
    torch = None
    nn = None


def _require_torch() -> None:
    if torch is None:
        raise ImportError("torch is required for the CNN pipeline")


class ConvFeatureExtractor(nn.Module if nn is not None else object):
    def __init__(
        self,
        n_channels: int,
        conv_channels: list[int] | None = None,
        kernel_size: int = 3,
        embed_dim: int = 8,
        dropout: float = 0.1,
    ) -> None:
        _require_torch()
        super().__init__()
        conv_channels = conv_channels or [16, 32]
        layers: list[nn.Module] = []
        in_ch = n_channels
        for out_ch in conv_channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(in_ch, 1)
        self.embed_dim = in_ch

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        h = self.conv(x)
        emb = self.pool(h).squeeze(-1)
        return self.head(emb).squeeze(-1)

    def embed(self, x: "torch.Tensor") -> "torch.Tensor":
        h = self.conv(x)
        return self.pool(h).squeeze(-1)


@dataclass
class CNNResult:
    model: object
    train_predictions: pd.Series
    eval_predictions: pd.Series
    channel_saliency: pd.Series
    history: list[float]
    lookback: int
    channel_names: list[str]


def build_window_tensor(
    panel: pd.DataFrame,
    target: pd.Series,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Build (N, C, L) windows ending at t; label = target[t] (= r_{t+1} if contracted)."""
    df = panel.join(target.rename("y"), how="inner").dropna()
    values = df.drop(columns=["y"]).to_numpy(dtype=np.float32)
    y = df["y"].to_numpy(dtype=np.float32)
    idx = df.index
    xs, ys, ids = [], [], []
    for i in range(lookback - 1, len(df)):
        xs.append(values[i - lookback + 1 : i + 1].T)
        ys.append(y[i])
        ids.append(idx[i])
    if not xs:
        return np.empty((0, panel.shape[1], lookback)), np.empty((0,)), pd.DatetimeIndex([])
    return np.stack(xs), np.asarray(ys), pd.DatetimeIndex(ids)


def _channel_saliency(
    model: "ConvFeatureExtractor",
    X: np.ndarray,
    channel_names: list[str],
    device: "torch.device",
) -> pd.Series:
    model.eval()
    xt = torch.from_numpy(X).to(device).requires_grad_(True)
    out = model(xt).sum()
    model.zero_grad(set_to_none=True)
    out.backward()
    sal = (xt.grad.abs() * xt.abs()).mean(dim=(0, 2)).detach().cpu().numpy()
    return pd.Series(sal, index=channel_names, name="cnn_channel_saliency").sort_values(
        ascending=False
    )


def fit_cnn(
    panel_train: pd.DataFrame,
    target_train: pd.Series,
    panel_eval: pd.DataFrame,
    target_eval: pd.Series,
    *,
    lookback: int = 32,
    conv_channels: list[int] | None = None,
    kernel_size: int = 3,
    embed_dim: int = 8,
    dropout: float = 0.1,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> CNNResult:
    """Fit on train windows; predict + saliency on eval windows."""
    _require_torch()
    torch.manual_seed(seed)

    if list(panel_train.columns) != list(panel_eval.columns):
        raise ValueError("panel_train and panel_eval must share the same columns/order")

    X_tr, y_tr, idx_tr = build_window_tensor(panel_train, target_train, lookback)
    X_ev, y_ev, idx_ev = build_window_tensor(panel_eval, target_eval, lookback)
    if len(y_tr) == 0:
        raise ValueError("Not enough training data for CNN windows")
    if len(y_ev) == 0:
        raise ValueError("Not enough eval data for CNN windows")

    device = torch.device("cpu")
    model = ConvFeatureExtractor(
        n_channels=X_tr.shape[1],
        conv_channels=conv_channels,
        kernel_size=kernel_size,
        embed_dim=embed_dim,
        dropout=dropout,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    history: list[float] = []
    model.train()
    for _ in range(epochs):
        total = 0.0
        n = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(yb)
            n += len(yb)
        history.append(total / max(n, 1))

    model.eval()
    with torch.no_grad():
        train_pred = model(torch.from_numpy(X_tr).to(device)).cpu().numpy()
        eval_pred = model(torch.from_numpy(X_ev).to(device)).cpu().numpy()

    saliency = _channel_saliency(model, X_ev, list(panel_train.columns), device)

    return CNNResult(
        model=model,
        train_predictions=pd.Series(train_pred, index=idx_tr, name="cnn_train_pred"),
        eval_predictions=pd.Series(eval_pred, index=idx_ev, name="cnn_eval_pred"),
        channel_saliency=saliency,
        history=history,
        lookback=lookback,
        channel_names=list(panel_train.columns),
    )
