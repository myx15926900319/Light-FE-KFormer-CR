from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from .losses import CoolingResilienceLoss


def make_loader(
    normal_tokens: np.ndarray,
    heat_tokens: np.ndarray,
    ci_normal: np.ndarray,
    ci_heat: np.ndarray,
    cr: np.ndarray,
    *,
    batch_size: int = 64,
    shuffle: bool = False,
) -> DataLoader:
    normal = torch.as_tensor(normal_tokens, dtype=torch.float32)
    heat = torch.as_tensor(heat_tokens, dtype=torch.float32)
    y_n = torch.as_tensor(np.asarray(ci_normal), dtype=torch.float32).reshape(-1, 1)
    y_h = torch.as_tensor(np.asarray(ci_heat), dtype=torch.float32).reshape(-1, 1)
    y_cr = torch.as_tensor(np.asarray(cr), dtype=torch.float32).reshape(-1, 1)
    n = normal.shape[0]
    if not (heat.shape[0] == y_n.shape[0] == y_h.shape[0] == y_cr.shape[0] == n):
        raise ValueError("All token and target arrays must contain the same number of rows")
    dataset = TensorDataset(normal, heat, y_n, y_h, y_cr)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _target_dict(y_n: torch.Tensor, y_h: torch.Tensor, y_cr: torch.Tensor):
    return {"ci_normal": y_n, "ci_heat": y_h, "cr": y_cr}


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: CoolingResilienceLoss,
    *,
    device: str = "cpu",
) -> Dict[str, float]:
    model.to(device)
    model.train()
    total_loss = 0.0
    total_rows = 0
    for normal, heat, y_n, y_h, y_cr in loader:
        normal, heat = normal.to(device), heat.to(device)
        y_n, y_h, y_cr = y_n.to(device), y_h.to(device), y_cr.to(device)
        optimizer.zero_grad()
        pred = model(normal, heat)
        loss, _ = loss_fn(pred, _target_dict(y_n, y_h, y_cr))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * normal.size(0)
        total_rows += normal.size(0)
    return {"loss": total_loss / max(total_rows, 1)}


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: CoolingResilienceLoss,
    *,
    device: str = "cpu",
) -> Dict[str, float]:
    model.to(device)
    model.eval()
    total_loss = 0.0
    total_rows = 0
    pred_cr, true_cr = [], []
    pred_n, true_n = [], []
    pred_h, true_h = [], []

    with torch.no_grad():
        for normal, heat, y_n, y_h, y_cr in loader:
            normal, heat = normal.to(device), heat.to(device)
            y_n, y_h, y_cr = y_n.to(device), y_h.to(device), y_cr.to(device)
            pred = model(normal, heat)
            loss, _ = loss_fn(pred, _target_dict(y_n, y_h, y_cr))
            total_loss += loss.item() * normal.size(0)
            total_rows += normal.size(0)

            pred_cr.append(pred["cr"].cpu().numpy().ravel())
            true_cr.append(y_cr.cpu().numpy().ravel())
            pred_n.append(pred["ci_normal"].cpu().numpy().ravel())
            true_n.append(y_n.cpu().numpy().ravel())
            pred_h.append(pred["ci_heat"].cpu().numpy().ravel())
            true_h.append(y_h.cpu().numpy().ravel())

    def cat(parts):
        return np.concatenate(parts) if parts else np.array([], dtype=float)

    pcr, tcr = cat(pred_cr), cat(true_cr)
    pn, tn = cat(pred_n), cat(true_n)
    ph, th = cat(pred_h), cat(true_h)

    metrics = {
        "loss": total_loss / max(total_rows, 1),
        "cr_mae": float(mean_absolute_error(tcr, pcr)),
        "cr_rmse": float(np.sqrt(mean_squared_error(tcr, pcr))),
        "ci_normal_mae": float(mean_absolute_error(tn, pn)),
        "ci_heat_mae": float(mean_absolute_error(th, ph)),
    }
    if len(tcr) >= 2 and np.std(tcr) > 0:
        metrics["cr_r2"] = float(r2_score(tcr, pcr))
        metrics["cr_spearman"] = float(
            pd.Series(tcr).rank().corr(pd.Series(pcr).rank())
        )
    else:
        metrics["cr_r2"] = float("nan")
        metrics["cr_spearman"] = float("nan")
    return metrics


def fit_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 200,
    lr: float = 5e-4,
    weight_decay: float = 1e-5,
    patience: int = 25,
    device: str = "cpu",
    loss_fn: Optional[CoolingResilienceLoss] = None,
) -> Tuple[torch.nn.Module, Dict[str, list]]:
    loss_fn = loss_fn or CoolingResilienceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "val_loss": []}
    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    bad_epochs = 0

    for _ in range(epochs):
        train_stats = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device=device
        )
        val_stats = evaluate(model, val_loader, loss_fn, device=device)
        history["train_loss"].append(train_stats["loss"])
        history["val_loss"].append(val_stats["loss"])

        if val_stats["loss"] < best_val - 1e-8:
            best_val = val_stats["loss"]
            best_state = deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model, history
