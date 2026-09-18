from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class AutoEncoder(nn.Module):
    """Lightweight feature compressor preserved from the original FE-KFormer."""

    def __init__(self, input_dim: int, bottleneck_dim: int = 8, hidden_dim: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(x))


def fit_autoencoder(
    train_x: np.ndarray,
    val_x: Optional[np.ndarray] = None,
    *,
    bottleneck_dim: int = 8,
    hidden_dim: int = 16,
    lr: float = 1e-3,
    epochs: int = 150,
    batch_size: int = 64,
    patience: int = 20,
    device: str = "cpu",
) -> Tuple[AutoEncoder, Dict[str, list]]:
    """Fit one shared autoencoder on training-state rows only.

    For a semantic group, callers should concatenate normal and heatwave
    training rows before calling this function. Validation rows are never used
    to fit weights; they only control early stopping.
    """
    train = np.asarray(train_x, dtype=np.float32)
    if train.ndim != 2:
        raise ValueError(f"Expected 2D training array, got {train.shape}")
    val = None if val_x is None else np.asarray(val_x, dtype=np.float32)

    model = AutoEncoder(
        input_dim=train.shape[1],
        bottleneck_dim=bottleneck_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(torch.from_numpy(train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "val_loss": []}
    best_state = deepcopy(model.state_dict())
    best_loss = float("inf")
    bad_epochs = 0

    for _ in range(epochs):
        model.train()
        total = 0.0
        count = 0
        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            recon = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
            count += xb.size(0)
        train_loss = total / max(count, 1)
        history["train_loss"].append(train_loss)

        model.eval()
        with torch.no_grad():
            if val is None:
                monitor_loss = train_loss
            else:
                vb = torch.from_numpy(val).to(device)
                monitor_loss = criterion(model(vb), vb).item()
        history["val_loss"].append(float(monitor_loss))

        if monitor_loss < best_loss - 1e-8:
            best_loss = float(monitor_loss)
            best_state = deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model, history


def encode_array(
    model: AutoEncoder,
    x: np.ndarray,
    *,
    batch_size: int = 1024,
    device: str = "cpu",
) -> np.ndarray:
    """Encode a 2D array into bottleneck tokens."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.shape}")
    loader = DataLoader(TensorDataset(torch.from_numpy(arr)), batch_size=batch_size)
    model = model.to(device)
    model.eval()
    chunks = []
    with torch.no_grad():
        for (xb,) in loader:
            chunks.append(model.encode(xb.to(device)).cpu().numpy())
    if not chunks:
        return np.empty((0, model.bottleneck_dim), dtype=np.float32)
    return np.concatenate(chunks, axis=0)
