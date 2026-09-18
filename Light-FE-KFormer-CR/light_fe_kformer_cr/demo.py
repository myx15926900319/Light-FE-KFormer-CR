from __future__ import annotations

import json
import random

import numpy as np
import pandas as pd
import torch

from .data import compute_cr, group_split
from .losses import CoolingResilienceLoss
from .model import LightFEKFormerCR
from .tokens import SemanticTokenPipeline
from .train import evaluate, fit_model, make_loader


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_synthetic_pairs(n_parks: int = 30, pairs_per_park: int = 3, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = n_parks * pairs_per_park
    park_index = np.repeat(np.arange(n_parks), pairs_per_park)
    area = rng.uniform(2.0, 30.0, n)
    green_ratio = rng.uniform(0.35, 0.9, n)
    shape = rng.uniform(1.0, 2.5, n)
    ndvi_n = rng.uniform(0.45, 0.85, n)
    ndvi_h = np.clip(ndvi_n - rng.uniform(0.0, 0.12, n), 0.1, 0.9)

    df = pd.DataFrame(
        {
            "park_id": [f"park_{i:03d}" for i in park_index],
            "pair_id": [f"pair_{i:04d}" for i in range(n)],
            "area": area,
            "shape_index": shape,
            "green_ratio": green_ratio,
            "ndvi_park_normal": ndvi_n,
            "ndvi_park_heat": ndvi_h,
        }
    )
    for scale in (100, 300, 500):
        df[f"ndvi_{scale}"] = rng.uniform(0.2, 0.75, n)
        df[f"ndbi_{scale}"] = rng.uniform(-0.15, 0.45, n)
        df[f"building_{scale}"] = rng.uniform(0.1, 0.85, n)
        df[f"green_{scale}"] = rng.uniform(0.1, 0.8, n)

    df["temperature_normal"] = rng.uniform(30.0, 36.5, n)
    df["temperature_heat"] = rng.uniform(37.0, 43.0, n)
    df["wind_speed_normal"] = rng.uniform(0.5, 4.5, n)
    df["wind_speed_heat"] = rng.uniform(0.2, 3.5, n)
    df["solar_radiation_normal"] = rng.uniform(250.0, 800.0, n)
    df["solar_radiation_heat"] = rng.uniform(350.0, 900.0, n)

    build = df["building_300"].to_numpy()
    ci_normal = (
        0.7
        + 2.5 * green_ratio
        + 0.06 * np.sqrt(area)
        + 0.9 * ndvi_n
        - 0.8 * build
        + rng.normal(0, 0.15, n)
    )
    retention = np.clip(
        0.95
        - 0.012 * (df["temperature_heat"].to_numpy() - 37.0)
        - 0.25 * build
        + 0.18 * green_ratio
        + rng.normal(0, 0.04, n),
        0.35,
        1.15,
    )
    df["ci_normal"] = np.clip(ci_normal, 0.6, None)
    df["ci_heat"] = df["ci_normal"].to_numpy() * retention
    return compute_cr(df, min_ci_normal=0.5)


def run_demo(epochs: int = 5) -> dict:
    set_seed(42)
    df = make_synthetic_pairs()
    train_df, val_df, test_df = group_split(df, random_state=42)

    token_pipeline = SemanticTokenPipeline(
        bottleneck_dim=8,
        use_openfe=False,  # The demo does not require the optional dependency.
        ae_epochs=5,
        ae_patience=3,
        device="cpu",
    ).fit(train_df, val_df)

    tr_n, tr_h = token_pipeline.transform(train_df)
    va_n, va_h = token_pipeline.transform(val_df)
    te_n, te_h = token_pipeline.transform(test_df)

    train_loader = make_loader(
        tr_n,
        tr_h,
        train_df["ci_normal"].to_numpy(),
        train_df["ci_heat"].to_numpy(),
        train_df["cr"].to_numpy(),
        batch_size=32,
        shuffle=True,
    )
    val_loader = make_loader(
        va_n,
        va_h,
        val_df["ci_normal"].to_numpy(),
        val_df["ci_heat"].to_numpy(),
        val_df["cr"].to_numpy(),
        batch_size=64,
    )
    test_loader = make_loader(
        te_n,
        te_h,
        test_df["ci_normal"].to_numpy(),
        test_df["ci_heat"].to_numpy(),
        test_df["cr"].to_numpy(),
        batch_size=64,
    )

    model = LightFEKFormerCR(token_dim=8)
    loss_fn = CoolingResilienceLoss(lambda_consistency=0.2)
    model, _ = fit_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        patience=max(2, epochs),
        device="cpu",
        loss_fn=loss_fn,
    )
    return evaluate(model, test_loader, loss_fn, device="cpu")


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2, ensure_ascii=False))
