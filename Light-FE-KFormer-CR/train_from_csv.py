from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

from light_fe_kformer_cr.data import compute_cr, group_split
from light_fe_kformer_cr.losses import CoolingResilienceLoss
from light_fe_kformer_cr.model import LightFEKFormerCR
from light_fe_kformer_cr.tokens import SemanticTokenPipeline
from light_fe_kformer_cr.train import evaluate, fit_model, make_loader


def parse_args():
    p = argparse.ArgumentParser(description="Train Light-FE-KFormer-CR from paired park data")
    p.add_argument("csv", type=Path, help="Paired normal/heatwave CSV")
    p.add_argument("--output", type=Path, default=Path("outputs"))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--ae-epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--min-ci-normal", type=float, default=0.5)
    p.add_argument("--no-openfe", action="store_true", help="Disable optional OpenFE stage")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cpu or cuda; default chooses cuda when available")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.csv)
    df = compute_cr(df, min_ci_normal=args.min_ci_normal)
    train_df, val_df, test_df = group_split(df, random_state=args.seed)

    token_pipeline = SemanticTokenPipeline(
        bottleneck_dim=8,
        use_openfe=not args.no_openfe,
        ae_epochs=args.ae_epochs,
        device=device,
    ).fit(train_df, val_df)

    train_n, train_h = token_pipeline.transform(train_df)
    val_n, val_h = token_pipeline.transform(val_df)
    test_n, test_h = token_pipeline.transform(test_df)

    def loader(frame, n_tokens, h_tokens, shuffle=False):
        return make_loader(
            n_tokens,
            h_tokens,
            frame["ci_normal"].to_numpy(),
            frame["ci_heat"].to_numpy(),
            frame["cr"].to_numpy(),
            batch_size=args.batch_size,
            shuffle=shuffle,
        )

    train_loader = loader(train_df, train_n, train_h, True)
    val_loader = loader(val_df, val_n, val_h)
    test_loader = loader(test_df, test_n, test_h)

    model = LightFEKFormerCR(token_dim=8)
    loss_fn = CoolingResilienceLoss(lambda_cr=1.0, lambda_consistency=0.2)
    model, history = fit_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        device=device,
        loss_fn=loss_fn,
    )
    metrics = evaluate(model, test_loader, loss_fn, device=device)

    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output / "light_fe_kformer_cr.pt")
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
