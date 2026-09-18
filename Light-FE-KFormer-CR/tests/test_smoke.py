import numpy as np
import torch

from light_fe_kformer_cr.losses import CoolingResilienceLoss
from light_fe_kformer_cr.model import LightFEKFormerCR
from light_fe_kformer_cr.train import make_loader, train_one_epoch, evaluate


def test_one_training_step_and_evaluation_are_finite():
    rng = np.random.default_rng(4)
    n = 20
    normal = rng.normal(size=(n, 5, 8)).astype(np.float32)
    heat = rng.normal(size=(n, 5, 8)).astype(np.float32)
    ci_normal = rng.uniform(1.0, 5.0, n).astype(np.float32)
    ci_heat = ci_normal * rng.uniform(0.5, 1.0, n).astype(np.float32)
    cr = ci_heat / ci_normal

    loader = make_loader(normal, heat, ci_normal, ci_heat, cr, batch_size=8, shuffle=True)
    model = LightFEKFormerCR(token_dim=8)
    loss_fn = CoolingResilienceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_stats = train_one_epoch(model, loader, optimizer, loss_fn, device='cpu')
    eval_stats = evaluate(model, loader, loss_fn, device='cpu')

    assert np.isfinite(train_stats['loss'])
    assert np.isfinite(eval_stats['loss'])
    assert np.isfinite(eval_stats['cr_mae'])
    assert np.isfinite(eval_stats['cr_rmse'])
