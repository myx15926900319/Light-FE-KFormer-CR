import torch

from light_fe_kformer_cr.losses import CoolingResilienceLoss


def test_consistency_term_is_zero_for_ratio_consistent_predictions():
    loss_fn = CoolingResilienceLoss(lambda_cr=1.0, lambda_consistency=0.5)
    pred = {
        'ci_normal': torch.tensor([[4.0], [2.0]]),
        'ci_heat': torch.tensor([[3.0], [1.0]]),
        'cr': torch.tensor([[0.75], [0.5]]),
    }
    target = {
        'ci_normal': torch.tensor([[4.0], [2.0]]),
        'ci_heat': torch.tensor([[3.0], [1.0]]),
        'cr': torch.tensor([[0.75], [0.5]]),
    }
    total, parts = loss_fn(pred, target)
    assert torch.isfinite(total)
    assert parts['consistency'].item() < 1e-8
    assert total.item() < 1e-8


def test_loss_is_finite_when_predictions_are_imperfect():
    loss_fn = CoolingResilienceLoss()
    pred = {
        'ci_normal': torch.tensor([[0.01], [3.0]]),
        'ci_heat': torch.tensor([[1.0], [2.0]]),
        'cr': torch.tensor([[1.0], [0.8]]),
    }
    target = {
        'ci_normal': torch.tensor([[2.0], [4.0]]),
        'ci_heat': torch.tensor([[1.2], [2.5]]),
        'cr': torch.tensor([[0.6], [0.625]]),
    }
    total, parts = loss_fn(pred, target)
    assert torch.isfinite(total)
    assert all(torch.isfinite(v) for v in parts.values())
