import torch

from light_fe_kformer_cr.model import KFormerEncoder, LightFEKFormerCR


def test_kformer_encoder_shapes_match_semantic_tokens():
    encoder = KFormerEncoder(token_dim=8, num_factors=4, factor_dim=8, num_heads=2)
    x = torch.randn(6, 5, 8)
    fused, similarity = encoder(x)
    assert fused.shape == (6, 32)
    assert similarity.shape == (6, 4, 5)


def test_kformer_input_projection_supports_different_token_dimension():
    encoder = KFormerEncoder(token_dim=6, num_factors=3, factor_dim=8, num_heads=2)
    x = torch.randn(4, 5, 6)
    fused, similarity = encoder(x)
    assert fused.shape == (4, 24)
    assert similarity.shape == (4, 3, 5)


def test_light_fe_kformer_cr_returns_three_positive_predictions_and_two_similarity_maps():
    model = LightFEKFormerCR(
        token_dim=8,
        num_factors=4,
        factor_dim=8,
        num_heads=2,
        head_hidden_dim=32,
    )
    normal = torch.randn(7, 5, 8)
    heat = torch.randn(7, 5, 8)
    out = model(normal, heat)
    assert out['ci_normal'].shape == (7, 1)
    assert out['ci_heat'].shape == (7, 1)
    assert out['cr'].shape == (7, 1)
    assert torch.all(out['ci_normal'] > 0)
    assert torch.all(out['ci_heat'] > 0)
    assert torch.all(out['cr'] > 0)
    assert out['similarity_normal'].shape == (7, 4, 5)
    assert out['similarity_heat'].shape == (7, 4, 5)
    # There is exactly one shared KFormer encoder object for both states.
    assert hasattr(model, 'encoder')
    assert not hasattr(model, 'normal_encoder')
    assert not hasattr(model, 'heat_encoder')
