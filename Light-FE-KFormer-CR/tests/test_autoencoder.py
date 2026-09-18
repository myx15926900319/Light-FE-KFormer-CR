import numpy as np
import torch

from light_fe_kformer_cr.autoencoder import AutoEncoder, encode_array, fit_autoencoder


def test_autoencoder_reconstruction_and_embedding_shapes():
    model = AutoEncoder(input_dim=12, bottleneck_dim=8, hidden_dim=16)
    x = torch.randn(5, 12)
    recon = model(x)
    z = model.encode(x)
    assert recon.shape == (5, 12)
    assert z.shape == (5, 8)


def test_fit_autoencoder_and_encode_array_returns_finite_tokens():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(40, 10)).astype(np.float32)
    val = rng.normal(size=(10, 10)).astype(np.float32)
    model, history = fit_autoencoder(
        train,
        val,
        bottleneck_dim=4,
        hidden_dim=8,
        epochs=3,
        batch_size=16,
        patience=3,
        device='cpu',
    )
    z = encode_array(model, val, device='cpu')
    assert z.shape == (10, 4)
    assert np.isfinite(z).all()
    assert len(history['train_loss']) >= 1
