import numpy as np
import pandas as pd

from light_fe_kformer_cr.tokens import SemanticTokenPipeline


def make_pairs(n=16):
    rng = np.random.default_rng(1)
    data = {
        'park_id': [f'p{i // 2}' for i in range(n)],
        'pair_id': [f'pair{i}' for i in range(n)],
        'area': rng.uniform(1, 20, n),
        'shape_index': rng.uniform(1, 2, n),
        'green_ratio': rng.uniform(0.4, 0.9, n),
        'ndvi_park_normal': rng.uniform(0.5, 0.9, n),
        'ndvi_park_heat': rng.uniform(0.4, 0.8, n),
        'ci_normal': rng.uniform(1.0, 5.0, n),
    }
    for scale in (100, 300, 500):
        data[f'ndvi_{scale}'] = rng.uniform(0.2, 0.8, n)
        data[f'ndbi_{scale}'] = rng.uniform(-0.2, 0.5, n)
        data[f'building_{scale}'] = rng.uniform(0.1, 0.9, n)
        data[f'green_{scale}'] = rng.uniform(0.1, 0.9, n)
    for state in ('normal', 'heat'):
        data[f'temperature_{state}'] = rng.uniform(30, 42, n)
        data[f'wind_speed_{state}'] = rng.uniform(0.2, 5, n)
        data[f'solar_radiation_{state}'] = rng.uniform(100, 900, n)
    data['ci_heat'] = np.asarray(data['ci_normal']) * rng.uniform(0.4, 1.1, n)
    return pd.DataFrame(data)


def test_semantic_token_pipeline_produces_five_shared_state_tokens():
    df = make_pairs()
    train = df.iloc[:12].copy()
    val = df.iloc[12:].copy()
    pipeline = SemanticTokenPipeline(
        bottleneck_dim=4,
        use_openfe=False,
        ae_epochs=2,
        ae_patience=2,
        device='cpu',
    )
    pipeline.fit(train, val)
    normal, heat = pipeline.transform(val)
    assert normal.shape == (4, 5, 4)
    assert heat.shape == (4, 5, 4)
    assert np.isfinite(normal).all()
    assert np.isfinite(heat).all()
    assert set(pipeline.autoencoders_) == {'park', 'buf100', 'buf300', 'buf500', 'weather'}
