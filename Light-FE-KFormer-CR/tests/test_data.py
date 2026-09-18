import numpy as np
import pandas as pd

from light_fe_kformer_cr.data import RobustStandardizer, compute_cr, group_split


def test_compute_cr_filters_unstable_normal_ci():
    df = pd.DataFrame({
        'ci_normal': [2.0, 0.2, 4.0],
        'ci_heat': [1.0, 0.1, 5.0],
    })
    out = compute_cr(df, min_ci_normal=0.5)
    assert len(out) == 2
    assert np.allclose(out['cr'].to_numpy(), [0.5, 1.25])


def test_group_split_keeps_parks_disjoint():
    df = pd.DataFrame({
        'park_id': np.repeat([f'p{i}' for i in range(10)], 3),
        'x': np.arange(30),
    })
    train, val, test = group_split(df, group_col='park_id', random_state=7)
    train_ids = set(train.park_id)
    val_ids = set(val.park_id)
    test_ids = set(test.park_id)
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert train_ids | val_ids | test_ids == set(df.park_id)


def test_robust_standardizer_uses_fit_statistics_only():
    train = np.array([[0.0], [1.0], [2.0], [3.0]])
    test = np.array([[100.0]])
    scaler = RobustStandardizer(lower_q=0.0, upper_q=1.0).fit(train)
    transformed = scaler.transform(test)
    # Training upper bound is 3, so test value must first clip to 3.
    expected = (3.0 - train.mean()) / train.std()
    assert np.allclose(transformed[0, 0], expected)
