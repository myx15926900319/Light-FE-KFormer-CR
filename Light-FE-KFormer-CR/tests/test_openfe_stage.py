import pandas as pd
import pytest

from light_fe_kformer_cr.openfe_stage import (
    FEATURE_GROUPS,
    OpenFEGroupEnhancer,
    extract_group,
)


def test_extract_group_uses_state_specific_column_when_available_and_static_otherwise():
    df = pd.DataFrame({
        'area': [10.0, 20.0],
        'shape_index': [1.2, 1.4],
        'green_ratio': [0.8, 0.7],
        'ndvi_park_normal': [0.72, 0.65],
        'ndvi_park_heat': [0.61, 0.58],
    })
    normal = extract_group(df, 'park', 'normal')
    heat = extract_group(df, 'park', 'heat')
    assert list(normal.columns) == FEATURE_GROUPS['park']
    assert list(heat.columns) == FEATURE_GROUPS['park']
    assert normal['area'].tolist() == heat['area'].tolist()
    assert normal['ndvi_park'].tolist() == [0.72, 0.65]
    assert heat['ndvi_park'].tolist() == [0.61, 0.58]


def test_extract_group_reports_missing_required_feature():
    df = pd.DataFrame({'area': [1.0]})
    with pytest.raises(KeyError, match='shape_index'):
        extract_group(df, 'park', 'normal')


def test_openfe_adapter_has_clear_error_when_optional_dependency_missing():
    df = pd.DataFrame({name: [0.1, 0.2] for name in FEATURE_GROUPS['weather']})
    enhancer = OpenFEGroupEnhancer(top_k=2, n_jobs=1)
    try:
        import openfe  # noqa: F401
    except ModuleNotFoundError:
        with pytest.raises(ImportError, match='openfe'):
            enhancer.fit(df, df, [1.0, 2.0], [1.1, 2.1])
