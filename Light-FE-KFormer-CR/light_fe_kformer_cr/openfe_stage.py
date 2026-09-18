from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


# Five physically meaningful semantic tokens replace the original 15 arbitrary
# combinations of area/shape/fragmentation/connectivity categories.
FEATURE_GROUPS: Dict[str, List[str]] = {
    "park": ["area", "shape_index", "green_ratio", "ndvi_park"],
    "buf100": ["ndvi_100", "ndbi_100", "building_100", "green_100"],
    "buf300": ["ndvi_300", "ndbi_300", "building_300", "green_300"],
    "buf500": ["ndvi_500", "ndbi_500", "building_500", "green_500"],
    "weather": ["temperature", "wind_speed", "solar_radiation"],
}

STATES = ("normal", "heat")


def extract_group(df: pd.DataFrame, group: str, state: str) -> pd.DataFrame:
    """Return one semantic feature group for one state.

    For each base feature, `<feature>_<state>` is preferred when present;
    otherwise the unsuffixed column is treated as static and shared between
    normal and heatwave states. Returned columns always use base feature names,
    so both states have the exact same schema.
    """
    if group not in FEATURE_GROUPS:
        raise KeyError(f"Unknown feature group: {group}")
    if state not in STATES:
        raise ValueError(f"state must be one of {STATES}, got {state!r}")

    result = {}
    for base in FEATURE_GROUPS[group]:
        state_col = f"{base}_{state}"
        if state_col in df.columns:
            source = state_col
        elif base in df.columns:
            source = base
        else:
            raise KeyError(
                f"Missing required feature '{base}': expected '{state_col}' "
                f"or static column '{base}'"
            )
        result[base] = pd.to_numeric(df[source], errors="coerce")
    return pd.DataFrame(result, index=df.index)


@dataclass
class OpenFEGroupEnhancer:
    """Thin, leakage-safe adapter around the optional `openfe` package.

    Fit this object only on the training split. Normal and heatwave training
    rows are stacked before fitting, and their respective CI values are stacked
    as the supervised target. The same generated feature definitions are then
    used to transform both states and every downstream split.
    """

    top_k: int = 8
    n_jobs: int = 4
    include_original: bool = False

    def _imports(self):
        try:
            from openfe import OpenFE, transform, tree_to_formula
        except ModuleNotFoundError as exc:
            raise ImportError(
                "The optional dependency 'openfe' is required for OpenFE feature "
                "enhancement. Install it in the training environment before using "
                "OpenFEGroupEnhancer."
            ) from exc
        return OpenFE, transform, tree_to_formula

    def fit(
        self,
        normal_x: pd.DataFrame,
        heat_x: pd.DataFrame,
        ci_normal: Sequence[float],
        ci_heat: Sequence[float],
    ) -> "OpenFEGroupEnhancer":
        OpenFE, _, tree_to_formula = self._imports()
        self._validate_pair(normal_x, heat_x)
        fit_x = pd.concat(
            [normal_x.reset_index(drop=True), heat_x.reset_index(drop=True)],
            axis=0,
            ignore_index=True,
        )
        fit_y = pd.DataFrame(
            {
                "ci": np.concatenate(
                    [np.asarray(ci_normal, dtype=float), np.asarray(ci_heat, dtype=float)]
                )
            }
        )
        if len(fit_x) != len(fit_y):
            raise ValueError("Feature rows and CI labels must have matching lengths")

        self.model_ = OpenFE()
        self.model_.fit(data=fit_x, label=fit_y, n_jobs=self.n_jobs)
        self.selected_features_ = list(self.model_.new_features_list[: self.top_k])
        self.feature_formulas_ = [tree_to_formula(f) for f in self.selected_features_]
        self.base_columns_ = list(normal_x.columns)
        return self

    def transform_pair(
        self, normal_x: pd.DataFrame, heat_x: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        _, transform, _ = self._imports()
        if not hasattr(self, "selected_features_"):
            raise RuntimeError("OpenFEGroupEnhancer must be fit before transform")
        self._validate_pair(normal_x, heat_x)
        normal_aug, heat_aug = transform(
            normal_x,
            heat_x,
            self.selected_features_,
            n_jobs=self.n_jobs,
        )
        if self.include_original:
            return normal_aug, heat_aug
        k = len(self.selected_features_)
        if k == 0:
            # Keep base inputs if OpenFE returns no generated feature.
            return normal_x.copy(), heat_x.copy()
        normal_generated = normal_aug.iloc[:, -k:].copy()
        heat_generated = heat_aug.iloc[:, -k:].copy()
        normal_generated.columns = self.feature_formulas_
        heat_generated.columns = self.feature_formulas_
        return normal_generated, heat_generated

    @staticmethod
    def _validate_pair(normal_x: pd.DataFrame, heat_x: pd.DataFrame) -> None:
        if list(normal_x.columns) != list(heat_x.columns):
            raise ValueError("Normal and heatwave feature schemas must be identical")
        if len(normal_x) != len(heat_x):
            raise ValueError("Normal and heatwave feature rows must be paired")
