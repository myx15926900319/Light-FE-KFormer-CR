from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def compute_cr(
    df: pd.DataFrame,
    ci_normal_col: str = "ci_normal",
    ci_heat_col: str = "ci_heat",
    min_ci_normal: float = 0.5,
    cr_col: str = "cr",
) -> pd.DataFrame:
    """Filter numerically unstable pairs and compute cooling resilience.

    CR is defined as CI_heat / CI_normal.  The denominator is filtered rather
    than hidden behind a tiny epsilon because pairs with almost no normal-state
    cooling do not have a stable ratio interpretation.
    """
    required = {ci_normal_col, ci_heat_col}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    ci_n = pd.to_numeric(out[ci_normal_col], errors="coerce")
    ci_h = pd.to_numeric(out[ci_heat_col], errors="coerce")
    valid = np.isfinite(ci_n) & np.isfinite(ci_h) & (ci_n >= min_ci_normal)
    out = out.loc[valid].copy()
    out[cr_col] = out[ci_heat_col].astype(float) / out[ci_normal_col].astype(float)
    return out.reset_index(drop=True)


def group_split(
    df: pd.DataFrame,
    group_col: str = "park_id",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows by park/group so the same park never leaks across splits."""
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1")
    if group_col not in df:
        raise KeyError(f"Missing group column: {group_col}")
    if df[group_col].nunique() < 3:
        raise ValueError("At least three unique groups are required")

    first = GroupShuffleSplit(
        n_splits=1, train_size=train_size, random_state=random_state
    )
    train_idx, remainder_idx = next(first.split(df, groups=df[group_col]))
    train = df.iloc[train_idx].copy()
    remainder = df.iloc[remainder_idx].copy()

    relative_val = val_size / (val_size + test_size)
    second = GroupShuffleSplit(
        n_splits=1, train_size=relative_val, random_state=random_state + 1
    )
    val_idx, test_idx = next(
        second.split(remainder, groups=remainder[group_col])
    )
    val = remainder.iloc[val_idx].copy()
    test = remainder.iloc[test_idx].copy()
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


@dataclass
class RobustStandardizer:
    """Winsorize and standardize using *training-only* fitted statistics."""

    lower_q: float = 0.05
    upper_q: float = 0.95
    eps: float = 1e-8

    def fit(self, x: np.ndarray) -> "RobustStandardizer":
        x = self._as_2d(x)
        if not (0.0 <= self.lower_q < self.upper_q <= 1.0):
            raise ValueError("Quantiles must satisfy 0 <= lower_q < upper_q <= 1")
        self.lower_ = np.quantile(x, self.lower_q, axis=0)
        self.upper_ = np.quantile(x, self.upper_q, axis=0)
        clipped = np.clip(x, self.lower_, self.upper_)
        self.mean_ = clipped.mean(axis=0)
        std = clipped.std(axis=0)
        self.std_ = np.where(std < self.eps, 1.0, std)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        for name in ("lower_", "upper_", "mean_", "std_"):
            if not hasattr(self, name):
                raise RuntimeError("RobustStandardizer must be fit before transform")
        x = self._as_2d(x)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        clipped = np.clip(x, self.lower_, self.upper_)
        return (clipped - self.mean_) / self.std_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    @staticmethod
    def _as_2d(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {arr.shape}")
        return arr
