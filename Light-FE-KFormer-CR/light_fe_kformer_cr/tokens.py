from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .autoencoder import AutoEncoder, encode_array, fit_autoencoder
from .data import RobustStandardizer
from .openfe_stage import FEATURE_GROUPS, OpenFEGroupEnhancer, extract_group


@dataclass
class SemanticTokenPipeline:
    """Build five semantic tokens shared across normal and heatwave states.

    The fitting order is deliberately leakage-safe:
      1) optional OpenFE on training rows only,
      2) robust scaler on training rows only,
      3) one shared autoencoder per semantic group on stacked normal+heat rows.
    """

    bottleneck_dim: int = 8
    hidden_dim: int = 16
    use_openfe: bool = True
    openfe_top_k: int = 8
    openfe_n_jobs: int = 4
    ae_epochs: int = 150
    ae_patience: int = 20
    ae_batch_size: int = 64
    device: str = "cpu"

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
    ) -> "SemanticTokenPipeline":
        self.enhancers_: Dict[str, Optional[OpenFEGroupEnhancer]] = {}
        self.scalers_: Dict[str, RobustStandardizer] = {}
        self.autoencoders_: Dict[str, AutoEncoder] = {}

        if "ci_normal" not in train_df or "ci_heat" not in train_df:
            raise KeyError("train_df must contain ci_normal and ci_heat")

        for group in FEATURE_GROUPS:
            train_n = extract_group(train_df, group, "normal")
            train_h = extract_group(train_df, group, "heat")

            enhancer = None
            if self.use_openfe:
                enhancer = OpenFEGroupEnhancer(
                    top_k=self.openfe_top_k,
                    n_jobs=self.openfe_n_jobs,
                    include_original=False,
                )
                enhancer.fit(
                    train_n,
                    train_h,
                    train_df["ci_normal"].to_numpy(),
                    train_df["ci_heat"].to_numpy(),
                )
                train_n, train_h = enhancer.transform_pair(train_n, train_h)

            train_stacked = np.vstack(
                [train_n.to_numpy(dtype=float), train_h.to_numpy(dtype=float)]
            )
            scaler = RobustStandardizer().fit(train_stacked)
            train_scaled = scaler.transform(train_stacked)

            val_scaled = None
            if val_df is not None and len(val_df) > 0:
                val_n = extract_group(val_df, group, "normal")
                val_h = extract_group(val_df, group, "heat")
                if enhancer is not None:
                    val_n, val_h = enhancer.transform_pair(val_n, val_h)
                val_stacked = np.vstack(
                    [val_n.to_numpy(dtype=float), val_h.to_numpy(dtype=float)]
                )
                val_scaled = scaler.transform(val_stacked)

            ae, _ = fit_autoencoder(
                train_scaled,
                val_scaled,
                bottleneck_dim=self.bottleneck_dim,
                hidden_dim=self.hidden_dim,
                epochs=self.ae_epochs,
                batch_size=self.ae_batch_size,
                patience=self.ae_patience,
                device=self.device,
            )
            self.enhancers_[group] = enhancer
            self.scalers_[group] = scaler
            self.autoencoders_[group] = ae

        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "autoencoders_"):
            raise RuntimeError("SemanticTokenPipeline must be fit before transform")

        normal_tokens = []
        heat_tokens = []
        for group in FEATURE_GROUPS:
            normal = extract_group(df, group, "normal")
            heat = extract_group(df, group, "heat")
            enhancer = self.enhancers_[group]
            if enhancer is not None:
                normal, heat = enhancer.transform_pair(normal, heat)

            scaler = self.scalers_[group]
            normal_scaled = scaler.transform(normal.to_numpy(dtype=float))
            heat_scaled = scaler.transform(heat.to_numpy(dtype=float))
            ae = self.autoencoders_[group]
            normal_z = encode_array(ae, normal_scaled, device=self.device)
            heat_z = encode_array(ae, heat_scaled, device=self.device)
            normal_tokens.append(normal_z[:, None, :])
            heat_tokens.append(heat_z[:, None, :])

        return (
            np.concatenate(normal_tokens, axis=1).astype(np.float32),
            np.concatenate(heat_tokens, axis=1).astype(np.float32),
        )
