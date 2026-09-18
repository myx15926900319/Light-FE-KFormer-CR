# Light-FE-KFormer-CR Design

## Goal
Convert the uploaded FE-KFormer core scripts into a lightweight cooling-resilience model that preserves OpenFE feature enhancement, shared autoencoder compression, and learnable-keypoint attention while predicting normal cooling intensity, heatwave cooling intensity, and cooling resilience jointly.

## Data model
Each row is a paired observation for one park under a normal-hot state and a heatwave state. Required identifiers: `park_id`, `pair_id`. Targets: `ci_normal`, `ci_heat`; `cr` is either provided or computed as `ci_heat / ci_normal` after filtering unstable `ci_normal` values.

Feature groups are semantic tokens: `park`, `buf100`, `buf300`, `buf500`, `weather`. Normal and heatwave states use the same group definitions and the same fitted preprocessing/autoencoder parameters.

## Architecture
1. Group-wise feature enhancement: optional OpenFE is fit only on training rows, jointly over normal and heatwave states, with state-specific CI labels. Five semantic groups are retained instead of all 15 category combinations.
2. Group-wise autoencoders: one shared autoencoder per semantic group compresses enhanced features to an 8-dimensional token. Normal and heatwave samples share the same encoder.
3. Shared KFormer encoder: five tokens are processed by learnable keypoints, keypoint self-attention, and keypoint-to-token cross-attention. The encoder returns a fused latent vector and token similarities.
4. Paired differential head: shared encoder outputs `F_N` and `F_H`; `F_H-F_N` and `|F_H-F_N|` feed a resilience head. State-specific heads predict `CI_N` and `CI_H` from their own latent features.
5. Loss: SmoothL1 for three targets plus a consistency penalty enforcing `CR ≈ CI_H / CI_N`.

## Leakage controls
All clipping/scaling statistics, OpenFE feature generation, and autoencoder fitting are learned only from the training split. Splits are grouped by `park_id` so one park never occurs in multiple splits.

## Lightweight constraints
- Five semantic tokens only.
- Default bottleneck dimension: 8.
- Default learnable factors: 4.
- Attention heads: 2.
- No deep TransformerEncoder stack; preserve the original KFormer attention mechanism.
