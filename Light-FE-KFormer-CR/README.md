# Light-FE-KFormer-CR

This project is a cleaned, lightweight redesign of the uploaded FE-KFormer core scripts for **urban green-space cooling resilience** under normal-hot vs heatwave conditions.

## Model lineage

Original idea retained:

`OpenFE -> AutoEncoder -> Learnable Keypoints -> Self/Cross Attention -> Regression`

Modified task:

`Normal state + Heatwave state -> shared KFormer -> CI_normal + CI_heat + CR`

with the consistency constraint:

`CR ~= CI_heat / CI_normal`.

## Five semantic tokens

1. `park`: `area`, `shape_index`, `green_ratio`, `ndvi_park`
2. `buf100`: `ndvi_100`, `ndbi_100`, `building_100`, `green_100`
3. `buf300`: `ndvi_300`, `ndbi_300`, `building_300`, `green_300`
4. `buf500`: `ndvi_500`, `ndbi_500`, `building_500`, `green_500`
5. `weather`: `temperature`, `wind_speed`, `solar_radiation`

For any feature, a state-specific column such as `temperature_normal` / `temperature_heat` is preferred. If no suffixed column exists, the unsuffixed column is treated as static and shared by both states.

## Required target/ID columns

- `park_id`: park/group identifier used for leakage-safe train/validation/test splitting
- `pair_id`: optional but recommended paired-observation identifier
- `ci_normal`: observed normal-hot cooling intensity (deg C)
- `ci_heat`: observed heatwave cooling intensity (deg C)

`cr` is computed as `ci_heat / ci_normal`; rows with unstable small `ci_normal` are filtered before training.

## Important leakage controls

- Split by `park_id`, never by random rows.
- Fit OpenFE only on the training split.
- Fit clipping/scaling only on training rows.
- Fit each autoencoder only on training rows.
- Normal and heatwave states share the same OpenFE definitions, scaler, autoencoder, and KFormer encoder.

## Install

```bash
pip install -r requirements.txt
```

The `openfe` package is optional only if you run with `--no-openfe`. The full FE-KFormer lineage uses OpenFE.

## Synthetic smoke demo

```bash
python -m light_fe_kformer_cr.demo
```

The demo intentionally disables OpenFE so that the architecture can be smoke-tested even in an environment where `openfe` is unavailable.

## Train your paired Wuhan data

```bash
python train_from_csv.py wuhan_pairs.csv --output outputs
```

If OpenFE is not installed:

```bash
python train_from_csv.py wuhan_pairs.csv --no-openfe
```

## Files

- `data.py`: CR labels, group-wise split, leakage-safe robust scaling
- `openfe_stage.py`: semantic feature groups and optional OpenFE feature enhancement
- `autoencoder.py`: shared group-wise feature compression
- `tokens.py`: builds `[N, 5, D]` normal and heatwave semantic tokens
- `model.py`: shared KFormer encoder + differential multitask heads
- `losses.py`: CI/CR SmoothL1 losses + physical consistency loss
- `train.py`: loaders, training, validation and test metrics
- `train_from_csv.py`: end-to-end training entry point

## Verification note

The non-OpenFE path is covered by automated tests and a synthetic end-to-end demo. The OpenFE adapter follows the API used by the uploaded author's `openfe.py`, but it cannot be executed in environments where the optional `openfe` package is not installed.
