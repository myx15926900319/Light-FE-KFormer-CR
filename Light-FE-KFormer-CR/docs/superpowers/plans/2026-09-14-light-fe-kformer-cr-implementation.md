# Light-FE-KFormer-CR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested, runnable Light-FE-KFormer-CR package from the uploaded FE-KFormer architecture.

**Architecture:** Preserve FE-KFormer's OpenFE → AutoEncoder → learnable-keypoint attention lineage, but use five semantic multiscale tokens and paired normal/heatwave encoding. Predict `ci_normal`, `ci_heat`, and `cr` jointly and add a resilience consistency loss.

**Tech Stack:** Python 3.9+, PyTorch, NumPy, pandas, scikit-learn; optional OpenFE for supervised feature generation.

**Spec:** `docs/superpowers/specs/2026-09-14-light-fe-kformer-cr-design.md`

## Global Constraints
- Group split by `park_id`.
- Training-only fit for all preprocessing.
- Normal and heatwave states share preprocessing, autoencoders, and KFormer encoder.
- Five tokens: park, buf100, buf300, buf500, weather.
- Default token dimension 8, four keypoints, two attention heads.

---

### Task 1: Data schema and leakage-safe preprocessing
**Files:** Create `light_fe_kformer_cr/data.py`; Test `tests/test_data.py`.
**Interfaces:** `compute_cr`, `group_split`, `RobustStandardizer`.
- [ ] Write tests for CR computation, unstable CI filtering, grouped split disjointness, and train-only scaling.
- [ ] Run tests and confirm failure before implementation.
- [ ] Implement minimal data utilities.
- [ ] Run tests and confirm pass.

### Task 2: Shared autoencoder token compressor
**Files:** Create `light_fe_kformer_cr/autoencoder.py`; Test `tests/test_autoencoder.py`.
**Interfaces:** `AutoEncoder`, `fit_autoencoder`, `encode_array`.
- [ ] Write shape and reconstruction-path tests.
- [ ] Run failing tests.
- [ ] Implement autoencoder and helpers.
- [ ] Run passing tests.

### Task 3: KFormer encoder and paired CR model
**Files:** Create `light_fe_kformer_cr/model.py`; Test `tests/test_model.py`.
**Interfaces:** `KFormerEncoder`, `LightFEKFormerCR`.
- [ ] Write tests for `[B,T,D]` input, similarity shape, shared encoder, and three positive predictions.
- [ ] Run failing tests.
- [ ] Implement the encoder and multitask model.
- [ ] Run passing tests.

### Task 4: Cooling-resilience multitask loss
**Files:** Create `light_fe_kformer_cr/losses.py`; Test `tests/test_losses.py`.
**Interfaces:** `CoolingResilienceLoss`.
- [ ] Write tests for finite loss and near-zero consistency on ratio-consistent predictions.
- [ ] Run failing tests.
- [ ] Implement SmoothL1 multitask + consistency loss.
- [ ] Run passing tests.

### Task 5: Optional OpenFE semantic-group stage
**Files:** Create `light_fe_kformer_cr/openfe_stage.py`; Test `tests/test_openfe_stage.py`.
**Interfaces:** `FEATURE_GROUPS`, `OpenFEGroupEnhancer`, state-column mapping helpers.
- [ ] Test semantic group extraction independently of optional OpenFE dependency.
- [ ] Run failing tests.
- [ ] Implement extraction and optional OpenFE adapter with explicit dependency error.
- [ ] Run passing tests.

### Task 6: End-to-end trainer and synthetic smoke test
**Files:** Create `light_fe_kformer_cr/train.py`, `light_fe_kformer_cr/demo.py`, `requirements.txt`, `README.md`; Test `tests/test_smoke.py`.
**Interfaces:** `train_one_epoch`, `evaluate`, synthetic token-pair demo.
- [ ] Write smoke test that performs one optimizer step and finite loss.
- [ ] Run failing test.
- [ ] Implement training/evaluation helpers and demo.
- [ ] Run entire test suite.
