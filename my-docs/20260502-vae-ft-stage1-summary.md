# VAE Finetune Stage-1 Summary & Stage-2 Action Guide

**Date**: 2026-05-02
**Branch**: `vae-finetune`
**Run**: `results/coart_feat18_20260423_three_branch_ws_v0/` (steps 0–159300)
**Analysis ckpt**: `ema_0.9999_dec_step0155000.pt` (+ adjacent online ckpts at 145k / 150k)
**Audience**: future-self resuming, plus collaborator code review

---

## 0. Executive summary (TL;DR)

1. **Stage-1 produced a usable VAE.** EMA decode of 8 golden assets gives mean CD = 6.6e-6, mean NC = 0.95; mesh→corep→latent→corep→mesh round-trip works end-to-end.
2. **One of the two new heads is dead weight.** Diagnostic ablation shows `p2_head` zeroed has < 1% CD impact on every golden asset and `oracle_p2` does not improve geometry — the 18-channel three-branch design wasted ~6 % of IO parameters on a path that 97.79 % of cubes never exercise.
3. **The other new head learned but is undertrained.** `ef_head` is causally essential (zeroing it produces empty mesh on 8/8 assets), but `oracle_ef` cuts `n_components` by 60–99 % per asset — i.e. predicted ef is good enough to *build* a mesh, far from good enough for *clean* topology.
4. **Training overshot the optimum step.** Best `val_recon` was hit at **step 110000 (0.189)**; from step 75k onwards `train_kl` climbs monotonically 15.2 → 52.8 (3.5×) under `lambda_kl=1e-6`. The 155k ckpt is past peak but not catastrophic — it should be replaced by step ≈ 110k–130k as Stage-1's "best" artifact for downstream DiT.
5. **Stage-2 priorities**: (a) drop or fix `p2`; (b) add conditional + topology-aware loss for `ef`; (c) raise `lambda_kl` and cap `max_steps` ≤ 130k or add early stopping; (d) re-run the same diagnostic pipeline on the new ckpt to confirm fixes.

---

## 1. Background & motivation

### 1.1 Why a finetune at all
TRELLIS.2 ships a Shape-VAE pretrained on 6-channel features `(vertex 3 + intersected 3)`. The CoReP pipeline (`corep_fast`) consumes/produces an 18-channel feature `[p1 (3), p2 (3), edge_weights (6), face_weights (6)]` on the same SC-VAE backbone. Stage-1 finetuned the pretrained shape VAE so that

```
mesh ──corep_fast──► 18-ch feats ──Encoder──► latent z ──Decoder──► 18-ch pred ──feature_to_mesh──► mesh'
```

is a self-consistent loop with bounded geometry loss, enabling downstream DiT training over the latent.

### 1.2 The 18-channel layout (verified by 3-way independent scan, 131M cubes)
| Channels | Semantics | Distribution |
|---|---|---|
| 0–2 | `p1_xyz` (lower-z representative point) | always present, mean (0.5, 0.5, 0.5), std 0.23 |
| 3–5 | `p2_xyz` (higher-z point) | **= [0,0,0] in 97.79 % of cubes**; non-zero only on 2.21 % |
| 6–11 | `edge_weights` (ordinal int counts) | 99.1 % ∈ {0, 1}, range 0..22 |
| 12–17 | `face_weights` (ordinal int counts) | **99.83 % = 0**, range 0..5 |

This sparsity is the dominant fact of life for Stage-1 design and the root cause of the p2 collapse described in §6.

---

## 2. Architecture: three-branch IO

### 2.1 The design choice (decided 2026-04-22; see `logs/findings_feat18_vae_launch.md`)

The pretrained encoder/decoder use a single `sp.SparseLinear(6 → C₀)` / `(C_end → 7)` IO. The 18-channel payload has three semantic blocks, so we split IO into three branches:

```
Feat18EncIO                              Feat18DecIO
─────────                                ─────────
p1_branch  : Linear(3 → C₀)              p1_head : Linear(C_end → 3)
p2_branch  : Linear(3 → C₀)              p2_head : Linear(C_end → 3)
ef_branch  : Linear(12 → C₀)             ef_head : Linear(C_end → 12)

forward(x):                              forward(x):
  h = p1(x[:,0:3]) + p2(x[:,3:6]) +        out = concat(p1(x), p2(x), ef(x))
      ef(x[:,6:18])
```

Defined in `coart/vae/io_stems.py`; constructed by `coart/vae/build.py:build_models(io_arch="three_branch")`.

### 2.2 Warm-start strategy (`coart/vae/build.py:_apply_warmstart_three_branch`)

| Linear | Init source |
|---|---|
| `p1_branch` | `pretrained.input_layer.weight[:, 0:3]`, full bias |
| `p2_branch` | `pretrained.input_layer.weight[:, 0:3]`, **bias = 0** |
| `ef_branch` | xavier_uniform (no warm-start) |
| `p1_head` | `pretrained.output_layer.weight[0:3, :]`, full bias |
| `p2_head` | `pretrained.output_layer.weight[0:3, :]`, **bias = 0** |
| `ef_head` | xavier_uniform (no warm-start) |

Backbone weights load via `strict=False` (skipping `input_layer.*` / `output_layer.*` shape mismatches).

### 2.3 Why these choices and what they imply

- **Three independent branches** preserve z-sort signal — no permutation ambiguity from sharing weights between `p1` and `p2`.
- **Warm-starting both point branches** from the same pretrained vertex column halves the angular coverage but biases p1/p2 towards the pretrained representation manifold; xavier ef would otherwise have to compete for limited finetune capacity.
- **`p2.bias = 0`** was the critical *defect* in retrospect: when input p2 is zero (97.79 % of cubes), the only signal `p2_branch` produces is its bias; if bias starts at zero and only gets gradient signal from the rare 2.21 % multi-point cubes, the branch effectively never has a meaningful contribution to the encoder sum. This is what §6 measured.

### 2.4 What did NOT change vs. native pretraining

- Backbone (5-stage SparseConvNeXt blocks `(0, 4, 8, 16, 4)`, model channels `(64, 128, 256, 512, 1024)`, latent_channels=32).
- KL prior (standard normal), subdivision prediction heads at decoder levels, EMA mechanics.
- `compute_vae_loss` block decomposition: `total = recon + lambda_kl * kl + lambda_subdiv * subdiv`, where `recon = MSE(pred, target)` over all 18 channels uniformly.

---

## 3. Stage-1 training configuration

From `results/coart_feat18_20260423_three_branch_ws_v0/config.json`:

| Group | Setting | Value | Note |
|---|---|---|---|
| Pretrained | `enc_pretrained` | `microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16` | |
| | `dec_pretrained` | `microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16` | |
| IO | `io_arch` / `warmstart_io` | `three_branch` / `True` | §2 |
| Optim | `lr` | `1e-5` | aligned with native ft-512 |
| | `lr_unfreeze_warmup_steps` | 500 | linear ramp from 0 |
| | `freeze_backbone_steps` | 2000 | only IO trainable for first 2k |
| | `grad_clip_max` / `grad_clip_pct` | 1.0 / 95 | adaptive clipper |
| | `use_bf16` | True | autocast |
| Data | `batch_size` | 1 | bs=1 per GPU × 8 GPU = 8 effective |
| | `bucket_sampler` / `max_voxels` | True / 500000 | OOM guard |
| | `max_translate` | 16 | augmentation (voxel units) |
| | `val_split_mod` | 200 | sha mod 200 → 209 held-out items |
| Loss | `lambda_kl` | **1e-6** | (Stage-2 candidate to raise) |
| | `lambda_subdiv` | **0.1** | (Stage-2 candidate to raise) |
| Schedule | `max_steps` | 200000 | actually ran to ~159300 (preempted) |
| | `i_log` / `i_save` / `i_val` | 100 / 5000 / 5000 | |
| | `first_deep_eval_step` | 10000 | golden asset eval cadence |
| EMA | `use_ema` / `ema_rate` | True / 0.9999 | |
| | `rolling_ckpts` / `rolling_ckpts_ema` | 5 / 1 | |

Hardware: 8 × H100/A100 on `host-10-240-99-117` (training) + `host-10-240-99-119` (analysis).

---

## 4. Training trajectory (steps 100 → 159300)

### 4.1 Per-block recon and KL (sampled)

| Step | val_recon | train_recon | train_recon_p1 | train_recon_p2 | train_recon_ef | train_kl |
|---:|---:|---:|---:|---:|---:|---:|
| 5000 | 0.614 | 0.315 | 0.029 | 0.018 | 0.462 | 21.19 |
| 25000 | 0.436 | 0.223 | 0.020 | 0.009 | 0.328 | 19.04 |
| 50000 | 0.294 | 0.230 | 0.018 | 0.008 | 0.340 | 15.11 |
| 75000 | 0.214 | 0.192 | 0.014 | 0.005 | 0.284 | **15.21** (KL min) |
| 100000 | 0.190 | 0.144 | 0.011 | 0.005 | 0.213 | 16.94 |
| **110000** | **0.189** (val min) | 0.266 | 0.012 | 0.008 | 0.394 | 21.30 |
| 120000 | 0.191 | 0.227 | 0.013 | 0.006 | 0.336 | 26.86 |
| 122500 | — | **0.062** (train min) | — | — | 0.091 | 28.11 |
| 130000 | 0.196 | 0.233 | 0.013 | 0.007 | 0.345 | 33.13 |
| 140000 | 0.195 | 0.066 | 0.009 | 0.003 | 0.096 | 43.43 |
| 150000 | 0.197 | 0.099 | 0.010 | 0.006 | 0.145 | 48.64 |
| 155000 | 0.196 | 0.120 | 0.012 | 0.006 | 0.176 | **52.80** |
| 159300 | — | 0.299 | 0.015 | 0.009 | 0.442 | 57.84 |

### 4.2 Headline observations

- **`val_recon` is flat from step 110k onwards** (0.189 → 0.196). Not classical "val loss going up" overfitting, but **convergence + drift**.
- **`train_kl` is monotonically blowing up** from step 75k: 15.2 → 52.8, a 3.5× increase. With `lambda_kl=1e-6`, the KL contribution to `total` is < 6e-5 — the optimizer has effectively no incentive to keep the encoder posterior close to the prior. This is the *signature* of an unconstrained VAE encoder freely using its 32 latent channels.
- **`train_recon` is highly noisy** post-100k (0.062 to 0.299 swings between consecutive log points) due to bs=1, bf16 noise, and the KL trade-off. *Train loss is not a reliable convergence indicator at this batch size.*
- **`recon_ef` dominates `recon_total`** by ~30×: `recon_ef ≈ 0.18` vs `recon_p1 ≈ 0.012` at the late ckpts. With the uniform MSE summation, ef channels dictate recon convergence.
- **`recon_p2` keeps slowly improving** (min 0.0027 at step 156300), but starting from 0.018 at step 5000 it only ever drops 6.7×, vs. the ef branch's ~5× drop. p2 is "trained" by gradient but the signal is dilute — see §6 for why this convergence is mostly cosmetic.

### 4.3 Deep-eval geometry (golden 8) at logged checkpoints

| Step | mean_cd | mean_nc | helmet_cd | val_p95_cd | val_p95_nc |
|---:|---:|---:|---:|---:|---:|
| 50100 | — | — | — | — | 0.538 (NC min) |
| 100100 | — | — | **1.2795e-05** (helmet min) | — | — |
| 145100 | — | — | — | — | **0.99135** (NC max) |
| 155100 (latest) | 6.56e-06 | 0.947 | 1.30e-05 | **7.41e-06** (val_p95 CD min) | 0.991 |

Geometry is *stable through 155k* — eval-time degradation is not visible. The KL drift hurts the encoder representation (Stage 2 DiT will care) more than the immediate reconstruction.

### 4.4 The "best ckpt" verdict by metric

| Metric | Argmin step | Value |
|---|---:|---:|
| `val_recon` | 110000 | 0.189 |
| `helmet/cd` | 100100 | 1.2795e-05 |
| `train_kl` | 51100 | 14.70 |
| `aggregate/mean_cd` (extrapolated) | ≈100k–130k | (stable) |

**Recommended Stage-1 deliverable for downstream DiT**: re-evaluate the **EMA at step 110k–130k** vs. step 155k on the full 8-asset golden set + a held-out val subset, pick the better one. The 155k EMA is what we have analyzed and it is *not* catastrophic, just suboptimal.

---

## 5. Diagnostic methodology (this analysis)

### 5.1 Question
Did finetuning produce *effective* outputs for the new IO parameters (`p2_branch / p2_head / ef_branch / ef_head`), or did training collapse them to constants? Pure recon-loss tracking cannot answer this — both heads can achieve near-zero MSE by emitting the channel mean (e.g. `face_weights` is 0 in 99.83 % of voxels).

### 5.2 Three orthogonal probes (`docs/superpowers/specs/2026-05-02-vae-finetune-effectiveness-analysis-design.md`)

#### A. Static weight drift
For each of the 6 IO Linears, compute against (i) reconstructed step-0 init under the same seed and (ii) ckpts at step 145k / 150k:
- `rel_frob_drift = ‖W_now − W_ref‖_F / ‖W_ref‖_F`
- `rms_elem_drift = ‖W_now − W_ref‖_F / √numel`
- `bias_drift_per_dim = ‖b_now − b_ref‖₂ / √d_out`
- top-5 singular values, **effective rank** `exp(H(σ̄))`, condition number
- `mean_row_cosine(W_now, W_ref)` row-by-row

**Interpretation**: small drift vs. step-0 + small bias_drift = *dead*; large drift + slope still non-trivial between 145k→155k = *learning*; mid-range plateau = *plateau, defer to functional probe*.

#### B. Activation & per-channel statistics
Forward 200 val items in fp32 (bf16 off, deterministic):
- Hook `Feat18EncIO` to record per-branch contribution `‖p1_branch(f[:,0:3])‖`, `‖p2_branch(f[:,3:6])‖`, `‖ef_branch(f[:,6:18])‖`, partitioned by *signal-bearing* mask (p2 != 0 or ef != 0).
- Per-channel pred-vs-target stats (denormalised): pearson_r, MSE on all voxels, **MSE conditional on target ≠ 0** (the only honest MSE — eliminates the "predicting zero gives free MSE" artefact for sparse channels), zero-rate gap.

**Interpretation**: pearson_r < 0.05 + pred_zero_rate ≥ target_zero_rate + 0.5 → dead; 0.05 ≤ r < 0.3 → undertrained; r ≥ 0.3 → alive.

#### C. Causal head ablation on golden assets
On each of 8 golden assets, run encoder → mu (no sampling) → decoder → 18-ch pred, then substitute *in normalised space* one of:

| Condition | Substitution |
|---|---|
| `full` | none |
| `zero_ef` | `pred[:, 6:18] := 0` |
| `zero_p2` | `pred[:, 3:6] := 0` |
| `oracle_ef` | `pred[:, 6:18] := target[:, 6:18]` |
| `oracle_p2` | `pred[:, 3:6] := target[:, 3:6]` |

Then `denormalize → feature_to_mesh → corep_to_exp5_vertices → CD / NC / F-score / topology`. The five-way comparison directly answers: does the head contribute? does it learn the right thing?

**Interpretation**: `|ΔCD_zero| < 5 %` → dead (zeroing changed nothing); `ΔCD_zero ≥ 5 %` AND `ΔCD_oracle ≤ -30 %` → undertrained (oracle dramatically improves); else → alive. *Special-cased*: if `zero_X` produces empty mesh on every asset, the head is causally essential → alive (this special case was added during analysis when smoke results showed all-zero_ef-empty-mesh — see commit `2bf0554`).

### 5.3 Implementation
- Pure helpers + orchestration in `coart/analysis/{weight_drift, activation_stats, head_ablation, report}.py` — additive, no modification of `coart/vae/` or `coart/eval/`.
- CLI `scripts/coart_analyze_finetune.py` glues the three probes + report assembly.
- 8-GPU sharded sweep via `scripts/run_analyze_finetune_8gpu.sh` (each rank handles `n_val // 8` items + `assets[rank::8]`; rank 0 also runs Section A; merge step concatenates and writes `report.md`).
- 26 CPU unit tests for the pure helpers; smoke + full e2e validation on `host-10-240-99-119`.

---

## 6. Diagnostic findings @ step 155k

### 6.1 TL;DR table (auto-generated)

| Head | Drift | Functional | Causal | Combined |
|---|---|---|---|---|
| `p2_head` | undertrained | alive (per-channel r 0.41–0.46) | **dead** | **DEAD** |
| `ef_head` | alive | alive | alive | **ALIVE (undertrained)** |

### 6.2 Static weight drift vs. step-0

| Linear | rel_frob_drift | mean_row_cosine | bias_drift | eff_rank / max | verdict |
|---|---:|---:|---:|---:|---|
| `p1_branch` | 0.644 | 0.716 | 0.098 | 2.99 / 3 | full-range learning |
| `p2_branch` | 0.472 | 0.875 | 0.098 | 2.98 / 3 | learning, kept warm-start direction |
| **`ef_branch`** | **1.471** | **−0.052** | 0.285 | 11.69 / 12 | **rotated to ~orthogonal direction; full-rank** |
| `p1_head` | 0.400 | 0.918 | 0.025 | 2.99 / 3 | full-range learning |
| `p2_head` | 0.662 | 0.776 | 0.080 | 2.99 / 3 | learning, partial rotation |
| **`ef_head`** | **4.945** | **0.002** | 0.347 | 10.72 / 12 | **fully relearned from xavier; full-rank** |

Drift between step 145k→155k for every Linear is < 2 %, confirming convergence (the model has plateaued, not exploded).

**Reading**:
- `ef_branch / ef_head` are *not collapsed*. row_cosine ≈ 0 means rows rotated into completely new directions (xavier rows are random anyway, so this just says learning happened); rel_frob ~ 5 means the magnitude is 5× the random init scale.
- `p2_branch / p2_head` show moderate drift but `mean_row_cosine ~ 0.78–0.88` — they kept most of the warm-start (pretrained vertex) direction.

### 6.3 Activation statistics over 200 val items (≈84 M voxels)

#### Encoder branch contribution norms (mean ‖branch(input)‖ per voxel)

| Branch | norm_all | norm_signal | norm_zero | n_signal | n_zero |
|---|---:|---:|---:|---:|---:|
| `p1_branch` | 1.13 | 1.13 | — | 84.7 M | 0 |
| **`p2_branch`** | 1.83 | **0.85** | **1.84** | 1.35 M | 83.3 M |
| `ef_branch` | 3.91 | 3.99 | 2.89 | 78.8 M | 5.9 M |

**Crucial finding**: `p2_branch` produces a *larger* contribution on voxels where input p2 is *zero* (1.84) than on voxels where p2 is non-zero (0.85). This is exactly the signature of "branch is essentially a constant bias" — when input is zero, output is `bias` (norm 1.84); when input is non-zero, the linear combination accidentally produces something smaller. Compared to `ef_branch`, where signal voxels produce *larger* output (3.99 vs 2.89), `p2_branch` is functionally degenerate.

#### Per-channel pred-vs-target (key channels)

| ch | block | pearson_r | mse_overall | mse_conditional | pred_zero | target_zero | label |
|---:|---|---:|---:|---:|---:|---:|---|
| 0 | p1 | **0.90** | 0.010 | 0.010 | 0.011 | 0.037 | excellent |
| 1 | p1 | 0.89 | 0.011 | 0.010 | 0.010 | 0.037 | excellent |
| 2 | p1 | 0.90 | 0.011 | 0.010 | 0.010 | 0.039 | excellent |
| 3 | p2 | **0.41** | 0.004 | **0.18** | 0.880 | 0.985 | **noisy on signal** |
| 4 | p2 | 0.42 | 0.004 | 0.18 | 0.871 | 0.985 | noisy on signal |
| 5 | p2 | 0.46 | 0.005 | 0.23 | 0.866 | 0.984 | noisy on signal |
| 6 | edge | **0.94** | 0.026 | 0.05 | 0.466 | 0.683 | very good |
| 9 | edge | 0.94 | 0.033 | 0.04 | 0.235 | 0.446 | very good |
| 12 | face | 0.83 | 5e-4 | **0.24** | 0.994 | 0.998 | good but conditional MSE high |
| 17 | face | 0.83 | 5e-4 | 0.21 | 0.994 | 0.998 | good but conditional MSE high |

**Reading**:
- p1 channels are excellently learnt (r ≈ 0.90, conditional MSE ≈ 0.01).
- **p2 channels score r ≈ 0.41–0.46** with **conditional MSE 0.18–0.23** (i.e. on the rare voxels where p2 ≠ 0, the prediction is essentially random — std target ≈ 0.07, MSE 0.18 ≫ random-baseline level).
- **edge channels** (ch 6–11): r ≈ 0.94, conditional MSE 0.04–0.05 — well learnt.
- **face channels** (ch 12–17): r ≈ 0.80–0.85 overall, but conditional MSE 0.20–0.28 on ef ≠ 0 voxels — pred is correct *most* of the time because most targets are zero, but on the rare non-zero voxels predictions are off. This is what `oracle_ef` compensates for in the causal probe.

### 6.4 Causal head ablation (8 golden assets, 5 conditions)

| Asset | Condition | CD | n_components | Notes |
|---|---|---:|---:|---|
| helmet | full | 1.29e-5 | **602 688** | |
| helmet | zero_ef | — | — | **mesh_empty** |
| helmet | zero_p2 | 1.29e-5 (Δ +0.4 %) | 602 688 | identical |
| helmet | oracle_ef | 1.27e-5 (Δ −1.5 %) | **222 792** (−63 %) | topology cleaner |
| helmet | oracle_p2 | 1.28e-5 (Δ −0.7 %) | 602 741 | identical |
| triple_sphere | full | 1.65e-5 | 77 | |
| triple_sphere | zero_ef | — | — | mesh_empty |
| triple_sphere | oracle_ef | 1.66e-5 | **13** (−83 %) | |
| val_p25 | full | 2.11e-6 | 325 | |
| val_p25 | zero_ef | — | — | mesh_empty |
| val_p25 | oracle_ef | 2.13e-6 | **2** (−99 %) | clean topology |
| val_p40 | full | 3.11e-6 | 89 | |
| val_p40 | zero_ef | — | — | mesh_empty |
| val_p40 | oracle_ef | 3.11e-6 | **2** (−98 %) | clean |
| val_p80 | full | 2.49e-6 | 46 | |
| val_p80 | oracle_ef | 2.49e-6 | **1** | water-tight-ish |

Pattern (consistent across all 8 assets):
- `zero_ef` → **`feature_to_mesh` returns empty mesh in 8/8 cases**. Maximum-entropy "alive" signal: ef channels are causally indispensable for mesh extraction.
- `zero_p2` → identical CD/NC/F-score/topology to `full`. p2_head output is unused by `feature_to_mesh` for these assets (most are single-point cubes; the rare double-point asset `val_p10` shows < 2 % difference).
- `oracle_ef` → **`n_components` drops 60–99 %** depending on asset, while CD changes < 2 %. ef_head learns *vertex placement* well (CD reflects this) but *connectivity / topology* poorly.
- `oracle_p2` → no measurable improvement on any metric.

### 6.5 Combined verdict

- **`ef_head` and `ef_branch`**: alive, full-rank, Pearson r ≥ 0.83 per channel, causally indispensable for mesh extraction. Topology gap vs. oracle is huge — the head needs targeted training improvement, not architecture change.
- **`p2_head` and `p2_branch`**: dead by every causal measure. Functional pearson_r non-zero (0.41) only because 98.5 % of targets are zero and predicting near-zero gets you r > 0 trivially; on the rare non-zero voxels, predictions are essentially noise (cond MSE 0.18–0.23). The encoder branch behaves like a constant bias (norm on signal voxels < norm on zero voxels). The decoder head's output is ignored by `feature_to_mesh` for all 8 golden assets.

---

## 7. Stage-1 lessons learned

### L1. p2 was dead by *design*, not by training failure
- **Cause**: 97.79 % of cubes are single-point (p2 = 0). The `p2.bias = 0` warm-start blocks any signal at zero input. The branch only ever sees gradient through the rare 2.21 % multi-point cubes — and at bs=1 with 8 GPUs, that's ~0.18 effective batches per step on average where p2 actually backpropagates a useful signal. With 159k steps × 0.18 ≈ 29k effective gradient updates for p2, vs ~159k for p1/ef.
- **Symptom**: low conditional MSE didn't show up because non-zero p2 supervision is rare; cosmetic per-block recon convergence (recon_p2 at 0.0027–0.009) hid the fact that the function being learnt was approximately "constant".
- **Implication**: any p2 redesign in Stage-2 must either (a) provide stronger supervision per-voxel-where-it-matters, or (b) just remove p2 and treat all cubes as single-point.

### L2. ef was alive but topology-blind
- **Cause**: `compute_vae_loss` uses uniform `F.mse_loss(pred, target)` over all 18 channels. With 99.1 % of edge values in {0, 1} and 99.83 % of face values = 0, the channel-MSE budget is dominated by *zero targets*. A model that predicts ~0 everywhere achieves a globally tiny MSE while utterly missing the rare non-zero edges/faces.
- **Symptom**: per-channel pearson_r is high (≥ 0.83) because correlation is invariant to the zero-prediction trick; but conditional MSE on ef ≠ 0 voxels is 0.04–0.28; oracle_ef cuts n_components by 60–99 %.
- **Implication**: Stage-2 must add *conditional* loss terms (weight non-zero ef voxels 5–10×) or topology-aware regularization (n_components / Euler delta from GT), or both.

### L3. Unconstrained KL
- **Cause**: `lambda_kl = 1e-6` is the inherited default from native ft-512, intended for an already-pretrained encoder representation. After three-branch IO surgery, the encoder is freshly mapped from 18 channels and re-learns its representation — it has no incentive to keep `mu, logvar` near the prior under this tiny λ.
- **Symptom**: `train_kl` from 15.2 (step 75k) → 52.8 (step 155k), monotonic.
- **Implication**: Raise `lambda_kl` toward 1e-4 (or anneal). Downstream DiT training over the latent will be sensitive to encoder posterior drift.

### L4. Plateau ≠ stop
- Best `val_recon` at **step 110k**; from step 100k to 159k val_recon stays in [0.189, 0.197]. The training loop kept running another 50k+ steps with no eval improvement, while KL drifted. This is wasteful and slightly harmful.
- **Implication**: enable early stopping or cap `max_steps`. A 130k-step Stage-2 run probably yields the same eval as 200k.

### L5. Per-block loss decomposition was useful, but not enough
- Stage-1 already logs `recon_p1 / recon_p2 / recon_ef` separately. This was the only reason we could even see that ef dominates and p1 / p2 converge fast.
- **However**, none of these per-block losses revealed the dead-p2 / topology-blind-ef problems — they require *causal* probing (zero / oracle ablation) and *conditional* statistics (conditional MSE on ef ≠ 0).
- **Implication**: continue per-block logging in Stage-2; **add periodic in-loop eval of conditional MSE** (cheap) and **periodic head-ablation deep-eval** (run our diagnostic CLI at every i_save × 5 cadence).

### L6. Infrastructure is solid (after fighting 5 critical bugs early on)
- **Final state**: resume / atomic save / EMA + online-split rolling ckpts / wandb auto-resume / bucket sampler / max_voxels cap all work across multiple session restarts. No data loss, no divergent runs (after the date-drift bug fixed earlier on the branch), no OOMs.
- **Path to here**: 5 production-blocking bugs were navigated during launch (April 2026). Each cost hours-to-days of debug time and the *fixes* are non-obvious; they should NOT be re-derived in Stage-2. Full DDP / collective config preserved in **Appendix A**; the bug summary table is below for posterity.

| # | Symptom | Root cause | Fix commit | Lesson |
|---|---|---|---|---|
| 1 | resume "optimizer param-group size mismatch" | base mode saved with all params unfrozen; resume rebuilt optim while still frozen | `db437ad` | resume must peek `misc_step*.pt` to recover trainable state before optim init |
| 2 | NCCL ALLREDUCE timeout (SeqNum=224, NumelIn=18 vs 34962) | `logger.flush_if_due` called all-reduce only on rank-0; other ranks early-exit → shape-mismatched collectives forever | `ecf30e2` | every collective op must run on every rank; logger went rank-0-local |
| 3 | step 5342 epoch-boundary silent deadlock (all ranks stuck in `loss.backward()`) | DDP `find_unused_parameters=False` + `output_layer.ef_head` referenced multiple times in forward (final feats + subdivision head share path) → reducer marks param-ready twice → undefined behavior | `a86d9e9` (after `dc85c6f`, `10ced5b`) | sparse conv + multi-use param requires **`static_graph=True`**. `find_unused=True` raises clear error; `find_unused=False` deadlocks silently. Neither is acceptable without `static_graph`. |
| 4 | step-time wildly unstable under `bucket_sort_mode=shuffle` (0.5s ↔ 150s); triton autotune 60–70 % CPU long-tail | voxel count jumps across steps → triton recompiles new shapes constantly | `--bucket_sort_mode ascending` CLI | sampler.py docstring already noted: ascending is JIT-friendly; default was wrong |
| 5 | NCCL default 10-min timeout too tight for cold JIT / large-asset forward | inherent | `10ced5b` (1h timeout) | multi-rank cold JIT can take 15–50 min; default timeout misclassifies as fault |

The working DDP construction:
```python
DDP(model, device_ids=[local_rank], output_device=local_rank,
    bucket_cap_mb=128, find_unused_parameters=False, static_graph=True,
    gradient_as_bucket_view=True, broadcast_buffers=False)
dist.init_process_group("nccl", timeout=timedelta(hours=1))
```
- Same diagnostic methodology and 8-GPU launcher transfer to Stage-2 unchanged.

### L7. 8-GPU sharded analysis is fast
- Smoke (n_val=8): ~30 min wall (rank 0 stuck on helmet's mesh ablation).
- Full (n_val=200): ~30 min wall, dominated by helmet on rank 0.
- Without sharding, single-GPU full run would be ~3.5 h. Sharding gives ~7× speedup on this workload. Note: helmet is the worst-case asset (n_components ≈ 600k); future runs could distribute by `n_voxels`-balanced bin-packing instead of round-robin to flatten the rank-0 bottleneck.

---

## 8. Stage-2 action guide

Numbered for cross-reference. Each item lists the *change*, *why* (linked to a Stage-1 lesson), *expected effect*, and *risk*.

### 8.0 — Hard constraints from downstream DiT (must NOT break)

The Stage-1 VAE is the input to a downstream DiT finetune. Stage-2 changes must preserve the API contract so the DiT does not have to retrain from scratch:

| Invariant | Why | Stage-2 implication |
|---|---|---|
| `latent_channels = 32` | DiT's bottleneck dimension is fixed | do not change |
| `μ / σ` distribution near `N(0, I)` | DiT was conditioned to expect this prior | tighten KL (8.4) is *good*; don't introduce schemes that re-shape the latent (e.g. flow priors) |
| Decoder output 18-ch `[p1(3), p2(3), edge(6), face(6)]` | downstream `feature_to_mesh` expects this layout | **if 8.1 drops p2 → 15-ch output requires an upstream `corep_fast` change too**; treat as coordinated upgrade, not a Stage-2-internal-only change |
| EMA decoder is the published artifact | DiT ingests the EMA, not the online ckpt | `--use_ema --ema_rate 0.9999` stays |

**Stage-2 architectural changes (8.1) that touch the 18-ch interface require a coordinated `corep_fast` schema bump and a downstream DiT migration plan.** Do not ship a 15-ch decoder until the DiT side is ready.

### 8.1 — Decide p2 fate: **drop it** (recommended)
- **Why** L1: p2 is dead weight; 97.79 % of cubes never use it.
- **Change**: switch to a 2-branch IO `p_branch (3) + ef_branch (12)` with a 15-channel feat. Update `corep_fast` upstream to drop p2 (or pad with zeros). All single-point cubes (the typical case) are unaffected; multi-point cubes lose the second-vertex signal — re-evaluate whether multi-point cubes were ever benefiting (the 2.21 % minority).
- **Effect**: -2 Linears (~150k params), simpler loss, no dead branch dragging gradient noise into the encoder sum.
- **Risk**: multi-point cubes' geometry could degrade. Mitigation: keep p2 alive by Plan B below if degradation > 5 % CD on multi-point val subset.
- **Plan B (if p2 must stay)**: replace `p2.bias = 0` warm-start with the non-zero pretrained bias (already used for p1) **and** add a permutation-invariant Chamfer-2 supervision term on multi-point cubes only. Treat 2.21 % cubes with 50× per-cube loss weight.

### 8.2 — Add conditional MSE on ef channels
- **Why** L2: uniform MSE rewards predicting zero; conditional MSE on ef ≠ 0 voxels is 0.04–0.28.
- **Change** (in `coart/vae/loss.py:compute_vae_loss`):
  ```python
  ef_signal_mask = target[:, 6:18].abs().sum(dim=1) > 0  # voxel has any non-zero ef
  recon_ef_cond = F.mse_loss(pred[ef_signal_mask, 6:18], target[ef_signal_mask, 6:18]) \
                  if ef_signal_mask.any() else torch.zeros((), device=pred.device)
  total = recon + lambda_kl * kl + lambda_subdiv * subdiv \
          + lambda_ef_cond * recon_ef_cond
  ```
  with `lambda_ef_cond` between 0.5 and 2.0 (start 1.0). Log `recon_ef_cond` separately.
- **Effect**: forces the model to learn ef on rare non-zero voxels; should reduce oracle_ef topology gap (Stage-1: oracle drops n_components 60–99 %; Stage-2 target: ≤ 30 %).
- **Risk**: signal voxels are sparse → noisy gradient. Bs=1 makes this worse. Mitigation: bs ≥ 2 if possible, or accumulate gradients over 2 steps.

### 8.3 — Add topology-aware regularization (optional, conditional on 8.2's gain)
- **Why** L2: Stage-1 ef predictions give "good vertices, bad topology" — the loss has no signal that connectivity matters.
- **Change**: at the loss layer, periodically (every K steps) decode a small minibatch through `feature_to_mesh`, compute `n_components` and `euler`, add `lambda_topo * |n_components_pred − n_components_gt| / n_components_gt` (clipped). Sparse signal but directly aligned with the failure mode. Or — simpler — penalize the L1 distance between predicted ef-channel-zero patterns and target zero patterns (forces the *placement* of non-zero ef to match target, not just the magnitudes).
- **Effect**: cleaner topology after Stage-1 saw oracle_ef drop n_components 60–99 %.
- **Risk**: `feature_to_mesh` is non-differentiable, so the topology metric goes through a regularizer-only path (no backprop through mesh). The L1-zero-pattern alternative *is* differentiable through ef channels and is the safer first try.

### 8.4 — Raise `lambda_kl` to 1e-4 (10× of Stage-1)
- **Why** L3: KL drifted 3.5× during Stage-1. Downstream DiT will see this as drift in latent statistics.
- **Change**: `--lambda_kl 1e-4`. If you observe under-fitting at step 30–50k (recon stops dropping early), reduce to 5e-5. If KL still drifts, anneal: start at 1e-6, cosine ramp to 1e-4 by step 50k.
- **Effect**: KL stays bounded < 25 (Stage-1 step-75k baseline). Encoder posterior is stable. Downstream DiT training has well-conditioned latent.
- **Risk**: tighter KL means worse recon. Watch `val_recon` at step 50–100k — if it plateaus above 0.25 (vs Stage-1's 0.19), back off λ.

### 8.5 — Cap `max_steps` at 130000 (or use early stopping)
- **Why** L4: best val at step 110k; drift after that.
- **Change**: `--max_steps 130000`, OR add an early-stop hook: if `val_recon` hasn't improved by > 0.5 % in 25k steps, stop.
- **Effect**: 35 % less compute, same artifact quality.
- **Risk**: nil; can always resume past 130k if metrics are still moving.

### 8.6 — Bump `lambda_subdiv` from 0.1 to 0.3
- **Why** subdiv prediction is closely tied to ef channels (they both encode "is there a face / edge here") and a stronger subdiv signal indirectly helps ef.
- **Change**: `--lambda_subdiv 0.3`.
- **Effect**: cleaner subdiv predictions; possibly smaller oracle_ef gap.
- **Risk**: subdiv loss could dominate; back off if `recon` flatlines.

### 8.7 — Save best-by-val checkpoints separately
- **Change**: in `coart/vae/train.py`, when `val_recon` hits a new minimum, save a `best_val_step{S}.pt` ckpt (and corresponding EMA). Keep the rolling-K otherwise.
- **Effect**: explicit "best Stage-2 artifact" without manual ckpt cherry-picking.
- **Risk**: nil; just disk.

### 8.8 — Run the diagnostic pipeline at every i_save × 5 cadence
- **Why** L5: per-block loss is not enough; we need head-ablation periodically.
- **Change**: add a hook to training loop or a scheduled cron that runs `bash scripts/run_analyze_finetune_8gpu.sh ... --n_val 50 --skip_drift` every 25k steps after step 50k. Output goes into `analysis_step<S>/` per ckpt; results are diff-able across steps.
- **Effect**: catch late-stage regression on ef topology / p2-revival within one eval cycle instead of after the run finishes.
- **Risk**: 8-GPU 5-min interruption of training every 25k steps. Acceptable.

### 8.9 — Re-evaluate Stage-1 110k EMA before declaring 155k as Stage-1 deliverable
- **Why** L4: best val_recon at 110k; deep_eval helmet CD min at 100k.
- **Change**: dispatch a one-off deep-eval (existing `scripts/coart_ema_eval.py`) on the EMA at step 110000 and step 130000. Compare to step 155000.
- **Effect**: pick the actual best Stage-1 artifact for downstream DiT.
- **Risk**: nil; cheap.

---

## 9. Operational procedure (for collaborator review)

### 9.1 Run Stage-1 finetune (reference)
```
ssh host-10-240-99-117 "cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  torchrun --nproc_per_node 8 -m coart.vae \
    --run_tag <tag> --resume_from latest \
    --io_arch three_branch --warmstart_io \
    --lr 1e-5 --lambda_kl 1e-6 --lambda_subdiv 0.1 \
    --max_steps 200000 \
    --bucket_sampler --max_voxels 500000 \
    --use_ema --ema_rate 0.9999 \
    --use_wandb --wandb_project coart-vae"
```
Outputs land in `results/coart_feat18_{YYYYMMDD}_{run_tag}/` (auto-resolved across midnight; see `coart/vae/config.py:_build_output_dir`).

### 9.2 Run the diagnostic analysis (Stage-1 form, no per-step)
```
ssh host-10-240-99-119 "cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
  bash scripts/run_analyze_finetune_8gpu.sh \
    results/coart_feat18_20260423_three_branch_ws_v0 155000 200"
```
Wall-clock ~30 min on 8 × A100/H100 (rank-0 helmet bottleneck). Output:
- `<ckpt_dir>/analysis_step155000/report.md` — markdown headline + tables.
- `<ckpt_dir>/analysis_step155000/tables/{weight_drift_summary, activation_contributions, per_channel_pred_vs_target, ablation_metrics}.csv`
- `<ckpt_dir>/analysis_step155000/figures/*.png` — SV spectra, per-channel histograms, pred-vs-target scatter, ablation Δ-CD bar chart.

### 9.3 Single-process / partial mode
```
.venv/bin/python scripts/coart_analyze_finetune.py \
  --ckpt_dir <DIR> --step <S> --use_ema --n_val 200 \
  --mode all                # full single-process
  --skip_drift              # any combination of skip flags
  --use_online              # analyse non-EMA ckpt
```

### 9.4 Reproducibility caveats
- `reconstruct_step0_state_dicts` pins `torch.manual_seed`/`torch.cuda.manual_seed_all` before model construction; the xavier draw is "a representative xavier under the same distribution", not the exact training-launch tensor (Stage-1 didn't pin its own seed). The drift baseline is therefore good to ±10 % on `rel_frob_drift` but may differ from the literal step-0 weights.
- Forward pass in fp32 (not bf16) to remove autocast noise from per-channel zero-rate / pearson_r. Wandb-logged training-time metrics may differ slightly.

### 9.5 Where the code lives (additive only — nothing in `coart/vae/` or `coart/eval/` was modified)
```
coart/analysis/
  ├── __init__.py
  ├── weight_drift.py        # §A
  ├── activation_stats.py    # §B
  ├── head_ablation.py       # §C
  └── report.py              # auto-generates report.md
scripts/
  ├── coart_analyze_finetune.py        # CLI
  └── run_analyze_finetune_8gpu.sh     # 8-GPU launcher
coart/tests/
  ├── test_analysis_weight_drift.py    (6 tests)
  ├── test_analysis_activation_stats.py (4 tests)
  ├── test_analysis_head_ablation.py    (11 tests)
  └── test_analysis_report.py           (5 tests)
```

### 9.6 Design + plan documents
- `docs/superpowers/specs/2026-05-02-vae-finetune-effectiveness-analysis-design.md` — full spec.
- `docs/superpowers/plans/2026-05-02-vae-finetune-effectiveness-analysis.md` — 11-task implementation plan (executed via subagent-driven development with stage-level batch reviews; commit chain `72ba7f6 → 2bf0554`).

---

## 10. Open questions for Stage-2

1. **Should we drop p2 entirely or keep + repair?** 8.1 vs Plan B. Drop is simpler and reflects data reality (97.79 % single-point); keep+repair preserves capability for the 2.21 % multi-point cubes which may matter for thin sheet meshes (`val_p10` had 1236 components and was the only asset where `oracle_p2` slightly improved CD).
2. **Is `lambda_kl=1e-4` enough, or do we need annealing?** Stage-1 KL went 21 → 52 unrestrained. Stage-2 baseline target: `train_kl < 25` at step 130k.
3. **Is conditional MSE alone sufficient for ef, or do we need the topology regularizer too?** Try 8.2 first; only add 8.3 if oracle_ef gap is still > 30 %.
4. **What's the right Stage-2 max_steps given a 10× larger `lambda_kl`?** KL constraint usually slows recon convergence. Estimate: 130k–160k; calibrate on the val_recon plateau.
5. **Are the 2.21 % multi-point cubes a homogeneous population, or do they cluster (e.g. always a particular asset class)?** If clustered, weighted loss is straightforward; if scattered, importance sampling at the dataloader level may be cleaner.

---

## 11. Artifacts inventory

### Stage-1 training run
- `results/coart_feat18_20260423_three_branch_ws_v0/`
  - `config.json` — full Stage-1 config
  - `ckpt_step{145000, 150000, 155000}.pt` — online ckpts (rolling 5)
  - `ema_0.9999_{enc,dec}_step0155000.pt` — latest EMA (rolling 1)
  - `misc_step0155000.pt` — optimizer / step / wandb ID for resume
  - `tb_logs/events.out.tfevents.*` — full scalar history (101 tags)
  - `wandb/run-*` — wandb runs (3 sessions across resumes)
  - `meshes_step{0..155000}/` — periodic mesh dumps

### Stage-1 evaluation
- `ema_eval_step0155000.json` — 8-asset golden set CD/NC/F-score on EMA at step 155k (mean_cd 6.56e-6, mean_nc 0.947).

### This analysis
- `results/coart_feat18_20260423_three_branch_ws_v0/analysis_step155000/`
  - `report.md`, `tables/*.csv`, `figures/*.png` (see §9.2).
- New code committed on `vae-finetune` branch (12 commits from `72ba7f6` to `2bf0554`).

### Pre-launch design context
- `logs/findings_feat18_vae_launch.md` — pre-launch design audit (warm-start math, lr alignment, EMA / resume / OOM risks).
- `logs/findings_corep_merge.md` — deeper warm-start math (subagent A/B research; orthogonality of `input_layer.W[:,0:3]` vs `[:,3:6]`).
- `logs/findings_oom_quality.md` — OOM observations during early runs.

---

## 12. Changelog of this document

| Date | Change |
|---|---|
| 2026-05-02 | Initial draft — Stage-1 retrospective + Stage-2 action guide. Author: siyuan + Claude (vae-finetune branch, post step-155k analysis). |
| 2026-05-02 | Merged content from archived `20260423-coart-vae-finetune-full-summary.md`: added §8.0 DiT API invariants, §L6 expanded with 5-bug history, Appendix A DDP config, Appendix B `coart/` module tree, Appendix C operational pitfalls. |

---

## Appendix A — Stage-1 final DDP / collective configuration

The DDP construction below is the result of debugging the 5 bugs in §L6. Stage-2 should reuse this verbatim unless there is a specific reason to deviate.

```python
# coart/common/dist_utils.py
from datetime import timedelta
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(
    "nccl",
    timeout=timedelta(hours=1),  # default 10 min misclassifies cold JIT as fault
)

model = DDP(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
    bucket_cap_mb=128,                  # grad bucket coalescing size
    find_unused_parameters=False,       # required: every param must be used every step
    static_graph=True,                  # required: ef_head is referenced multiple times
                                        # in forward (final feats + subdivision head)
    gradient_as_bucket_view=True,       # zero-copy bucket
    broadcast_buffers=False,            # model has no BN running stats
)
```

Mandatory cooperating settings:
- `--bucket_sort_mode ascending` (CLI flag): voxel count monotonically increases per epoch → triton autotune compiles each shape once. `shuffle` mode caused step-time variance 0.5s ↔ 150s (bug #4).
- Logger may NOT call collective ops (all_reduce / all_gather) inside its rank-0 buffer-flush path — use rank-0-local writes (bug #2 fix).
- Resume must peek `misc_step*.pt` to restore trainable-param mask before constructing the optimizer (bug #1 fix).

Cold-JIT timing for first launch on 8 ranks: **15–50 min** (each rank runs independent triton autotune; NFS triton cache partially shared via `coart/__init__.py:TRITON_CACHE_DIR`). **Do not kill a stalled-looking launcher in the first hour.** Use `py-spy dump --pid <PID>` to distinguish triton-autotune-loop from genuine NCCL deadlock (bug #5 fix raised the fault threshold to 1 h, but py-spy is still useful for diagnosis).

---

## Appendix B — `coart/` module map (orientation for new contributors)

The Stage-1 code is organised task-first (`vae/`, `dit/`, `eval/` are sibling tasks; shared infra in `common/` and `data/`). This is intentional so the Stage-2 changes (which are mostly in `vae/`) do not touch shared modules.

```
coart/
├── __init__.py                 # sets TRITON_CACHE_DIR=<repo>/.cache/triton (NFS-shared)
├── README.md                   # module-level Chinese docs
├── common/
│   ├── flex_gemm_patch.py      # SubMConv3dFunction.backward fix for frozen weights
│   ├── dist_utils.py           # init_dist (1h timeout), unwrap, wrap_ddp (static_graph=True)
│   ├── ema.py                  # EMAModel shadow params + state_dict {decay, shadow}
│   ├── checkpoint.py           # atomic_save + rolling-K per-prefix
│   └── logging.py              # CoartTBLogger (rank-0-only, wandb+TB+image+object3d+alert)
├── data/
│   ├── feat18_dataset.py       # Feat18Dataset (sha-hash val split, max_voxels cap, dict return)
│   ├── stats.py                # normalize/denormalize + load_stats
│   └── samplers.py             # BucketedDistributedSampler (ascending mode mandatory)
├── vae/
│   ├── io_stems.py             # Feat18EncIO / Feat18DecIO three-branch
│   ├── build.py                # build_models + load_pretrained_into + warm-start
│   ├── config.py               # @dataclass VaeTrainConfig + argparse + output-dir resolver
│   ├── loss.py                 # compute_vae_loss (no render; per-block recon)
│   ├── sampling.py             # dump_samples (eval-time mesh export)
│   ├── train.py                # main loop (base/resume mode, unfreeze schedule, deep-eval hook, watchdog)
│   └── __main__.py             # CLI entry: python -m coart.vae
├── eval/
│   ├── golden_assets.json      # 8-asset static manifest
│   ├── golden_baseline.json    # Layer V baseline (helmet / triple_sphere EXP-5 hardcoded)
│   ├── normalization.py        # corep_to_exp5_vertices, normalize_mesh_exp5_inplace
│   ├── metrics.py              # wraps scripts/eval/{eval_metrics, ovoxel_repr_test}
│   ├── deep_eval.py            # run_deep_eval() — encoder→decoder→feature_to_mesh→metrics
│   └── watchdog.py             # 3 alert conditions (grad_spike / ef_diverge / helmet_bad)
├── analysis/                   # ADDED in this analysis (Stage-1 retrospective tooling)
│   ├── weight_drift.py
│   ├── activation_stats.py
│   ├── head_ablation.py
│   └── report.py
├── dit/                        # placeholder for downstream DiT
└── tests/
    ├── test_build_warmstart.py / test_io_stems.py / test_loss.py / test_ema.py / ... (53 tests)
    └── test_analysis_*.py      (26 analysis tests)

scripts/
├── launch_coart.sh                    # production launcher (sources .coart.env, exec torchrun)
├── coart_analyze_finetune.py          # diagnostic CLI (this analysis)
├── run_analyze_finetune_8gpu.sh       # 8-GPU sharded launcher (this analysis)
├── coart_ema_eval.py                  # one-off EMA deep-eval
├── coart_build_golden.py              # build/refresh golden_assets NPZs
└── coart_renormalize_golden.py        # bring legacy NPZs to exp5 schema
```

---

## Appendix C — Stage-1 operational pitfalls (battle-tested)

These are not bugs — they are environmental gotchas that cost time during Stage-1 launch and will likely recur in Stage-2 if not anticipated.

1. **8-GPU DDP cold-JIT is 15–50 min on first launch.** Each rank runs independent triton autotune; the NFS cache (`<repo>/.cache/triton`) helps after the first run but the very first launch is slow. **Do not kill the launcher** in the first hour even if logs appear frozen.

2. **`nvidia-smi` 100 % util + 0 % mem util** can mean either NCCL-blocking-on-collective *or* a triton-autotune busy loop. Distinguish via `py-spy dump --pid <PID>` (needs sudo); if you see `pyfunctorch` / `triton.runtime` frames it is autotune; if `_C.NCCL_*` frames it is a collective deadlock (debug bug #2 / #3 territory).

3. **`~/.netrc` is per-node-local.** wandb auth needs `~/.netrc` (or `WANDB_API_KEY` env) on *every* training node. Multi-node runs require either symlinking via NFS or env-var injection in `launch_coart.sh`.

4. **`WANDB_MODE=offline` still validates the API key on first init.** Setting only `WANDB_MODE=offline` without prior `wandb login` raises during `wandb.init()`. Either run `wandb login` once on each node or set `WANDB_API_KEY` in the launcher env.

5. **Sparse-conv + DDP MUST use `static_graph=True`.** `output_layer.ef_head` is referenced twice in forward (final feats + subdivision head share path). With `find_unused_parameters=False` this silently deadlocks at the next epoch boundary; with `find_unused_parameters=True` this raises "marked ready twice". Only `static_graph=True` makes it work. See bug #3 in §L6.

6. **bf16 autocast is fine for forward, but loss / KL must be fp32.** `compute_vae_loss` calls `.float()` on inputs internally — Stage-2 should preserve this pattern.

7. **`max_voxels=500000` is a guard, not a typical value.** Single-asset peak voxels is ~499 885 (helmet); typical is 75k–250k. Lowering this for a "smaller batch" experiment risks dropping helmet-class assets entirely from training, which will degrade the deep-eval helmet score.

8. **`bs=1` is intentional and not casually upgradable.** bs=2 forces `max_voxels` down to ~250k and *would* drop the heaviest assets; gradient-accumulation gives effective bs=N at 1/N the step-rate but sparse-conv VAE has no measurable bs benefit over bs=1×8-rank DDP (verified empirically in Stage-1 early experiments).

9. **The helmet asset dominates wall-time on every eval / ablation pass.** Its predicted mesh has ~600k components — `feature_to_mesh` and `compute_topo_metrics` are slow on this scale. Stage-1's 8-GPU sharded analysis took ~30 min wall, ~25 min of which was rank-0's helmet ablation. If wall-time matters, distribute assets by `n_voxels`-balanced bin-packing instead of round-robin.

10. **Atomic ckpt save (`tmp → rename`) is not optional.** Stage-1 saw multiple session preemptions; the rolling-K (3 online, 1 EMA) plus atomic save ensured no run was ever lost. Stage-2 reuses the same `coart/common/checkpoint.py`.
