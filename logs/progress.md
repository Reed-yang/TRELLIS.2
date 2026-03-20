# TRELLIS.2 Improvement — Progress Log

## Phase 0: Gap Measurement (2026-03-19 ~ 2026-03-20)

### Goal
Quantify the quality gap between SC-VAE reconstruction upper bound and DiT generation quality, to determine Phase 1 priority (improve VAE or DiT).

### Timeline

| Date | Milestone |
|------|-----------|
| 03-19 | Evaluation metrics module (CD, F-score, NC, PSNR, SSIM) implemented and tested |
| 03-19 | Pilot data: 30 Objaverse models downloaded (diverse LVIS categories) |
| 03-19 | Encoder weights downloaded from HF Hub (shape_enc_next_dc_f16c32_fp16, 354M params) |
| 03-19 | gap_measurement.py: Path A (VAE encode-decode) + Path B (DiT generation) complete |
| 03-19 | First run with normal-map placeholder images — DiT/VAE CD ratio = 176x |
| 03-20 | Blender 3.0 installed, CYCLES conditioning images rendered (6-GPU parallel) |
| 03-20 | Second run with Blender images — DiT/VAE CD ratio = 406x (unaligned) |
| 03-20 | Discovered coordinate axis misalignment issue between GT and DiT output |
| 03-20 | Added 24-rotation alignment — DiT/VAE CD ratio = **49x** (aligned, final) |

### Current Status: **Phase 0 COMPLETE**

### Key Result

| Metric | VAE Recon | DiT Gen (aligned) | Ratio |
|--------|-----------|-------------------|-------|
| CD | 0.000071 | 0.00348 | 49x |
| F-score | 0.775 | 0.321 | 2.4x worse |
| NC | 0.932 | 0.741 | 1.26x worse |

**Decision: DiT is the bottleneck.** Per roadmap-v2.md decision rule, Phase 1 should prioritize DiT optimization (P2b promoted to highest priority).

### Deliverables
- `scripts/eval_metrics.py` — Reusable evaluation metrics with 24-rotation alignment
- `scripts/gap_measurement.py` — Full pipeline with multi-GPU support
- `scripts/prepare_pilot_data.py` — Objaverse pilot data downloader
- `scripts/render_blender_cond.py` — Blender CYCLES conditioning renderer
- `experiments/gap_measurement/` — Normal-map experiment results
- `experiments/gap_measurement_blender/` — Blender experiment results (final)

---

## Next: Phase 1 (Not Started)

Per gap measurement results, proposed priority reordering:
1. **P2b (was Phase 2)** → DiT improvements (condition injection, normal guidance, timestep sampling)
2. **P1c + P1d** → Decode-后 mesh refinement + hole fixing (independent of VAE/DiT choice)
3. **P1a + P1b** → SC-VAE improvements (lower priority given DiT is bottleneck)
