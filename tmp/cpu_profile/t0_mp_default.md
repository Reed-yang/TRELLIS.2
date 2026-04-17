# T0 Decisive Experiment — MP Branch (default multiprocessing)

## Environment

- **Host:** `host-10-240-99-116`
- **GPU:** `cuda:0` (physical index 3 via `CUDA_VISIBLE_DEVICES=3`) — NVIDIA H100 80GB HBM3
- **Baseline HEAD:** `030f638` (plan commit); on-disk branch `post-profile-sync-elim`
- **Driver:** `tmp/cpu_profile/t0_driver.py` (mode=default, trials=3)
- **Fixture:** F2 = `trimesh.creation.icosphere(subdivisions=3, radius=0.4)` exported to temp `.ply`
- **Resolution:** 256
- **Call under test:** `corep_pipeline(mesh_path, 256, device)` (warmup excluded)
- **`num_workers`:** not set → uses pipeline default (multiprocessing enabled)

## Raw wall-time samples

| Trial | Wall-time (s) |
|------:|--------------:|
|     1 |         8.328 |
|     2 |         8.701 |
|     3 |         8.631 |

Sorted: `[8.328, 8.631, 8.701]`

## Aggregate

| Metric | Value (s) |
|--------|----------:|
| min    | 8.328 |
| median | 8.631 |
| max    | 8.701 |
| range  | 0.373 |

Relative spread: `(max - min) / median ≈ 4.3 %`.

## Pipeline output counts (V / F)

All three measured trials + warmup produced identical counts:

- `V = 551 079`
- `F = 1 102 152`

Output is deterministic across the warmup + 3 trials (expected, same fixture + default seeds).

## Notes on variance / anomalies

- Variance ~4 % between min and max; median is in the middle, no outliers.
- Warmup ran before timing (not included in samples) to absorb import / CUDA init / first-call JIT.
- `V` / `F` counts identical across warmup and all 3 trials → pipeline is reproducible under default MP settings on this fixture.
- No warnings, no exceptions in the log.

## Artifacts

- Raw JSON: `tmp/cpu_profile/t0_default.json`
- Full stdout log: `tmp/cpu_profile/t0_default.log`
- Driver: `tmp/cpu_profile/t0_driver.py`
- This report: `tmp/cpu_profile/t0_mp_default.md`

## Coordination with serial branch

- Parallel subagent running `--mode serial` on 116 GPU 4; no interference.
- Controller will aggregate both branches' medians to compute MP-vs-serial delta.
