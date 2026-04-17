# T0 Decisive Experiment — SERIAL branch

## Setup
- **Host:** `host-10-240-99-116`
- **GPU:** CUDA device 4 (masked to `cuda:0` in-process via `CUDA_VISIBLE_DEVICES=4`)
- **Baseline HEAD:** `030f638`
- **Fixture:** `trimesh.creation.icosphere(subdivisions=3, radius=0.4)` -> temp `.ply`
- **Resolution:** 256
- **Driver:** `tmp/cpu_profile/t0_driver.py` (shared with MP branch)
- **Command:**
  ```
  CUDA_VISIBLE_DEVICES=4 .venv/bin/python tmp/cpu_profile/t0_driver.py --mode serial --trials 3
  ```

## Monkeypatch verification

Driver patches `multiprocessing` BEFORE `corep_fast` import:
```python
import multiprocessing as mp
import multiprocessing.pool as mp_pool
mp.Pool = SerialPool
mp_pool.Pool = SerialPool
```

### Probe results (`tmp/cpu_profile/t0_probe.py` + `t0_probe2.py`)

1. `from multiprocessing import Pool as _Pool` AFTER `maybe_force_serial()` -> `_Pool is SerialPool == True`. Confirmed `mp.Pool` rebinding is the effective patch channel.
2. All 4 stage modules (`s4_face_point`, `s6_collapse`, `s7_rank_assign`, `s8_collapse`) import `Pool` via `from multiprocessing import Pool as _Pool` **inside** the relevant functions — not at module top level. So `patch_stage_pools()` finds no module-level `_Pool`/`Pool` aliases to rewrite (it is a no-op in this codebase).
3. This is actually fine: because the imports are function-scoped, every call re-resolves `Pool` through `mp.Pool` and therefore gets `SerialPool`. Probe 2 confirmed after running an end-to-end tiny pipeline that `mp.Pool is SerialPool` still holds.

**Patch coverage:** full. Every `Pool()` instantiation inside stage code returns `SerialPool` because the lookup goes through the patched `multiprocessing.Pool` each time.

**Grepped Pool import sites (all function-scoped):**
- `corep_fast/stages/s4_face_point.py:160, 1186`
- `corep_fast/stages/s6_collapse.py:924`
- `corep_fast/stages/s7_rank_assign.py:1314`
- `corep_fast/stages/s8_collapse.py:1426, 1767, 1935`

No `ProcessPoolExecutor` or other `concurrent.futures` entry points observed in the grep — this was not verified exhaustively though. (See "Concerns" below.)

## Results

| Trial | Wall-time (s) |
|-------|---------------|
| 1     | 75.397        |
| 2     | 75.312        |
| 3     | 75.285        |

- **Median:** **75.312 s**
- **Min:** 75.285 s
- **Max:** 75.397 s
- **Spread:** 0.112 s (0.15 % of median) — well below the 5 % target, confirming serial mode is deterministic.

## Output geometry
- **V (vertices):** 551,079
- **F (faces):** 1,102,152
- Counts identical across warmup + all 3 trials -> bit-stable nw=1 behaviour.

## Anomalies / Concerns
1. `patch_stage_pools()` is a no-op in this codebase (function-scoped imports, nothing to patch at module level). The patch is still effective via `mp.Pool`. Flagged for clarity; not a correctness issue.
2. Grep only covered `multiprocessing.Pool` imports. If any stage uses `concurrent.futures.ProcessPoolExecutor` or spawns via raw `Process`, those would NOT be serialised by this monkeypatch. Based on the 0.15 % trial variance + bit-stable V/F, serialisation appears complete, but the controller should cross-check against the MP-branch V/F to confirm there is no silent parallel path.
3. 75.3 s end-to-end is MUCH larger than the ~10 s baseline we have been targeting on the GPU pipeline. This is consistent with corep_pipeline being CPU-bound with heavy per-item work, and suggests MP is probably giving a real speedup at this resolution (contradicting the initial hypothesis that MP overhead dominates). Final verdict requires the MP branch number for comparison.

## Files
- `tmp/cpu_profile/t0_serial.md` (this file)
- `tmp/cpu_profile/t0_serial.json` (raw samples + metadata)
- `tmp/cpu_profile/t0_serial.log` (stdout from the run)
- `tmp/cpu_profile/t0_driver.py` (shared driver, mode=serial branch)
- `tmp/cpu_profile/t0_probe.py`, `t0_probe2.py` (monkeypatch sanity checks)
