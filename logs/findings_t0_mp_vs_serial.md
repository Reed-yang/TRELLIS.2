# T0 Decisive Experiment — MP vs Serial End-to-End

**Date:** 2026-04-17
**Purpose:** Before committing to V2 plan's W2 (persistent MP pool) and W6 (s4 Stage D),
decide whether MP is net-positive. Two subagents ran `corep_pipeline` at res=256 on
host-10-240-99-116, one with default MP, one with all `multiprocessing.Pool` calls
monkeypatched to run serially.

## Result

| Mode | Median wall (s) | V count | F count | Variance |
|---|---:|---:|---:|---:|
| MP default (GPU 3) | **8.631** | 551,079 | 1,102,152 | 4.3% |
| Serial nw=1 (GPU 4) | **75.312** | 551,079 | 1,102,152 | 0.15% |

**Serial is 8.7× slower than MP (+66.7 s).** MP is emphatically net-positive at current
pipeline shape. V2 plan's direction (keep MP, optimize with W2 persistent pool) is correct.

## Why the earlier hypothesis was wrong

Main-thread cProfile reported `_thread.lock.acquire 2.7 s` and worker cProfile reported
only ~60 ms of Python self-time across all stages. I interpreted the 2.7 s lock wait as
pure MP overhead that serial could reclaim.

This was wrong because cProfile `self`-time is **Python-only**. Worker time spent inside
torch/numpy C extensions doesn't appear as worker self-time but still shows up as
`lock.acquire` in the main thread. The real per-worker compute is ~seconds, not ~ms, so
removing MP shifts that cost back to the main thread and serializes what was parallel.

## Decisions

1. **Keep MP.** Plan's W2/W4/W5/W6/W7 stay.
2. **Serial (nw=1) is bit-deterministic** — 0.15% wall variance, identical V/F across
   warmup + 3 trials. Use nw=1 as the golden-snapshot basis for W4/W5 correctness gates.
3. **MP is nondeterministic** (pre-existing). F3 default-MP runs differ by ~0.7% of
   vertex set (subagent's earlier finding). Root cause almost certainly hash
   randomization (`PYTHONHASHSEED` default random). Fix via:
   - Set `PYTHONHASHSEED=0` in pool initializer (part of T3 W2 persistent pool scope).
   - Audit `set`/`dict`-iteration-based tiebreakers in s4/s6/s7/s8.
4. **Plan update** — T1 forces `num_workers=1` in `_run_pipeline` (via SerialPool
   monkeypatch pattern from `tmp/cpu_profile/t0_driver.py`). Golden gate is bit-exact
   for W4/W5 validation. A secondary MP-mode fuzzy check lands in T9.

## Artifacts

- `tmp/cpu_profile/t0_driver.py` — shared driver (both branches)
- `tmp/cpu_profile/t0_mp_default.md`, `t0_default.json`, `t0_default.log` — MP branch
- `tmp/cpu_profile/t0_serial.md`, `t0_serial.json`, `t0_serial.log` — serial branch
- `tmp/cpu_profile/t0_probe.py`, `t0_probe2.py` — serial monkeypatch sanity checks

## Cost

- Two subagents in parallel, ~10 min each. Total ~15 min wall, negligible $.
- Saved: days of W2/W6 work that would have been misdirected if plan had moved to
  "delete MP" based on the earlier cProfile misread.
