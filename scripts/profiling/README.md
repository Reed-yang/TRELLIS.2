# Coart DiT Profiling Infrastructure

A small toolkit for **iteratively measuring DiT training step-time / memory /
throughput** across config tweaks. Designed so you can flip a knob, fire one
command, and have a JSON + markdown diff vs your last baseline within ~3-5
minutes (depending on warmup steps).

## Files

| Script | Role |
|---|---|
| `profile_dit.py` | Run a short, instrumented training pass with config overrides; emit `result.json` |
| `baseline_extract.py` | Parse `log.txt` of an *existing* training run (live or finished) into the same `result.json` schema — no new training needed |
| `profile_compare.py` | Render markdown diff table over N `result.json` files (first = baseline) |
| `profile_sweep.py` | Run a YAML-defined matrix of cells sequentially, then auto-call `profile_compare` |
| `sweeps/example.yaml` | Template sweep |

All paths are absolute under `/mnt/novita2/siyuan/workspace/TRELLIS.2`. Result
files land in `logs/profile_results/` and per-run output dirs land in
`results/profile_dit_runs/`.

## Quick start — three core workflows

### 1. Snapshot an existing/running training as baseline

When you have a live training run and want to capture its current
steady-state numbers as a baseline (no GPU cost — just reads the log):

```bash
.venv/bin/python scripts/profiling/baseline_extract.py \
    --output-dir results/coart_dit_shape_20260502_201753_v3 \
    --label v3_baseline \
    --warmup-steps 100  # skip the cold-start spike
```

This writes `logs/profile_results/v3_baseline_<ts>.json` with:
- step time mean / p50 / p95 / p99
- throughput (steps/h, tok/s) — populated if the run was using our
  instrumented trainer subclass
- per-t-bin train loss
- elastic memory mean / mem_ratio mean
- hold-out eval loss (if `i_eval` was set)

### 2. Run a single profile probe with a config tweak

Run 30 warmup + 100 active steps with one config override on the idle
node host-10-240-99-118:

```bash
.venv/bin/python scripts/profiling/profile_dit.py \
    --label split1_target082 \
    --host host-10-240-99-118 \
    --num-gpus 8 \
    --warmup-steps 30 --active-steps 100 \
    --override trainer.args.batch_split=1 \
    --override trainer.args.elastic.args.target_ratio=0.82
```

The script will:

1. Clone the base config, apply both overrides, force `i_log=1` (every
   step → log.txt), and disable `i_save` / `i_eval` / `wandb` to keep
   the probe cheap.
2. Push the tmp config to `scripts/profiling/.tmp_configs/<run>.json`.
3. Build + ssh-execute the train command on the chosen host.
4. After exit, parse `log.txt` and write `logs/profile_results/<run>.json`.

### 3. Compare two or more runs

```bash
.venv/bin/python scripts/profiling/profile_compare.py \
    --runs logs/profile_results/v3_baseline_*.json \
           logs/profile_results/split1_target082_*.json \
    --out logs/profile_compare/cmp_<ts>.md
```

The first run is the baseline; subsequent runs show `(±X% ✅/❌/≈)`
deltas vs baseline. Per-t-bin loss and hold-out eval loss tables also
included if available.

### 4. Sweep a matrix

```bash
.venv/bin/python scripts/profiling/profile_sweep.py \
    --sweep scripts/profiling/sweeps/example.yaml
```

`example.yaml` defines six cells: baseline + five tweaks (target_ratio
high/low, max_tokens drop, batch_size_per_gpu push, batch_split revert).
The sweep:

1. Calls `profile_dit.py` for each cell (sequential — never parallel,
   to avoid GPU contention).
2. Aggregates result JSONs into a single `compare.md`.

Cell names become both the result.json filename prefix and the column
header in the comparison table.

## Override path syntax

Use **dotted JSON paths** to navigate the config tree:

```
--override trainer.args.batch_split=1
--override trainer.args.elastic.args.target_ratio=0.82
--override dataset.args.max_tokens=4096
--override trainer.args.optimizer.args.lr=1e-5
```

Values are **JSON-decoded**:
- `0.82` → float
- `12` → int
- `true` / `false` → bool
- `[1,2,3]` → list
- `"foo"` → str (use quotes for strings that look like JSON literals)

## Result JSON schema (output of profile_dit & baseline_extract)

```jsonc
{
  "label": "split1_target082",
  "kind": "profile_run",            // or "baseline_extract"
  "run_id": "split1_target082_20260502_215100",
  "host": "host-10-240-99-118",
  "num_gpus": 8,
  "wallclock_s": 312.4,
  "subprocess_exit": 0,
  "warmup_steps": 30,
  "active_steps": 100,
  "tmp_config_path": "...",
  "output_dir": "results/profile_dit_runs/...",
  "stdout_log": "logs/profile_run_*.log",
  "config": { /* full resolved config */ },
  "overrides": ["trainer.args.batch_split=1", "..."],
  "summary": {
    "n_rows_total": 130,
    "n_rows_after_warmup": 100,
    "step_range": [31, 130],
    "warmup_steps": 30,
    "metrics": {
      "time/step":   {"n":100, "mean":2.41, "p50":2.40, "p95":2.69, "p99":2.95, "min":2.18, "max":3.10, "stddev":0.18},
      "perf/throughput_step_per_h": {...},
      "perf/throughput_tok_per_s":  {...},
      "perf/dataloader_wait_s":     {...},
      "perf/mem_peak_gb":           {...},
      "perf/mem_alloc_gb":          {...},
      "loss/loss":      {...},
      "status/grad_norm":{...},
      "elastic/input_size":{...},
      "elastic/memory":  {...},
      "elastic/mem_ratio":{...}
    },
    "loss_bins": {
      "loss/bin_0/mse": {...},
      ...
      "loss/bin_9/mse": {...}
    },
    "eval": {
      "eval/loss": {...},
      "eval/bin_*/mse": {...}
    }
  }
}
```

## What the schema captures (and why)

| Field | Why it matters |
|---|---|
| `time/step` p50 / p95 / p99 | **Median** = central tendency; **p95–p99** = tail spike, the thing that DDP barriers wait on |
| `perf/throughput_step_per_h` | Direct ETA driver |
| `perf/throughput_tok_per_s` | Normalised compute throughput; isolates kernel efficiency from batch composition |
| `perf/dataloader_wait_s` | I/O bottleneck signal |
| `perf/mem_peak_gb` / `mem_alloc_gb` | OOM headroom; gap between peak and alloc reveals fragmentation |
| `elastic/mem_ratio` | Live snapshot of how aggressive the elastic controller is — should approach `target_ratio` after warmup |
| `elastic/input_size` | Sanity-check that the LPT bin-pack is producing the expected token sum |
| `loss/bin_*` | Per-t-bin convergence; flow-matching often regresses non-uniformly across t |
| `eval/*` | Held-out (overfitting) signal — only present when `i_eval > 0` |

## Cost of a single probe

With `--warmup-steps 30 --active-steps 100`, an 8-GPU profile probe on
H100 takes ≈ **5–7 minutes** (≈ 60s NCCL init + 30 × 2s warmup +
100 × ~2s active). Sweep of 6 cells ≈ 35–45 min sequential.

## Important: never share GPUs with production training

The toolkit always writes its own `results/profile_dit_runs/<run>/`
output dir. **But** the actual training process needs whole GPUs. If
v4 is running on host-10-240-99-119, point profiles at 117 or 118 with
`--host`. The launcher refuses to wedge into an already-busy GPU —
verify with `nvidia-smi` first.

## When to use which workflow

```
Need to compare what's already trained?           -> baseline_extract
Need to test a single 'what if' config tweak?     -> profile_dit
Need to scan a 2D matrix (BS × target_ratio)?     -> profile_sweep
Done with measurements; want a writeup?           -> profile_compare on the result.jsons
```

## Limitations / TODO

- No `torch.profiler` Chrome-trace integration yet — train.py already
  has a `--profile` flag; can be added as a `profile_dit.py --profiler`
  switch later.
- No automatic NCCL barrier-time isolation. If you want this, run
  `NCCL_DEBUG=INFO` and grep `Coll Allreduce` deltas; not exposed
  in the result JSON.
- No per-rank breakdown — all metrics are rank-0 (the trainer writes
  log.txt only on rank 0). Per-rank cost imbalance has to be inferred
  from `time/step` p99 vs p50.
- No automatic OOM detection — a sweep cell that OOMs will surface as
  `subprocess_exit != 0` in the result and a missing summary; check the
  per-cell stdout log.
