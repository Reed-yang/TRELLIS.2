# Objaverse-XL Sketchfab Bulk Download (slurm)

Wrapper around `objaverse.xl.download_objects` tuned for our 6-node gpu partition.
Target: `/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/`.

## Files

- `download_sketchfab.py` — Python wrapper with `--processes` tuning, structured
  per-rank summary/log/failures output. Idempotent (resume works).
- `run_tune.slurm` — single-node processes sweep (32/64/96/128 × 500 objs).
- `run_scale.slurm` — 2-node scale-out test at the tuned processes value.
- `run_full.slurm` — 6-node array job covering all 167k unprocessed objects.
- `analyze_summaries.py` — collate rank*_summary.json into a comparison table.

## Workflow

```bash
# 1. Make sure metadata.csv is in place (one-time; already done):
.venv/bin/python data_toolkit/build_metadata.py ObjaverseXL --source sketchfab \
    --root /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab

# 2. Tune processes (runs 4 buckets sequentially, ~10 min):
sbatch tools/data_download/run_tune.slurm
# Wait for completion, then:
.venv/bin/python tools/data_download/analyze_summaries.py --tag tune-p

# 3. (Optional) Verify 2-node scaling at the picked value:
sbatch --export=ALL,PROCESSES=96 tools/data_download/run_scale.slurm
.venv/bin/python tools/data_download/analyze_summaries.py --tag scale

# 4. Full 6-node run (~30 min at 96 procs if HF not rate-limiting):
sbatch --export=ALL,PROCESSES=96 tools/data_download/run_full.slurm

# 5. Merge new_records/*.csv into raw/metadata.csv:
.venv/bin/python data_toolkit/build_metadata.py ObjaverseXL --source sketchfab \
    --root /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab
```

## Output layout

```
/mnt/novita2/data/video_obj/
├── .objaverse_cache/                      # shared annotations parquet (one-time ~500 MB)
└── ObjaverseXL_sketchfab/
    ├── metadata.csv                        # 168k full index
    ├── statistics.txt                      # downloaded count
    ├── raw/
    │   ├── hf-objaverse-v1/glbs/000-XXX/*.glb
    │   ├── new_records/part_{rank}_{tag}.csv    # per-rank downloaded list
    │   └── merged_records/                 # archived after build_metadata
    └── logs/download/
        ├── rank{R}_{tag}.log               # loguru stdout+debug
        ├── rank{R}_{tag}_summary.json      # structured metrics
        ├── rank{R}_{tag}_failures.csv      # sha256 that didn't finish
        └── sbatch_{job}_{array}.log        # slurm stdout
```

## Resume semantics

Rerunning the same sbatch is safe:

1. `download_sketchfab.py` filters out rows whose `local_path` already exists
   in `raw/metadata.csv` (updated by `build_metadata.py`).
2. Inside `oxl.download_objects`, any target file already on disk is skipped.

So after a job failure / preemption, just resubmit; only the missing tail is
re-downloaded.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Many 429 in rank log | HF CDN throttling 6-node concurrency | Lower `PROCESSES`, resubmit |
| Rank exits early with 0 downloads | metadata already fully covered | Check `filtered N already-downloaded` in log |
| `ModuleNotFoundError: objaverse` | wrong python | Must use `.venv/bin/python`, not system `python` |
| Nodes all land on 117 | partition has only 1 idle | Expected; slurm queues the rest as the partition frees up |
