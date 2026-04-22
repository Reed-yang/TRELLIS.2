# preprocess-by-rank — offline easy→hard ranking + resume-safe multi-node launcher

Offline pipeline that (1) scans 168 k GLB assets, (2) ranks them from
easy to hard by predicted encode time + OOM probability, then
(3) launches `precompute_feat18.py` across one-or-more nodes and
one-or-more GPUs per node, tracking progress on the shared filesystem
so **resume is instant** and **any prefix of the rank can be scaled up
later** by simply extending the work list.

The launcher is a thin wrapper around `precompute_feat18.py`. All the
per-mesh success / failure bookkeeping lives in that script; the
launcher just fans ranks out across GPUs and hides the
`CUDA_VISIBLE_DEVICES` plumbing.

---

## Contents

```
scripts/preprocess-by-rank/
├── scan_168k.py            # 1. GLB header peek → scan_168k.parquet
├── build_labels.py         # 2. parse prior rank*.log → labels.parquet
├── fit_and_rank.py         # 3. sklearn GBM score + tier + rank → full_ranked.csv
├── launch_full_rank.sh     # 4. multi-node / multi-GPU / asymmetric-cluster launcher
├── status.py               # 5. any-time progress snapshot (--watch supported)
├── compute_stats.py        # 6. post-run aggregate mean/std across all .npz
└── out/                    #    scan / labels / ranked artefacts
```

---

## Pipeline at a glance

```
┌──────────────┐ scan    ┌────────────────────┐ labels ┌───────────────┐
│ raw/*.glb    │────────▶│ scan_168k.parquet  │───────▶│ labels.parquet│
│  (168 k)     │         │ (file_mb, faces…)  │        │ (sha, status, │
└──────────────┘         └────────────────────┘        │  enc_time)    │
                                                       └───────┬───────┘
                                                               │
                                                   fit + rank  ▼
                                                  ┌───────────────────────┐
                                                  │ full_ranked.csv (168k)│
                                                  │ tier 0..3, rank 0..N-1│
                                                  └───────────┬───────────┘
                                                              │
                              launch_full_rank.sh (per node)  ▼
                                          ┌────────────────────────────┐
                                          │ precompute_feat18.py × N   │
                                          │  ranks on M nodes (shared  │
                                          │  output dir, resume safe)  │
                                          └─────────────┬──────────────┘
                                                        ▼
                                          ┌────────────────────────────┐
                                          │ OUT_DIR/data/<sha>.npz     │
                                          │ + OUT_DIR/data/*.failed    │
                                          │ + OUT_DIR/failed_rank*.txt │
                                          └────────────────────────────┘
```

---

## 0. Prerequisites

- A Python venv at `${REPO_ROOT}/.venv` with torch, corep_fast deps,
  sklearn, pandas. The launcher uses this by default — override via
  `PYTHON_BIN` if you maintain a different path.
- The output directory must be writable by the user launching. First-
  time setup (once):

  ```bash
  sudo chown -R $(id -u):$(id -g) /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/feat18_512
  sudo find   /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/feat18_512/ -type d -exec chmod g+s {} \;
  sudo chmod -R g+rw /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/feat18_512/
  ```

  Setgid on every subdir lets a second user in the same primary group
  keep writing without inheriting their own gid.

---

## 1. (Re)generate the ranking

Only needed when the raw asset set or the labelled subset changes.
Cheap — under a minute total:

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
.venv/bin/python scripts/preprocess-by-rank/scan_168k.py      # ~15 s
.venv/bin/python scripts/preprocess-by-rank/build_labels.py   # ~1  s
.venv/bin/python scripts/preprocess-by-rank/fit_and_rank.py   # ~20 s
```

Outputs in `scripts/preprocess-by-rank/out/`:

| artefact              | role                                                          |
|-----------------------|---------------------------------------------------------------|
| `scan_168k.parquet`   | lightweight GLB header features for all 168k                  |
| `labels.parquet`      | `(sha, status, enc_time)` parsed from the existing 9895 logs  |
| `full_ranked.csv`     | **authoritative work-list** (168k rows, ordered easy→hard)    |
| `ranked_all.parquet`  | same ordering + richer schema for analysis                    |
| `top_20k.csv`         | convenience slice (first 20 000 rows)                         |
| `summary.md`          | tier sizes + cumulative ETA table per milestone               |

Tuning the filter (all optional):

| flag                 | default  | effect                                           |
|----------------------|----------|--------------------------------------------------|
| `--max_file_mb`      | `15.0`   | hard-threshold on GLB file size                  |
| `--max_n_faces`      | `300000` | hard-threshold on total triangle count           |
| `--max_n_meshes`     | `30`     | hard-threshold on sub-mesh / primitive count     |
| `--max_pred_enc_s`   | `20.0`   | soft-threshold on predicted encode time          |
| `--max_oom_prob`     | `0.15`   | soft-threshold on predicted OOM probability      |
| `--n_target`         | `20000`  | slice size for `top_20k.csv`                     |
| `--n_gpus`           | `8`      | used in the ETA summary only                     |

The cumulative ETA + OOM-risk table in `summary.md` is the best guide
for deciding where to stop in the rank.

---

## 2. Launching

`launch_full_rank.sh` fans work across GPUs on **one** node. To scale
to multiple nodes, run it separately on each node with a consistent
global coordinate (same `WORLD_SIZE`, disjoint `RANK_START`).

### 2.1 Single node, 8 GPUs (the common case)

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
NODE_IDX=0 NODE_COUNT=1 GPUS_PER_NODE=8 \
  bash scripts/preprocess-by-rank/launch_full_rank.sh
```

### 2.2 Smoke test first — 20 rows on 1 GPU

```bash
NODE_IDX=0 NODE_COUNT=1 GPUS_PER_NODE=1 EXTRA_ARGS="--limit 20" \
  bash scripts/preprocess-by-rank/launch_full_rank.sh
```

### 2.3 Two symmetric nodes × 8 GPUs (16 ranks)

On node 0:

```bash
NODE_IDX=0 NODE_COUNT=2 GPUS_PER_NODE=8 \
  bash scripts/preprocess-by-rank/launch_full_rank.sh
```

On node 1 (separately):

```bash
NODE_IDX=1 NODE_COUNT=2 GPUS_PER_NODE=8 \
  bash scripts/preprocess-by-rank/launch_full_rank.sh
```

### 2.4 Asymmetric: node 0 = 8 GPUs, node 1 = 4 idle GPUs (e.g. 0,5,6,7)

Override the global rank layout explicitly so both sides agree on
`WORLD_SIZE` and carve up disjoint rank ranges:

```bash
# node 0 — global ranks 0..7, CUDA devices 0..7 (default)
NODE_IDX=0 NODE_COUNT=2 GPUS_PER_NODE=8  WORLD_SIZE=12 RANK_START=0 \
  bash scripts/preprocess-by-rank/launch_full_rank.sh

# node 1 — global ranks 8..11, CUDA devices 0,5,6,7
NODE_IDX=1 NODE_COUNT=2 GPUS_PER_NODE=4  WORLD_SIZE=12 RANK_START=8 \
  CUDA_DEVICES=0,5,6,7 \
  bash scripts/preprocess-by-rank/launch_full_rank.sh
```

Contract: `WORLD_SIZE` identical on every node, and the ranges
`[RANK_START, RANK_START + GPUS_PER_NODE)` across all nodes partition
`[0, WORLD_SIZE)` with no overlap.

### 2.5 Launching over SSH (detach-safe)

Start each node from your local shell without tying up an SSH socket:

```bash
ssh host-10-240-99-117 'setsid nohup env \
    NODE_IDX=0 NODE_COUNT=2 GPUS_PER_NODE=8 WORLD_SIZE=12 RANK_START=0 \
    bash /mnt/novita2/siyuan/workspace/TRELLIS.2/scripts/preprocess-by-rank/launch_full_rank.sh \
  > /mnt/novita2/siyuan/workspace/TRELLIS.2/logs/launch_117_$(date +%Y%m%d_%H%M%S).log \
  2>&1 < /dev/null & echo pid=$!'
```

`setsid` + `nohup` + redirect + `< /dev/null` makes the workers live
on after you disconnect.

### 2.6 Full environment variable reference

All optional. Every variable has a sensible default.

| var                 | default                                                                         | purpose                                                                     |
|---------------------|---------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| `NODE_IDX`          | `0`                                                                             | 0-based index of this node in the cluster                                   |
| `NODE_COUNT`        | `1`                                                                             | total nodes participating                                                   |
| `GPUS_PER_NODE`     | `8`                                                                             | number of GPUs this node will use                                           |
| `WORLD_SIZE`        | `NODE_COUNT * GPUS_PER_NODE`                                                    | global rank count; override for asymmetric clusters                         |
| `RANK_START`        | `NODE_IDX * GPUS_PER_NODE`                                                      | global rank that this node's `local_rank=0` maps to                         |
| `CUDA_DEVICES`      | `0,1,…,GPUS_PER_NODE-1`                                                         | comma-separated CUDA device IDs to use (one per local rank)                 |
| `PYTHON_BIN`        | `${REPO}/.venv/bin/python`                                                      | interpreter; falls back to `$PATH` python if the venv one is missing        |
| `DATASET_ROOT`      | `/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab`                             | root containing `raw/…glb`                                                  |
| `METADATA_CSV`      | `${REPO}/scripts/preprocess-by-rank/out/full_ranked.csv`                        | input work-list; must have `sha256,local_path` columns                      |
| `RESOLUTION`        | `512`                                                                           | voxel grid resolution                                                       |
| `OUT_DIR`           | `${DATASET_ROOT}/feat18_${RESOLUTION}`                                          | output `data/*.npz` dir (shared across nodes)                               |
| `LOG_DIR`           | `${OUT_DIR}/../logs/precompute_feat18_r${RESOLUTION}_full_n${N}x${G}`           | per-rank log dir (rotated with timestamp)                                   |
| `NUM_WORKERS`       | `(nproc - 4) / GPUS_PER_NODE`                                                   | CPU forks per rank (stock formula is wrong under multi-node, we override)   |
| `MAX_MESH_FILE_MB`  | unset                                                                           | skip meshes larger than X MiB on disk (slow-load guard)                     |
| `EXTRA_ARGS`        | unset                                                                           | extra args to `precompute_feat18.py` (e.g. `--limit 1000`, `--verbose`)     |

Positional `$@` after the launcher name is also forwarded to every
rank.

---

## 3. Resume semantics

Just re-run the **exact same command**. There is no state file, no
lock file, no manual cleanup.

The launcher passes `--skip_done_fast` to `precompute_feat18.py`,
which:

1. lists `OUT_DIR/data/*.npz`           → set of done sha256
2. lists `OUT_DIR/data/*.npz.failed`    → set of failure-sentinel sha256
3. filters both out of the rank's work queue **before** the loop starts

This enumerates 10 k files in ~0.1 s, versus ~5-10 minutes in the old
per-row `np.load` resume path. Consequences:

- A rank that was already done exits immediately with "0 ok, 0 failed".
- A rank that was mid-way through resumes exactly where it left off;
  no work is repeated.
- Per-rank running stats (`stats_rank<r>.npz`) written at finalisation
  cover **only the rows encoded by this run**. Run `compute_stats.py`
  once at the end to get a global mean/std over every npz on disk
  (§5).

### 3.1 Scaling up the work-list mid-run

Because the shared output directory is the only state, you can:

- launch the first batch against `top_20k.csv`
- while it runs, decide to extend to 50 k; kill nothing
- launch a second batch against a bigger CSV (`top_50k.csv`)
- the second batch instantly skips everything already on disk and
  picks up where the first stopped
- the first batch either finishes normally or is killed by you — either
  way, zero lost work

### 3.2 Adding / removing a node mid-run

Ranks must partition `[0, WORLD_SIZE)`. You **cannot** add or drop a
node without re-launching the existing nodes — the striding is fixed
at launch time. Practical recipe:

1. `pkill -f precompute_feat18` on all existing nodes.
2. Re-launch every node with the new `WORLD_SIZE` + reassigned
   `RANK_START` / `CUDA_DEVICES`.
3. Fast resume skips everything already on disk; restart cost is
   seconds (directory listing + model import), not minutes.

### 3.3 Recovering from a poisoned CUDA context

`precompute_feat18.py` has built-in handling: if a mesh produces a
context-poisoning exception ("CUDA error", "illegal memory access",
etc.), the rank writes a `.failed` sentinel for that sha and `os.execv`s
itself. The launcher's nohup parent is unaware, and the new child
resumes from `skip_done_fast` with the sentinel keeping the bad mesh
out of its queue.

---

## 4. Monitoring

`status.py` reads only from `OUT_DIR` and the per-rank log dir — safe
to run concurrently with an active job.

```bash
# one-shot snapshot
.venv/bin/python scripts/preprocess-by-rank/status.py

# live refresh every 30s (Ctrl-C to stop)
.venv/bin/python scripts/preprocess-by-rank/status.py --watch 30

# 12 GPUs total for the ETA math (117 × 8 + 120 × 4)
.venv/bin/python scripts/preprocess-by-rank/status.py --n_gpus 12 --watch 30

# dump per-sha status snapshot
.venv/bin/python scripts/preprocess-by-rank/status.py --emit_csv /tmp/per_sha.csv

# also print the next 20 pending (hardest not-yet-run) rows
.venv/bin/python scripts/preprocess-by-rank/status.py --show_next 20
```

Output columns:

- per-tier done / failed / pending counts from `full_ranked.csv`
- median + mean observed `enc_time` (last ~1 000 ok encodes across all ranks)
- predicted total enc-seconds left for pending rows, and ETA hours

For raw per-rank tqdm tail:

```bash
tail -F /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/logs/precompute_feat18_r512_full_n*/*_global0_*.log
```

### 4.1 Quick filesystem-based throughput measurement

Count freshly-written npz files in a recent window to get a node- and
log-independent throughput number:

```bash
DATADIR=/mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/feat18_512/data
now=$(date +%s)
for window in 60 300 900; do
  c=$(find "$DATADIR" -maxdepth 1 -name '*.npz' -newermt "@$((now - window))" | wc -l)
  rate=$(awk "BEGIN {printf \"%.2f\", $c/$window}")
  echo "last ${window}s: ${c} new npz → ${rate} /s"
done
```

Combines the contribution of every rank on every node and does not
depend on parsing tqdm output.

---

## 5. After the run — global mean / std

Because `--skip_done_fast` bypasses per-row stats accumulation, the
per-rank `stats_rank<r>.npz` only covers this run's fresh encodes. To
get a **global** mean / std across every npz on disk:

```bash
.venv/bin/python scripts/preprocess-by-rank/compute_stats.py \
  --out_dir /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/feat18_512
```

Parallel npz reader (~400 files / s on 128 cores). Writes
`OUT_DIR/stats_global.npz` with `{mean, std, n_voxels}` in the exact
schema the inline finaliser used — drop-in for the SC-VAE fine-tune
pipeline.

---

## 6. End-to-end recipe for the 168k job

```bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2

# 1. regenerate ranking (optional if it already exists)
.venv/bin/python scripts/preprocess-by-rank/scan_168k.py
.venv/bin/python scripts/preprocess-by-rank/build_labels.py
.venv/bin/python scripts/preprocess-by-rank/fit_and_rank.py --n_target 20000

# 2. smoke test one GPU / 20 rows on the chosen node
NODE_IDX=0 NODE_COUNT=1 GPUS_PER_NODE=1 EXTRA_ARGS="--limit 20" \
  bash scripts/preprocess-by-rank/launch_full_rank.sh

# 3. launch production on node 117 (8 GPUs) + node 120 (4 idle GPUs)
ssh host-10-240-99-117 'setsid nohup env \
    NODE_IDX=0 NODE_COUNT=2 GPUS_PER_NODE=8 WORLD_SIZE=12 RANK_START=0 \
    bash /mnt/novita2/siyuan/workspace/TRELLIS.2/scripts/preprocess-by-rank/launch_full_rank.sh \
  > /mnt/novita2/siyuan/workspace/TRELLIS.2/logs/launch_117_$(date +%Y%m%d_%H%M%S).log \
  2>&1 < /dev/null & echo pid=$!'

ssh host-10-240-99-120 'setsid nohup env \
    NODE_IDX=1 NODE_COUNT=2 GPUS_PER_NODE=4 WORLD_SIZE=12 RANK_START=8 \
    CUDA_DEVICES=0,5,6,7 \
    bash /mnt/novita2/siyuan/workspace/TRELLIS.2/scripts/preprocess-by-rank/launch_full_rank.sh \
  > /mnt/novita2/siyuan/workspace/TRELLIS.2/logs/launch_120_$(date +%Y%m%d_%H%M%S).log \
  2>&1 < /dev/null & echo pid=$!'

# 4. monitor
.venv/bin/python scripts/preprocess-by-rank/status.py --n_gpus 12 --watch 60

# 5. when done, aggregate stats
.venv/bin/python scripts/preprocess-by-rank/compute_stats.py
```

---

## 7. Troubleshooting

**`Permission denied` writing to `feat18_512/` or `index_rank*.csv`**
— the output dir is owned by another user and the group is not shared.
Follow §0 (chown + setgid + g+rw).

**`CUDA_DEVICES (0,5,6,7) has 4 entries but GPUS_PER_NODE=8`**
— `GPUS_PER_NODE` must equal the length of `CUDA_DEVICES`. Set
`GPUS_PER_NODE=4` in the example above.

**`ERROR: WORLD_SIZE=10 must be >= RANK_START+GPUS_PER_NODE=12`**
— the rank ranges overlap or fall outside `[0, WORLD_SIZE)`. Recompute
the layout so each node owns exactly `GPUS_PER_NODE` consecutive ranks
and they tile `[0, WORLD_SIZE)` cleanly.

**`METADATA_CSV not found`**
— the default path assumes the repo is at the conventional location.
Pass an absolute `METADATA_CSV=…` if it is not.

**Two launches on the same node racing the same rank**
— the launcher is stateless; if you relaunch without pkill first,
you will have duplicate ranks. `os.replace` makes this correct (no
corruption) but wasteful. Always `pkill -f precompute_feat18` before
relaunching on a node.

**One GPU on the target node is busy with someone else's job**
— use `CUDA_DEVICES=0,5,6,7` (or whichever are idle) plus matching
`GPUS_PER_NODE`. Busy GPUs are untouched; their owner is not affected.

**fast-resume did not filter the expected count**
— check `OUT_DIR/data/` for leftover `*.npz.failed` sentinels. Those
count as "skip" too. To force a retry on a failed sha, delete the
sentinel and relaunch.

**rates differ sharply between nodes**
— expected on shared nodes. §4.1 gives a filesystem-based rate; combine
it with `nvidia-smi` on each node to attribute the gap to either other
users' GPU jobs or CPU contention (high `NUM_WORKERS` on a loaded box
hurts more than helps).

---

## Design notes

- **Why one CSV, not per-rank shards?** The rank striding in
  `precompute_feat18.py` (`metadata.iloc[rank :: world_size]`) is
  already a clean partition. A single CSV keeps the rank ordering
  reproducible across launches and avoids fragile per-rank file
  handling.

- **Why setgid on the output dir?** The job is commonly launched by
  different users on shared clusters. Setgid + group-writable lets
  the second user resume or extend the run without chowning anything.

- **Why `.failed` sentinels as well as `failed_rank<r>.txt`?** The
  per-sha sentinel is the authoritative "do not retry" marker and
  lives next to the npz. The per-rank txt file is append-only and
  useful for post-mortem aggregation / blame-by-rank. Both are written
  synchronously so a crashed rank loses at most one mesh of
  information.

- **Why rotate rank logs per launch?** Restarts overwrite the tqdm bar
  line a lot; keeping the previous session's tail as a separate file
  preserves the "what broke last time" evidence.
