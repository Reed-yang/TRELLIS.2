# coart_data_v0 - DiT Data Pipeline (10K asset milestone)

See spec: `docs/superpowers/specs/2026-05-02-coart-dit-data-pipeline-design.md`
See plan: `docs/superpowers/plans/2026-05-02-coart-dit-data-pipeline.md`

## Quickstart

### 1. Pre-flight - pick 10K sha
```bash
.venv/bin/python scripts/coart_data_v0/pick_instances.py \
  --metadata_csv     /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/metadata.csv \
  --raw_metadata_csv /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/raw/metadata.csv \
  --feat18_dir       /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/feat18_512/data \
  --aesthetic_min 4.5 --n 10000 --seed 0 \
  --out /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/coart_dit_data_v0/instances_10k.csv
```

### 2. SS-flow IoU validation gate (~15 min)
```bash
.venv/bin/python scripts/coart_compare_occupancy.py \
  --golden_dir datasets/coart_golden \
  --resolutions 64,32,16 \
  --out logs/findings_ss_flow_iou.md
```
Read the verdict line in the output report.

### 3. Render (5-7 h on 24 GPU)
```bash
STAGE=render NODES="117 118 119" NUM_GPUS_PER_NODE=8 bash scripts/coart_data_v0/run.sh
```

### 4. DINO cache (~10-30 min, can overlap with render)
```bash
STAGE=dino NODES="118 119" NUM_GPUS_PER_NODE=8 bash scripts/coart_data_v0/run.sh
```

### 5. SLat cache (~20 min)
```bash
STAGE=slat NODES="119" NUM_GPUS_PER_NODE=8 \
  VAE_CKPT=results/coart_feat18_20260423_three_branch_ws_v0/ema_0.9999_enc_step0155000.pt \
  VAE_TAG=vae_three_branch_ws_v0_ema_s0155000 \
  bash scripts/coart_data_v0/run.sh
```

### 6. Manifest
```bash
STAGE=manifest bash scripts/coart_data_v0/run.sh
```

## Smoke run (100 assets, < 30 min)

```bash
.venv/bin/python scripts/coart_data_v0/pick_instances.py \
  --metadata_csv     /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/metadata.csv \
  --raw_metadata_csv /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/raw/metadata.csv \
  --feat18_dir       /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/feat18_512/data \
  --aesthetic_min 4.5 --n 100 --seed 1 \
  --out /mnt/novita2/data/video_obj/ObjaverseXL_sketchfab/coart_dit_data_v0/instances_smoke100.csv

INSTANCES=instances_smoke100.csv NODES="119" NUM_GPUS_PER_NODE=1 LIMIT=100 \
  STAGE=render bash scripts/coart_data_v0/run.sh
# repeat for STAGE=dino, slat, manifest
```

## VAE re-encode (when coart.vae ckpt updates)

Only re-run STAGE=slat with a new VAE_TAG. Render + DINO caches are reusable.

## Output layout

See spec section 4.1.

## Re-running on partial failure

All stages are idempotent (resume via output existence check). Just re-launch the same command.
