#!/usr/bin/env bash
cd /mnt/novita2/siyuan/workspace/TRELLIS.2 && \
CUDA_VISIBLE_DEVICES=4 PYTHONHASHSEED=0 .venv/bin/python - <<'PYEOF' 2>&1 | tee tmp/followup_baseline/vram_peak.log
import torch
import trimesh
from corep_fast.pipeline import corep_pipeline
mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.4)
mesh.export('/tmp/fix_baseline.ply')
torch.cuda.reset_peak_memory_stats()
corep_pipeline('/tmp/fix_baseline.ply', 256, torch.device('cuda:0'))
print(f"peak_alloc_MB={torch.cuda.max_memory_allocated()/1024**2:.1f}")
print(f"peak_reserved_MB={torch.cuda.max_memory_reserved()/1024**2:.1f}")
PYEOF
