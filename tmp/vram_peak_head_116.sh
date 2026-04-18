#!/usr/bin/env bash
set -e
cd /mnt/novita2/siyuan/workspace/TRELLIS.2
mkdir -p tmp/followup_baseline
CUDA_VISIBLE_DEVICES=4 PYTHONHASHSEED=0 .venv/bin/python - <<'PYEOF' 2>&1 | tee tmp/followup_baseline/vram_peak_head.log
import torch
import trimesh
from corep_fast.pipeline import corep_pipeline
mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.4)
mesh.export('/tmp/fix_vram_head.ply')
# Warmup (allocator + cudnn init)
torch.cuda.reset_peak_memory_stats()
corep_pipeline('/tmp/fix_vram_head.ply', 256, torch.device('cuda:0'))
print(f"[warmup] peak_alloc={torch.cuda.max_memory_allocated()/1024**2:.1f}MB  peak_reserved={torch.cuda.max_memory_reserved()/1024**2:.1f}MB")
# Clean run
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
corep_pipeline('/tmp/fix_vram_head.ply', 256, torch.device('cuda:0'))
print(f"[clean] peak_alloc={torch.cuda.max_memory_allocated()/1024**2:.1f}MB  peak_reserved={torch.cuda.max_memory_reserved()/1024**2:.1f}MB")
PYEOF
