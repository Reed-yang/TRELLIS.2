"""
Generate side-by-side comparison videos of Layer R / Layer C / Layer V meshes
for the baseline test models.

Reads persisted .ply files from
    results/baseline_experiments/EXP6_corep_recon/work_{model}_{res}/
and writes one MP4 per model to
    results/baseline_experiments/videos/{model}_comparison.mp4

Animation phases (15 s @ 24 fps = 360 frames total):
    Phase 1 (0-3s,   frames 0-71):   full mesh,          yaw +120 deg
    Phase 2 (3-6s,   frames 72-143): slice reveal,       yaw +120 deg
    Phase 3 (6-12s,  frames 144-287): slice at center,   yaw +240 deg
    Phase 4 (12-15s, frames 288-359): slice retract,     yaw +120 deg

The slice plane has fixed normal (-1, +1, +1)/sqrt(3) and moves along that
axis from outer corner (-0.9, +0.9, +0.9) to center (0, 0, 0) and back.

Usage:
    .venv/bin/python scripts/render/render_rcv_comparison.py
    .venv/bin/python scripts/render/render_rcv_comparison.py --models icosphere
"""
import argparse
import math
import os
import sys
import traceback
from dataclasses import dataclass

import numpy as np

# Project root on sys.path so trellis2 and scripts.* imports resolve.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


DEFAULT_MODELS = [
    'bowl', 'icosphere', 'parallel_planes', 'nested_spheres',
    'bugatti', 'spacesuit', 'helmet',
]

MESH_ROOT = 'results/baseline_experiments/EXP6_corep_recon'
DEFAULT_OUT_DIR = 'results/baseline_experiments/videos'

LABELS = ['Layer R (O-Voxel)', 'Layer C (CoReP)', 'Layer V (O-Voxel + VAE)']

SLICE_NORMAL = np.array([-1.0, 1.0, 1.0]) / math.sqrt(3.0)
SLICE_OUTER = np.array([-0.9, 0.9, 0.9])
SLICE_CENTER = np.array([0.0, 0.0, 0.0])


@dataclass
class AnimConfig:
    fps: int = 24
    duration: float = 15.0
    resolution: int = 768       # per-layer render resolution
    mesh_resolution: int = 512  # which EXP6 work_dir to read from
    pitch_deg: float = 25.0
    fov_deg: float = 40.0
    label_band_px: int = 60

    @property
    def total_frames(self) -> int:
        return int(round(self.fps * self.duration))

    @property
    def phase_boundaries(self) -> list:
        """Return frame indices [p1_end, p2_end, p3_end, p4_end]."""
        f = self.fps
        return [
            int(round(3.0 * f)),   # end of phase 1
            int(round(6.0 * f)),   # end of phase 2
            int(round(12.0 * f)),  # end of phase 3
            int(round(15.0 * f)),  # end of phase 4 (== total_frames)
        ]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--models', nargs='+', default=DEFAULT_MODELS,
                   help='Model names to render. Default: all 7 baseline models.')
    p.add_argument('--resolution', type=int, default=768,
                   help='Per-layer render resolution. Default: 768.')
    p.add_argument('--fps', type=int, default=24, help='Video framerate. Default: 24.')
    p.add_argument('--duration', type=float, default=15.0,
                   help='Video duration in seconds. Default: 15.0.')
    p.add_argument('--mesh-resolution', type=int, default=512,
                   help='Which EXP6 @res directory to read meshes from. Default: 512.')
    p.add_argument('--pitch', type=float, default=25.0, help='Camera pitch in degrees.')
    p.add_argument('--fov', type=float, default=40.0, help='Camera FOV in degrees.')
    p.add_argument('--out-dir', default=DEFAULT_OUT_DIR, help='Output directory for MP4 files.')
    p.add_argument('--overwrite', action='store_true', help='Re-render even if MP4 exists.')
    return p.parse_args()


def main():
    args = parse_args()
    config = AnimConfig(
        fps=args.fps,
        duration=args.duration,
        resolution=args.resolution,
        mesh_resolution=args.mesh_resolution,
        pitch_deg=args.pitch,
        fov_deg=args.fov,
    )
    print(f'Models: {args.models}')
    print(f'Config: {config}')
    print(f'Total frames: {config.total_frames}')
    print(f'Phase boundaries (frame idx): {config.phase_boundaries}')
    print(f'Output dir: {args.out_dir}')


if __name__ == '__main__':
    main()
