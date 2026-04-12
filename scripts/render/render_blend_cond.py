"""
Render conditioning images from .blend files (with PBR materials) using Blender CYCLES.

Unlike render_blender_cond.py which uses .obj (no materials), this script opens
.blend files directly so CYCLES renders with full PBR textures.

Usage:
    python scripts/render/render_blend_cond.py \
        --manifest experiments/component_eval/test_set/manifest_pbr.json \
        --blend_dir datasets/Toys4k/toys4k_blend_files \
        --output_dir experiments/component_eval/renders_cond_pbr \
        --num_views 16 --rank 0 --world_size 4
"""

import os
import sys
import json
import math
import argparse
import numpy as np
from subprocess import call, DEVNULL, STDOUT
from tqdm import tqdm

BLENDER_PATH = '/tmp/blender-3.0.1-linux-x64/blender'
BLENDER_SCRIPT = os.path.join(os.path.dirname(__file__), '..', '..', 'data_toolkit', 'blender_script', 'render_cond.py')

# Camera distribution (same as render_blender_cond.py)
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

def radical_inverse(base, n):
    val = 0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val

def halton_sequence(dim, n):
    return radical_inverse(PRIMES[dim], n)

def build_cond_views(num_views):
    views = []
    for i in range(num_views):
        yaw = halton_sequence(0, i) * 2 * math.pi
        pitch_raw = halton_sequence(1, i)
        pitch = math.asin(pitch_raw * 2 - 1)
        fov_raw = halton_sequence(2, i)
        fov_deg = 10 + fov_raw * 60  # 10-70 degrees
        fov_rad = math.radians(fov_deg)
        radius = 1.5 / math.tan(fov_rad / 2)
        views.append({
            'yaw': yaw,
            'pitch': pitch,
            'radius': radius,
            'fov': fov_rad,
        })
    return views


def render_single_blend(blend_path, uid, output_dir, num_views=16, resolution=1024):
    """Render conditioning images from a .blend file."""
    asset_dir = os.path.join(output_dir, uid)
    os.makedirs(asset_dir, exist_ok=True)

    # Skip if already rendered
    if os.path.exists(os.path.join(asset_dir, 'transforms.json')):
        return True

    cond_views = build_cond_views(num_views)

    # Key difference: pass blend file as BOTH the Blender input file AND --object
    # This makes Blender open the .blend (loading all materials), then render_cond.py
    # skips init_scene/load_object and just calls delete_invisible_objects()
    cmd = [
        BLENDER_PATH,
        os.path.realpath(blend_path),  # Open blend file directly
        '-b',
        '-P', os.path.realpath(BLENDER_SCRIPT),
        '--',
        '--object', os.path.realpath(blend_path),
        '--cond_views', json.dumps(cond_views),
        '--cond_resolution', str(resolution),
        '--cond_output_folder', asset_dir,
        '--engine', 'CYCLES',
    ]

    ret = call(cmd, stdout=DEVNULL, stderr=DEVNULL)
    return os.path.exists(os.path.join(asset_dir, 'transforms.json'))


def main():
    parser = argparse.ArgumentParser(description="Render conditioning images from .blend files")
    parser.add_argument("--manifest", required=True, help="Path to PBR manifest JSON")
    parser.add_argument("--blend_dir", default="datasets/Toys4k/toys4k_blend_files",
                        help="Root directory of .blend files")
    parser.add_argument("--output_dir", default="experiments/component_eval/renders_cond_pbr",
                        help="Output directory for rendered images")
    parser.add_argument("--num_views", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()

    # Load manifest
    with open(args.manifest) as f:
        manifest = json.load(f)

    # Map UIDs to blend files
    import glob
    blend_files = {}
    for bf in glob.glob(os.path.join(args.blend_dir, '*', '*', '*.blend')):
        parts = bf.split(os.sep)
        uid = parts[-2]
        blend_files[uid] = bf

    # Shard
    start = len(manifest) * args.rank // args.world_size
    end = len(manifest) * (args.rank + 1) // args.world_size
    shard = manifest[start:end]
    print(f"Rank {args.rank}/{args.world_size}: rendering {len(shard)} assets (index {start}-{end})")

    os.makedirs(args.output_dir, exist_ok=True)

    success = 0
    fail = 0
    for item in tqdm(shard, desc=f"Rank {args.rank}"):
        uid = item['uid']
        blend_path = blend_files.get(uid)
        if blend_path is None:
            print(f"  No blend file for {uid}, skipping")
            fail += 1
            continue

        ok = render_single_blend(blend_path, uid, args.output_dir,
                                  args.num_views, args.resolution)
        if ok:
            success += 1
        else:
            print(f"  FAILED: {uid}")
            fail += 1

    print(f"Rank {args.rank} done: {success} success, {fail} fail")


if __name__ == "__main__":
    main()
