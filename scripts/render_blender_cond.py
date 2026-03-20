# scripts/render_blender_cond.py
"""
Render Blender CYCLES conditioning images for pilot meshes.
Calls the existing data_toolkit/blender_script/render_cond.py via Blender subprocess.
Produces images identical to the DiT training distribution.
"""

import os
import sys
import json
import math
import argparse
import numpy as np
from subprocess import call, DEVNULL
from tqdm import tqdm
from PIL import Image

# Blender path (same as data_toolkit/render_cond.py)
BLENDER_PATH = '/tmp/blender-3.0.1-linux-x64/blender'
BLENDER_SCRIPT = os.path.join(os.path.dirname(__file__), '..', 'data_toolkit', 'blender_script', 'render_cond.py')

# Camera distribution functions (copied from data_toolkit/utils.py to avoid import issues)
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

def hammersley_sequence(dim, n, num_samples):
    return [n / num_samples] + [radical_inverse(PRIMES[i], n) for i in range(dim - 1)]

def sphere_hammersley_sequence(n, num_samples, offset=(0, 0)):
    u, v = hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return [phi, theta]


def build_cond_views(num_views=16):
    """Build camera view parameters matching training distribution."""
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y)
        pitchs.append(p)

    fov_min, fov_max = 10, 70
    radius_min = np.sqrt(3) / 2 / np.sin(fov_max / 360 * np.pi)
    radius_max = np.sqrt(3) / 2 / np.sin(fov_min / 360 * np.pi)
    k_min = 1 / radius_max**2
    k_max = 1 / radius_min**2
    ks = np.random.uniform(k_min, k_max, (1000000,))
    radius = [1 / np.sqrt(k) for k in ks]
    fov = [2 * np.arcsin(np.sqrt(3) / 2 / r) for r in radius]

    return [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f}
            for y, p, r, f in zip(yaws, pitchs, radius, fov)]


def render_single_mesh(mesh_path, uid, output_root, num_views=16, resolution=1024):
    """
    Render conditioning images for a single mesh using Blender CYCLES.

    Args:
        mesh_path: path to .glb/.obj mesh file
        uid: unique identifier for this mesh
        output_root: root directory for renders_cond/
        num_views: number of views to render (default 16, matches training)
        resolution: image resolution (default 1024, matches training)

    Returns:
        True if successful, False otherwise
    """
    output_folder = os.path.join(output_root, 'renders_cond', uid)
    os.makedirs(output_folder, exist_ok=True)

    # Skip if already rendered
    if os.path.exists(os.path.join(output_folder, 'transforms.json')):
        return True

    # Resolve symlinks
    mesh_path = os.path.realpath(mesh_path)

    # Build camera views
    cond_views = build_cond_views(num_views)

    args = [
        BLENDER_PATH, '-b', '-P', os.path.realpath(BLENDER_SCRIPT),
        '--',
        '--object', mesh_path,
        '--cond_views', json.dumps(cond_views),
        '--cond_resolution', str(resolution),
        '--cond_output_folder', output_folder,
        '--engine', 'CYCLES',
    ]

    ret = call(args, stdout=DEVNULL, stderr=DEVNULL)
    return os.path.exists(os.path.join(output_folder, 'transforms.json'))


def pick_best_view(renders_dir, uid):
    """
    Pick the best front-facing view from rendered views using camera parameters.
    Scoring: prefer low |elevation|, azimuth near 0° or 180°, FOV in [30°, 50°].
    Falls back to 000.png if transforms.json is missing.
    """
    transforms_path = os.path.join(renders_dir, uid, 'transforms.json')
    if not os.path.exists(transforms_path):
        fallback = os.path.join(renders_dir, uid, '000.png')
        return fallback if os.path.exists(fallback) else None

    with open(transforms_path) as f:
        data = json.load(f)

    best_view = None
    best_score = float('inf')

    for frame in data.get('frames', []):
        m = frame['transform_matrix']
        cam_pos = [m[0][3], m[1][3], m[2][3]]
        r = math.sqrt(sum(c**2 for c in cam_pos))
        if r < 1e-6:
            continue

        elev_deg = math.degrees(math.asin(cam_pos[2] / r))
        azim_deg = math.degrees(math.atan2(cam_pos[1], cam_pos[0]))
        fov_deg = math.degrees(frame.get('camera_angle_x', 0.7))

        # Prefer: |elevation| small, azimuth near 0° or ±180°, FOV near 40°
        azim_front = min(abs(azim_deg), abs(abs(azim_deg) - 180))
        score = abs(elev_deg) * 1.0 + azim_front * 0.5 + abs(fov_deg - 40) * 0.3

        if score < best_score:
            best_score = score
            best_view = frame['file_path']

    if best_view:
        path = os.path.join(renders_dir, uid, best_view)
        if os.path.exists(path):
            return path

    fallback = os.path.join(renders_dir, uid, '000.png')
    return fallback if os.path.exists(fallback) else None


def render_from_manifest(manifest_path, output_dir, num_views=16, rank=0, world_size=1):
    """
    Render conditioning images for all assets in a manifest.
    Each asset gets num_views Blender CYCLES renders at 1024x1024.
    Supports multi-process parallelism via rank/world_size sharding.
    """
    with open(manifest_path) as f:
        items = json.load(f)

    # Shard for parallel execution
    if world_size > 1:
        start = len(items) * rank // world_size
        end = len(items) * (rank + 1) // world_size
        items = items[start:end]

    print(f"Rendering {len(items)} assets (rank {rank}/{world_size}), {num_views} views each")

    for item in tqdm(items, desc="Blender Rendering"):
        uid = item['uid']
        mesh_path = item['mesh_path']
        asset_dir = os.path.join(output_dir, uid)

        # Skip if already rendered (check last view exists)
        if os.path.exists(os.path.join(asset_dir, f"{num_views-1:04d}.png")):
            continue

        os.makedirs(asset_dir, exist_ok=True)

        # Call existing Blender subprocess
        cmd = [
            BLENDER_PATH, '--background', '--python', BLENDER_SCRIPT,
            '--', mesh_path, asset_dir,
            '--num_views', str(num_views),
            '--resolution', '1024',
        ]
        try:
            call(cmd, stdout=DEVNULL, stderr=DEVNULL, timeout=300)
        except Exception as e:
            print(f"  Render failed for {uid}: {e}")


def update_manifest_with_renders(manifest_path, output_dir, num_views=16):
    """Update manifest.json with renders_dir and num_views fields."""
    with open(manifest_path) as f:
        items = json.load(f)
    for item in items:
        item['renders_dir'] = os.path.join(os.path.abspath(output_dir), item['uid'])
        item['num_views'] = num_views
    with open(manifest_path, 'w') as f:
        json.dump(items, f, indent=2)
    print(f"Updated {len(items)} entries in {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Render Blender conditioning images for pilot meshes")
    parser.add_argument("--manifest", type=str,
                        help="Path to pilot data manifest.json")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/gap_measurement_blender",
                        help="Output directory for experiment")
    parser.add_argument("--render_output", type=str,
                        default="experiments/component_eval/renders_cond",
                        help="Output directory for manifest-based batch rendering")
    parser.add_argument("--num_views", type=int, default=16,
                        help="Number of views per model (default: 16, matches training)")
    parser.add_argument("--resolution", type=int, default=1024,
                        help="Render resolution (default: 1024, matches training)")
    parser.add_argument("--rank", type=int, default=0,
                        help="Process rank for multi-process parallelism")
    parser.add_argument("--world_size", type=int, default=1,
                        help="Total number of processes for parallelism")
    args = parser.parse_args()

    # Manifest-based batch rendering mode (component eval pipeline)
    if args.manifest:
        render_from_manifest(args.manifest, args.render_output, args.num_views, args.rank, args.world_size)
        return

    # Verify Blender is installed
    if not os.path.exists(BLENDER_PATH):
        print(f"ERROR: Blender not found at {BLENDER_PATH}")
        print("Install with: data_toolkit/render_cond.py's _install_blender() or manually")
        sys.exit(1)

    # Load manifest
    with open(args.manifest) as f:
        manifest = json.load(f)

    renders_dir = os.path.join(args.output_dir, 'renders_cond')
    os.makedirs(renders_dir, exist_ok=True)

    # Render each mesh
    successes = 0
    failures = []
    for item in tqdm(manifest, desc="Rendering with Blender"):
        uid = item['uid']
        mesh_path = item['mesh_path']

        ok = render_single_mesh(mesh_path, uid, args.output_dir,
                                num_views=args.num_views,
                                resolution=args.resolution)
        if ok:
            successes += 1
        else:
            failures.append(uid)
            print(f"  FAILED: {uid}")

    print(f"\nRendering complete: {successes}/{len(manifest)} succeeded")
    if failures:
        print(f"Failed UIDs: {failures}")

    # Create updated manifest with Blender image paths
    updated_manifest = []
    for item in manifest:
        uid = item['uid']
        blender_image = pick_best_view(renders_dir, uid)
        if blender_image:
            updated_item = dict(item)
            updated_item['image_path'] = os.path.abspath(blender_image)
            updated_item['image_source'] = 'blender_cycles'
            updated_manifest.append(updated_item)
        else:
            print(f"  Skipping {uid}: no rendered image")

    # Save updated manifest
    manifest_dir = os.path.join(args.output_dir, 'pilot_data')
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_path = os.path.join(manifest_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(updated_manifest, f, indent=2)

    print(f"Updated manifest: {manifest_path} ({len(updated_manifest)} models)")


if __name__ == "__main__":
    main()
