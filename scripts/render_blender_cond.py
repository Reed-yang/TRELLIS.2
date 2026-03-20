# scripts/render_blender_cond.py
"""
Render Blender CYCLES conditioning images for pilot meshes.
Calls the existing data_toolkit/blender_script/render_cond.py via Blender subprocess.
Produces images identical to the DiT training distribution.
"""

import os
import sys
import json
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
    Pick the first rendered view as conditioning image.
    Returns path to the selected PNG, or None if not found.
    """
    view_path = os.path.join(renders_dir, uid, '000.png')
    if os.path.exists(view_path):
        return view_path
    return None


def main():
    parser = argparse.ArgumentParser(description="Render Blender conditioning images for pilot meshes")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to pilot data manifest.json")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/gap_measurement_blender",
                        help="Output directory for experiment")
    parser.add_argument("--num_views", type=int, default=16,
                        help="Number of views per model (default: 16, matches training)")
    parser.add_argument("--resolution", type=int, default=1024,
                        help="Render resolution (default: 1024, matches training)")
    args = parser.parse_args()

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
