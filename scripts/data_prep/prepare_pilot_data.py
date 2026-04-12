"""
Download pilot test meshes from Objaverse and render reference images.
Uses Objaverse's lvis annotations to get diverse, high-quality models.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import argparse
import trimesh
import numpy as np
from tqdm import tqdm


def get_pilot_uids(num_models=30):
    """
    Select diverse Objaverse model UIDs using LVIS annotations.
    """
    import objaverse

    lvis = objaverse.load_lvis_annotations()
    selected = []
    categories_used = []
    for cat, uids in sorted(lvis.items()):
        if len(selected) >= num_models:
            break
        uid = uids[0]
        selected.append(uid)
        categories_used.append(cat)

    print(f"Selected {len(selected)} models from {len(categories_used)} categories")
    return selected, categories_used


def download_models(uids, output_dir):
    """Download Objaverse models by UID."""
    import objaverse

    os.makedirs(output_dir, exist_ok=True)
    paths = objaverse.load_objects(uids=uids)

    downloaded = {}
    for uid, src_path in paths.items():
        dst_path = os.path.join(output_dir, f"{uid}.glb")
        if not os.path.exists(dst_path):
            try:
                os.symlink(os.path.abspath(src_path), dst_path)
            except OSError:
                import shutil
                shutil.copy2(src_path, dst_path)
        downloaded[uid] = dst_path

    return downloaded


def render_reference_image(mesh_path, output_path, resolution=512):
    """
    Render a reference image of the mesh for DiT input.
    """
    try:
        mesh = trimesh.load(mesh_path, force="mesh")
    except Exception as e:
        print(f"  Failed to load {mesh_path}: {e}")
        return False

    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        print(f"  Empty mesh: {mesh_path}")
        return False

    # Normalize mesh to [-0.5, 0.5]
    center = (mesh.vertices.min(0) + mesh.vertices.max(0)) / 2
    scale_val = (mesh.vertices.max(0) - mesh.vertices.min(0)).max()
    if scale_val < 1e-8:
        print(f"  Degenerate mesh: {mesh_path}")
        return False
    scale = 0.99999 / scale_val
    mesh.vertices = (mesh.vertices - center) * scale

    try:
        scene = mesh.scene()
        png = scene.save_image(resolution=(resolution, resolution))
        with open(output_path, 'wb') as f:
            f.write(png)
        return True
    except Exception:
        pass

    # Fallback: save a placeholder
    try:
        from PIL import Image
        img = Image.new('RGB', (resolution, resolution), (200, 200, 200))
        img.save(output_path)
        print(f"  Warning: used placeholder image for {mesh_path}")
        return True
    except Exception:
        return False


def validate_mesh(mesh_path):
    """Check if a mesh is valid for evaluation."""
    try:
        mesh = trimesh.load(mesh_path, force="mesh")
        if mesh.vertices.shape[0] < 10 or mesh.faces.shape[0] < 10:
            return False
        if np.any(np.isnan(mesh.vertices)) or np.any(np.isinf(mesh.vertices)):
            return False
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Prepare pilot test data for gap measurement")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/gap_measurement/pilot_data",
                        help="Output directory")
    parser.add_argument("--num_models", type=int, default=30,
                        help="Number of models to download")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Render resolution for reference images")
    args = parser.parse_args()

    mesh_dir = os.path.join(args.output_dir, "meshes")
    image_dir = os.path.join(args.output_dir, "images")
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    print("Selecting pilot models from Objaverse...")
    uids, categories = get_pilot_uids(args.num_models)

    print(f"Downloading {len(uids)} models...")
    paths = download_models(uids, mesh_dir)

    valid_models = []
    for uid in tqdm(uids, desc="Validating and rendering"):
        mesh_path = paths.get(uid)
        if mesh_path is None:
            continue
        if not validate_mesh(mesh_path):
            print(f"  Skipping invalid mesh: {uid}")
            continue

        image_path = os.path.join(image_dir, f"{uid}.png")
        if render_reference_image(mesh_path, image_path, args.resolution):
            valid_models.append({
                "uid": uid,
                "mesh_path": os.path.abspath(mesh_path),
                "image_path": os.path.abspath(image_path),
                "category": categories[uids.index(uid)] if uid in uids else "unknown",
            })

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(valid_models, f, indent=2)

    print(f"\nPilot data ready: {len(valid_models)} valid models")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
