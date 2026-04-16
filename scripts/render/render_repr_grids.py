"""
Generate improved comparison grids for O-Voxel repr test.

For each model, creates a single large image:
  Rows: GT | Layer A 512 | Layer A 1024 | Layer B 512 | Layer B 1024
  Cols: 8 viewpoints

Uses per-model camera parameters to handle extreme aspect ratios.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
import open3d as o3d

OUTPUT_ROOT = "experiments/ovoxel_repr_test"
NVIEWS = 8
RENDER_RES = 512
MAX_FACES = 500_000

# Per-model camera parameters
CAMERA_PARAMS = {
    "helmet": {"r": 2.0, "fov": 40, "elevation": 15},
    "bugatti": {"r": 3.5, "fov": 20, "elevation": 30},   # farther + higher for flat car
    "spacesuit": {"r": 2.0, "fov": 40, "elevation": 15},
}


def decimate_mesh(tm_mesh, target_faces):
    """Decimate mesh using open3d if above target face count."""
    if len(tm_mesh.faces) <= target_faces:
        return tm_mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(tm_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(tm_mesh.faces)
    o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_faces)
    return trimesh.Trimesh(
        vertices=np.asarray(o3d_mesh.vertices),
        faces=np.asarray(o3d_mesh.triangles),
        process=False,
    )


def render_views(tm_mesh, model_id):
    """Render normal maps from multiple views."""
    from trellis2.representations import Mesh as TrellisMesh
    from trellis2.utils.render_utils import render_snapshot

    tm_mesh = decimate_mesh(tm_mesh, MAX_FACES)

    trellis_mesh = TrellisMesh(
        vertices=torch.from_numpy(tm_mesh.vertices.copy()).float().cuda(),
        faces=torch.from_numpy(tm_mesh.faces.copy()).int().cuda(),
    )

    cam = CAMERA_PARAMS[model_id]
    result = render_snapshot(
        trellis_mesh,
        resolution=RENDER_RES,
        nviews=NVIEWS,
        r=cam["r"], fov=cam["fov"],
        offset=(0, cam["elevation"] / 180 * np.pi),
        return_types=["normal"],
    )

    return [Image.fromarray(nmap) for nmap in result["normal"]]


def make_grid(model_id, rows_config):
    """
    Create a comparison grid.
    rows_config: list of (label, mesh_path_or_None) tuples.
      mesh_path_or_None: path to mesh file, or None to use GT.
    """
    gt_path = os.path.join(OUTPUT_ROOT, "data", f"{model_id}.obj")
    gt_mesh = trimesh.load(gt_path, process=False)

    cell_w, cell_h = RENDER_RES, RENDER_RES
    label_w = 160
    n_rows = len(rows_config)

    grid_w = label_w + cell_w * NVIEWS
    grid_h = cell_h * n_rows

    grid = Image.new("RGB", (grid_w, grid_h), (30, 30, 30))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (IOError, OSError):
        font = ImageFont.load_default()
        font_small = font

    for i, (label, mesh_path) in enumerate(rows_config):
        y = i * cell_h

        # Draw label
        draw.text((8, y + cell_h // 2 - 20), label, fill="white", font=font)

        # Load and render mesh
        if mesh_path is None:
            mesh = gt_mesh
        else:
            if not os.path.exists(mesh_path):
                draw.text((8, y + cell_h // 2 + 5), "(missing)", fill=(150, 150, 150), font=font_small)
                continue
            mesh = trimesh.load(mesh_path, process=False)

        try:
            images = render_views(mesh, model_id)
            for j, img in enumerate(images):
                grid.paste(img.convert("RGB"), (label_w + j * cell_w, y))
        except Exception as e:
            draw.text((8, y + cell_h // 2 + 5), f"render err", fill=(255, 100, 100), font=font_small)
            print(f"  Render failed for {label}: {e}")

    return grid


def main():
    os.makedirs(os.path.join(OUTPUT_ROOT, "previews"), exist_ok=True)

    for model_id in ["helmet", "bugatti", "spacesuit"]:
        print(f"Generating grid for {model_id}...")

        rows = [
            ("GT", None),
            ("A @512", os.path.join(OUTPUT_ROOT, "layer_a", f"{model_id}_512", "recon.ply")),
            ("A @1024", os.path.join(OUTPUT_ROOT, "layer_a", f"{model_id}_1024", "recon.ply")),
            ("B @512", os.path.join(OUTPUT_ROOT, "layer_b", f"{model_id}_512", "recon.ply")),
            ("B @1024", os.path.join(OUTPUT_ROOT, "layer_b", f"{model_id}_1024", "recon.ply")),
        ]

        grid = make_grid(model_id, rows)
        out_path = os.path.join(OUTPUT_ROOT, "previews", f"{model_id}_comparison.png")
        grid.save(out_path)
        print(f"  Saved: {out_path} ({grid.size[0]}x{grid.size[1]})")


if __name__ == "__main__":
    main()
