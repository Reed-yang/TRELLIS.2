"""
Generate synthetic test models M4-M7 for baseline experiments.

Usage:
    python scripts/eval/baseline_generate_models.py --output-dir results/baseline_experiments/data
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import numpy as np
import trimesh


def generate_bowl(output_path):
    """M4: Bowl — clean open surface with single boundary loop."""
    # Half of a UV sphere (open hemisphere)
    sphere = trimesh.creation.uv_sphere(radius=0.4, count=[32, 32])
    # Keep only top half (z > -0.05 to make a shallow bowl)
    mask = sphere.vertices[:, 2] > -0.05
    face_mask = mask[sphere.faces].all(axis=1)
    bowl = sphere.submesh([np.where(face_mask)[0]], append=True)
    # Center at origin
    bowl.vertices -= bowl.vertices.mean(axis=0)
    scale = 0.99999 / (bowl.vertices.max(0) - bowl.vertices.min(0)).max()
    bowl.vertices *= scale
    bowl.export(output_path)
    print(f"  M4 bowl: {len(bowl.vertices)} verts, {len(bowl.faces)} faces -> {output_path}")


def generate_parallel_planes(output_path_template, resolution):
    """M5: Parallel planes at parameterized gap distances.

    gap d is specified in voxel widths at the given resolution.
    Generates planes for d in {0.5, 1.0, 2.0, 4.0}.
    """
    voxel_size = 1.0 / resolution
    for d in [0.5, 1.0, 2.0, 4.0]:
        gap = d * voxel_size
        # Two axis-aligned square planes centered at origin
        half_size = 0.3
        z_offset = gap / 2
        verts_top = [
            [-half_size, -half_size, z_offset],
            [half_size, -half_size, z_offset],
            [half_size, half_size, z_offset],
            [-half_size, half_size, z_offset],
        ]
        faces_top = [[0, 1, 2], [0, 2, 3]]
        verts_bot = [
            [-half_size, -half_size, -z_offset],
            [half_size, -half_size, -z_offset],
            [half_size, half_size, -z_offset],
            [-half_size, half_size, -z_offset],
        ]
        faces_bot = [[4, 6, 5], [4, 7, 6]]  # flipped normal
        all_verts = np.array(verts_top + verts_bot, dtype=np.float64)
        all_faces = np.array(faces_top + faces_bot, dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=False)
        out_path = output_path_template.format(d=d, res=resolution)
        mesh.export(out_path)
        print(f"  M5 planes(d={d}): gap={gap:.6f} -> {out_path}")


def generate_nested_spheres(output_path):
    """M6: Two nested spheres — tests internal structure preservation."""
    outer = trimesh.creation.icosphere(subdivisions=3, radius=0.4)
    inner = trimesh.creation.icosphere(subdivisions=3, radius=0.2)
    combined = trimesh.util.concatenate([outer, inner])
    combined.export(output_path)
    print(f"  M6 nested: {len(combined.vertices)} verts, {len(combined.faces)} faces -> {output_path}")


def generate_icosphere(output_path):
    """M7: Icosphere — smooth reference / sanity check."""
    ico = trimesh.creation.icosphere(subdivisions=4, radius=0.4)
    ico.export(output_path)
    print(f"  M7 icosphere: {len(ico.vertices)} verts, {len(ico.faces)} faces -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/baseline_experiments/data")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Generating synthetic test models...")
    generate_bowl(os.path.join(args.output_dir, "bowl.obj"))
    for res in [512, 1024]:
        generate_parallel_planes(
            os.path.join(args.output_dir, "parallel_planes_d{d}_res{res}.obj"),
            resolution=res,
        )
    generate_nested_spheres(os.path.join(args.output_dir, "nested_spheres.obj"))
    generate_icosphere(os.path.join(args.output_dir, "icosphere.obj"))
    print("Done.")


if __name__ == "__main__":
    main()
