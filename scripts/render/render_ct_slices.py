"""
CT-scan style cross-section rendering for O-Voxel reconstructed meshes.

Renders cross-section contours and clipping-plane normal maps at multiple
heights through the mesh, revealing internal structure.

Usage:
    python scripts/render/render_ct_slices.py --model helmet --layer b --res 512
    python scripts/render/render_ct_slices.py --model helmet --layer b --res 512 1024 --n-slices 12
    python scripts/render/render_ct_slices.py --all
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

OUTPUT_ROOT = "experiments/ovoxel_repr_test"
MODELS = ["helmet", "bugatti", "spacesuit"]
RESOLUTIONS = [512, 1024]


# ---------------------------------------------------------------------------
# Cross-section contour rendering (2D slices like CT)
# ---------------------------------------------------------------------------

def get_cross_section_contours(mesh, axis, position):
    """Get cross-section contour paths at a given position along an axis."""
    origin = [0, 0, 0]
    normal = [0, 0, 0]
    origin[axis] = position
    normal[axis] = 1.0
    try:
        section = mesh.section(plane_origin=origin, plane_normal=normal)
        if section is None:
            return None
        return section
    except Exception:
        return None


def render_contour_slice(mesh, axis, position, img_size=512, padding=20):
    """Render a single cross-section contour as a PIL image."""
    section = get_cross_section_contours(mesh, axis, position)

    img = Image.new("RGB", (img_size, img_size), (20, 20, 20))
    draw = ImageDraw.Draw(img)

    if section is None:
        return img

    # Get 2D path by projecting onto the plane perpendicular to axis
    try:
        planar, transform = section.to_planar()
    except Exception:
        return img

    if len(planar.entities) == 0:
        return img

    # Map 2D coords to image space
    # The mesh is in [-0.5, 0.5], so map that to image coords
    usable = img_size - 2 * padding
    scale = usable / 1.0  # [-0.5, 0.5] → [0, usable]

    for entity in planar.entities:
        points_2d = planar.vertices[entity.points]
        # Map from [-0.5, 0.5] to pixel coords
        px = (points_2d[:, 0] + 0.5) * scale + padding
        py = (0.5 - points_2d[:, 1]) * scale + padding  # flip Y
        coords = list(zip(px.tolist(), py.tolist()))
        if len(coords) >= 2:
            draw.line(coords, fill=(0, 220, 180), width=2)

    return img


def render_ct_grid(mesh, n_slices=8, axis=1, img_size=256):
    """Render a grid of cross-section contours along an axis (like CT scan)."""
    bbox = mesh.bounding_box.bounds
    lo, hi = bbox[0][axis], bbox[1][axis]
    margin = (hi - lo) * 0.05
    positions = np.linspace(lo + margin, hi - margin, n_slices)

    slices = []
    for pos in positions:
        img = render_contour_slice(mesh, axis, pos, img_size=img_size)
        slices.append(img)

    return slices, positions


# ---------------------------------------------------------------------------
# Clipping-plane 3D rendering (half-mesh normal maps)
# ---------------------------------------------------------------------------

MAX_RENDER_FACES_CT = 200_000

def render_clipped_normal_maps(mesh, clip_axis, clip_positions, view_axis=0,
                               resolution=512):
    """Render normal maps of mesh clipped at various positions.

    Clips the mesh by removing all faces whose centroid is beyond clip_position
    along clip_axis, then renders from a fixed viewpoint along view_axis.
    """
    import torch
    from trellis2.representations import Mesh as TrellisMesh
    from trellis2.utils.render_utils import render_snapshot

    images = []
    verts_np = mesh.vertices.copy()
    faces_np = mesh.faces.copy()
    centroids = verts_np[faces_np].mean(axis=1)  # [F, 3]

    for clip_pos in clip_positions:
        # Keep faces whose centroid is below clip_pos on clip_axis
        mask = centroids[:, clip_axis] < clip_pos
        kept_faces = faces_np[mask]

        if len(kept_faces) < 10:
            images.append(Image.new("RGB", (resolution, resolution), (40, 40, 40)))
            continue

        # Reindex vertices
        used_verts = np.unique(kept_faces.flatten())
        vert_map = np.full(len(verts_np), -1, dtype=np.int64)
        vert_map[used_verts] = np.arange(len(used_verts))
        new_verts = verts_np[used_verts]
        new_faces = vert_map[kept_faces]

        # Decimate if too large
        if len(new_faces) > MAX_RENDER_FACES_CT:
            import open3d as o3d
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(new_verts)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(new_faces)
            o3d_mesh = o3d_mesh.simplify_quadric_decimation(MAX_RENDER_FACES_CT)
            new_verts = np.asarray(o3d_mesh.vertices)
            new_faces = np.asarray(o3d_mesh.triangles)

        torch.cuda.empty_cache()

        trellis_mesh = TrellisMesh(
            vertices=torch.from_numpy(new_verts).float().cuda(),
            faces=torch.from_numpy(new_faces).int().cuda(),
        )

        try:
            if view_axis == 0:
                yaw_offset = 0
            elif view_axis == 1:
                yaw_offset = np.pi / 2
            else:
                yaw_offset = 0

            result = render_snapshot(
                trellis_mesh,
                resolution=resolution,
                nviews=1,
                r=2, fov=40,
                offset=(yaw_offset, 15 / 180 * np.pi),
                return_types=["normal"],
            )
            images.append(Image.fromarray(result["normal"][0]))
        except Exception as e:
            print(f"    Render failed at clip={clip_pos:.3f}: {e}")
            images.append(Image.new("RGB", (resolution, resolution), (40, 40, 40)))
        finally:
            del trellis_mesh
            torch.cuda.empty_cache()

    return images


# ---------------------------------------------------------------------------
# Grid assembly
# ---------------------------------------------------------------------------

def assemble_ct_grid(slice_images, positions, title, cell_size=256):
    """Assemble slice images into a labeled grid."""
    n = len(slice_images)
    cols = min(n, 8)
    rows = (n + cols - 1) // cols

    label_h = 24
    grid_w = cols * cell_size
    grid_h = rows * (cell_size + label_h) + 40  # +40 for title

    grid = Image.new("RGB", (grid_w, grid_h), (30, 30, 30))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except (IOError, OSError):
        font = ImageFont.load_default()
        title_font = font

    # Title
    draw.text((10, 8), title, fill="white", font=title_font)

    for idx, (img, pos) in enumerate(zip(slice_images, positions)):
        row = idx // cols
        col = idx % cols
        x = col * cell_size
        y = row * (cell_size + label_h) + 40

        resized = img.resize((cell_size, cell_size), Image.LANCZOS)
        grid.paste(resized.convert("RGB"), (x, y))

        label = f"y={pos:.3f}"
        draw.text((x + 4, y + cell_size + 2), label, fill=(180, 180, 180), font=font)

    return grid


def assemble_comparison_grid(gt_slices, recon_slices_dict, positions, model_id,
                             layer, cell_size=192):
    """Assemble GT vs recon comparison: rows=[GT, res1, res2], cols=slice positions."""
    n_slices = len(gt_slices)
    resolutions = sorted(recon_slices_dict.keys())
    n_rows = 1 + len(resolutions)

    label_w = 80
    grid_w = label_w + n_slices * cell_size
    grid_h = n_rows * (cell_size + 4) + 50

    grid = Image.new("RGB", (grid_w, grid_h), (30, 30, 30))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except (IOError, OSError):
        font = ImageFont.load_default()
        title_font = font

    title = f"CT Slices: {model_id} (Layer {layer.upper()})"
    draw.text((10, 8), title, fill="white", font=title_font)

    # Column headers (slice positions)
    for j, pos in enumerate(positions):
        x = label_w + j * cell_size + cell_size // 2 - 20
        draw.text((x, 30), f"{pos:.2f}", fill=(150, 150, 150), font=font)

    y_offset = 50
    # GT row
    draw.text((5, y_offset + cell_size // 2 - 8), "GT", fill="white", font=font)
    for j, img in enumerate(gt_slices):
        resized = img.resize((cell_size, cell_size), Image.LANCZOS)
        grid.paste(resized.convert("RGB"), (label_w + j * cell_size, y_offset))
    y_offset += cell_size + 4

    # Recon rows
    for res in resolutions:
        draw.text((5, y_offset + cell_size // 2 - 8), str(res), fill="white", font=font)
        for j, img in enumerate(recon_slices_dict[res]):
            resized = img.resize((cell_size, cell_size), Image.LANCZOS)
            grid.paste(resized.convert("RGB"), (label_w + j * cell_size, y_offset))
        y_offset += cell_size + 4

    return grid


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def render_model_ct(model_id, layer, resolutions, n_slices=8, clip_axis=1):
    """Render CT-style views for one model across resolutions."""
    print(f"\n{'='*60}")
    print(f"CT Rendering: {model_id}, Layer {layer}, axis={clip_axis}")
    print(f"{'='*60}")

    # Load GT mesh
    gt_path = os.path.join(OUTPUT_ROOT, "data", f"{model_id}.obj")
    if not os.path.exists(gt_path):
        print(f"  GT mesh not found: {gt_path}")
        return
    gt_mesh = trimesh.load(gt_path, process=False)
    print(f"  GT: {len(gt_mesh.vertices)} verts, {len(gt_mesh.faces)} faces")

    # Determine slice positions from GT bounding box
    bbox = gt_mesh.bounding_box.bounds
    lo, hi = bbox[0][clip_axis], bbox[1][clip_axis]
    margin = (hi - lo) * 0.05
    positions = np.linspace(lo + margin, hi - margin, n_slices)

    out_dir = os.path.join(OUTPUT_ROOT, "previews", "ct_slices")
    os.makedirs(out_dir, exist_ok=True)

    # --- Contour CT slices ---
    print(f"  Rendering GT contour slices...")
    gt_contour_slices = []
    for pos in positions:
        gt_contour_slices.append(render_contour_slice(gt_mesh, clip_axis, pos, img_size=256))

    recon_contour_slices = {}
    for res in resolutions:
        recon_path = os.path.join(OUTPUT_ROOT, f"layer_{layer}", f"{model_id}_{res}", "recon.ply")
        if not os.path.exists(recon_path):
            print(f"  Recon not found: {recon_path}")
            continue
        recon_mesh = trimesh.load(recon_path, process=False)
        print(f"  [{res}] Rendering contour slices... ({len(recon_mesh.faces)} faces)")
        slices = []
        for pos in positions:
            slices.append(render_contour_slice(recon_mesh, clip_axis, pos, img_size=256))
        recon_contour_slices[res] = slices

    if recon_contour_slices:
        contour_grid = assemble_comparison_grid(
            gt_contour_slices, recon_contour_slices, positions,
            model_id, layer, cell_size=192,
        )
        contour_path = os.path.join(out_dir, f"{model_id}_layer{layer}_contours.png")
        contour_grid.save(contour_path)
        print(f"  Saved: {contour_path}")

    # --- Clipping-plane 3D normal maps ---
    # Use fewer slices for 3D rendering (expensive)
    clip_positions = positions[::max(1, len(positions) // 6)][:6]
    print(f"  Rendering GT clipping-plane normal maps ({len(clip_positions)} slices)...")
    gt_clip_images = render_clipped_normal_maps(
        gt_mesh, clip_axis, clip_positions, resolution=384,
    )

    recon_clip_images = {}
    for res in resolutions:
        recon_path = os.path.join(OUTPUT_ROOT, f"layer_{layer}", f"{model_id}_{res}", "recon.ply")
        if not os.path.exists(recon_path):
            continue
        recon_mesh = trimesh.load(recon_path, process=False)
        print(f"  [{res}] Rendering clipping-plane normal maps...")
        imgs = render_clipped_normal_maps(
            recon_mesh, clip_axis, clip_positions, resolution=384,
        )
        recon_clip_images[res] = imgs

    if recon_clip_images:
        clip_grid = assemble_comparison_grid(
            gt_clip_images, recon_clip_images, clip_positions,
            model_id, f"{layer}_clip", cell_size=256,
        )
        clip_path = os.path.join(out_dir, f"{model_id}_layer{layer}_clipped.png")
        clip_grid.save(clip_path)
        print(f"  Saved: {clip_path}")


def main():
    parser = argparse.ArgumentParser(description="CT-scan style cross-section rendering")
    parser.add_argument("--model", nargs="+", default=MODELS,
                        help="Models to render")
    parser.add_argument("--layer", choices=["a", "b"], default="b",
                        help="Layer to render")
    parser.add_argument("--res", nargs="+", type=int, default=RESOLUTIONS,
                        help="Resolutions to compare")
    parser.add_argument("--n-slices", type=int, default=8,
                        help="Number of cross-section slices")
    parser.add_argument("--axis", type=int, default=1, choices=[0, 1, 2],
                        help="Clipping axis (0=X, 1=Y, 2=Z)")
    parser.add_argument("--all", action="store_true",
                        help="Render all models, both layers")
    args = parser.parse_args()

    if args.all:
        for model_id in MODELS:
            for layer in ["a", "b"]:
                render_model_ct(model_id, layer, args.res, args.n_slices, args.axis)
    else:
        for model_id in args.model:
            render_model_ct(model_id, args.layer, args.res, args.n_slices, args.axis)


if __name__ == "__main__":
    main()
