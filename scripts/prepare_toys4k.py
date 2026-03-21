"""
Prepare Toys4k-PBR test set for component evaluation.

Downloads/processes the Toys4k dataset, applies PBR material filtering
(base_color + metallic + roughness), computes complexity metrics, and
outputs manifest.json + metadata.json for downstream evaluation.

Toys4k dataset: https://github.com/rehg-lab/lowshot-shapebias
The dataset contains ~3,229 toy assets; after PBR filtering ~473 remain.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import argparse
import trimesh
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


MESH_EXTENSIONS = {'.glb', '.gltf', '.obj'}


def discover_meshes(data_dir):
    """
    Recursively discover all mesh files under data_dir.
    Returns list of (mesh_path, category) tuples.
    Category is inferred from the parent directory name.
    """
    meshes = []
    data_path = Path(data_dir)

    for ext in MESH_EXTENSIONS:
        for mesh_path in sorted(data_path.rglob(f"*{ext}")):
            # Infer category from parent folder name
            rel = mesh_path.relative_to(data_path)
            parts = rel.parts
            category = parts[0] if len(parts) > 1 else "unknown"
            meshes.append((str(mesh_path), category))

    # Deduplicate by stem (prefer .glb > .gltf > .obj)
    seen_stems = {}
    priority = {'.glb': 0, '.gltf': 1, '.obj': 2}
    for mesh_path, category in meshes:
        p = Path(mesh_path)
        stem_key = str(p.parent / p.stem)
        ext_priority = priority.get(p.suffix.lower(), 99)
        if stem_key not in seen_stems or ext_priority < seen_stems[stem_key][1]:
            seen_stems[stem_key] = ((mesh_path, category), ext_priority)

    return [v[0] for v in sorted(seen_stems.values(), key=lambda x: x[0][0])]


def check_pbr_materials(mesh_path):
    """
    Check whether a mesh has complete PBR metallic-roughness materials.

    A mesh passes the PBR filter if at least one geometry/material satisfies:
      - has baseColorTexture OR baseColorFactor
      - has metallicFactor, roughnessFactor, OR metallicRoughnessTexture

    Returns (has_pbr: bool, material_info: dict).
    """
    try:
        scene = trimesh.load(mesh_path, process=False)
    except Exception as e:
        return False, {"error": str(e)}

    materials = []

    # Collect materials from scene or single mesh
    if isinstance(scene, trimesh.Scene):
        for geom_name, geom in scene.geometry.items():
            if hasattr(geom, 'visual') and hasattr(geom.visual, 'material'):
                materials.append(geom.visual.material)
    elif hasattr(scene, 'visual') and hasattr(scene.visual, 'material'):
        materials.append(scene.visual.material)

    if not materials:
        return False, {"reason": "no_materials_found"}

    has_base_color = False
    has_metallic_roughness = False

    for mat in materials:
        if not isinstance(mat, trimesh.visual.material.PBRMaterial):
            continue

        # Check base color: texture or factor
        bc_texture = getattr(mat, 'baseColorTexture', None)
        bc_factor = getattr(mat, 'baseColorFactor', None)
        if bc_texture is not None or bc_factor is not None:
            has_base_color = True

        # Check metallic-roughness: factors or combined texture
        mr_texture = getattr(mat, 'metallicRoughnessTexture', None)
        m_factor = getattr(mat, 'metallicFactor', None)
        r_factor = getattr(mat, 'roughnessFactor', None)
        if mr_texture is not None or m_factor is not None or r_factor is not None:
            has_metallic_roughness = True

    info = {
        "num_materials": len(materials),
        "has_base_color": has_base_color,
        "has_metallic_roughness": has_metallic_roughness,
    }
    return (has_base_color and has_metallic_roughness), info


def compute_mesh_complexity(mesh_path):
    """
    Load mesh and compute complexity metrics.
    Returns dict with face_count, vertex_count, or None on failure.
    """
    try:
        scene = trimesh.load(mesh_path, force="mesh", process=False)
    except Exception:
        # Try loading as scene and concatenate
        try:
            scene_obj = trimesh.load(mesh_path, process=False)
            if isinstance(scene_obj, trimesh.Scene):
                meshes = [g for g in scene_obj.geometry.values()
                          if isinstance(g, trimesh.Trimesh)]
                if not meshes:
                    return None
                scene = trimesh.util.concatenate(meshes)
            else:
                return None
        except Exception:
            return None

    if not isinstance(scene, trimesh.Trimesh):
        return None

    if scene.vertices.shape[0] == 0 or scene.faces.shape[0] == 0:
        return None

    return {
        "face_count": int(scene.faces.shape[0]),
        "vertex_count": int(scene.vertices.shape[0]),
    }


def assign_complexity_tiers(entries):
    """
    Assign complexity tier (1=low, 2=medium, 3=high) based on face count terciles.
    Modifies entries in-place.
    """
    if not entries:
        return

    face_counts = sorted([e["face_count"] for e in entries])
    n = len(face_counts)
    t1 = face_counts[n // 3]
    t2 = face_counts[2 * n // 3]

    for entry in entries:
        fc = entry["face_count"]
        if fc <= t1:
            entry["tier"] = 1
        elif fc <= t2:
            entry["tier"] = 2
        else:
            entry["tier"] = 3

    return t1, t2


def build_metadata(entries, tier_thresholds, total_discovered, total_pbr_passed):
    """Build summary metadata dict."""
    face_counts = [e["face_count"] for e in entries]
    vertex_counts = [e["vertex_count"] for e in entries]

    tier_counts = defaultdict(int)
    category_counts = defaultdict(int)
    for e in entries:
        tier_counts[e["tier"]] += 1
        category_counts[e["category"]] += 1

    return {
        "dataset": "Toys4k-PBR",
        "total_discovered": total_discovered,
        "total_pbr_passed": total_pbr_passed,
        "total_valid": len(entries),
        "complexity_tiers": {
            "tier_1_max_faces": tier_thresholds[0] if tier_thresholds else None,
            "tier_2_max_faces": tier_thresholds[1] if tier_thresholds else None,
        },
        "tier_distribution": dict(sorted(tier_counts.items())),
        "category_distribution": dict(sorted(category_counts.items())),
        "face_count_stats": {
            "min": int(np.min(face_counts)) if face_counts else 0,
            "max": int(np.max(face_counts)) if face_counts else 0,
            "mean": float(np.mean(face_counts)) if face_counts else 0,
            "median": float(np.median(face_counts)) if face_counts else 0,
        },
        "vertex_count_stats": {
            "min": int(np.min(vertex_counts)) if vertex_counts else 0,
            "max": int(np.max(vertex_counts)) if vertex_counts else 0,
            "mean": float(np.mean(vertex_counts)) if vertex_counts else 0,
            "median": float(np.median(vertex_counts)) if vertex_counts else 0,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Toys4k-PBR test set for component evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process pre-downloaded Toys4k data
  python scripts/prepare_toys4k.py --data_dir /path/to/toys4k

  # Skip PBR filter (include all meshes, useful for testing)
  python scripts/prepare_toys4k.py --data_dir /path/to/toys4k --skip_pbr_filter

  # Custom output directory
  python scripts/prepare_toys4k.py --data_dir /path/to/toys4k --output_dir /path/to/output

Download:
  The Toys4k dataset (~3,229 toy assets) can be obtained from:
    - Project page: https://github.com/rehg-lab/lowshot-shapebias
    - The dataset should be extracted so that mesh files (.glb/.gltf/.obj)
      are organized under category subdirectories within data_dir.
        """,
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to pre-downloaded Toys4k data directory containing mesh files"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="experiments/component_eval/test_set",
        help="Output directory for manifest and metadata (default: experiments/component_eval/test_set)"
    )
    parser.add_argument(
        "--skip_pbr_filter", action="store_true",
        help="Skip PBR material filtering (include all valid meshes)"
    )
    args = parser.parse_args()

    # Validate data directory
    if not os.path.isdir(args.data_dir):
        print(f"Error: data_dir does not exist: {args.data_dir}")
        print("Please download the Toys4k dataset first. See --help for details.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Discover mesh files
    print(f"Discovering mesh files in {args.data_dir} ...")
    meshes = discover_meshes(args.data_dir)
    total_discovered = len(meshes)
    print(f"  Found {total_discovered} mesh files")

    if total_discovered == 0:
        print("Error: No mesh files found. Check that data_dir contains .glb/.gltf/.obj files.")
        sys.exit(1)

    # Step 2: PBR filtering
    pbr_passed = []
    pbr_skipped = 0
    pbr_failed = 0

    if args.skip_pbr_filter:
        print("PBR filtering: SKIPPED (--skip_pbr_filter)")
        pbr_passed = meshes
    else:
        print("Checking PBR materials ...")
        for mesh_path, category in tqdm(meshes, desc="PBR filter"):
            has_pbr, info = check_pbr_materials(mesh_path)
            if has_pbr:
                pbr_passed.append((mesh_path, category))
            elif "error" in info:
                pbr_skipped += 1
            else:
                pbr_failed += 1

        print(f"  PBR passed: {len(pbr_passed)}")
        print(f"  PBR failed: {pbr_failed}")
        if pbr_skipped > 0:
            print(f"  Load errors: {pbr_skipped}")

    total_pbr_passed = len(pbr_passed)

    # Step 3: Compute complexity metrics
    print("Computing mesh complexity ...")
    entries = []
    load_failures = 0

    for mesh_path, category in tqdm(pbr_passed, desc="Complexity"):
        complexity = compute_mesh_complexity(mesh_path)
        if complexity is None:
            load_failures += 1
            continue

        # Generate a stable UID from the relative path
        # Use parent directory name (e.g., "airplane_000") since mesh files are often
        # named generically (e.g., "mesh.obj")
        rel_path = os.path.relpath(mesh_path, args.data_dir)
        rel_parts = Path(rel_path).parts
        if len(rel_parts) >= 2:
            uid = rel_parts[-2]  # e.g., "airplane_000"
        else:
            uid = Path(rel_path).stem
        # Ensure uniqueness by prepending category if not already included
        if not uid.startswith(category):
            uid = f"{category}__{uid}"

        entries.append({
            "uid": uid,
            "mesh_path": os.path.abspath(mesh_path),
            "category": category,
            "face_count": complexity["face_count"],
            "vertex_count": complexity["vertex_count"],
        })

    if load_failures > 0:
        print(f"  Warning: {load_failures} meshes failed to load for complexity computation")

    if not entries:
        print("Error: No valid meshes remaining after filtering.")
        sys.exit(1)

    # Step 4: Assign complexity tiers
    print("Assigning complexity tiers ...")
    tier_thresholds = assign_complexity_tiers(entries)
    tier_counts = defaultdict(int)
    for e in entries:
        tier_counts[e["tier"]] += 1
    print(f"  Tier 1 (low):    {tier_counts[1]} assets  (faces <= {tier_thresholds[0]})")
    print(f"  Tier 2 (medium): {tier_counts[2]} assets  (faces <= {tier_thresholds[1]})")
    print(f"  Tier 3 (high):   {tier_counts[3]} assets  (faces > {tier_thresholds[1]})")

    # Step 5: Sort entries by uid for determinism
    entries.sort(key=lambda e: e["uid"])

    # Step 6: Write manifest.json
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(entries, f, indent=2)
    print(f"\nManifest written: {manifest_path}  ({len(entries)} entries)")

    # Step 7: Write metadata.json
    metadata = build_metadata(entries, tier_thresholds, total_discovered, total_pbr_passed)
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata written: {metadata_path}")

    # Summary
    print(f"\n{'='*50}")
    print(f"Toys4k-PBR test set preparation complete")
    print(f"  Discovered:   {total_discovered}")
    print(f"  PBR filtered: {total_pbr_passed}")
    print(f"  Valid output: {len(entries)}")
    print(f"  Output dir:   {os.path.abspath(args.output_dir)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
