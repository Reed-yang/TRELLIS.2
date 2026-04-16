"""
Filter Toys4k assets by PBR material presence using Blender files.

Runs inside Blender: blender -b -P filter_pbr_blender.py -- --blend_dir ... --output ...

An asset passes PBR filter if at least one material has a Principled BSDF node with:
  - Base Color input linked to a texture node (Image Texture or similar)
  - Metallic input linked OR non-default value (!=0.0)
  - Roughness input linked OR non-default value (!=0.5)

This matches the paper's Toys4k-PBR subset (~473 assets).
"""

import bpy
import os
import sys
import json
import glob


def check_pbr_current_file():
    """Check PBR materials in the currently loaded Blender file."""
    pbr_materials = []
    non_pbr_materials = []

    for mat in bpy.data.materials:
        if mat is None or not mat.use_nodes:
            continue
        tree = mat.node_tree

        for node in tree.nodes:
            if node.type != 'BSDF_PRINCIPLED':
                continue

            # Check Base Color: must have a linked texture
            bc = node.inputs.get('Base Color')
            has_base_color_tex = bc is not None and bc.is_linked

            # Check Metallic: linked or non-default
            met = node.inputs.get('Metallic')
            has_metallic = False
            if met is not None:
                has_metallic = met.is_linked or abs(met.default_value - 0.0) > 1e-6

            # Check Roughness: linked or non-default
            rough = node.inputs.get('Roughness')
            has_roughness = False
            if rough is not None:
                has_roughness = rough.is_linked or abs(rough.default_value - 0.5) > 1e-6

            if has_base_color_tex and has_metallic and has_roughness:
                pbr_materials.append(mat.name)
            else:
                non_pbr_materials.append(mat.name)
            break  # only check first Principled BSDF per material

    return len(pbr_materials) > 0, pbr_materials, non_pbr_materials


def main():
    # Parse args after '--'
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        print("Usage: blender -b -P filter_pbr_blender.py -- --blend_dir DIR --output FILE [--rank R --world_size W]")
        sys.exit(1)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--blend_dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    args = parser.parse_args(argv)

    # Find all blend files
    blend_files = sorted(glob.glob(os.path.join(args.blend_dir, '*', '*', '*.blend')))
    print(f"Found {len(blend_files)} blend files")

    # Shard
    start = len(blend_files) * args.rank // args.world_size
    end = len(blend_files) * (args.rank + 1) // args.world_size
    shard = blend_files[start:end]
    print(f"Rank {args.rank}: processing {len(shard)} files ({start}-{end})")

    results = []
    for i, blend_path in enumerate(shard):
        # Extract UID from path: .../category/uid/uid.blend
        parts = blend_path.split(os.sep)
        uid = parts[-2]  # e.g., "bus_003"
        category = parts[-3]  # e.g., "bus"

        try:
            bpy.ops.wm.open_mainfile(filepath=blend_path)
            is_pbr, pbr_mats, non_pbr_mats = check_pbr_current_file()
            results.append({
                'uid': uid,
                'category': category,
                'blend_path': blend_path,
                'is_pbr': is_pbr,
                'pbr_material_count': len(pbr_mats),
                'total_materials': len(pbr_mats) + len(non_pbr_mats),
            })
        except Exception as e:
            results.append({
                'uid': uid,
                'category': category,
                'blend_path': blend_path,
                'is_pbr': False,
                'error': str(e),
            })

        if (i + 1) % 100 == 0:
            print(f"  [{args.rank}] {i+1}/{len(shard)} checked, "
                  f"{sum(1 for r in results if r['is_pbr'])} PBR so far")

    # Write results
    output_path = args.output if args.world_size == 1 else f"{args.output}.rank{args.rank}"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    pbr_count = sum(1 for r in results if r['is_pbr'])
    print(f"Rank {args.rank} done: {pbr_count}/{len(results)} passed PBR filter -> {output_path}")


main()
