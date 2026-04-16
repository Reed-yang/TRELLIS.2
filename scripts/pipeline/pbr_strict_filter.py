"""Generate strict PBR UID list: all 3 PBR inputs must be texture-linked. Run inside Blender."""
import bpy, os, sys, json, glob

blend_dir = "datasets/Toys4k/toys4k_blend_files"
blend_files = sorted(glob.glob(os.path.join(blend_dir, '*', '*', '*.blend')))

pbr_uids = []

for i, bf in enumerate(blend_files):
    parts = bf.split(os.sep)
    uid = parts[-2]
    category = parts[-3]

    try:
        bpy.ops.wm.open_mainfile(filepath=bf)
    except:
        continue

    is_strict_pbr = False
    for mat in bpy.data.materials:
        if mat is None or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type != 'BSDF_PRINCIPLED':
                continue
            bc = node.inputs.get('Base Color')
            met = node.inputs.get('Metallic')
            rough = node.inputs.get('Roughness')
            if (bc and bc.is_linked and met and met.is_linked and rough and rough.is_linked):
                is_strict_pbr = True
                break
        if is_strict_pbr:
            break

    if is_strict_pbr:
        pbr_uids.append(uid)

    if (i + 1) % 500 == 0:
        print(f"  [{i+1}/{len(blend_files)}] strict PBR: {len(pbr_uids)}")

output_path = "experiments/component_eval/test_set/pbr_strict_uids.json"
with open(output_path, 'w') as f:
    json.dump(sorted(pbr_uids), f, indent=2)
print(f"DONE: {len(pbr_uids)} strict PBR UIDs -> {output_path}")
