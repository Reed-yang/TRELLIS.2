"""Test different PBR filter strictness levels. Run inside Blender."""
import bpy, os, sys, json, glob

blend_dir = "datasets/Toys4k/toys4k_blend_files"
blend_files = sorted(glob.glob(os.path.join(blend_dir, '*', '*', '*.blend')))

strict_count = 0
loose_count = 0
texture_linked_count = 0

for i, bf in enumerate(blend_files):
    try:
        bpy.ops.wm.open_mainfile(filepath=bf)
    except:
        continue

    has_strict = False
    has_loose = False
    has_tex_linked = False

    for mat in bpy.data.materials:
        if mat is None or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type != 'BSDF_PRINCIPLED':
                continue
            bc = node.inputs.get('Base Color')
            met = node.inputs.get('Metallic')
            rough = node.inputs.get('Roughness')

            bc_linked = bc is not None and bc.is_linked
            met_linked = met is not None and met.is_linked
            rough_linked = rough is not None and rough.is_linked
            met_nondefault = met is not None and (met.is_linked or abs(met.default_value) > 1e-6)
            rough_nondefault = rough is not None and (rough.is_linked or abs(rough.default_value - 0.5) > 1e-6)

            if bc_linked and met_linked and rough_linked:
                has_strict = True
            if bc_linked and met_nondefault and rough_nondefault:
                has_loose = True
            if bc_linked and (met_linked or rough_linked):
                has_tex_linked = True
            break

    if has_strict: strict_count += 1
    if has_loose: loose_count += 1
    if has_tex_linked: texture_linked_count += 1

    if (i + 1) % 500 == 0:
        print(f"  [{i+1}/{len(blend_files)}] strict={strict_count} loose={loose_count} partial={texture_linked_count}")

print(f"FILTER_RESULT: strict(all_linked)={strict_count}, loose(bc_linked+nondefault)={loose_count}, partial(bc+any_linked)={texture_linked_count}, total={len(blend_files)}")
