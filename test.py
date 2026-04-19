import torch
from corep_fast.pipeline import corep_pipeline
import os
import trimesh
import numpy as np


os.makedirs("tmp/test_fast", exist_ok=True)




# dummy sphere mesh
# mesh = trimesh.creation.icosphere(subdivisions=3, radius=1)

# dummy cube mesh
# mesh = trimesh.creation.box(extents=[1, 1, 1])

# dummy plane mesh
# vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
# faces = np.array([[0, 1, 2], [1, 2, 3]])
# mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

# dummy double layer plane mesh
# vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 0.01], [1, 0, 0.01], [0, 1, 0.01], [1, 1, 0.01]])
# faces = np.array([[0, 1, 2], [1, 2, 3], [4, 5, 6], [5, 6, 7]])
# mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

# dummy big small double layer plane mesh
# vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 0.01], [.5, 0, 0.01], [0, .5, 0.01], [.5, .5, 0.01]])
# faces = np.array([[0, 1, 2], [1, 2, 3], [4, 5, 6], [5, 6, 7]])
# mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

# dummy triple layer plane mesh
# vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 0.01], [1, 0, 0.01], [0, 1, 0.01], [1, 1, 0.01], [0, 0, 0.02], [1, 0, 0.02], [0, 1, 0.02], [1, 1, 0.02]])
# faces = np.array([[0, 1, 2], [1, 2, 3], [4, 5, 6], [5, 6, 7], [8, 9, 10], [9, 10, 11]])
# mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

# dummy triple layer plane leaning mesh
# vertices = np.array([[0, 0, 0], [1, 0, 0.1], [0, 1, 1], [1, 1, 1.1], [0, 0, 0.01], [1, 0, 0.11], [0, 1, 1.01], [1, 1, 1.11], [0, 0, 0.02], [1, 0, 0.12], [0, 1, 1.02], [1, 1, 1.12]])
# faces = np.array([[0, 1, 2], [1, 2, 3], [4, 5, 6], [5, 6, 7], [8, 9, 10], [9, 10, 11]])
# mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

# dummy triple layer plane slightly leaning mesh
# vertices = np.array([[0, 0, 0], [1, 0, 0.1], [0, 1, 0], [1, 1, 0.1], [0, 0, 0.01], [1, 0, 0.11], [0, 1, 0.01], [1, 1, 0.11], [0, 0, 0.02], [1, 0, 0.12], [0, 1, 0.02], [1, 1, 0.12]])
# faces = np.array([[0, 1, 2], [1, 2, 3], [4, 5, 6], [5, 6, 7], [8, 9, 10], [9, 10, 11]])
# mesh = trimesh.Trimesh(vertices=vertices, faces=faces)


# # dummy triple layer sphere mesh
mesh1 = trimesh.creation.icosphere(subdivisions=3, radius=1)
mesh2 = trimesh.creation.icosphere(subdivisions=3, radius=1.01)
mesh3 = trimesh.creation.icosphere(subdivisions=3, radius=1.02)
mesh = trimesh.Trimesh(vertices=np.concatenate([mesh1.vertices, mesh2.vertices, mesh3.vertices]), faces=np.concatenate([mesh1.faces, mesh2.faces + len(mesh1.vertices), mesh3.faces + len(mesh1.vertices) + len(mesh2.vertices)]))



mesh.export("tmp/test_mesh/dummy.ply")

# 输入任意 .ply / .obj 三角网格，输出重建后的 .ply
# batch, vertices, faces = corep_pipeline(
#     mesh_path="tmp/test_mesh/dummy.ply",
#     resolution=32,
#     # mesh_path="tmp/test_mesh/banana_plant_with_pot.glb",
#     # resolution=256,               # 体素分辨率（越高越精细，越慢）
#     device=torch.device("cuda:0"),
#     output_path="tmp/test_fast/output.ply",     # 可选，写 PLY 文件
#     merge_decimals=5,             # 顶点焊接精度（小数位）
#     num_workers=None,             # MP 工人数，None=自动
# )

# 返回值:
#   batch:    CubeBatch   — 完整的编码状态（所有阶段输出）
#   vertices: (V, 3) float32 — 重建网格顶点（归一化坐标 [0,1]³）
#   faces:    (F, 3) int32   — 重建网格三角面








from corep_fast.pipeline import mesh_to_param, param_to_mesh

device = torch.device("cuda:0")

# mesh → minimal param representation (runs s1-s4 only)
param = mesh_to_param(
    # mesh_path="tmp/test_mesh/banana_plant_with_pot.glb",
    mesh_path="/mnt/novita2/siyuan/workspace/TRELLIS.2/datasets/ObjaverseXL_sketchfab/raw/hf-objaverse-v1/glbs/000-086/6eba14662bd048f9bc1ca10e63b5622f.glb",
    resolution=512,
    # mesh_path="tmp/test_mesh/dummy.ply",
    # resolution=32,
    device=device,
    num_workers=1,
)

print(f"\n========== CorepParam ==========")
print(f"resolution:     {param.resolution}")
print(f"cube_indices:   {param.cube_indices.shape} {param.cube_indices.dtype}")
print(f"edge_weights:   {param.edge_weights.shape} {param.edge_weights.dtype}")
print(f"face_weights:   {param.face_weights.shape} {param.face_weights.dtype}")
print(f"point_values:   {param.point_values.shape} {param.point_values.dtype}")
print(f"point_offsets:  {param.point_offsets.shape} {param.point_offsets.dtype}")
print(f"num_boundary:   {param.num_boundary.shape} {param.num_boundary.dtype}")
print(f"================================\n")

# param → mesh reconstruction (re-runs s6+s7+s8 from minimal data)
vertices, faces = param_to_mesh(
    param,
    device=device,
    merge_decimals=5,
    num_workers=16,
)
print(f"重建完成: V={vertices.shape[0]}, F={faces.shape[0]}")

from corep_fast.stages.s8_collapse import _write_ply_ascii
_write_ply_ascii(vertices, faces, "tmp/test_fast/param_output.ply")
print("Saved to tmp/test_fast/param_output.ply")


param_fw_unique = param.face_weights
batch_fw_full = batch.face_weights
from corep_fast.pipeline import _unique_to_full_face_weights
param_fw_full = _unique_to_full_face_weights(param_fw_unique, param.cube_indices, param.resolution)
batch_fw_unique = batch.face_weights[:, [0, 1, 4, 5, 10, 11]]
breakpoint()

# check if the face weights are the same for each row
print(param_fw_full != batch_fw_full)
mask_if_same = (param_fw_full == batch_fw_full.cpu().numpy()).all(axis=1)
mask_if_different = ~mask_if_same
print(param_fw_full[mask_if_different][:2])
print(batch_fw_full[mask_if_different][:2].cpu().numpy())

# breakpoint()