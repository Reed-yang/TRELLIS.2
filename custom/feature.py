from utils import load_pickle, save_pickle, fetch_np_array
from voxelize import voxelize

from feature_volume import feature_volume
from feature_edge import feature_edge, visualize_cube_edge_weights
from feature_point import feature_point, visualize_single_cube
from feature_face import feature_face, visualize_cube_face_weights

from collapse import reconstruct_mesh
from collapse_point import collapse_point
from collapse_face import collapse_face_inner, collapse_face_boundary, visualize_loops


import trimesh
import numpy as np
import os
import shutil


if __name__ == "__main__":
    # mesh_path = 'tmp/test_mesh/turbine__turbofan_engine__jet_engine.ply'
    # mesh_path = 'tmp/test_mesh/shivaji_maharaj_cloth.ply'
    output_directory = "tmp/test_feature"
    resolution = 8

    # mesh = trimesh.load(mesh_path)

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
    
    # dummy triple layer plane mesh
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 0.01], [1, 0, 0.01], [0, 1, 0.01], [1, 1, 0.01], [0, 0, 0.02], [1, 0, 0.02], [0, 1, 0.02], [1, 1, 0.02]])
    faces = np.array([[0, 1, 2], [1, 2, 3], [4, 5, 6], [5, 6, 7], [8, 9, 10], [9, 10, 11]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # dummy triple layer plane leaning mesh
    # vertices = np.array([[0, 0, 0], [1, 0, 0.1], [0, 1, 0], [1, 1, 0.1], [0, 0, 0.01], [1, 0, 0.11], [0, 1, 0.01], [1, 1, 0.11], [0, 0, 0.02], [1, 0, 0.12], [0, 1, 0.02], [1, 1, 0.12]])
    # faces = np.array([[0, 1, 2], [1, 2, 3], [4, 5, 6], [5, 6, 7], [8, 9, 10], [9, 10, 11]])
    # mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # # dummy triple layer sphere mesh
    # mesh1 = trimesh.creation.icosphere(subdivisions=3, radius=1)
    # mesh2 = trimesh.creation.icosphere(subdivisions=3, radius=1.01)
    # mesh3 = trimesh.creation.icosphere(subdivisions=3, radius=1.02)
    # mesh = trimesh.Trimesh(vertices=np.concatenate([mesh1.vertices, mesh2.vertices, mesh3.vertices]), faces=np.concatenate([mesh1.faces, mesh2.faces + len(mesh1.vertices), mesh3.faces + len(mesh1.vertices) + len(mesh2.vertices)]))

    # clean output directory
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory, exist_ok=True)


    norm_mesh, boundaries, face_registers, boundary_registers, nm_registers = voxelize(mesh, output_directory, resolution)
    print('voxelize', face_registers[0])

    face_registers, boundary_registers = feature_volume(face_registers, boundary_registers, norm_mesh, boundaries, output_directory)
    print('feature_volume', face_registers[0])

    face_registers = feature_edge(norm_mesh, resolution, face_registers, output_directory, debug=False)
    print('feature_edge', face_registers[0])

    face_registers = feature_face(norm_mesh, resolution, face_registers, boundaries, boundary_registers, output_directory, debug=True)
    print('feature_face', face_registers[0])

    face_registers = feature_point(norm_mesh, resolution, face_registers, output_directory, debug=False)
    print('feature_point', face_registers[0])

    # face_registers = load_pickle(f'{output_directory}/face_registers.pkl')
    # norm_mesh = trimesh.load(f'{output_directory}/norm_mesh.ply')




    inner_mask = fetch_np_array(face_registers, 'num_boundary') == 0
    boundary_mask = fetch_np_array(face_registers, 'num_boundary') > 0
    inner_registers = [face_registers[i] for i in range(len(face_registers)) if inner_mask[i]]
    boundary_registers = [face_registers[i] for i in range(len(face_registers)) if boundary_mask[i]]


    solved_uturn, ambiguous_uturn, unsolvable_uturn = collapse_face_inner(inner_registers)
    print('collapse_face', len(solved_uturn), len(ambiguous_uturn), len(unsolvable_uturn))
    if len(solved_uturn) > 0:
        print('solved_uturn', solved_uturn[0])

    os.makedirs(f'{output_directory}/debug', exist_ok=True)
    # for cube_data in unsolvable_uturn:
    #     visualize_cube_edge_weights(cube_data, resolution, norm_mesh).export(f'{output_directory}/debug/tmp_edge_weights_{cube_data["cube_indices"]}.ply')
    #     visualize_cube_face_weights(cube_data, resolution, norm_mesh).export(f'{output_directory}/debug/tmp_face_weights_{cube_data["cube_indices"]}.ply')

    solved_boundary, ambiguous_boundary, unsolvable_boundary = collapse_face_boundary(boundary_registers)
    print('collapse_face_boundary', len(solved_boundary), len(ambiguous_boundary), len(unsolvable_boundary))
    if len(solved_boundary) > 0:
        print('solved_boundary', solved_boundary[0])
    os.makedirs(f'{output_directory}/debug', exist_ok=True)
    # for cube_data in unsolvable_boundary:
    #     visualize_cube_edge_weights(cube_data, resolution, norm_mesh).export(f'{output_directory}/debug/tmp_edge_weights_{cube_data["cube_indices"]}.ply')
    #     visualize_cube_face_weights(cube_data, resolution, norm_mesh).export(f'{output_directory}/debug/tmp_face_weights_{cube_data["cube_indices"]}.ply')

    face_registers = [*solved_uturn, *solved_boundary]
    # for cube_data in face_registers:
    #     visualize_loops(cube_data, resolution).export(f'{output_directory}/debug/tmp_loops_{cube_data["cube_indices"]}.ply')

    # breakpoint()

    face_registers = collapse_point(face_registers, resolution)
    print('collapse_point', face_registers[0])

    breakpoint()

    mesh_data = reconstruct_mesh(resolution, face_registers, output_filepath=f'{output_directory}/collapse_mesh.ply')


    breakpoint()

