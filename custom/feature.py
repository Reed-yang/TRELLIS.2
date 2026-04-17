from utils import load_pickle, save_pickle, fetch_np_array, voxels_to_mesh
from voxelize import voxelize

from feature_volume import feature_volume
from feature_edge import feature_edge, visualize_cube_edge_weights
from feature_point import feature_point, visualize_single_cube
from feature_face import feature_face, visualize_cube_face_weights

# from collapse import reconstruct_mesh, mark_exception
# from collapse_point import collapse_point_inner, collapse_point_boundary
# from collapse_face import collapse_face_inner, collapse_face_boundary, visualize_loops


import trimesh
import numpy as np
import os
import shutil

def full_to_unique_weights(full_weights, geometry_type='edge'):
    """
    Reduces full duplicate weights to a minimal set of unique weights by extracting 
    the "owned" geometry components of each cell (e.g., bottom, front, and left boundaries).
    
    Parameters:
    - full_weights: np.ndarray of shape (n, 18) for edges or (n, 12) for faces/triangles.
                    (n can be a sparse subset of cubes)
    - geometry_type: str, either 'edge' or 'face'
    
    Returns:
    - np.ndarray of shape (n, 6) containing only the unique weights.
    """
    if geometry_type == 'edge':
        if full_weights.shape[-1] != 18:
            raise ValueError(f"Expected 18 edges, got {full_weights.shape[-1]}")
        # A cell uniquely 'owns' its origin-aligned edges and the diagonals of its owned faces.
        # 0: X-edge (Bottom Front)
        # 3: Y-edge (Bottom Left)
        # 8: Z-edge (Front-Left Vertical)
        # 12: XY-diagonal (Bottom face)
        # 14: XZ-diagonal (Front face)
        # 17: YZ-diagonal (Left face)
        unique_indices = [0, 3, 8, 12, 14, 17]
        return full_weights[..., unique_indices]
        
    elif geometry_type == 'face':
        if full_weights.shape[-1] != 12:
            raise ValueError(f"Expected 12 faces, got {full_weights.shape[-1]}")
        # A cell uniquely 'owns' its Bottom, Front, and Left faces.
        # 0, 1: Bottom face halves
        # 4, 5: Front face halves
        # 10, 11: Left face halves
        unique_indices = [0, 1, 4, 5, 10, 11]
        return full_weights[..., unique_indices]
        
    else:
        raise ValueError("geometry_type must be 'edge' or 'face'")


def _shift_grid(grid, dx, dy, dz, mode='clip'):
    """
    Helper function to access neighbor elements in the 3D grid.
    A shift of dx=1 means the cell at (x,y,z) inherits the value from (x+1,y,z).
    """
    shifted = grid
    if mode == 'wrap':
        # Periodic boundaries (toroidal topology)
        if dx == 1: shifted = np.roll(shifted, shift=-1, axis=0)
        if dy == 1: shifted = np.roll(shifted, shift=-1, axis=1)
        if dz == 1: shifted = np.roll(shifted, shift=-1, axis=2)
    elif mode == 'clip':
        # Bounded box boundaries (duplicates the outermost internal faces for the outer boundary)
        shifted = np.copy(grid)
        if dx == 1: shifted[:-1, :, :] = shifted[1:, :, :]
        if dy == 1: shifted[:, :-1, :] = shifted[:, 1:, :]
        if dz == 1: shifted[:, :, :-1] = shifted[:, :, 1:]
    else:
        raise ValueError(f"Unknown mode: {mode}")
        
    return shifted


def unique_to_full_weights(unique_weights, cube_indices, res, geometry_type='edge', mode='clip'):
    """
    Reconstructs the full duplicate weights for a sparse array of active cubes.
    It scatters the sparse weights to a dense grid, fetches shared geometry from neighbors, 
    and gathers the results back to the original sparse format.
    
    Parameters:
    - unique_weights: np.ndarray of shape (n, 6) containing the minimal weights.
    - cube_indices: np.ndarray of shape (n, 3), the (x, y, z) coordinates of the active cubes.
    - res: int, the resolution/dimension of the full bounding grid (res x res x res).
    - geometry_type: str, either 'edge' or 'face'
    - mode: str, 'clip' (best for bounded 0-1 space) or 'wrap' (for periodic topology)
    
    Returns:
    - np.ndarray of shape (n, 18) or (n, 12) containing the reconstructed full weights for the n active cubes.
    """
    n = unique_weights.shape[0]
    
    # Extract x, y, z coordinates
    x, y, z = cube_indices[:, 0], cube_indices[:, 1], cube_indices[:, 2]
    
    # Scatter: Create a full dense 3D grid of zeros and place our active cube weights directly
    U = np.zeros((res, res, res, 6), dtype=unique_weights.dtype)
    U[x, y, z] = unique_weights
    
    if geometry_type == 'edge':
        F = np.zeros((res, res, res, 18), dtype=unique_weights.dtype)
        
        # Reference owned base edges
        E_X = U[..., 0]  # E0: X-edge
        E_Y = U[..., 1]  # E3: Y-edge
        E_Z = U[..., 2]  # E8: Z-edge
        D_XY = U[..., 3] # E12: Bottom diag
        D_XZ = U[..., 4] # E14: Front diag
        D_YZ = U[..., 5] # E17: Left diag

        # --- Reconstruct the 12 Standard Edges ---
        F[..., 0] = E_X                                   # 0: Bottom Front 
        F[..., 1] = _shift_grid(E_Y, 1, 0, 0, mode)       # 1: Bottom Right (Y-edge of x+1)
        F[..., 2] = _shift_grid(E_X, 0, 1, 0, mode)       # 2: Bottom Back (X-edge of y+1)
        F[..., 3] = E_Y                                   # 3: Bottom Left 
        
        F[..., 4] = _shift_grid(E_X, 0, 0, 1, mode)       # 4: Top Front (X-edge of z+1)
        F[..., 5] = _shift_grid(E_Y, 1, 0, 1, mode)       # 5: Top Right (Y-edge of x+1, z+1)
        F[..., 6] = _shift_grid(E_X, 0, 1, 1, mode)       # 6: Top Back (X-edge of y+1, z+1)
        F[..., 7] = _shift_grid(E_Y, 0, 0, 1, mode)       # 7: Top Left (Y-edge of z+1)

        F[..., 8] = E_Z                                   # 8: Front-Left Vertical
        F[..., 9] = _shift_grid(E_Z, 1, 0, 0, mode)       # 9: Front-Right Vert (Z-edge of x+1)
        F[..., 10]= _shift_grid(E_Z, 1, 1, 0, mode)       # 10: Back-Right Vert (Z-edge of x+1, y+1)
        F[..., 11]= _shift_grid(E_Z, 0, 1, 0, mode)       # 11: Back-Left Vert (Z-edge of y+1)

        # --- Reconstruct the 6 Diagonal Edges ---
        F[..., 12] = D_XY                                 # 12: Bottom Diagonal
        F[..., 13] = _shift_grid(D_XY, 0, 0, 1, mode)     # 13: Top Diagonal (XY-diag of z+1)
        F[..., 14] = D_XZ                                 # 14: Front Diagonal
        F[..., 15] = _shift_grid(D_YZ, 1, 0, 0, mode)     # 15: Right Diagonal (YZ-diag of x+1)
        F[..., 16] = _shift_grid(D_XZ, 0, 1, 0, mode)     # 16: Back Diagonal (XZ-diag of y+1)
        F[..., 17] = D_YZ                                 # 17: Left Diagonal

        # Gather: Pull back ONLY the original n cubes using the 3D indices
        return F[x, y, z]

    elif geometry_type == 'face':
        F = np.zeros((res, res, res, 12), dtype=unique_weights.dtype)
        
        # Reference owned base triangles
        T_Bot1, T_Bot2 = U[..., 0], U[..., 1]
        T_Frt1, T_Frt2 = U[..., 2], U[..., 3]
        T_Lft1, T_Lft2 = U[..., 4], U[..., 5]

        # --- Reconstruct the 12 Triangles ---
        F[..., 0] = T_Bot1                                # T0: Bottom 1
        F[..., 1] = T_Bot2                                # T1: Bottom 2
        F[..., 2] = _shift_grid(T_Bot1, 0, 0, 1, mode)    # T2: Top 1 (Bottom 1 of z+1)
        F[..., 3] = _shift_grid(T_Bot2, 0, 0, 1, mode)    # T3: Top 2 (Bottom 2 of z+1)

        F[..., 4] = T_Frt1                                # T4: Front 1
        F[..., 5] = T_Frt2                                # T5: Front 2
        F[..., 6] = _shift_grid(T_Lft1, 1, 0, 0, mode)    # T6: Right 1 (Left 1 of x+1)
        F[..., 7] = _shift_grid(T_Lft2, 1, 0, 0, mode)    # T7: Right 2 (Left 2 of x+1)

        F[..., 8] = _shift_grid(T_Frt1, 0, 1, 0, mode)    # T8: Back 1 (Front 1 of y+1)
        F[..., 9] = _shift_grid(T_Frt2, 0, 1, 0, mode)    # T9: Back 2 (Front 2 of y+1)

        F[..., 10] = T_Lft1                               # T10: Left 1
        F[..., 11] = T_Lft2                               # T11: Left 2

        # Gather: Pull back ONLY the original n cubes using the 3D indices
        return F[x, y, z]


if __name__ == "__main__":
    mesh_path = 'tmp/test_mesh/banana_plant_with_pot.glb'
    output_directory = "tmp/test_feature"
    resolution = 256

    mesh = trimesh.load(mesh_path, force='mesh')

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
    # mesh1 = trimesh.creation.icosphere(subdivisions=3, radius=1)
    # mesh2 = trimesh.creation.icosphere(subdivisions=3, radius=1.01)
    # mesh3 = trimesh.creation.icosphere(subdivisions=3, radius=1.02)
    # mesh = trimesh.Trimesh(vertices=np.concatenate([mesh1.vertices, mesh2.vertices, mesh3.vertices]), faces=np.concatenate([mesh1.faces, mesh2.faces + len(mesh1.vertices), mesh3.faces + len(mesh1.vertices) + len(mesh2.vertices)]))

    # clean output directory
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory, exist_ok=True)


    norm_mesh, boundaries, face_registers, boundary_registers, nm_registers = voxelize(mesh, output_directory, resolution)
    # print('voxelize', face_registers[0])

    face_registers, boundary_registers = feature_volume(face_registers, boundary_registers, norm_mesh, boundaries, output_directory)
    # print('feature_volume', face_registers[0])

    face_registers = feature_edge(norm_mesh, resolution, face_registers, output_directory, debug=False)
    # print('feature_edge', face_registers[0])

    face_registers = feature_face(norm_mesh, resolution, face_registers, boundaries, boundary_registers, output_directory, debug=False)
    # print('feature_face', face_registers[0])

    face_registers = feature_point(norm_mesh, resolution, face_registers, output_directory, debug=False)
    # print('feature_point', face_registers[0])

    # face_registers = load_pickle(f'{output_directory}/face_registers.pkl')
    # norm_mesh = trimesh.load(f'{output_directory}/norm_mesh.ply')

    
    cube_indices = fetch_np_array(face_registers, 'cube_indices')
    # num_components = fetch_np_array(face_registers, 'num_components')
    edge_weights_full = fetch_np_array(face_registers, 'edge_weights')
    face_weights_full = fetch_np_array(face_registers, 'face_weights')
    num_boundary = fetch_np_array(face_registers, 'num_boundary')
    component_points = [cube_data['component_points'] for cube_data in face_registers]
    first_2_component_points = [component_points[i][:2] if len(component_points[i]) >= 2 else component_points[i]+[[0., 0., 0.]] for i in range(len(component_points))]
    first_2_component_points = np.array([cp[0] + cp[1] for cp in first_2_component_points])

    edge_weights = full_to_unique_weights(edge_weights_full, geometry_type='edge')
    face_weights = full_to_unique_weights(face_weights_full, geometry_type='face')

    save_pickle(f'{output_directory}/feature/occ.pkl', cube_indices)
    save_pickle(f'{output_directory}/feature/occ_boundary.pkl', num_boundary)
    save_pickle(f'{output_directory}/feature/weights_edge.pkl', edge_weights)
    save_pickle(f'{output_directory}/feature/weights_face.pkl', face_weights)
    save_pickle(f'{output_directory}/feature/points_2.pkl', first_2_component_points)
    save_pickle(f'{output_directory}/feature/component_points.pkl', component_points)

    edge_weights_full_reconstructed = unique_to_full_weights(edge_weights, cube_indices, resolution, geometry_type='edge')
    face_weights_full_reconstructed = unique_to_full_weights(face_weights, cube_indices, resolution, geometry_type='face')
    is_equal = np.allclose(edge_weights_full, edge_weights_full_reconstructed) and np.allclose(face_weights_full, face_weights_full_reconstructed)
    print('is_equal', is_equal)
    breakpoint()


    # inner_mask = fetch_np_array(face_registers, 'num_boundary') == 0
    # boundary_mask = fetch_np_array(face_registers, 'num_boundary') > 0
    # inner_registers = [face_registers[i] for i in range(len(face_registers)) if inner_mask[i]]
    # boundary_registers = [face_registers[i] for i in range(len(face_registers)) if boundary_mask[i]]


    # solved_uturn, ambiguous_uturn, unsolvable_uturn = collapse_face_inner(inner_registers)
    # print('collapse_face', len(solved_uturn), len(ambiguous_uturn), len(unsolvable_uturn))
    # # if len(solved_uturn) > 0:
    # #     print('solved_uturn', solved_uturn[0])

    # os.makedirs(f'{output_directory}/debug', exist_ok=True)
    # # for cube_data in unsolvable_uturn:
    # #     visualize_cube_edge_weights(cube_data, resolution, norm_mesh).export(f'{output_directory}/debug/tmp_edge_weights_{cube_data["cube_indices"]}.ply')
    # #     visualize_cube_face_weights(cube_data, resolution, norm_mesh).export(f'{output_directory}/debug/tmp_face_weights_{cube_data["cube_indices"]}.ply')

    # solved_boundary, ambiguous_boundary, unsolvable_boundary = collapse_face_boundary(boundary_registers)
    # print('collapse_face_boundary', len(solved_boundary), len(ambiguous_boundary), len(unsolvable_boundary))
    # # if len(solved_boundary) > 0:
    # #     print('solved_boundary', solved_boundary[0])
    # os.makedirs(f'{output_directory}/debug', exist_ok=True)
    # # for cube_data in unsolvable_boundary:
    # #     visualize_cube_edge_weights(cube_data, resolution, norm_mesh).export(f'{output_directory}/debug/tmp_edge_weights_{cube_data["cube_indices"]}.ply')
    # #     visualize_cube_face_weights(cube_data, resolution, norm_mesh).export(f'{output_directory}/debug/tmp_face_weights_{cube_data["cube_indices"]}.ply')

    # # face_registers = [*solved_uturn, *solved_boundary]
    # # face_registers = [*solved_uturn]

    # # for cube_data in face_registers:
    # #     visualize_loops(cube_data, resolution).export(f'{output_directory}/debug/tmp_loops_{cube_data["cube_indices"]}.ply')

    # # breakpoint()

    # face_registers = collapse_point_inner(solved_uturn, resolution, debug=False, output_directory=output_directory)
    # face_registers_with_boundary = collapse_point_boundary(solved_boundary, resolution, debug=False, output_directory=output_directory)
    # # print('collapse_point', face_registers[0])

    # exception_registers = mark_exception([*ambiguous_uturn, *unsolvable_uturn, *ambiguous_boundary, *unsolvable_boundary])
    # # exception_indices = fetch_np_array(exception_registers, 'cube_indices')
    # # voxels_to_mesh(exception_indices, resolution, color=[255, 0, 0, 128]).export(f'{output_directory}/debug/exception_voxels.ply')
    
    # # breakpoint()

    # face_registers = [*face_registers, *face_registers_with_boundary, *exception_registers]
    
    # mesh_data = reconstruct_mesh(resolution, face_registers, output_filepath=f'{output_directory}/collapse_mesh.ply')


    # breakpoint()

