import numpy as np
import trimesh
import pickle
import multiprocessing as mp
import os
from tqdm import tqdm

from utils import load_pickle, fetch_np_array, save_pickle
from voxelize import visualize_face_registers
from collapse_edge import reconstruct_loops, reconstruct_loops_multiple_cubes


# Global variables for worker processes to avoid memory duplication
_worker_vertices = None
_worker_faces = None
_worker_step = None
_worker_v_offsets = None
_worker_edge_indices = None

def _init_worker(vertices, faces, res):
    """Initializer function to set up read-only global data for each worker."""
    global _worker_vertices, _worker_faces, _worker_step
    global _worker_v_offsets, _worker_edge_indices
    
    _worker_vertices = vertices
    _worker_faces = faces
    _worker_step = 1.0 / res
    
    _worker_v_offsets = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=float)

    _worker_edge_indices = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
        # (0, 2), (4, 6), (0, 5), (1, 6), (2, 7), (0, 7)
        (0, 2), (4, 6), (1, 4), (1, 6), (2, 7), (0, 7)
    ]

def _process_cube(data):
    """Worker function to process a single cube using pre-loaded global topology."""
    ix, iy, iz = data['cube_indices']
    f_idx = data['face_indices']
    
    if not f_idx:
        return [0] * 18
        
    base_pos = np.array([ix, iy, iz]) * _worker_step
    cube_verts = base_pos + (_worker_v_offsets * _worker_step)
    
    cube_triangles = _worker_vertices[_worker_faces[f_idx]]
    
    V0 = cube_triangles[:, 0, :]
    V1 = cube_triangles[:, 1, :]
    V2 = cube_triangles[:, 2, :]
    
    E1 = V1 - V0
    E2 = V2 - V0
    
    edge_weights = []
    EPSILON = 1e-8
    
    for e_start, e_end in _worker_edge_indices:
        O = cube_verts[e_start]
        D = cube_verts[e_end] - O
        
        P = np.cross(D, E2)
        det = np.sum(E1 * P, axis=1)
        
        valid = np.abs(det) > EPSILON
        
        inv_det = np.zeros_like(det)
        inv_det[valid] = 1.0 / det[valid]
        
        T = O - V0
        u = np.sum(T * P, axis=1) * inv_det
        
        valid_u = valid & (u >= 0.0) & (u <= 1.0)
        
        Q = np.cross(T, E1)
        v = np.sum(D * Q, axis=1) * inv_det
        
        valid_v = valid_u & (v >= 0.0) & (u + v <= 1.0)
        
        t = np.sum(E2 * Q, axis=1) * inv_det
        
        intersects = valid_v & (t >= -EPSILON) & (t <= 1.0 + EPSILON)
        edge_weights.append(int(np.sum(intersects)))
        
    return edge_weights

def calculate_edge_crossings(mesh, res, cube_data, output_dir, batch_size=10000, num_workers=None):
    """
    Calculates the number of mesh faces crossing 18 specific edges of sub-cubes
    in a regular voxel grid, appending the counts to each cube's dictionary.
    
    Args:
        mesh: An object with `.vertices` (N, 3) and `.faces` (M, 3) attributes 
              representing the triangulated surface mesh normalized to a (0,1) cube.
        res (int): The resolution of the grid. Sub-cube size is 1.0 / res.
        cube_data (list of dict): List containing dicts with:
            - 'cube_indices': tuple (ix, iy, iz)
            - 'face_indices': list of integer face indices intersecting the cube.
        batch_size (int): Number of cubes to process in memory at once.
        num_workers (int): Number of parallel processes to spawn.
            
    Returns:
        list of dict: The updated list with 'edge_weights' (list of 18 ints) 
                      added to each dictionary.
    """
    # Ensure mesh data is formatted as NumPy arrays for rapid vectorized operations
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
        
    # Initialize pool with shared read-only mesh data to save memory across workers
    pool = mp.Pool(processes=num_workers, initializer=_init_worker, initargs=(vertices, faces, res))
    
    try:
        # Process strictly in batches to cap peak memory usage at a constant level
        with tqdm(total=len(cube_data), desc="Calculating Edge Crossings") as pbar:
            for i in range(0, len(cube_data), batch_size):
                batch = cube_data[i : i + batch_size]
                
                # Map the batch across available workers
                chunk_size = max(1, len(batch) // (num_workers * 4))
                batch_results = pool.map(_process_cube, batch, chunksize=chunk_size)
                
                # Update dictionaries in place
                for data, weights in zip(batch, batch_results):
                    data['edge_weights'] = weights
                    
                pbar.update(len(batch))
    finally:
        pool.close()
        pool.join()
        
    save_pickle(f'{output_dir}/face_registers.pkl', cube_data)
    return cube_data


def visualize_edge_weights(weights, edge_radius=0.015, sphere_radius=0.04):
    """
    Creates a 3D visualization mesh of a sub-cube with spheres evenly placed 
    on edges representing the integer crossing weights.
    
    Args:
        weights (list of int): 18 integers representing the crossings on each edge.
        edge_radius (float): Thickness of the cube wireframe edges.
        sphere_radius (float): Size of the crossing indicator spheres.
        
    Returns:
        trimesh.Trimesh: A single concatenated mesh containing the wireframe and spheres.
    """
    v_offsets = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=float)
    
    edge_indices = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
        # (0, 2), (4, 6), (0, 5), (1, 6), (2, 7), (0, 7)
        (0, 2), (4, 6), (1, 4), (1, 6), (2, 7), (0, 7)
    ]
    
    meshes = []
    
    # Build wireframe using thin cylinders
    for start_idx, end_idx in edge_indices:
        p1 = v_offsets[start_idx]
        p2 = v_offsets[end_idx]
        edge_mesh = trimesh.creation.cylinder(radius=edge_radius, segment=(p1, p2))
        
        # Color original edges light grey and diagonals light blue for clarity
        edge_mesh.visual.face_colors = [200, 200, 200, 255]
        meshes.append(edge_mesh)
        
    # Build spheres for weights
    for i, weight in enumerate(weights):
        if weight > 0:
            start_idx, end_idx = edge_indices[i]
            p1 = v_offsets[start_idx]
            p2 = v_offsets[end_idx]
            
            # Distribute spheres evenly along the segment
            # e.g., weight 1 -> t=0.5; weight 2 -> t=0.33, 0.66
            t_values = np.linspace(1/(weight+1), weight/(weight+1), weight)
            
            for t in t_values:
                pos = p1 + t * (p2 - p1)
                sphere = trimesh.creation.icosphere(radius=sphere_radius)
                sphere.apply_translation(pos)
                
                # Color code spheres: Red for regular edges, Blue for diagonals
                if i < 12:
                    sphere.visual.face_colors = [220, 50, 50, 255]
                else:
                    sphere.visual.face_colors = [50, 100, 220, 255]
                    
                meshes.append(sphere)
                
    # Combine everything into a single mesh object
    return trimesh.util.concatenate(meshes)


def visualize_cube_edge_weights(cube_data, res, mesh=None):
    edge_mesh = visualize_edge_weights(cube_data['edge_weights'])
    # scale the edge mesh to the size of the cube
    edge_mesh.apply_scale(1.0 / res)
    edge_mesh.apply_translation(np.array(cube_data['cube_indices']) * (1.0 / res))
    return edge_mesh

def feature_edge(mesh, res, face_registers, output_dir, debug=False):
    results = calculate_edge_crossings(mesh, res, face_registers, output_dir)
    if debug:
        os.makedirs(f'{output_dir}/debug', exist_ok=True)
        for result in results:
            visualize_edge_weights(result['edge_weights']).export(f'{output_dir}/debug/tmp_edge_weights_{result["cube_indices"][0]}_{result["cube_indices"][1]}_{result["cube_indices"][2]}.ply')
        for result in results:
            visualize_face_registers(mesh, res, result['cube_indices'], result['face_indices']).export(f'{output_dir}/debug/tmp_cube_{result["cube_indices"][0]}_{result["cube_indices"][1]}_{result["cube_indices"][2]}.ply')
    # breakpoint()
    results = reconstruct_loops_multiple_cubes(results)
    save_pickle(f'{output_dir}/face_registers.pkl', results)
    return results


if __name__ == "__main__":

    load_dir = 'tmp/test_rep'
    output_dir = 'tmp/test_rep'
    res = 1024

    mesh = trimesh.load(f'{load_dir}/norm_mesh.ply')
    face_mapping = load_pickle(f'{load_dir}/volume_feature_face.pkl')

    results = calculate_edge_crossings(mesh, res, face_mapping)
    edge_features = fetch_np_array(results, 'edge_weights')
    # breakpoint()

    # print(edge_features[1])
    visualize_edge_weights(edge_features[1]).export('tmp/tmp_edge_weight.ply')
    # visualize_face_registers(mesh, res, face_mapping[1]['cube_indices'], face_mapping[1]['face_indices']).export('tmp/tmp_cube.ply')

    # extracted_loops = reconstruct_loops(edge_features[0])
    # print(f"Found {len(extracted_loops)} loop(s).")
    # for step in extracted_loops[1]:
    #     print(f"Face {step['face']:2} | Entered via edge {step['edge_in']:2} -> Exited via edge {step['edge_out']:2}")
    # breakpoint()

    results = reconstruct_loops_multiple_cubes(results)
    save_pickle(f'{output_dir}/edge_features.pkl', results)

    breakpoint()
