import os
import pickle
import trimesh
import multiprocessing
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from utils import load_pickle, voxels_to_mesh, fetch_np_array
from voxelize import visualize_boundary_registers

# Global variable for the worker processes to hold the adjacency graph in memory
_worker_face_adj = None
_worker_edge_adj = None

def _init_worker(face_adj):
    """Initializer for multiprocessing pool to set up the global adjacency graph."""
    global _worker_face_adj
    _worker_face_adj = face_adj

def _process_single_cube(cube_dict):
    """Worker function to process a single cube."""
    global _worker_face_adj
    new_dict = cube_dict.copy()
    face_indices = new_dict.get("face_indices", [])
    
    if not face_indices:
        new_dict["num_components"] = 0
        return new_dict
        
    face_set = set(face_indices)
    visited = set()
    num_components = 0
    
    # Use Depth-First Search (DFS) on the precomputed adjacency graph
    # This completely avoids creating trimesh objects and is orders of magnitude faster
    for face in face_indices:
        if face not in visited:
            num_components += 1
            stack = [face]
            visited.add(face)
            
            while stack:
                curr = stack.pop()
                for neighbor in _worker_face_adj[curr]:
                    if neighbor in face_set and neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
                        
    new_dict["num_components"] = num_components
    return new_dict

def _init_edge_worker(edge_adj):
    """Initializer for multiprocessing pool to set up the global edge adjacency graph."""
    global _worker_edge_adj
    _worker_edge_adj = edge_adj

def _process_single_edge_cube(cube_dict):
    """Worker function to process a single boundary edge cube."""
    global _worker_edge_adj
    new_dict = cube_dict.copy()
    edge_indices = new_dict.get("edge_indices", [])
    
    if not edge_indices:
        new_dict["num_boundary"] = 0
        return new_dict
        
    edge_set = set(edge_indices)
    visited = set()
    num_components = 0
    
    # Use Depth-First Search (DFS) on the precomputed edge adjacency graph
    for edge in edge_indices:
        if edge not in visited:
            num_components += 1
            stack = [edge]
            visited.add(edge)
            
            while stack:
                curr = stack.pop()
                for neighbor in _worker_edge_adj[curr]:
                    if neighbor in edge_set and neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
                        
    new_dict["num_boundary"] = num_components
    return new_dict


def volume_feature_face(cube_data_list, mesh, save_dir, filename="face_registers.pkl", batch_size=10000, num_workers=None):
    """
    Processes a list of dictionaries containing cube and face indices, 
    calculates the number of connected components for the mesh faces in each cube, 
    and saves the updated data to a pickle file.
    
    Args:
        cube_data_list (list of dict): List of dicts, e.g., 
                                       [{"cube_indices": (i, j, k), "face_indices": [f1, f2, ...]}, ...]
        mesh (trimesh.Trimesh): The original normalized 3D mesh.
        save_dir (str): Directory where the output pickle file will be saved.
        filename (str): Name of the output pickle file.
        batch_size (int): Number of items to process simultaneously to cap memory usage.
        num_workers (int, optional): Number of CPU cores to use. Defaults to CPU count - 1.
        
    Returns:
        list of dict: The updated list of dictionaries with the 'num_components' key added.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        
    updated_cube_data = []
    
    print("Precomputing face adjacency graph... (This is done once and speeds up processing vastly)")
    # Build an adjacency list: index is face_id, value is a list of adjacent face_ids
    face_adj = [[] for _ in range(len(mesh.faces))]
    for f1, f2 in mesh.face_adjacency:
        face_adj[f1].append(f2)
        face_adj[f2].append(f1)
        
    print(f"Starting multiprocessing pool with {num_workers} workers...")
    
    # Use a Pool to parallelize. We process in explicit batches to keep the memory footprint bounded.
    with multiprocessing.Pool(processes=num_workers, initializer=_init_worker, initargs=(face_adj,)) as pool:
        # Loop through data in batches, updating progress with tqdm
        for i in tqdm(range(0, len(cube_data_list), batch_size), desc="Processing Cube Batches"):
            batch = cube_data_list[i : i + batch_size]
            
            # Map the processing function over the current batch (blocks until batch completes)
            batch_results = pool.map(_process_single_cube, batch)
            updated_cube_data.extend(batch_results)
            
    # Save the updated list to the specified directory using pickle
    save_path = os.path.join(save_dir, filename)
    with open(save_path, "wb") as f:
        pickle.dump(updated_cube_data, f)
        
    print(f"Processed {len(updated_cube_data)} cubes and saved to {save_path}")
    
    return updated_cube_data

def volume_feature_boundary(cube_data_list, boundary_edges, save_dir, filename="boundary_registers.pkl", batch_size=10000, num_workers=None):
    """
    Processes a list of dictionaries containing cube and edge indices, 
    calculates the number of connected components for the edges in each cube, 
    and saves the updated data to a pickle file.
    
    Args:
        cube_data_list (list of dict): List of dicts, e.g., 
                                       [{"cube_indices": (i, j, k), "edge_indices": [e1, e2, ...]}, ...]
        boundary_edges (array-like): Array of shape (N, 2, 3) representing edge start and end coordinates.
        save_dir (str): Directory where the output pickle file will be saved.
        filename (str): Name of the output pickle file.
        batch_size (int): Number of items to process simultaneously.
        num_workers (int, optional): Number of CPU cores to use.
        
    Returns:
        list of dict: The updated list of dictionaries with the 'num_components' key added.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        
    updated_cube_data = []
    
    print("Precomputing edge adjacency graph... (This is done once and speeds up processing vastly)")
    # Convert to numpy array in case it's a TrackedArray or other array-like object
    boundary_array = np.asarray(boundary_edges)
    N = boundary_array.shape[0]
    
    # To find shared vertices, we round coordinates to 6 decimal places and find unique vertices
    flat_points = np.round(boundary_array.reshape(-1, 3), decimals=6)
    _, inverse_indices = np.unique(flat_points, axis=0, return_inverse=True)
    edge_vertex_ids = inverse_indices.reshape(-1, 2)
    
    # Build vertex to edges mapping
    vertex_to_edges = defaultdict(list)
    for edge_idx, (v1, v2) in enumerate(edge_vertex_ids):
        vertex_to_edges[v1].append(edge_idx)
        vertex_to_edges[v2].append(edge_idx)
        
    # Build edge adjacency list
    edge_adj = [[] for _ in range(N)]
    for edge_idx, (v1, v2) in enumerate(edge_vertex_ids):
        adj = set(vertex_to_edges[v1]) | set(vertex_to_edges[v2])
        adj.remove(edge_idx)
        edge_adj[edge_idx] = list(adj)
        
    print(f"Starting multiprocessing pool with {num_workers} workers...")
    
    with multiprocessing.Pool(processes=num_workers, initializer=_init_edge_worker, initargs=(edge_adj,)) as pool:
        for i in tqdm(range(0, len(cube_data_list), batch_size), desc="Processing Boundary Batches"):
            batch = cube_data_list[i : i + batch_size]
            batch_results = pool.map(_process_single_edge_cube, batch)
            updated_cube_data.extend(batch_results)
            
    save_path = os.path.join(save_dir, filename)
    with open(save_path, "wb") as f:
        pickle.dump(updated_cube_data, f)
        
    print(f"Processed {len(updated_cube_data)} boundary cubes and saved to {save_path}")
    
    return updated_cube_data

def visualize_volume_feature(volume_indices_face, volume_num_face, volume_indices_boundary, volume_num_boundary, res, save_dir):
    mask = np.where(volume_num_face == 1)[0]
    voxels_to_mesh(volume_indices_face[mask], res, color=[255, 255, 255, 255]).export(f'{save_dir}/volume_face_1.ply')
    mask = np.where(volume_num_face == 2)[0]
    voxels_to_mesh(volume_indices_face[mask], res, color=[0, 255, 255, 255]).export(f'{save_dir}/volume_face_2.ply')
    mask = np.where(volume_num_face > 2)[0]
    voxels_to_mesh(volume_indices_face[mask], res, color=[0, 0, 255, 255]).export(f'{save_dir}/volume_face_3+.ply')

    mask = np.where(volume_num_boundary == 1)[0]
    voxels_to_mesh(volume_indices_boundary[mask], res, color=[255, 255, 0, 255]).export(f'{save_dir}/volume_boundary_1.ply')
    mask = np.where(volume_num_boundary == 2)[0]
    voxels_to_mesh(volume_indices_boundary[mask], res, color=[0, 255, 0, 255]).export(f'{save_dir}/volume_boundary_2.ply')
    mask = np.where(volume_num_boundary > 2)[0]
    voxels_to_mesh(volume_indices_boundary[mask], res, color=[0, 0, 255, 255]).export(f'{save_dir}/volume_boundary_3+.ply')

def get_neighborhood_components(cube_indices, components_count, res=None):
    """
    For a given set of cube indices and their connected component counts, 
    returns a 3x3x3 neighborhood grid for each cube containing the neighbor's component counts.

    Args:
        cube_indices (np.ndarray): An (N, 3) array of integer coordinates for the cubes.
        components_count (np.ndarray): An (N,) array of integers representing the number 
                                       of connected components in the respective cube.
        res (int, optional): The resolution of the grid (i.e., sliced into res**3 cubes). 
                             If None, it infers the bounding box from the max index.

    Returns:
        np.ndarray: An (N, 3, 3, 3) array where each element contains the component 
                    counts of the surrounding 3x3x3 neighborhood.
    """
    N = cube_indices.shape[0]
    
    # Handle the empty edge case
    if N == 0:
        return np.zeros((0, 3, 3, 3), dtype=components_count.dtype)

    # Determine the maximum bounds of the grid
    if res is None:
        grid_shape = np.max(cube_indices, axis=0) + 1
    else:
        grid_shape = (res, res, res)

    # Create a dense grid padded by 1 on all sides to elegantly handle boundary lookups.
    # We initialize it with zeros (0 components for empty/out-of-bounds space).
    padded_grid = np.zeros(
        (grid_shape[0] + 2, grid_shape[1] + 2, grid_shape[2] + 2),
        dtype=components_count.dtype
    )

    # Populate the grid. We shift the indices by +1 to account for the padding.
    padded_grid[cube_indices[:, 0] + 1, 
                cube_indices[:, 1] + 1, 
                cube_indices[:, 2] + 1] = components_count

    # Create a 3D grid of offsets representing a 3x3x3 neighborhood: (0, 1, 2)
    # The start index for a neighborhood in the padded grid corresponds perfectly 
    # to the original unpadded cube index! 
    # (Unpadded coord `x` means padded center `x+1`, meaning neighborhood starts at `x`).
    dx, dy, dz = np.meshgrid(np.arange(3), np.arange(3), np.arange(3), indexing='ij')

    # Expand dimensions to leverage NumPy broadcasting to index everything simultaneously.
    # cube_indices: shape transforms from (N, 3) -> (N, 1, 1, 1) per axis
    # dx, dy, dz: shape transforms from (3, 3, 3) -> (1, 3, 3, 3)
    x_idx = cube_indices[:, 0, None, None, None] + dx[None, :, :, :]
    y_idx = cube_indices[:, 1, None, None, None] + dy[None, :, :, :]
    z_idx = cube_indices[:, 2, None, None, None] + dz[None, :, :, :]

    # Extract the (N, 3, 3, 3) neighborhoods cleanly
    neighborhoods = padded_grid[x_idx, y_idx, z_idx]

    return neighborhoods

def feature_volume(face_registers, boundary_registers, mesh, boundaries, output_dir):
    face_result = volume_feature_face(face_registers, mesh, output_dir)
    boundary_result = volume_feature_boundary(boundary_registers, boundaries, output_dir)
    return face_result, boundary_result

# Example usage (for testing purposes):
if __name__ == "__main__":
    load_dir = 'tmp/test_rep'
    output_dir = 'tmp/test_rep'
    res = 1024

    mesh = trimesh.load(f'{load_dir}/norm_mesh.ply')
    face_mapping = load_pickle(f'{load_dir}/voxel_face_mapping_res{res}.pkl')

    boundary = load_pickle(f'{load_dir}/open_boundary.pkl')
    boundary_mapping = load_pickle(f'{load_dir}/voxel_boundary_mapping_res{res}.pkl')

    
    result = volume_feature_face(face_mapping, mesh, output_dir)
    for item in result[:10]:
        print(f"Cube: {item['cube_indices']}, Faces: {len(item['face_indices'])}, Components: {item['num_components']}")
        
    # Testing the fetch_np_array functionality
    volume_indices_face, volume_num_face = fetch_np_array(result, ["cube_indices", "num_components"])
    print(f"\nFetched Array Shapes -> Cube Indices: {volume_indices_face.shape}, Components: {volume_num_face.shape}")


    # Testing the boundary functionality
    boundary_result = volume_feature_boundary(boundary_mapping, boundary, output_dir)
    for b_item in boundary_result[:10]:
        print(f"Boundary Cube: {b_item['cube_indices']}, Edges: {len(b_item['edge_indices'])}, Components: {b_item['num_components']}")

    volume_indices_boundary, volume_num_boundary = fetch_np_array(boundary_result, ["cube_indices", "num_components"])
    print(f"\nFetched Array Shapes -> Cube Indices: {volume_indices_boundary.shape}, Components: {volume_num_boundary.shape}")

    # mask = np.where(comps_arr == 2)[0]
    # merged_boundary = visualize_boundary_registers(
    #         boundaries=boundary, 
    #         res=res, 
    #         cube_index=boundary_mapping[mask[0]]['cube_indices'], 
    #         edge_indices=boundary_mapping[mask[0]]['edge_indices']
    #     )
    # merged_boundary.export('tmp/test_rep/sample_volume_boundary.ply')

    visualize_volume_feature(volume_indices_face, volume_num_face, volume_indices_boundary, volume_num_boundary, res, output_dir)

    volume_indices_face_neighbor_components = get_neighborhood_components(volume_indices_face, volume_num_face)
    volume_indices_boundary_neighbor_components = get_neighborhood_components(volume_indices_boundary, volume_num_boundary)

    # breakpoint()
