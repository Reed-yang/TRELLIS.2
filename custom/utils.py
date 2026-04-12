import numpy as np
import trimesh
import json
import ijson
import orjson
import os
import pickle
from tqdm import tqdm
from typing import List, Dict, Any, Tuple



def voxels_to_mesh(coords, res, color=[128, 128, 128, 128]):
    """
    Converts a sparse voxel tensor into a merged trimesh object and normalizes
    it to fit within a [-0.5, 0.5] cube centered at (0,0,0).
    """
    # Normalize and shift to center at (0,0,0)
    coords_normalized = (coords.astype(np.float32) / res) - 0.5
    voxel_size = 1.0 / res
    
    unit_cube_vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=np.float32) * voxel_size
    
    unit_cube_faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
        [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]
    ])

    num_voxels = coords.shape[0]
    all_vertices = unit_cube_vertices[None, :, :] + coords_normalized[:, None, :]
    all_vertices = all_vertices.reshape(-1, 3)
    
    offsets = np.arange(num_voxels) * 8
    all_faces = unit_cube_faces[None, :, :] + offsets[:, None, None]
    all_faces = all_faces.reshape(-1, 3)

    # Grey color with alpha
    colors = np.full((len(all_vertices), 4), color, dtype=np.uint8)

    return trimesh.Trimesh(vertices=all_vertices + 0.5, faces=all_faces, vertex_colors=colors, process=False)

def points_to_spheres_mesh(points, res, subdivisions=1):
    """
    Converts a point cloud into a mesh of red spheres centered at (0,0,0).
    Assumes input points are in range [0, 1].
    """
    # Shift points to center the space at (0,0,0)
    shifted_points = points - 0.5
    
    radius = 0.1 / res
    base_sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    v_base, f_base = base_sphere.vertices, base_sphere.faces
    num_v = len(v_base)
    num_points = points.shape[0]

    all_vertices = (v_base[None, :, :] + shifted_points[:, None, :]).reshape(-1, 3)
    offsets = np.arange(num_points) * num_v
    all_faces = (f_base[None, :, :] + offsets[:, None, None]).reshape(-1, 3)

    colors = np.zeros((len(all_vertices), 4), dtype=np.uint8)
    colors[:, 0], colors[:, 3] = 255, 255 

    return trimesh.Trimesh(vertices=all_vertices, faces=all_faces, vertex_colors=colors, process=False)

def flags_to_cylinders_mesh(flags, coords, res, radius_ratio=0.05):
    """
    Creates blue cylinders for edges centered at (0,0,0).
    """
    voxel_size = 1.0 / res
    radius = voxel_size * radius_ratio
    blue = [0, 0, 255, 255]
    
    row_idx, col_idx = np.where(flags)
    if len(row_idx) == 0:
        return trimesh.Trimesh()

    # Normalize starting point and shift by 0.5 to center at (0,0,0)
    starts = ((coords[row_idx].astype(np.float32) + 1.0) / res) - 0.5
    
    base_cyl = trimesh.creation.cylinder(radius=radius, height=voxel_size, sections=8)
    base_cyl.apply_translation([0, 0, -voxel_size / 2])
    
    v_base, f_base = base_cyl.vertices, base_cyl.faces
    num_v, num_f = len(v_base), len(f_base)
    num_edges = len(row_idx)

    rot_x = trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0])
    rot_y = trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0])
    rot_z = np.eye(4)
    rots = [rot_x, rot_y, rot_z]

    all_v = np.zeros((num_edges * num_v, 3), dtype=np.float32)
    all_f = np.zeros((num_edges * num_f, 3), dtype=np.int64)
    
    for d in range(3):
        mask = (col_idx == d)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
            
        v_rot = trimesh.transformations.transform_points(v_base, rots[d])
        d_starts = starts[mask]
        
        for i, g_idx in enumerate(indices):
            v_start = g_idx * num_v
            f_start = g_idx * num_f
            all_v[v_start : v_start + num_v] = v_rot + d_starts[i]
            all_f[f_start : f_start + num_f] = f_base + v_start

    colors = np.full((len(all_v), 4), blue, dtype=np.uint8)
    return trimesh.Trimesh(vertices=all_v, faces=all_f, vertex_colors=colors, process=False)



def fast_json_loads(data):
    return orjson.loads(data)
def fast_json_dumps(data):
    return orjson.dumps(data).decode('utf-8')

def iter_feature_json(file_path):
    """
    Iterates over the generated volume features JSON (JSON Lines format) line by line.
    Using a generator prevents out-of-memory errors for massive files (e.g., 13GB+).
    
    Args:
        file_path (str): Path to the output JSON file.
        
    Yields:
        dict: The parsed JSON object for a single line.
    """
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                yield fast_json_loads(line)

def extract_array_from_json(file_path, keys):
    """
    Efficiently reads a massive JSON Lines file ONCE and extracts multiple keys into NumPy arrays.
    This avoids massive memory spikes by discarding heavy surface data immediately.
    
    Args:
        file_path (str): Path to the JSON Lines file.
        keys (list of str): List of string keys to extract (e.g., ['index', 'non_manifold_num']).
        
    Returns:
        dict: A dictionary mapping each requested key to its corresponding NumPy array.
    """
    extracted_data = {key: [] for key in keys}
    
    # Get file size for an accurate byte-based progress bar
    file_size = os.path.getsize(file_path)
    
    with open(file_path, 'r') as f, tqdm(total=file_size, unit='B', unit_scale=True, desc="Extracting features") as pbar:
        for line in f:
            # Update progress bar by the byte length of the line read
            pbar.update(len(line))
            
            line = line.strip()
            if not line:
                continue
                
            # Parse the line using our optimized JSON loader
            item = fast_json_loads(line)
            
            # Extract only the required fields, letting Python garbage collect the rest
            for key in keys:
                if key in item:
                    extracted_data[key].append(item[key])
                    
    # Convert lists to NumPy arrays at the very end
    return {key: np.array(val) for key, val in extracted_data.items()}

def fetch_np_array(data_list, keys):
    """
    Fetches values from a list of dictionaries for specified key(s)
    and formats them into numpy arrays of shape (n, *).
    
    Args:
        data_list (list of dict): The list of dictionaries to extract from.
        keys (str or list of str): The key or list of keys to extract.
        
    Returns:
        np.ndarray or tuple of np.ndarray: The extracted data as numpy arrays.
    """
    if isinstance(keys, str):
        return np.array([d[keys] for d in data_list])
    elif isinstance(keys, (list, tuple)):
        return tuple(np.array([d[key] for d in data_list]) for key in keys)
    else:
        raise ValueError("keys must be a string or a list/tuple of strings")

def load_pickle(filepath: str) -> List[Dict[str, Any]]:
    """
    Loads a voxel mapping list of dictionaries from a pickle file.
    
    Args:
        filepath: String, path to the pickle file.
        
    Returns:
        List of dictionaries containing cube indices and feature (face/edge) indices.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")
        
    with open(filepath, "rb") as in_file:
        result_list = pickle.load(in_file)
        
    return result_list


def save_pickle(save_path, data):
    """
    Saves a Python object to a file using pickle.
    
    Args:
        save_path (str): The full path (including filename) where data should be saved.
        data (any): The Python object to serialize.
    """
    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 'wb' stands for Write Binary, which pickle requires
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    RES = 512
    # Verify centering: a point at RES/2 (256) should now be at ~0
    example_coords = np.array([[256, 256, 256]])
    mesh = voxels_to_mesh(example_coords, RES)
    print(f"Voxel mesh center bounds: {mesh.bounds}")