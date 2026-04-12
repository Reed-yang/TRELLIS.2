
import os
import json
import numpy as np
import pickle
import trimesh
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Any, Tuple

from visualize import edges_to_cylinders_mesh
from utils import save_pickle, load_pickle


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Normalizes a mesh into a unit cube [0, 1]^3 while keeping axis proportions.
    The mesh is centered at [0.5, 0.5, 0.5] and its maximum dimension will be exactly 1.

    Args:
        mesh (trimesh.Trimesh): The input mesh.

    Returns:
        trimesh.Trimesh: The normalized mesh.
    """
    # Create a copy to avoid modifying the original mesh in-place
    normalized_mesh = mesh.copy()
    normalized_mesh.merge_vertices(merge_tex=True, merge_norm=True)
    normalized_mesh.remove_unreferenced_vertices()
    mask = normalized_mesh.unique_faces() & normalized_mesh.nondegenerate_faces()
    normalized_mesh.update_faces(mask)

    # Get the bounding box of the mesh
    min_bounds, max_bounds = normalized_mesh.bounds
    extents = max_bounds - min_bounds
    center = (max_bounds + min_bounds) / 2.0

    # Find the maximum extent to scale uniformly (preserving proportions)
    max_extent = np.max(extents)

    # Prevent division by zero in case of a degenerate mesh (e.g., a single point)
    if max_extent == 0:
        return normalized_mesh

    # Step 1: Translate the mesh so its center is at the origin (0, 0, 0)
    normalized_mesh.vertices -= center

    # Step 2: Scale uniformly so the maximum dimension spans exactly 1
    normalized_mesh.vertices /= max_extent
    normalized_mesh.vertices *= 0.947 

    # Step 3: Translate the mesh so its center is at (0.5, 0.5, 0.5)
    # This places it securely inside the [0, 1]^3 unit cube
    normalized_mesh.vertices[:, 2] += 0.513
    normalized_mesh.vertices[:, 1] += 0.506
    normalized_mesh.vertices[:, 0] += 0.489

    return normalized_mesh

def extract_boundaries(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Extracts the boundary edges of open surfaces in a mesh.
    A boundary edge is defined as an edge that is linked to exactly one face.

    Args:
        mesh (trimesh.Trimesh): The input mesh.

    Returns:
        np.ndarray: An (n, 2, 3) array of vertex coordinates representing the boundary edges.
    """
    # Get all edges from the faces and sort the vertex indices of each edge.
    # Sorting ensures that edge (A, B) and edge (B, A) are treated as the same edge.
    edges = np.sort(mesh.edges, axis=1)
    
    # Find unique edges and count how many faces share each edge
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    
    # Boundary edges are those that belong to exactly one face (count == 1)
    boundary_edges = unique_edges[counts == 1]
    
    # Map vertex indices to their actual 3D coordinates
    return mesh.vertices[boundary_edges]

def extract_non_manifolds(mesh: trimesh.Trimesh) -> Tuple[np.ndarray, List[List[int]], np.ndarray, List[List[int]]]:
    """
    Extracts non-manifold edges and vertices, along with the indices of their neighbor faces.
    
    Args:
        mesh (trimesh.Trimesh): The input mesh.
        
    Returns:
        Tuple containing:
        - nm_edges_coords: (N, 2, 3) array of non-manifold edge coordinates.
        - nm_edges_neighbors: List of lists containing neighbor face indices for each non-manifold edge.
        - nm_vertices_coords: (M, 3) array of non-manifold vertex coordinates.
        - nm_vertices_neighbors: List of lists containing neighbor face indices for each non-manifold vertex.
    """
    # 1. Non-manifold edges (shared by > 2 faces)
    edges = np.sort(mesh.edges, axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    nm_edge_v_indices = unique_edges[counts > 2]
    
    if len(nm_edge_v_indices) > 0:
        nm_edges_coords = mesh.vertices[nm_edge_v_indices]
    else:
        nm_edges_coords = np.empty((0, 2, 3))
        
    # Fast mapping of edges to faces to find neighbors
    edge_to_faces = defaultdict(list)
    for f_idx, face in enumerate(mesh.faces):
        e1 = tuple(sorted([face[0], face[1]]))
        e2 = tuple(sorted([face[1], face[2]]))
        e3 = tuple(sorted([face[2], face[0]]))
        edge_to_faces[e1].append(f_idx)
        edge_to_faces[e2].append(f_idx)
        edge_to_faces[e3].append(f_idx)
        
    nm_edges_neighbors = [edge_to_faces[tuple(e)] for e in nm_edge_v_indices]
    
    # 2. Non-manifold vertices
    # Any vertex attached to a non-manifold edge is intrinsically non-manifold
    nm_v_set = set(nm_edge_v_indices.flatten())
    
    # Also find "bowtie" vertices: vertices where faces sharing them form >1 connected component.
    vertex_to_faces = defaultdict(list)
    for f_idx, face in enumerate(mesh.faces):
        for v in face:
            vertex_to_faces[v].append(f_idx)
            
    face_adj_dict = defaultdict(list)
    for f1, f2 in mesh.face_adjacency:
        face_adj_dict[f1].append(f2)
        face_adj_dict[f2].append(f1)
        
    for v, f_list in vertex_to_faces.items():
        if v in nm_v_set or len(f_list) <= 1:
            continue
        
        # BFS to count connected components in the 1-ring umbrella of the vertex
        visited = set()
        components = 0
        f_set = set(f_list)
        
        for start_f in f_list:
            if start_f not in visited:
                components += 1
                if components > 1:
                    nm_v_set.add(v)  # Disconnected umbrella found!
                    break
                queue = [start_f]
                visited.add(start_f)
                while queue:
                    curr_f = queue.pop(0)
                    for adj_f in face_adj_dict[curr_f]:
                        if adj_f in f_set and adj_f not in visited:
                            visited.add(adj_f)
                            queue.append(adj_f)
                            
    nm_vertex_indices = np.array(list(nm_v_set), dtype=int)
    if len(nm_vertex_indices) > 0:
        nm_vertices_coords = mesh.vertices[nm_vertex_indices]
        nm_vertices_neighbors = [vertex_to_faces[v] for v in nm_vertex_indices]
    else:
        nm_vertices_coords = np.empty((0, 3))
        nm_vertices_neighbors = []
        
    return nm_edges_coords, nm_edges_neighbors, nm_vertices_coords, nm_vertices_neighbors


def process_face_to_grid(mesh: trimesh.Trimesh, res: int, save_dir: str) -> List[Dict[str, Any]]:
    """
    Slices the [0, 1]^3 space into res**3 smaller cubes and registers mesh faces to them.
    
    Args:
        mesh: trimesh.Trimesh object (assumed to be normalized to [0,1] bounding box)
        res: Integer, resolution of the voxel grid (e.g., 32 means 32x32x32 grid)
        save_dir: String, path to the directory where results will be saved.
        
    Returns:
        List of dictionaries containing non-empty cubes and their intersecting faces.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Optional: Force normalization to [0, 1] just in case it wasn't strictly enforced
    min_b, max_b = mesh.bounds
    if np.any(min_b < 0.0) or np.any(max_b > 1.0):
        print("Warning: Mesh bounds are outside [0, 1]. The logic assumes a unit cube space.")
    
    voxel_size = 1.0 / res
    h = np.array([voxel_size / 2.0] * 3)  # Half-extents of the voxel
    
    triangles = mesh.vertices[mesh.faces] # Shape: (N, 3, 3)
    num_faces = len(triangles)
    
    # 1. Get bounding box for every triangle to quickly isolate candidate voxels
    min_bounds = triangles.min(axis=1)
    max_bounds = triangles.max(axis=1)
    
    # Convert metric coordinates to integer grid indices
    min_idx = np.floor(min_bounds / voxel_size).astype(int)
    max_idx = np.floor(max_bounds / voxel_size).astype(int)
    
    # Clip to [0, res-1] to handle any float precision edge-cases on the [1.0] boundary
    min_idx = np.clip(min_idx, 0, res - 1)
    max_idx = np.clip(max_idx, 0, res - 1)
    
    # Dictionary to hold face indices for each non-empty cube (i, j, k)
    cube_dict = defaultdict(list)
    
    for f in tqdm(range(num_faces), desc="Mapping faces to voxels"):
        i_min, j_min, k_min = min_idx[f]
        i_max, j_max, k_max = max_idx[f]
        
        # Optimization: If the triangle's bounding box is entirely within a single voxel, 
        # it definitely intersects it. No need for complex exact intersection math.
        if i_min == i_max and j_min == j_max and k_min == k_max:
            cube_dict[(i_min, j_min, k_min)].append(int(f))
            continue
            
        # 2. Generate coordinates for all candidate voxels overlapping the triangle's bounding box
        i_vals = np.arange(i_min, i_max + 1)
        j_vals = np.arange(j_min, j_max + 1)
        k_vals = np.arange(k_min, k_max + 1)
        
        # Meshgrid to get all combinations of (i, j, k)
        I, J, K = np.meshgrid(i_vals, j_vals, k_vals, indexing='ij')
        I_flat = I.flatten()
        J_flat = J.flatten()
        K_flat = K.flatten()
        
        # Convert grid indices to spatial centers of the voxels
        centers = np.stack([I_flat + 0.5, J_flat + 0.5, K_flat + 0.5], axis=1) * voxel_size
        v0, v1, v2 = triangles[f]
        
        # 3. Perform exact Triangle-AABB intersection (vectorized against all candidate voxels)
        valid_mask = vectorized_triangle_box_intersect(v0, v1, v2, centers, h)
        
        # Append face to the passing voxels
        valid_indices = np.where(valid_mask)[0]
        for idx in valid_indices:
            cube_pos = (int(I_flat[idx]), int(J_flat[idx]), int(K_flat[idx]))
            cube_dict[cube_pos].append(int(f))

    # 4. Format into the requested list of dictionaries
    result_list = []
    for (i, j, k), face_indices in cube_dict.items():
        result_list.append({
            "cube_indices": (int(i), int(j), int(k)),
            "face_indices": face_indices
        })
        
    # 5. Save output to save_dir as Pickle
    save_path = os.path.join(save_dir, f"face_registers.pkl")
    with open(save_path, "wb") as out_file:
        pickle.dump(result_list, out_file)
        
    print(f"Registered {num_faces} faces across {len(result_list)} unique small cubes.")
    print(f"Results saved to {save_path}")
    
    return result_list


def vectorized_triangle_box_intersect(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, 
                                      centers: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Separating Axis Theorem (SAT) for Triangle vs multiple AABBs.
    Extremely fast vectorized check to see which bounding boxes actually intersect the triangle.
    """
    M = centers.shape[0]
    valid = np.ones(M, dtype=bool)
    
    # Edges of the triangle
    e0 = v1 - v0
    e1 = v2 - v1
    e2 = v0 - v2
    n = np.cross(e0, e1)
    
    # If the triangle is degenerate (points are collinear), skip
    if np.allclose(n, 0):
        return np.zeros(M, dtype=bool)
        
    # --- Test 1: Triangle Plane vs Box ---
    r_plane = np.dot(h, np.abs(n))
    # Distance from center to plane: dot(n, v0 - center)
    dist_to_plane = np.abs(np.dot(n, v0) - np.dot(centers, n))
    valid &= (dist_to_plane <= r_plane)
    
    if not np.any(valid): 
        return valid
        
    # --- Test 2: The 9 Edge Cross-Product Axes ---
    axes = [
        np.array([0, -e0[2], e0[1]]), np.array([e0[2], 0, -e0[0]]), np.array([-e0[1], e0[0], 0]),
        np.array([0, -e1[2], e1[1]]), np.array([e1[2], 0, -e1[0]]), np.array([-e1[1], e1[0], 0]),
        np.array([0, -e2[2], e2[1]]), np.array([e2[2], 0, -e2[0]]), np.array([-e2[1], e2[0], 0])
    ]
    
    for axis in axes:
        if np.allclose(axis, 0):
            continue
            
        # Project triangle vertices onto the axis
        p0, p1, p2 = np.dot(v0, axis), np.dot(v1, axis), np.dot(v2, axis)
        min_p = min(p0, p1, p2)
        max_p = max(p0, p1, p2)
        
        # Projected box radius
        r_axis = h[0]*abs(axis[0]) + h[1]*abs(axis[1]) + h[2]*abs(axis[2])
        
        # Project box centers onto axis
        proj_centers = np.dot(centers, axis)
        
        # Check overlap: (min_p <= proj_centers + r) AND (max_p >= proj_centers - r)
        overlap = (min_p - proj_centers <= r_axis) & (max_p - proj_centers >= -r_axis)
        
        valid &= overlap
        if not np.any(valid): 
            return valid

    # --- Test 3: Triangle AABB vs Box AABB ---
    for dim in range(3):
        min_p = min(v0[dim], v1[dim], v2[dim])
        max_p = max(v0[dim], v1[dim], v2[dim])
        proj_centers = centers[:, dim]
        overlap = (min_p - proj_centers <= h[dim]) & (max_p - proj_centers >= -h[dim])
        valid &= overlap
        if not np.any(valid):
            return valid

    return valid


def process_boundary_to_grid(boundaries: np.ndarray, res: int, save_dir: str) -> List[Dict[str, Any]]:
    """
    Slices the [0, 1]^3 space into res**3 smaller cubes and registers boundary edges to them.
    
    Args:
        boundaries: np.ndarray of shape (N, 2, 3), coordinates of boundary edges.
        res: Integer, resolution of the voxel grid (e.g., 32 means 32x32x32 grid).
        save_dir: String, path to the directory where results will be saved.
        
    Returns:
        List of dictionaries containing non-empty cubes and their intersecting edges.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    voxel_size = 1.0 / res
    h = np.array([voxel_size / 2.0] * 3)  # Half-extents of the voxel
    
    num_edges = len(boundaries)
    if num_edges == 0:
        print("No boundaries found to process.")
        return []
    
    # 1. Get bounding box for every segment to quickly isolate candidate voxels
    min_bounds = boundaries.min(axis=1)
    max_bounds = boundaries.max(axis=1)
    
    # Convert metric coordinates to integer grid indices
    min_idx = np.floor(min_bounds / voxel_size).astype(int)
    max_idx = np.floor(max_bounds / voxel_size).astype(int)
    
    # Clip to [0, res-1] to handle any float precision edge-cases on the [1.0] boundary
    min_idx = np.clip(min_idx, 0, res - 1)
    max_idx = np.clip(max_idx, 0, res - 1)
    
    # Dictionary to hold edge indices for each non-empty cube (i, j, k)
    cube_dict = defaultdict(list)
    
    for e in tqdm(range(num_edges), desc="Mapping boundaries to voxels"):
        i_min, j_min, k_min = min_idx[e]
        i_max, j_max, k_max = max_idx[e]
        
        # Optimization: If the segment's bounding box is entirely within a single voxel, 
        # it definitely intersects it. No need for complex exact intersection math.
        if i_min == i_max and j_min == j_max and k_min == k_max:
            cube_dict[(i_min, j_min, k_min)].append(int(e))
            continue
            
        # 2. Generate coordinates for all candidate voxels overlapping the segment's bounding box
        i_vals = np.arange(i_min, i_max + 1)
        j_vals = np.arange(j_min, j_max + 1)
        k_vals = np.arange(k_min, k_max + 1)
        
        # Meshgrid to get all combinations of (i, j, k)
        I, J, K = np.meshgrid(i_vals, j_vals, k_vals, indexing='ij')
        I_flat = I.flatten()
        J_flat = J.flatten()
        K_flat = K.flatten()
        
        # Convert grid indices to spatial centers of the voxels
        centers = np.stack([I_flat + 0.5, J_flat + 0.5, K_flat + 0.5], axis=1) * voxel_size
        v0, v1 = boundaries[e]
        
        # 3. Perform exact Segment-AABB intersection (vectorized against all candidate voxels)
        valid_mask = vectorized_segment_box_intersect(v0, v1, centers, h)
        
        # Append edge to the passing voxels
        valid_indices = np.where(valid_mask)[0]
        for idx in valid_indices:
            cube_pos = (int(I_flat[idx]), int(J_flat[idx]), int(K_flat[idx]))
            cube_dict[cube_pos].append(int(e))

    # 4. Format into the requested list of dictionaries
    result_list = []
    for (i, j, k), edge_indices in cube_dict.items():
        result_list.append({
            "cube_indices": (int(i), int(j), int(k)),
            "edge_indices": edge_indices
        })
        
    # 5. Save output to save_dir as Pickle
    save_path = os.path.join(save_dir, f"boundary_registers.pkl")
    with open(save_path, "wb") as out_file:
        pickle.dump(result_list, out_file)
        
    print(f"Registered {num_edges} edges across {len(result_list)} unique small cubes.")
    print(f"Results saved to {save_path}")
    
    return result_list


def vectorized_segment_box_intersect(v0: np.ndarray, v1: np.ndarray, 
                                     centers: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Separating Axis Theorem (SAT) for 3D Line Segment vs multiple AABBs.
    """
    c = (v0 + v1) / 2.0
    d = (v1 - v0) / 2.0
    d_abs = np.abs(d)
    
    # Distance from AABB centers to segment center
    c_prime = c - centers  # Shape: (M, 3)
    
    # --- Test 1: Bounding Box Check (3 axes parallel to AABB face normals) ---
    valid = np.all(np.abs(c_prime) <= (h + d_abs), axis=1)
    
    if not np.any(valid):
        return valid
        
    # --- Test 2: Cross Product Checks (3 axes parallel to the cross products of edges) ---
    # Adding a tiny epsilon to handle floating point imprecision
    eps = 1e-8
    
    cross_x = np.abs(c_prime[:, 1] * d[2] - c_prime[:, 2] * d[1])
    rad_x = h[1] * d_abs[2] + h[2] * d_abs[1]
    valid &= (cross_x <= rad_x + eps)
    
    if not np.any(valid): 
        return valid
    
    cross_y = np.abs(c_prime[:, 2] * d[0] - c_prime[:, 0] * d[2])
    rad_y = h[2] * d_abs[0] + h[0] * d_abs[2]
    valid &= (cross_y <= rad_y + eps)
    
    if not np.any(valid): 
        return valid
    
    cross_z = np.abs(c_prime[:, 0] * d[1] - c_prime[:, 1] * d[0])
    rad_z = h[0] * d_abs[1] + h[1] * d_abs[0]
    valid &= (cross_z <= rad_z + eps)
    
    return valid


def process_non_manifolds_to_grid(nm_edges: np.ndarray, nm_vertices: np.ndarray, res: int, save_dir: str) -> List[Dict[str, Any]]:
    """
    Slices the [0, 1]^3 space into res**3 smaller cubes and registers non-manifold features to them.
    
    Args:
        nm_edges: np.ndarray of shape (N, 2, 3), coordinates of non-manifold edges.
        nm_vertices: np.ndarray of shape (M, 3), coordinates of non-manifold vertices.
        res: Integer, resolution of the voxel grid.
        save_dir: String, path to the directory where results will be saved.
        
    Returns:
        List of dictionaries containing non-empty cubes and their intersecting feature indices.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    voxel_size = 1.0 / res
    h = np.array([voxel_size / 2.0] * 3)  # Half-extents of the voxel
    
    # Dictionary to hold feature indices for each cube
    cube_dict = defaultdict(lambda: {"nm_edge_indices": [], "nm_vertex_indices": []})
    
    # 1. Process Edges (Same approach as boundary edges)
    num_edges = len(nm_edges)
    if num_edges > 0:
        min_bounds = nm_edges.min(axis=1)
        max_bounds = nm_edges.max(axis=1)
        
        min_idx = np.floor(min_bounds / voxel_size).astype(int)
        max_idx = np.floor(max_bounds / voxel_size).astype(int)
        min_idx = np.clip(min_idx, 0, res - 1)
        max_idx = np.clip(max_idx, 0, res - 1)
        
        for e in tqdm(range(num_edges), desc="Mapping nm-edges to voxels"):
            i_min, j_min, k_min = min_idx[e]
            i_max, j_max, k_max = max_idx[e]
            
            if i_min == i_max and j_min == j_max and k_min == k_max:
                cube_dict[(i_min, j_min, k_min)]["nm_edge_indices"].append(int(e))
                continue
                
            i_vals = np.arange(i_min, i_max + 1)
            j_vals = np.arange(j_min, j_max + 1)
            k_vals = np.arange(k_min, k_max + 1)
            
            I, J, K = np.meshgrid(i_vals, j_vals, k_vals, indexing='ij')
            I_flat = I.flatten()
            J_flat = J.flatten()
            K_flat = K.flatten()
            
            centers = np.stack([I_flat + 0.5, J_flat + 0.5, K_flat + 0.5], axis=1) * voxel_size
            v0, v1 = nm_edges[e]
            
            valid_mask = vectorized_segment_box_intersect(v0, v1, centers, h)
            valid_indices = np.where(valid_mask)[0]
            for idx in valid_indices:
                cube_pos = (int(I_flat[idx]), int(J_flat[idx]), int(K_flat[idx]))
                cube_dict[cube_pos]["nm_edge_indices"].append(int(e))

    # 2. Process Vertices (Fast point-in-AABB logic)
    num_vertices = len(nm_vertices)
    if num_vertices > 0:
        grid_indices = np.floor(nm_vertices / voxel_size).astype(int)
        grid_indices = np.clip(grid_indices, 0, res - 1)
        for v_idx, (i, j, k) in enumerate(grid_indices):
            cube_dict[(int(i), int(j), int(k))]["nm_vertex_indices"].append(int(v_idx))

    # 3. Format into the requested list of dictionaries
    result_list = []
    for (i, j, k), data in cube_dict.items():
        result_list.append({
            "cube_indices": (int(i), int(j), int(k)),
            "nm_edge_indices": data["nm_edge_indices"],
            "nm_vertex_indices": data["nm_vertex_indices"]
        })
        
    # 4. Save output to save_dir
    save_path = os.path.join(save_dir, f"nm_registers.pkl")
    with open(save_path, "wb") as out_file:
        pickle.dump(result_list, out_file)
        
    print(f"Registered {num_edges} nm-edges and {num_vertices} nm-vertices across {len(result_list)} unique small cubes.")
    
    return result_list


def load_voxel_mapping(filepath: str) -> List[Dict[str, Any]]:
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


def visualize_face_registers(mesh: trimesh.Trimesh, res: int, cube_index: Tuple[int, int, int], face_indices: List[int]) -> trimesh.Trimesh:
    """
    Extracts the specified faces into a submesh, creates a tubular wireframe mesh 
    representing the voxel cube, and merges both into a single trimesh.Trimesh.
    
    Args:
        mesh: The original trimesh.Trimesh object.
        res: The resolution of the voxel grid.
        cube_index: Tuple (i, j, k) representing the voxel's grid indices.
        face_indices: List of integer face indices to extract.
        
    Returns:
        A single trimesh.Trimesh containing both the faces and the cube wireframe.
    """
    scale_factor = 1.0
    voxel_size = (1.0 / res) * scale_factor
    i, j, k = cube_index
    
    # Calculate the precise center of the voxel
    center = np.array([i + 0.5, j + 0.5, k + 0.5]) * voxel_size
    
    # 1. Extract submesh and scale it
    sub_mesh = mesh.submesh([face_indices], append=True)
    sub_mesh.apply_scale(scale_factor)
    
    if hasattr(sub_mesh.visual, 'face_colors'):
        # Paint submesh light blue
        sub_mesh.visual.face_colors = [100, 200, 255, 200]
        
    # 2. Create wireframe as an actual mesh (cylinders for edges) so it can be merged
    box = trimesh.creation.box(extents=[voxel_size, voxel_size, voxel_size])
    box.apply_translation(center)
    
    # Create a thin cylinder for each edge of the box
    edge_radius = voxel_size * 0.02  # thickness is 2% of the voxel size
    cylinders = []
    
    for edge in box.edges_unique:
        p0 = box.vertices[edge[0]]
        p1 = box.vertices[edge[1]]
        # Create a cylinder segment for this edge
        cyl_mesh = trimesh.creation.cylinder(radius=edge_radius, segment=[p0, p1])
        cylinders.append(cyl_mesh)
        
    wireframe_mesh = trimesh.util.concatenate(cylinders)
    if hasattr(wireframe_mesh.visual, 'face_colors'):
        # Paint wireframe dark gray
        wireframe_mesh.visual.face_colors = [50, 50, 50, 255]
        
    # 3. Merge the submesh and wireframe mesh into a single object
    merged_mesh = trimesh.util.concatenate([sub_mesh, wireframe_mesh])
    
    return merged_mesh


def visualize_boundary_registers(boundaries: np.ndarray, res: int, cube_index: Tuple[int, int, int], edge_indices: List[int]) -> trimesh.Trimesh:
    """
    Extracts the specified boundary edges into a cylinder mesh, creates a tubular wireframe mesh 
    representing the voxel cube, and merges both into a single trimesh.Trimesh.
    
    Args:
        boundaries: np.ndarray of shape (N, 2, 3), coordinates of boundary edges.
        res: The resolution of the voxel grid.
        cube_index: Tuple (i, j, k) representing the voxel's grid indices.
        edge_indices: List of integer edge indices to extract.
        
    Returns:
        A single trimesh.Trimesh containing both the boundary edges and the cube wireframe.
    """
    scale_factor = 1000.0
    voxel_size = (1.0 / res) * scale_factor
    i, j, k = cube_index
    
    # Calculate the precise center of the voxel
    center = np.array([i + 0.5, j + 0.5, k + 0.5]) * voxel_size
    
    # 1. Create mesh for the selected boundary edges and scale them
    selected_boundaries = boundaries[edge_indices] * scale_factor
    boundary_radius = voxel_size * 0.02  # Make boundaries slightly thicker than the box wireframe
    boundary_cylinders = []
    
    for edge_segment in selected_boundaries:
        cyl = trimesh.creation.cylinder(radius=boundary_radius, segment=edge_segment)
        boundary_cylinders.append(cyl)
        
    if boundary_cylinders:
        boundary_mesh = trimesh.util.concatenate(boundary_cylinders)
        if hasattr(boundary_mesh.visual, 'face_colors'):
            # Paint boundaries red for high visibility
            boundary_mesh.visual.face_colors = [255, 255, 0, 255]
    else:
        # Fallback to an empty mesh if no boundaries were passed
        boundary_mesh = trimesh.Trimesh()
        
    # 2. Create wireframe as an actual mesh (cylinders for edges) so it can be merged
    box = trimesh.creation.box(extents=[voxel_size, voxel_size, voxel_size])
    box.apply_translation(center)
    
    # Create a thin cylinder for each edge of the box
    edge_radius = voxel_size * 0.02  # Thickness matching face registers
    cylinders = []
    
    for edge in box.edges_unique:
        p0 = box.vertices[edge[0]]
        p1 = box.vertices[edge[1]]
        # Create a cylinder segment for this edge
        cyl_mesh = trimesh.creation.cylinder(radius=edge_radius, segment=[p0, p1])
        cylinders.append(cyl_mesh)
        
    wireframe_mesh = trimesh.util.concatenate(cylinders)
    if hasattr(wireframe_mesh.visual, 'face_colors'):
        # Paint wireframe dark gray
        wireframe_mesh.visual.face_colors = [50, 50, 50, 255]
        
    # 3. Merge the boundary mesh and wireframe mesh into a single object
    merged_mesh = trimesh.util.concatenate([boundary_mesh, wireframe_mesh])
    
    return merged_mesh


def visualize_non_manifolds(
    mesh: trimesh.Trimesh, 
    nm_edges: np.ndarray, 
    nm_vertices: np.ndarray, 
    nm_edges_neighbors: List[List[int]], 
    nm_vertices_neighbors: List[List[int]], 
    res: int, 
    cube_index: Tuple[int, int, int], 
    nm_edge_indices: List[int], 
    nm_vertex_indices: List[int]
) -> trimesh.Trimesh:
    """
    Visualizes the non-manifold edges, vertices, and their neighbor faces inside a voxel.
    
    Args:
        mesh: The original trimesh.Trimesh object.
        nm_edges: Array of non-manifold edges.
        nm_vertices: Array of non-manifold vertices.
        nm_edges_neighbors: Neighbor face mappings for edges.
        nm_vertices_neighbors: Neighbor face mappings for vertices.
        res: The resolution of the voxel grid.
        cube_index: Tuple (i, j, k).
        nm_edge_indices: List of non-manifold edge indices to extract.
        nm_vertex_indices: List of non-manifold vertex indices to extract.
        
    Returns:
        A single trimesh.Trimesh containing the neighbors, features, and the cube wireframe.
    """
    scale_factor = 1000.0
    voxel_size = (1.0 / res) * scale_factor
    i, j, k = cube_index
    center = np.array([i + 0.5, j + 0.5, k + 0.5]) * voxel_size
    
    meshes_to_merge = []
    
    # 1. Gather all unique neighbor face indices
    face_indices = set()
    for e_idx in nm_edge_indices:
        face_indices.update(nm_edges_neighbors[e_idx])
    for v_idx in nm_vertex_indices:
        face_indices.update(nm_vertices_neighbors[v_idx])
        
    # 2. Extract submesh of contextual neighbor faces
    if len(face_indices) > 0:
        sub_mesh = mesh.submesh([list(face_indices)], append=True)
        sub_mesh.apply_scale(scale_factor)
        if hasattr(sub_mesh.visual, 'face_colors'):
            sub_mesh.visual.face_colors = [255, 165, 0, 180]  # Translucent orange faces
        meshes_to_merge.append(sub_mesh)
        
    # 3. Create cylinders for non-manifold edges
    edge_radius = voxel_size * 0.04
    if len(nm_edge_indices) > 0:
        selected_edges = nm_edges[nm_edge_indices] * scale_factor
        for edge_segment in selected_edges:
            cyl = trimesh.creation.cylinder(radius=edge_radius, segment=edge_segment)
            if hasattr(cyl.visual, 'face_colors'):
                cyl.visual.face_colors = [255, 0, 0, 255] # Solid red edges
            meshes_to_merge.append(cyl)
            
    # 4. Create spheres for non-manifold vertices
    vertex_radius = voxel_size * 0.06
    if len(nm_vertex_indices) > 0:
        selected_vertices = nm_vertices[nm_vertex_indices] * scale_factor
        for vertex_pt in selected_vertices:
            sph = trimesh.creation.icosphere(radius=vertex_radius)
            sph.apply_translation(vertex_pt)
            if hasattr(sph.visual, 'face_colors'):
                sph.visual.face_colors = [0, 255, 0, 255] # Solid green vertices
            meshes_to_merge.append(sph)
            
    # 5. Create wireframe box
    box = trimesh.creation.box(extents=[voxel_size, voxel_size, voxel_size])
    box.apply_translation(center)
    wireframe_radius = voxel_size * 0.02
    cylinders = []
    
    for edge in box.edges_unique:
        p0 = box.vertices[edge[0]]
        p1 = box.vertices[edge[1]]
        cyl_mesh = trimesh.creation.cylinder(radius=wireframe_radius, segment=[p0, p1])
        cylinders.append(cyl_mesh)
        
    wireframe_mesh = trimesh.util.concatenate(cylinders)
    if hasattr(wireframe_mesh.visual, 'face_colors'):
        wireframe_mesh.visual.face_colors = [50, 50, 50, 255] # Dark Gray wireframe
    meshes_to_merge.append(wireframe_mesh)
    
    return trimesh.util.concatenate(meshes_to_merge)


def voxelize(mesh: trimesh.Trimesh, output_directory: str, resolution: int) -> Tuple[trimesh.Trimesh, np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    os.makedirs(output_directory, exist_ok=True)
    norm_mesh = normalize_mesh(mesh)
    norm_mesh.export(f'{output_directory}/norm_mesh.ply')

    boundaries = extract_boundaries(norm_mesh)
    save_pickle(f'{output_directory}/open_boundary.pkl', boundaries)

    edges = edges_to_cylinders_mesh(boundaries)
    edges.export(f'{output_directory}/open_boundary.ply')

    nm_edges, nm_edges_neighbors, nm_vertices, nm_vertices_neighbors = extract_non_manifolds(norm_mesh)
    save_pickle(f'{output_directory}/non_manifold.pkl', [nm_edges, nm_edges_neighbors, nm_vertices, nm_vertices_neighbors])

    face_registers = process_face_to_grid(norm_mesh, res=resolution, save_dir=output_directory)
    boundary_registers = process_boundary_to_grid(boundaries, res=resolution, save_dir=output_directory)
    nm_registers = process_non_manifolds_to_grid(nm_edges, nm_vertices, res=resolution, save_dir=output_directory)

    return norm_mesh, boundaries, face_registers, boundary_registers, nm_registers

# --- Example Usage ---
if __name__ == "__main__":
    # mesh_path = 'tmp/test_mesh/shivaji_maharaj_cloth.ply'
    mesh_path = 'tmp/test_mesh/turbine__turbofan_engine__jet_engine.ply'
    output_directory = "tmp/test_rep"
    resolution = 1024

    os.makedirs(output_directory, exist_ok=True)


    original_mesh = trimesh.load(mesh_path)

    # Normalize the mesh
    norm_mesh = normalize_mesh(original_mesh)
    norm_mesh.export('tmp/test_rep/norm_mesh.ply')

    # --- Test Boundary Extraction ---
    print("\n--- Boundary Extraction Test ---")
    
    boundaries = extract_boundaries(norm_mesh)
    save_pickle('tmp/test_rep/open_boundary.pkl', boundaries)

    edges = edges_to_cylinders_mesh(boundaries)
    edges.export('tmp/test_rep/open_boundary.ply')

    nm_edges, nm_edges_neighbors, nm_vertices, nm_vertices_neighbors = extract_non_manifolds(norm_mesh)
    save_pickle('tmp/test_rep/non_manifold.pkl', [nm_edges, nm_edges_neighbors, nm_vertices, nm_vertices_neighbors])

    # 2. Run the voxel processing
    
    # 3. Retrieve results
    face_registers = process_face_to_grid(norm_mesh, res=resolution, save_dir=output_directory)
    boundary_registers = process_boundary_to_grid(boundaries, res=resolution, save_dir=output_directory)
    nm_registers = process_non_manifolds_to_grid(nm_edges, nm_vertices, res=resolution, save_dir=output_directory)

    
    # Look at a sample
    if face_registers:
        print(f"\nExample of one dict entry:")
        print(f"Cube index: {face_registers[0]['cube_indices']}")
        print(f"Face indices count: {len(face_registers[0]['face_indices'])}")

        merged = visualize_face_registers(
            mesh=norm_mesh, 
            res=resolution, 
            cube_index=face_registers[0]['cube_indices'], 
            face_indices=face_registers[0]['face_indices']
        )
        merged.export('tmp/test_rep/sample_face.ply')

    if boundary_registers:
        print(f"\nExample of one boundary dict entry:")
        print(f"Cube index: {boundary_registers[0]['cube_indices']}")
        print(f"Edge indices count: {len(boundary_registers[0]['edge_indices'])}")

        merged_boundary = visualize_boundary_registers(
            boundaries=boundaries, 
            res=resolution, 
            cube_index=boundary_registers[0]['cube_indices'], 
            edge_indices=boundary_registers[0]['edge_indices']
        )
        merged_boundary.export('tmp/test_rep/sample_boundary.ply')

    if nm_registers:
        print(f"\nExample of one non-manifold dict entry:")
        print(f"Cube index: {nm_registers[0]['cube_indices']}")
        print(f"NM Edge indices count: {len(nm_registers[0]['nm_edge_indices'])}")
        print(f"NM Vertex indices count: {len(nm_registers[0]['nm_vertex_indices'])}")

        merged_nm = visualize_non_manifolds(
            mesh=norm_mesh,
            nm_edges=nm_edges,
            nm_vertices=nm_vertices,
            nm_edges_neighbors=nm_edges_neighbors,
            nm_vertices_neighbors=nm_vertices_neighbors,
            res=resolution,
            cube_index=nm_registers[0]['cube_indices'],
            nm_edge_indices=nm_registers[0]['nm_edge_indices'],
            nm_vertex_indices=nm_registers[0]['nm_vertex_indices']
        )
        merged_nm.export('tmp/test_rep/sample_non_manifold.ply')








