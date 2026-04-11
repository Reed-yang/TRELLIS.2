import numpy as np
import trimesh
import networkx as nx
import multiprocessing as mp
from tqdm import tqdm

from volume_feature import get_neighborhood_components
from utils import load_pickle, voxels_to_mesh, fetch_np_array


def filter_single_component_neighborhoods(neighborhoods, indices):
    """
    Filters the neighborhoods to extract those where the center cube has exactly 1 component,
    and all other surrounding cubes have no more than 1 component.

    Args:
        neighborhoods (np.ndarray): An (N, 3, 3, 3) array of component counts.
        indices (np.ndarray): An (N, 3) array of integer coordinates for the cubes.

    Returns:
        tuple: 
            - np.ndarray: An (N', 3, 3, 3) array containing only the elements satisfying the condition.
            - np.ndarray: An (N', 3) array containing the corresponding filtered indices.
    """
    if neighborhoods.shape[0] == 0:
        return neighborhoods, indices
        
    # Condition 1: The center element (1, 1, 1) is exactly 1
    center_is_one = neighborhoods[:, 1, 1, 1] == 1
    
    # Condition 2: All elements in the 3x3x3 grid are <= 1.
    # Since we already check the center is 1, checking if the maximum value
    # in the whole 3x3x3 sub-grid is <= 1 ensures no neighbors are > 1.
    all_le_one = neighborhoods.max(axis=(1, 2, 3)) <= 1
    
    # Combine conditions using bitwise AND
    valid_mask = center_is_one & all_le_one
    
    # Filter both arrays
    return neighborhoods[valid_mask], indices[valid_mask]


# Pre-compute all potential 6-connected edges in a flattened 3x3x3 grid (indices 0 to 26).
# Defined at the module level so worker processes can access it efficiently without pickling overhead.
_POTENTIAL_EDGES = []
for _x in range(3):
    for _y in range(3):
        for _z in range(3):
            _idx = _x * 9 + _y * 3 + _z
            # Link adjacent cubes on x, y, and z axes
            if _x < 2: _POTENTIAL_EDGES.append((_idx, (_x + 1) * 9 + _y * 3 + _z))
            if _y < 2: _POTENTIAL_EDGES.append((_idx, _x * 9 + (_y + 1) * 3 + _z))
            if _z < 2: _POTENTIAL_EDGES.append((_idx, _x * 9 + _y * 3 + _z + 1))


def _check_planarity_worker(grid_3d):
    """Worker function to check planarity of a single 3x3x3 neighborhood."""
    flat_grid = grid_3d.flatten()
    
    # Get the nodes (flattened indices where the value is 1)
    active_nodes = np.where(flat_grid == 1)[0]
    if len(active_nodes) == 0:
        return True
        
    active_set = set(active_nodes)

    # Build the graph
    G = nx.Graph()
    G.add_nodes_from(active_nodes)

    # Add an edge only if both adjacent cubes are '1'
    for u, v in _POTENTIAL_EDGES:
        if u in active_set and v in active_set:
            G.add_edge(u, v)

    # Check planarity using NetworkX's built-in algorithm
    is_planar, _ = nx.check_planarity(G)
    return is_planar


def filter_planar_graphs(neighborhoods, indices, batch_size=100000, num_workers=None):
    """
    Takes an (N, 3, 3, 3) array of neighborhoods (where elements are assumed 
    to be 0 or 1), forms a graph for each 3x3x3 grid, and filters out the grids 
    whose resulting graph is NOT planar. Uses multiprocessing and batching.
    
    Nodes are cubes with the value 1.
    Edges exist between nodes that share a facet (6-connected neighbors).

    Args:
        neighborhoods (np.ndarray): An (N, 3, 3, 3) array of component counts.
        indices (np.ndarray): An (N, 3) array of integer coordinates for the cubes.
        batch_size (int): Max number of neighborhoods to send to workers at once 
                          to cap memory overhead.
        num_workers (int, optional): Number of CPU cores to use. Defaults to all available.

    Returns:
        tuple: 
            - np.ndarray: An (N', 3, 3, 3) array containing only the planar neighborhoods.
            - np.ndarray: An (N', 3) array containing the corresponding filtered indices.
    """
    N = neighborhoods.shape[0]
    if N == 0:
        return neighborhoods, indices

    if num_workers is None:
        num_workers = mp.cpu_count()

    valid_mask = np.zeros(N, dtype=bool)

    # Process in batches to strictly control memory usage from multiprocessing overhead
    with mp.Pool(processes=num_workers) as pool:
        with tqdm(total=N, desc="Filtering planar graphs") as pbar:
            for start_idx in range(0, N, batch_size):
                end_idx = min(start_idx + batch_size, N)
                batch = neighborhoods[start_idx:end_idx]
                
                # Calculate an optimal chunksize based on current batch and workers.
                chunk_size = max(1, len(batch) // (num_workers * 4))
                
                # Use pool.imap to yield results lazily and update tqdm smoothly
                batch_results = []
                for is_planar in pool.imap(_check_planarity_worker, batch, chunksize=chunk_size):
                    batch_results.append(is_planar)
                    pbar.update(1)
                    
                valid_mask[start_idx:end_idx] = batch_results

    # Filter and return both arrays
    return neighborhoods[valid_mask], indices[valid_mask]


def generate_simulated_planar_mesh(neighborhoods, indices, res=None, min_idx=None, max_idx=None):
    """
    Generates a 3D surface mesh that simulates the local planar structure 
    for each 3x3x3 neighborhood. It fits a plane to the active voxels using PCA, 
    intersects the plane with the center voxel bounding box to form a planar polygon,
    and aggregates all these polygons into a single welded mesh.

    Args:
        neighborhoods (np.ndarray): An (N, 3, 3, 3) array of planar component counts.
        indices (np.ndarray): An (N, 3) array of corresponding cube coordinates.
        res (int, optional): The resolution of the full grid. If provided, 
                             vertices are scaled back to the (0, 1) unit cube domain.
        min_idx (int or array-like, optional): Minimum spatial index (inclusive) to process.
        max_idx (int or array-like, optional): Maximum spatial index (inclusive) to process.

    Returns:
        tuple:
            - vertices (np.ndarray): An (V, 3) array of mesh vertices.
            - faces (np.ndarray): An (F, 3) array of triangular face indices.
    """
    # Filter neighborhoods and indices if spatial bounding values are provided
    if min_idx is not None or max_idx is not None:
        valid_mask = np.ones(indices.shape[0], dtype=bool)
        if min_idx is not None:
            valid_mask &= np.all(indices >= min_idx, axis=1)
        if max_idx is not None:
            valid_mask &= np.all(indices <= max_idx, axis=1)
            
        neighborhoods = neighborhoods[valid_mask]
        indices = indices[valid_mask]

    if neighborhoods.shape[0] == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
        
    all_vertices = []
    all_faces = []
    
    # Define local voxel cube bounds from -0.5 to 0.5
    cube_verts = np.array([
        [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5]
    ])
    
    # 12 edges of the cube defined by vertex index pairs
    cube_edges = [
        (0,1), (1,2), (2,3), (3,0), # bottom face
        (4,5), (5,6), (6,7), (7,4), # top face
        (0,4), (1,5), (2,6), (3,7)  # vertical pillars
    ]
    
    for i in range(len(neighborhoods)):
        grid = neighborhoods[i]
        idx = indices[i]
        
        # Get local coords (-1 to 1) of active cubes relative to center
        pts = np.argwhere(grid == 1) - np.array([1, 1, 1])
        
        if len(pts) < 3:
            # Need at least a line of points to form a plane/quad
            continue
            
        # Calculate PCA to find the dominant normal vector
        pts_mean = np.mean(pts, axis=0)
        centered_pts = pts - pts_mean
        cov = centered_pts.T @ centered_pts
        
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        # Smallest eigenvalue's eigenvector corresponds to the plane's normal vector
        n = eigvecs[:, 0]
        u = eigvecs[:, 1]
        v = eigvecs[:, 2]
        
        # Heuristic to force consistent normal orientation (mostly pointing "Up")
        # This prevents adjacent planar pieces from having randomly flipped face windings
        if n[2] < 0 or (np.isclose(n[2], 0) and n[1] < 0) or (np.isclose(n[2], 0) and np.isclose(n[1], 0) and n[0] < 0):
            n = -n
            
        # Ensure (n, u, v) forms a right-handed coordinate system for predictable sorting
        if np.linalg.det(np.column_stack((n, u, v))) < 0:
            v = -v
            
        # We enforce the plane passes through (0,0,0) (the exact center of the middle cube)
        # Equation of plane: n . x = 0
        intersections = []
        for v1, v2 in cube_edges:
            p1 = cube_verts[v1]
            p2 = cube_verts[v2]
            
            d1 = np.dot(p1, n)
            d2 = np.dot(p2, n)
            
            # Check if the edge crosses the plane
            if (d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0):
                t = d1 / (d1 - d2)
                intersections.append(p1 + t * (p2 - p1))
            elif d1 == 0:
                intersections.append(p1)
            elif d2 == 0:
                intersections.append(p2)
                
        if len(intersections) < 3:
            continue
            
        # Filter down to unique intersection vertices (to form a clean polygon)
        intersections = np.array(intersections)
        rounded_inter = np.round(intersections, decimals=5)
        _, unique_idx = np.unique(rounded_inter, axis=0, return_index=True)
        intersections = intersections[unique_idx]
        
        if len(intersections) < 3:
            continue
            
        # Sort intersections angularly to construct the convex polygon
        poly_center = np.mean(intersections, axis=0)
        vecs = intersections - poly_center
        
        # Project vectors onto the 2D plane to get sorting angles
        angles = np.arctan2(np.dot(vecs, v), np.dot(vecs, u))
        sorted_points = intersections[np.argsort(angles)]
        
        # Translate from local (-0.5 to 0.5) to global grid coordinates
        global_points = sorted_points + idx + 0.5
        
        # Normalize into the (0, 1) unit cube domain if res is provided
        if res is not None:
            global_points = global_points / res
            
        # Triangulate the polygon using a triangle fan from the 0th vertex
        base_idx = len(all_vertices)
        all_vertices.extend(global_points)
        
        for j in range(1, len(global_points) - 1):
            all_faces.append([base_idx, base_idx + j, base_idx + j + 1])
            
    if len(all_vertices) > 0:
        all_vertices = np.array(all_vertices)
        all_faces = np.array(all_faces)
        
        # Weld duplicate vertices at the borders of adjacent cubes to form a continuous mesh
        rounded_verts = np.round(all_vertices, decimals=5)
        unique_verts, inverse_indices = np.unique(rounded_verts, axis=0, return_inverse=True)
        
        return unique_verts, inverse_indices[all_faces]
    else:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)


def generate_graph_mesh(neighborhoods, indices, res=None, min_idx=None, max_idx=None, thickness=0.1):
    """
    Generates a 3D surface mesh that visualizes the graph structure of the neighborhoods.
    Nodes are represented as small cubes, and edges (6-connected components) are represented 
    as thin rectangular prisms connecting the nodes. Extremely fast due to pure vectorization.

    Args:
        neighborhoods (np.ndarray): An (N, 3, 3, 3) array of planar component counts.
        indices (np.ndarray): An (N, 3) array of corresponding cube coordinates.
        res (int, optional): The resolution of the full grid. If provided, 
                             vertices are scaled back to the (0, 1) unit cube domain.
        min_idx (int or array-like, optional): Minimum spatial index (inclusive) to process.
        max_idx (int or array-like, optional): Maximum spatial index (inclusive) to process.
        thickness (float, optional): The thickness of the nodes and edges.

    Returns:
        tuple:
            - vertices (np.ndarray): An (V, 3) array of mesh vertices.
            - faces (np.ndarray): An (F, 3) array of triangular face indices.
    """
    # Filter neighborhoods and indices if spatial bounding values are provided
    if min_idx is not None or max_idx is not None:
        valid_mask = np.ones(indices.shape[0], dtype=bool)
        if min_idx is not None:
            valid_mask &= np.all(indices >= min_idx, axis=1)
        if max_idx is not None:
            valid_mask &= np.all(indices <= max_idx, axis=1)
            
        neighborhoods = neighborhoods[valid_mask]
        indices = indices[valid_mask]

    if neighborhoods.shape[0] == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
        
    N = neighborhoods.shape[0]
    
    # 1. Extract and deduplicate all unique nodes globally
    active_mask = neighborhoods == 1
    active_idx = np.argwhere(active_mask) # (K, 4) where cols are [batch_idx, x, y, z]
    batch_idx = active_idx[:, 0]
    local_xyz = active_idx[:, 1:] - 1
    global_nodes = indices[batch_idx] + local_xyz
    unique_nodes = np.unique(global_nodes, axis=0)
    
    # 2. Extract and deduplicate all unique edges globally
    neighborhoods_flat = neighborhoods.reshape(N, 27)
    edge_list = []
    
    for u, v in _POTENTIAL_EDGES:
        # Check where both nodes of the potential edge are active
        valid_edge_mask = (neighborhoods_flat[:, u] == 1) & (neighborhoods_flat[:, v] == 1)
        valid_idx = np.where(valid_edge_mask)[0]
        
        if len(valid_idx) == 0:
            continue
            
        u_coord = np.array(np.unravel_index(u, (3, 3, 3))) - 1
        v_coord = np.array(np.unravel_index(v, (3, 3, 3))) - 1
        
        u_global = indices[valid_idx] + u_coord
        v_global = indices[valid_idx] + v_coord
        
        edges = np.stack([u_global, v_global], axis=1) # (E_sub, 2, 3)
        edge_list.append(edges)
        
    if edge_list:
        all_edges = np.concatenate(edge_list, axis=0)
        # Flatten the inner dimensions to use np.unique on rows
        E = all_edges.shape[0]
        all_edges_flat = all_edges.reshape(E, 6)
        unique_edges_flat = np.unique(all_edges_flat, axis=0)
        unique_edges = unique_edges_flat.reshape(-1, 2, 3)
    else:
        unique_edges = np.zeros((0, 2, 3), dtype=int)
        
    # 3. Generate mesh geometry (boxes for nodes and edges)
    V_nodes = unique_nodes.shape[0]
    E_edges = unique_edges.shape[0]
    
    unit_box = np.array([
        [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5]
    ])
    
    unit_faces = np.array([
        [0, 2, 1], [0, 3, 2], # bottom
        [4, 5, 6], [4, 6, 7], # top
        [0, 1, 5], [0, 5, 4], # front
        [1, 2, 6], [1, 6, 5], # right
        [2, 3, 7], [2, 7, 6], # back
        [3, 0, 4], [3, 4, 7]  # left
    ])
    
    all_vertices_list = []
    
    # Add node boxes
    if V_nodes > 0:
        node_centers = unique_nodes + 0.5
        node_dims = np.full((V_nodes, 3), thickness)
        scaled_nodes = unit_box[None, :, :] * node_dims[:, None, :]
        translated_nodes = scaled_nodes + node_centers[:, None, :]
        all_vertices_list.append(translated_nodes.reshape(-1, 3))
        
    # Add edge boxes
    if E_edges > 0:
        p1 = unique_edges[:, 0, :]
        p2 = unique_edges[:, 1, :]
        edge_centers = (p1 + p2) / 2.0 + 0.5
        diffs = p2 - p1
        # Length along the connection axis is 1.0 + thickness
        edge_dims = np.where(diffs != 0, 1.0 + thickness, thickness)
        
        scaled_edges = unit_box[None, :, :] * edge_dims[:, None, :]
        translated_edges = scaled_edges + edge_centers[:, None, :]
        all_vertices_list.append(translated_edges.reshape(-1, 3))
        
    if not all_vertices_list:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
        
    vertices = np.concatenate(all_vertices_list, axis=0)
    
    # Normalize coordinates
    if res is not None:
        vertices = vertices / res
        
    # Construct faces
    total_boxes = V_nodes + E_edges
    offsets = (np.arange(total_boxes) * 8)[:, None, None]
    faces = unit_faces[None, :, :] + offsets
    faces = faces.reshape(-1, 3)
    
    return vertices, faces


# Example usage (for testing purposes):
if __name__ == "__main__":
    load_dir = 'tmp/test_rep'
    output_dir = 'tmp/test_rep'
    res = 1024

    vis_len = -1
    min_idx = 500
    max_idx = 700

    mesh = trimesh.load(f'{load_dir}/norm_mesh.ply')
    boundary = load_pickle(f'{load_dir}/open_boundary.pkl')
    
    volume_feature_face = load_pickle(f'{load_dir}/volume_feature_face.pkl')
    volume_feature_boundary = load_pickle(f'{load_dir}/volume_feature_boundary.pkl')
    
    volume_indices_face, volume_num_face = fetch_np_array(volume_feature_face, ["cube_indices", "num_components"])
    volume_indices_boundary, volume_num_boundary = fetch_np_array(volume_feature_boundary, ["cube_indices", "num_components"])
    
    volume_indices_face_neighbor_components = get_neighborhood_components(volume_indices_face, volume_num_face)
    volume_indices_boundary_neighbor_components = get_neighborhood_components(volume_indices_boundary, volume_num_boundary)
    
    volume_indices_face_single_component_neighborhoods, volume_indices_face_single_component_neighborhoods_indices = filter_single_component_neighborhoods(volume_indices_face_neighbor_components, volume_indices_face)

    print(f"Number of single component neighborhoods: {volume_indices_face_single_component_neighborhoods.shape[0]}")
    print(f"Number of total neighborhoods: {volume_indices_face_neighbor_components.shape[0]}")
    print(f"Percentage of single component neighborhoods: {volume_indices_face_single_component_neighborhoods.shape[0] / volume_indices_face_neighbor_components.shape[0] * 100}%")
    
    volume_indices_face_single_component_neighborhoods_planar, volume_indices_face_single_component_neighborhoods_planar_indices = filter_planar_graphs(volume_indices_face_single_component_neighborhoods, volume_indices_face_single_component_neighborhoods_indices)
    print(f"Number of planar neighborhoods: {volume_indices_face_single_component_neighborhoods_planar.shape[0]}")
    print(f"Number of total neighborhoods: {volume_indices_face_single_component_neighborhoods.shape[0]}")
    print(f"Percentage of planar neighborhoods: {volume_indices_face_single_component_neighborhoods_planar.shape[0] / volume_indices_face_single_component_neighborhoods.shape[0] * 100}%")
    
    print("percentage of single component neighborhoods that are planar: ", volume_indices_face_single_component_neighborhoods_planar.shape[0] / volume_indices_face_neighbor_components.shape[0] * 100)
    
    vertices, faces = generate_simulated_planar_mesh(volume_indices_face_single_component_neighborhoods_planar[:vis_len], volume_indices_face_single_component_neighborhoods_planar_indices[:vis_len], res, min_idx, max_idx)
    mesh = trimesh.Trimesh(vertices, faces)
    mesh.export(f'{output_dir}/simulated_planar_mesh.ply')

    vertices, faces = generate_graph_mesh(volume_indices_face_single_component_neighborhoods_planar[:vis_len], volume_indices_face_single_component_neighborhoods_planar_indices[:vis_len], res, min_idx, max_idx)
    mesh = trimesh.Trimesh(vertices, faces)
    mesh.export(f'{output_dir}/graph_mesh.ply')
    
    breakpoint()
    
    
    





