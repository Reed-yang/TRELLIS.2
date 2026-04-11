import collections
import multiprocessing as mp
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

def sort_loops_in_cube(cube_dict):
    """
    Assigns ranks to each edge intersection of loops in a cube to prevent intersections.
    
    Args:
        cube_dict (dict): A dictionary containing 'edge_weights' and 'loops'.
                          'edge_weights' is a list of 18 integers.
                          'loops' is a list of lists, where each inner list is a 
                          sequence of edge indices.
                          
    Returns:
        list of dicts: Each dict corresponds to a loop, structured as 
                       {'loop': [e1, e2, ...], 'rank': [r1, r2, ...]}.
    """
    edge_weights = cube_dict.get('edge_weights', [0]*18)
    input_loops = cube_dict.get('loops', [])
    
    if not input_loops:
        return []

    # Hardcoded topological mappings
    edges_vertices = {
        0: (0, 1), 1: (1, 2), 2: (2, 3), 3: (3, 0),
        4: (4, 5), 5: (5, 6), 6: (6, 7), 7: (7, 4),
        8: (0, 4), 9: (1, 5), 10: (2, 6), 11: (3, 7),
        12: (0, 2), 13: (4, 6), 14: (1, 4), 15: (1, 6),
        16: (2, 7), 17: (0, 7)
    }
    
    faces_edges = [
        (0, 1, 12),  # T0
        (2, 3, 12),  # T1
        (4, 5, 13),  # T2
        (6, 7, 13),  # T3
        (0, 8, 14),  # T4
        (4, 9, 14),  # T5
        (1, 10, 15), # T6
        (5, 9, 15),  # T7
        (2, 11, 16), # T8
        (6, 10, 16), # T9
        (3, 11, 17), # T10
        (7, 8, 17)   # T11
    ]
    
    def get_points_near(E, v, count, W):
        """
        Gets `count` points on edge E closest to vertex v.
        Rank 0 is closest to the start vertex (u1), Rank W-1 is closest to end (u2).
        Returns a list of tuples: (Edge, Rank) ordered by increasing distance to v.
        """
        if count <= 0:
            return []
        u1, u2 = edges_vertices[E]
        if v == u1:
            # Ranks 0, 1, 2... are closest to u1
            return [(E, r) for r in range(count)]
        elif v == u2:
            # Ranks W-1, W-2... are closest to u2
            return [(E, r) for r in range(W - 1, W - 1 - count, -1)]
        else:
            raise ValueError(f"Vertex {v} is not on edge {E}")

    # Build an adjacency graph of point connections inside the triangles
    # Nodes are tuples: (edge_index, rank)
    adj = collections.defaultdict(list)
    
    for T in faces_edges:
        E_A, E_B, E_C = T
        W_A = edge_weights[E_A]
        W_B = edge_weights[E_B]
        W_C = edge_weights[E_C]
        
        # Calculate how many curves go between each pair of edges in this triangle
        N_AB = (W_A + W_B - W_C) // 2
        N_BC = (W_B + W_C - W_A) // 2
        N_CA = (W_C + W_A - W_B) // 2
        
        # Find shared vertex for each corner
        v_AB = list(set(edges_vertices[E_A]) & set(edges_vertices[E_B]))[0]
        v_BC = list(set(edges_vertices[E_B]) & set(edges_vertices[E_C]))[0]
        v_CA = list(set(edges_vertices[E_C]) & set(edges_vertices[E_A]))[0]
        
        # Extract points ordered by proximity to the corner they cut off
        pts_A_near_AB = get_points_near(E_A, v_AB, N_AB, W_A)
        pts_B_near_AB = get_points_near(E_B, v_AB, N_AB, W_B)
        
        pts_B_near_BC = get_points_near(E_B, v_BC, N_BC, W_B)
        pts_C_near_BC = get_points_near(E_C, v_BC, N_BC, W_C)
        
        pts_C_near_CA = get_points_near(E_C, v_CA, N_CA, W_C)
        pts_A_near_CA = get_points_near(E_A, v_CA, N_CA, W_A)
        
        # Connect them: closest-to-closest, yielding perfectly nested non-intersecting chords
        for p1, p2 in zip(pts_A_near_AB, pts_B_near_AB):
            adj[p1].append(p2)
            adj[p2].append(p1)
            
        for p1, p2 in zip(pts_B_near_BC, pts_C_near_BC):
            adj[p1].append(p2)
            adj[p2].append(p1)
            
        for p1, p2 in zip(pts_C_near_CA, pts_A_near_CA):
            adj[p1].append(p2)
            adj[p2].append(p1)

    # Traverse the 2-regular graph to extract completed topological loops
    visited = set()
    traced_loops = []
    
    for node in list(adj.keys()):
        if node not in visited:
            cycle = []
            curr = node
            prev = None
            
            while True:
                cycle.append(curr)
                visited.add(curr)
                neighbors = adj[curr]
                
                # Topological safety check (should never happen on valid triangulated mesh)
                if len(neighbors) != 2: 
                    break
                    
                next_node = neighbors[0] if neighbors[0] != prev else neighbors[1]
                
                if next_node in visited:
                    break
                    
                prev = curr
                curr = next_node
            
            # Store the cycle if it forms a legitimate loop
            if len(cycle) >= 3:
                traced_loops.append(cycle)

    # Match the internally traced loops with the provided loops to align their ranks
    used_traced = [False] * len(traced_loops)
    results = []
    
    for in_loop in input_loops:
        k = len(in_loop)
        matched = False
        
        for i, t_loop in enumerate(traced_loops):
            if used_traced[i] or len(t_loop) != k: 
                continue
            
            t_edges = [p[0] for p in t_loop]
            
            # Check forward sequence matching
            for shift in range(k):
                if all(t_edges[(shift + j) % k] == in_loop[j] for j in range(k)):
                    ranks = [t_loop[(shift + j) % k][1] for j in range(k)]
                    results.append({'loop': in_loop, 'rank': ranks})
                    used_traced[i] = True
                    matched = True
                    break
            if matched: break
            
            # Check backward sequence matching
            for shift in range(k):
                if all(t_edges[(shift - j) % k] == in_loop[j] for j in range(k)):
                    ranks = [t_loop[(shift - j) % k][1] for j in range(k)]
                    results.append({'loop': in_loop, 'rank': ranks})
                    used_traced[i] = True
                    matched = True
                    break
            if matched: break
        
        # Fallback for topological discrepancies in input arrays
        if not matched:
            results.append({'loop': in_loop, 'rank': [0] * k})

    return results

def assign_points_to_loops(cube_dict, resolution):
    """
    Assigns each component point to a sorted loop to form a non-intersecting triplet surface.
    Uses the Hungarian algorithm to minimize the sum of squared distances between
    loop centroids and component points.
    
    Args:
        cube_dict (dict): Dictionary containing 'sorted_loops', 'component_points',
                          'cube_indices', and 'edge_weights'.
        resolution (float/int): The resolution of the grid to compute global coordinates.
        
    Returns:
        list: The modified 'sorted_loops' list with 'component_point' added to each loop.
    """
    sorted_loops = cube_dict.get('sorted_loops', [])
    component_points = cube_dict.get('component_points', [])
    edge_weights = cube_dict.get('edge_weights', [0]*18)
    
    if not sorted_loops or not component_points:
        return sorted_loops
        
    ix, iy, iz = cube_dict.get('cube_indices', (0, 0, 0))
    d = 1.0 / resolution
    x0, y0, z0 = ix * d, iy * d, iz * d
    
    # Hardcoded local-to-global vertex coordinate mapping
    vertices = {
        0: (x0, y0, z0),
        1: (x0+d, y0, z0),
        2: (x0+d, y0+d, z0),
        3: (x0, y0+d, z0),
        4: (x0, y0, z0+d),
        5: (x0+d, y0, z0+d),
        6: (x0+d, y0+d, z0+d),
        7: (x0, y0+d, z0+d)
    }
    
    edges_vertices = {
        0: (0, 1), 1: (1, 2), 2: (2, 3), 3: (3, 0),
        4: (4, 5), 5: (5, 6), 6: (6, 7), 7: (7, 4),
        8: (0, 4), 9: (1, 5), 10: (2, 6), 11: (3, 7),
        12: (0, 2), 13: (4, 6), 14: (1, 4), 15: (1, 6),
        16: (2, 7), 17: (0, 7)
    }
    
    loop_centroids = []
    for loop_data in sorted_loops:
        edges = loop_data['loop']
        ranks = loop_data['rank']
        
        cx, cy, cz = 0.0, 0.0, 0.0
        k = len(edges)
        
        for e, r in zip(edges, ranks):
            u1, u2 = edges_vertices[e]
            v1, v2 = vertices[u1], vertices[u2]
            W = edge_weights[e]
            
            # Calculate parametric intersection t
            # Rank 0 is closest to u1, Rank W-1 is closest to u2
            t = (r + 1) / (W + 1) if W > 0 else 0.5
            
            px = v1[0] + t * (v2[0] - v1[0])
            py = v1[1] + t * (v2[1] - v1[1])
            pz = v1[2] + t * (v2[2] - v1[2])
            
            cx += px
            cy += py
            cz += pz
            
        if k > 0:
            cx /= k
            cy /= k
            cz /= k
        
        loop_centroids.append((cx, cy, cz))
        
    # Cost matrix: Sum of squared Euclidean distances
    # Minimizing squared distance naturally penalizes crossing bipartite matches in 3D
    n_loops = len(loop_centroids)
    n_points = len(component_points)
    cost_matrix = []
    
    for i in range(n_loops):
        row = []
        for j in range(n_points):
            c = loop_centroids[i]
            p = component_points[j]
            dist_sq = (c[0]-p[0])**2 + (c[1]-p[1])**2 + (c[2]-p[2])**2
            row.append(dist_sq)
        cost_matrix.append(row)
        
    # Hungarian algorithm to find optimal non-intersecting matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Initialize defaults
    for loop_data in sorted_loops:
        loop_data['component_point'] = None
        
    # Apply optimal assignments
    for r, c in zip(row_ind, col_ind):
        sorted_loops[r]['component_point'] = component_points[c]
        
    return sorted_loops

def _process_cube_wrapper(args):
    """
    Wrapper function for multiprocessing to attach results directly to the dict.
    """
    cube_dict, resolution = args
    # Create a shallow copy to prevent shared state issues and ensure pure functions
    new_dict = cube_dict.copy()
    new_dict['sorted_loops'] = sort_loops_in_cube(new_dict)
    
    if resolution is not None:
        new_dict['sorted_loops'] = assign_points_to_loops(new_dict, resolution)
        
    return new_dict

def process_multiple_cubes(cube_dicts, resolution=None, batch_size=1000, num_workers=None):
    """
    Processes a list of cube dictionaries in parallel using batches to control memory footprint.
    
    Args:
        cube_dicts (list): List of cube dictionaries.
        resolution (float, optional): Grid resolution to pair component points to loops.
        batch_size (int): Number of cubes to serialize and process in one IPC batch.
        num_workers (int): Number of parallel worker processes. Default is CPU count.
        
    Returns:
        list: A new list of cube dictionaries containing the 'sorted_loops' key.
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
        
    results = []
    total_cubes = len(cube_dicts)
    
    with mp.Pool(processes=num_workers) as pool:
        # We iterate in chunks to limit the amount of serialized data in IPC pipes
        with tqdm(total=total_cubes, desc="Processing cubes") as pbar:
            for i in range(0, total_cubes, batch_size):
                batch = cube_dicts[i : i + batch_size]
                
                # Package args for multiprocessing
                batch_args = [(c, resolution) for c in batch]
                
                # pool.map automatically preserves the original input order
                batch_results = pool.map(_process_cube_wrapper, batch_args)
                results.extend(batch_results)
                pbar.update(len(batch))
                
    return results



def collapse_point(cube_dicts, resolution):
    processed_cubes = process_multiple_cubes(cube_dicts, resolution, batch_size=10000)
    return processed_cubes

# Example Usage
if __name__ == "__main__":
    example_cube = {
        'cube_indices': (1, 1, 2), 'face_indices': [0, 1], 'num_components': 1, 
        'edge_weights': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1], 
        'loops': [[8, 14, 9, 15, 10, 16, 11, 17]], 'num_loops': 1, 'error': None, 
        'face_weights': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'num_boundary': 0, 
        'component_points': [[0.375, 0.375, 0.513]]
    }
    
    # Simulate a list of cubes
    cube_list = [example_cube for _ in range(500)]
    
    # Process them using the parallel batch function and pass the resolution
    resolution = 4  # Example resolution
    processed_cubes = process_multiple_cubes(cube_list, resolution=resolution, batch_size=100)
    
    print(f"\nSuccessfully processed {len(processed_cubes)} cubes.")
    print("First processed cube's sorted loops (with paired point):")
    for l in processed_cubes[0]['sorted_loops']:
        print(l)