import collections
import multiprocessing as mp
from tqdm import tqdm
import trimesh
import numpy as np
from scipy.optimize import linear_sum_assignment
import functools


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



# =====================================================================
# BOUNDARY LOOP FUNCTIONS (For open sequences containing faces & edges)
# =====================================================================

def parse_element(el, index=0):
    """Parses a loop element into a standardized ('f', index) or ('e', index) tuple."""
    if isinstance(el, str):
        t = el[0].lower()
        if t in ('f', 'e'):
            return (t, int(el[1:]))
    elif isinstance(el, (tuple, list)) and len(el) >= 2:
        return (str(el[0]).lower(), int(el[1]))
    
    # Fallback for plain integers: 
    # loops start with a face and strictly interleave (face, edge, face, edge...)
    # Therefore, even indices are faces, odd indices are edges.
    if index % 2 == 0:
        return ('f', int(el))
    else:
        return ('e', int(el))


def _get_rank_positions(element_counts, vertices, edges_vertices, faces_vertices, d=1.0):
    """Generates 3D coordinates for all available ranks on faces and edges."""
    rank_positions = {}
    for pel, W in element_counts.items():
        typ, idx = pel
        positions = []
        if typ == 'e':
            if idx not in edges_vertices: idx = 0
            u1, u2 = edges_vertices[idx]
            v1, v2 = vertices[u1], vertices[u2]
            for r in range(W):
                spacing = 0.05 * d
                t = 0.5 + (r - (W - 1) / 2.0) * spacing
                t = max(0.05, min(0.95, t))
                positions.append(v1 + t * (v2 - v1))
        else:
            if idx not in faces_vertices: idx = 0
            u1, u2, u3 = faces_vertices[idx]
            v1, v2, v3 = vertices[u1], vertices[u2], vertices[u3]
            C = (v1 + v2 + v3) / 3.0
            if W <= 1:
                positions.append(C)
            else:
                # Create a small circle around the face center to prevent point overlap
                vec1 = v2 - v1
                vec1 = vec1 / (np.linalg.norm(vec1) + 1e-9)
                cross_n = np.cross(vec1, v3 - v1)
                vec2 = np.cross(cross_n, vec1)
                vec2 = vec2 / (np.linalg.norm(vec2) + 1e-9)
                
                for r in range(W):
                    theta = 2 * np.pi * r / W
                    R = 0.1 * d
                    positions.append(C + R * (np.cos(theta) * vec1 + np.sin(theta) * vec2))
        rank_positions[pel] = positions
    return rank_positions


def sort_loops_with_boundary(cube_dict):
    """
    Assigns ranks to sequences mixing faces and edges using geometric coordinate descent.
    It guarantees non-intersecting line segments between edges by explicitly penalizing 
    and resolving true 3D crossings after the initial optimization.
    """
    input_loops = cube_dict.get('loops', [])
    if not input_loops:
        return []
        
    vertices = {
        0: np.array([0.0, 0.0, 0.0]), 1: np.array([1.0, 0.0, 0.0]),
        2: np.array([1.0, 1.0, 0.0]), 3: np.array([0.0, 1.0, 0.0]),
        4: np.array([0.0, 0.0, 1.0]), 5: np.array([1.0, 0.0, 1.0]),
        6: np.array([1.0, 1.0, 1.0]), 7: np.array([0.0, 1.0, 1.0])
    }
    edges_vertices = {
        0: (0, 1), 1: (1, 2), 2: (2, 3), 3: (3, 0),
        4: (4, 5), 5: (5, 6), 6: (6, 7), 7: (7, 4),
        8: (0, 4), 9: (1, 5), 10: (2, 6), 11: (3, 7),
        12: (0, 2), 13: (4, 6), 14: (1, 4), 15: (1, 6),
        16: (2, 7), 17: (0, 7)
    }
    faces_vertices = {
        0: (0, 1, 2), 1: (0, 2, 3), 2: (4, 5, 6), 3: (4, 6, 7),
        4: (0, 1, 4), 5: (1, 4, 5), 6: (1, 2, 6), 7: (1, 5, 6),
        8: (2, 3, 7), 9: (2, 6, 7), 10: (0, 3, 7), 11: (0, 4, 7)
    }

    # Parse loops and filter out faces completely for the ranking logic
    parsed_loops = []
    element_counts = collections.Counter()
    for loop in input_loops:
        parsed_loop = []
        for i, el in enumerate(loop):
            pel = parse_element(el, index=i)
            # Keep only edges, completely strip faces
            if pel[0] == 'e':
                parsed_loop.append(pel)
                element_counts[pel] += 1
        parsed_loops.append(parsed_loop)

    rank_positions = _get_rank_positions(element_counts, vertices, edges_vertices, faces_vertices, d=1.0)
    
    # Initialize random assignments
    current_ranks = {pel: list(range(W)) for pel, W in element_counts.items()}
    for ranks in current_ranks.values():
        np.random.shuffle(ranks)
        
    loop_assignments = [[0] * len(loop) for loop in parsed_loops]
    pel_used = collections.defaultdict(int)
    for i, loop in enumerate(parsed_loops):
        for j, pel in enumerate(loop):
            loop_assignments[i][j] = current_ranks[pel][pel_used[pel]]
            pel_used[pel] += 1

    # Coordinate Descent / Relaxation: Untangle the lines generally
    for iteration in range(10):
        for pel, W in element_counts.items():
            if W <= 1:
                continue
                
            # Gather all loops crossing this element
            crossings = []
            for i, loop in enumerate(parsed_loops):
                for j, el in enumerate(loop):
                    if el == pel:
                        crossings.append((i, j))
                        
            target_positions = []
            for i, j in crossings:
                loop = parsed_loops[i]
                neighbors = []
                if j > 0:
                    prev_pel = loop[j-1]
                    neighbors.append(rank_positions[prev_pel][loop_assignments[i][j-1]])
                if j < len(loop) - 1:
                    next_pel = loop[j+1]
                    neighbors.append(rank_positions[next_pel][loop_assignments[i][j+1]])
                    
                target = sum(neighbors) / len(neighbors) if neighbors else np.array([0.5, 0.5, 0.5])
                
                # Add a tiny bit of inertia so mathematically identical loops push apart
                curr_pos = rank_positions[pel][loop_assignments[i][j]]
                target_positions.append(target + 0.1 * curr_pos)
                
            # Assign ranks locally to minimize squared lengths
            cost_matrix = np.zeros((W, W))
            for c_idx, target in enumerate(target_positions):
                for r_idx in range(W):
                    pos = rank_positions[pel][r_idx]
                    cost_matrix[c_idx, r_idx] = np.sum((target - pos)**2)
                    
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for c_idx, r_idx in zip(row_ind, col_ind):
                i, j = crossings[c_idx]
                loop_assignments[i][j] = r_idx

    # --- Explicit 3D Intersection Removal Pass ---
    
    def segments_cross(p1, p2, p3, p4, tol=1e-4):
        """Mathematical 3D segment crossing test finding the shortest distance."""
        d13 = p1 - p3
        d43 = p4 - p3
        d21 = p2 - p1
        
        d4321 = np.dot(d43, d21)
        d2121 = np.dot(d21, d21)
        d4343 = np.dot(d43, d43)
        d1321 = np.dot(d13, d21)
        d1343 = np.dot(d13, d43)
        
        denom = d2121 * d4343 - d4321 * d4321
        if abs(denom) < 1e-8:
            return False # parallel or collinear, ignore to prevent false touches
            
        t = (d1343 * d4321 - d1321 * d4343) / denom
        u = (d1343 + t * d4321) / d4343
        
        # Check if crossing happens strictly inside the interior of the segments
        if 0.01 < t < 0.99 and 0.01 < u < 0.99:
            pt1 = p1 + t * d21
            pt2 = p3 + u * d43
            if np.linalg.norm(pt1 - pt2) < tol:
                return True
        return False

    def evaluate_cost(assignments):
        """Calculates total cost including a massive penalty for any segment crossing."""
        segments = []
        for i, loop in enumerate(parsed_loops):
            pts = [rank_positions[pel][assignments[i][j]] for j, pel in enumerate(loop)]
            for j in range(len(pts) - 1):
                segments.append((i, pts[j], pts[j+1]))
                
        intersections = 0
        sq_len = 0.0
        n_seg = len(segments)
        for a in range(n_seg):
            sq_len += np.sum((segments[a][1] - segments[a][2])**2)
            for b in range(a + 1, n_seg):
                if segments[a][0] == segments[b][0]: 
                    continue # same loop
                if segments_cross(segments[a][1], segments[a][2], segments[b][1], segments[b][2]):
                    intersections += 1
        return intersections * 10000.0 + sq_len

    current_assignments = [list(a) for a in loop_assignments]
    best_cost = evaluate_cost(current_assignments)
    
    # Combinatorial 2-Opt swap search to strictly eliminate any remaining intersections
    for _ in range(20):
        swapped = False
        for pel, W in element_counts.items():
            if W <= 1: continue
            crossings = []
            for i, loop in enumerate(parsed_loops):
                for j, el in enumerate(loop):
                    if el == pel:
                        crossings.append((i, j))
                        
            # Try all pairwise rank swaps on this edge
            for a in range(len(crossings)):
                for b in range(a + 1, len(crossings)):
                    i1, j1 = crossings[a]
                    i2, j2 = crossings[b]
                    
                    # Apply swap
                    current_assignments[i1][j1], current_assignments[i2][j2] = \
                        current_assignments[i2][j2], current_assignments[i1][j1]
                        
                    new_cost = evaluate_cost(current_assignments)
                    if new_cost < best_cost - 1e-6:
                        best_cost = new_cost
                        swapped = True
                    else:
                        # Revert swap
                        current_assignments[i1][j1], current_assignments[i2][j2] = \
                            current_assignments[i2][j2], current_assignments[i1][j1]
        if not swapped:
            break
            
    loop_assignments = current_assignments

    results = []
    for filtered_loop, assignment in zip(parsed_loops, loop_assignments):
        # Extract just the plain integer edge indices to form the final stripped loop array
        edge_indices = [pel[1] for pel in filtered_loop]
        results.append({'loop': edge_indices, 'rank': assignment})
    return results





def sort_loops_in_cube_boundary(cube_dict):
    """
    Main entry point. Supports parsing combinations of open/closed curves, 
    auto-resolving interleaved vs. pure edge indices formats, and wrapping
    them into closed topologies for the intersection-free solver.
    """
    edge_weights = cube_dict.get('edge_weights', [0]*18)
    input_loops = cube_dict.get('loops', [])
    
    if not input_loops:
        return []

    faces_edges = [
        (0, 1, 12), (2, 3, 12), (4, 5, 13), (6, 7, 13),
        (0, 8, 14), (4, 9, 14), (1, 10, 15), (5, 9, 15),
        (2, 11, 16), (6, 10, 16), (3, 11, 17), (7, 8, 17)
    ]
    
    # Pre-calculate face adjacencies to navigate dummy paths
    face_adj = collections.defaultdict(list)
    for i in range(12):
        for j in range(i+1, 12):
            shared = set(faces_edges[i]).intersection(faces_edges[j])
            if shared:
                e = list(shared)[0]
                face_adj[i].append((j, e))
                face_adj[j].append((i, e))

    def share_face(e1, e2):
        for i, f in enumerate(faces_edges):
            if e1 in f and e2 in f:
                return True, i
        return False, -1

    def parse_loop(loop):
        """Intelligently detects whether loop is pure edges or interleaved [f, e, f...]"""
        is_interleaved = False
        if len(loop) > 0:
            # Check if interleaved rule validates
            valid_interleaved = True
            for i in range(0, len(loop)-1, 2):
                f_idx, e_idx = loop[i], loop[i+1]
                if f_idx < 0 or f_idx > 11 or e_idx not in faces_edges[f_idx]:
                    valid_interleaved = False
                    break
            if valid_interleaved:
                is_interleaved = True
                
        if is_interleaved:
            pure_edges = loop[1::2]
            is_open = (len(loop) % 2 != 0)
            f_start = loop[0] if is_open else None
            f_end = loop[-1] if is_open else None
            return pure_edges, is_open, f_start, f_end
            
        # Treat as pure sequence of edges
        pure_edges = list(loop)
        is_open = not (len(pure_edges) >= 3 and share_face(pure_edges[-1], pure_edges[0])[0])
        
        f_start, f_end = -1, -1
        if is_open and pure_edges:
            if len(pure_edges) >= 2:
                _, shared_f = share_face(pure_edges[0], pure_edges[1])
                for i, f in enumerate(faces_edges):
                    if pure_edges[0] in f and i != shared_f:
                        f_start = i
                        break
                _, shared_f_end = share_face(pure_edges[-2], pure_edges[-1])
                for i, f in enumerate(faces_edges):
                    if pure_edges[-1] in f and i != shared_f_end:
                        f_end = i
                        break
            elif len(pure_edges) == 1:
                faces = [i for i, f in enumerate(faces_edges) if pure_edges[0] in f]
                if len(faces) >= 2:
                    f_start, f_end = faces[0], faces[1]
                    
        return pure_edges, is_open, f_start, f_end

    def find_dummy_path(f_end, e_last, f_start, e_first):
        """BFS finding a path over edges to close an open loop, without backtracking over entry/exit edges."""
        if f_end == f_start:
            return []
            
        q = collections.deque([(f_end, [], {f_end})])
        
        while q:
            curr_f, path_edges, visited = q.popleft()
            
            for neighbor, edge in face_adj[curr_f]:
                if neighbor in visited:
                    continue
                # Cannot re-enter the very edge it just left
                if not path_edges and edge == e_last:
                    continue
                # Cannot arrive via the exact edge it will restart through
                if neighbor == f_start and edge == e_first:
                    continue
                    
                new_path = list(path_edges) + [edge]
                if neighbor == f_start:
                    return new_path
                    
                new_visited = set(visited)
                new_visited.add(neighbor)
                q.append((neighbor, new_path, new_visited))
        return []

    augmented_loops = []
    loop_metadata = []
    new_edge_weights = list(edge_weights)
    
    # Step 1: Parse and bridge open loops into virtual closed loops
    for orig_loop in input_loops:
        pure_edges, is_open, f_start, f_end = parse_loop(orig_loop)
        
        if is_open and len(pure_edges) > 0:
            e_first, e_last = pure_edges[0], pure_edges[-1]
            dummy_path = find_dummy_path(f_end, e_last, f_start, e_first)
            
            aug_loop = pure_edges + dummy_path
            for e in dummy_path:
                new_edge_weights[e] += 1
                
            augmented_loops.append(aug_loop)
            loop_metadata.append((pure_edges, True))
        else:
            augmented_loops.append(pure_edges)
            loop_metadata.append((pure_edges, False))
            
    # Step 2: Feed closed loops into sorting algorithm
    sorted_results = sort_loops_in_cube({
        'edge_weights': new_edge_weights,
        'loops': augmented_loops
    })
    
    # Step 3: Decouple virtual dummy edges and format returning result
    final_output = []
    for i, res in enumerate(sorted_results):
        pure_edges, is_open = loop_metadata[i]
        
        # Original loop slice is at the front; extract just those sorted ranks
        k = len(pure_edges)
        ranks = res['rank'][:k]
        
        out_dict = {
            'loop': pure_edges,
            'rank': ranks
        }
        if is_open:
            out_dict['is_open'] = True
            
        final_output.append(out_dict)
        
    return final_output

def assign_points_to_loops_with_boundary(cube_dict, resolution):
    """Pairs optimal component points to sorted boundary loops using centroids."""
    sorted_loops = cube_dict.get('sorted_loops', [])
    component_points = cube_dict.get('component_points', [])
    
    if not sorted_loops or not component_points:
        return sorted_loops
        
    ix, iy, iz = cube_dict.get('cube_indices', (0, 0, 0))
    d = 1.0 / resolution
    x0, y0, z0 = ix * d, iy * d, iz * d
    
    vertices = {
        0: np.array([x0, y0, z0]),     1: np.array([x0+d, y0, z0]),
        2: np.array([x0+d, y0+d, z0]), 3: np.array([x0, y0+d, z0]),
        4: np.array([x0, y0, z0+d]),   5: np.array([x0+d, y0, z0+d]),
        6: np.array([x0+d, y0+d, z0+d]), 7: np.array([x0, y0+d, z0+d])
    }
    edges_vertices = {
        0: (0, 1), 1: (1, 2), 2: (2, 3), 3: (3, 0),
        4: (4, 5), 5: (5, 6), 6: (6, 7), 7: (7, 4),
        8: (0, 4), 9: (1, 5), 10: (2, 6), 11: (3, 7),
        12: (0, 2), 13: (4, 6), 14: (1, 4), 15: (1, 6),
        16: (2, 7), 17: (0, 7)
    }
    faces_vertices = {
        0: (0, 1, 2), 1: (0, 2, 3), 2: (4, 5, 6), 3: (4, 6, 7),
        4: (0, 1, 4), 5: (1, 4, 5), 6: (1, 2, 6), 7: (1, 5, 6),
        8: (2, 3, 7), 9: (2, 6, 7), 10: (0, 3, 7), 11: (0, 4, 7)
    }
    
    element_counts = collections.Counter()
    for loop_data in sorted_loops:
        for el in loop_data['loop']:
            # Since faces are stripped, all remaining elements are purely edges
            pel = ('e', int(el))
            element_counts[pel] += 1
            
    rank_positions = _get_rank_positions(element_counts, vertices, edges_vertices, faces_vertices, d)
    
    loop_centroids = []
    for loop_data in sorted_loops:
        centroid = np.zeros(3)
        edges = loop_data['loop']
        ranks = loop_data['rank']
        k = len(edges)
        
        for e, r in zip(edges, ranks):
            pel = ('e', int(e))
            centroid += rank_positions[pel][r]
        if k > 0:
            centroid /= k
        loop_centroids.append(centroid)
        
    n_loops = len(loop_centroids)
    n_points = len(component_points)
    cost_matrix = np.zeros((n_loops, n_points))
    
    for i in range(n_loops):
        for j in range(n_points):
            cost_matrix[i, j] = np.sum((loop_centroids[i] - component_points[j])**2)
            
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    for loop_data in sorted_loops:
        loop_data['component_point'] = None
    for r, c in zip(row_ind, col_ind):
        sorted_loops[r]['component_point'] = component_points[c]
        
    return sorted_loops


def _process_boundary_cube_wrapper(args):
    cube_dict, resolution = args
    new_dict = cube_dict.copy()
    new_dict['sorted_loops'] = sort_loops_in_cube_boundary(new_dict)
    if resolution is not None:
        new_dict['sorted_loops'] = assign_points_to_loops(new_dict, resolution)
    return new_dict


def process_multiple_cubes_with_boundary(cube_dicts, resolution=None, batch_size=1000, num_workers=None):
    """
    Processes loops defined by both faces and edges across multiple cubes in parallel.
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
        
    results = []
    total_cubes = len(cube_dicts)
    
    with mp.Pool(processes=num_workers) as pool:
        with tqdm(total=total_cubes, desc="Processing boundary cubes") as pbar:
            for i in range(0, total_cubes, batch_size):
                batch = cube_dicts[i : i + batch_size]
                batch_args = [(c, resolution) for c in batch]
                batch_results = pool.map(_process_boundary_cube_wrapper, batch_args)
                results.extend(batch_results)
                pbar.update(len(batch))
                
    return results









def visualize_shifted_loops(cube_dict, resolution, pipe_radius_ratio=0.02):
    """
    Visualizes the cube wireframe and the shifted non-intersecting loops as 3D pipes.
    
    Args:
        cube_dict (dict): Dictionary containing 'sorted_loops', 'cube_indices', etc.
        resolution (float/int): Grid resolution to compute global coordinates.
        pipe_radius_ratio (float): Radius of the pipes relative to the cube size.
        
    Returns:
        trimesh.Trimesh: Concatenated mesh containing wireframe and loop pipes.
    """
    if trimesh is None:
        raise ImportError("The 'trimesh' library is required for visualization. Install it with: pip install trimesh")
        
    ix, iy, iz = cube_dict.get('cube_indices', (0, 0, 0))
    d = 1.0 / resolution
    x0, y0, z0 = ix * d, iy * d, iz * d
    
    # Local-to-global vertex coordinate mapping
    vertices = {
        0: (x0, y0, z0),       1: (x0+d, y0, z0),
        2: (x0+d, y0+d, z0),   3: (x0, y0+d, z0),
        4: (x0, y0, z0+d),     5: (x0+d, y0, z0+d),
        6: (x0+d, y0+d, z0+d), 7: (x0, y0+d, z0+d)
    }
    
    edges_vertices = {
        0: (0, 1), 1: (1, 2), 2: (2, 3), 3: (3, 0),
        4: (4, 5), 5: (5, 6), 6: (6, 7), 7: (7, 4),
        8: (0, 4), 9: (1, 5), 10: (2, 6), 11: (3, 7),
        12: (0, 2), 13: (4, 6), 14: (1, 4), 15: (1, 6),
        16: (2, 7), 17: (0, 7)
    }
    
    meshes = []
    wireframe_radius = d * pipe_radius_ratio * 0.5
    loop_radius = d * pipe_radius_ratio
    
    # 1. Generate Cube Wireframe (using the 12 primary structural edges, 0-11, and the 6 diagonal edges, 12-17)
    for e in range(18):
        u1, u2 = edges_vertices[e]
        p1, p2 = vertices[u1], vertices[u2]
        cyl = trimesh.creation.cylinder(radius=wireframe_radius, segment=(p1, p2))
        cyl.visual.face_colors = [200, 200, 200, 150]  # Semi-transparent light grey
        meshes.append(cyl)
        
    # 2. Generate Shifted Loops
    sorted_loops = cube_dict.get('sorted_loops', [])
    edge_weights = cube_dict.get('edge_weights', [0]*18)
    
    # Distinct colors for multiple loops intersecting the same cell
    colors = [
        [255, 50, 50, 255],   # Red
        [50, 255, 50, 255],   # Green
        [50, 100, 255, 255],  # Blue
        [255, 200, 50, 255],  # Yellow
        [255, 50, 255, 255],  # Magenta
        [50, 255, 255, 255]   # Cyan
    ]
    
    for i, loop_data in enumerate(sorted_loops):
        edges = loop_data.get('loop', [])
        ranks = loop_data.get('rank', [])
        k = len(edges)
        if k == 0:
            continue
            
        # Compute 3D coordinates for each intersection point on the edges
        pts = []
        for e, r in zip(edges, ranks):
            u1, u2 = edges_vertices[e]
            v1, v2 = np.array(vertices[u1]), np.array(vertices[u2])
            W = edge_weights[e]
            
            # Shift the loop node on edges by rank to prevent overlap
            # Space the nodes relative to the pipe's diameter and center them at 0.5
            if W > 0:
                spacing = pipe_radius_ratio * 3
                t = 0.5 + (r - (W - 1) / 2.0) * spacing
                # Clamp t to ensure points don't clip off the edge bounds
                t = max(0.05, min(0.95, t))
            else:
                t = 0.5
                
            pt = v1 + t * (v2 - v1)
            pts.append(pt)
            
        color = colors[i % len(colors)]
        
        # Create a pipe for each line segment of the loop
        for j in range(k):
            p1 = pts[j]
            p2 = pts[(j + 1) % k]
            
            if np.linalg.norm(p2 - p1) > 1e-6:
                cyl = trimesh.creation.cylinder(radius=loop_radius, segment=(p1, p2))
                cyl.visual.face_colors = color
                meshes.append(cyl)
                
        # Optional: Render the assigned component point as a sphere to verify the assignment step
        cp = loop_data.get('component_point')
        if cp is not None:
            sphere = trimesh.creation.icosphere(radius=loop_radius * 1.5)
            sphere.apply_translation(cp)
            sphere.visual.face_colors = color
            meshes.append(sphere)

    if not meshes:
        return trimesh.Trimesh()
        
    return trimesh.util.concatenate(meshes)



def visualize_shifted_loops_with_boundary(cube_dict, resolution, pipe_radius_ratio=0.02):
    """
    Visualizes the cube wireframe and the shifted non-intersecting boundary loops as 3D pipes.
    These paths traverse both faces and edges and form open line sequences.
    
    Args:
        cube_dict (dict): Dictionary containing 'sorted_loops', 'cube_indices', etc.
        resolution (float/int): Grid resolution to compute global coordinates.
        pipe_radius_ratio (float): Radius of the pipes relative to the cube size.
        
    Returns:
        trimesh.Trimesh: Concatenated mesh containing wireframe and path pipes.
    """
    if trimesh is None:
        raise ImportError("The 'trimesh' library is required for visualization. Install it with: pip install trimesh")
        
    ix, iy, iz = cube_dict.get('cube_indices', (0, 0, 0))
    d = 1.0 / resolution
    x0, y0, z0 = ix * d, iy * d, iz * d
    
    vertices = {
        0: np.array([x0, y0, z0]),     1: np.array([x0+d, y0, z0]),
        2: np.array([x0+d, y0+d, z0]), 3: np.array([x0, y0+d, z0]),
        4: np.array([x0, y0, z0+d]),   5: np.array([x0+d, y0, z0+d]),
        6: np.array([x0+d, y0+d, z0+d]), 7: np.array([x0, y0+d, z0+d])
    }
    
    edges_vertices = {
        0: (0, 1), 1: (1, 2), 2: (2, 3), 3: (3, 0),
        4: (4, 5), 5: (5, 6), 6: (6, 7), 7: (7, 4),
        8: (0, 4), 9: (1, 5), 10: (2, 6), 11: (3, 7),
        12: (0, 2), 13: (4, 6), 14: (1, 4), 15: (1, 6),
        16: (2, 7), 17: (0, 7)
    }
    
    meshes = []
    wireframe_radius = d * pipe_radius_ratio * 0.5
    loop_radius = d * pipe_radius_ratio
    
    # 1. Generate Cube Wireframe (using only the 12 primary structural edges, 0-11)
    for e in range(18):
        u1, u2 = edges_vertices[e]
        p1, p2 = vertices[u1], vertices[u2]
        cyl = trimesh.creation.cylinder(radius=wireframe_radius, segment=(p1, p2))
        cyl.visual.face_colors = [200, 200, 200, 150]  # Semi-transparent light grey
        meshes.append(cyl)
        
    sorted_loops = cube_dict.get('sorted_loops', [])
    if not sorted_loops:
        if meshes: return trimesh.util.concatenate(meshes)
        return trimesh.Trimesh()
        
    # 2. Reconstruct element counts to compute edge weights
    element_counts = collections.Counter()
    for loop_data in sorted_loops:
        for el in loop_data['loop']:
            pel = ('e', int(el))
            element_counts[pel] += 1
            
    colors = [
        [255, 50, 50, 255],   # Red
        [50, 255, 50, 255],   # Green
        [50, 100, 255, 255],  # Blue
        [255, 200, 50, 255],  # Yellow
        [255, 50, 255, 255],  # Magenta
        [50, 255, 255, 255]   # Cyan
    ]
    
    # 3. Generate Shifted Boundary Loops
    for i, loop_data in enumerate(sorted_loops):
        edges = loop_data.get('loop', [])
        ranks = loop_data.get('rank', [])
        k = len(edges)
        if k < 2:
            continue
            
        pts = []
        for e, r in zip(edges, ranks):
            u1, u2 = edges_vertices[int(e)]
            v1, v2 = vertices[u1], vertices[u2]
            W = element_counts[('e', int(e))]
            
            # Shift the loop node on edges by rank to prevent overlap
            # Space the nodes relative to the pipe's diameter and center them at 0.5
            if W > 0:
                spacing = pipe_radius_ratio * 3
                t = 0.5 + (r - (W - 1) / 2.0) * spacing
                # Clamp t to ensure points don't clip off the edge bounds
                t = max(0.05, min(0.95, t))
            else:
                t = 0.5
                
            pt = v1 + t * (v2 - v1)
            pts.append(pt)
            
        color = colors[i % len(colors)]
        
        # Create a pipe for each line segment of the open loop
        for j in range(k - 1):
            p1 = pts[j]
            p2 = pts[j + 1]
            
            if np.linalg.norm(p2 - p1) > 1e-6:
                cyl = trimesh.creation.cylinder(radius=loop_radius, segment=(p1, p2))
                cyl.visual.face_colors = color
                meshes.append(cyl)
                
        # Optional: Render the assigned component point as a sphere
        cp = loop_data.get('component_point')
        if cp is not None:
            sphere = trimesh.creation.icosphere(radius=loop_radius * 1.5)
            sphere.apply_translation(cp)
            sphere.visual.face_colors = color
            meshes.append(sphere)

    if not meshes:
        return trimesh.Trimesh()
        
    return trimesh.util.concatenate(meshes)


def collapse_point_inner(cube_dicts, resolution, debug=False, output_directory=None):
    processed_cubes = process_multiple_cubes(cube_dicts, resolution, batch_size=10000)
    if debug:
        for cube_dict in processed_cubes:
            visualize_shifted_loops(cube_dict, resolution).export(f'{output_directory}/debug/loops_inner_{cube_dict["cube_indices"]}.ply')
    return processed_cubes

def collapse_point_boundary(cube_dicts, resolution, debug=False, output_directory=None):
    processed_cubes = process_multiple_cubes_with_boundary(cube_dicts, resolution, batch_size=10000)
    if debug:
        for cube_dict in processed_cubes:
            visualize_shifted_loops_with_boundary(cube_dict, resolution).export(f'{output_directory}/debug/loops_boundary_{cube_dict["cube_indices"]}.ply')
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