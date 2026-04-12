from typing import List, Dict, Tuple, Set
import multiprocessing
from tqdm import tqdm

# Hardcoded topology based on the specific triangulated cube description
# Map of edge indices to their corresponding (vertex_a, vertex_b)
EDGE_VERTS = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # 0-3: Bottom edges
    (4, 5), (5, 6), (6, 7), (7, 4),  # 4-7: Top edges
    (0, 4), (1, 5), (2, 6), (3, 7),  # 8-11: Vertical edges
    (0, 2),                          # 12: Bottom diagonal
    (4, 6),                          # 13: Top diagonal
    # (0, 5),                          # 14: Front diagonal
    (1, 4),                          # 14: Front diagonal
    (1, 6),                          # 15: Right diagonal
    (2, 7),                          # 16: Back diagonal
    (0, 7)                           # 17: Left diagonal
]

# Triangles defined as a tuple of 3 edge indices
TRIANGLES = [
    (0, 1, 12),    # T0: Bottom face half 1
    (2, 3, 12),    # T1: Bottom face half 2
    (4, 5, 13),    # T2: Top face half 1
    (6, 7, 13),    # T3: Top face half 2
    # (0, 9, 14),    # T4: Front face half 1
    # (4, 8, 14),    # T5: Front face half 2
    (0, 8, 14),    # T4: Front face half 1
    (4, 9, 14),    # T5: Front face half 2
    (1, 10, 15),   # T6: Right face half 1
    (5, 9, 15),    # T7: Right face half 2
    (2, 11, 16),   # T8: Back face half 1
    (6, 10, 16),   # T9: Back face half 2
    (3, 11, 17),   # T10: Left face half 1
    (7, 8, 17)     # T11: Left face half 2
]

def _get_common_vertex(e1: int, e2: int) -> int:
    """Finds the shared vertex between two edges."""
    v1, v2 = EDGE_VERTS[e1]
    u1, u2 = EDGE_VERTS[e2]
    if v1 in (u1, u2): return v1
    if v2 in (u1, u2): return v2
    raise ValueError(f"Edges {e1} and {e2} do not share a vertex.")

def _get_ordered_points(e_idx: int, corner_v: int, k: int, weight: int) -> List[int]:
    """
    Returns the indices of the `k` points closest to `corner_v` on edge `e_idx`.
    To ensure no crossing, points are ordered by their distance from the corner.
    Point 0 is closest to the first vertex of the edge, Point (weight-1) is closest to the second.
    """
    u, v = EDGE_VERTS[e_idx]
    if corner_v == u:
        # Closest points start at 0 and move up
        return list(range(k))
    elif corner_v == v:
        # Closest points start at weight - 1 and move down
        return [weight - 1 - i for i in range(k)]
    else:
        raise ValueError(f"Vertex {corner_v} is not an endpoint of edge {e_idx}")

def reconstruct_loops(edge_weights: List[int]) -> List[List[Dict[str, int]]]:
    """
    Given a list of 18 edge weights, reconstructs the distinct, disjoint closed loops 
    on the surface of the triangulated cube using normal curve theory.
    """
    # 1. Validation
    if len(edge_weights) != 18:
        raise ValueError(f"Expected exactly 18 edge weights, got {len(edge_weights)}.")

    for t_idx, (e1, e2, e3) in enumerate(TRIANGLES):
        w1, w2, w3 = edge_weights[e1], edge_weights[e2], edge_weights[e3]
        if w1 + w2 < w3 or w2 + w3 < w1 or w3 + w1 < w2:
            raise ValueError(f"Triangle inequality violated in face T{t_idx} (edges {e1}, {e2}, {e3}).")
        if (w1 + w2 + w3) % 2 != 0:
            raise ValueError(f"Sum of edge weights in face T{t_idx} is not even.")

    # 2. Setup Adjacency Graph for the Points
    # Graph maps (edge_idx, point_idx) -> list of (triangle_idx, target_edge_idx, target_point_idx)
    adj: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}
    
    # Initialize adjacency map for every point on every edge
    for e in range(18):
        for p in range(edge_weights[e]):
            adj[(e, p)] = []

    # 3. Calculate Internal Arcs and Build Connections
    for t_idx, (e1, e2, e3) in enumerate(TRIANGLES):
        w1, w2, w3 = edge_weights[e1], edge_weights[e2], edge_weights[e3]
        
        # Calculate how many arcs isolate each corner of the triangle
        k12 = (w1 + w2 - w3) // 2  # Arcs connecting e1 and e2
        k23 = (w2 + w3 - w1) // 2  # Arcs connecting e2 and e3
        k31 = (w3 + w1 - w2) // 2  # Arcs connecting e3 and e1

        # Process each pair of edges in the triangle
        edge_pairs = [(e1, e2, k12), (e2, e3, k23), (e3, e1, k31)]
        
        for edge_A, edge_B, arcs_count in edge_pairs:
            if arcs_count == 0:
                continue
                
            common_v = _get_common_vertex(edge_A, edge_B)
            
            # Get the points ordered by distance from the shared corner to prevent self-intersections
            pts_A = _get_ordered_points(edge_A, common_v, arcs_count, edge_weights[edge_A])
            pts_B = _get_ordered_points(edge_B, common_v, arcs_count, edge_weights[edge_B])
            
            # Connect the isolated points pairwise across the triangle
            for i in range(arcs_count):
                p_A = pts_A[i]
                p_B = pts_B[i]
                adj[(edge_A, p_A)].append((t_idx, edge_B, p_B))
                adj[(edge_B, p_B)].append((t_idx, edge_A, p_A))

    # 4. Trace the Loops
    loops = []
    visited_nodes: Set[Tuple[int, int]] = set()

    for start_node in adj:
        if start_node in visited_nodes:
            continue
            
        current_loop = []
        curr_node = start_node
        prev_node = None
        
        # Traverse until we close the cycle
        while True:
            visited_nodes.add(curr_node)
            neighbors = adj[curr_node]
            
            # A valid normal curve manifold ensures exactly 2 paths out of any internal point
            assert len(neighbors) == 2, "Internal point does not have exactly degree 2!"
            
            n1_tri, n1_edge, n1_p = neighbors[0]
            n2_tri, n2_edge, n2_p = neighbors[1]
            
            node1 = (n1_edge, n1_p)
            node2 = (n2_edge, n2_p)

            # Choose the neighbor we didn't just come from
            if prev_node is None:
                chosen_tri, next_e, next_p = neighbors[0]
                next_node = node1
            else:
                if node1 == prev_node:
                    chosen_tri, next_e, next_p = neighbors[1]
                    next_node = node2
                else:
                    chosen_tri, next_e, next_p = neighbors[0]
                    next_node = node1

            # Record the trace step
            # current_loop.append({
            #     "face": chosen_tri,
            #     "edge_in": curr_node[0],
            #     "edge_out": next_e
            # })
            current_loop.append(curr_node[0])
            
            prev_node = curr_node
            curr_node = next_node
            
            if curr_node == start_node:
                break
                
        loops.append(current_loop)

    return loops


def _process_single_cube(cube_data: Dict) -> Dict:
    """Helper function for multiprocessing to process a single cube."""
    try:
        loops = reconstruct_loops(cube_data['edge_weights'])
        cube_data['loops'] = loops
        cube_data['num_loops'] = len(loops)
        cube_data['error'] = None
    except Exception as e:
        # Gracefully handle any invalid topology errors to prevent crashing the whole batch
        cube_data['loops'] = []
        cube_data['num_loops'] = 0
        cube_data['error'] = str(e)
    return cube_data


def reconstruct_loops_multiple_cubes(cubes_data: List[Dict], batch_size: int = 1000) -> List[Dict]:
    """
    Processes multiple cube configurations in parallel.
    Input: List of dicts, where each dict has an 'edge_weights' key.
    Returns: The updated list of dicts with a new 'loops' key.
    """
    results = []
    num_cores = multiprocessing.cpu_count()
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        with tqdm(total=len(cubes_data), desc="Processing cubes") as pbar:
            # Process in explicitly sized batches to limit IPC memory overhead
            for i in range(0, len(cubes_data), batch_size):
                batch = cubes_data[i:i + batch_size]
                
                # imap yields results efficiently as they are ready
                for processed_cube in pool.imap(_process_single_cube, batch):
                    results.append(processed_cube)
                    pbar.update(1)
                    
    return results


if __name__ == "__main__":
    # Test Example: A loop tightly circling Vertex 0. 
    # It cuts across edges 0, 3, 8, 12, 14, 17 once each. All other edges are 0.
    test_weights = [0] * 18
    for e in [0, 3, 8, 12, 14, 17]:
        test_weights[e] = 1
        
    extracted_loops = reconstruct_loops(test_weights)
    print(f"Found {len(extracted_loops)} loop(s).")
    for step in extracted_loops[0]:
        print(f"Face {step['face']:2} | Entered via edge {step['edge_in']:2} -> Exited via edge {step['edge_out']:2}")

    # Test Multiprocessing Example
    print("\nTesting multiprocessing on multiple cubes...")
    multi_cubes = [{'edge_weights': test_weights.copy()} for _ in range(500)]
    processed_cubes = reconstruct_loops_multiple_cubes(multi_cubes, batch_size=100)
    print(f"Processed {len(processed_cubes)} cubes successfully.")