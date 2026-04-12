from typing import List, Dict, Tuple, Set
import multiprocessing
import itertools
import math
from tqdm import tqdm
import collections
import numpy as np
import trimesh
from collections import defaultdict, Counter

# Hardcoded topology based on the specific triangulated cube description
# Map of edge indices to their corresponding (vertex_a, vertex_b)
EDGE_VERTS = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # 0-3: Bottom edges
    (4, 5), (5, 6), (6, 7), (7, 4),  # 4-7: Top edges
    (0, 4), (1, 5), (2, 6), (3, 7),  # 8-11: Vertical edges
    (0, 2),                          # 12: Bottom diagonal
    (4, 6),                          # 13: Top diagonal
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


# =========================================================================
# NEW: Extractor for ambiguous U-turns mapping without open boundaries
# =========================================================================

def get_canonical_loop(loop: List[int]) -> Tuple[int, ...]:
    """
    Deduplicates cyclic loops so [e1, f1, e2, f2] matches [e2, f2, e1, f1] 
    and its reversed trace [e1, f2, e2, f1].
    """
    n = len(loop) // 2
    best = tuple(loop)
    
    # Check forward shifts
    for i in range(n):
        shifted = tuple(loop[2*i:] + loop[:2*i])
        if shifted < best:
            best = shifted
            
    # Compute reversed loop taking faces sequence into account 
    # (e0->f0->e1 becomes e0<-f0<-e1 in reverse, so f0 leads to e0)
    if len(loop) > 0:
        loop_rev = [loop[0]] + loop[::-1][:-1]
        # Check backward shifts
        for i in range(n):
            shifted = tuple(loop_rev[2*i:] + loop_rev[:2*i])
            if shifted < best:
                best = shifted
                
    return best

def get_canonical_solution(loops: List[List[int]]) -> Tuple[Tuple[int, ...], ...]:
    """Sorts all loops in a solution to formulate a unique dictionary key."""
    return tuple(sorted([get_canonical_loop(l) for l in loops]))

def _trace_loops_for_uturn_assignment(ew: List[int], assignment: Tuple[Tuple[int, int, int], ...]) -> List[List[int]]:
    """Builds and traces the exact topology graph for a specific u-turn mapping."""
    adj: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = { (e, p): [] for e in range(18) for p in range(ew[e]) }
    
    for t_idx, (e1, e2, e3) in enumerate(TRIANGLES):
        u1, u2, u3 = assignment[t_idx]
        
        w1_prime = ew[e1] - 2 * u1
        w2_prime = ew[e2] - 2 * u2
        w3_prime = ew[e3] - 2 * u3
        
        k12 = (w1_prime + w2_prime - w3_prime) // 2
        k23 = (w2_prime + w3_prime - w1_prime) // 2
        k31 = (w3_prime + w1_prime - w2_prime) // 2
        
        def get_k_for_vertex(eA, target_v):
            if target_v == _get_common_vertex(e1, e2) and eA in (e1, e2): return k12
            if target_v == _get_common_vertex(e2, e3) and eA in (e2, e3): return k23
            if target_v == _get_common_vertex(e3, e1) and eA in (e3, e1): return k31
            return 0
            
        def assign_face_connections(eA, eB, k, v_common):
            if k == 0: return
            pts_A = _get_ordered_points(eA, v_common, k, ew[eA])
            pts_B = _get_ordered_points(eB, v_common, k, ew[eB])
            for i in range(k):
                adj[(eA, pts_A[i])].append((t_idx, eB, pts_B[i]))
                adj[(eB, pts_B[i])].append((t_idx, eA, pts_A[i]))
                
        # Assign standard corner-crossing arcs
        assign_face_connections(e1, e2, k12, _get_common_vertex(e1, e2))
        assign_face_connections(e2, e3, k23, _get_common_vertex(e2, e3))
        assign_face_connections(e3, e1, k31, _get_common_vertex(e3, e1))
        
        def assign_uturns(eA, u):
            if u == 0: return
            v0, v1 = EDGE_VERTS[eA]
            k_v0 = get_k_for_vertex(eA, v0)
            k_v1 = get_k_for_vertex(eA, v1)
            
            start_idx = k_v0
            end_idx = ew[eA] - k_v1
            
            if end_idx - start_idx != 2 * u:
                raise ValueError(f"U-turn bounds mismatch on edge {eA}")
                
            for i in range(u):
                p1 = start_idx + 2 * i
                p2 = start_idx + 2 * i + 1
                adj[(eA, p1)].append((t_idx, eA, p2))
                adj[(eA, p2)].append((t_idx, eA, p1))
                
        # Assign embedded U-turn arcs
        assign_uturns(e1, u1)
        assign_uturns(e2, u2)
        assign_uturns(e3, u3)
        
    loops = []
    visited = set()
    for start_node in adj:
        if start_node in visited:
            continue
            
        curr_node = start_node
        prev_node = None
        current_loop = []
        
        while True:
            visited.add(curr_node)
            neighbors = adj[curr_node]
            
            if len(neighbors) != 2:
                raise ValueError(f"Degree is not exactly 2 at node {curr_node}")
                
            # Filter the backwards direction
            if prev_node is None:
                f_idx, next_e, next_p = neighbors[0]
            else:
                n0_f, n0_e, n0_p = neighbors[0]
                if (n0_e, n0_p) == prev_node:
                    f_idx, next_e, next_p = neighbors[1]
                else:
                    f_idx, next_e, next_p = neighbors[0]
                    
            # Record edge and the face crossed
            current_loop.append(curr_node[0])
            current_loop.append(f_idx)
            
            prev_node = curr_node
            curr_node = (next_e, next_p)
            
            if curr_node == start_node:
                break
                
        loops.append(current_loop)
        
    return loops

def extract_loops_with_uturns(cubes_data: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Extracts topological loops attributing face_weights as inner-facet U-turns.
    
    Returns:
        (solved_uturn, ambiguous_uturn, unsolvable_uturn)
    """
    solved_uturn = []
    ambiguous_uturn = []
    unsolvable_uturn = []
    
    for cube in tqdm(cubes_data, desc="Extracting U-Turn Configurations"):
        ew = cube.get('edge_weights', [0] * 18)
        fw = cube.get('face_weights', [0] * 12)
        
        if sum(ew) == 0 and sum(fw) == 0:
            c = cube.copy()
            c['loops'] = []
            solved_uturn.append(c)
            continue
            
        face_valid_assignments = []
        is_possible = True
        
        # 1. Determine local valid U-turn mappings independently for all 12 faces
        for t_idx in range(12):
            valid_for_face = []
            W = fw[t_idx]
            e1, e2, e3 = TRIANGLES[t_idx]
            
            # Combinatorics for u1 + u2 + u3 = W (Face Weight)
            for u1 in range(W + 1):
                for u2 in range(W + 1 - u1):
                    u3 = W - u1 - u2
                    
                    # Remaining weights after subtracting 2 crossings per assigned U-Turn
                    w1 = ew[e1] - 2 * u1
                    w2 = ew[e2] - 2 * u2
                    w3 = ew[e3] - 2 * u3
                    
                    if w1 < 0 or w2 < 0 or w3 < 0:
                        continue
                    if w1 + w2 < w3 or w2 + w3 < w1 or w3 + w1 < w2:
                        continue
                    if (w1 + w2 + w3) % 2 != 0:
                        continue
                        
                    valid_for_face.append((u1, u2, u3))
                    
            if not valid_for_face:
                is_possible = False
                break
                
            face_valid_assignments.append(valid_for_face)
            
        if not is_possible:
            unsolvable_uturn.append(cube)
            continue
            
        # Optional safeguard against combinatorial explosion
        total_combinations = math.prod(len(v) for v in face_valid_assignments)
        if total_combinations > 100000:
            c = cube.copy()
            c['loops'] = []
            c['error'] = 'Too many ambiguous u-turn combinations'
            unsolvable_uturn.append(c)
            continue

        # 2. Evaluate global Cartesian product of the valid local mappings
        unique_solutions = {}
        
        for assignment in itertools.product(*face_valid_assignments):
            try:
                loops = _trace_loops_for_uturn_assignment(ew, assignment)
                canonical_sol = get_canonical_solution(loops)
                
                # Keep unique configurations
                if canonical_sol not in unique_solutions:
                    unique_solutions[canonical_sol] = loops
                    
            except Exception as e:
                # Discard topologies violating global tracing consistency
                print(e)
                continue
        
        # prune out backward u turn, i.e. 3 same face indices
        unique_solutions = list(unique_solutions.values())
        unique_solutions = [[loop[::2] for loop in sol] for sol in unique_solutions]
        uniq_sol = []
        for sol in unique_solutions:
            is_valid = True
            for loop in sol:
                if any(count >= 3 for count in Counter(loop).values()):
                    is_valid = False
                    break
            if is_valid:
                uniq_sol.append(sol)
        unique_solutions = uniq_sol

        # 3. Stratify the outcomes based on degree of ambiguity
        num_sols = len(unique_solutions)
        if num_sols == 0:
            unsolvable_uturn.append(cube)
        elif num_sols == 1:
            c = cube.copy()
            c['loops'] = unique_solutions[0]
            # c['loops'] = [loop[::2] for loop in c['loops']]
            c['num_loops'] = len(c['loops'])
            solved_uturn.append(c)
        else:
            c = cube.copy()
            # Store the matrix of solutions when mapping is ambiguous
            c['loops'] = unique_solutions
            # c['loops'] = [loop[::2] for loop in c['loops']]
            ambiguous_uturn.append(c)
            
    return solved_uturn, ambiguous_uturn, unsolvable_uturn



# =========================================================================
# NEW FUNCTIONALITY: Loop Extraction with Boundaries and U-Turns
# =========================================================================


def _generate_arcs_for_face(F_idx: int, config: Dict, edge_weights: List[int]) -> List[Tuple]:
    """Generates a non-intersecting list of arcs for a specific triangle configuration."""
    arcs = []
    eA, eB, eC = TRIANGLES[F_idx]
    
    corners = [ (eA, eB), (eB, eC), (eC, eA) ]
    
    # 1. K-Arcs (Normal curve connections)
    for E1, E2 in corners:
        v_shared = _get_common_vertex(E1, E2)
        k = config['k'][(E1, E2)]
        for i in range(k):
            # Port allocation follows distance from the shared vertex
            p1 = i if EDGE_VERTS[E1][0] == v_shared else edge_weights[E1] - 1 - i
            p2 = i if EDGE_VERTS[E2][0] == v_shared else edge_weights[E2] - 1 - i
            arcs.append( (("port", E1, p1), ("port", E2, p2), F_idx) )
            
    # 2. U-Turns and Open Boundaries
    bound_idx = 0
    for E in [eA, eB, eC]:
        u, v = EDGE_VERTS[E]
        # Identify the edge sharing the FIRST endpoint of E (u)
        E_u = eA if (u in EDGE_VERTS[eA] and eA != E) else (eB if (u in EDGE_VERTS[eB] and eB != E) else eC)
        k_u = config['k'][(E, E_u)]
        
        u_E = config['u'][E]
        b_E = config['b'][E]
        
        # U-turns occupy the pairs immediately following the k-arcs
        for i in range(u_E):
            p1 = k_u + 2*i
            p2 = k_u + 2*i + 1
            arcs.append( (("port", E, p1), ("port", E, p2), F_idx) )
            
        # Boundaries occupy the space following u-turns
        for i in range(b_E):
            p = k_u + 2*u_E + i
            arcs.append( (("port", E, p), ("bound", F_idx, bound_idx), F_idx) )
            bound_idx += 1
            
    return arcs


def _extract_loops_from_adj(adj: Dict) -> List[List[int]]:
    """Traces paths (boundaries) and cycles given the graph adjacency list."""
    visited_nodes = set()
    loops = []
    
    # Trace open paths starting from boundaries
    bound_nodes = sorted([n for n in adj if n[0] == "bound"])
    for start_node in bound_nodes:
        if start_node in visited_nodes: continue
        
        seq = []
        curr_node = start_node
        visited_nodes.add(curr_node)
        
        next_node, F_next = adj[curr_node][0]
        prev_node_temp = curr_node
        F_arrived = F_next
        
        while True:
            seq.append(F_next) # Pass through Face
            
            prev_node_temp = curr_node
            F_arrived = F_next
            curr_node = next_node
            visited_nodes.add(curr_node)
            
            if curr_node[0] == "bound":
                break # Terminated at another open boundary
                
            seq.append(curr_node[1]) # Pass through Edge
            
            neighbors = adj[curr_node]
            if neighbors[0] == (prev_node_temp, F_arrived):
                next_node, F_next = neighbors[1]
            else:
                next_node, F_next = neighbors[0]
                
        loops.append(seq)
        
    # Trace closed cycles from remaining unvisited ports
    port_nodes = sorted([n for n in adj if n[0] == "port"])
    for start_node in port_nodes:
        if start_node in visited_nodes: continue
        
        seq = []
        curr_node = start_node
        visited_nodes.add(curr_node)
        
        prev_node_temp, F_arrived = adj[curr_node][1]
        next_node, F_next = adj[curr_node][0]
        
        while True:
            seq.append(F_next) # Pass through Face
            
            prev_node_temp = curr_node
            F_arrived = F_next
            curr_node = next_node
            visited_nodes.add(curr_node)
            
            if curr_node == start_node:
                break # Reached the start, closing the cycle
                
            seq.append(curr_node[1]) # Pass through Edge
            
            neighbors = adj[curr_node]
            if neighbors[0] == (prev_node_temp, F_arrived):
                next_node, F_next = neighbors[1]
            else:
                next_node, F_next = neighbors[0]
                
        loops.append(seq)
        
    return loops


def extract_loops_with_boundaries(cube_dicts: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Extracts loops allowing open boundaries and u-turns.
    Returns: (solved_uturn, ambiguous_uturn, unsolvable_uturn)
    """
    solved_uturn = []
    ambiguous_uturn = []
    unsolvable_uturn = []
    
    for cube in cube_dicts:
        edge_weights = cube['edge_weights']
        face_weights = cube['face_weights']
        
        face_best_configs = []
        is_solvable = True
        
        # 1. Resolve local topologies for each of the 12 faces
        for f_idx in range(12):
            B_target = face_weights[f_idx]
            eA, eB, eC = TRIANGLES[f_idx]
            wA, wB, wC = edge_weights[eA], edge_weights[eB], edge_weights[eC]
            
            valid_configs = []
            
            # Brute force all possible k-arcs, validating remainder parity for u-turns
            for kAB in range(min(wA, wB) + 1):
                for kBC in range(min(wB, wC) + 1):
                    for kCA in range(min(wC, wA) + 1):
                        remA = wA - kAB - kCA
                        remB = wB - kAB - kBC
                        remC = wC - kBC - kCA
                        if remA < 0 or remB < 0 or remC < 0: continue
                        
                        for uA in range(remA // 2 + 1):
                            bA = remA - 2 * uA
                            for uB in range(remB // 2 + 1):
                                bB = remB - 2 * uB
                                for uC in range(remC // 2 + 1):
                                    bC = remC - 2 * uC
                                    
                                    B_total = bA + bB + bC
                                    U_total = uA + uB + uC
                                    
                                    # Hard constraint: 1 face weight exactly equals 1 U-turn OR 1 Open Boundary
                                    # If face weight is 0, this strictly enforces U_total=0 and B_total=0
                                    if B_total + U_total == B_target:
                                        valid_configs.append({
                                            'k': { (eA, eB): kAB, (eB, eA): kAB,
                                                   (eB, eC): kBC, (eC, eB): kBC,
                                                   (eC, eA): kCA, (eA, eC): kCA },
                                            'u': { eA: uA, eB: uB, eC: uC },
                                            'b': { eA: bA, eB: bB, eC: bC },
                                            'B_total': B_total,
                                            'U_total': U_total
                                        })
            
            if not valid_configs:
                is_solvable = False
                break
                
            # Score configurations: prefer open boundaries over u-turns when ambiguous
            def score_fn(c):
                return -c['B_total'] # Lower score is better (maximizes open boundaries)
                
            valid_configs.sort(key=score_fn)
            best_score = score_fn(valid_configs[0])
            best_configs = [c for c in valid_configs if score_fn(c) == best_score]
            face_best_configs.append(best_configs)
            
        if not is_solvable:
            unsolvable_uturn.append(cube)
            continue
            
        # 2. Ascertain ambiguity and assemble macro-configurations
        num_global = 1
        for bc in face_best_configs:
            num_global *= len(bc)
            
        # Capping combinatorial explosion limit
        all_global_configs = list(itertools.islice(itertools.product(*face_best_configs), 1000))
        
        all_loops_solutions = []
        for global_config in all_global_configs:
            # Enforce global topological rule: n open paths means exactly 2n open boundaries (must be even)
            if sum(c['B_total'] for c in global_config) % 2 != 0:
                continue

            arcs = []
            for f_idx in range(12):
                arcs.extend(_generate_arcs_for_face(f_idx, global_config[f_idx], edge_weights))
                
            adj = collections.defaultdict(list)
            for nodeA, nodeB, F in arcs:
                adj[nodeA].append( (nodeB, F) )
                adj[nodeB].append( (nodeA, F) )
                
            loops = _extract_loops_from_adj(adj)
            all_loops_solutions.append(loops)
            
        # 3. Categorize Output
        if not all_loops_solutions:
            unsolvable_uturn.append(cube)
        elif len(all_loops_solutions) == 1:
            cube['loops'] = all_loops_solutions[0]
            # cube['loops'] = [loop[1::2] for loop in cube['loops']]
            cube['num_loops'] = len(cube['loops'])
            solved_uturn.append(cube)
        else:
            cube['loops_solutions'] = all_loops_solutions
            # cube['loops_solutions'] = [[loop[1::2] for loop in loop_solution] for loop_solution in cube['loops_solutions']]
            ambiguous_uturn.append(cube)
            
    return solved_uturn, ambiguous_uturn, unsolvable_uturn
    


def visualize_loops(cube_dict: Dict, res: int) -> trimesh.Trimesh:
    """
    Creates a 3D mesh visualization of the extracted loops and the cube's wireframe.
    Nodes are shifted along edges/faces to prevent overlaps.
    
    Args:
        cube_dict: Dictionary containing 'cube_indices' and 'loops' / 'loops_solutions'.
        res: Grid resolution used to appropriately scale the bounds.
        
    Returns:
        trimesh.Trimesh: The concatenated 3D mesh of the visualization.
    """
    
    # Extract the resolved loops (fallback to the first solution if ambiguous)
    loops = cube_dict.get('loops', [])
    if not loops and 'loops_solutions' in cube_dict and len(cube_dict['loops_solutions']) > 0:
        loops = cube_dict['loops_solutions'][0]
        
    cx, cy, cz = cube_dict.get('cube_indices', (0, 0, 0))
    
    # 1. Base Geometry Generation (Scaled to local space)
    unit_V = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=float)
    
    V = (unit_V + np.array([cx, cy, cz])) / float(res)
    
    # 2. Pre-count visits to calculate shifts preventing overlap
    edge_counts = defaultdict(int)
    face_counts = defaultdict(int)
    
    for loop in loops:
        is_open = (len(loop) % 2 != 0)
        n_len = len(loop)
        for i, item in enumerate(loop):
            if i % 2 == 1:
                edge_counts[item] += 1   # Odd indices are edges
            else:
                # Even indices are faces. Determine if we keep this face node.
                keep_face = False
                if is_open and (i == 0 or i == n_len - 1):
                    keep_face = True # Terminal nodes of open boundary
                else:
                    prev_edge = loop[(i - 1) % n_len]
                    next_edge = loop[(i + 1) % n_len]
                    if prev_edge == next_edge:
                        keep_face = True # U-Turn
                
                if keep_face:
                    face_counts[item] += 1
                
    # 3. Generate distributed coordinates for overlapping edges
    edge_pts = defaultdict(list)
    for E, count in edge_counts.items():
        u, v = EDGE_VERTS[E]
        p_u, p_v = V[u], V[v]
        for i in range(count):
            # Linearly distribute points along the edge
            t = (i + 1) / (count + 1)
            edge_pts[E].append(p_u + t * (p_v - p_u))
            
    # Generate distributed coordinates for overlapping faces
    face_pts = defaultdict(list)
    for F, count in face_counts.items():
        t_idx = TRIANGLES[F]
        p0, p1, p2 = V[EDGE_VERTS[t_idx[0]][0]], V[EDGE_VERTS[t_idx[0]][1]], V[EDGE_VERTS[t_idx[1]][1]]
        # Calculate proper centroid from the 3 vertices
        v_idx = set(EDGE_VERTS[t_idx[0]] + EDGE_VERTS[t_idx[1]] + EDGE_VERTS[t_idx[2]])
        v_coords = np.array([V[idx] for idx in v_idx])
        C = np.mean(v_coords, axis=0)
        
        if count == 1:
            face_pts[F].append(C)
        else:
            # Create a small circular distribution around the face centroid
            n = np.cross(v_coords[1] - v_coords[0], v_coords[2] - v_coords[0])
            n_norm = np.linalg.norm(n)
            n = n / n_norm if n_norm > 1e-8 else np.array([0, 0, 1])
            
            u_dir = v_coords[1] - v_coords[0]
            u_dir = u_dir / np.linalg.norm(u_dir)
            v_dir = np.cross(n, u_dir)
            
            radius = 0.15 / res
            for i in range(count):
                angle = 2 * np.pi * i / count
                pt = C + radius * np.cos(angle) * u_dir + radius * np.sin(angle) * v_dir
                face_pts[F].append(pt)
                
    # 4. Construct meshes
    meshes = []
    
    # A. Render the Cube Wireframe (using all 18 edges)
    wire_radius = 0.01 / res
    wire_color = [150, 150, 150, 100]  # Semi-transparent grey
    for E in range(18):
        u, v = EDGE_VERTS[E]
        cyl = trimesh.creation.cylinder(radius=wire_radius, segment=[V[u], V[v]])
        cyl.visual.face_colors = wire_color
        meshes.append(cyl)
        
    # B. Render the extracted loops
    pipe_radius = 0.025 / res
    joint_radius = pipe_radius * 1.3
    
    for loop in loops:
        # Determine if path is open or closed based on parity
        # (Open loops start and end at a face -> odd length)
        is_open = (len(loop) % 2 != 0) 
        
        loop_color = np.random.randint(50, 220, size=3).tolist() + [255]
        
        pts = []
        n_len = len(loop)
        for i, item in enumerate(loop):
            if i % 2 == 1:
                pts.append(edge_pts[item].pop(0))
            else:
                keep_face = False
                if is_open and (i == 0 or i == n_len - 1):
                    keep_face = True
                else:
                    prev_edge = loop[(i - 1) % n_len]
                    next_edge = loop[(i + 1) % n_len]
                    if prev_edge == next_edge:
                        keep_face = True
                        
                if keep_face:
                    pts.append(face_pts[item].pop(0))
                
        num_segments = len(pts) - 1 if is_open else len(pts)
        
        for i in range(num_segments):
            p1 = pts[i]
            p2 = pts[(i + 1) % len(pts)]
            
            # Pipe connection
            if np.linalg.norm(p1 - p2) > 1e-6:
                cyl = trimesh.creation.cylinder(radius=pipe_radius, segment=[p1, p2])
                cyl.visual.face_colors = loop_color
                meshes.append(cyl)
            
            # Joint Sphere
            sph = trimesh.creation.icosphere(radius=joint_radius)
            sph.apply_translation(p1)
            sph.visual.face_colors = loop_color
            meshes.append(sph)
            
        # Draw the terminating node
        if not is_open:
            # Close the circle
            sph = trimesh.creation.icosphere(radius=joint_radius)
            sph.apply_translation(pts[-1])
            sph.visual.face_colors = loop_color
            meshes.append(sph)
        else:
            # Highlight Open Boundary endpoints (in bright Red)
            end_sph = trimesh.creation.icosphere(radius=joint_radius * 1.5)
            end_sph.apply_translation(pts[-1])
            end_sph.visual.face_colors = [255, 30, 30, 255]
            meshes.append(end_sph)
            
            start_sph = trimesh.creation.icosphere(radius=joint_radius * 1.5)
            start_sph.apply_translation(pts[0])
            start_sph.visual.face_colors = [255, 30, 30, 255]
            meshes.append(start_sph)
            
    if not meshes:
        return trimesh.Trimesh()
        
    return trimesh.util.concatenate(meshes)

def collapse_face_inner(face_registers: List[Dict]) -> List[Dict]:
    solved_uturn, ambiguous_uturn, unsolvable_uturn = extract_loops_with_uturns(face_registers)
    return solved_uturn, ambiguous_uturn, unsolvable_uturn


def collapse_face_boundary(face_registers: List[Dict]) -> List[Dict]:
    solved_boundary, ambiguous_boundary, unsolvable_boundary = extract_loops_with_boundaries(face_registers)
    return solved_boundary, ambiguous_boundary, unsolvable_boundary