import os
import pickle
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import trimesh

# Global variables for worker processes to avoid memory duplication
_worker_vertices = None
_worker_faces = None
_worker_step = None
_worker_v_offsets = None
_worker_triangles = None

# def _init_worker(vertices, faces, res):
#     """Initializer function to set up read-only global data for each worker."""
#     global _worker_vertices, _worker_faces, _worker_step
#     global _worker_v_offsets, _worker_triangles
    
#     _worker_vertices = vertices
#     _worker_faces = faces
#     _worker_step = 1.0 / res
    
#     _worker_v_offsets = np.array([
#         [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
#         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
#     ], dtype=float)

#     # Hardcoded mapping of Triangle Facets (T0 to T11)
#     # Format: ( (v0, v1, v2), (e0, e1, e2) )
#     # e0 is the edge between v0-v1, e1 between v1-v2, e2 between v2-v0
#     _worker_triangles = [
#         ((0, 1, 2), (0, 1, 12)),     # T0: Bottom Front-Right
#         ((2, 3, 0), (2, 3, 12)),     # T1: Bottom Back-Left
#         ((4, 5, 6), (4, 5, 13)),     # T2: Top Front-Right
#         ((6, 7, 4), (6, 7, 13)),     # T3: Top Back-Left
#         ((0, 1, 4), (0, 14, 8)),     # T4: Front Left
#         ((4, 1, 5), (14, 9, 4)),     # T5: Front Right
#         ((1, 2, 6), (1, 10, 15)),    # T6: Right Front
#         ((5, 1, 6), (9, 15, 5)),     # T7: Right Back
#         ((2, 3, 7), (2, 11, 16)),    # T8: Back Right
#         ((6, 2, 7), (10, 16, 6)),    # T9: Back Left
#         ((3, 0, 7), (3, 17, 11)),    # T10: Left Back
#         ((7, 0, 4), (17, 8, 7))      # T11: Left Front
#     ]

# def intersect_facet_with_mesh(V0, V1, V2, mesh_triangles):
#     """
#     Vectorized extraction of intersection segments between a single cube facet 
#     (V0, V1, V2) and an array of mesh triangles.
#     """
#     E1 = V1 - V0
#     E2 = V2 - V0
#     Nc = np.cross(E1, E2)
#     len_Nc = np.linalg.norm(Nc)
#     if len_Nc < 1e-12:
#         return []
#     Nc = Nc / len_Nc
    
#     M0 = mesh_triangles[:, 0, :]
#     M1 = mesh_triangles[:, 1, :]
#     M2 = mesh_triangles[:, 2, :]
    
#     # Signed distances to the plane of the facet
#     d0 = np.sum((M0 - V0) * Nc, axis=1)
#     d1 = np.sum((M1 - V0) * Nc, axis=1)
#     d2 = np.sum((M2 - V0) * Nc, axis=1)
    
#     has_pos = (d0 > 1e-8) | (d1 > 1e-8) | (d2 > 1e-8)
#     has_neg = (d0 < -1e-8) | (d1 < -1e-8) | (d2 < -1e-8)
    
#     # Identify edges strictly on the plane
#     d0_zero = np.abs(d0) <= 1e-8
#     d1_zero = np.abs(d1) <= 1e-8
#     d2_zero = np.abs(d2) <= 1e-8
#     coplanar_edge = (d0_zero & d1_zero) | (d1_zero & d2_zero) | (d2_zero & d0_zero)
    
#     # Triangle crosses the plane or has an edge exactly on the plane
#     intersects_plane = (has_pos & has_neg) | coplanar_edge
    
#     valid_idx = np.where(intersects_plane)[0]
#     if len(valid_idx) == 0:
#         return []
        
#     segments = []
#     for i in valid_idx:
#         m0, m1, m2 = M0[i], M1[i], M2[i]
#         da, db, dc = d0[i], d1[i], d2[i]
        
#         pts = []
#         if coplanar_edge[i]:
#             if d0_zero[i] and d1_zero[i]: pts.extend([m0, m1])
#             if d1_zero[i] and d2_zero[i]: pts.extend([m1, m2])
#             if d2_zero[i] and d0_zero[i]: pts.extend([m2, m0])
#         else:
#             for a, b, d_a, d_b in [(m0, m1, da, db), (m1, m2, db, dc), (m2, m0, dc, da)]:
#                 if d_a * d_b < -1e-14:
#                     t = d_a / (d_a - d_b)
#                     pts.append(a + t * (b - a))
#                 elif abs(d_a) <= 1e-8:
#                     pts.append(a)
                    
#         # Filter for unique points to form the segment
#         unique_pts = []
#         for p in pts:
#             if not any(np.linalg.norm(p - up) < 1e-6 for up in unique_pts):
#                 unique_pts.append(p)
                
#         if len(unique_pts) >= 2:
#             P1, P2 = unique_pts[0], unique_pts[1]
            
#             # Sutherland-Hodgman style segment clipping against the facet boundaries
#             dP = P2 - P1
#             t_min, t_max = 0.0, 1.0
            
#             valid = True
#             for j in range(3):
#                 A = [V0, V1, V2][j]
#                 B = [V1, V2, V0][j]
#                 edge = B - A
#                 nk = np.cross(Nc, edge) # Inward-pointing normal in the plane
                
#                 Ck_P1 = np.dot(P1 - A, nk)
#                 dot = np.dot(dP, nk)
                
#                 if dot > 1e-8:
#                     t_min = max(t_min, -Ck_P1 / dot)
#                 elif dot < -1e-8:
#                     t_max = min(t_max, -Ck_P1 / dot)
#                 else:
#                     if Ck_P1 < -1e-8: # Parallel and strictly outside
#                         valid = False
#                         break
                        
#             # Accept if segment overlap is valid and has noticeable length
#             if valid and t_min <= t_max + 1e-8 and t_max - t_min > 1e-6:
#                 segments.append((P1 + t_min * dP, P1 + t_max * dP))
                
#     return segments

# def _process_cube(data):
#     """Worker function to compute face weights using adjacency graph topology."""
#     ix, iy, iz = data['cube_indices']
#     f_idx = data['face_indices']
    
#     # Initialize 12 zeros for the 12 triangular facets
#     face_weights = [0] * 12
    
#     if not f_idx:
#         data['face_weights'] = face_weights
#         return data
        
#     base_pos = np.array([ix, iy, iz]) * _worker_step
#     cube_verts = base_pos + (_worker_v_offsets * _worker_step)
    
#     mesh_triangles = _worker_vertices[_worker_faces[f_idx]]
    
#     for t_idx, ((v0, v1, v2), edge_indices) in enumerate(_worker_triangles):
#         V0 = cube_verts[v0]
#         V1 = cube_verts[v1]
#         V2 = cube_verts[v2]
        
#         segments = intersect_facet_with_mesh(V0, V1, V2, mesh_triangles)
#         if not segments:
#             continue
            
#         # 1. Build a topological graph from the line segments
#         nodes = []
#         edges = []
#         for p1, p2 in segments:
#             idx1 = -1
#             for i, n in enumerate(nodes):
#                 if np.linalg.norm(p1 - n) < 1e-5:
#                     idx1 = i
#                     break
#             if idx1 == -1:
#                 idx1 = len(nodes)
#                 nodes.append(p1)
                
#             idx2 = -1
#             for i, n in enumerate(nodes):
#                 if np.linalg.norm(p2 - n) < 1e-5:
#                     idx2 = i
#                     break
#             if idx2 == -1:
#                 idx2 = len(nodes)
#                 nodes.append(p2)
                
#             if idx1 != idx2:
#                 # Ensure edges are recorded uniquely to avoid synthetic degree inflations
#                 edge_tuple = tuple(sorted((idx1, idx2)))
#                 edges.append(edge_tuple)
                
#         if not edges:
#             continue
            
#         unique_edges = set(edges)
#         adj = {i: [] for i in range(len(nodes))}
#         for u, v in unique_edges:
#             adj[u].append(v)
#             adj[v].append(u)
            
#         # 2. Extract Connected Components
#         visited = set()
#         components = []
#         for i in range(len(nodes)):
#             if i not in visited:
#                 comp = []
#                 q = [i]
#                 visited.add(i)
#                 while q:
#                     curr = q.pop(0)
#                     comp.append(curr)
#                     for neighbor in adj[curr]:
#                         if neighbor not in visited:
#                             visited.add(neighbor)
#                             q.append(neighbor)
#                 components.append(comp)
                
#         # 3. Analyze Endpoint Placements
#         for comp in components:
#             # Graph nodes with a single connection natively form the open ends of any polyline
#             endpoints = [n for n in comp if len(adj[n]) == 1]
#             if not endpoints:
#                 continue # Loops do not interact with boundaries and are ignored
                
#             # Keep counts of which edge (or inside state "-1") these endpoints terminate on
#             edge_counts = {-1: 0}
#             for e_idx in edge_indices:
#                 edge_counts[e_idx] = 0
                
#             for ep in endpoints:
#                 P = nodes[ep]
#                 assigned_edges = []
                
#                 # Check distance to all 3 boundaries
#                 for j in range(3):
#                     A = cube_verts[[v0, v1, v2][j]]
#                     B = cube_verts[[v1, v2, v0][j]]
#                     edge_vec = B - A
#                     length = np.linalg.norm(edge_vec)
#                     if length < 1e-12: continue
                    
#                     t = np.dot(P - A, edge_vec) / (length**2)
#                     if -1e-5 <= t <= 1 + 1e-5:
#                         proj = A + t * edge_vec
#                         if np.linalg.norm(P - proj) < 1e-4:
#                             assigned_edges.append(edge_indices[j])
                            
#                 if not assigned_edges:
#                     assigned_edges.append(-1)
                    
#                 for e_id in assigned_edges:
#                     edge_counts[e_id] += 1
                    
#             # 4. Resolve Topology Metrics
#             for e_id, count in edge_counts.items():
#                 if e_id == -1:
#                     # Every endpoint ending *inside* the facet increments open boundaries natively
#                     face_weights[t_idx] += count
#                 else:
#                     # Every connected matching pair of endpoints arriving on the *same* edge makes 1 U-Turn
#                     face_weights[t_idx] += count // 2
                    
#     data['face_weights'] = face_weights
#     if (ix, iy, iz) == (2, 11, 10):
#         print(edge_counts)
#         breakpoint()
#     return data

# def calculate_face_weights(mesh, res, cube_data, output_dir, batch_size=10000, num_workers=None):
#     """
#     Calculates U-turns and open-boundary components on the 12 triangular facets of each subcube.
    
#     Args:
#         mesh: An object with `.vertices` (N, 3) and `.faces` (M, 3) attributes.
#         res (int): Grid resolution.
#         cube_data (list of dict): Subcubes and their intersected mesh data.
#         output_dir (str): Location to persist `.pkl` result file.
#         batch_size (int): Cubes chunked in memory per batch.
#         num_workers (int): Cores to employ.
            
#     Returns:
#         list of dict: Updated cube dictionaries inclusive of `face_weights` (length 12 integers).
#     """
#     vertices = np.asarray(mesh.vertices)
#     faces = np.asarray(mesh.faces)
    
#     if num_workers is None:
#         num_workers = max(1, mp.cpu_count() - 1)
        
#     pool = mp.Pool(processes=num_workers, initializer=_init_worker, initargs=(vertices, faces, res))
    
#     updated_data = []
#     try:
#         with tqdm(total=len(cube_data), desc="Calculating Face Weights (U-Turns & Boundaries)") as pbar:
#             for i in range(0, len(cube_data), batch_size):
#                 batch = cube_data[i : i + batch_size]
#                 chunk_size = max(1, len(batch) // (num_workers * 4))
                
#                 batch_results = pool.map(_process_cube, batch, chunksize=chunk_size)
#                 updated_data.extend(batch_results)
                
#                 pbar.update(len(batch))
#     finally:
#         pool.close()
#         pool.join()
        
#     os.makedirs(output_dir, exist_ok=True)
#     with open(os.path.join(output_dir, 'face_registers.pkl'), 'wb') as f:
#         pickle.dump(updated_data, f)
        
#     return updated_data

def _init_worker(vertices, faces, res):
    """Initializer function to set up read-only global data for each worker."""
    global _worker_vertices, _worker_faces, _worker_step
    global _worker_v_offsets, _worker_triangles
    
    _worker_vertices = vertices
    _worker_faces = faces
    _worker_step = 1.0 / res
    
    _worker_v_offsets = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=float)

    # Hardcoded mapping of Triangle Facets (T0 to T11)
    # Format: ( (v0, v1, v2), (e0, e1, e2) )
    # e0 is the edge between v0-v1, e1 between v1-v2, e2 between v2-v0
    _worker_triangles = [
        ((0, 1, 2), (0, 1, 12)),     # T0: Bottom Front-Right
        ((2, 3, 0), (2, 3, 12)),     # T1: Bottom Back-Left
        ((4, 5, 6), (4, 5, 13)),     # T2: Top Front-Right
        ((6, 7, 4), (6, 7, 13)),     # T3: Top Back-Left
        ((0, 1, 4), (0, 14, 8)),     # T4: Front Left
        ((4, 1, 5), (14, 9, 4)),     # T5: Front Right
        ((1, 2, 6), (1, 10, 15)),    # T6: Right Front
        ((5, 1, 6), (9, 15, 5)),     # T7: Right Back
        ((2, 3, 7), (2, 11, 16)),    # T8: Back Right
        ((6, 2, 7), (10, 16, 6)),    # T9: Back Left
        ((3, 0, 7), (3, 17, 11)),    # T10: Left Back
        ((7, 0, 4), (17, 8, 7))      # T11: Left Front
    ]

def intersect_facet_with_mesh(V0, V1, V2, mesh_triangles):
    """
    Vectorized extraction of intersection segments between a single cube facet 
    (V0, V1, V2) and an array of mesh triangles.
    """
    E1 = V1 - V0
    E2 = V2 - V0
    Nc = np.cross(E1, E2)
    len_Nc = np.linalg.norm(Nc)
    if len_Nc < 1e-12:
        return []
    Nc = Nc / len_Nc
    
    M0 = mesh_triangles[:, 0, :]
    M1 = mesh_triangles[:, 1, :]
    M2 = mesh_triangles[:, 2, :]
    
    # Signed distances to the plane of the facet
    d0 = np.sum((M0 - V0) * Nc, axis=1)
    d1 = np.sum((M1 - V0) * Nc, axis=1)
    d2 = np.sum((M2 - V0) * Nc, axis=1)
    
    has_pos = (d0 > 1e-8) | (d1 > 1e-8) | (d2 > 1e-8)
    has_neg = (d0 < -1e-8) | (d1 < -1e-8) | (d2 < -1e-8)
    
    # Identify edges strictly on the plane
    d0_zero = np.abs(d0) <= 1e-8
    d1_zero = np.abs(d1) <= 1e-8
    d2_zero = np.abs(d2) <= 1e-8
    coplanar_edge = (d0_zero & d1_zero) | (d1_zero & d2_zero) | (d2_zero & d0_zero)
    
    # Triangle crosses the plane or has an edge exactly on the plane
    intersects_plane = (has_pos & has_neg) | coplanar_edge
    
    valid_idx = np.where(intersects_plane)[0]
    if len(valid_idx) == 0:
        return []
        
    segments = []
    for i in valid_idx:
        m0, m1, m2 = M0[i], M1[i], M2[i]
        da, db, dc = d0[i], d1[i], d2[i]
        
        pts = []
        if coplanar_edge[i]:
            if d0_zero[i] and d1_zero[i]: pts.extend([m0, m1])
            if d1_zero[i] and d2_zero[i]: pts.extend([m1, m2])
            if d2_zero[i] and d0_zero[i]: pts.extend([m2, m0])
        else:
            for a, b, d_a, d_b in [(m0, m1, da, db), (m1, m2, db, dc), (m2, m0, dc, da)]:
                if d_a * d_b < -1e-14:
                    t = d_a / (d_a - d_b)
                    pts.append(a + t * (b - a))
                elif abs(d_a) <= 1e-8:
                    pts.append(a)
                    
        # Filter for unique points to form the segment
        unique_pts = []
        for p in pts:
            if not any(np.linalg.norm(p - up) < 1e-8 for up in unique_pts):
                unique_pts.append(p)
                
        if len(unique_pts) >= 2:
            P1, P2 = unique_pts[0], unique_pts[1]
            
            # Sutherland-Hodgman style segment clipping against the facet boundaries
            dP = P2 - P1
            t_min, t_max = 0.0, 1.0
            
            valid = True
            for j in range(3):
                A = [V0, V1, V2][j]
                B = [V1, V2, V0][j]
                edge = B - A
                nk = np.cross(Nc, edge) # Inward-pointing normal in the plane
                
                Ck_P1 = np.dot(P1 - A, nk)
                dot = np.dot(dP, nk)
                
                if dot > 1e-8:
                    t_min = max(t_min, -Ck_P1 / dot)
                elif dot < -1e-8:
                    t_max = min(t_max, -Ck_P1 / dot)
                else:
                    if Ck_P1 < -1e-8: # Parallel and strictly outside
                        valid = False
                        break
                        
            # Accept if segment overlap is valid and has noticeable length
            if valid and t_min <= t_max + 1e-8 and t_max - t_min > 1e-8:
                segments.append((P1 + t_min * dP, P1 + t_max * dP))
                
    return segments

def _process_cube(data):
    """Worker function to compute face weights using adjacency graph topology."""
    ix, iy, iz = data['cube_indices']
    f_idx = data['face_indices']
    
    # Initialize 12 zeros for the 12 triangular facets
    face_weights = [0] * 12
    
    if not f_idx:
        data['face_weights'] = face_weights
        return data
        
    base_pos = np.array([ix, iy, iz]) * _worker_step
    cube_verts = base_pos + (_worker_v_offsets * _worker_step)
    
    mesh_triangles = _worker_vertices[_worker_faces[f_idx]]
    
    for t_idx, ((v0, v1, v2), edge_indices) in enumerate(_worker_triangles):
        V0 = cube_verts[v0]
        V1 = cube_verts[v1]
        V2 = cube_verts[v2]
        
        segments = intersect_facet_with_mesh(V0, V1, V2, mesh_triangles)
        if not segments:
            continue
            
        # 1. Build a topological graph from the line segments
        nodes = []
        edges = []
        for p1, p2 in segments:
            idx1 = -1
            for i, n in enumerate(nodes):
                if np.linalg.norm(p1 - n) < 1e-8:
                    idx1 = i
                    break
            if idx1 == -1:
                idx1 = len(nodes)
                nodes.append(p1)
                
            idx2 = -1
            for i, n in enumerate(nodes):
                if np.linalg.norm(p2 - n) < 1e-8:
                    idx2 = i
                    break
            if idx2 == -1:
                idx2 = len(nodes)
                nodes.append(p2)
                
            if idx1 != idx2:
                # Ensure edges are recorded uniquely to avoid synthetic degree inflations
                edge_tuple = tuple(sorted((idx1, idx2)))
                edges.append(edge_tuple)
                
        if not edges:
            continue
            
        unique_edges = set(edges)
        adj = {i: [] for i in range(len(nodes))}
        for u, v in unique_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # 2. Extract Connected Components
        visited = set()
        components = []
        for i in range(len(nodes)):
            if i not in visited:
                comp = []
                q = [i]
                visited.add(i)
                while q:
                    curr = q.pop(0)
                    comp.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append(neighbor)
                components.append(comp)
                
        # 3. Analyze Endpoint Placements
        for comp in components:
            # Graph nodes with a single connection natively form the open ends of any polyline
            endpoints = [n for n in comp if len(adj[n]) == 1]
            if not endpoints:
                continue # Loops do not interact with boundaries and are ignored
                
            # Keep counts of which edge (or inside state "-1") these endpoints terminate on
            edge_counts = {-1: 0}
            for e_idx in edge_indices:
                edge_counts[e_idx] = 0
                
            for ep in endpoints:
                P = nodes[ep]
                assigned_edges = []
                
                # Check distance to all 3 boundaries
                for j in range(3):
                    A = cube_verts[[v0, v1, v2][j]]
                    B = cube_verts[[v1, v2, v0][j]]
                    edge_vec = B - A
                    length = np.linalg.norm(edge_vec)
                    if length < 1e-12: continue
                    
                    t = np.dot(P - A, edge_vec) / (length**2)
                    if -1e-8 <= t <= 1 + 1e-8:
                        proj = A + t * edge_vec
                        if np.linalg.norm(P - proj) < 1e-8:
                            assigned_edges.append(edge_indices[j])
                            
                if not assigned_edges:
                    assigned_edges.append(-1)
                    
                for e_id in assigned_edges:
                    edge_counts[e_id] += 1
                    
            # 4. Resolve Topology Metrics
            for e_id, count in edge_counts.items():
                if e_id != -1:
                    # Every connected matching pair of endpoints arriving on the *same* edge makes 1 U-Turn
                    face_weights[t_idx] += count // 2
                    
    data['face_weights'] = face_weights
    return data

def calculate_face_weights(mesh, res, cube_data, output_dir, batch_size=10000, num_workers=None):
    """
    Calculates U-turns on the 12 triangular facets of each subcube.
    
    Args:
        mesh: An object with `.vertices` (N, 3) and `.faces` (M, 3) attributes.
        res (int): Grid resolution.
        cube_data (list of dict): Subcubes and their intersected mesh data.
        output_dir (str): Location to persist `.pkl` result file.
        batch_size (int): Cubes chunked in memory per batch.
        num_workers (int): Cores to employ.
            
    Returns:
        list of dict: Updated cube dictionaries inclusive of `face_weights` (length 12 integers).
    """
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
        
    pool = mp.Pool(processes=num_workers, initializer=_init_worker, initargs=(vertices, faces, res))
    
    updated_data = []
    try:
        with tqdm(total=len(cube_data), desc="Calculating Face Weights (U-Turns)") as pbar:
            for i in range(0, len(cube_data), batch_size):
                batch = cube_data[i : i + batch_size]
                chunk_size = max(1, len(batch) // (num_workers * 4))
                
                batch_results = pool.map(_process_cube, batch, chunksize=chunk_size)
                updated_data.extend(batch_results)
                
                pbar.update(len(batch))
    finally:
        pool.close()
        pool.join()
        
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'face_registers.pkl'), 'wb') as f:
        pickle.dump(updated_data, f)
        
    return updated_data



def extract_boundary_weights(res: int, boundaries: np.ndarray, cube_dicts: list) -> list:
    """
    Extracts mesh open boundary intersection data on 12 triangulated cube facets.
    
    Args:
        res (int): Resolution of the grid. The unit space (0,1) is split into res**3 cubes.
        boundaries (np.ndarray): Array of shape (n, 2, 3) representing n open boundary segments.
        cube_dicts (list): List of dictionaries, each containing:
            - 'cube_indices': tuple of (x, y, z) cube indices
            - 'edge_indices': list of boundary indices from the `boundaries` array
            - 'face_weights': list of 12 integer weights for each triangle facet
            
    Returns:
        list: The modified list of dictionaries.
    """
    
    # Base vertex coordinates for a unit cube [0, 1]^3 
    # Mapped strictly according to your specifications:
    unit_V = np.array([
        [0, 0, 0], # 0: Front-Left-Bottom
        [1, 0, 0], # 1: Front-Right-Bottom
        [1, 1, 0], # 2: Back-Right-Bottom
        [0, 1, 0], # 3: Back-Left-Bottom
        [0, 0, 1], # 4: Front-Left-Top
        [1, 0, 1], # 5: Front-Right-Top
        [1, 1, 1], # 6: Back-Right-Top
        [0, 1, 1]  # 7: Back-Left-Top
    ], dtype=float)

    # 12 triangles strictly mapped from your provided edge combinations
    # Each row specifies the 3 vertex indices forming the triangle
    tri_indices = np.array([
        [0, 1, 2],  # T0: Bottom face half 1 (Edges: 0, 1, 12)
        [2, 3, 0],  # T1: Bottom face half 2 (Edges: 2, 3, 12)
        [4, 5, 6],  # T2: Top face half 1    (Edges: 4, 5, 13)
        [6, 7, 4],  # T3: Top face half 2    (Edges: 6, 7, 13)
        [0, 1, 4],  # T4: Front face half 1  (Edges: 0, 8, 14)
        [4, 5, 1],  # T5: Front face half 2  (Edges: 4, 9, 14)
        [1, 2, 6],  # T6: Right face half 1  (Edges: 1, 10, 15)
        [5, 6, 1],  # T7: Right face half 2  (Edges: 5, 9, 15)
        [2, 3, 7],  # T8: Back face half 1   (Edges: 2, 11, 16)
        [6, 7, 2],  # T9: Back face half 2   (Edges: 6, 10, 16)
        [3, 0, 7],  # T10: Left face half 1  (Edges: 3, 11, 17)
        [7, 4, 0]   # T11: Left face half 2  (Edges: 7, 8, 17)
    ])

    # A robust, scale-invariant epsilon to handle floating point inaccuracy
    eps = 1e-12

    for cube in cube_dicts:
        cx, cy, cz = cube['cube_indices']
        
        # Scale and translate the base unit cube coordinates into actual 3D space
        V = (unit_V + np.array([cx, cy, cz])) / float(res)

        # Batch-extract the geometric coordinates for the 12 triangles' 3 vertices
        tri_v0 = V[tri_indices[:, 0]]  # Shape: (12, 3)
        tri_v1 = V[tri_indices[:, 1]]  # Shape: (12, 3)
        tri_v2 = V[tri_indices[:, 2]]  # Shape: (12, 3)

        # Initialize the new 12-dimensional boundary_weights array
        boundary_weights = np.zeros(12, dtype=int)

        # Iterate only through the relevant boundary indices registered to this specific cube
        edge_idxs = cube['edge_indices']
        for idx in edge_idxs:
            p1 = boundaries[idx, 0]  # Start point of the boundary segment (3,)
            p2 = boundaries[idx, 1]  # End point of the boundary segment (3,)

            # =======================================================================
            # Möller-Trumbore Segment-Triangle Intersection (Batched over 12 facets)
            # =======================================================================
            direction = p2 - p1
            edge1 = tri_v1 - tri_v0
            edge2 = tri_v2 - tri_v0

            h = np.cross(direction, edge2)  # Shape: (12, 3)
            a = np.sum(edge1 * h, axis=1)   # Shape: (12,)

            # Filter out segments parallel to the facet
            valid = np.abs(a) > eps
            if not np.any(valid):
                continue

            f = np.zeros_like(a)
            f[valid] = 1.0 / a[valid]

            s = p1 - tri_v0  # Shape: (12, 3)
            u = f * np.sum(s * h, axis=1)  # Barycentric U (12,)

            q = np.cross(s, edge1)  # Shape: (12, 3)
            v = f * np.sum(direction * q, axis=1)  # Barycentric V (12,)

            t = f * np.sum(edge2 * q, axis=1)  # Intersection parameter along segment (12,)

            # Intersection valid if: 
            # 1. Barycentric coords are valid (hits inside triangle)
            # 2. t is between 0 and 1 (hit on the finite segment between p1 and p2)
            intersect = valid & \
                        (u >= -eps) & (u <= 1.0 + eps) & \
                        (v >= -eps) & (u + v <= 1.0 + eps) & \
                        (t >= -eps) & (t <= 1.0 + eps)

            # Increment weights for facets that this boundary intersected
            boundary_weights += intersect.astype(int)

        # Update and save the results inside the current dictionary
        cube['boundary_weights'] = boundary_weights.tolist()
        # cube['face_weights'] = [fw + bw for fw, bw in zip(cube['face_weights'], cube['boundary_weights'])]

    return cube_dicts


def visualize_cube_face_weights(cube_dict, res, mesh=None):
    """
    Creates a 3D visualization mesh for a given cube, showing its wireframe, 
    small light green cubes representing the magnitude of the face weights,
    and optionally the intersecting original mesh faces.
    
    Args:
        cube_dict (dict): Dictionary containing 'cube_indices', 'face_weights', and 'face_indices'.
        res (int): Grid resolution.
        mesh (trimesh.Trimesh, optional): The original mesh object to extract faces from.
        
    Returns:
        trimesh.Trimesh: A concatenated mesh visualizing the cube and its face weights.
    """
    ix, iy, iz = cube_dict['cube_indices']
    step = 1.0 / res
    base_pos = np.array([ix, iy, iz]) * step
    
    v_offsets = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=float)
    
    cube_verts = base_pos + (v_offsets * step)
    cube_center = np.mean(cube_verts, axis=0)
    
    triangles = [
        ((0, 1, 2), (0, 1, 12)),     # T0: Bottom Front-Right
        ((2, 3, 0), (2, 3, 12)),     # T1: Bottom Back-Left
        ((4, 5, 6), (4, 5, 13)),     # T2: Top Front-Right
        ((6, 7, 4), (6, 7, 13)),     # T3: Top Back-Left
        ((0, 1, 4), (0, 14, 8)),     # T4: Front Left
        ((4, 1, 5), (14, 9, 4)),     # T5: Front Right
        ((1, 2, 6), (1, 10, 15)),    # T6: Right Front
        ((5, 1, 6), (9, 15, 5)),     # T7: Right Back
        ((2, 3, 7), (2, 11, 16)),    # T8: Back Right
        ((6, 2, 7), (10, 16, 6)),    # T9: Back Left
        ((3, 0, 7), (3, 17, 11)),    # T10: Left Back
        ((7, 0, 4), (17, 8, 7))      # T11: Left Front
    ]
    
    meshes = []
    
    # 1. Create Wireframe (Cylinders for the 18 edges matching topology)
    edge_indices = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
        (0, 2), (4, 6), (1, 4), (1, 6), (2, 7), (0, 7)
    ]
    
    cyl_radius = step * 0.015
    for e_start, e_end in edge_indices:
        p1 = cube_verts[e_start]
        p2 = cube_verts[e_end]
        dist = np.linalg.norm(p2 - p1)
        if dist > 1e-8:
            cyl = trimesh.creation.cylinder(radius=cyl_radius, segment=[p1, p2])
            cyl.visual.face_colors = [120, 120, 120, 255]  # Gray wireframe
            meshes.append(cyl)
            
    # 2. Add light green cubes for face weights
    face_weights = cube_dict.get('face_weights', [0]*12)
    box_size = step * 0.1
    
    for t_idx, ((v0, v1, v2), _) in enumerate(triangles):
        weight = face_weights[t_idx]
        if weight <= 0:
            continue
            
        V0 = cube_verts[v0]
        V1 = cube_verts[v1]
        V2 = cube_verts[v2]
        
        centroid = (V0 + V1 + V2) / 3.0
        
        # Compute inward normal to push cubes inside
        E1 = V1 - V0
        E2 = V2 - V0
        normal = np.cross(E1, E2)
        n_len = np.linalg.norm(normal)
        if n_len > 1e-12:
            normal /= n_len
        else:
            normal = np.array([0.0, 0.0, 1.0])
            
        # Ensure normal points inwards towards the center of the cube
        if np.dot(normal, cube_center - centroid) < 0:
            normal = -normal
        
        # Shrink the area slightly so the cubes don't overlap with the edge wireframes
        shrink = 0.7
        V0_s = centroid + shrink * (V0 - centroid)
        V1_s = centroid + shrink * (V1 - centroid)
        V2_s = centroid + shrink * (V2 - centroid)
        
        # Place small indicator cubes distributed across the facet
        for w in range(weight):
            box = trimesh.creation.box(extents=[box_size, box_size, box_size])
            
            if weight == 1:
                # If only one weight, place it right at the centroid
                pos = centroid
            else:
                # Use a low-discrepancy sequence to distribute multiple cubes inside the triangle
                f = ((w + 1) * 0.618033988749895) % 1.0
                g = ((w + 1) * 0.732050807568877) % 1.0
                if f + g > 1.0:
                    f = 1.0 - f
                    g = 1.0 - g
                    
                pos = V0_s + f * (V1_s - V0_s) + g * (V2_s - V0_s)
            
            # Shift the position inwards along the normal so it sits inside the main cube
            pos = pos + normal * (box_size * 0.6)
            
            box.apply_translation(pos)
            box.visual.face_colors = [144, 238, 144, 255]  # Light Green
            meshes.append(box)

    # 3. Add original intersecting mesh faces
    if mesh is not None and cube_dict.get('face_indices'):
        f_idx = cube_dict['face_indices']
        m_verts = np.asarray(mesh.vertices)
        m_faces = np.asarray(mesh.faces)[f_idx]
        
        # Create a submesh for the original intersecting faces
        sub_mesh = trimesh.Trimesh(vertices=m_verts, faces=m_faces, process=False)
        # Apply a distinct color (e.g., semi-transparent light blue) to differentiate it
        sub_mesh.visual.face_colors = [173, 216, 230, 180]
        meshes.append(sub_mesh)
            
    if meshes:
        return trimesh.util.concatenate(meshes)
    else:
        return trimesh.Trimesh()


def combine_dict_lists(list_a, list_b):
    """
    Combines dicts from list_b into list_a based on matching 'cube_indices'.
    Adds 'boundary_weights' element-wise to 'face_weights'.
    """
    # Create a lookup dictionary for fast O(1) access to list_a's elements
    lookup_a = {d['cube_indices']: d for d in list_a}
    
    for dict_b in list_b:
        cube_indices = dict_b.get('cube_indices')
        
        # Check if the matching cube_indices exists in list_a
        if cube_indices in lookup_a:
            dict_a = lookup_a[cube_indices]
            
            # 1. Element-wise addition of boundary_weights to face_weights
            face_w = dict_a.get('face_weights', [])
            boundary_w = dict_b.get('boundary_weights', [])
            
            # Using zip to add corresponding elements together
            dict_a['face_weights'] = [f + b for f, b in zip(face_w, boundary_w)]
            
            # 2. Add the rest of the contents of dict_b into dict_a
            dict_a.update(dict_b)
            
    # Add 'num_boundary': 0 to dicts in list_a that weren't in list_b
    for dict_a in list_a:
        if 'num_boundary' not in dict_a:
            dict_a['num_boundary'] = 0
            
    # Return the modified list_a
    return list_a


def feature_face(mesh, res, face_registers, boundaries, boundary_registers, output_dir, debug=False):
    """
    Calculates U-turns and open-boundary components on the 12 triangular facets of each subcube.
    """
    updated_data = calculate_face_weights(mesh, res, face_registers, output_dir)
    boundary_registers = extract_boundary_weights(res, boundaries, boundary_registers)
    combined_result = combine_dict_lists(updated_data, boundary_registers)
    if debug:
        os.makedirs(os.path.join(output_dir, 'debug'), exist_ok=True)
        for cube_data in combined_result:
            cube_mesh = visualize_cube_face_weights(cube_data, res, mesh)
            cube_mesh.export(os.path.join(output_dir, 'debug', f'face_weights_{cube_data["cube_indices"]}.ply'))
    return combined_result