import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from itertools import islice
import trimesh
from utils import load_pickle, save_pickle
from functools import partial
import os
import collections

# 1. Hardcode Local Vertices Mapping
# Using a (0,1) local coordinate system: +x=Right, +y=Back, +z=Top
VERTICES = np.array([
    [0, 0, 0],  # 0: Bottom front-left
    [1, 0, 0],  # 1: Bottom front-right
    [1, 1, 0],  # 2: Bottom back-right
    [0, 1, 0],  # 3: Bottom back-left
    [0, 0, 1],  # 4: Top front-left
    [1, 0, 1],  # 5: Top front-right
    [1, 1, 1],  # 6: Top back-right
    [0, 1, 1]   # 7: Top back-left
], dtype=float)

# 2. Hardcode Local Edge Definitions 
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # 0-3: Bottom edges
    (4, 5), (5, 6), (6, 7), (7, 4),  # 4-7: Top edges
    (0, 4), (1, 5), (2, 6), (3, 7),  # 8-11: Vertical edges
    (0, 2), (4, 6),                  # 12-13: Bottom & Top Diagonals
    (1, 4), (1, 6), (2, 7), (0, 7)   # 14-17: Face Diagonals (Front, Right, Back, Left)
]

# Precompute midpoints of the 18 local edges
EDGE_MIDPOINTS = np.array([(VERTICES[u] + VERTICES[v]) / 2.0 for u, v in EDGES])

def compute_fast_normal(midpoints, cube_indices=None):
    """Computes the plane normal for a loop using a fast, vectorized Newell's Method."""
    midpoints = np.array(midpoints)
    if len(midpoints) < 3:
        return np.array([0.0, 0.0, 1.0])  # Fallback for degenerate loops
    
    # Vectorized Newell's Method
    v0 = midpoints
    v1 = np.roll(midpoints, -1, axis=0)
    
    nx = np.sum((v0[:, 1] - v1[:, 1]) * (v0[:, 2] + v1[:, 2]))
    ny = np.sum((v0[:, 2] - v1[:, 2]) * (v0[:, 0] + v1[:, 0]))
    nz = np.sum((v0[:, 0] - v1[:, 0]) * (v0[:, 1] + v1[:, 1]))
    
    normal = np.array([nx, ny, nz])
    norm = np.linalg.norm(normal)
    
    if norm > 1e-6:
        normal = normal / norm
    else:
        # Fallback to simple cross product of first 3 points if degenerate
        normal = np.cross(midpoints[1] - midpoints[0], midpoints[2] - midpoints[0])
        norm = np.linalg.norm(normal)
        normal = normal / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])
    
    # Standardize orientation: choose the normal that points away from the middle 
    # of the small cube.
    centroid = np.mean(midpoints, axis=0)
    cube_center = np.array([0.5, 0.5, 0.5])
    direction_to_plane = centroid - cube_center
    
    if np.linalg.norm(direction_to_plane) > 1e-6:
        # If normal points opposite to the direction from center to plane, flip it
        if np.dot(direction_to_plane, normal) < 0:
            normal = -normal
    else:
        # Fallback if the plane passes exactly through the center of the cube
        idx = np.argmax(np.abs(normal))
        if normal[idx] < 0:
            normal = -normal
        
    return normal

def distribute_normals(n, m):
    """
    Given a normal 'n', returns 'm' normals such that one is 'n' and the 
    rest are distributed as far away from each other as possible on the sphere.
    """
    n = n / np.linalg.norm(n)
    
    if m == 1:
        return [n]
    if m == 2:
        return [n, -n]
    if m == 3:
        # Equilateral triangle configuration on the normal's great circle
        u = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(n, u)) > 0.9:
            u = np.array([0.0, 1.0, 0.0])
        # Orthogonalize u against n
        u = u - np.dot(u, n) * n
        u = u / np.linalg.norm(u)
        
        n2 = -0.5 * n + (np.sqrt(3)/2) * u
        n3 = -0.5 * n - (np.sqrt(3)/2) * u
        return [n, n2, n3]
    if m == 4:
        # Regular tetrahedron configuration
        u = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(n, u)) > 0.9:
            u = np.array([0.0, 1.0, 0.0])
        u = u - np.dot(u, n) * n
        u = u / np.linalg.norm(u)
        v = np.cross(n, u)
        
        sin_theta = np.sqrt(8)/3
        cos_theta = -1/3
        n2 = cos_theta * n + sin_theta * u
        n3 = cos_theta * n + sin_theta * (-0.5 * u + (np.sqrt(3)/2) * v)
        n4 = cos_theta * n + sin_theta * (-0.5 * u - (np.sqrt(3)/2) * v)
        return [n, n2, n3, n4]
        
    # Fallback for m >= 5: Physics-based repulsive particle simulation
    # Initialize with one point at 'n' and the rest randomized
    rng = np.random.RandomState(42)  # Fixed seed for determinism
    pts = rng.randn(m, 3)
    pts[0] = n
    for i in range(m):
        pts[i] /= np.linalg.norm(pts[i])
        
    for _ in range(200): # Relaxation steps
        for i in range(1, m): # Skip the 0th vector; it stays fixed at `n`
            force = np.zeros(3)
            for j in range(m):
                if i == j: continue
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist > 1e-5:
                    force += diff / (dist**3) # Inverse square force
            pts[i] += force * 0.05
            pts[i] /= np.linalg.norm(pts[i])
            
    return [pts[i] for i in range(m)]

def get_connected_components(faces):
    """Partitions a local subset of faces into connected components by shared vertices."""
    v2f = {}
    for i, f in enumerate(faces):
        for v in f:
            if v not in v2f:
                v2f[v] = []
            v2f[v].append(i)
            
    adj = {i: [] for i in range(len(faces))}
    for f_list in v2f.values():
        for i in range(len(f_list)):
            for j in range(i+1, len(f_list)):
                adj[f_list[i]].append(f_list[j])
                adj[f_list[j]].append(f_list[i])
                
    visited = set()
    components = []
    for i in range(len(faces)):
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
    return components

def process_single_cube(cube_data, vertices=None, mesh_faces=None, res=1):
    """Processes a single cube dictionary. Extracted for multiprocessing."""
    loops = cube_data.get('loops', [])
    cube_indices = cube_data.get('cube_indices')
    face_indices = cube_data.get('face_indices', [])
    
    # Calculate `d` parameter based on the number of loops
    num_loops = cube_data.get('num_loops', len(loops))
    if num_loops <= 1:
        cube_data['d'] = 0.0
    else:
        cube_data['d'] = 0.5 * (1.0 - 1.8 / num_loops)
    
    # Fast Center Point Calculation (Approximated using triangle centroids)
    if vertices is not None and mesh_faces is not None and cube_indices is not None and len(face_indices) > 0:
        

        # Calculate the global 3D center of the small cube
        cube_center = (np.array(cube_indices) + 0.5) / res
        local_faces = mesh_faces[face_indices]

        # Pre-calculate the bounding box of the small cube
        cube_min = np.array(cube_indices) / res
        cube_max = (np.array(cube_indices) + 1.0) / res
        
        components = get_connected_components(local_faces)

        # Extract vertices for the intersecting faces
        comp_verts = vertices[local_faces]  # Shape: (NumFaces, 3, 3)
        
        # Fast, vectorized calculation to find the closest point on the face to the cube center
        if len(comp_verts) > 0:
            v0 = comp_verts[:, 0, :]
            v1 = comp_verts[:, 1, :]
            v2 = comp_verts[:, 2, :]
            
            # 1. Compute face normals
            cross_prod = np.cross(v1 - v0, v2 - v0)
            norm_mag = np.linalg.norm(cross_prod, axis=1, keepdims=True)
            # Avoid division by zero for degenerate triangles
            normals = cross_prod / np.where(norm_mag == 0, 1e-8, norm_mag)
            
            # 2. Project the cube_center onto the 3D plane of each face
            v0_to_center = cube_center - v0
            dist_to_plane = np.sum(v0_to_center * normals, axis=1, keepdims=True)
            face_pts = cube_center - dist_to_plane * normals
            
            # 3. Constrain the projected points to the bounding box of the individual face
            # This keeps the projection from wandering off if the face is very small
            face_min = np.min(comp_verts, axis=1)
            face_max = np.max(comp_verts, axis=1)
            face_pts = np.clip(face_pts, face_min, face_max)
            
            # 4. Constrain the points to be strictly within the bounding box of the small cube
            face_pts = np.clip(face_pts, cube_min, cube_max)
        else:
            face_pts = np.empty((0, 3))
        
        # Squared Euclidean distance from the accurate face points to the cube center
        dists = np.sum((face_pts - cube_center)**2, axis=1)  
        
        closest_points = []
        for comp in components:
            # Extract distances for this specific component
            comp_dists = dists[comp]
            best_local_idx = np.argmin(comp_dists)
            best_global_idx = comp[best_local_idx]
            closest_points.append(face_pts[best_global_idx])
                
        if closest_points:
            avg_pt = np.mean(closest_points, axis=0)
            # Ensure the final averaged point is safely clipped within the small cube
            avg_pt = np.clip(avg_pt, cube_min, cube_max)
            cube_data['center_point'] = avg_pt.tolist()

            cube_data['component_points'] = [np.clip(comp_pt, cube_min, cube_max).tolist() for comp_pt in closest_points]
        else:
            cube_data['center_point'] = None

    if not loops:
        cube_data['normals'] = []
        return cube_data
        
    std_normals = []
    
    # 1. Compute fast Newell's best-fit normal for each loop
    for loop in loops:
        original_loop = [edge_idx for edge_idx in loop if edge_idx < 12]
        loop_midpoints = [EDGE_MIDPOINTS[edge_idx] for edge_idx in original_loop]
        # Pass cube_indices explicitly to enforce orientation logic
        n = compute_fast_normal(loop_midpoints, cube_indices)
        std_normals.append(n)
        
    # 2. Cluster identical normals (using un-oriented cosine similarity)
    clusters = [] # List of index lists
    for i, n in enumerate(std_normals):
        placed = False
        for cluster in clusters:
            base_n = std_normals[cluster[0]]
            # Check absolute dot product to see if they are practically the same plane
            if abs(np.dot(n, base_n)) > 0.95: 
                cluster.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
            
    # 3. Process clusters and distribute multi-component parallel normals
    final_normals = [None] * len(std_normals)
    for cluster in clusters:
        m = len(cluster)
        base_normal = std_normals[cluster[0]]
        
        # Use our distribution function to handle same-normal collisions
        distributed = distribute_normals(base_normal, m)
        
        for idx, new_n in zip(cluster, distributed):
            final_normals[idx] = new_n.tolist()
            
    # Add the computed and resolved normals directly to the dict
    cube_data['normals'] = final_normals
    return cube_data

def batched(iterable, n):
    """Batch data into lists of length n. Keeps IPC queue memory bounded."""
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch

def process_mesh_intersections(mesh, res, cube_dicts, output_dir, batch_size=5000, num_workers=None):
    """
    Main entry point. Iterates through the list of small cubes, extracts edge 
    loops, computes optimal normals, checks for conflicts, and updates the list.
    Accelerated using multiprocessing and tracked with tqdm.
    """
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1) # Leave one core free
        
    # Extract vertices and faces defensively
    vertices, mesh_faces = None, None
    if mesh is not None:
        if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
            vertices = mesh.vertices
            mesh_faces = mesh.faces
        elif isinstance(mesh, tuple) and len(mesh) == 2:
            vertices, mesh_faces = mesh
            
    total_items = len(cube_dicts) if hasattr(cube_dicts, '__len__') else None
    processed_dicts = []
    
    # Bind the mesh context efficiently
    worker_func = partial(process_single_cube, vertices=vertices, mesh_faces=mesh_faces, res=res)
    
    # Use batched processing to ensure memory usage stays constant 
    with mp.Pool(processes=num_workers) as pool:
        with tqdm(total=total_items, desc="Processing Intersections") as pbar:
            for batch in batched(cube_dicts, batch_size):
                # Process the batch synchronously so we don't build a huge backlog queue
                batch_results = pool.map(worker_func, batch)
                processed_dicts.extend(batch_results)
                pbar.update(len(batch_results))
                
    save_pickle(f'{output_dir}/face_registers.pkl', processed_dicts)
    return processed_dicts

def visualize_centers(cube_dicts):
    """
    Visualizes the computed center points by creating and returning a trimesh PointCloud.
    """
    points = []
    for d in cube_dicts:
        cp = d.get('center_point')
        if cp is not None:
            points.append(cp)
            
    if not points:
        # Return an empty point cloud if no points exist
        return trimesh.PointCloud(vertices=np.empty((0, 3)))
        
    return trimesh.PointCloud(vertices=np.array(points))

def visualize_loops(res, cube_data, pipe_radius=None):
    """
    Visualizes loops from subdivided cube data by rendering them as 3D pipes.
    
    Args:
        res (int): Resolution of the grid. The unit cube is divided into res**3 small cubes.
        cube_data (list of dict): Data describing the state and loops of each cube.
        pipe_radius (float, optional): Radius of the rendered pipe. Defaults to 2% of a small cube's edge.
        
    Returns:
        trimesh.Trimesh: A single concatenated mesh containing all the loops.
    """
    meshes = []
    
    # Base edge length of a single small cube
    s = 1.0 / res
    
    if pipe_radius is None:
        pipe_radius = s * 0.02

    # Hardcoded Edge Mapping (0 to 17) mapping to vertex indices
    edge_to_verts = [
        # Original 12 cube edges
        (0, 1), (1, 2), (2, 3), (3, 0),       # 0-3: Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),       # 4-7: Top face
        (0, 4), (1, 5), (2, 6), (3, 7),       # 8-11: Vertical edges
        
        # Added 6 diagonal edges
        (0, 2),                               # 12: Bottom Diagonal
        (4, 6),                               # 13: Top Diagonal
        (1, 4),                               # 14: Front Diagonal
        (1, 6),                               # 15: Right Diagonal
        (2, 7),                               # 16: Back Diagonal
        (0, 7)                                # 17: Left Diagonal
    ]

    # Pre-defined nice colors for distinguishable loops (RGBA)
    color_palette = [
        [214, 39, 40, 255],   # Red
        [31, 119, 180, 255],  # Blue
        [44, 160, 44, 255],   # Green
        [255, 127, 14, 255],  # Orange
        [148, 103, 189, 255], # Purple
        [140, 86, 75, 255],   # Brown
        [227, 119, 194, 255], # Pink
        [23, 190, 207, 255]   # Cyan
    ]

    for data in cube_data:
        cx, cy, cz = data['cube_indices']
        loops = data.get('loops', [])
        
        if not loops:
            continue

        # Define the exact bounds of this specific small cube
        x0, y0, z0 = cx * s, cy * s, cz * s
        x1, y1, z1 = (cx + 1) * s, (cy + 1) * s, (cz + 1) * s

        # Hardcoded Vertices Mapping (0 to 7)
        verts = np.array([
            [x0, y0, z0], # 0: front-left-bottom
            [x1, y0, z0], # 1: front-right-bottom
            [x1, y1, z0], # 2: back-right-bottom
            [x0, y1, z0], # 3: back-left-bottom
            [x0, y0, z1], # 4: front-left-top
            [x1, y0, z1], # 5: front-right-top
            [x1, y1, z1], # 6: back-right-top
            [x0, y1, z1]  # 7: back-left-top
        ])

        # Step 1: Count edge usage to handle shifting if multiple loops share an edge
        edge_to_loop_indices = collections.defaultdict(list)
        for loop_idx, loop in enumerate(loops):
            for e in loop:
                edge_to_loop_indices[e].append(loop_idx)

        # Step 2: Compute shifted midpoint coordinates for each loop using the edge
        point_map = collections.defaultdict(list)
        for e, l_indices in edge_to_loop_indices.items():
            n_users = len(l_indices)
            
            # Get geometry of the edge
            v0 = verts[edge_to_verts[e][0]]
            v1 = verts[edge_to_verts[e][1]]
            mid = (v0 + v1) / 2.0
            vec = v1 - v0
            L = np.linalg.norm(vec)
            dir_vec = vec / L if L > 0 else np.array([1, 0, 0])

            # Calculate safe spacing bounds (up to 50% of the total edge length)
            max_span = 0.5 * L
            spacing = max_span / max(n_users, 2)

            for i, l_idx in enumerate(l_indices):
                # Shift points symmetrically along the edge's direction
                shift = (i - (n_users - 1) / 2.0) * spacing
                pt = mid + shift * dir_vec
                point_map[(e, l_idx)].append(pt)

        # Step 3: Construct the 3D meshes for the loops
        for loop_idx, loop in enumerate(loops):
            if len(loop) < 2:
                continue

            loop_points = []
            used_counts = collections.defaultdict(int)
            
            # Resolve points in case the same loop visits the same edge multiple times
            for e in loop:
                idx = used_counts[e]
                pt = point_map[(e, loop_idx)][idx]
                loop_points.append(pt)
                used_counts[e] += 1

            # Loops are closed, link the last node back to the first
            loop_points.append(loop_points[0])
            color = color_palette[loop_idx % len(color_palette)]

            # Generate cylinders and joint spheres
            for i in range(len(loop_points) - 1):
                p0 = loop_points[i]
                p1 = loop_points[i+1]

                # Prevent generating zero-length cylinders
                if np.linalg.norm(p1 - p0) < 1e-6:
                    continue

                # Pipe segment
                cyl = trimesh.creation.cylinder(radius=pipe_radius, segment=[p0, p1])
                cyl.visual.face_colors = color
                meshes.append(cyl)

                # Joint sphere (appending at p0 for every segment guarantees all joints are covered)
                sph = trimesh.creation.icosphere(radius=pipe_radius, subdivisions=2)
                sph.apply_translation(p0)
                sph.visual.face_colors = color
                meshes.append(sph)

    # Concatenate all generated primitives into one continuous mesh object
    if meshes:
        return trimesh.util.concatenate(meshes)
    else:
        return trimesh.Trimesh()

def visualize_single_cube(cube_data, mesh, res=1):
    """
    Visualizes a single cube's data: small cube wireframe, center point sphere, 
    normal arrows, and the intersected local faces.
    Returns a single concatenated trimesh.Trimesh object.
    """
    import trimesh.creation
    meshes = []
    
    SCALE = 1.0
    
    cube_indices = np.array(cube_data.get('cube_indices'))
    cube_size = (1.0 / res) * SCALE
    global_cube_min = cube_indices * cube_size
    
    # 1. Wireframe of the small cube (using thin cylinders for edges)
    # Define the 8 corners of the local cube
    corners = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ]) * cube_size + global_cube_min
    
    # 12 boundary edges of a cube
    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]
    edge_radius = 0.01 * cube_size
    for u, v in edges:
        segment = corners[v] - corners[u]
        length = np.linalg.norm(segment)
        cyl = trimesh.creation.cylinder(radius=edge_radius, height=length)
        
        # Align Z axis to segment
        transform = trimesh.geometry.align_vectors([0, 0, 1], segment / length)
        # Translate to midpoint
        transform[:3, 3] = (corners[u] + corners[v]) / 2.0
        cyl.apply_transform(transform)
        cyl.visual.face_colors = [150, 150, 150, 255] # Gray wireframe
        meshes.append(cyl)
        
    # 2. Sphere for the center point
    cp = cube_data.get('center_point')
    if cp is not None:
        cp = np.array(cp) * SCALE
        sphere = trimesh.creation.icosphere(radius=0.04 * cube_size)
        sphere.apply_translation(cp)
        sphere.visual.face_colors = [255, 0, 0, 255] # Red center
        meshes.append(sphere)
        
        # 3. Arrows for normals
        normals = cube_data.get('normals', [])
        arrow_length = 0.4 * cube_size
        for n in normals:
            n = np.array(n)
            if np.linalg.norm(n) < 1e-6: continue
            n = n / np.linalg.norm(n)
            
            # Shaft
            shaft = trimesh.creation.cylinder(radius=0.01 * cube_size, height=arrow_length * 0.75)
            shaft_transform = trimesh.geometry.align_vectors([0, 0, 1], n)
            shaft_transform[:3, 3] = cp + n * (arrow_length * 0.375) # Midpoint of shaft
            shaft.apply_transform(shaft_transform)
            shaft.visual.face_colors = [0, 255, 0, 255] # Green normal
            
            # Head (cone)
            head = trimesh.creation.cone(radius=0.03 * cube_size, height=arrow_length * 0.25)
            head_transform = trimesh.geometry.align_vectors([0, 0, 1], n)
            # Base of the cone is at the end of the shaft
            head_transform[:3, 3] = cp + n * (arrow_length * 0.75) 
            head.apply_transform(head_transform)
            head.visual.face_colors = [0, 255, 0, 255]
            
            meshes.append(shaft)
            meshes.append(head)
            
    # 4. Include the faces listed in the dictionary
    face_indices = cube_data.get('face_indices', [])
    if len(face_indices) > 0 and mesh is not None:
        vertices, mesh_faces = None, None
        if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
            vertices = mesh.vertices
            mesh_faces = mesh.faces
        elif isinstance(mesh, tuple) and len(mesh) == 2:
            vertices, mesh_faces = mesh
            
        if vertices is not None and mesh_faces is not None:
            # Get the relevant faces
            local_faces = mesh_faces[face_indices]
            # Construct a submesh and scale the vertices
            scaled_vertices = vertices * SCALE
            submesh = trimesh.Trimesh(vertices=scaled_vertices, faces=local_faces, process=True)
            submesh.visual.face_colors = [0, 150, 255, 200] # Blue-ish faces
            meshes.append(submesh)
    
    loop_mesh = visualize_loops(res, [cube_data])
    meshes.append(loop_mesh)

    for mesh_point in cube_data['component_points']:
        mesh_point_mesh = trimesh.creation.icosphere(radius=0.04 * cube_size)
        mesh_point_mesh.apply_translation(mesh_point)
        mesh_point_mesh.visual.face_colors = [255, 255, 0, 255] # Yellow mesh point
        meshes.append(mesh_point_mesh)

    if not meshes:
        return trimesh.Trimesh()
        
    return trimesh.util.concatenate(meshes)


def feature_point(mesh, res, loop_points, output_dir, debug=False):
    results = process_mesh_intersections(mesh, res, loop_points, output_dir)
    os.makedirs(f'{output_dir}/debug', exist_ok=True)
    if debug:
        for i, result in enumerate(results):
            visualize_single_cube(result, mesh, res).export(f'{output_dir}/debug/loop_points_{result["cube_indices"]}.ply')
    return results

if __name__ == "__main__":

    load_dir = 'tmp/test_rep'
    output_dir = 'tmp/test_rep'
    res = 1024

    mesh = trimesh.load(f'{load_dir}/norm_mesh.ply')
    results = load_pickle(f'{load_dir}/edge_features.pkl')

    results = process_mesh_intersections(mesh, res, results, output_dir)
    # breakpoint()
    save_pickle(f'{output_dir}/loop_points.pkl', results)

    visualize_centers(results).export(f'{output_dir}/loop_points.ply')

    breakpoint()