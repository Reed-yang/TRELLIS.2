import numpy as np
import trimesh
import networkx as nx
from collections import defaultdict
import multiprocessing as mp
from tqdm import tqdm
from utils import save_pickle
import os

def clip_polygon_against_plane(polygon, plane_normal, plane_point):
    """
    Clips a 3D convex polygon against a 3D plane using the Sutherland-Hodgman algorithm.
    """
    if not polygon:
        return []
        
    clipped = []
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        
        d1 = np.dot(plane_normal, p1 - plane_point)
        d2 = np.dot(plane_normal, p2 - plane_point)
        
        if d1 >= 0:
            clipped.append(p1)
        
        if (d1 >= 0 and d2 < 0) or (d1 < 0 and d2 >= 0):
            # Calculate intersection point via linear interpolation
            t = d1 / (d1 - d2)
            p_intersect = p1 + t * (p2 - p1)
            clipped.append(p_intersect)
            
    return clipped

def get_local_connected_components(face_indices, mesh):
    """
    Identifies connected components from a subset of face indices.
    Two faces are connected if they share an edge.
    """
    faces_vertices = mesh.faces[face_indices]
    edge_to_faces = defaultdict(list)
    
    # Map edges to the faces that share them
    for idx, f_verts in zip(face_indices, faces_vertices):
        # Sort vertices so edges (A, B) and (B, A) hash to the same tuple
        e1 = tuple(sorted((f_verts[0], f_verts[1])))
        e2 = tuple(sorted((f_verts[1], f_verts[2])))
        e3 = tuple(sorted((f_verts[2], f_verts[0])))
        
        edge_to_faces[e1].append(idx)
        edge_to_faces[e2].append(idx)
        edge_to_faces[e3].append(idx)
        
    # Build an adjacency graph
    G = nx.Graph()
    G.add_nodes_from(face_indices)
    
    for f_list in edge_to_faces.values():
        if len(f_list) > 1:
            # Connect all faces sharing this edge (handles standard and non-manifold edges)
            for i in range(len(f_list)):
                for j in range(i + 1, len(f_list)):
                    G.add_edge(f_list[i], f_list[j])
                    
    # Yield the connected components as lists of face indices
    return [list(comp) for comp in nx.connected_components(G)]


# ==========================================
# Multiprocessing Worker Definitions
# ==========================================

# Globals to hold read-only data for each process. 
# This prevents the memory/IPC overhead of pickling the mesh for every single task.
_worker_mesh = None
_worker_res = None

def _init_worker(mesh, res):
    """Initialize worker processes with shared read-only mesh and resolution."""
    global _worker_mesh, _worker_res
    _worker_mesh = mesh
    _worker_res = res

def _process_cube_worker(cube_dict):
    """
    Worker function that computes components and centers for a single cube.
    """
    cube_indices = cube_dict['cube_indices']
    face_indices = cube_dict['face_indices']
    
    # We return a new copied dict to prevent mutating the original input reference directly
    res_dict = cube_dict.copy()
    
    if not face_indices:
        res_dict['component_points'] = []
        return res_dict
        
    # 1. Determine local bounding box bounds
    min_bound = np.array(cube_indices) / _worker_res
    max_bound = (np.array(cube_indices) + 1.0) / _worker_res
    
    # Planes defined by (normal, point on plane) facing INWARD to the keeping region
    planes = [
        (np.array([ 1,  0,  0]), min_bound),
        (np.array([-1,  0,  0]), max_bound),
        (np.array([ 0,  1,  0]), min_bound),
        (np.array([ 0, -1,  0]), max_bound),
        (np.array([ 0,  0,  1]), min_bound),
        (np.array([ 0,  0, -1]), max_bound)
    ]
    
    # 2. Extract topological connected components
    components = get_local_connected_components(face_indices, _worker_mesh)
    component_points = []
    
    for comp_faces in components:
        comp_faces_verts = _worker_mesh.faces[comp_faces]
        
        total_area = 0.0
        weighted_centroid = np.zeros(3)
        all_clipped_polygons = []
        
        # 3. Exactly clip each triangle to the cube's bounding box
        for f_verts in comp_faces_verts:
            poly = [
                _worker_mesh.vertices[f_verts[0]], 
                _worker_mesh.vertices[f_verts[1]], 
                _worker_mesh.vertices[f_verts[2]]
            ]
            
            # Successively clip against the 6 AABB planes
            for normal, point in planes:
                poly = clip_polygon_against_plane(poly, normal, point)
                if len(poly) < 3:
                    break
                    
            # 4. Triangulate the resulting convex polygon to find area/centroid
            if len(poly) >= 3:
                p0 = poly[0]
                for i in range(1, len(poly) - 1):
                    p1 = poly[i]
                    p2 = poly[i+1]
                    
                    cross = np.cross(p1 - p0, p2 - p0)
                    area = 0.5 * np.linalg.norm(cross)
                    centroid = (p0 + p1 + p2) / 3.0
                    
                    total_area += area
                    weighted_centroid += centroid * area
                    
                all_clipped_polygons.append(poly)
        
        # 5. Determine the actual on-mesh point
        if total_area > 1e-12:
            # The area-weighted center of mass of the clipped geometry
            target_center = weighted_centroid / total_area
            
            # Convert the polygon soup back into a localized temporary mesh
            clipped_vertices = []
            clipped_faces = []
            
            for poly in all_clipped_polygons:
                idx_start = len(clipped_vertices)
                clipped_vertices.extend(poly)
                for i in range(1, len(poly) - 1):
                    clipped_faces.append([idx_start, idx_start + i, idx_start + i + 1])
                    
            # Snap the floating centroid precisely back to the closest mesh surface within the cube
            clipped_mesh = trimesh.Trimesh(vertices=clipped_vertices, faces=clipped_faces, process=False)
            closest, _, _ = clipped_mesh.nearest.on_surface([target_center])
            center_pt = closest[0]
            
        else:
            # Fallback: if the geometry inside the box vanishes strictly due to numerical 
            # grazing boundaries, snap the geometric center of the box to the component.
            cube_center = (min_bound + max_bound) / 2.0
            comp_mesh = _worker_mesh.submesh([comp_faces], append=True)
            closest, _, _ = comp_mesh.nearest.on_surface([cube_center])
            center_pt = closest[0]
        
        component_points.append(center_pt.tolist())
        
    res_dict['component_points'] = component_points
    return res_dict


def compute_component_centers(res, mesh, cube_data_list, batch_size=1000, num_processes=None):
    """
    Computes a representative center point on each connected component of mesh faces 
    constrained within spatial resolution sub-cubes.
    
    :param res: Integer resolution of the unit cube grid.
    :param mesh: trimesh.Trimesh object.
    :param cube_data_list: List of dictionaries containing cube and face info.
    :param batch_size: Number of items to queue per worker at a time (keeps memory usage flat).
    :param num_processes: Number of CPU cores to utilize (defaults to all available).
    :return: Updated list of dictionaries with 'component_points'.
    """
    results = []
    
    # Pool setup with initializer avoids pickling the large `mesh` for each queue item.
    with mp.Pool(processes=num_processes, initializer=_init_worker, initargs=(mesh, res)) as pool:
        
        # imap uses internal chunking based on `batch_size` to ensure IPC memory 
        # stays constant and doesn't load millions of tasks at once into the queue.
        task_iterator = pool.imap(_process_cube_worker, cube_data_list, chunksize=batch_size)
        
        # Wrap the lazy iterator with tqdm to track progress efficiently
        for updated_cube_dict in tqdm(task_iterator, total=len(cube_data_list), desc="Processing Sub-Cubes"):
            results.append(updated_cube_dict)
            
    return results

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


def feature_point(mesh, res, face_registers, output_dir, debug=False):
    results = compute_component_centers(res, mesh, face_registers)
    save_pickle(f'{output_dir}/face_registers.pkl', results)
    if debug:
        for i, result in enumerate(results):
            visualize_single_cube(result, mesh, res).export(f'{output_dir}/debug/loop_points_{result["cube_indices"]}.ply')
    return results

# ==========================================
# Example usage block (Synthetic testing)
# ==========================================
if __name__ == "__main__":
    # Create a normalized spherical mesh
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.3)
    mesh.apply_translation([0.5, 0.5, 0.5]) # Center it in the [0, 1]^3 domain
    
    # Simulate the input
    res = 2  # The space is broken into 2x2x2 = 8 smaller cubes
    
    # Grab arbitrary faces that we know intersect with the octant closest to the origin
    example_faces = [i for i, centroid in enumerate(mesh.triangles_center) if np.all(centroid < 0.5)]
    
    # We will simulate a bunch of cubes to show off tqdm and multiprocessing
    mock_input_data = [
        {
            'cube_indices': (0, 0, 0), 
            'face_indices': example_faces, 
            'num_components': 1  # For reference only; the function dynamically calculates this
        }
    ] * 500  # Multiplied just to demonstrate progress bar filling up
    
    print(f"Initial Data created with {len(mock_input_data)} total tasks to process.")
    
    # Run the batched, multiprocessed function
    # batch_size limits how many items are actively enqueued at a time
    updated_data = compute_component_centers(res, mesh, mock_input_data, batch_size=50)
    
    print("\nResulting Points (from first task):", updated_data[0]['component_points'])