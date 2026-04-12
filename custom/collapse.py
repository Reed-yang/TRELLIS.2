"""
Cube Topology and Intersection Reformer
=======================================
This module defines the strict geometric and index mapping for a triangulated 
unit cube and provides a utility to reform intersection data into a structured 
point-loop format.
"""


import itertools
import os
import shutil
import pickle
import numpy as np
import trimesh
import collections
import multiprocessing as mp
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment



# =============================================================================
# THE GEOMETRY AND INDEX MAPPING
# =============================================================================

# Vertices (0 to 7) mapped to their relative (x, y, z) coordinates 
# assuming a standard unit cube structure for reference.
VERTICES = {
    0: "Bottom front-left",
    1: "Bottom front-right",
    2: "Bottom back-right",
    3: "Bottom back-left",
    4: "Top front-left",
    5: "Top front-right",
    6: "Top back-right",
    7: "Top back-left"
}

# Edges (0 to 17) mapped to (vertex1, vertex2)
EDGES = {
    # Original 12 cube edges
    0: (0, 1),   # Bottom Front
    1: (1, 2),   # Bottom Right
    2: (2, 3),   # Bottom Back
    3: (3, 0),   # Bottom Left
    4: (4, 5),   # Top Front
    5: (5, 6),   # Top Right
    6: (6, 7),   # Top Back
    7: (7, 4),   # Top Left
    8: (0, 4),   # Front-Left Vertical
    9: (1, 5),   # Front-Right Vertical
    10: (2, 6),  # Back-Right Vertical
    11: (3, 7),  # Back-Left Vertical
    
    # Added 6 diagonal edges (one per square face)
    12: (0, 2),  # Bottom Diagonal
    13: (4, 6),  # Top Diagonal
    14: (1, 4),  # Front Diagonal
    15: (1, 6),  # Right Diagonal
    16: (2, 7),  # Back Diagonal
    17: (0, 7)   # Left Diagonal
}

# Triangles (0 to 11) defined by a tuple of 3 edge indices
TRIANGLES = {
    0: (0, 1, 12),   1: (2, 3, 12),    # Bottom face halves
    2: (4, 5, 13),   3: (6, 7, 13),    # Top face halves
    4: (0, 8, 14),   5: (4, 9, 14),    # Front face halves
    6: (1, 10, 15),  7: (5, 9, 15),    # Right face halves
    8: (2, 11, 16),  9: (6, 10, 16),   # Back face halves
    10: (3, 11, 17), 11: (7, 8, 17)    # Left face halves
}

# =============================================================================
# REFORM FUNCTION
# =============================================================================

def reform_intersection_data(cube_data_list):
    """
    Reforms a list of cube intersection dictionaries.
    
    For each dictionary, it extracts the 'mesh_points' and 'loops', pairs them up,
    and creates a new sub-dictionary for each pair formatted as:
    {'mesh_point': (x, y, z), 'loop': [edge1, edge2, ...]}
    
    These new dictionaries are added to the original dictionary under the 
    key 'structured_loops'.
    
    Args:
        cube_data_list (list of dict): The input list of cube intersection data.
        
    Returns:
        list of dict: The modified list of dictionaries.
    """
    for data in cube_data_list:
        structured_loops = []
        
        # Safely extract lists (guaranteed by prompt to be length num_loops)
        loops = data.get('loops', [])
        # mesh_points = data.get('mesh_points', [])
        mesh_points = data.get('component_points', [])
        num_loops = data.get('num_loops', 0)

        if num_loops != len(mesh_points):
            structured_loops.append({})
            continue
            
        # Create paired dictionary for each topological component (loop)
        for i in range(num_loops):
            # Convert the mesh point coordinate list to a tuple (x, y, z)
            pt_tuple = tuple(mesh_points[i])
            loop_edges = loops[i]
            
            # Construct the newly requested sub-dictionary format
            structured_loop_dict = {
                'mesh_point': pt_tuple,
                'loop': loop_edges
            }
            
            structured_loops.append(structured_loop_dict)
            
        # Add the new dicts to the original dict
        data['structured_loops'] = structured_loops
        
    return cube_data_list


def extract_original_cube_edges(cube_data_list):
    """
    Iterates through the reformed cube data and extracts the original 
    cube edges (indices 0-11) from each structured loop.
    
    The extracted edges are added to the loop's dictionary under the 
    key 'original_cube_edges'.
    
    Args:
        cube_data_list (list of dict): The list of reformed intersection data.
        
    Returns:
        list of dict: The modified list of dictionaries.
    """
    for data in cube_data_list:
        for loop_data in data.get('structured_loops', []):
            # Filter the loop for original cube edges (indices 0 to 11 inclusive)
            original_edges = [edge for edge in loop_data.get('loop', []) if edge < 12]
            loop_data['original_cube_edges'] = original_edges
            
    return cube_data_list


def get_global_edge(ix, iy, iz, local_edge):
    """
    Maps a local edge index (0-11) of a cube at (ix, iy, iz) 
    to a unique global edge identifier.
    """
    if local_edge == 0: return ('x', ix, iy, iz)         # (0, 1) Bottom Front
    elif local_edge == 1: return ('z', ix+1, iy, iz)       # (1, 2) Bottom Right
    elif local_edge == 2: return ('x', ix, iy, iz+1)       # (2, 3) Bottom Back
    elif local_edge == 3: return ('z', ix, iy, iz)         # (3, 0) Bottom Left
    elif local_edge == 4: return ('x', ix, iy+1, iz)       # (4, 5) Top Front
    elif local_edge == 5: return ('z', ix+1, iy+1, iz)     # (5, 6) Top Right
    elif local_edge == 6: return ('x', ix, iy+1, iz+1)     # (6, 7) Top Back
    elif local_edge == 7: return ('z', ix, iy+1, iz)       # (7, 4) Top Left
    elif local_edge == 8: return ('y', ix, iy, iz)         # (0, 4) Front-Left Vertical
    elif local_edge == 9: return ('y', ix+1, iy, iz)       # (1, 5) Front-Right Vertical
    elif local_edge == 10: return ('y', ix+1, iy, iz+1)    # (2, 6) Back-Right Vertical
    elif local_edge == 11: return ('y', ix, iy, iz+1)      # (3, 7) Back-Left Vertical
    return None

def get_surrounding_cubes(g_edge):
    """
    Given a global edge, returns the 4 neighboring cube coordinates 
    in a cyclic order to ensure a valid, non-self-intersecting quad.
    """
    axis, nx, ny, nz = g_edge
    if axis == 'x':
        return [(nx, ny, nz), (nx, ny-1, nz), (nx, ny-1, nz-1), (nx, ny, nz-1)]
    elif axis == 'y':
        return [(nx, ny, nz), (nx-1, ny, nz), (nx-1, ny, nz-1), (nx, ny, nz-1)]
    elif axis == 'z':
        return [(nx, ny, nz), (nx-1, ny, nz), (nx-1, ny-1, nz), (nx, ny-1, nz)]

# def process_shared_edge_geometry(grid_2x2_lists):
#     """
#     Processes a 2x2 grid of cubes (4 lists of dicts) sharing an edge to resolve 
#     multi-cube intersection points and create connecting triangles.
    
#     Args:
#         resolution (int): Grid resolution (used to calculate physical coordinates).
#         grid_2x2_lists (list of lists of dict): 4 lists of dictionaries representing the 4 cubes.
        
#     Returns:
#         tuple: (new_vertices, triangles)
#     """
#     # 1. Deduce the base cube index and edge direction
#     valid_indices = []
#     for cube_list in grid_2x2_lists:
#         if cube_list and 'cube_indices' in cube_list[0]:
#             valid_indices.append(cube_list[0]['cube_indices'])
            
#     if not valid_indices:
#         return [], []
        
#     min_idx = [min(idx[i] for idx in valid_indices) for i in range(3)]
#     max_idx = [max(idx[i] for idx in valid_indices) for i in range(3)]
    
#     # The axis where the minimum index equals the maximum index is the shared edge axis
#     edge_axis = -1
#     for i in range(3):
#         if min_idx[i] == max_idx[i]:
#             edge_axis = i
#             break
            
#     if edge_axis == -1:
#         # Fallback if topology is malformed, guess Z
#         edge_axis = 2 

#     # Determine local edge for a specific relative position mapping to your strict constants
#     def get_local_edge(dx, dy, dz):
#         if edge_axis == 2:    # Z-aligned
#             if dx == 0 and dy == 0: return 10 # Back-Right
#             if dx == 1 and dy == 0: return 11 # Back-Left
#             if dx == 0 and dy == 1: return 9  # Front-Right
#             if dx == 1 and dy == 1: return 8  # Front-Left
#         elif edge_axis == 0:  # X-aligned
#             if dy == 0 and dz == 0: return 6  # Top-Back
#             if dy == 1 and dz == 0: return 4  # Top-Front
#             if dy == 0 and dz == 1: return 2  # Bottom-Back
#             if dy == 1 and dz == 1: return 0  # Bottom-Front
#         elif edge_axis == 1:  # Y-aligned
#             if dx == 0 and dz == 0: return 5  # Top-Right
#             if dx == 1 and dz == 0: return 7  # Top-Left
#             if dx == 0 and dz == 1: return 1  # Bottom-Right
#             if dx == 1 and dz == 1: return 3  # Bottom-Left
#         return -1

#     # 2. Extract points registered to the shared edge
#     cube_points = {}
#     for cube_list in grid_2x2_lists:
#         if not cube_list:
#             continue
#         idx = cube_list[0]['cube_indices']
#         dx = idx[0] - min_idx[0]
#         dy = idx[1] - min_idx[1]
#         dz = idx[2] - min_idx[2]
        
#         local_edge = get_local_edge(dx, dy, dz)
        
#         pts = []
#         for d in cube_list:
#             for loop in d.get('structured_loops', []):
#                 if local_edge in loop.get('original_cube_edges', []):
#                     pts.append(loop['mesh_point'])
                    
#         # Sort points by the edge axis to align multiple intersections (ranks)
#         pts.sort(key=lambda pt: pt[edge_axis])
        
#         # Store points keyed by relative position
#         if edge_axis == 2:   key = (dx, dy)
#         elif edge_axis == 0: key = (dy, dz)
#         elif edge_axis == 1: key = (dx, dz)
#         cube_points[key] = pts

#     # 3. Create vertices and triangles per rank
#     new_vertices = []
#     triangles = []
    
#     max_rank = 0
#     if cube_points:
#         max_rank = max(len(pts) for pts in cube_points.values())
        
#     # Standard neighbor cycle for up to 4 connecting triangles
#     neighbors = [(0, 0), (1, 0), (1, 1), (0, 1)]
    
#     for rank in range(max_rank):
#         rank_pts = []
#         pt_map = {}
        
#         for uv in neighbors:
#             if uv in cube_points and rank < len(cube_points[uv]):
#                 pt = cube_points[uv][rank]
#                 rank_pts.append(pt)
#                 pt_map[uv] = pt
                
#         if not rank_pts:
#             continue
            
#         # Average the coordinates of the valid points for this rank
#         avg_x = sum(p[0] for p in rank_pts) / len(rank_pts)
#         avg_y = sum(p[1] for p in rank_pts) / len(rank_pts)
#         avg_z = sum(p[2] for p in rank_pts) / len(rank_pts)
        
#         proj_pt = (avg_x, avg_y, avg_z)
        
#         # # Project to the shared edge by locking the non-varying axes based on cube width (1.0/resolution)
#         # if edge_axis == 0:
#         #     proj_pt = (avg_x, (min_idx[1] + 1) / resolution, (min_idx[2] + 1) / resolution)
#         # elif edge_axis == 1:
#         #     proj_pt = ((min_idx[0] + 1) / resolution, avg_y, (min_idx[2] + 1) / resolution)
#         # else: # edge_axis == 2
#         #     proj_pt = ((min_idx[0] + 1) / resolution, (min_idx[1] + 1) / resolution, avg_z)
            
#         new_vertices.append(proj_pt)
        
#         # # Generate up to 4 triangles using adjacent neighboring points
#         for i in range(4):
#             uv1 = neighbors[i]
#             uv2 = neighbors[(i + 1) % 4]
#             if uv1 in pt_map and uv2 in pt_map:
#                 triangles.append((proj_pt, pt_map[uv1], pt_map[uv2]))
        
#         # add triangle only if 4 points are present
#         # if len(rank_pts) == 4:
#         #     triangles.append((proj_pt, pt_map[neighbors[0]], pt_map[neighbors[1]]))
#         #     triangles.append((proj_pt, pt_map[neighbors[1]], pt_map[neighbors[2]]))
#         #     triangles.append((proj_pt, pt_map[neighbors[2]], pt_map[neighbors[3]]))
#         #     triangles.append((proj_pt, pt_map[neighbors[3]], pt_map[neighbors[0]]))



#     return new_vertices, triangles


def mark_exception(cube_data_list):
    """
    Marks the cubes that are exceptions to the collapse point algorithm.
    """
    return [{'exception': True, 'cube_indices': cube_data['cube_indices'], 'sorted_loops': [{'component_point': cube_data['component_points'][0]}]} for cube_data in cube_data_list]


# def process_shared_edge_geometry(grid_2x2_lists):
#     """
#     Processes a 2x2 grid of cubes (4 lists of dicts) sharing an edge to resolve 
#     multi-cube intersection points and create connecting triangles.
#     Vertices are strictly connected based on pre-calculated topological ranks.
    
#     Args:
#         grid_2x2_lists (list of lists of dict): 4 lists of dictionaries representing the 4 cubes.
        
#     Returns:
#         tuple: (new_vertices, triangles)
#     """
#     # 1. Deduce the base cube index and edge direction
#     valid_indices = []
#     for cube_list in grid_2x2_lists:
#         if cube_list and 'cube_indices' in cube_list[0]:
#             valid_indices.append(cube_list[0]['cube_indices'])
            
#     if not valid_indices:
#         return [], []
        
#     min_idx = [min(idx[i] for idx in valid_indices) for i in range(3)]
#     max_idx = [max(idx[i] for idx in valid_indices) for i in range(3)]
    
#     # The axis where the minimum index equals the maximum index is the shared edge axis
#     edge_axis = -1
#     for i in range(3):
#         if min_idx[i] == max_idx[i]:
#             edge_axis = i
#             break
            
#     if edge_axis == -1:
#         # Fallback if topology is malformed, guess Z
#         edge_axis = 2 

#     # Determine local edge for a specific relative position mapping to your strict constants
#     def get_local_edge(dx, dy, dz):
#         if edge_axis == 2:    # Z-aligned
#             if dx == 0 and dy == 0: return 10 # Back-Right
#             if dx == 1 and dy == 0: return 11 # Back-Left
#             if dx == 0 and dy == 1: return 9  # Front-Right
#             if dx == 1 and dy == 1: return 8  # Front-Left
#         elif edge_axis == 0:  # X-aligned
#             if dy == 0 and dz == 0: return 6  # Top-Back
#             if dy == 1 and dz == 0: return 4  # Top-Front
#             if dy == 0 and dz == 1: return 2  # Bottom-Back
#             if dy == 1 and dz == 1: return 0  # Bottom-Front
#         elif edge_axis == 1:  # Y-aligned
#             if dx == 0 and dz == 0: return 5  # Top-Right
#             if dx == 1 and dz == 0: return 7  # Top-Left
#             if dx == 0 and dz == 1: return 1  # Bottom-Right
#             if dx == 1 and dz == 1: return 3  # Bottom-Left
#         return -1

#     # 2. Extract points and organize them by their pre-calculated rank
#     # Structure: points_by_rank[rank][relative_uv_tuple] = point
#     points_by_rank = collections.defaultdict(dict)
    
#     for cube_list in grid_2x2_lists:
#         if not cube_list:
#             continue
#         idx = cube_list[0]['cube_indices']
#         dx = idx[0] - min_idx[0]
#         dy = idx[1] - min_idx[1]
#         dz = idx[2] - min_idx[2]
        
#         local_edge = get_local_edge(dx, dy, dz)
#         if local_edge == -1:
#             continue
            
#         # Determine relative 2D coordinate for quad corners around the edge
#         if edge_axis == 2:   uv = (dx, dy)
#         elif edge_axis == 0: uv = (dy, dz)
#         elif edge_axis == 1: uv = (dx, dz)
        
#         for d in cube_list:
#             for loop_data in d.get('sorted_loops', []):
#                 edges = loop_data.get('loop', [])
#                 ranks = loop_data.get('rank', [])
                
#                 # Check if this loop intersects the shared local edge
#                 # Use enumerate to capture cases where a single loop crosses the same edge multiple times
#                 for i, edge in enumerate(edges):
#                     if edge == local_edge:
#                         rank = ranks[i]
                        
#                         # Normalize rank orientation to the global positive axis
#                         # Edges 2, 6 (-X) and 3, 7 (-Y) go in the negative direction.
#                         if local_edge in [2, 6, 3, 7]:
#                             W = d.get('edge_weights', [0]*18)[local_edge]
#                             normalized_rank = (W - 1) - rank
#                         else:
#                             normalized_rank = rank
                            
#                         pt = loop_data.get('component_point')
#                         if pt is not None:
#                             points_by_rank[normalized_rank][uv] = pt

#     # 3. Create vertices and triangles per rank group
#     new_vertices = []
#     triangles = []
    
#     # Standard neighbor cycle for up to 4 connecting triangles
#     neighbors = [(0, 0), (1, 0), (1, 1), (0, 1)]
    
#     for rank, pt_map in sorted(points_by_rank.items()):
#         rank_pts = list(pt_map.values())
        
#         if not rank_pts:
#             continue
            
#         # Average the coordinates of the valid points matching this specific rank
#         avg_x = sum(p[0] for p in rank_pts) / len(rank_pts)
#         avg_y = sum(p[1] for p in rank_pts) / len(rank_pts)
#         avg_z = sum(p[2] for p in rank_pts) / len(rank_pts)
        
#         proj_pt = (avg_x, avg_y, avg_z)
#         new_vertices.append(proj_pt)
        
#         # # Generate up to 4 triangles using adjacent neighboring points for this rank
#         # for i in range(4):
#         #     uv1 = neighbors[i]
#         #     uv2 = neighbors[(i + 1) % 4]
#         #     if uv1 in pt_map and uv2 in pt_map:
#         #         triangles.append((proj_pt, pt_map[uv1], pt_map[uv2]))

#         # only when there are 4 points
#         if len(rank_pts) == 4:
#             triangles.append((proj_pt, pt_map[neighbors[0]], pt_map[neighbors[1]]))
#             triangles.append((proj_pt, pt_map[neighbors[1]], pt_map[neighbors[2]]))
#             triangles.append((proj_pt, pt_map[neighbors[2]], pt_map[neighbors[3]]))
#             triangles.append((proj_pt, pt_map[neighbors[3]], pt_map[neighbors[0]]))

#     return new_vertices, triangles


# def process_shared_edge_geometry(grid_2x2_lists):
#     """
#     Processes a 2x2 grid of cubes (4 lists of dicts) sharing an edge to resolve 
#     multi-cube intersection points and create connecting triangles.
#     Vertices are strictly connected based on pre-calculated topological ranks.
#     Exception cubes ignore loops and inject their first component point into 
#     any matching neighboring rank groups.
    
#     Args:
#         grid_2x2_lists (list of lists of dict): 4 lists of dictionaries representing the 4 cubes.
        
#     Returns:
#         tuple: (new_vertices, triangles)
#     """
#     # 1. Deduce the base cube index and edge direction
#     valid_indices = []
#     for cube_list in grid_2x2_lists:
#         if cube_list and 'cube_indices' in cube_list[0]:
#             valid_indices.append(cube_list[0]['cube_indices'])
            
#     if not valid_indices:
#         return [], []
        
#     min_idx = [min(idx[i] for idx in valid_indices) for i in range(3)]
#     max_idx = [max(idx[i] for idx in valid_indices) for i in range(3)]
    
#     # The axis where the minimum index equals the maximum index is the shared edge axis
#     edge_axis = -1
#     for i in range(3):
#         if min_idx[i] == max_idx[i]:
#             edge_axis = i
#             break
            
#     if edge_axis == -1:
#         # Fallback if topology is malformed, guess Z
#         edge_axis = 2 

#     # Determine local edge for a specific relative position mapping to strict constants
#     def get_local_edge(dx, dy, dz):
#         if edge_axis == 2:    # Z-aligned
#             if dx == 0 and dy == 0: return 10 # Back-Right
#             if dx == 1 and dy == 0: return 11 # Back-Left
#             if dx == 0 and dy == 1: return 9  # Front-Right
#             if dx == 1 and dy == 1: return 8  # Front-Left
#         elif edge_axis == 0:  # X-aligned
#             if dy == 0 and dz == 0: return 6  # Top-Back
#             if dy == 1 and dz == 0: return 4  # Top-Front
#             if dy == 0 and dz == 1: return 2  # Bottom-Back
#             if dy == 1 and dz == 1: return 0  # Bottom-Front
#         elif edge_axis == 1:  # Y-aligned
#             if dx == 0 and dz == 0: return 5  # Top-Right
#             if dx == 1 and dz == 0: return 7  # Top-Left
#             if dx == 0 and dz == 1: return 1  # Bottom-Right
#             if dx == 1 and dz == 1: return 3  # Bottom-Left
#         return -1

#     # 2. Extract points and organize them by their pre-calculated rank
#     # Structure: points_by_rank[rank][relative_uv_tuple] = point
#     points_by_rank = collections.defaultdict(dict)
    
#     # Keep track of exception points that need to act as wildcards (uv -> point)
#     exception_points = {}
    
#     for cube_list in grid_2x2_lists:
#         if not cube_list:
#             continue
#         idx = cube_list[0]['cube_indices']
#         dx = idx[0] - min_idx[0]
#         dy = idx[1] - min_idx[1]
#         dz = idx[2] - min_idx[2]
        
#         local_edge = get_local_edge(dx, dy, dz)
#         if local_edge == -1:
#             continue
            
#         # Determine relative 2D coordinate for quad corners around the edge
#         if edge_axis == 2:   uv = (dx, dy)
#         elif edge_axis == 0: uv = (dy, dz)
#         elif edge_axis == 1: uv = (dx, dz)
        
#         for d in cube_list:
#             # ---> NEW CHECK: Handle Exception Cubes <---
#             if d.get('exception') is True:
#                 pt = None
#                 # Safely attempt to extract the first component point
#                 if d.get('sorted_loops') and len(d['sorted_loops']) > 0:
#                     pt = d['sorted_loops'][0].get('component_point')
#                 elif d.get('component_points') and len(d['component_points']) > 0:
#                     pt = d['component_points'][0]
#                 elif d.get('component_point'):
#                     pt = d.get('component_point')
                
#                 if pt is not None:
#                     exception_points[uv] = pt
                
#                 # Continue early to ignore the regular loops in this exception cube
#                 continue
#             # -------------------------------------------
                
#             for loop_data in d.get('sorted_loops', []):
#                 edges = loop_data.get('loop', [])
#                 ranks = loop_data.get('rank', [])
                
#                 # Check if this loop intersects the shared local edge
#                 # Use enumerate to capture cases where a single loop crosses the same edge multiple times
#                 for i, edge in enumerate(edges):
#                     if edge == local_edge:
#                         rank = ranks[i]
                        
#                         # Normalize rank orientation to the global positive axis
#                         # Edges 2, 6 (-X) and 3, 7 (-Y) go in the negative direction.
#                         if local_edge in [2, 6, 3, 7]:
#                             W = d.get('edge_weights', [0]*18)[local_edge]
#                             normalized_rank = (W - 1) - rank
#                         else:
#                             normalized_rank = rank
                            
#                         pt = loop_data.get('component_point')
#                         if pt is not None:
#                             points_by_rank[normalized_rank][uv] = pt

#     # ---> DISTRIBUTE EXCEPTIONS: Act as wildcards for neighboring ranks <---
#     # This fulfills the goal of connecting to "any neighbors".
#     for rank, pt_map in points_by_rank.items():
#         for uv, exc_pt in exception_points.items():
#             if uv not in pt_map:
#                 pt_map[uv] = exc_pt

#     # Fallback: if there are no typical loops crossing this edge at all, but we 
#     # have multiple exception points, group them in a default rank (Rank 0) so they connect.
#     if not points_by_rank and len(exception_points) > 0:
#         points_by_rank[0] = exception_points

#     # 3. Create vertices and triangles per rank group
#     new_vertices = []
#     triangles = []
    
#     # Standard neighbor cycle for up to 4 connecting triangles
#     neighbors = [(0, 0), (1, 0), (1, 1), (0, 1)]
    
#     for rank, pt_map in sorted(points_by_rank.items()):
#         rank_pts = list(pt_map.values())
        
#         if not rank_pts:
#             continue
            
#         # Average the coordinates of the valid points matching this specific rank
#         avg_x = sum(p[0] for p in rank_pts) / len(rank_pts)
#         avg_y = sum(p[1] for p in rank_pts) / len(rank_pts)
#         avg_z = sum(p[2] for p in rank_pts) / len(rank_pts)
        
#         proj_pt = (avg_x, avg_y, avg_z)
#         new_vertices.append(proj_pt)
        
#         # Only create triangles when there are 4 points forming a complete quad around the edge
#         if len(rank_pts) == 4:
#             # We connect the newly created averaged (projection) point to the neighboring perimeter points
#             triangles.append((proj_pt, pt_map[neighbors[0]], pt_map[neighbors[1]]))
#             triangles.append((proj_pt, pt_map[neighbors[1]], pt_map[neighbors[2]]))
#             triangles.append((proj_pt, pt_map[neighbors[2]], pt_map[neighbors[3]]))
#             triangles.append((proj_pt, pt_map[neighbors[3]], pt_map[neighbors[0]]))

#         # # Generate up to 4 triangles using adjacent neighboring points for this rank
#         # for i in range(4):
#         #     uv1 = neighbors[i]
#         #     uv2 = neighbors[(i + 1) % 4]
#         #     if uv1 in pt_map and uv2 in pt_map:
#         #         triangles.append((proj_pt, pt_map[uv1], pt_map[uv2]))

#     return new_vertices, triangles



def process_shared_edge_geometry(grid_2x2_lists):
    """
    Processes a 2x2 grid of cubes (4 lists of dicts) sharing an edge to resolve 
    multi-cube intersection points and create connecting triangles.
    Vertices are strictly connected based on pre-calculated topological ranks.
    Exception cubes ignore loops and inject their first component point into 
    any matching neighboring rank groups.
    
    Args:
        grid_2x2_lists (list of lists of dict): 4 lists of dictionaries representing the 4 cubes.
        
    Returns:
        tuple: (new_vertices, triangles)
    """
    # 1. Deduce the base cube index and edge direction
    valid_indices = []
    for cube_list in grid_2x2_lists:
        if cube_list and 'cube_indices' in cube_list[0]:
            valid_indices.append(cube_list[0]['cube_indices'])
            
    if not valid_indices:
        return [], []
        
    min_idx = [min(idx[i] for idx in valid_indices) for i in range(3)]
    max_idx = [max(idx[i] for idx in valid_indices) for i in range(3)]
    
    # The axis where the minimum index equals the maximum index is the shared edge axis
    edge_axis = -1
    for i in range(3):
        if min_idx[i] == max_idx[i]:
            edge_axis = i
            break
            
    if edge_axis == -1:
        # Fallback if topology is malformed, guess Z
        edge_axis = 2 

    # Determine local edge for a specific relative position mapping to strict constants
    def get_local_edge(dx, dy, dz):
        if edge_axis == 2:    # Z-aligned
            if dx == 0 and dy == 0: return 10 # Back-Right
            if dx == 1 and dy == 0: return 11 # Back-Left
            if dx == 0 and dy == 1: return 9  # Front-Right
            if dx == 1 and dy == 1: return 8  # Front-Left
        elif edge_axis == 0:  # X-aligned
            if dy == 0 and dz == 0: return 6  # Top-Back
            if dy == 1 and dz == 0: return 4  # Top-Front
            if dy == 0 and dz == 1: return 2  # Bottom-Back
            if dy == 1 and dz == 1: return 0  # Bottom-Front
        elif edge_axis == 1:  # Y-aligned
            if dx == 0 and dz == 0: return 5  # Top-Right
            if dx == 1 and dz == 0: return 7  # Top-Left
            if dx == 0 and dz == 1: return 1  # Bottom-Right
            if dx == 1 and dz == 1: return 3  # Bottom-Left
        return -1

    # 2. Extract points and organize them by their pre-calculated rank
    # Structure: points_by_rank[rank][relative_uv_tuple] = point
    points_by_rank = collections.defaultdict(dict)
    
    # Keep track of exception points that need to act as wildcards (uv -> point)
    exception_points = {}
    cube_by_uv = {}
    shared_edge_weight = 0
    
    for cube_list in grid_2x2_lists:
        if not cube_list:
            continue
        idx = cube_list[0]['cube_indices']
        dx = idx[0] - min_idx[0]
        dy = idx[1] - min_idx[1]
        dz = idx[2] - min_idx[2]
        
        local_edge = get_local_edge(dx, dy, dz)
        if local_edge == -1:
            continue
            
        # Determine relative 2D coordinate for quad corners around the edge
        if edge_axis == 2:   uv = (dx, dy)
        elif edge_axis == 0: uv = (dy, dz)
        elif edge_axis == 1: uv = (dx, dz)
        
        cube_by_uv[uv] = cube_list
        
        for d in cube_list:
            # Track max shared edge weight for condition evaluation
            w = d.get('edge_weights', [0]*18)[local_edge]
            if w > 0:
                shared_edge_weight = max(shared_edge_weight, w)

            # ---> NEW CHECK: Handle Exception Cubes <---
            if d.get('exception') is True:
                pt = None
                # Safely attempt to extract the first component point
                if d.get('sorted_loops') and len(d['sorted_loops']) > 0:
                    pt = d['sorted_loops'][0].get('component_point')
                elif d.get('component_points') and len(d['component_points']) > 0:
                    pt = d['component_points'][0]
                elif d.get('component_point'):
                    pt = d.get('component_point')
                
                if pt is not None:
                    exception_points[uv] = pt
                
                # Continue early to ignore the regular loops in this exception cube
                continue
            # -------------------------------------------
                
            for loop_data in d.get('sorted_loops', []):
                edges = loop_data.get('loop', [])
                ranks = loop_data.get('rank', [])
                
                # Check if this loop intersects the shared local edge
                # Use enumerate to capture cases where a single loop crosses the same edge multiple times
                for i, edge in enumerate(edges):
                    if edge == local_edge:
                        rank = ranks[i]
                        
                        # Normalize rank orientation to the global positive axis
                        # Edges 2, 6 (-X) and 3, 7 (-Y) go in the negative direction.
                        if local_edge in [2, 6, 3, 7]:
                            W = d.get('edge_weights', [0]*18)[local_edge]
                            normalized_rank = (W - 1) - rank
                        else:
                            normalized_rank = rank
                            
                        pt = loop_data.get('component_point')
                        if pt is not None:
                            points_by_rank[normalized_rank][uv] = pt

    # ---> NEW: Conditional Promotion of Disconnected Cubes to Exceptions <---
    all_have_components = len(cube_by_uv) == 4 and all(
        any(d.get('num_components', 0) > 0 for d in cl) for cl in cube_by_uv.values()
    )
    all_incomplete = all(len(pt_map) < 4 for pt_map in points_by_rank.values())

    if all_have_components and shared_edge_weight > 0 and all_incomplete:
        neighbors_pairs = [((0, 0), (1, 0)), ((1, 0), (1, 1)), ((1, 1), (0, 1)), ((0, 1), (0, 0))]
        new_exception_uvs = set()
        
        for uv1, uv2 in neighbors_pairs:
            connected = False
            for pt_map in points_by_rank.values():
                if uv1 in pt_map and uv2 in pt_map:
                    connected = True
                    break
            if not connected:
                # Pair doesn't connect, flag both as exceptions
                if uv1 in cube_by_uv: new_exception_uvs.add(uv1)
                if uv2 in cube_by_uv: new_exception_uvs.add(uv2)

        for uv in new_exception_uvs:
            if uv not in exception_points:
                pt = None
                # Safely attempt to extract the first component point
                for d in cube_by_uv[uv]:
                    if d.get('sorted_loops') and len(d['sorted_loops']) > 0:
                        pt = d['sorted_loops'][0].get('component_point')
                    elif d.get('component_points') and len(d['component_points']) > 0:
                        pt = d['component_points'][0]
                    elif d.get('component_point'):
                        pt = d.get('component_point')
                    if pt is not None:
                        break
                
                if pt is not None:
                    exception_points[uv] = pt
                
                # Remove this uv's regular loop points so it only acts as an exception wildcard
                for pt_map in points_by_rank.values():
                    pt_map.pop(uv, None)
                    
        # Cleanup any ranks that became completely empty after popping
        empty_ranks = [r for r, pm in points_by_rank.items() if not pm]
        for r in empty_ranks:
            del points_by_rank[r]

    # ---> DISTRIBUTE EXCEPTIONS: Act as wildcards for neighboring ranks <---
    # This fulfills the goal of connecting to "any neighbors".
    for rank, pt_map in points_by_rank.items():
        for uv, exc_pt in exception_points.items():
            if uv not in pt_map:
                pt_map[uv] = exc_pt

    # Fallback: if there are no typical loops crossing this edge at all, but we 
    # have multiple exception points, group them in a default rank (Rank 0) so they connect.
    if not points_by_rank and len(exception_points) > 0:
        points_by_rank[0] = exception_points

    # 3. Create vertices and triangles per rank group
    new_vertices = []
    triangles = []
    
    # Standard neighbor cycle for up to 4 connecting triangles
    neighbors = [(0, 0), (1, 0), (1, 1), (0, 1)]
    
    for rank, pt_map in sorted(points_by_rank.items()):
        rank_pts = list(pt_map.values())
        
        if not rank_pts:
            continue
            
        # Average the coordinates of the valid points matching this specific rank
        avg_x = sum(p[0] for p in rank_pts) / len(rank_pts)
        avg_y = sum(p[1] for p in rank_pts) / len(rank_pts)
        avg_z = sum(p[2] for p in rank_pts) / len(rank_pts)
        
        proj_pt = (avg_x, avg_y, avg_z)
        new_vertices.append(proj_pt)
        
        # Only create triangles when there are 4 points forming a complete quad around the edge
        if len(rank_pts) == 4:
            # We connect the newly created averaged (projection) point to the neighboring perimeter points
            triangles.append((proj_pt, pt_map[neighbors[0]], pt_map[neighbors[1]]))
            triangles.append((proj_pt, pt_map[neighbors[1]], pt_map[neighbors[2]]))
            triangles.append((proj_pt, pt_map[neighbors[2]], pt_map[neighbors[3]]))
            triangles.append((proj_pt, pt_map[neighbors[3]], pt_map[neighbors[0]]))

    return new_vertices, triangles


def merge_close_vertices_and_faces(vertices, faces, tolerance_decimals=5):
    """
    Merges vertices that are closely collocated to reduce mesh size and 
    removes resulting degenerate or duplicate faces.
    
    Args:
        vertices (list): List of (x, y, z) coordinate tuples.
        faces (list): List of (v1, v2, v3) index tuples.
        tolerance_decimals (int): Number of decimal places to round to for merging.
        
    Returns:
        tuple: (new_vertices, new_faces, new_vertex_to_index)
    """
    new_vertices = []
    old_to_new = {}
    spatial_hash = {}
    
    # 1. Weld close vertices
    for i, v in enumerate(vertices):
        # Quantize vertex to the target precision for spatial hashing
        v_round = (round(v[0], tolerance_decimals), 
                   round(v[1], tolerance_decimals), 
                   round(v[2], tolerance_decimals))
        
        if v_round in spatial_hash:
            old_to_new[i] = spatial_hash[v_round]
        else:
            new_idx = len(new_vertices)
            spatial_hash[v_round] = new_idx
            old_to_new[i] = new_idx
            new_vertices.append(v)
            
    new_faces = []
    seen_faces = set()
    
    # 2. Re-index faces, remove degenerates, and deduplicate
    for face in faces:
        # Map face to the new welded vertex indices
        nf = (old_to_new[face[0]], old_to_new[face[1]], old_to_new[face[2]])
        
        # Check for degeneracy (needs 3 distinct vertices)
        if nf[0] != nf[1] and nf[1] != nf[2] and nf[2] != nf[0]:
            # Canonicalize winding order to cleanly detect duplicate identical faces
            min_i = min(nf)
            if nf[0] == min_i:
                canonical_f = nf
            elif nf[1] == min_i:
                canonical_f = (nf[1], nf[2], nf[0])
            else:
                canonical_f = (nf[2], nf[0], nf[1])
            
            if canonical_f not in seen_faces:
                seen_faces.add(canonical_f)
                new_faces.append(nf)  # Preserve original relative winding orientation
                
    # 3. Rebuild precise index mapping for the ongoing builder
    new_vertex_to_index = {}
    for i, v in enumerate(new_vertices):
        v_exact = (round(v[0], 6), round(v[1], 6), round(v[2], 6))
        if v_exact not in new_vertex_to_index:
            new_vertex_to_index[v_exact] = i
            
    return new_vertices, new_faces, new_vertex_to_index


def generate_global_mesh(resolution, cube_data_list, output_filepath="output_mesh.ply", batch_size=1000, merge_decimals=5):
    """
    Iterates over the effective res**3 grid space resolving shared edges.
    Streams the results directly to a 3D PLY file using algorithmic deduplication
    to strictly prevent OOM errors and Pipe constraints.
    
    Args:
        resolution (int): Resolution of the grid.
        cube_data_list (list of dict): Reformed list of cube dicts.
        output_filepath (str): The destination path for the .ply file.
        batch_size (int): Deprecated/unused, left for signature compatibility.
        merge_decimals (int): Coordinate precision for merging closely located vertices on the fly.
        
    Returns:
        str: The filepath of the generated mesh.
    """
    # 1. Group dicts by cube index for fast grid lookup
    cube_map = {}
    for data in cube_data_list:
        idx = data.get('cube_indices')
        if not idx:
            continue
            
        # CRITICAL MEMORY FIX: Strip down dictionary data to strictly what is needed 
        # to avoid RAM saturation from massive underlying geometry lists.
        loops = []
        # for loop in data.get('structured_loops', []):
        #     loops.append({
        #         'mesh_point': loop.get('mesh_point'),
        #         'original_cube_edges': loop.get('original_cube_edges', []),
        #         'loop': loop.get('loop', [])
        #     })
        for loop in data.get('sorted_loops', []):
            loops.append({
                'component_point': loop.get('component_point'),
                'rank': loop.get('rank', []),
                'loop': loop.get('loop', [])
            })
            
        # pruned_data = {
        #     'cube_indices': idx,
        #     'structured_loops': loops
        # }
        pruned_data = {
            'cube_indices': idx,
            'sorted_loops': loops,
            'edge_weights': data.get('edge_weights', [0]*18),
            'exception': data.get('exception', False),
            'num_components': data.get('num_components', 0),
        }
        
        if idx not in cube_map:
            cube_map[idx] = []
        cube_map[idx].append(pruned_data)

    def get_grid_input(indices_list):
        grid = []
        valid_count = 0
        for idx in indices_list:
            if idx in cube_map:
                grid.append(cube_map[idx])
                valid_count += 1
            else:
                grid.append([{'cube_indices': idx, 'structured_loops': []}])
        return grid, valid_count

    # 2. O(1) Memory Task Generator utilizing "Ownership" Deduplication
    def task_generator():
        # Iterate only over active populated cubes
        for idx in cube_map:
            x, y, z = idx
            
            # The 12 bounding edges belonging to this cube
            local_edges = [
                ('X', x, y, z), ('X', x, y+1, z), ('X', x, y, z+1), ('X', x, y+1, z+1),
                ('Y', x, y, z), ('Y', x+1, y, z), ('Y', x, y, z+1), ('Y', x+1, y, z+1),
                ('Z', x, y, z), ('Z', x+1, y, z), ('Z', x, y+1, z), ('Z', x+1, y+1, z)
            ]
            
            for axis, a, b, c in local_edges:
                # Assess bounds and neighbors
                if axis == 'X':
                    if not (0 <= a < resolution and 1 <= b < resolution and 1 <= c < resolution): continue
                    neighbors = [(a, b-1, c-1), (a, b, c-1), (a, b-1, c), (a, b, c)]
                elif axis == 'Y':
                    if not (1 <= a < resolution and 0 <= b < resolution and 1 <= c < resolution): continue
                    neighbors = [(a-1, b, c-1), (a, b, c-1), (a-1, b, c), (a, b, c)]
                elif axis == 'Z':
                    if not (1 <= a < resolution and 1 <= b < resolution and 0 <= c < resolution): continue
                    neighbors = [(a-1, b-1, c), (a, b-1, c), (a-1, b, c), (a, b, c)]
                else:
                    continue
                    
                # Find all valid, actively populated neighbors sharing this exact edge
                active_neighbors = [n for n in neighbors if n in cube_map]
                if not active_neighbors: 
                    continue
                
                # CRITICAL MEMORY FIX: Only yield the task if the *current* cube represents 
                # the lexicographically smallest neighbor. This algorithmically guarantees 
                # no edge is processed twice without needing to store millions of elements in a set().
                if min(active_neighbors) == idx:
                    grid, count = get_grid_input(neighbors)
                    if count >= 2:
                        yield grid
                
    # Memory optimization: Store only vertex hashes mapped to integer indices
    vertex_to_index = {}
    seen_faces = set()
    current_vertex_index = 0  # PLY format utilizes 0-based indexing
    total_faces = 0
    
    verts_filepath = output_filepath + ".verts.tmp"
    faces_filepath = output_filepath + ".faces.tmp"
    
    # 3. Process lazily and stream directly to disk (Bypasses Multiprocessing Pipe Queue entirely)
    with open(verts_filepath, 'w') as f_verts, open(faces_filepath, 'w') as f_faces:
        tasks = task_generator()
        pbar = tqdm(tasks, desc="Processing & Streaming Edges", unit="edges", dynamic_ncols=True)
        
        for grid in pbar:
            verts, tris = process_shared_edge_geometry(grid)
            
            for t in tris:
                idxs = []
                for v in (t[0], t[1], t[2]):
                    # Dynamic weld: round to target decimals immediately
                    v_round = (round(v[0], merge_decimals), round(v[1], merge_decimals), round(v[2], merge_decimals))
                    
                    if v_round not in vertex_to_index:
                        vertex_to_index[v_round] = current_vertex_index
                        # Immediately stream the new vertex to the disk (no 'v' prefix for PLY)
                        f_verts.write(f"{v[0]} {v[1]} {v[2]}\n")
                        current_vertex_index += 1
                        
                    idxs.append(vertex_to_index[v_round])
                
                # Check for degeneracy (needs 3 distinct vertices)
                nf = tuple(idxs)
                if nf[0] != nf[1] and nf[1] != nf[2] and nf[2] != nf[0]:
                    # Canonicalize winding order to detect and strip duplicate identical faces
                    min_i = min(nf)
                    if nf[0] == min_i:
                        canonical_f = nf
                    elif nf[1] == min_i:
                        canonical_f = (nf[1], nf[2], nf[0])
                    else:
                        canonical_f = (nf[2], nf[0], nf[1])
                    
                    if canonical_f not in seen_faces:
                        seen_faces.add(canonical_f)
                        # Immediately stream the face to the disk (prefix '3' for triangle in PLY)
                        f_faces.write(f"3 {idxs[0]} {idxs[1]} {idxs[2]}\n")
                        total_faces += 1
                
            pbar.set_postfix({"verts": current_vertex_index, "faces": total_faces})

    # 4. Final step: Concatenate the temp files into a standard PLY file structure
    print(f"\nAssembling final mesh into {output_filepath}...")
    with open(output_filepath, 'w') as f_out:
        # Write the strict PLY header
        f_out.write("ply\n")
        f_out.write("format ascii 1.0\n")
        f_out.write(f"element vertex {current_vertex_index}\n")
        f_out.write("property float x\n")
        f_out.write("property float y\n")
        f_out.write("property float z\n")
        f_out.write(f"element face {total_faces}\n")
        f_out.write("property list uchar int vertex_indices\n")
        f_out.write("end_header\n")
        
        # Append all vertices
        with open(verts_filepath, 'r') as f_verts:
            shutil.copyfileobj(f_verts, f_out)
            
        # Append all faces
        with open(faces_filepath, 'r') as f_faces:
            shutil.copyfileobj(f_faces, f_out)
            
    # Cleanup temporary split files
    os.remove(verts_filepath)
    os.remove(faces_filepath)
    print("Mesh generation complete!")

    return output_filepath

def reconstruct_mesh(resolution, cube_data_list, output_filepath="output_mesh.ply"):
    """
    Reconstructs a mesh from the cube data list.
    
    Args:
        resolution (int): The resolution of the mesh.
        cube_data_list (list of dict): The list of cube data.
    """
    reformed_data = reform_intersection_data(cube_data_list)
    final_data = extract_original_cube_edges(reformed_data)
    return generate_global_mesh(resolution, final_data, output_filepath=output_filepath)




if __name__ == "__main__":
    # Test data mimicking the structure provided in the prompt
    sample_data = [
        {
            'cube_indices': (693, 542, 605), 
            'face_indices': [12, 13, 14, 15, 16, 17, 18, 33547, 33548, 33549, 33550, 33551, 33552, 33553, 33649, 33650, 33651, 33652, 33653], 
            'num_components': 1, 
            'edge_weights': [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1], 
            'loops': [[4, 13, 6, 16, 11, 17, 8, 14]], 
            'num_loops': 1, 
            'error': None, 
            'd': 0.0, 
            'center_point': [0.6774748166402181, 0.5299870769182841, 0.5912693738937378], 
            'normals': [[-0.7071067811865476, 0.0, 0.7071067811865476]], 
            'mesh_points': [[0.6774748166402181, 0.5299870769182841, 0.5912693738937378]]
        }
    ]
    
    # Run the reformer
    reformed_data = reform_intersection_data(sample_data)

    # Run the new extraction function for original edges
    final_data = extract_original_cube_edges(reformed_data)
    
    # Print the restructured key from the first dictionary to verify
    print("Reformed Output:")
    for item in final_data[0]['structured_loops']:
        print(item)


# Test the shared edge geometry processing
    res = 1000  # Large enough to accommodate index 694
    # Mock a 2x2 grid sharing a Z-edge (X and Y vary, Z is constant at 605)
    mock_grid_2x2_data = [
        {'cube_indices': (693, 542, 605), 'structured_loops': [{'mesh_point': (0.6, 0.5, 0.55), 'original_cube_edges': [10]}]},
        {'cube_indices': (694, 542, 605), 'structured_loops': [{'mesh_point': (0.65, 0.55, 0.56), 'original_cube_edges': [11]}]},
        {'cube_indices': (693, 543, 605), 'structured_loops': [{'mesh_point': (0.55, 0.65, 0.54), 'original_cube_edges': [9]}]},
        {'cube_indices': (694, 543, 605), 'structured_loops': [{'mesh_point': (0.62, 0.62, 0.55), 'original_cube_edges': [8]}]}
    ]
    
    # Run global mesh generator on mock grid list
    output_path = generate_global_mesh(res, mock_grid_2x2_data)
    
    print(f"\nGenerated Mesh saved successfully to: {output_path}")

    breakpoint()