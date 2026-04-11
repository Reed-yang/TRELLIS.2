import os
import numpy as np
import trimesh

from utils import voxels_to_mesh, extract_array_from_json


def edges_to_cylinders_mesh(edges, color=[255, 255, 0, 255], radius=2e-4):
    """
    Creates a single 3D mesh consisting of cylindrical pipes for each edge.
    
    Parameters:
    -----------
    edges : np.ndarray
        Shape (n, 2, 3) representing n pairs of 3D vertices.
        edges[:, 0, :] are the start points, edges[:, 1, :] are the end points.
    color : list or tuple
        RGBA or RGB color for the pipes. Default is Yellow [255, 255, 0, 255].
    radius : float
        The radius (thickness) of the pipes.
        
    Returns:
    --------
    trimesh.Trimesh
        A single mesh containing all the combined pipes.
    """
    # Validate input shape
    edges = np.asarray(edges)
    if edges.ndim != 3 or edges.shape[1:] != (2, 3):
        raise ValueError(f"Expected edges to have shape (n, 2, 3), but got {edges.shape}")
        
    pipe_meshes = []
    
    for edge in edges:
        start_pt, end_pt = edge
        
        # Calculate length of the edge
        length = np.linalg.norm(end_pt - start_pt)
        
        # Skip degenerate edges (where start and end points are effectively the same)
        if length < 1e-8:
            continue
            
        # Create a cylinder exactly between the start and end points
        pipe = trimesh.creation.cylinder(radius=radius, segment=edge)
        
        # Apply the color
        pipe.visual.vertex_colors = color
        
        pipe_meshes.append(pipe)
        
    # If no valid edges were found, return an empty mesh
    if not pipe_meshes:
        return trimesh.Trimesh()
        
    # Concatenate all individual pipe meshes into one single mesh
    combined_mesh = trimesh.util.concatenate(pipe_meshes)
    
    return combined_mesh




if __name__ == '__main__':

    input_json_path = '/home/lagwein/trellis/TRELLIS.2/tmp/corep/volume_feature.json'
    volume_feature_keys = ['index', 'resolution', 'non_manifold_num', 'edge_border_num', 'flat_sheet_num']

    # vis_num = 1000000
    vis_num = -1
    vis_dir = 'tmp/corep/visualize'


    volume_feature = extract_array_from_json(input_json_path, volume_feature_keys)

    index = volume_feature['index']
    resolution = volume_feature['resolution'][0]
    non_manifold_num = volume_feature['non_manifold_num']
    edge_border_num = volume_feature['edge_border_num']
    flat_sheet_num = volume_feature['flat_sheet_num']

    index = index[:vis_num]
    non_manifold_num = non_manifold_num[:vis_num]
    edge_border_num = edge_border_num[:vis_num]
    flat_sheet_num = flat_sheet_num[:vis_num]

    os.makedirs(vis_dir, exist_ok=True)

    voxel_occ = voxels_to_mesh(index, resolution)
    voxel_occ.export(f'{vis_dir}/voxel_occ.ply')

    voxel_eb = voxels_to_mesh(index[edge_border_num>0], resolution, [255, 255, 0, 255])
    voxel_eb.export(f'{vis_dir}/voxel_eb.ply')

    voxel_fs = voxels_to_mesh(index[flat_sheet_num>0], resolution, [0, 128, 255, 255])
    voxel_fs.export(f'{vis_dir}/voxel_fs.ply')



    # breakpoint()




