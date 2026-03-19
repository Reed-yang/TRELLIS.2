"""
Evaluation metrics for Gap Measurement pipeline.
Geometric: Chamfer Distance, F-score, Normal Consistency
Rendering: PSNR, SSIM on normal maps (wraps trellis2 utilities)
"""

import torch
import numpy as np
import trimesh as tm


# ---------------------------------------------------------------------------
# Point cloud sampling
# ---------------------------------------------------------------------------

def sample_points_and_normals(mesh, num_points=10000):
    """
    Sample points and face normals from a trimesh mesh surface.

    Args:
        mesh: trimesh.Trimesh object
        num_points: number of points to sample

    Returns:
        points: [N, 3] torch float tensor
        normals: [N, 3] torch float tensor
    """
    points, face_indices = tm.sample.sample_surface(mesh, num_points)
    normals = mesh.face_normals[face_indices]
    return (
        torch.from_numpy(np.asarray(points)).float(),
        torch.from_numpy(np.asarray(normals)).float(),
    )


def trellis_mesh_to_trimesh(trellis_mesh):
    """Convert a trellis2.representations.Mesh to trimesh.Trimesh."""
    return tm.Trimesh(
        vertices=trellis_mesh.vertices.detach().cpu().numpy(),
        faces=trellis_mesh.faces.detach().cpu().numpy(),
        process=False,
    )


# ---------------------------------------------------------------------------
# Geometric metrics
# ---------------------------------------------------------------------------

def _chunked_min_dists(src, tgt, chunk_size=2048):
    """Compute min L2 distances from each point in src to nearest point in tgt."""
    min_dists = []
    for i in range(0, len(src), chunk_size):
        chunk = src[i:i + chunk_size]
        dists = torch.cdist(chunk, tgt)  # [chunk, M]
        min_dists.append(dists.min(dim=1)[0])
    return torch.cat(min_dists)


def chamfer_distance(points1, points2, chunk_size=2048):
    """
    Bidirectional Chamfer Distance (mean of squared L2 distances).

    Args:
        points1: [N, 3] tensor on GPU
        points2: [M, 3] tensor on GPU
        chunk_size: batch size for chunked cdist to avoid OOM

    Returns:
        Scalar float: mean bidirectional CD
    """
    d1 = _chunked_min_dists(points1, points2, chunk_size)
    d2 = _chunked_min_dists(points2, points1, chunk_size)
    return ((d1 ** 2).mean() + (d2 ** 2).mean()).item() / 2


def f_score(points1, points2, threshold=0.01, chunk_size=2048):
    """
    F-score: harmonic mean of precision and recall at distance threshold.

    Args:
        points1: [N, 3] tensor on GPU (predicted)
        points2: [M, 3] tensor on GPU (ground truth)
        threshold: distance threshold (in model coordinate space [-0.5, 0.5])
        chunk_size: batch size for chunked cdist

    Returns:
        Scalar float in [0, 1]
    """
    d1 = _chunked_min_dists(points1, points2, chunk_size)
    d2 = _chunked_min_dists(points2, points1, chunk_size)
    precision = (d1 < threshold).float().mean()
    recall = (d2 < threshold).float().mean()
    denom = precision + recall
    if denom < 1e-8:
        return 0.0
    return (2 * precision * recall / denom).item()


def normal_consistency(points1, normals1, points2, normals2, chunk_size=2048):
    """
    Normal Consistency: mean |cos(angle)| between matched point normals.
    For each point in points1, find nearest point in points2, compare normals.

    Args:
        points1, normals1: [N, 3] tensors on GPU
        points2, normals2: [M, 3] tensors on GPU

    Returns:
        Scalar float in [0, 1]
    """
    nn_indices = []
    for i in range(0, len(points1), chunk_size):
        chunk = points1[i:i + chunk_size]
        dists = torch.cdist(chunk, points2)
        nn_indices.append(dists.argmin(dim=1))
    nn_indices = torch.cat(nn_indices)

    matched_normals = normals2[nn_indices]
    cos_sim = torch.abs((normals1 * matched_normals).sum(dim=1))
    return cos_sim.mean().item()


# ---------------------------------------------------------------------------
# Rendering-based metrics
# ---------------------------------------------------------------------------

def render_normal_maps(trellis_mesh, nviews=8, resolution=512):
    """
    Render normal maps of a trellis2 Mesh from multiple views.

    Args:
        trellis_mesh: trellis2.representations.Mesh on CUDA
        nviews: number of views
        resolution: render resolution

    Returns:
        List of [3, H, W] float tensors (normal maps in [0, 1])
    """
    from trellis2.utils.render_utils import render_snapshot

    result = render_snapshot(
        trellis_mesh,
        resolution=resolution,
        nviews=nviews,
        r=2, fov=40,
        return_types=["normal"],
    )
    normal_maps = []
    for nmap in result["normal"]:
        t = torch.from_numpy(nmap).float() / 255.0  # [H, W, 3]
        normal_maps.append(t.permute(2, 0, 1))  # [3, H, W]
    return normal_maps


def compute_rendering_metrics(normal_maps_pred, normal_maps_gt):
    """
    Compute PSNR and SSIM between predicted and GT normal maps.

    Args:
        normal_maps_pred: list of [3, H, W] float tensors
        normal_maps_gt: list of [3, H, W] float tensors

    Returns:
        dict with 'psnr' and 'ssim' (mean across views)
    """
    from trellis2.utils.loss_utils import psnr, ssim

    psnr_vals = []
    ssim_vals = []
    for pred, gt in zip(normal_maps_pred, normal_maps_gt):
        pred = pred.unsqueeze(0).cuda()  # [1, 3, H, W]
        gt = gt.unsqueeze(0).cuda()
        psnr_vals.append(psnr(pred, gt).item())
        ssim_vals.append(ssim(pred, gt).item())

    return {
        "psnr": float(np.mean(psnr_vals)),
        "ssim": float(np.mean(ssim_vals)),
    }
