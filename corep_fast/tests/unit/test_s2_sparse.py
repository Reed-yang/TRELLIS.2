"""Property test: sparse label-prop matches dense reference on random inputs."""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="s2 sparse tests require CUDA",
)


def _random_batch(N, M, p_edge, seed):
    """Generate synthetic padded_faces (N, M) int32, a mesh face_adj (F, 3)
    int32, and a valid-mask. p_edge controls adjacency density."""
    g = torch.Generator().manual_seed(seed)
    counts = torch.randint(1, M + 1, (N,), generator=g)  # per-cube valid count
    padded = torch.full((N, M), -1, dtype=torch.int32)
    F_max = int(counts.sum().item() * 2)
    face_ids = torch.arange(F_max, dtype=torch.int32)
    for n in range(N):
        c = int(counts[n])
        perm = torch.randperm(F_max, generator=g)[:c]
        padded[n, :c] = face_ids[perm]
    # Random face_adj — each face has 3 neighbors chosen from F_max uniformly
    face_adj = torch.randint(0, F_max, (F_max, 3), generator=g, dtype=torch.int32)
    # Simulate edge density by zeroing (setting to sentinel) a fraction:
    drop = torch.rand(F_max, 3, generator=g) > p_edge
    face_adj[drop] = -1
    mask = torch.arange(M).unsqueeze(0) < counts.unsqueeze(1)  # (N, M) bool
    return padded, face_adj, mask


@pytest.mark.parametrize("seed", list(range(10)))
def test_sparse_matches_dense(seed):
    """On 10 random small inputs, sparse and dense label-prop give the same
    num_components per cube."""
    dev = torch.device("cuda")
    N, M = 4, 12  # small enough that dense path fits easily
    padded, face_adj, mask = _random_batch(N, M, p_edge=0.4, seed=seed)
    padded = padded.to(dev); face_adj = face_adj.to(dev); mask = mask.to(dev)

    from corep_fast.stages.s2_components_sparse import _sparse_num_components
    from corep_fast.stages.s2_components import _label_propagation_sequential

    nc_sparse = _sparse_num_components(padded, face_adj, mask).cpu()
    labels_seq = _label_propagation_sequential(padded, face_adj[padded.long()],
                                               mask, N, M, dev)
    nc_seq = torch.zeros(N, dtype=torch.int32)
    for n in range(N):
        valid_labels = labels_seq[n][mask[n]].unique()
        nc_seq[n] = valid_labels.numel()

    assert torch.equal(nc_sparse, nc_seq.to(dev).cpu()), (
        f"seed={seed}: sparse {nc_sparse.tolist()} vs seq {nc_seq.tolist()}")
