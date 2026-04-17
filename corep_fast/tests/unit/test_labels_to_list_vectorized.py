"""W_L2L: vectorized _labels_to_list_of_lists must match legacy bucket loop."""
import numpy as np
import pytest
import torch

from corep_fast.stages.s4_face_point import _labels_to_list_of_lists


# Legacy implementation copied from s4_face_point.py:966-1012 at anchor HEAD
# (bit-identical reference for parity tests).
def _labels_to_list_of_lists_legacy(batched_labels, batched_face_ids, face_counts):
    N, M = batched_labels.shape
    if N == 0:
        return []
    order = torch.argsort(batched_labels, dim=1, stable=True)
    sorted_labels = batched_labels.gather(1, order)
    sorted_fids = batched_face_ids.gather(1, order)
    sorted_labels_cpu = sorted_labels.cpu().numpy()
    sorted_fids_cpu = sorted_fids.cpu().numpy()
    counts_cpu = face_counts.cpu().numpy()
    result = []
    for i in range(N):
        n_i = int(counts_cpu[i])
        if n_i == 0:
            result.append([])
            continue
        row_labels = sorted_labels_cpu[i, :n_i]
        row_fids = sorted_fids_cpu[i, :n_i]
        components = []
        cur_label = int(row_labels[0])
        cur_comp = [int(row_fids[0])]
        for k in range(1, n_i):
            lbl = int(row_labels[k])
            if lbl != cur_label:
                components.append(cur_comp)
                cur_comp = []
                cur_label = lbl
            cur_comp.append(int(row_fids[k]))
        components.append(cur_comp)
        result.append(components)
    return result


def _random_batch(N, M, seed=0):
    rng = np.random.RandomState(seed)
    counts = rng.randint(0, M + 1, size=N).astype(np.int64)
    labels = np.full((N, M), M, dtype=np.int64)  # pad = SENTINEL=M
    fids = np.full((N, M), -1, dtype=np.int64)
    for i in range(N):
        n = counts[i]
        labels[i, :n] = rng.randint(0, max(1, n), size=n)
        fids[i, :n] = rng.randint(0, 1_000_000, size=n)
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return (
        torch.from_numpy(labels).to(dev),
        torch.from_numpy(fids).to(dev),
        torch.from_numpy(counts).to(dev),
    )


def test_matches_legacy_random_100():
    labels, fids, counts = _random_batch(N=100, M=20, seed=0)
    legacy = _labels_to_list_of_lists_legacy(labels, fids, counts)
    new = _labels_to_list_of_lists(labels, fids, counts)
    assert len(legacy) == len(new)
    for i, (L_i, N_i) in enumerate(zip(legacy, new)):
        assert len(L_i) == len(N_i), f"row {i}: comp count mismatch"
        for ci, (Lc, Nc) in enumerate(zip(L_i, N_i)):
            assert [int(x) for x in Lc] == [int(x) for x in Nc], \
                f"row {i} comp {ci}: {Lc} != {Nc}"


def test_empty_rows():
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    labels = torch.zeros((5, 10), dtype=torch.int64, device=dev)
    fids = torch.zeros((5, 10), dtype=torch.int64, device=dev)
    counts = torch.zeros((5,), dtype=torch.int64, device=dev)
    result = _labels_to_list_of_lists(labels, fids, counts)
    assert result == [[], [], [], [], []]


def test_all_same_label_one_component_per_row():
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    N, M = 4, 5
    labels = torch.zeros((N, M), dtype=torch.int64, device=dev)
    fids = torch.arange(N * M, dtype=torch.int64, device=dev).reshape(N, M)
    counts = torch.full((N,), M, dtype=torch.int64, device=dev)
    result = _labels_to_list_of_lists(labels, fids, counts)
    assert len(result) == N
    for i in range(N):
        assert len(result[i]) == 1
        assert [int(x) for x in result[i][0]] == list(range(i * M, (i + 1) * M))


def test_all_distinct_labels_singleton_components():
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    N, M = 3, 4
    labels = torch.arange(M, dtype=torch.int64, device=dev).view(1, M).expand(N, M).contiguous()
    fids = torch.arange(N * M, dtype=torch.int64, device=dev).reshape(N, M)
    counts = torch.full((N,), M, dtype=torch.int64, device=dev)
    result = _labels_to_list_of_lists(labels, fids, counts)
    assert len(result) == N
    for i in range(N):
        assert len(result[i]) == M
        for k in range(M):
            assert [int(x) for x in result[i][k]] == [i * M + k]
