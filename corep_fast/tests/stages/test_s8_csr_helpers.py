"""Tests for CSR expansion helpers used in direct-tensor s8 path."""
import torch
import pytest
from corep_fast.stages.s8_collapse import _csr_expand_to_items


class TestCsrExpandToItems:
    def test_simple_case(self):
        """offsets=[0,2,5,6] -> [0,0,1,1,1,2]"""
        offsets = torch.tensor([0, 2, 5, 6], dtype=torch.int64)
        result = _csr_expand_to_items(offsets, total=6)
        expected = torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.int64)
        assert torch.equal(result, expected)

    def test_empty_groups_in_middle(self):
        """offsets=[0,2,2,5] -> group 1 is empty -> [0,0,2,2,2]"""
        offsets = torch.tensor([0, 2, 2, 5], dtype=torch.int64)
        result = _csr_expand_to_items(offsets, total=5)
        expected = torch.tensor([0, 0, 2, 2, 2], dtype=torch.int64)
        assert torch.equal(result, expected)

    def test_total_zero(self):
        """total=0 -> empty tensor."""
        offsets = torch.tensor([0, 0, 0], dtype=torch.int64)
        result = _csr_expand_to_items(offsets, total=0)
        assert result.numel() == 0

    def test_gpu(self):
        """Runs on CUDA if available."""
        if not torch.cuda.is_available():
            pytest.skip("no CUDA")
        offsets = torch.tensor([0, 3, 5], dtype=torch.int64, device='cuda')
        result = _csr_expand_to_items(offsets, total=5)
        expected = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64, device='cuda')
        assert torch.equal(result, expected)
