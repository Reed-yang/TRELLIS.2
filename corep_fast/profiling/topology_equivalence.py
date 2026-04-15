"""
5-layer topology equivalence checker between two CubeBatches.

Layers (see spec §7.2):
    L1 — integer fields: num_components, num_boundary, edge_weights,
         face_weights, status  (exact equality)
    L2 — loop set per cube, compared via canonical form
    L3 — rank assignment on loop edges
    L4 — loop ↔ component_point matching
    L5 — 3D component point coordinates (rtol=1e-4, atol=1e-6)

L1 is implemented in this task.  L2 in Task 8, L3/L4/L5 in Task 9.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from corep_fast.containers import CubeBatch


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

@dataclass
class Mismatch:
    """A single equivalence violation."""
    field: str
    cube_idx: int
    value_a: object = None
    value_b: object = None
    detail: str = ''


@dataclass
class LayerReport:
    layer_name: str
    passed: bool = True
    num_cubes: int = 0
    mismatches: list[Mismatch] = field(default_factory=list)

    def add_mismatch(self, m: Mismatch) -> None:
        self.passed = False
        self.mismatches.append(m)


@dataclass
class EquivalenceReport:
    layer1: Optional[LayerReport] = None
    layer2: Optional[LayerReport] = None
    layer3: Optional[LayerReport] = None
    layer4: Optional[LayerReport] = None
    layer5: Optional[LayerReport] = None

    def all_passed(self) -> bool:
        return all(
            (r is None or r.passed)
            for r in (self.layer1, self.layer2, self.layer3, self.layer4, self.layer5)
        )

    def pretty_print(self) -> str:
        lines = ["Topology Equivalence Report"]
        for r in (self.layer1, self.layer2, self.layer3, self.layer4, self.layer5):
            if r is None:
                continue
            status = "PASS" if r.passed else f"FAIL ({len(r.mismatches)} mismatches)"
            lines.append(f"  {r.layer_name}: {status}")
            if not r.passed and r.mismatches:
                first = r.mismatches[0]
                lines.append(
                    f"    First mismatch: cube {first.cube_idx}, field {first.field!r}, "
                    f"A={first.value_a!r}, B={first.value_b!r}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Layer 1: integer fields
# ---------------------------------------------------------------------------

_L1_FIELDS = [
    'num_components',
    'num_boundary',
    'edge_weights',
    'face_weights',
    'status',
]


def check_layer1_integer_fields(cb_a: CubeBatch, cb_b: CubeBatch) -> LayerReport:
    report = LayerReport(layer_name='L1 integer fields', num_cubes=cb_a.num_cubes)

    if cb_a.num_cubes != cb_b.num_cubes:
        report.add_mismatch(Mismatch(
            field='shape',
            cube_idx=0,
            value_a=cb_a.num_cubes,
            value_b=cb_b.num_cubes,
            detail=f'num_cubes differs: {cb_a.num_cubes} vs {cb_b.num_cubes}',
        ))
        return report

    for fname in _L1_FIELDS:
        ta = getattr(cb_a, fname).cpu()
        tb = getattr(cb_b, fname).cpu()
        if ta.shape != tb.shape:
            report.add_mismatch(Mismatch(
                field=fname,
                cube_idx=0,
                value_a=tuple(ta.shape),
                value_b=tuple(tb.shape),
                detail=f'{fname} shape differs',
            ))
            continue
        if not torch.equal(ta, tb):
            if ta.dim() == 1:
                diff_mask = (ta != tb)
                cube_idx = int(diff_mask.nonzero(as_tuple=False)[0].item())
                va = int(ta[cube_idx].item())
                vb = int(tb[cube_idx].item())
            else:
                diff_mask = (ta != tb).any(dim=-1)
                cube_idx = int(diff_mask.nonzero(as_tuple=False)[0].item())
                va = ta[cube_idx].tolist()
                vb = tb[cube_idx].tolist()
            report.add_mismatch(Mismatch(
                field=fname, cube_idx=cube_idx, value_a=va, value_b=vb,
            ))
    return report


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

_ALL_LAYERS = ['l1', 'l2', 'l3', 'l4', 'l5']


def check_topology_equivalence(
    cb_a: CubeBatch,
    cb_b: CubeBatch,
    layers: list[str] = None,
) -> EquivalenceReport:
    if layers is None:
        layers = _ALL_LAYERS

    report = EquivalenceReport()

    if 'l1' in layers:
        report.layer1 = check_layer1_integer_fields(cb_a, cb_b)
        if not report.layer1.passed:
            return report

    if 'l2' in layers:
        report.layer2 = check_layer2_loop_structure(cb_a, cb_b)
        if not report.layer2.passed:
            return report

    if 'l3' in layers:
        report.layer3 = check_layer3_rank_assignment(cb_a, cb_b)
        if not report.layer3.passed:
            return report

    if 'l4' in layers:
        report.layer4 = check_layer4_point_matching(cb_a, cb_b)
        if not report.layer4.passed:
            return report

    if 'l5' in layers:
        report.layer5 = check_layer5_point_coordinates(cb_a, cb_b)

    return report


# ---------------------------------------------------------------------------
# Loop canonical form (spec §14.C)
# ---------------------------------------------------------------------------

def canonicalize_loop(edges: list[int] | torch.Tensor) -> tuple[int, ...]:
    if isinstance(edges, torch.Tensor):
        seq = edges.tolist()
    else:
        seq = list(edges)
    if not seq:
        raise ValueError("canonicalize_loop: empty loop")

    n = len(seq)
    min_idx = min(range(n), key=lambda i: seq[i])
    fwd = tuple(seq[(min_idx + k) % n] for k in range(n))

    rev = list(reversed(seq))
    min_idx_r = min(range(n), key=lambda i: rev[i])
    rev_canon = tuple(rev[(min_idx_r + k) % n] for k in range(n))

    return fwd if fwd <= rev_canon else rev_canon


def canonicalize_loop_set(loops: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    canon_each = [canonicalize_loop(l) for l in loops]
    return tuple(sorted(canon_each))


# ---------------------------------------------------------------------------
# Layer 2: Loop set equivalence
# ---------------------------------------------------------------------------

def _extract_loops_for_cube(cb: CubeBatch, cube_idx: int) -> list[list[int]]:
    loop_lo = int(cb.loop_cube_off[cube_idx].item())
    loop_hi = int(cb.loop_cube_off[cube_idx + 1].item())
    loops = []
    for l_idx in range(loop_lo, loop_hi):
        e_lo = int(cb.loop_edge_off[l_idx].item())
        e_hi = int(cb.loop_edge_off[l_idx + 1].item())
        loops.append(cb.loop_edge_val[e_lo:e_hi].tolist())
    return loops


def check_layer2_loop_structure(cb_a: CubeBatch, cb_b: CubeBatch) -> LayerReport:
    report = LayerReport(layer_name='L2 loop structure', num_cubes=cb_a.num_cubes)

    for cube_idx in range(cb_a.num_cubes):
        loops_a = _extract_loops_for_cube(cb_a, cube_idx)
        loops_b = _extract_loops_for_cube(cb_b, cube_idx)

        if len(loops_a) != len(loops_b):
            report.add_mismatch(Mismatch(
                field='loop_count',
                cube_idx=cube_idx,
                value_a=len(loops_a),
                value_b=len(loops_b),
            ))
            continue

        canon_a = canonicalize_loop_set(loops_a) if loops_a else ()
        canon_b = canonicalize_loop_set(loops_b) if loops_b else ()
        if canon_a != canon_b:
            report.add_mismatch(Mismatch(
                field='loop_set',
                cube_idx=cube_idx,
                value_a=canon_a,
                value_b=canon_b,
            ))

    return report


# ---------------------------------------------------------------------------
# Layer 3: Rank assignment
# ---------------------------------------------------------------------------

def check_layer3_rank_assignment(cb_a: CubeBatch, cb_b: CubeBatch) -> LayerReport:
    """Exact equality of loop_edge_rank across all loops of all cubes."""
    report = LayerReport(layer_name='L3 rank assignment', num_cubes=cb_a.num_cubes)

    for cube_idx in range(cb_a.num_cubes):
        loop_lo_a = int(cb_a.loop_cube_off[cube_idx].item())
        loop_hi_a = int(cb_a.loop_cube_off[cube_idx + 1].item())
        loop_lo_b = int(cb_b.loop_cube_off[cube_idx].item())
        loop_hi_b = int(cb_b.loop_cube_off[cube_idx + 1].item())

        for li_a, li_b in zip(range(loop_lo_a, loop_hi_a), range(loop_lo_b, loop_hi_b)):
            e_lo_a = int(cb_a.loop_edge_off[li_a].item())
            e_hi_a = int(cb_a.loop_edge_off[li_a + 1].item())
            e_lo_b = int(cb_b.loop_edge_off[li_b].item())
            e_hi_b = int(cb_b.loop_edge_off[li_b + 1].item())

            rank_a = cb_a.loop_edge_rank[e_lo_a:e_hi_a]
            rank_b = cb_b.loop_edge_rank[e_lo_b:e_hi_b]
            if rank_a.shape != rank_b.shape or not torch.equal(rank_a, rank_b):
                report.add_mismatch(Mismatch(
                    field='loop_edge_rank',
                    cube_idx=cube_idx,
                    value_a=rank_a.tolist(),
                    value_b=rank_b.tolist(),
                ))
                break
    return report


# ---------------------------------------------------------------------------
# Layer 4: Loop ↔ point matching
# ---------------------------------------------------------------------------

def check_layer4_point_matching(cb_a: CubeBatch, cb_b: CubeBatch) -> LayerReport:
    """Exact equality of loop_point_match per cube."""
    report = LayerReport(layer_name='L4 point matching', num_cubes=cb_a.num_cubes)

    for cube_idx in range(cb_a.num_cubes):
        loop_lo_a = int(cb_a.loop_cube_off[cube_idx].item())
        loop_hi_a = int(cb_a.loop_cube_off[cube_idx + 1].item())
        loop_lo_b = int(cb_b.loop_cube_off[cube_idx].item())
        loop_hi_b = int(cb_b.loop_cube_off[cube_idx + 1].item())

        match_a = cb_a.loop_point_match[loop_lo_a:loop_hi_a]
        match_b = cb_b.loop_point_match[loop_lo_b:loop_hi_b]

        if match_a.shape != match_b.shape or not torch.equal(match_a, match_b):
            report.add_mismatch(Mismatch(
                field='loop_point_match',
                cube_idx=cube_idx,
                value_a=match_a.tolist(),
                value_b=match_b.tolist(),
            ))
    return report


# ---------------------------------------------------------------------------
# Layer 5: Component point coordinates (float tolerance)
# ---------------------------------------------------------------------------

_L5_RTOL = 1e-4
_L5_ATOL = 1e-6


def check_layer5_point_coordinates(
    cb_a: CubeBatch,
    cb_b: CubeBatch,
    rtol: float = _L5_RTOL,
    atol: float = _L5_ATOL,
) -> LayerReport:
    report = LayerReport(layer_name='L5 point coordinates', num_cubes=cb_a.num_cubes)

    for cube_idx in range(cb_a.num_cubes):
        lo_a = int(cb_a.point_offsets[cube_idx].item())
        hi_a = int(cb_a.point_offsets[cube_idx + 1].item())
        lo_b = int(cb_b.point_offsets[cube_idx].item())
        hi_b = int(cb_b.point_offsets[cube_idx + 1].item())

        pts_a = cb_a.point_values[lo_a:hi_a]
        pts_b = cb_b.point_values[lo_b:hi_b]

        if pts_a.shape != pts_b.shape:
            report.add_mismatch(Mismatch(
                field='point_values',
                cube_idx=cube_idx,
                value_a=tuple(pts_a.shape),
                value_b=tuple(pts_b.shape),
                detail='point count differs',
            ))
            continue

        if pts_a.numel() > 0 and not torch.allclose(pts_a.cpu(), pts_b.cpu(), rtol=rtol, atol=atol):
            max_diff = (pts_a.cpu() - pts_b.cpu()).abs().max().item()
            report.add_mismatch(Mismatch(
                field='point_values',
                cube_idx=cube_idx,
                value_a=pts_a.tolist(),
                value_b=pts_b.tolist(),
                detail=f'max abs diff = {max_diff:.2e}',
            ))
    return report
