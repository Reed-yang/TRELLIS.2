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

    # L2/L3/L4/L5 added in Tasks 8 and 9
    return report
