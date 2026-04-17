"""Deterministic icosphere subdiv-3 builder, matches Phase 2 benchmark exactly.

Reference: tmp/e2e_profile_m2.py:127 uses radius=0.4, subdivisions=3.
Result: V=642, F=1280.
"""
import trimesh


def build_icosphere_subdiv3() -> trimesh.Trimesh:
    """Build deterministic icosphere matching Phase 2 benchmark (radius=0.4)."""
    return trimesh.creation.icosphere(subdivisions=3, radius=0.4)


if __name__ == "__main__":
    m = build_icosphere_subdiv3()
    print(f"V={m.vertices.shape[0]} F={m.faces.shape[0]}")
    assert m.vertices.shape[0] == 642, f"expected V=642, got {m.vertices.shape[0]}"
    assert m.faces.shape[0] == 1280, f"expected F=1280, got {m.faces.shape[0]}"
    print("OK")
