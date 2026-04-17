# tmp/profile_deep/test_monkeypatch_import.py
"""Smoke test: monkeypatch imports cleanly and patches all 7 stage entries."""
import sys


def test_patches_all_stages():
    # Clean slate
    for mod in list(sys.modules.keys()):
        if mod.startswith('corep_fast'):
            del sys.modules[mod]
    # Apply patches
    from tmp.profile_deep import monkeypatch_nvtx
    monkeypatch_nvtx.apply_stage_nvtx()
    # Verify each entry function has been wrapped
    import corep_fast.stages.s1_voxelize as s1
    import corep_fast.stages.s2_components as s2
    import corep_fast.stages.s3_edge_weights as s3
    import corep_fast.stages.s4_face_point as s4
    import corep_fast.stages.s6_collapse as s6
    import corep_fast.stages.s7_rank_assign as s7
    import corep_fast.stages.s8_collapse as s8
    for fn, expected_name in [
        (s1.s1_voxelize, "s1_voxelize"),
        (s2.s2_components, "s2_components"),
        (s3.s3_edge_weights, "s3_edge_weights"),
        (s4.s4_face_point, "s4_face_point"),
        (s6.s6_collapse, "s6_collapse"),
        (s7.s7_rank_assign, "s7_rank_assign"),
        (s8.decode_from_cubebatch, "s8_decode"),
    ]:
        assert hasattr(fn, "_nvtx_wrapped"), f"{expected_name} not wrapped"
        assert fn._nvtx_wrapped == expected_name, f"{fn._nvtx_wrapped} != {expected_name}"
    print(f"OK: 7 stages patched.")


def test_substage_events_register():
    # Clean import state
    for mod in list(sys.modules.keys()):
        if mod.startswith('corep_fast'):
            del sys.modules[mod]
    from tmp.profile_deep import monkeypatch_nvtx
    monkeypatch_nvtx.apply_stage_nvtx()
    monkeypatch_nvtx.apply_substage_events()
    assert len(monkeypatch_nvtx._SUBSTAGE_TIMINGS) == 0, "should start empty"
    # Verify a known sub-stage fn is patched
    import corep_fast.stages.s4_face_point as s4
    assert hasattr(s4._compute_face_weights_gpu, "_event_wrapped")
    import corep_fast.stages.s6_collapse as s6
    assert hasattr(s6._fastpath_gpu_build_adjacency, "_event_wrapped")
    import corep_fast.stages.s7_rank_assign as s7
    assert hasattr(s7._phase1_gpu_rank_assign, "_event_wrapped")
    print("OK: sub-stage events registered.")


if __name__ == "__main__":
    test_patches_all_stages()
    test_substage_events_register()
