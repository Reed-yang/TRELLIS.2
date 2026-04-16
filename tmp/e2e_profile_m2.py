"""M2 profile: same as M1 but with direct-tensor + GPU fw + direct grids enabled.

Uses COREP_FAST_* env vars which should all default to '1' (on).

Usage:
    python tmp/e2e_profile_m2.py --res 256
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import trimesh

from corep_fast.containers import MeshTensors
from corep_fast.stages.s1_voxelize import s1_voxelize
from corep_fast.stages.s2_components import s2_components
from corep_fast.stages.s3_edge_weights import s3_edge_weights
from corep_fast.stages.s4_face_point import s4_face_point
from corep_fast.stages.s6_collapse import s6_collapse
from corep_fast.stages.s7_rank_assign import s7_rank_assign
from corep_fast.stages.s8_collapse import decode_from_cubebatch


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def profile(mesh_path, resolution, device):
    results = {}
    mesh = trimesh.load(mesh_path)

    gc.collect()
    _sync()
    t0 = time.perf_counter()
    mt = MeshTensors.from_trimesh(mesh, resolution, device=device)
    _sync()
    results['load'] = time.perf_counter() - t0

    for (name, fn_args) in [
        ('s1', lambda b: s1_voxelize(mt, resolution, device)),
    ]:
        pass  # unused, kept for reference

    gc.collect()
    _sync()
    t0 = time.perf_counter()
    batch = s1_voxelize(mt, resolution, device)
    _sync()
    results['s1'] = time.perf_counter() - t0
    results['cubes'] = batch.num_cubes

    gc.collect()
    _sync()
    t0 = time.perf_counter()
    batch = s2_components(batch, mt)
    _sync()
    results['s2'] = time.perf_counter() - t0

    gc.collect()
    _sync()
    t0 = time.perf_counter()
    batch = s3_edge_weights(batch, mt)
    _sync()
    results['s3'] = time.perf_counter() - t0

    gc.collect()
    _sync()
    t0 = time.perf_counter()
    batch = s4_face_point(batch, mt)
    _sync()
    results['s4'] = time.perf_counter() - t0

    gc.collect()
    _sync()
    t0 = time.perf_counter()
    batch = s6_collapse(batch)
    _sync()
    results['s6'] = time.perf_counter() - t0

    gc.collect()
    _sync()
    t0 = time.perf_counter()
    batch = s7_rank_assign(batch)
    _sync()
    results['s7'] = time.perf_counter() - t0

    gc.collect()
    _sync()
    t0 = time.perf_counter()
    v, f = decode_from_cubebatch(batch, merge_decimals=5)
    _sync()
    results['s8'] = time.perf_counter() - t0
    results['V'] = int(v.shape[0]) if hasattr(v, 'shape') else len(v)
    results['F'] = int(f.shape[0]) if hasattr(f, 'shape') else len(f)

    results['s1_s7'] = sum(results[k] for k in ('s1','s2','s3','s4','s6','s7'))
    results['e2e'] = results['load'] + results['s1_s7'] + results['s8']
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res', type=int, default=256)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
    print(f"M2 flags: S8_DIRECT_TENSOR={os.environ.get('COREP_FAST_S8_DIRECT_TENSOR','1')} "
          f"S4_GPU_FW={os.environ.get('COREP_FAST_S4_GPU_FW','1')} "
          f"S8_DIRECT_GRIDS={os.environ.get('COREP_FAST_S8_DIRECT_GRIDS','1')}")

    _ = torch.zeros(10, device=device)
    _sync()

    mesh_path = '/tmp/e2e_m2_bench.ply'
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.4)
    mesh.export(mesh_path)

    print(f"\n=== Resolution = {args.res} ===")
    new = profile(mesh_path, args.res, device)
    for k in ('load','s1','s2','s3','s4','s6','s7','s8','s1_s7','e2e'):
        print(f"  {k:<8}: {new[k]:8.3f}s")
    print(f"  cubes={new['cubes']}  V={new['V']}  F={new['F']}")

    # Compare with baseline
    baseline_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'tmp', 'e2e_profile_results.json')
    try:
        with open(baseline_path) as fp:
            baseline = json.load(fp)
        b = baseline[f'res_{args.res}']
        old = {
            's1': b['s1_voxelize'],
            's2': b['s2_feature_volume'],
            's3': b['s3_feature_edge'],
            's4': b['s4a_feature_face'] + b['s4b_feature_point'],
            's6': b['s6_collapse_face'],
            's7': b['s7_collapse_point'],
            's8_torch': b['s8_torch'],
            's8_custom': b['s8_custom'],
            's1_s7': b['s1_to_s7_total'],
            'e2e_torch': b['e2e_torch'],
            'e2e_custom': b['e2e_custom'],
        }

        print(f"\n--- Speedup vs e2e_custom (true baseline = {old['e2e_custom']:.2f}s) ---")
        print(f"  {'Stage':<10} {'OLD':>10} {'NEW':>10} {'Speedup':>10}")
        for k in ('s1','s2','s3','s4','s6','s7','s1_s7'):
            sp = old[k] / max(new[k], 1e-6)
            print(f"  {k:<10} {old[k]:>10.3f} {new[k]:>10.3f} {sp:>9.2f}x")
        sp_s8 = old['s8_custom'] / max(new['s8'], 1e-6)
        print(f"  {'s8':<10} {old['s8_custom']:>10.3f} {new['s8']:>10.3f} {sp_s8:>9.2f}x")
        sp_e2e = old['e2e_custom'] / max(new['e2e'], 1e-6)
        print(f"  {'e2e':<10} {old['e2e_custom']:>10.3f} {new['e2e']:>10.3f} {sp_e2e:>9.2f}x ⭐")

        print(f"\n--- vs e2e_torch (including triton-s8 optimization) ---")
        sp_torch = old['e2e_torch'] / max(new['e2e'], 1e-6)
        print(f"  e2e     {old['e2e_torch']:>10.3f} {new['e2e']:>10.3f} {sp_torch:>9.2f}x")
    except (FileNotFoundError, KeyError) as e:
        print(f"WARNING: Failed to load baseline: {e}")

    if args.out:
        with open(args.out, 'w') as fp:
            json.dump({
                'new': new,
                'm2_flags': {
                    'S8_DIRECT_TENSOR': os.environ.get('COREP_FAST_S8_DIRECT_TENSOR','1'),
                    'S4_GPU_FW': os.environ.get('COREP_FAST_S4_GPU_FW','1'),
                    'S8_DIRECT_GRIDS': os.environ.get('COREP_FAST_S8_DIRECT_GRIDS','1'),
                },
            }, fp, indent=2, default=str)
        print(f"\nSaved to {args.out}")


if __name__ == '__main__':
    main()
