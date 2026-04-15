# Phase 1 Stage Rewriting Priority

> Decision date: 2026-04-15
> Based on: profiling/runs/baseline_custom_v0.json (resolution 256, 3 meshes: icosphere, parallel_planes, nested_spheres)
> Machine: CPU-only profiling (custom/ is pure Python+NumPy, no GPU ops)

## Stage Time Rankings (from baseline profile)

Averaged across 3 meshes at resolution 256 (total cubes range: 118K–344K):

| Rank | Stage | Avg Wall Time (s) | % of Total | Decision |
|------|-------|-------------------:|-----------:|----------|
| 1 | s8_collapse (reconstruct_mesh) | 60.6 | 48.5% | Rewrite first — dominates pipeline |
| 2 | s3_feature_edge | 15.1 | 11.7% | Rewrite second |
| 3 | s4_feature_face | 14.2 | 10.9% | Rewrite third (bundle with s3) |
| 4 | s6_collapse_face | 11.9 | 9.5% | Rewrite fourth — hardest stage (combinatorial) |
| 5 | s7_collapse_point | 11.0 | 9.2% | Rewrite fifth (coupled with s6) |
| 6 | s4_feature_point | 8.8 | 7.0% | Rewrite sixth |
| 7 | s2_feature_volume | 2.5 | 2.1% | Rewrite seventh |
| 8 | s1_voxelize | 1.5 | 1.2% | Rewrite last — already fast |

## Key Observations

1. **s8_collapse dominates** at ~48% of total time. This is the mesh reconstruction / deduplication step. The custom/ implementation has O(n²) vertex dedup and sequential face stitching — this is the #1 vectorization target.

2. **s3 + s4_face together are ~23%** — these are the edge weight and face weight computation stages. Both involve per-cube per-triangle geometric calculations that vectorize naturally.

3. **s6_collapse_face at ~9.5%** — despite being the algorithmically hardest stage (combinatorial enumeration), it's only the 4th largest bottleneck at resolution 256. At higher resolutions or with more complex meshes, this may grow due to the O(W³)^12 inner loop. Still worth rewriting early due to complexity.

4. **s1_voxelize is tiny (~1.2%)** — despite being single-threaded in custom/, the voxelization step is fast at 256. May become more significant at 1024 resolution.

5. **GPU memory = 0 throughout** — custom/ is pure CPU. All stages will benefit from GPU migration.

## Final Phase 1 Task Order

1. **s8_collapse** — 48% of runtime. Global vertex dedup + face stitching. Torch `unique` + scatter-based merging should give 50-200× speedup.
2. **s3_feature_edge** — 12% of runtime. Per-cube edge weight computation. Dense tensor ops, straightforward vectorization.
3. **s4_feature_face** — 11% of runtime. Per-cube face weight computation. Includes Sutherland-Hodgman clipping (spec §6.4).
4. **s6_collapse_face** — 9.5% of runtime. Combinatorial loop extraction. Three-layer strategy (algebraic pruning → vectorized enumeration → bucketed dispatch). Hardest to implement but essential for correctness.
5. **s7_collapse_point** — 9% of runtime. Loop-to-point matching. Coupled with s6 output, rewrite together.
6. **s4_feature_point** — 7% of runtime. Component point computation.
7. **s2_feature_volume** — 2% of runtime. Connected component counting per cube.
8. **s1_voxelize** — 1% of runtime. Triangle-voxel intersection. Last priority but still worth Torch for batch pipeline consistency.

## Notes

- Profile data is from resolution 256 on 3 small-to-medium meshes. Resolution 512 profile is pending (running in background) and may shift rankings slightly — especially s6 may grow with more complex geometry.
- The 48% s8_collapse dominance was unexpected — it was not pre-guessed as the top hotspot. This validates the Profiling-First methodology.
- All times are CPU wall-clock. GPU speedups will compound because GPU parallelism benefits all stages simultaneously.
