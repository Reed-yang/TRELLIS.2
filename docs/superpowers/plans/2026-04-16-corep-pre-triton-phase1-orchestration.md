# CoReP Pre-Triton Phase 1 Orchestration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute Phase 1 of the pre-Triton final-pass evaluation: re-establish baseline, dispatch 4 parallel explore subagents to analyze s4/s6/s7/s8 residual CPU/Python sections, then synthesize their reports into a Phase 2 implementation master plan.

**Architecture:** Read-only orchestration. Main Claude writes baseline scripts + subagent charters, dispatches 4 `Explore` subagents in parallel, verifies their outputs, then writes a Phase 2 implementation plan in writing-plans format (which becomes input to Phase 2 execution). No `corep_fast/` source code changes in this plan.

**Tech Stack:** SSH to 119 node (GPU 0/1 idle), `tmp/run_*.sh` script + `ssh` pattern, `Agent` tool with `Explore` subagent type, markdown analysis reports, `superpowers:writing-plans` skill for the master plan output.

**Spec:** `docs/superpowers/specs/2026-04-16-corep-pre-triton-final-pass-design.md`

---

## Pre-Task Setup Checklist

Before starting Task 1, verify these are true:

- [ ] Working directory is `/mnt/novita2/siyuan/workspace/TRELLIS.2`
- [ ] Current git branch is `gpu-pipeline`
- [ ] `git status` is clean
- [ ] `tmp/m2_final_res128.json` and `tmp/m2_final_res256.json` exist (M2 baseline reference)
- [ ] Memory says profiling on 119 node (per `feedback_profiling_on_119.md`)
- [ ] SSH to 119 node works: `ssh host-10-240-99-119 'hostname'` succeeds

If any fail, STOP and report. Do not proceed.

---

## Task 1: Re-Establish Baseline on 119 Node

**Files:**
- Create: `tmp/run_pretriton_baseline_119_gpu0.sh`
- Create: `tmp/pretriton_baseline_res128.json` (output)
- Create: `tmp/pretriton_baseline_res256.json` (output)
- Reference: `tmp/e2e_profile_m2.py` (existing profile script)

- [ ] **Step 1.1: Read existing M2 profile script and baseline data**

Read `tmp/e2e_profile_m2.py` to confirm it accepts `--res` and writes JSON. Check the M2 baseline numbers:

```bash
cat tmp/m2_final_res128.json
cat tmp/m2_final_res256.json
```

Expected: e2e ~8.5s @ res=128, ~17.9-20.3s @ res=256. These are M2 baseline.

- [ ] **Step 1.2: Write SSH-runnable baseline script**

Create `tmp/run_pretriton_baseline_119_gpu0.sh`:

```bash
#!/bin/bash
# Re-run M2 baseline on 119 GPU 0 to confirm starting point for pre-Triton work.
# Output: tmp/pretriton_baseline_res128.json + tmp/pretriton_baseline_res256.json
set -euo pipefail
cd /mnt/novita2/siyuan/workspace/TRELLIS.2

export CUDA_VISIBLE_DEVICES=0
echo "=== res=128 (3 runs, take median) ==="
for i in 1 2 3; do
    echo "--- run $i ---"
    .venv/bin/python tmp/e2e_profile_m2.py --res 128 --out tmp/pretriton_baseline_res128_run${i}.json
done
echo "=== res=256 (3 runs, take median) ==="
for i in 1 2 3; do
    echo "--- run $i ---"
    .venv/bin/python tmp/e2e_profile_m2.py --res 256 --out tmp/pretriton_baseline_res256_run${i}.json
done

# Pick the run with the median e2e for each resolution
.venv/bin/python -c "
import json, glob, sys
for res in [128, 256]:
    runs = []
    for p in sorted(glob.glob(f'tmp/pretriton_baseline_res{res}_run*.json')):
        with open(p) as f: runs.append((p, json.load(f)))
    runs.sort(key=lambda x: x[1]['new']['e2e'])
    median_path, median_data = runs[len(runs)//2]
    final_path = f'tmp/pretriton_baseline_res{res}.json'
    with open(final_path, 'w') as f: json.dump(median_data, f, indent=2)
    print(f'{final_path}: e2e={median_data[\"new\"][\"e2e\"]:.2f}s (from {median_path})')
"
echo "DONE"
```

Make it executable:

```bash
chmod +x tmp/run_pretriton_baseline_119_gpu0.sh
```

- [ ] **Step 1.3: Verify the script syntax (don't run yet)**

```bash
bash -n tmp/run_pretriton_baseline_119_gpu0.sh
```

Expected: no output (success).

- [ ] **Step 1.4: Verify `tmp/e2e_profile_m2.py` supports `--res` and `--out` flags (PRE-VERIFIED 2026-04-16)**

```bash
grep -E 'argparse|--res|--out' tmp/e2e_profile_m2.py | head -10
```

Expected output (verified 2026-04-16):
```
    python tmp/e2e_profile_m2.py --res 256
import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--res', type=int, default=256)
    parser.add_argument('--out', type=str, default=None)
```

If different, STOP and ask user — profile script may have changed. Do not modify it without explicit permission since `tmp/` is shared.

- [ ] **Step 1.5: Run baseline on 119 GPU 0 via SSH**

```bash
ssh host-10-240-99-119 'bash /mnt/novita2/siyuan/workspace/TRELLIS.2/tmp/run_pretriton_baseline_119_gpu0.sh' 2>&1 | tee tmp/run_pretriton_baseline_119_gpu0.log
```

Expected: prints `--- run i ---` for each of 6 runs, then `DONE`. Total wall time ~3-5 min.

- [ ] **Step 1.6: Verify baseline output JSON exists and is sane**

```bash
cat tmp/pretriton_baseline_res128.json | python3 -m json.tool | head -25
cat tmp/pretriton_baseline_res256.json | python3 -m json.tool | head -25
```

Expected: both JSONs parse, have `new.e2e` between 6-12s for res=128 and 14-22s for res=256, and contain s1/s2/s3/s4/s6/s7/s8 timing keys.

- [ ] **Step 1.7: Commit baseline JSONs (NOT the per-run intermediate JSONs)**

```bash
git add -f tmp/pretriton_baseline_res128.json tmp/pretriton_baseline_res256.json tmp/run_pretriton_baseline_119_gpu0.sh
git commit -m "perf(pretriton): re-establish baseline on 119 GPU 0 for pre-Triton work"
```

If `tmp/` is gitignored (likely), use `-f` as shown.

---

## Task 2: Write 4 Subagent Charters

**Files:**
- Create: `tmp/pretriton_charter_s4.md`
- Create: `tmp/pretriton_charter_s6.md`
- Create: `tmp/pretriton_charter_s7.md`
- Create: `tmp/pretriton_charter_s8.md`

These are the prompts that will be passed verbatim to the 4 `Explore` subagents in Task 3. They must be self-contained because subagents have no conversation history.

- [ ] **Step 2.1: Write the shared prompt-prefix block (used in all 4 charters)**

Save the following as `tmp/pretriton_charter_shared_prefix.md`:

```markdown
# Pre-Triton Analysis (READ-ONLY) — Shared Context

## Your Task

You are an `Explore` subagent assigned to deeply analyze ONE CoReP-fast pipeline stage and produce a markdown report. **You will NOT modify any code.** You only read, profile-data analyze, and write a single markdown report file.

## Working Directory

`/mnt/novita2/siyuan/workspace/TRELLIS.2`

## Project Context

CoReP-fast is the GPU-accelerated rewrite of TRELLIS.2's voxelization pipeline. After M1 + M2 milestones, on H100 res=256 icosphere subdiv=3:
- e2e: 17.92s vs custom baseline 141.05s = **7.87x speedup**
- Remaining bottlenecks: s4 (24%), s6 (14%), s7 (29%), s8 (31%)

Goal: identify all torch / vectorization / algorithm-layer optimizations remaining BEFORE writing Triton kernels. Push every CPU/Python residual section either to GPU torch or document why it is Triton-only.

## Reference Documents (READ THESE)

- Spec for this evaluation: `docs/superpowers/specs/2026-04-16-corep-pre-triton-final-pass-design.md`
- M2 spec: `docs/superpowers/specs/2026-04-16-corep-fast-m2-full-efficiency.md`
- M1 spec: `docs/superpowers/specs/2026-04-16-corep-full-vectorization-design.md`
- s8 historical analysis: `my-docs/20260415-corep-fast-stage2-analysis.md`
- e2e profiling history: `my-docs/20260415-e2e-profiling-acceleration-analysis.md` (esp. appendix D/E/F)
- Deep-review: `my-docs/20260415-deep-review-torch-vectorization.md`
- Repo overview: `corep_fast/README.md`
- Stage 2 v2 vectorization spec (s8 specifically): `docs/superpowers/specs/2026-04-15-corep-fast-stage2-v2-torch-vectorization.md`

## Baseline Profile Data

Read both:
- `tmp/pretriton_baseline_res128.json`
- `tmp/pretriton_baseline_res256.json`

These are the freshly re-measured baselines. Use them as the "current time" for your stage.

## Output Requirements

Write a SINGLE markdown file: `tmp/pretriton_<your_stage>_analysis.md`

The file MUST follow this exact structure (sections A through E). Total length < 300 lines.

```
# <Your Stage> Pre-Triton Analysis

## A. 当前实现快照（针对 corep_fast/stages/<your_file>.py 当前 HEAD）
- 关键函数 + 行号
- 已 GPU 化部分（标注 M1/M2 提交）vs 仍 CPU 部分
- profile 数据来源（commit + 数据文件路径）

## B. 残留 CPU/Python 段的算法本质
对每个 CPU 段：
- 算法描述（伪代码或文字 < 30 行）
- 为什么 M1/M2 把它留在 CPU
- 数据规模 res=128 / 256（item 数量、平均/最大单元工作量）
- 单元算法复杂度（O 表示 + 常数估算）

## C. Torch 化可行性矩阵
| 段 | 提议的 torch 方案 | 风险（精确性 / 内存峰值 / 实现复杂度） | 预期收益 (s, res=256) | 推荐 |
|---|---|---|---|---|
推荐取值: do / skip / triton-only

## D. 数据 contract 检查
- 输入张量: shape / dtype / 来源 stage
- 输出张量: shape / dtype / 下游 consumer
- 与其它 stage 的 contract 是否需要变动
- 与 custom/ baseline 在 V/F 等价上的依赖路径

## E. Triton handoff
若该 stage 的某段不做 torch，剩余给 Triton 的 kernel 应该长什么样：
- 提议的 kernel 输入张量列表（shape / dtype）
- 提议的 grid / block / shared mem 布局
- 估算的 GPU 工作量（FLOPs + bytes）
- 与 Phase 2 torch 优化的接口契合度
```

## Hard Rules

1. **Do NOT modify any source files** — you are read-only
2. **Do NOT run any code** — analysis only (you may read existing profile JSONs but not invoke profile scripts)
3. **Do NOT speculate** — every claim about current code must cite file:line
4. **Do NOT exceed 300 lines** in your output report
5. **Use Chinese-simplified** for the analysis (matches user preference)
```

Save this as `tmp/pretriton_charter_shared_prefix.md` for reference; it gets prepended to each per-stage charter below.

- [ ] **Step 2.2: Write S4 charter**

Create `tmp/pretriton_charter_s4.md` containing the shared prefix above followed by:

```markdown
---

## Your Stage: S4 (face_point)

### File

`corep_fast/stages/s4_face_point.py` (1197 lines)

### Functions to focus on

- `s4_face_point` (line 66) — entry point
- `_compute_face_weights_mp` (line 132) + `_fw_worker_indexed` (line 171) — old MP path
- `_compute_face_weights_gpu` (line 1119) — M2 P2 GPU path

### Specific Investigation Questions

1. The M2 P2 GPU path replaces ~80% of the per-cube CPU work, but Stage C (BFS U-turn count) still runs on CPU MP. Map out exactly which sub-functions of `_compute_face_weights_gpu` still hit Python: pair expansion / plane-tri / clip / compact / **BFS** / scatter. For each, label GPU or CPU and time fraction.
2. The BFS U-turn algorithm: per (cube, facet) it has a small graph (≤ ~16 segments). Can this be vectorized into a batched GPU graph algorithm (e.g., parallel union-find or Bellman-Ford bounded iterations)?
3. The component_centroid path uses GPU `closest_point_on_mesh` (commit `2625ff1`). Are there still Python loops or per-call .item() syncs in the surrounding wrapper code? Is the CSR construction `comp_point_off/val` purely tensor ops?
4. The MP `_fw_worker_indexed` for the old fallback path: when is it actually triggered now? If never, can it be removed?
5. Is there fork/serialization overhead in the GPU path's MP step that could be eliminated by switching to async GPU execution?

### Output File

`tmp/pretriton_s4_analysis.md`

### Stage-specific reading

- M2 spec sections 2.1-2.11 (full s4 GPU-fw redesign)
- Stage C MP code path inside `_compute_face_weights_gpu`
- `corep_fast/geom/closest_point.py` and `corep_fast/geom/plane_tri_intersect.py`
- `corep_fast/geom/sh_clip.py`
```

- [ ] **Step 2.3: Write S6 charter**

Create `tmp/pretriton_charter_s6.md` (shared prefix + below):

```markdown
---

## Your Stage: S6 (collapse / loop extraction)

### File

`corep_fast/stages/s6_collapse.py` (655 lines)

### Functions to focus on

- `s6_collapse` (line 502) — entry point
- `_collapse_fast` (line 72) — fast-path (face_weights all zero)
- `_collapse_with_uturns` (line 296) — slow-path (with U-turn enumeration)
- `_collapse_with_uturns_tracked` (line 379) — slow-path with assignment tracking
- `_s6_worker` (line 476) — MP worker
- per-cube loop at line 580 + MP launch at line 595

### Specific Investigation Questions

1. The fast-path `_collapse_fast`: how often does it hit (% of cubes)? Is it pure GPU or Python? If Python, is there a vectorizable batched version of the closed-form loop trace?
2. The slow-path `_collapse_with_uturns`: it does (u1, u2, u3) cartesian product enumeration per cube with budget 100K. What's the actual distribution of product sizes? p50/p99/max at res=256? If most are tiny (≤32), can we batch all slow-path cubes into a (N_slow, MAX_PRODUCT, 3) tensor and do one GPU-side trace?
3. The graph-trace within slow-path uses Hierholzer-style. Can this be expressed as parallel union-find + canonicalization, where canonicalization is the only sequential step?
4. The MP `_s6_worker`: is the per-task payload large (i.e., serialization overhead is real)? Look at `chunking.py` if used.
5. `uturn_assignment` output (M1 contract): is the (N, 12, 3) tensor materialized on GPU or rebuilt from MP worker outputs?

### Output File

`tmp/pretriton_s6_analysis.md`

### Stage-specific reading

- M1 spec section 4.4 (s6 redesign)
- `_get_canonical_loop` (line 162) and `_get_canonical_solution` (line 181)
```

- [ ] **Step 2.4: Write S7 charter**

Create `tmp/pretriton_charter_s7.md` (shared prefix + below):

```markdown
---

## Your Stage: S7 (rank_assign)

### File

`corep_fast/stages/s7_rank_assign.py` (752 lines)

### Functions to focus on

- `s7_rank_assign` (line 490) — entry point
- `_s7_rank_worker` (line 427) — Phase 1 rank tracing MP worker
- `_match_loops_to_ranks` (line 259) — cyclic alignment
- `_trace_with_ranks_fast` (line 70) — fast path
- `_trace_with_ranks_uturn_assignment` (line 153) — slow path consuming s6's uturn_assignment
- `_compute_centroids` (line 314) — GPU centroid interpolation
- `_hungarian_match` (line 364) — Hungarian wrapper
- `_hungarian_worker` (line 462) — MP worker (currently serialized per commit `24ad133`)
- `_extract_s6_loops` (line 402) and `_extract_component_points` (line 414) — per-cube CSR extraction

### Specific Investigation Questions

1. Phase 1 (rank tracing) MP loop: this is the largest s7 bottleneck. The trace per-cube is sequential graph walk on a small (≤12 edges) graph. Is there a batched GPU formulation (parallel BFS layer-by-layer over all cubes) that avoids per-cube Python?
2. `_extract_s6_loops` and `_extract_component_points` per cube: are they per-cube Python loops? If so, can they be replaced by vectorized CSR gathers?
3. `_match_loops_to_ranks`: cyclic alignment per loop is O(K²) per loop. Can this be batched over L loops as a (L, K, K) tensor matmul/comparison?
4. `_compute_centroids`: confirmed GPU. Are there any .item() or .cpu() syncs?
5. Hungarian: commit `24ad133` made it serial scipy. M1 spec said "inherently sequential" but is there a batched GPU Sinkhorn / auction substitute? Quote real ROI estimate (cost matrix is small per cube).
6. The Phase 1 rank tracing for fast-path cubes (uturn_assignment == -1) vs slow-path: are they done in different code branches? If so, is the fast-path branch GPU-feasible while slow-path remains MP?

### Output File

`tmp/pretriton_s7_analysis.md`

### Stage-specific reading

- M1 spec section 4.5 (s7 redesign)
- M1 spec on the claim "graph traversal is inherently sequential" — challenge it
- Recent commits 24ad133, b1476ca, 4e2d8d9
```

- [ ] **Step 2.5: Write S8 charter**

Create `tmp/pretriton_charter_s8.md` (shared prefix + below):

```markdown
---

## Your Stage: S8 (collapse / mesh decode)

### File

`corep_fast/stages/s8_collapse.py` (2200 lines)

### Functions to focus on

- `decode_from_cubebatch` (line 141) — public entry
- `_cubebatch_to_tensors_direct` (line 534) — M2 P1 direct path
- `process_geometry_vectorized` (line 830) — main vectorized geometry
- `_process_shared_edges_torch` (line 1143) — orchestrates vectorized path + Step E candidate fallback
- `_build_grids_from_tensors` (line 1420) — M2 P3 grid construction
- Step E candidate fallback (line 1222 onward) with MP `_Pool` (line 1279) — the main remaining Python loop

### Specific Investigation Questions

1. The 4-cube candidate fallback (~26% of edges, ~80% of triangles): the historical Stage 2 v2 analysis (`my-docs/20260415-corep-fast-stage2-analysis.md` §3.3) found only **0.05%** of 4-cube edges actually diverge from the GPU path. Why are we still routing all 4-cube edges to fallback? Can a tighter predicate (e.g., `4-cube AND any neighbor has num_loops ≥ 2`) restrict fallback to <2% while keeping correctness?
2. The encoding mismatch (`_EDGE_OFFSET_TABLE` vs `custom/collapse.py::get_local_edge`) called out in the Stage 2 v2 analysis §4.1: is this still present in current code? If so, can a one-time encoding alignment + a hybrid GPU-99.95%/Python-0.05% approach close it?
3. `_process_shared_edges_from_tensors` (line 1480): does it bypass `_build_grids_from_tensors` for the GPU path, or is the M2 P3 work still re-built per call?
4. `_cubebatch_to_dicts` (line 181): is it still called when `USE_DIRECT_TENSOR_S8=1`? If so where? If not, is the dead code worth removing?
5. The `_weld_and_dedup` vertex welding (Stage 1 milestone, ~3.5s historical): is it still on the hot path? Does it use `torch.unique` end-to-end?
6. Per-element `.item()` syncs anywhere in the s8 hot path?

### Output File

`tmp/pretriton_s8_analysis.md`

### Stage-specific reading

- The Stage 2 v2 analysis is the most important ref doc here
- M2 spec section 1 (s8 P1 direct path)
- M2 spec section 3 (s8 P3 candidate)
```

---

## Task 3: Dispatch 4 Explore Subagents in Parallel

**Files:**
- Read: `tmp/pretriton_charter_s{4,6,7,8}.md`
- Output: `tmp/pretriton_s{4,6,7,8}_analysis.md` (written by subagents)

- [ ] **Step 3.1: Verify all 4 charters exist and are non-empty**

```bash
ls -la tmp/pretriton_charter_s{4,6,7,8}.md
wc -l tmp/pretriton_charter_s{4,6,7,8}.md
```

Expected: all 4 exist, each > 100 lines.

- [ ] **Step 3.2: Read each charter file content (so you can pass it to Agent tool)**

Read each of the 4 files using the `Read` tool to get the exact prompt content for each subagent.

- [ ] **Step 3.3: Dispatch all 4 Explore subagents in a SINGLE message**

This step is critical — the 4 `Agent` tool calls MUST be in the SAME assistant message so they execute in parallel. Do not dispatch sequentially.

For each of the 4 stages, call the `Agent` tool with:
- `subagent_type`: `"Explore"`
- `description`: e.g. `"Pre-Triton analysis of S4"` (one per stage)
- `prompt`: the full content of the corresponding `tmp/pretriton_charter_s<stage>.md` file
- Specify `thoroughness: very thorough` in the prompt body

Pseudocode for the assistant turn:

```
[in single message:]
  Agent(subagent_type=Explore, description="Pre-Triton S4 analysis", prompt=<contents of tmp/pretriton_charter_s4.md>)
  Agent(subagent_type=Explore, description="Pre-Triton S6 analysis", prompt=<contents of tmp/pretriton_charter_s6.md>)
  Agent(subagent_type=Explore, description="Pre-Triton S7 analysis", prompt=<contents of tmp/pretriton_charter_s7.md>)
  Agent(subagent_type=Explore, description="Pre-Triton S8 analysis", prompt=<contents of tmp/pretriton_charter_s8.md>)
```

Expected: 4 subagents start in parallel. Wall time ~30-90 min depending on subagent depth. The summary returned by each subagent is NOT what we use — we read the markdown files they produced.

- [ ] **Step 3.4: While waiting (no action), check baseline data is good**

If subagents take more than 60 min, in the meantime, re-verify `tmp/pretriton_baseline_res*.json` shows the expected M2 numbers. Do NOT dispatch additional work until subagents return.

---

## Task 4: Verify Subagent Outputs

**Files:**
- Read: `tmp/pretriton_s{4,6,7,8}_analysis.md`

- [ ] **Step 4.1: Confirm all 4 reports exist**

```bash
ls -la tmp/pretriton_s{4,6,7,8}_analysis.md
wc -l tmp/pretriton_s{4,6,7,8}_analysis.md
```

Expected: all 4 exist, each between 80-300 lines.

If any are missing, the corresponding subagent failed. Re-dispatch ONLY that subagent with its charter — do not duplicate the others.

- [ ] **Step 4.2: Read all 4 reports**

Use the `Read` tool for each. Take notes on:
- Does each follow the A-E structure?
- Does Section C have a recommendation (do / skip / triton-only) for every residual section?
- Does Section D explicitly mention any cross-stage contract changes?
- Does Section E describe a concrete Triton kernel handoff?

- [ ] **Step 4.3: Identify any quality gaps**

For each report, list (in your scratchpad, not a file):
- Sections that are vague or incomplete
- Recommendations that lack a numerical ROI estimate
- Any cross-stage interaction not flagged

If there's a critical gap (e.g., missing Section C entirely), re-dispatch that subagent with a more focused prompt. Otherwise proceed.

- [ ] **Step 4.4: Commit the 4 analysis reports**

```bash
git add -f tmp/pretriton_charter_*.md tmp/pretriton_s{4,6,7,8}_analysis.md
git commit -m "docs(pretriton): add Phase 1 stage analyses (4 explore subagents)"
```

---

## Task 5: Synthesize the Phase 2 Master Plan

**Files:**
- Read: `tmp/pretriton_s{4,6,7,8}_analysis.md`, `tmp/pretriton_baseline_res{128,256}.json`
- Create: `docs/superpowers/plans/2026-04-16-corep-pre-triton-implementation.md`

The master plan IS the Phase 2 implementation plan. It must be written in `superpowers:writing-plans` format so a fresh subagent (or main Claude inline) can execute it task-by-task. **Do not delegate this step to a subagent** — main Claude must do the synthesis because cross-stage interaction reasoning requires holding all 4 reports in context.

- [ ] **Step 5.1: Read all 4 reports + baseline JSONs**

Read fully:
- `tmp/pretriton_s4_analysis.md`
- `tmp/pretriton_s6_analysis.md`
- `tmp/pretriton_s7_analysis.md`
- `tmp/pretriton_s8_analysis.md`
- `tmp/pretriton_baseline_res128.json`
- `tmp/pretriton_baseline_res256.json`

- [ ] **Step 5.2: Build the optimization registry**

In your scratchpad, build a table of EVERY individual optimization mentioned in any of the 4 reports:

| ID | Stage | Description | Recommendation | Predicted ΔT @ res=256 | Risk | Files | Cross-stage? |
|----|-------|-------------|----------------|------------------------|------|-------|--------------|
| O1 | s8 | Tighten 4-cube fallback predicate | do | -2.0s | low | s8_collapse.py:1222-1290 | no |
| O2 | s4 | GPU U-turn BFS via batched union-find | do | -2.5s | medium | s4_face_point.py:1119-... | no |
| ... | | | | | | | |

Sort by `Predicted ΔT desc`. Filter `Recommendation == do`. This is the work registry.

- [ ] **Step 5.3: Decide the two batches for Phase 2**

Pick top 2 by ROI as Batch 1 (parallel on GPU 0/1). Cross-check that the 2 chosen don't modify the same file. If they do, demote one to Batch 2 and pick the next ROI. Repeat for Batch 2. Document any optimizations skipped or held.

- [ ] **Step 5.4: Write the master plan**

Create `docs/superpowers/plans/2026-04-16-corep-pre-triton-implementation.md` following the writing-plans skill structure:

Required header:

```markdown
# CoReP Pre-Triton Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement all torch / vectorization / algorithm-layer optimizations identified in Phase 1 analyses, restricted to those marked "do" in the bottleneck registry. Push remaining bottlenecks to a state where only Triton can further accelerate them.

**Architecture:** 2 git worktrees on 119 GPU 0/1, two batches of 2 stages each. Each worktree owns one stage's optimization, writes A/B test first (TDD), implements + per-stage profile + commit.

**Tech Stack:** PyTorch 2.x on H100, multiprocessing fallback removal, GPU CSR gathers, batched union-find / parallel BFS as appropriate.

**Spec:** `docs/superpowers/specs/2026-04-16-corep-pre-triton-final-pass-design.md`
**Phase 1 reports:** `tmp/pretriton_s{4,6,7,8}_analysis.md`
**Baseline:** res=128 e2e <X>s, res=256 e2e <Y>s (from `tmp/pretriton_baseline_res*.json`)
**Soft target:** res=256 e2e ≤ 13s (from spec §5)

---
```

Then for each "do" optimization, write a Task block following the writing-plans format:

```markdown
### Task N: [optimization-id] — [stage] [short description]

**Files:**
- Modify: `corep_fast/stages/<file>.py:<line-range>`
- Test: `corep_fast/tests/<path>/test_<id>.py` (new)

- [ ] **Step N.1: Write A/B test that captures current behavior**

[exact code]

- [ ] **Step N.2: Run test to confirm it passes against current code**

[exact command + expected output]

- [ ] **Step N.3: [the actual optimization implementation]**

[exact code]

- [ ] **Step N.4: Run A/B test to verify no regression**

- [ ] **Step N.5: Per-stage profile to verify ΔT prediction**

[exact bash command on 119 GPU N]

- [ ] **Step N.6: Commit on `pre-triton/<stage>` branch**

[exact git commands]
```

Tasks should be ordered:
1. Setup (worktree creation per `superpowers:using-git-worktrees`)
2. Batch 1 (2 tasks, marked as parallel — assistant invokes both worktrees in same message)
3. Synchronization checkpoint (verify both batch-1 worktrees clean)
4. Batch 2 (2 tasks, parallel)
5. Synchronization checkpoint
6. Merge tasks (4 worktrees → `pre-triton` integration branch)
7. Final e2e profile (Phase 3)
8. Triton handoff doc (Phase 3)

- [ ] **Step 5.5: Self-review the master plan**

Per writing-plans skill checklist:
1. Spec coverage: every "do" optimization in the registry has a Task
2. No placeholders ("TBD", "implement later", etc.)
3. Type/function name consistency across tasks
4. Cross-stage contracts documented in Setup task
5. Acceptance criteria from spec §5 mapped to verification tasks

Fix issues inline. Then check: `wc -l docs/superpowers/plans/2026-04-16-corep-pre-triton-implementation.md` — should be 400-1500 lines.

- [ ] **Step 5.6: Commit the master plan**

```bash
git add -f docs/superpowers/plans/2026-04-16-corep-pre-triton-implementation.md
git commit -m "docs(pretriton): synthesize Phase 2 master plan from 4 stage analyses"
```

---

## Task 6: Phase 1 Wrap-up + Handoff

- [ ] **Step 6.1: Print summary of Phase 1 outputs**

To the user, summarize:
- Baseline res=128 e2e and res=256 e2e (from JSONs)
- Number of optimizations identified per stage (from registry)
- Total predicted ΔT
- Batch 1 and Batch 2 contents
- Soft-target feasibility judgment (will we hit ≤13s?)

- [ ] **Step 6.2: Hand off to user for Phase 2 execution decision**

Ask the user:
> "Phase 1 complete. The Phase 2 master plan is at `docs/superpowers/plans/2026-04-16-corep-pre-triton-implementation.md`. Want me to (1) review it together first, (2) start Batch 1 immediately via subagent-driven-development, or (3) defer to a future session?"

- [ ] **Step 6.3: Update memory with new project state**

Save a project memory:

```markdown
---
name: pre_triton_phase1_complete
description: Pre-Triton Phase 1 analysis complete; Phase 2 master plan ready for execution
type: project
---
Pre-Triton final-pass evaluation Phase 1 complete on 2026-04-16.

**Status:**
- 4 stage analyses written: tmp/pretriton_s{4,6,7,8}_analysis.md
- Phase 2 master plan: docs/superpowers/plans/2026-04-16-corep-pre-triton-implementation.md
- Baseline reconfirmed on 119 GPU 0: res=128 <X>s, res=256 <Y>s
- Total ROI predicted: <Z>s reduction (vs 17.92s M2 baseline)

**Why:** User asked for one final torch/algorithm-layer pass before Triton.

**How to apply:** Phase 2 execution waits for user go-ahead. Use 119 GPU 0/1 worktrees. Spec at docs/superpowers/specs/2026-04-16-corep-pre-triton-final-pass-design.md.
```

---

## Acceptance Criteria for THIS Plan (Phase 1 only)

- [ ] All 4 `tmp/pretriton_s*_analysis.md` files exist, each follows the A-E structure
- [ ] `tmp/pretriton_baseline_res128.json` and `_res256.json` exist with sane numbers
- [ ] `docs/superpowers/plans/2026-04-16-corep-pre-triton-implementation.md` exists, in writing-plans format, with no placeholders
- [ ] All commits made: baseline, charters + analyses, master plan
- [ ] Memory updated with Phase 1 complete state
- [ ] User briefed on Phase 2 decision

Phase 2 (actual code optimization) is OUT OF SCOPE for this plan — covered by the master plan.

---

## Notes for the Executing Engineer

- Subagents have NO conversation history — every charter must be self-contained
- The 4 subagent dispatches MUST be in a single message for parallelism
- The master plan is written by main Claude (not a subagent) because cross-stage synthesis requires full context
- 119 GPU 0/1 are the only allowed profile/test machines — never run benchmarks locally per `feedback_profiling_on_119.md`
- All SSH commands must `cd /mnt/novita2/siyuan/workspace/TRELLIS.2 &&` first per CLAUDE.md
- This is a research/analysis plan — no `corep_fast/` source code is modified by THIS plan
