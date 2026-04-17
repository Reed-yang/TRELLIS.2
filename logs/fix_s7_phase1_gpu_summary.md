# s7 Phase1 GPU 修复总结 (2026-04-17)

## 问题
`COREP_FAST_S7_PHASE1_GPU=1` (W2) 在多 loop cube 上产生错误 rank
→ 三层 icosphere @ res=32 丢 45% 几何 (14026/23740 vs baseline 25358/50708)。

## 根因
`_phase1_gpu_rank_assign` 对每条 s6 loop 独立 `alive.argmax` 取最小 rank0，
缺失 CPU `_match_loops_to_ranks` 的 `used_traced[i]=True` 双射消费约束。
当 cube 内多条 s6 loop 共享 edge sequence 时，GPU 让它们塌到同一 rank。

此外 GPU walker 对 **intra-loop 连续重复 edge** (U-turn 模式，如 `[..., 14, 14, ...]`)
处理失败（alive mask 全 False）。

## Baseline 对齐 (custom/)
`custom/collapse_point.py:147-183` 是 baseline 双射匹配 canonical 算法：
- `used_traced = [False] * len(traced_loops)`
- 对每条 s6 loop 尝试 forward/backward cyclic shift 匹配 traced_loop
- 匹配即 `used_traced[i] = True`，跳过已消费
- 未匹配 fallback 到 `[0] * k`

`corep_fast/stages/s7_rank_assign.py::_match_loops_to_ranks` (L260-309)
是 custom 的直译。GPU 路径必须复现同语义。

## 修复方案
`corep_fast/stages/s7_rank_assign.py::_phase1_gpu_rank_assign` (L1041-1142)

### 1. (cube, edge_sequence) 分组 + kth-True 选择
替换 `first_idx = alive.argmax(dim=1)`:
- `row_key = [cube_id, edges_0..K_max-1]` per loop (shape `(L, K_max+1) int32`)
- `_, group_ids = torch.unique(row_key, return_inverse=True, dim=0)`
- stable-sort + cummax segment reset → `within_group_idx` (组内第 0/1/2/... 条)
- cumsum trick: 取 `alive` 的第 `within_group_idx`-个 True
- 组数超过 alive 总数时 fallback 全 0（镜像 CPU "used_traced 耗尽" 分支）

### 2. CPU fallback for walker edge cases
`any_match=False` AND `status==OK` 的 cube (cube 966 / cube 954 等 4 例)
调用 `_s7_rank_worker` 重跑，`chosen_ranks` 写回。

## 验证 (node 116)
- **三层球 A/B 测试**: rank_diff=0, match_diff=0 (0 / 50497 mismatch) ✓
- **单层球 A/B** (res 32/64/128): 3/3 PASS ✓
- **全 regression suite**: 30/30 PASS ✓
- **8-GPU 并行 flag 矩阵**:
  - W2=0: V=25358 F=50708 (baseline)
  - W2=1: V=25361 F=50720 (+0.01%，浮点累加噪声)
- **e2e 修复验证 (HEAD 默认 W2=1)**: V=25361 F=50720
  - vs d780be8 (25358/50708): +3V/+12F = 0.01% (rank 张量位级相等，
    V/F 差来自 Phase 2 GPU scatter_mean 非确定性顺序)

## 新增测试
`corep_fast/tests/regression/test_s7_phase1_gpu_ab.py::test_s7_phase1_gpu_matches_cpu_triple_sphere`

使用三层嵌套 icosphere (r=1.00/1.01/1.02, s=3) @ res=32，覆盖：
- 多 loop cube + 重复 edge sequence (680 个)
- intra-loop 重复 edge (4 个)

## 改动文件
- `corep_fast/stages/s7_rank_assign.py` (+~75 行 kth-True 选择器 + CPU fallback)
- `corep_fast/tests/regression/test_s7_phase1_gpu_ab.py` (+~80 行 新测试 + 共享 helpers)

## 证据
- `logs/findings_s7_phase1_gpu_regression.md` — 原始根因报告
- `tmp/test_s7_triple_{before,after}_fix.log` — A/B test 前后对比
- `tmp/flag_matrix_{116,parallel_116,after_fix}.log` — flag 矩阵 (顺序 + 8 GPU 并行)
- `tmp/verify_all_116.log` — pipeline + 30 regression tests 全通过
- `tmp/test_fast/output.ply` (V=25361) — 修复后几何输出
