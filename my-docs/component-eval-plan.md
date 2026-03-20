# TRELLIS.2 组件级评估方案（简略版）

> 详细 spec: `docs/superpowers/specs/2026-03-20-component-eval-design.md`
> 详细 plan: `docs/superpowers/plans/2026-03-20-component-eval.md`
> 方法论调研: `logs/eval_methodology_survey.md`

## 目标

系统性测量 TRELLIS.2 各组件（SC-VAE、Structure DiT、Shape DiT、Material DiT）的能力上界，定位瓶颈组件，指导后续优化方向。

## 核心决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 测试集 | Toys4k-PBR (~473) | 论文标准测试集，可与 Table 1 直接对比 |
| 条件图 | 16 视角 Blender CYCLES | 匹配训练分布（Hammersley，FoV 10-70°） |
| DiT 评估策略 | Best-of-16 | 取最低 CD 视角作为能力上界，多卡可控 |
| 后处理 | 同时评估 with/without fill_holes | 隔离后处理贡献 |
| 阶段拆解 | GT injection（5 条件） | 逐阶段替换为 GT，精确量化各 DiT stage 误差 |
| 复杂度分层 | 3 tier（面数 + Canny 边缘比） | 分析 gap 在不同难度上的分布 |

## 三阶段设计

### Phase A — VAE 重建基线（全量 473）
- VAE encode → decode，raw output（无后处理）
- 指标：CD（1M + 100K 两版）、F-score（5 阈值）、NC、PSNR、SSIM、LPIPS
- 与论文 Table 1 对比验证管线正确性
- 预估 ~30 min / 1 GPU

### Phase B — DiT 生成诊断（全量 473 × 16 视角）
- 每个物体 16 张条件图，各跑一次 DiT，取 CD 最低为上界
- 同时输出 with/without fill_holes 两版
- 7,568 次推理，~4 小时 / 8×H100
- 输出：per-object gap ratio、best/worst case、by-tier 分层

### Phase C — 阶段级瓶颈定位（抽样 100）
- 从 Phase B 结果分层抽样 100 个物体
- 5 个实验条件：

| 条件 | Structure | Shape | Material | 目的 |
|------|-----------|-------|----------|------|
| Baseline | DiT | DiT | DiT | 参考线 |
| C1 | **GT** | DiT | DiT | 量化结构预测误差 |
| C2 | **GT** | **GT** | DiT | 隔离材质 DiT |
| C3 | DiT | DiT | **GT** | 隔离几何管线 |
| C4 | **GT** | DiT | **GT** | 隔离 Shape DiT |

- 分析：Baseline vs C1 = 结构贡献；C1 vs C4 = 材质贡献；C4 vs VAE = Shape DiT 剩余 gap

## 指标升级（vs Phase 0）

| 项目 | Phase 0 | 本次 |
|------|---------|------|
| 采样点 | 10K | 100K（Phase B/C）/ 1M（Phase A） |
| F-score 阈值 | 0.01 | 0.005, 0.01, 0.05, 0.1, 0.2 |
| 对齐 | 24-rotation only | 24-rotation + ICP 细化 |
| 渲染指标 | PSNR, SSIM | + LPIPS（感知相似度） |
| 法线图渲染 | 8 view, 任意参数 | 4 view 论文配置 + 8 view 覆盖 |
| 对齐后渲染 | 未对齐 | 对齐后再渲染 |

## 实现概要

```
scripts/
├── prepare_toys4k.py     # 数据准备：下载、PBR 筛选、复杂度分层
├── render_blender_cond.py # 扩展：manifest 批量渲染 16 视角
├── eval_metrics.py       # 扩展：ICP、LPIPS、多阈值 F-score
├── component_eval.py     # 主管线：--phase a|b|c, 多 GPU
└── report_gen.py         # 报告生成器

experiments/component_eval/
├── test_set/             # manifest.json + metadata
├── renders_cond/         # 16 视角条件图
├── phase_a/results/      # VAE 基线
├── phase_b/results/      # DiT best-of-16
└── phase_c/results/      # GT injection
```

## 执行顺序

1. **先启动 Blender 渲染**（~10-20 小时，最长单步）
2. 并行：扩展 eval_metrics + 写 component_eval.py + report_gen.py
3. Phase A（~30 min）→ 验证管线
4. Phase B（~4 小时，8 GPU 并行）→ merge rank CSVs → 生成报告
5. Phase C（~40 min，8 GPU 并行）→ 生成阶段归因报告
6. 汇总结论 → 写 `my-docs/component-eval-summary.md`

## 预期产出

回答 5 个关键问题：
1. VAE 在 Toys4k-PBR 上的重建上界是多少？（对标论文）
2. DiT 在理想条件下的生成上界是多少？（真实 gap）
3. fill_holes 后处理贡献多大？
4. 哪个 DiT stage 是主要瓶颈？（指导 Phase 1）
5. gap 是否随物体复杂度变化？（定位 DiT 弱项）
