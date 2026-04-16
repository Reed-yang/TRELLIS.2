# Phase 0: Gap Measurement — Summary

> 快速参考文档。详细记录见 `logs/progress.md` 和 `logs/findings.md`。

## 结论

**DiT 是当前系统瓶颈**，不是 VAE。修正条件图选择 + 对齐后，DiT 生成质量 (CD) 比 VAE 重建上界差 **~19 倍**。

→ Phase 1 应将 **DiT 改进 (P2b)** 提升为最高优先级。

## 数据

- 30 个 Objaverse 模型（多样 LVIS 类别），Blender CYCLES 渲染条件图像
- 24 轴旋转对齐修正坐标系差异
- 最佳正面视角选择（基于仰角/方位角/FOV 评分函数）

### 修正后结果（v2, best-view + aligned）

| 指标 | VAE 重建 | DiT 生成 | Gap |
|------|---------|---------|-----|
| CD | 0.000071 | 0.00133 | **18.7x** |
| F-score | 0.774 | 0.360 | 2.2x |
| NC | 0.932 | 0.779 | 1.2x |

### 修正前结果对比

| 版本 | CD ratio | 问题 |
|------|---------|------|
| 原始保存结果 | 406x | 未应用 24-rotation 对齐 |
| 对齐后 baseline | 49x | 随机 Blender 视角 000.png（含极暗/极端角度条件图）|
| **最终 v2** | **18.7x** | 最佳正面视角 + 对齐 |

### Best / Worst DiT Cases

| Best Cases | DiT CD | Ratio | Worst Cases | DiT CD | Ratio |
|-----------|--------|-------|-------------|--------|-------|
| Tabasco_sauce | 0.000038 | 1.3x | armband | 0.008496 | 176.9x |
| antenna | 0.000064 | 4.4x | alligator | 0.007177 | 142.1x |
| aerosol_can | 0.000089 | 2.5x | anklet | 0.005834 | 236.9x |
| Sharpie | 0.000109 | 12.7x | alarm_clock | 0.003082 | 36.2x |
| apron | 0.000135 | 6.6x | CD_player | 0.002324 | 39.9x |

## 关键发现

1. **坐标轴错位**是评估 3D 生成模型时最容易踩的坑（GT 和 DiT 坐标系不一致，未对齐前 gap 被高估 8 倍）
2. **条件图质量对 DiT 性能影响巨大**（随机视角 → 正面视角，CD ratio 从 49x 降至 18.7x，2.6 倍改善）
3. 最佳 case（Tabasco_sauce, antenna 等）DiT 已接近 VAE 水平（1-5x），说明 DiT 在"简单形状 + 好条件图"下能力不差
4. 最差 case 仍有 100x+ gap，主要集中在细长/复杂形状（alligator, anklet, armband）
5. `data_toolkit/datasets/` 模块缺失，标准测试集（Toys4K、Sketchfab）暂不可用

## 工具链

```
scripts/
├── eval_metrics.py          # CD, F-score, NC + 24-rotation alignment
├── reeval_gap.py            # 从保存的 OBJ 重新计算对齐指标 + 预览图
├── gap_measurement.py       # 完整评估管线 (多GPU: --rank/--world_size)
├── prepare_pilot_data.py    # Objaverse 数据下载
└── render_blender_cond.py   # Blender CYCLES 条件图像渲染（含最佳视角选择）
```

## 修订后优先级

```
Phase 1 (revised):
  P2b → DiT 改进 (条件注入 / normal guidance / timestep sampling)  ← 提升
  P1c → Mesh refinement (不依赖 VAE/DiT 选择)
  P1d → 小孔修复 (不依赖 VAE/DiT 选择)
  P1a → Curvature-weighted loss (降低，VAE 不是瓶颈)
  P1b → 渲染监督增强 (降低)
```
