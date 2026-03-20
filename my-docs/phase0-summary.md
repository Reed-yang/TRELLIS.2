# Phase 0: Gap Measurement — Summary

> 快速参考文档。详细记录见 `logs/progress.md` 和 `logs/findings.md`。

## 结论

**DiT 是当前系统瓶颈**，不是 VAE。DiT 生成质量 (CD) 比 VAE 重建上界差 **~49 倍**（对齐后）。

→ Phase 1 应将 **DiT 改进 (P2b)** 提升为最高优先级。

## 数据

- 30 个 Objaverse 模型（多样 LVIS 类别），Blender CYCLES 渲染条件图像
- 24 轴旋转对齐修正坐标系差异

| 指标 | VAE 重建 | DiT 生成 | Gap |
|------|---------|---------|-----|
| CD | 0.000071 | 0.00348 | 49x |
| F-score | 0.775 | 0.321 | 2.4x |
| NC | 0.932 | 0.741 | 1.3x |

## 关键发现

1. **坐标轴错位**是评估 3D 生成模型时最容易踩的坑（GT 和 DiT 坐标系不一致，未对齐前 gap 被高估 8 倍）
2. 法线图作为条件图像反而比 Blender 渲染效果好（可能因为几何信息更直接）
3. `data_toolkit/datasets/` 模块缺失，标准测试集（Toys4K、Sketchfab）暂不可用

## 工具链

```
scripts/
├── eval_metrics.py          # CD, F-score, NC + 24-rotation alignment
├── gap_measurement.py       # 完整评估管线 (多GPU: --rank/--world_size)
├── prepare_pilot_data.py    # Objaverse 数据下载
└── render_blender_cond.py   # Blender CYCLES 条件图像渲染
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
