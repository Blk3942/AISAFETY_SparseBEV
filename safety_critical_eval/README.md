# `safety_critical_eval`（本地 Safety-Critical nuScenes 检测评测）

本目录基于工程内的离线 nuScenes devkit（`offline_nuscenes_eval/`）新建一套**可配置的安全关键评估策略**，用于在本地生成“非官方 NDS/mAP/TP error”指标（更偏向近距离、不同类别权重、以及对朝向/速度/尺度的安全敏感加权）。

> 注意：该策略会改变 mAP 与部分 TP error 的定义，因此结果**不等同**官方 nuScenes NDS，不能与榜单直接对比；适用于内部分析与安全评测。

---

## 功能点（与你的需求一一对应）

1. **距离过滤（全局半径 X）**：只有目标到自车的距离 \(\lt X\) 的 GT/预测才纳入评估（X 为配置量）。
2. **mAP 类别加权**：mAP 在类别维度做加权平均（每类权重为配置量）。
3. **scale_err 维度加权**：对 `w/l/h` 三个尺寸误差分别加权（权重为配置量）。
4. **orient_err 分段加权**：基于“自车航向 vs 目标 yaw”的角度偏差（两个区间）对 `orient_err` 加权（区间范围与权重为配置量）。
5. **vel_err 分段加权**：同上，对 `vel_err` 加权（两个区间）。

---

## 如何运行

在项目根目录执行：

```bash
python tools/safety_critical_nuscenes_eval.py <path/to/results_nusc.json> --dataroot <path/to/nuscenes_root> --version v1.0-trainval
```

默认输出到 `results_nusc.json` 同级目录的 `safety_critical_eval_out/`，其中包含：

- `metrics_summary.json`
- `metrics_details.json`

---

## 配置文件

默认配置位于：

- `safety_critical_eval/nuscenes/eval/detection/configs/safety_critical.json`

你可以通过脚本参数 `--config safety_critical` 选择该配置（脚本默认即用它），或自行新增 json 文件并通过 `--config <name>` 选择。

