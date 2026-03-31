# `offline_nuscenes_eval`（离线 nuScenes 评测包）说明

本目录提供一套**可离线使用**的 nuScenes devkit 评测实现（主要用于 **3D 检测 NDS / mAP**），并在工程内通过脚本入口进行调用，从而在**不安装** `nuscenes-devkit` 的情况下完成与官方一致的评测。

> 更“算法细节导向”的完整说明请直接阅读：`offline_nuscenes_eval/EVALUATION.md`。

---

## 这个目录解决什么问题

- **离线评测**：在服务器/内网环境里不方便 `pip install nuscenes-devkit` 时，仍可完成 nuScenes 检测指标计算（mAP、mATE/mASE/mAOE/mAVE/mAAE、NDS）。
- **与 MMDet3D 对齐**：评测流程与 MMDetection3D 在 `NuScenesDataset._evaluate_single()` 中调用的 `DetectionEval` 逻辑一致（读 `results_nusc.json` → 运行评测 → 输出 `metrics_summary.json`）。
- **最小依赖**：提供精简版 `nuscenes.NuScenes` 数据库加载器（见 `nuscenes/nuscenes_db.py`），避免加载与检测评测无关的重型组件。

---

## 代码结构速览

- **检测评测核心**
  - `nuscenes/eval/detection/evaluate.py`：`DetectionEval`（评测主入口类）
  - `nuscenes/eval/detection/algo.py`：匹配、PR 曲线、AP 与 TP 误差累计
  - `nuscenes/eval/detection/data_classes.py`：指标聚合与 **NDS** 计算
  - `nuscenes/eval/detection/configs/detection_cvpr_2019.json`：默认配置（官方 CVPR 2019）
- **通用评测工具**
  - `nuscenes/eval/common/loaders.py`：读取预测/GT（要求格式与官方一致）
  - `nuscenes/eval/common/config.py`：`config_factory()`
- **精简 NuScenes DB**
  - `nuscenes/nuscenes_db.py`：精简版 `NuScenes`（仅满足评测所需的 JSON 表查询、速度估计等）
- **工程入口（推荐用这个跑）**
  - `tools/official_nuscenes_eval.py`：把 `offline_nuscenes_eval/` 加入 `sys.path` 优先级最高，然后 `import nuscenes` 并调用 `DetectionEval`

---

## 如何运行（推荐方式）

在项目根目录执行：

```bash
python tools/official_nuscenes_eval.py <path/to/results_nusc.json> --dataroot <path/to/nuscenes_root> --version v1.0-trainval
```

关键参数：

- **`result_path`**：`results_nusc.json`，需为 nuScenes devkit / MMDet3D 兼容的提交格式
- **`--dataroot`**：nuScenes 数据根目录（包含 `v1.0-trainval/` 或 `v1.0-mini/` 等）
- **`--version`**：
  - `v1.0-trainval` 默认 `eval_set=val`
  - `v1.0-mini` 默认 `eval_set=mini_val`
  - 其他版本需显式传 `--eval-set`

输出：

- 默认输出到 `results_nusc.json` 同级目录下的 `nuscenes_eval_out/`
- 其中最重要的是 `metrics_summary.json`（含 `nd_score`、`mean_ap`、`tp_errors` 等）

---

## 结果里常看的指标含义

- **mAP**：对 10 类检测结果，在 4 个中心距离阈值 \(0.5, 1, 2, 4\)（米）上的 AP 做平均，再在类别维平均。
- **mATE/mASE/mAOE/mAVE/mAAE**：分别对应 `trans_err/scale_err/orient_err/vel_err/attr_err` 的类间平均（在 `dist_th_tp=2.0m` 下计算 TP 误差）。
- **NDS**：默认配置中 `mean_ap_weight=5`，即 mAP 权重为 5，5 个 TP 子指标各 1，总计 10 份归一化。

`tools/official_nuscenes_eval.py` 里还提供了 `nds_from_summary()`，会用 `metrics_summary.json` 字段**按官方公式重算一次 NDS**做自检。

---

## 常见问题与排查

- **报错/断言：预测 sample_token 与 GT 不一致**
  - 说明你的 `results_nusc.json` 覆盖的 `sample_token` 集合与所选 `eval_set` 不匹配（多/少帧都会失败）。
  - 重点检查：`--version`、`--eval-set`、以及你导出预测时使用的数据划分（val/mini_val）。

- **想画曲线（PR/TP curve）但报缺依赖**
  - 曲线渲染会引入额外可视化依赖（如 matplotlib 等）。若你只要数值指标，保持默认 `--render-curves` 关闭、`--plot-examples 0` 即可。

---

## 进一步阅读

- **算法与数据流细节**：`offline_nuscenes_eval/EVALUATION.md`
- **入口脚本**：`tools/official_nuscenes_eval.py`
- **上游版权信息**：`offline_nuscenes_eval/ATTRIBUTION.txt`

