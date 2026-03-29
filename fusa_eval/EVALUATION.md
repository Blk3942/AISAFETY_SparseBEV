# fusa_eval：nuScenes 检测评测逻辑说明（FUSA 专用副本）

本目录由 **`offline_nuscenes_eval/`** 复制而来，供 **`tools/fusa_eval.py`** 单独依赖；可在本目录内修改而不影响标准 NDS 包。

本文说明目录内 **nuScenes 3D 目标检测** 离线评测的数据流、匹配规则、指标定义。算法实现来自 [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit)（Apache-2.0），与 MMDetection3D 中 `NuScenesDataset._evaluate_single()` 所调用的逻辑一致。

---

## 1. 目录与职责

| 路径 | 作用 |
|------|------|
| `nuscenes/eval/detection/` | **核心**：`DetectionEval`、`algo.py`（匹配与 AP）、`data_classes.py`（配置与 NDS）、`constants.py` |
| `nuscenes/eval/common/` | **数据**：`loaders.py`（读预测/GT、距离、过滤）、`data_classes.py`（`EvalBoxes`）、`utils.py`（距离/朝向等）、`config.py`（`config_factory`） |
| `nuscenes/eval/detection/configs/detection_cvpr_2019.json` | **默认评测配置**（距离阈值、类别范围、NDS 权重等） |
| `nuscenes/nuscenes_db.py` | **精简数据库**：只读 JSON 表，提供 `get` / `box_velocity` / reverse index，供 `load_gt`、`add_center_dist`、`filter_eval_boxes` 使用 |
| `nuscenes/utils/box.py` | **最小 `Box`**：仅 `corners()`，用于自行车架过滤 |
| `nuscenes/utils/data_classes.py` | 导出 `Box` |
| `nuscenes/utils/geometry_utils.py` | `points_in_box` 等（官方拷贝） |
| `nuscenes/utils/splits.py` | 官方 scene 划分（train/val/test/mini 等） |
| `ATTRIBUTION.txt` | 上游版权与许可证说明 |

项目根目录下 **`tools/fusa_eval.py`** 会将本目录（**`fusa_eval/`**）加入 `sys.path` 首位，从而 **无需 `pip install nuscenes`** 即可 `import nuscenes` 并跑完整评测。

---

## 2. 端到端流程（`DetectionEval`）

入口类：`nuscenes.eval.detection.evaluate.DetectionEval`（历史上别名 `NuScenesEval`）。

### 2.1 初始化 `__init__`

按顺序完成：

1. **`load_prediction(result_path, max_boxes_per_sample, DetectionBox)`**  
   - 读取 `results_nusc.json`：`meta` + `results`（按 `sample_token` 组织预测框列表）。  
   - 反序列化为 `EvalBoxes`，每框为 `DetectionBox`（含 `translation/size/rotation/velocity/detection_name/detection_score/attribute_name` 等）。  
   - 断言每帧预测数 ≤ `max_boxes_per_sample`（默认 500）。

2. **`load_gt(nusc, eval_set, DetectionBox)`**  
   - 用 `create_splits_scenes()` 得到当前 split 的 **scene 名称列表**。  
   - 遍历数据集中所有 `sample`，仅保留所属 `scene` 在 split 内的样本。  
   - 对每个样本的 `sample_annotation`：映射到 10 类检测名（`category_to_detection_name`），构造 GT `DetectionBox`（`detection_score=-1`，速度由 `nusc.box_velocity` 估计）。  
   - **要求**：预测 JSON 中出现的 `sample_token` 集合必须与该 split 的 GT **完全一致**（多一个少一个都会 `assert` 失败）。

3. **`add_center_dist(nusc, boxes)`**  
   - 对每个框，根据该 sample 的 `LIDAR_TOP` 关键帧的 **ego pose**，计算框中心相对自车的平移 `ego_translation`，并写入 `ego_dist`（用于 XY 平面距离过滤）。

4. **`filter_eval_boxes(nusc, boxes, class_range)`**（对预测与 GT **分别**做）  
   - **距离**：`ego_dist < class_range[类别]`（米），不同类半径不同（见配置 JSON）。  
   - **点数**：去掉 `num_pts == 0` 的 GT/预测（评估框内无雷达/激光点的目标）。  
   - **自行车架**：对 `bicycle` / `motorcycle`，若中心点落在官方标注的 `static_object.bicycle_rack` 3D 框内则剔除（需 `Box.corners` + `points_in_box`）。

### 2.2 核心计算 `evaluate()`

1. **双重循环**：对每个 **类别** `class_name` × 每个 **中心距离阈值** `dist_th ∈ {0.5, 1, 2, 4}` 米，调用 **`accumulate(...)`**（见第 3 节），得到 `DetectionMetricData`（101 个 recall 插值点上的 precision、置信度及各 TP 误差曲线）。

2. **AP**：对每个 `(class, dist_th)`，用 **`calc_ap(metric_data, min_recall, min_precision)`** 在 recall≥`min_recall` 段上对 precision 做裁剪与平均（官方实现，与论文设定一致）。

3. **TP 子指标**：对 **固定阈值 `dist_th_tp = 2.0` m** 下的 `metric_data`，按类调用 **`calc_tp(..., metric_name)`**，得到 `trans_err / scale_err / orient_err / vel_err / attr_err`。  
   - **类别特例**：`traffic_cone` 的 `attr_err/vel_err/orient_err` 记为 `nan`；`barrier` 的 `attr_err/vel_err` 记为 `nan`（与官方一致，后续 `nanmean` 聚合）。

4. 汇总为 **`DetectionMetrics`**：类间平均得到 `mean_ap`、`tp_errors`，再算 **NDS**（第 4 节）。

### 2.3 `main()` 与产物

- **`evaluate()`** 后若 `render_curves=True`，会调用 `render()` 画 PR/TP 曲线（依赖 matplotlib 等，见第 6 节）。  
- 写出 **`metrics_summary.json`**、`metrics_details.json`（与 pip 版 devkit 相同字段）。  
- 控制台打印 `mAP、mATE/mASE/mAOE/mAVE/mAAE、NDS`（`mATE` 等即 `tp_errors` 的展示名）。

---

## 3. 匹配与 AP：`accumulate`（`algo.py`）

对固定 **类别** 与 **距离阈值 `dist_th`**：

1. 统计该类 GT 总数 `npos`。若为 0，返回「无预测」形式的 `DetectionMetricData`。

2. 收集所有该类的预测，按 **`detection_score` 降序** 排序，依次处理每个预测：

   - 在同一 `sample_token` 内、尚未匹配的 GT 中，找 **距离函数最小** 的 GT（默认 **`center_distance`**：XY 平面中心 L2 距离）。  
   - 若最小距离 **&lt; `dist_th`** → 记为 **TP**：记录 `trans_err`（中心 L2）、`vel_err`（速度 L2）、`scale_err`（1 − scale IoU）、`orient_err`（朝向差，`barrier` 用 π 周期）、`attr_err`（1 − 属性是否一致）等。  
   - 否则 → **FP**。  
   - 每个 GT 在单帧内 **最多匹配一个** 预测（贪心、按分数从高到低）。

3. 沿排序序列累积 TP/FP，得到 precision–recall 曲线，再 **插值到 101 个 recall 点**；同时将各误差沿置信度对齐插值，供 `calc_tp` 在指定 recall 段上取平均。

**要点**：这是 **中心距离匹配的多阈值 AP**，不是 BEV IoU 匹配；四个 `dist_th` 的 AP 再对阈值维平均，得到该类 `mean_dist_aps`，再对 10 类平均得 **mAP**。

---

## 4. NDS（nuScenes Detection Score）

定义在 `DetectionMetrics.nd_score`（`detection/data_classes.py`）。

1. 对 5 个 TP 误差在 **类间** 取平均（`nanmean`），得到 `tp_errors`：`trans_err, scale_err, orient_err, vel_err, attr_err`。  
   - 对外展示名：**mATE, mASE, mAOE, mAVE, mAAE**（见 MMDet3D `ErrNameMapping`）。

2. **TP 分数**：`tp_scores[k] = max(0, 1 - tp_errors[k])`。

3. **NDS**（默认 `mean_ap_weight = 5`，见 `detection_cvpr_2019.json`）：

\[
\text{NDS} = \frac{w \cdot \text{mAP} + \sum_{k} \text{tp\_scores}[k]}{w + 5},
\quad w = 5
\]

即 **mAP 占 5 份权重**，5 个 TP 指标各占 1 份，再除以 10 **归一化**到约 \([0,1]\) 量级。

`tools/fusa_eval.py` 中的 **`nds_from_summary()`** 用同一公式对 `metrics_summary.json` 做自检，应与 `nd_score` 一致。

---

## 5. 默认配置 `detection_cvpr_2019.json`

| 字段 | 含义 |
|------|------|
| `class_range` | 各类评估半径（米），超出圆域的框不参与评测 |
| `dist_fcn` | `center_distance`（仅支持这一种；其他字符串会报错） |
| `dist_ths` | AP 用的四个中心距离阈值（米） |
| `dist_th_tp` | **TP 误差（mATE 等）** 使用的匹配阈值，固定为 **2.0** |
| `min_recall` / `min_precision` | AP 曲线裁剪下界（默认 0.1） |
| `max_boxes_per_sample` | 每帧最多预测框数 |
| `mean_ap_weight` | NDS 中 mAP 的权重 **w**（默认 5） |

通过 `nuscenes.eval.common.config.config_factory('detection_cvpr_2019')` 加载。

---

## 6. 离线包中的修改与限制

### 6.1 与 pip 版 `nuscenes` 的差异

- **`nuscenes/nuscenes_db.py`**：替代完整 `nuscenes.NuScenes`，不加载地图 raster、不加载 lidarseg/panoptic、无 `NuScenesExplorer`，仅满足 **检测评测** 所需表结构与查询。  
- **`eval/detection/evaluate.py`**：将 **`render` 相关 import 延迟**到 `render()` 与 `plot_examples>0` 分支，避免默认评测依赖 matplotlib / `LidarPointCloud`。  
- **`utils/data_classes.py`**：仅导出精简 **`Box`**，避免拉入完整 `data_classes.py` 的 cv2 等依赖。

### 6.2 使用建议

- **推荐**：`render_curves=False`、`plot_examples=0`（与 `tools/fusa_eval.py` 默认一致）→ 仅需 **numpy、pyquaternion、tqdm**。  
- **若** 需要 PDF 曲线或 BEV 可视化：需安装 **matplotlib** 等，并补齐 devkit 中 `render` / `visualize_sample` 对 `LidarPointCloud` 等符号的依赖（或改用完整 pip `nuscenes`）。

---

## 7. 如何运行（与工程衔接）

```bash
cd AISAFETY_SparseBEV
python tools/fusa_eval.py /path/to/submission/results_nusc.json \
  --dataroot /path/to/nuscenes \
  --version v1.0-trainval
```

- **`--version v1.0-trainval`** 且未指定 `--eval-set` 时，默认 **`eval_set=val`**（与 MMDet3D 一致）。  
- **`--version v1.0-mini`** 默认 **`mini_val`**。  
- 输出目录默认在结果 JSON 同级的 `nuscenes_eval_out/`，内含 **`metrics_summary.json`**。

MMDet3D / 本仓库 **`val.py`** 流程等价于：模型推理 → 格式化为 `results_nusc.json` → 调用同一套 `DetectionEval.main(render_curves=False)` → 读取 `nd_score`、`mean_ap` 等写进 `detail['pts_bbox_NuScenes/...']`。

**说明**：本目录为 FUSA 专用评测包；标准 NDS 请使用 **`offline_nuscenes_eval/`** + **`tools/official_nuscenes_eval.py`**。

---

## 8. 参考链接

- nuScenes 检测挑战说明：<https://www.nuscenes.org/object-detection>  
- devkit 源码：<https://github.com/nutonomy/nuscenes-devkit>  
- 本目录版权声明：`ATTRIBUTION.txt`
