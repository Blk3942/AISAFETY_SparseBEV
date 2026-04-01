# `detection_safety_critical.json` 配置说明

本文说明 `safety_critical_eval/nuscenes/eval/detection/configs/detection_safety_critical.json` 的各字段含义、单位、默认值与对评测结果的影响。

该配置用于脚本 `tools/safety_critical_nuscenes_eval.py`（默认 `--config detection_safety_critical`）。

---

## 1. 与官方 nuScenes detection 配置一致的字段

这些字段与 `offline_nuscenes_eval` 的 `detection_cvpr_2019.json` 含义一致：

- **`class_range`**（dict[class->meters]）
  - 各类别的最大评测半径（单位：米）。
  - 目标到自车的 2D 距离 `ego_dist` 超过该半径会被过滤（GT 与预测都会过滤）。
  - **取值范围**
    - **强约束**：所有值应为 **正数**（>0）。
    - **建议范围**：\[0, 100\] m（按业务需求，越大越接近官方全量评测）。

- **`dist_fcn`**（string）
  - 匹配函数名。当前实现只支持 `center_distance`（BEV 平面中心点距离）。
  - **取值范围**
    - **强约束**：只能是 `center_distance`。

- **`dist_ths`**（list[float]，meters）
  - 计算 AP 时使用的中心距离阈值集合（单位：米）。
  - 每个类别会在每个阈值上计算一次 AP，然后对这些阈值求平均得到该类别的 `mean_dist_ap`。
  - **取值范围**
    - **强约束**：列表非空；每个阈值应为 **正数**（>0）。
    - **建议范围**：常用 \([0.1, 10]\) m；nuScenes 默认 `[0.5, 1.0, 2.0, 4.0]`。

- **`dist_th_tp`**（float，meters）
  - 计算 TP 类误差（`trans_err/scale_err/orient_err/vel_err/attr_err`）时使用的匹配阈值（单位：米）。
  - 要求 `dist_th_tp` 必须在 `dist_ths` 中。
  - **取值范围**
    - **强约束**：必须满足 `dist_th_tp ∈ dist_ths` 且 `dist_th_tp > 0`。
    - **建议范围**：通常取 `dist_ths` 中“中等严格”的一个值（nuScenes 默认 2.0m）。

- **`min_recall`**（float，0~1）
  - AP/TP 误差计算时，截断低召回区间的下界（nuScenes 默认 0.1）。
  - **取值范围**
    - **强约束**：\[0, 1\]。
    - **建议范围**：\[0.0, 0.3\]（越大越强调高 recall 区间，数值会更“苛刻”）。

- **`min_precision`**（float，0~1）
  - 计算 AP 时，precision 低于该值的部分会被裁剪（nuScenes 默认 0.1）。
  - **取值范围**
    - **强约束**：\[0, 1\)（严格小于 1）。
    - **建议范围**：\[0.0, 0.3\]。

- **`max_boxes_per_sample`**（int）
  - 每帧最多允许的预测框数量（默认 500）。
  - **取值范围**
    - **强约束**：正整数（≥1）。
    - **建议范围**：\[50, 2000\]；太小会截断预测影响上限，太大只会增加计算量。

- **`mean_ap_weight`**（int/float）
  - NDS 中 mAP 的权重 \(w\)。
  - NDS 计算：
    \[
    \mathrm{NDS} = \frac{w\cdot \mathrm{mean\_ap} + \sum_k \max(0, 1-\mathrm{tp\_errors}[k])}{w+5}
    \]
  - **取值范围**
    - **强约束**：应为非负数（≥0）。若设为 0，则 NDS 完全由 5 个 TP 分数决定。
    - **建议范围**：\[1, 10\]；nuScenes 默认 5。

---

## 2. Safety-Critical 新增字段（本策略的核心）

### 2.1 `safety_max_dist`（float，meters）

- **含义**：全局安全评测半径 \(X\)（单位：米）。
- **作用**：只有 `ego_dist < safety_max_dist` 的 GT/预测才参与后续匹配与指标计算。
- **注意**：
  - 这是在 `class_range` 的基础上**再加一层过滤**，最终有效半径为 `min(class_range[class], safety_max_dist)`。
  - **取值范围**
    - **强约束**：建议为正数（>0）。若设为 `null`（或不提供）代表不启用全局半径过滤。
    - **建议范围**：\[5, 60\] m（常见安全关键关注近距离目标）。

### 2.2 `class_ap_weights`（dict[class->weight]）

- **含义**：mAP 类别权重。
- **作用**：`mean_ap` 不再是简单类别均值，而是类别加权平均：
  \[
  \mathrm{mean\_ap}=\frac{\sum_{c} w_c \cdot \mathrm{AP}_c}{\sum_{c} w_c}
  \]
  其中 \(\mathrm{AP}_c\) 是该类在 `dist_ths` 上平均后的 `mean_dist_ap`。
- **要求**：必须包含全部 10 个检测类别键（car/truck/.../barrier）。
  
  - **取值范围**
    - **强约束**：每个权重应为非负数（≥0），且所有权重不应全为 0（否则会退化）。
    - **建议范围**：\[0, 5\]；一般用 0（忽略该类）、1（默认）、2~5（强调）。

### 2.3 `scale_dim_weights`（dict，keys: `w/l/h`）

- **含义**：`scale_err` 的维度权重（宽/长/高）。
- **作用**：本策略下的 `scale_err` **不再使用官方的 `1 - scale_iou`**，而是定义为三维尺寸相对误差的加权均值：
  \[
  e_w=\frac{|w^{pred}-w^{gt}|}{\max(|w^{gt}|,\epsilon)},\;
  e_l=\frac{|l^{pred}-l^{gt}|}{\max(|l^{gt}|,\epsilon)},\;
  e_h=\frac{|h^{pred}-h^{gt}|}{\max(|h^{gt}|,\epsilon)}
  \]
  \[
  \mathrm{scale\_err}=\frac{\alpha_w e_w + \alpha_l e_l + \alpha_h e_h}{\alpha_w+\alpha_l+\alpha_h}
  \]
- **单位**：无量纲。
  - **取值范围**
    - **强约束**：`w/l/h` 必须齐全；权重建议为非负数（≥0），且不应全为 0。
    - **建议范围**：\[0, 5\]。

### 2.4 `orient_weighting`（dict）

- **字段**
  - **`split_deg`**：角度分段阈值（单位：度，范围通常 0~180）
  - **`w_small`**：当角度偏差 \(\le split\_deg\) 时的权重
  - **`w_large`**：当角度偏差 \(> split\_deg\) 时的权重
  - **取值范围**
    - **强约束**：`split_deg` 建议在 \([0, 180]\)；`w_small/w_large` 建议为非负数（≥0），且不要同时为 0。
    - **建议范围**：
      - `split_deg`: \([0, 90]\) 更常用（区分“近同向/迎面”与“横向”）
      - `w_small/w_large`: \([0, 5]\)
- **角度偏差的定义**：\(|yaw_{obj} - yaw_{ego}|\)，并 wrap 到 \([0,\pi]\)。
  - `yaw_ego` 来自该帧 `ego_pose.rotation`
  - `yaw_obj` 使用 **GT box 的 rotation** 提取（本策略实现如此）
- **作用**：对每个匹配到的 TP 框，`orient_err` 乘以该权重，并用“加权累计均值”汇总到 `DetectionMetricData` 曲线上。

### 2.5 `vel_weighting`（dict）

字段含义与 `orient_weighting` 完全一致，但作用对象是 `vel_err`：

- `vel_err` 先按官方定义计算（速度向量 L2），再按上述角度分段权重加权，并用“加权累计均值”汇总。
  - **取值范围**
    - 与 `orient_weighting` 相同：`split_deg ∈ [0,180]`（建议 \([0,90]\)），`w_small/w_large ≥ 0`（建议 \([0,5]\)）。

---

## 3. 常用配置配方（示例）

> 以下仅示例思路，你可以直接改 json 中对应字段。

- **只看近距离（例如 20m）**
  - `safety_max_dist: 20.0`

- **强调弱势类别（例如 pedestrian / bicycle）**
  - `class_ap_weights.pedestrian: 2.0`
  - `class_ap_weights.bicycle: 2.0`

- **尺度更关注高度（例如高低误差更敏感）**
  - `scale_dim_weights.h: 2.0`（w/l 保持 1.0）

- **对“横向朝向”更敏感（大角度权重更大）**
  - `orient_weighting.split_deg: 45.0`
  - `orient_weighting.w_small: 1.0`
  - `orient_weighting.w_large: 3.0`

- **速度误差只在“迎面/同向”时更重要（小角度权重大）**
  - `vel_weighting.split_deg: 30.0`
  - `vel_weighting.w_small: 2.0`
  - `vel_weighting.w_large: 1.0`

---

## 4. 与官方指标的可比性说明

该配置会改变：

- `mean_ap` 的类别聚合方式（加权而非均值）
- `scale_err` 的定义（相对尺寸误差加权而非 `1-scale_iou`）
- `orient_err/vel_err` 的聚合（按角度分段加权）
- 以及额外的全局半径过滤 `safety_max_dist`

因此输出的 `nd_score` 是**安全关键定制版 NDS**，仅用于本地安全评估与策略对比，不应与官方榜单 NDS 直接比较。

