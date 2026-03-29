# FUSA 检测评测加权配置说明

本文说明 `fusa_weighting.example.json` 及同类 JSON 的字段含义与用法。加权逻辑在 `fusa_eval` 内实现，**仅在使用 `tools/fusa_eval.py` 并传入 `--fusa-weighting` 时生效**；不传该参数时与官方 nuScenes 检测评测一致。

---

## 1. 如何启用

在工程根目录（`AISAFETY_SparseBEV`）下执行：

```bash
python tools/fusa_eval.py <results_nusc.json> \
  --dataroot <nuScenes根目录> \
  --version v1.0-mini \
  --fusa-weighting fusa_eval/configs/fusa_weighting.example.json \
  --output-dir <输出目录>
```

- 将路径换成你自己的 `results_nusc.json` 与 nuScenes 数据根目录。
- 也可复制 `fusa_weighting.example.json` 为自定义文件名后修改，再通过 `--fusa-weighting` 指向该文件。

**与官方指标对齐的回归配置**：使用 `fusa_weighting.identity.json`（全 1 权重），用于确认实现与官方 `DetectionEval` 一致。

---

## 2. 单个 GT 框的权重公式

对每个参与评测的 **真值（GT）框**，先计算平面自车距离 `ego_dist`（与 nuScenes devkit 中 `add_center_dist` 一致，单位：米），再按下式相乘：

\[
w_{\mathrm{GT}} = w_{\mathrm{class}} \times w_{\mathrm{ego}}(d) \times w_{\mathrm{attr}}
\]

- \(w_{\mathrm{class}}\)：该检测类别在 `class_weights` 中的系数（未写明的类别默认为 `1.0`）。
- \(w_{\mathrm{ego}}(d)\)：由 `ego_distance_bins` 根据距离 \(d=\texttt{ego\_dist}\) 查表得到。
- \(w_{\mathrm{attr}}\)：由 `attribute_weights` 根据类别与属性字符串得到。

最终 \(w_{\mathrm{GT}}\) 会截断为不小于 `1e-9`，避免为 0。

该权重用于：

- **加权 recall 分母** `npos`（某类所有 GT 的 \(w_{\mathrm{GT}}\) 之和）；
- **匹配后 TP 误差曲线**的加权累积（与每次匹配到的 GT 权重一致）。

**Precision** 仍按官方定义为 **TP/(TP+FP)**（按预测条数计数），避免权重过大时出现大于 1 的“伪精度”。

---

## 3. 字段说明

### 3.1 `description`（可选）

字符串，仅作文档说明，不参与计算。

---

### 3.2 `class_weights`

- **类型**：对象，键为 nuScenes **检测类名**（须与下列一致），值为正数。
- **检测类名列表**（共 10 类）：

  `car`, `truck`, `bus`, `trailer`, `construction_vehicle`, `pedestrian`, `motorcycle`, `bicycle`, `traffic_cone`, `barrier`

- **未出现的类别**：视为 `1.0`。
- **第二层作用**：在得到每类 `mean_dist_aps` 与各类 TP 误差后，**跨类聚合 mAP 与 mATE/mASE/…** 时，再按该类 `class_weights` 做加权平均（见 `FusaDetectionMetrics`）。

---

### 3.3 `ego_distance_bins`

- **类型**：数组，每项为 `{ "up_to_m": number | null, "weight": number }`。
- 配置加载后会按 `up_to_m` **升序**排序（`null` 视为正无穷）。
- **查表规则**：对平面距离 \(d\)，从左到右找**第一个**满足 \(d < \texttt{up\_to\_m}\) 的项，使用其 `weight`；若列表为空，则等价于单条 `{ "up_to_m": +∞, "weight": 1.0 }`。

示例（与 `example.json` 一致）：

| 条件 | 权重 |
|------|------|
| \(d < 10\) m | 1.5 |
| \(10 \le d < 30\) m | 1.0 |
| \(d \ge 30\) m | 0.7 |

---

### 3.4 `attribute_weights`

- **`default`**：全局默认系数（未命中 `per_class` 时仍可能用到该默认值逻辑，见代码）。
- **`per_class`**：按检测类名分组，每类下为「属性名 → 系数」的映射。

**特殊键名**：

| 键 | 含义 |
|----|------|
| `_empty` | 该 GT **无** attribute 时（空字符串） |
| `_default` | 该类已配置 `per_class`，但当前属性名不在表中时 |

**属性名**须为 nuScenes 官方字符串，例如：

- `pedestrian.moving`, `pedestrian.sitting_lying_down`, `pedestrian.standing`
- `cycle.with_rider`, `cycle.without_rider`
- `vehicle.moving`, `vehicle.parked`, `vehicle.stopped`

仅当 `per_class` 中**包含**该类时，才会按该类的 `_empty` / `_default` / 具体属性名解析；否则回退到 `attribute_weights.default`。

---

### 3.5 `nds`（可选）

用于调整 **nuScenes 风格 NDS** 的聚合方式（在加权 mAP 与加权 TP 误差已算出的前提下）：

| 子字段 | 含义 |
|--------|------|
| `mean_ap_weight` | NDS 中 mAP 项的权重（与官方 `detection_cvpr_2019` 中 `mean_ap_weight` 对应；不传则沿用检测配置 `DetectionConfig` 中的值）。 |
| `tp_metric_multipliers` | 对五个 TP 指标在换算成「得分」`max(0, 1 - err)` 后的**额外倍率**，键名为：`trans_err`, `scale_err`, `orient_err`, `vel_err`, `attr_err`。未写的键视为 `1.0`。 |

NDS 公式仍为：

\[
\text{NDS} = \frac{w_{\mathrm{mAP}} \cdot \mathrm{mAP} + \sum_k s_k}{w_{\mathrm{mAP}} + 5}
\]

其中 \(s_k\) 为各 TP 分项得分（已乘 `tp_metric_multipliers` 时按实现为准），\(w_{\mathrm{mAP}}\) 为 `mean_ap_weight`。

---

## 4. 输出结果

启用 `--fusa-weighting` 时，`metrics_summary.json` 中会包含：

- `fusa_enabled: true`
- `fusa_weighting`：完整配置回显（便于复现实验）

此时顶层的 `mean_ap`、`nd_score`、`tp_errors` 等即为 **FUSA 加权后的指标**；若需与榜单完全一致的官方指标，请**不使用** `--fusa-weighting`，或另跑一份 `identity` 配置对比。

---

## 5. 相关文件

| 文件 | 说明 |
|------|------|
| `fusa_weighting.example.json` | 示例：行人/两轮等加权、距离分箱、车辆属性等 |
| `fusa_weighting.identity.json` | 全 1，用于与官方指标对齐检查 |
| `tools/fusa_eval.py` | 命令行入口 |

实现代码位置（仅作查阅）：`fusa_eval/nuscenes/eval/detection/fusa_weighting.py`、`algo_fusa.py`、`metrics_fusa.py`、`evaluate.py`。
