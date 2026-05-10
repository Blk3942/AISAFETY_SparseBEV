# `safety_critical_eval` 总结说明（含关键公式）

本文基于当前目录下已有文件进行汇总：

- `README.md`
- `nuscenes/eval/detection/configs/detection_safety_critical.json`
- `nuscenes/eval/detection/configs/CONFIG.md`
- `nuscenes/eval/detection/algo.py`
- `nuscenes/eval/detection/data_classes.py`
- `OFFLINE_VS_SAFETY_COMPARISON.md`

---

## 1. 这套评估在做什么

`safety_critical_eval` 是在官方 nuScenes 检测评测流程基础上的安全关键定制版，核心目标是：

- **更关注近距离目标**（通过 `safety_max_dist`）
- **更关注高风险类别**（通过 `class_ap_weights`）
- **让误差对风险语境更敏感**（`scale_err` 维度加权、`orient_err/vel_err` 角度分段加权）

> 结论：它输出的是“安全关键口径指标”，不是官方榜单口径，不能直接与官方 NDS 横向比较。

---

## 2. 与官方流程一致的部分

- 匹配主框架仍是 DETECTION eval 的标准流程：
  - 按类、按中心距离阈值（`dist_ths`）做匹配
  - 计算 AP（`calc_ap`）与 TP 误差（`calc_tp`）
- 仍使用中心距离匹配（`dist_fcn=center_distance`）
- 仍保留 NDS 的总体结构（mAP + 5 个 TP 分项）

---

## 3. Safety 版新增/改动点

## 3.1 距离过滤（近场优先）

只保留满足以下条件的框：

\[
\text{ego\_dist} < \min(\text{class\_range[class]}, \text{safety\_max\_dist})
\]

含义：先满足类别半径，再满足全局安全半径 `safety_max_dist`。

---

## 3.2 mAP 跨类聚合改为类别加权

先得到每类在各阈值上的平均 AP（`mean_dist_aps[class]`），再做加权平均：

\[
\text{mean\_ap}
=
\frac{\sum_c w_c \cdot \text{AP}_c}{\sum_c w_c}
\]

- \(w_c\)：`class_ap_weights[c]`
- \(\text{AP}_c\)：该类在 `dist_ths` 上平均后的 AP

---

## 3.3 `scale_err` 改为 `w/l/h` 相对误差加权

单个匹配对的尺度误差定义为：

\[
e_w=\frac{|w^{pred}-w^{gt}|}{\max(|w^{gt}|,\epsilon)},
\quad
e_l=\frac{|l^{pred}-l^{gt}|}{\max(|l^{gt}|,\epsilon)},
\quad
e_h=\frac{|h^{pred}-h^{gt}|}{\max(|h^{gt}|,\epsilon)}
\]

\[
\text{scale\_err}
=
\frac{\alpha_w e_w + \alpha_l e_l + \alpha_h e_h}{\alpha_w+\alpha_l+\alpha_h}
\]

- \(\alpha_w,\alpha_l,\alpha_h\)：`scale_dim_weights.w/l/h`
- \(\epsilon=10^{-6}\)

---

## 3.4 `orient_err` 与 `vel_err` 角度分段加权

先计算角度偏差（自车航向 vs 目标 yaw）：

\[
\Delta\theta = \left| \text{wrapToPi}(yaw_{obj}-yaw_{ego}) \right| \in [0,\pi]
\]

两段权重函数（以 `split_deg` 为分界）：

\[
w(\Delta\theta)=
\begin{cases}
w_{\text{small}}, & \Delta\theta \le \theta_{split}\\
w_{\text{large}}, & \Delta\theta > \theta_{split}
\end{cases}
\]

然后对误差序列采用加权累计均值（实现中用 `cumweighted_mean`）：

\[
\overline{e}_{1:t}
=
\frac{\sum_{i=1}^{t} w_i e_i}{\sum_{i=1}^{t} w_i}
\]

分别应用于：

- `orient_err`（使用 `orient_weighting`）
- `vel_err`（使用 `vel_weighting`）

---

## 4. NDS 仍按同结构计算（但输入已被定制）

先把 TP 误差转分数：

\[
\text{tp\_score}_k=\max(0,\,1-\text{tp\_error}_k)
\]

再聚合：

\[
\text{NDS}
=
\frac{w_{ap}\cdot \text{mean\_ap} + \sum_{k\in\{\text{trans,scale,orient,vel,attr}\}}\text{tp\_score}_k}
{w_{ap}+5}
\]

- \(w_{ap}=\text{mean\_ap\_weight}\)
- 注意：这里的 `mean_ap/scale/orient/vel` 已经是 safety 口径版本

---

## 5. 默认配置（当前项目）

来自 `detection_safety_critical.json`：

- `safety_max_dist = 30.0`
- `class_ap_weights`：
  - `pedestrian=3.0`, `motorcycle=2.0`, `bicycle=2.0`，其余多为 `1.0`
- `scale_dim_weights`：
  - `w=5.0, l=5.0, h=3.0`
- `orient_weighting`：
  - `split_deg=45`, `w_small=5.0`, `w_large=2.0`
- `vel_weighting`：
  - `split_deg=45`, `w_small=5.0`, `w_large=2.0`

这体现了“近场 + 弱势交通参与者 + 朝向/速度风险语境”的评估偏好。

---

## 6. 实践建议

- 对外/论文/榜单对比：使用 `offline_nuscenes_eval`
- 安全目标优化：并行使用 `safety_critical_eval`
- 推荐报告方式：同一模型同时给出两套分数，避免“只看官方均值”掩盖安全场景退化

