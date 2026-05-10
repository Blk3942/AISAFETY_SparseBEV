# `offline_nuscenes_eval` vs `safety_critical_eval` 评估对比说明

本文对比本项目中的两套 nuScenes 检测评估实现：

- **`offline_nuscenes_eval`**：官方口径离线评估（与 nuScenes devkit / MMDet3D 一致）
- **`safety_critical_eval`**：安全关键定制评估（在官方流程基础上引入安全权重与过滤策略）

---

## 1. 两者定位

- **`offline_nuscenes_eval`**
  - 目标：复现官方指标（mAP、mATE/mASE/mAOE/mAVE/mAAE、NDS）
  - 优点：可与公开结果/论文/榜单直接对齐
  - 使用场景：模型基线对比、对外报告、标准验证

- **`safety_critical_eval`**
  - 目标：把评估重点从“整体平均表现”转向“安全敏感表现”
  - 特点：可配置近距关注、类别权重、尺度/朝向/速度误差加权
  - 使用场景：功能安全分析、风险导向优化、场景优先级评估

---

## 2. 关键差异总览

| 维度 | `offline_nuscenes_eval`（官方口径） | `safety_critical_eval`（定制口径） |
|---|---|---|
| 目标过滤 | 按官方 `class_range` | `class_range` + 全局 `safety_max_dist`（可配置） |
| mAP 聚合 | 类别均值（官方定义） | 类别加权均值（`class_ap_weights`） |
| `scale_err` | `1 - scale_iou`（官方） | `w/l/h` 相对误差加权（`scale_dim_weights`） |
| `orient_err` | 官方角度误差 | 按“自车航向 vs 目标 yaw”分段加权 |
| `vel_err` | 官方速度误差（L2） | 按“自车航向 vs 目标 yaw”分段加权 |
| NDS 可比性 | 与官方可直接比 | 非官方定制 NDS，不建议与榜单直接比 |

---

## 3. 流程层面的核心变化（Safety 版）

`safety_critical_eval` 在官方检测流程上主要引入以下变化：

1. **近距优先过滤**
   - 仅评估 `ego_dist < safety_max_dist` 的目标（GT 与预测都过滤）
   - 让评估更聚焦“对安全更关键”的近距离交互目标

2. **类别重要性显式建模**
   - `mean_ap` 按 `class_ap_weights` 加权
   - 可提升行人/二轮等高风险类别对最终结果的影响

3. **尺度误差可解释化**
   - 用 `w/l/h` 三维相对误差加权替代单一 `1-scale_iou`
   - 便于按业务需求强调“高/宽/长”中更敏感的维度

4. **朝向与速度的风险上下文加权**
   - `orient_err` 与 `vel_err` 按角度偏差分段加权（两区间）
   - 将“横穿/迎面/同向”等相对姿态差异纳入指标权重

---

## 4. 为什么 `safety_critical_eval` 更有优势（在安全评估语境下）

以下优势针对“安全场景评估”，不是榜单可比性：

- **更符合风险分布**  
  现实风险往往集中在近距目标；`safety_max_dist` 能避免远距样本稀释近场安全问题。

- **更符合业务优先级**  
  不同类别安全影响不同（如行人 > 静态障碍）；类别加权能把评估目标与产品安全目标对齐。

- **更可解释、更可调**  
  `scale_err` 拆成 `w/l/h` 可配置权重，便于定位“哪种尺寸误差在拖后腿”。

- **更贴近交互风险**  
  朝向/速度误差与相对姿态（ego vs object yaw）相关性强；分段加权让指标更敏感于高风险交互工况。

- **可作为安全优化闭环指标**  
  可直接把配置映射为“安全策略偏好”，形成训练-评估-迭代的闭环，而不仅是追求单一平均分。

---

## 5. 使用建议

- **对外汇报/学术对比**：优先用 `offline_nuscenes_eval`
- **内部安全优化**：补充跑 `safety_critical_eval`
- **最佳实践**：两套指标并行
  - 官方指标保证可比性
  - Safety 指标衡量风险敏感改进是否真实有效

---

## 6. 一句话结论

`offline_nuscenes_eval` 解决“**是否与行业标准一致**”，`safety_critical_eval` 解决“**是否更安全、更符合业务风险优先级**”。在安全导向项目中，后者是官方指标的重要补充，并且在近场与高风险类别上更有决策价值。

