# 面向安全关键场景的自动驾驶3D目标检测评估体系

> **论文框架草稿** · 基于 SafetyAI-SparseBEV 项目
> 
> 关键词：3D目标检测、安全关键评估、nuScenes、BEV感知、风险敏感指标、近场感知

---

## 摘要（Abstract）

现有自动驾驶感知评估体系（如 nuScenes Detection Score, NDS）以分类平均准确度为核心，对**近场高风险目标**的漏检与误差缺乏区分刻画，难以直接指导以安全为目标的模型优化。本文提出一套**安全关键口径（Safety-Critical）3D目标检测评估体系**，在 nuScenes 官方评测框架之上引入：（1）近场有效半径过滤；（2）弱势交通参与者类别加权 mAP；（3）尺度误差语义化重定义（w/l/h 相对误差）；（4）基于自车-目标相对朝向的误差分段加权机制。同时构建支撑多次实验纵向对比的 **MySQL 评估数据仓**，实现 GT 一次入库、多次 run 复用。以 SparseBEV 模型为基线，在 nuScenes val 集上对比官方指标与安全指标，揭示常规高 NDS 下潜在的近场检测弱点，为风险优先级导向的感知优化提供可复现的量化依据。

---

## 1 引言（Introduction）

### 1.1 研究背景

- 自动驾驶感知系统需在**毫秒级**内完成对周围环境的精确理解，任何漏检或几何误差在高速行驶中均可能酿成安全事故。
- nuScenes 数据集与 NDS 体系已成为业界标准 [Caesar et al., 2020]，但其 mAP 对所有类别等权、检测距离上限达 50 m，导致**近场低速行人/骑行者**对最终指标贡献有限，模型可在整体 NDS 优秀的同时在高危目标上存在显著缺陷。
- 现有安全性评估研究多聚焦于对抗样本鲁棒性或仿真场景构造，较少从**指标体系设计**角度系统解决"官方榜单分数与实际安全能力脱节"的问题。

### 1.2 研究动机

- **近场优先**：距自车 30 m 以内的目标与碰撞风险直接相关；50 m 外的远场目标在决策层通常不触发紧急制动。
- **类别差异**：行人、摩托车、自行车在碰撞事故中死亡率显著高于乘用车，其检测重要性不应与大型卡车等权。
- **误差语义**：官方 scale IoU 在细长目标（行人、骑行者）上对长度误差不敏感；朝向与速度误差在**迎向自车**的目标上危险程度远高于侧方或远离目标。

### 1.3 主要贡献

1. 提出**安全关键 nuScenes 检测评估配置**（Safety-Critical Detection Config），在官方评测骨架上最小化改动，保持匹配框架兼容性。
2. 设计并实现**评估数据仓**（MySQL Schema），支持 GT 资产化复用与多实验横向/纵向对比。
3. 构建配套**可视化对比工具链**，直观展示官方口径与安全口径过滤差异。
4. 以 SparseBEV 为基线进行系统性实验，量化两套指标的差异与关联，并揭示安全盲区。

---

## 2 相关工作（Related Work）

### 2.1 多目标3D检测基准与评估指标

- **nuScenes 评估体系** [Caesar et al., 2020]：NDS = (mAP × 5 + Σ TP-Score) / 10；TP 误差含 ATE、ASE、AOE、AVE、AAE。
- **Waymo Open Dataset** [Sun et al., 2020]：3D mAP + mAPH（朝向加权）。
- **KITTI** [Geiger et al., 2012]：2D / BEV / 3D AP 三档，易/中/难三级。
- 现有体系均为**任务精度导向**，非**风险导向**。

### 2.2 安全关键场景定义与评估

- 基于仿真的安全测试 [Suo et al., 2021; Rempe et al., 2022]：生成对抗场景，关注碰撞率。
- 关键切入场景（Cut-in）、行人横穿等场景化分析：聚焦场景拓扑，与感知指标解耦。
- **本文区别**：不构造对抗场景，而是在真实数据分布上**重新定义评估权重与范围**，关注检测层而非规划层。

### 2.3 BEV 感知模型

- **CenterPoint** [Yin et al., 2021]、**BEVDet** [Huang et al., 2022]、**BEVFormer** [Li et al., 2022]：基于 Transformer 的多摄像头 BEV 检测主流范式。
- **SparseBEV** [Liu et al., 2023]：稀疏查询 + 自适应特征采样，在 nuScenes 测试集上实现顶级 NDS，本文以其作为基线验证评估体系。

---

## 3 安全关键评估体系设计（Methodology）

### 3.1 官方评测框架回顾

设检测类别集合为 $\mathcal{C}$，距离阈值集合为 $\mathcal{D} = \{0.5, 1, 2, 4\}$ m，则：

$$\text{mAP}_{\text{official}} = \frac{1}{|\mathcal{C}|} \sum_{c \in \mathcal{C}} \frac{1}{|\mathcal{D}|} \sum_{d \in \mathcal{D}} \text{AP}(c, d)$$

TP 误差（ATE、ASE、AOE、AVE、AAE）在 $d_{\text{tp}} = 2$ m 阈值下匹配后计算，最终：

$$\text{NDS} = \frac{1}{10} \left( 5 \cdot \text{mAP} + \sum_{k} \max\left(0, 1 - \text{TP}_k\right) \right)$$

### 3.2 近场有效半径过滤

定义安全最大检测距离 $r_{\text{safety}}$（默认 30 m），对类别 $c$ 的实际评估半径为：

$$r_c^{\text{eff}} = \min\left(r_c^{\text{class}}, r_{\text{safety}}\right)$$

仅保留满足 $\text{ego\_dist}(b) \leq r_c^{\text{eff}}$ 的 GT box $b$ 及对应预测框参与评估。该操作将评估焦点收窄至**危险决策域**，排除远场噪声。

### 3.3 类别加权 mAP

引入类别重要性权重向量 $\mathbf{w} = \{w_c\}_{c \in \mathcal{C}}$（行人 $w=3.0$，摩托/自行车 $w=2.0$，其余 $w=1.0$），定义：

$$\text{mAP}_{\text{safety}} = \frac{\sum_{c \in \mathcal{C}} w_c \cdot \overline{\text{AP}}_c}{\sum_{c \in \mathcal{C}} w_c}$$

其中 $\overline{\text{AP}}_c = \frac{1}{|\mathcal{D}|} \sum_{d \in \mathcal{D}} \text{AP}(c, d)$。

### 3.4 语义化尺度误差

官方 ASE 使用 $1 - \text{scale\_IoU}$，在细长目标上对单维度误差不敏感。本文改为尺寸相对误差的加权均值：

$$\text{scale\_err}_{\text{safety}} = \frac{w_w \cdot \varepsilon_w + w_l \cdot \varepsilon_l + w_h \cdot \varepsilon_h}{w_w + w_l + w_h}$$

其中 $\varepsilon_{\dim} = \left|\frac{\hat{s}_{\dim} - s_{\dim}}{s_{\dim}}\right|$，默认权重 $w_w = w_l = 5,\ w_h = 3$（宽长对 BEV 占位判断更关键）。

### 3.5 基于相对朝向的误差分段加权

定义**自车-目标相对方位角** $\Delta\theta$（GT box yaw 与自车航向之差），按 $\Delta\theta_{\text{split}} = 45°$ 分两档：

- **迎向/正面区** $|\Delta\theta| \leq 45°$：高权重 $\alpha_{\text{front}} = 5$（朝向/速度误差危险程度高）
- **侧方/背向区** $|\Delta\theta| > 45°$：低权重 $\alpha_{\text{side}} = 2$

对 TP 集合 $\mathcal{T}$ 中每个匹配对 $(i)$ 的朝向误差 $e_i^{\text{orient}}$，定义加权累积均值：

$$\text{orient\_err}_{\text{safety}} = \frac{\sum_{i \in \mathcal{T}} \alpha(\Delta\theta_i) \cdot e_i^{\text{orient}}}{\sum_{i \in \mathcal{T}} \alpha(\Delta\theta_i)}$$

速度误差 $\text{vel\_err}_{\text{safety}}$ 采用相同加权机制。

### 3.6 安全定制 NDS

$$\text{NDS}_{\text{safety}} = \frac{5 \cdot \text{mAP}_{\text{safety}} + \sum_{k} \max(0, 1 - \text{TP}_k^{\text{safety}})}{10}$$

**注意**：$\text{NDS}_{\text{safety}}$ 与官方 NDS 不可横向比较，仅用于内部不同配置/模型间的相对比较。

### 3.7 安全评估配置汇总

| 参数 | 官方值 | Safety 值 |
|------|--------|-----------|
| `safety_max_dist` | 无 | 30 m |
| 类别权重（行人） | 1.0 | **3.0** |
| 类别权重（摩托/自行车） | 1.0 | **2.0** |
| `scale_err` 定义 | 1 − scale_IoU | w/l/h 相对误差加权 |
| 尺度权重 (w, l, h) | 不适用 | **(5, 5, 3)** |
| orient/vel 分段角度 | 不分段 | **45°** |
| 高风险方向权重 | 1.0 | **5.0** |
| 低风险方向权重 | 1.0 | **2.0** |

---

## 4 评估数据仓架构（Evaluation Data Warehouse）

### 4.1 设计原则

- **GT 资产化**：nuScenes GT 数据一次性导入，不随实验重复存储
- **Run 隔离**：每次推理实验作为独立 `experiment_run`，可携带配置、权重路径、git commit、延迟等元信息
- **双口径并行**：`metrics_official` 与 `metrics_safety_critical` 分表存储，支持并行对比
- **细粒度可溯源**：框级 `match_pair`、类级 `metrics_per_class`、距离分箱 `metrics_per_distance` 支持下钻分析

### 4.2 核心实体关系

```
nuscenes_dataset
    └── nuscenes_scene ──── ground_truth_sample
                                ├── ground_truth_box        ← ego_dist, visibility_level
                                ├── ground_truth_ego        ← yaw_rad, speed_mps
                                └── nuscenes_sample_camera  ← 6路相机文件名/标定

experiment_run
    └── prediction_sample
            └── prediction_box

experiment_run ─── metrics_official
             └─── metrics_safety_critical
             └─── metrics_per_class
             └─── metrics_per_distance
             └─── confusion_matrix_cell
             └─── match_pair              ← TP/FP/FN 框级证据
             └─── run_tag                 ← 自由标签 KV

视图: v_run_summary  ← 合并双口径 NDS/mAP/样本数
```

### 4.3 数据导入流水线

```
nuScenes JSON → import_gt_to_db.py       → GT 基础数据入库
             → import_camera_to_db.py    → 相机标定与帧文件入库
             → import_ego_to_db.py       → 自车位姿与速度入库
             → import_scene_visibility.py → GT box 可见度等级入库
             → fill_ego_dist_fast.py     → 批量回填 ego_dist 字段

val.py 推理结果 → import_run_to_db.py    → Run 信息 + 预测框 + 双口径 metrics 入库
                → fill_missing_db.py     → 混淆矩阵/距离分箱/延迟/标签 等补全
```

---

## 5 实验（Experiments）

### 5.1 实验配置

- **数据集**：nuScenes v1.0-trainval，val split（6019 样本）
- **基线模型**：SparseBEV（r50_nuimg_704×256 配置，CenterHead，时序帧数 N）
- **评估工具**：`tools/official_nuscenes_eval.py`（官方口径）& `tools/safety_critical_nuscenes_eval.py`（安全口径）
- **硬件**：XXX GPU，平均推理延迟 XX ms/frame

### 5.2 主要结果对比

#### 5.2.1 整体指标

| 指标 | 官方口径 | Safety 口径 | 差值 / 说明 |
|------|----------|-------------|-------------|
| mAP | — | — | Safety 提高行人/骑行者权重，通常 ↓ |
| NDS | — | — | 不可互比 |
| mATE (m) | — | — | 匹配逻辑相同，数值相近 |
| mASE | — | — | 定义不同，Safety ↑（尺寸误差更严格） |
| mAOE (rad) | — | — | Safety 对迎向目标加权，通常 ↑ |
| mAVE (m/s) | — | — | Safety 对迎向目标加权，通常 ↑ |

#### 5.2.2 类别级 AP 对比（Safety vs Official）

| 类别 | AP_official | AP_safety（单类）| 权重倍数 | 备注 |
|------|-------------|-----------------|----------|------|
| car | — | — | 1× | — |
| truck | — | — | 1× | — |
| bus | — | — | 1× | — |
| trailer | — | — | 1× | — |
| construction_vehicle | — | — | 1× | — |
| pedestrian | — | — | **3×** | 安全关键 |
| motorcycle | — | — | **2×** | 安全关键 |
| bicycle | — | — | **2×** | 安全关键 |
| traffic_cone | — | — | 1× | — |
| barrier | — | — | 1× | — |

#### 5.2.3 近场（0–30 m）vs 远场（30–50 m）性能分析

- 官方评测混入远场目标时 mAP 的变化幅度
- 近场 AP 各类别排名与官方排名的差异
- 近场 ATE/ASE/AOE 相较全场的变化趋势

### 5.3 可见度分级分析

基于 `visibility_level`（1–4，对应遮挡程度），分析：
- 低可见度（level 1–2）目标的 AP 与误差
- 安全口径对低可见度目标的检测弱点暴露程度

### 5.4 混淆矩阵分析

- 行人 ↔ 骑行者类间混淆率
- 安全口径与官方口径下混淆矩阵差异（`confusion_matrix_cell` 表）
- 主要误检/漏检场景的相机图像定性分析

### 5.5 朝向/速度误差的方向性分析

- TP 样本中，迎向（$|\Delta\theta| \leq 45°$）vs 侧方/背向的 AOE/AVE 分布
- 说明加权机制的合理性：迎向目标误差更大且对安全影响更显著

---

## 6 可视化案例分析（Qualitative Analysis）

### 6.1 官方过滤 vs Safety 过滤对比图

- 同一帧图像中，官方有效 GT（黑框）与 safety 有效 GT（彩色框，按距离着色）的差异
- 代表性场景：30 m 外行人在官方 mAP 中计入、在安全指标中不计的情况

### 6.2 安全盲区案例

- 模型在官方 NDS > X 时仍存在的**近场行人漏检**案例
- BEV 视角下安全距离圆（30 m）内漏检框的分布热力图

### 6.3 尺度误差语义化案例

- 行人目标 scale_IoU 较高但 w/l/h 相对误差较大的案例（e.g. 高度误差 40% 但 IoU 仍达 0.6）

---

## 7 讨论（Discussion）

### 7.1 两套指标的互补性

- 官方 NDS：用于横向对比、对外报告、与榜单对齐
- Safety NDS：用于内部优化迭代、风险优先级判断，**不替代**官方指标

### 7.2 参数敏感性

- `safety_max_dist` 对 mAP 的影响曲线（20 m / 30 m / 40 m）
- 类别权重的 Ablation：仅行人加权 vs 行人+骑行者加权 vs 全部加权

### 7.3 局限性

- 当前 Safety 体系仍基于**离线评估**，未考虑时序漏检（连续帧漏检的累积风险）
- 朝向分段基于 GT box yaw，未考虑预测框朝向误差对碰撞轨迹的影响
- 数据仓目前针对 nuScenes，扩展至其他数据集需重新定义 GT 导入接口

### 7.4 未来工作

- 引入**时序连续性**指标（MOTA-style tracking 级安全评估）
- 与规划层对接：将感知误差映射至 TTC（Time-to-Collision）降级幅度
- 扩展至 Waymo / Argoverse 2 数据集
- 在线评估集成：将 Safety 指标嵌入训练 loss 监控流水线

---

## 8 结论（Conclusion）

本文提出并实现了面向自动驾驶安全的 nuScenes 3D目标检测评估体系扩展框架，通过近场半径过滤、类别风险加权、语义化尺度误差及朝向分段加权四项机制，使评估指标能够量化感知系统在**安全决策关键域**的实际能力，弥补了官方 NDS 对近场高危目标不够敏感的缺陷。配套的 MySQL 评估数据仓与可视化工具链进一步提升了实验的可复现性与可解释性。以 SparseBEV 为基线的系统实验验证了安全口径与官方口径之间的差异，揭示了模型在常规指标良好时潜在的安全盲区，为以风险优先级为导向的感知模型迭代提供了可量化依据。

---

## 参考文献（References）

- [1] Caesar, H. et al. (2020). *nuScenes: A multimodal dataset for autonomous driving*. CVPR.
- [2] Liu, S. et al. (2023). *SparseBEV: High-Performance Sparse 3D Object Detection from Multi-Camera Videos*. ICCV.
- [3] Li, Z. et al. (2022). *BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers*. ECCV.
- [4] Huang, J. et al. (2022). *BEVDet: High-Performance Multi-Camera 3D Object Detection in Bird-Eye-View*. arXiv.
- [5] Yin, T. et al. (2021). *Center-based 3D Object Detection and Tracking*. CVPR.
- [6] Sun, P. et al. (2020). *Scalability in Perception for Autonomous Driving: Waymo Open Dataset*. CVPR.
- [7] Geiger, A. et al. (2012). *Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite*. CVPR.
- [8] Suo, S. et al. (2021). *TrafficSim: Learning to Simulate Realistic Multi-Agent Behaviors*. CVPR.
- [9] Rempe, D. et al. (2022). *Generating Useful Accident-Prone Driving Scenarios via a Learned Traffic Prior*. CVPR.

---

## 附录（Appendix）

### A. 安全评估配置文件（detection_safety_critical.json）说明

完整参数列表与取值范围，以及各参数对 mAP / TP 误差的影响方向。

### B. 数据库表结构速查

核心表字段定义、索引策略与典型查询 SQL 示例（按 run_id 汇总、按类别下钻、按距离分箱等）。

### C. 可视化工具使用说明

各脚本调用参数、依赖环境、输出图像格式说明。

### D. Ablation 补充表格

`safety_max_dist`、`class_ap_weights`、`orient_split_deg` 等参数 Ablation 完整数值表。
