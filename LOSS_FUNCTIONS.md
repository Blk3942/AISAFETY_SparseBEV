# 本项目损失函数说明（SparseBEV / SparseBEVHead）

本文基于本仓库源码梳理 **训练时的损失函数定义、匹配策略（Hungarian）与关键张量含义**，用于快速理解/修改训练配置。

核心入口：

- `models/sparsebev.py`：训练前向中调用 `self.pts_bbox_head.loss(...)`
- `models/sparsebev_head.py`：实现主要 loss（分类/回归 + 可选 DN 去噪损失）
- `models/bbox/assigners/hungarian_assigner_3d.py`：Hungarian 匹配（决定正负样本）
- `models/bbox/utils.py`：`normalize_bbox()`（回归目标的参数化方式）
- `configs/r50_nuimg_704x256.py`：loss 与 assigner 的权重配置

---

## 1. 总览：训练时优化目标由哪些项组成

在 `SparseBEVHead.loss()` 中，loss 以字典形式返回，主要由以下两类组成：

- **主分支（matching queries）**
  - `loss_cls`：分类损失（最后一层 decoder）
  - `loss_bbox`：3D 框回归损失（最后一层 decoder）
  - `d{k}.loss_cls` / `d{k}.loss_bbox`：中间 decoder 层的辅助损失（`k` 为 decoder 层编号，从 0 开始）
- **可选：Query Denoising（DN）分支**
  - `loss_cls_dn` / `loss_bbox_dn`：最后一层 decoder 的 DN 损失
  - `d{k}.loss_cls_dn` / `d{k}.loss_bbox_dn`：中间 decoder 层的 DN 辅助损失

> 配置里还提供了 `loss_iou`（GIoU），但在当前 `SparseBEVHead` 实现中没有参与计算；且默认权重为 0（见配置文件）。

---

## 2. 预测与 GT 的关键张量约定

### 2.1 预测输出

`SparseBEVHead.forward()` 输出（省略非关键字段）：

- `all_cls_scores`：形状通常为 `[num_dec_layers, B, Q, num_classes]`
- `all_bbox_preds`：形状通常为 `[num_dec_layers, B, Q, 10]`

其中每个 query 的 box 参数在本项目中按如下顺序组织（见 `SparseBEVHead.forward()` 里对 `bbox_preds` 的拼接）：

- `bbox_pred = [cx, cy, w, l, cz, h, sin(yaw), cos(yaw), vx, vy]`（共 10 维）

### 2.2 GT 组织方式

训练时 `SparseBEV.forward_train()` 会把 `gt_bboxes_3d` / `gt_labels_3d` 填入 `img_metas`，并调用：

- `SparseBEVHead.loss(gt_bboxes_list, gt_labels_list, preds_dicts, ...)`

其中 `gt_bboxes_list` 会被转换为：

- `gt = concat(gravity_center, box_tensor[:,3:])`
- 结果张量形状为 `[num_gt, 9]`（常见为 `[cx, cy, cz, w, l, h, yaw, vx, vy]`；具体由上游 3D box tensor 定义决定）

---

## 3. 匹配策略：HungarianAssigner3D（决定正负样本）

本项目采用 DETR 风格的 **一对一匹配**：每个 GT 至多匹配一个 query，每个 query 至多匹配一个 GT。匹配由 `HungarianAssigner3D.assign()` 完成。

### 3.1 匹配代价（cost）由哪些项组成

总 cost：

\[
\text{cost} = \text{cls\_cost} + \text{reg\_cost} + \text{iou\_cost}
\]

但在默认配置中 `iou_cost.weight = 0.0`，因此等价于：

\[
\text{cost} = \text{cls\_cost} + \text{reg\_cost}
\]

默认配置见 `configs/r50_nuimg_704x256.py`：

- `cls_cost`: `FocalLossCost(weight=2.0)`（实现来自 mmdet）
- `reg_cost`: `BBox3DL1Cost(weight=0.25)`（本仓库 `models/bbox/match_costs/match_cost.py`）
- `iou_cost`: `IoUCost(weight=0.0)`（来自 mmdet，默认不生效）

### 3.2 回归 cost 的参数化与权重（code_weights）

匹配时会先把 GT 做参数化（`normalize_bbox()`），并在 cost 计算前按维度加权：

- `bbox_pred ← bbox_pred * code_weights`
- `gt_norm ← normalize_bbox(gt) * code_weights`

默认 `code_weights`（见配置）：

- `[2, 2, 1, 1, 1, 1, 1, 1, 1, 1]`（对应 10 维回归向量）

> `HungarianAssigner3D` 里 `with_velo=True` 时会对全部维度做 L1 cost；否则只用前 8 维（不含速度）。

### 3.3 依赖说明

Hungarian 匹配使用 `scipy.optimize.linear_sum_assignment`，因此训练环境需要安装 `scipy`，否则会报错提示安装。

---

## 4. 回归目标的定义：`normalize_bbox()`

本项目回归的不是直接的 \([cx,cy,cz,w,l,h,yaw,vx,vy]\)，而是把部分维度做了变换（用于数值稳定/周期性）：

对输入 `bboxes[..., :]`：

- **尺寸对数**：`w, l, h` 使用 `log()` 变换
- **朝向**：用 `sin(yaw), cos(yaw)` 代替 `yaw`（消除角度周期不连续）

输出（含速度时）为：

\[
\text{norm}(b) = [cx, cy, \log w, \log l, cz, \log h, \sin(yaw), \cos(yaw), vx, vy]
\]

对应实现：`models/bbox/utils.py::normalize_bbox()`

---

## 5. 具体损失项定义

### 5.1 分类损失：`loss_cls`（FocalLoss，sigmoid）

在 `SparseBEVHead.loss_single()` 与 `SparseBEVHead.dn_loss_single()` 中，分类损失统一通过 `self.loss_cls(...)` 计算。

默认配置（`configs/r50_nuimg_704x256.py`）：

- `type='FocalLoss'`
- `use_sigmoid=True`
- `gamma=2.0`
- `alpha=0.25`
- `loss_weight=2.0`

标签构造方式（DETR 风格）：

- 未匹配（负样本）query 的 label 被置为 `num_classes`（背景类索引）
- 匹配到 GT 的 query label 为对应 `gt_labels`

归一化（`avg_factor`）：

- 主分支：`cls_avg_factor = num_pos + bg_cls_weight * num_neg`（再做分布式 reduce_mean）
- DN 分支：`avg_factor = num_total_pos`（再 reduce_mean）

### 5.2 回归损失：`loss_bbox`（L1Loss）

回归损失同样在 `loss_single()` 与 `dn_loss_single()` 中计算，形式为 **加权 L1**：

\[
\mathcal{L}_{bbox} = \frac{1}{Z}\sum_i \sum_d w_d \cdot \lvert \hat{b}_{i,d} - b^{*}_{i,d} \rvert
\]

- `\hat{b}`：预测框（decoder 输出的 10 维向量）
- `b*`：`normalize_bbox()` 后的回归目标
- `w_d`：`code_weights`（逐维权重）
- `Z`：`avg_factor = num_total_pos`（分布式平均后 clamp 到至少 1）

默认配置：

- `loss_bbox = L1Loss(loss_weight=0.25)`

数值稳定：

- 对 `normalized_bbox_targets` 做 `isfinite` 过滤（避免 `nan/inf` 参与计算）
- 最终 loss 用 `torch.nan_to_num` 做兜底

### 5.3 DN（Query Denoising）分支损失

当满足以下条件时启用 DN loss：

- `SparseBEVHead.training == True`
- 配置 `query_denoising=True`

DN 机制要点（见 `SparseBEVHead.prepare_for_dn_input()`）：

- 从每张图的 GT 构造“已知目标”query，复制为 `query_denoising_groups` 组
- 对 box center 与 label 注入噪声（`dn_bbox_noise_scale` / `dn_label_noise_scale`）
- 在 transformer 输入侧通过 `attn_mask` 隔离 DN 与正常匹配 queries 的注意力

DN loss 形式与主分支一致：

- `loss_cls_dn`：FocalLoss
- `loss_bbox_dn`：L1Loss（对 `normalize_bbox(known_bboxs)`）

默认 DN 权重：

- `dn_weight = 1.0`（在 `dn_loss_single()` 中对 loss 乘这个系数）

---

## 6. 默认配置速查（r50_nuimg_704x256）

与损失/匹配最相关的配置片段位于 `configs/r50_nuimg_704x256.py`：

- `pts_bbox_head.loss_cls`: FocalLoss（`loss_weight=2.0`）
- `pts_bbox_head.loss_bbox`: L1Loss（`loss_weight=0.25`）
- `pts_bbox_head.code_weights`: `[2,2,1,1,1,1,1,1,1,1]`
- `train_cfg.pts.assigner`:
  - `cls_cost.weight=2.0`
  - `reg_cost.weight=0.25`
  - `iou_cost.weight=0.0`

---

## 7. 修改建议（常见改法）

- **想加强定位（cx,cy）约束**：提高 `code_weights[0:2]` 或提高 `loss_bbox.loss_weight`
- **想降低速度回归影响**：降低 `code_weights[8:10]`（若速度维参与回归/匹配）
- **想让匹配更依赖回归**：提高 `train_cfg.assigner.reg_cost.weight`
- **不想用 DN**：设置 `query_denoising=False`（并可移除相关中间 loss 记录）

