# 基于 SparseBEV / nuScenes `val` 结果的 SQL 数据库搭建思路

本文档含**基础仓**与**分析增强**（来源：`docs/优化建议.md`）：逐类/逐距离段指标、实验标签、`match_pair` 证据链、物化视图等。

## 1. 目标与数据来源

| 层次 | 来源 | 说明 |
|------|------|------|
| 真值（**与单次推理解耦**） | nuScenes **Full Dataset** 某一发行版下的 **全集**标注（如 `v1.0-trainval` / `v1.0-mini` / `v1.0-test`） | 按发行版入库一次；**不挂 `run_id`** |
| 推理结果 | `submission/pts_bbox/results_nusc.json`（或等价路径） | 每条推理运行独立存储 |
| 官方指标 | `nuscenes_eval_out/metrics_summary.json` 或 `val.py` 终端输出 | NDS、mAP、mATE… |
| Safety 指标 | `safety_critical_eval_out/metrics_summary.json` | 加权 mAP、`nd_score`、各类 TP 误差 |
| 混淆矩阵 | **评测脚本二次汇总**（见 §6） | 匹配后对 **GT 类 × 预测类** 计数 |

**设计要点**

- **GT 库**：面向 nuScenes 某一 `version` 对应的数据根目录做 **一次性全量导入**（该版本下可达的全部 `sample` / `sample_annotation`），供所有历史与未来推理复用。  
- **推理**：每次 `val`/推理对应一条 `experiment_run`，仅写入预测、聚合指标与混淆矩阵；通过 **数据集外键 + 样本级外键** 绑定到同一套 GT。  
- **兼容性**：mini 训推可在库中单独登记 `v1.0-mini` 数据集行；全量 trainval 使用 `v1.0-trainval`；二者互不覆盖。

---

## 2. 实体关系（概念）

```
nuscenes_dataset
      │
      ├──< ground_truth_sample ──< ground_truth_box
      │
      └──< experiment_run ──┬── metrics_official (1:1)
                            ├── metrics_safety_critical (1:1)
                            ├── metrics_per_class (1:N)          ← 逐类指标
                            ├── metrics_per_distance (1:N)       ← 逐距离段
                            ├── run_tag (1:N)                    ← 可选标签
                            ├── confusion_matrix_cell (1:N)
                            ├── match_pair (1:N)                 ← 框级匹配证据
                            └── prediction_sample (1:N) ──> ground_truth_sample (N:1)
                                      └── prediction_box (1:N)
```

- **GT 侧**：仅依赖 `nuscenes_dataset`，**不**依赖 `experiment_run`。  
- **推理侧**：`prediction_sample` 同时持有 `run_id` 与 `gt_sample_id`。

---

## 3. 核心表结构建议

### 3.1 nuScenes 数据集发行版（GT 全集作用域）

```sql
CREATE TABLE nuscenes_dataset (
    dataset_id        BIGSERIAL PRIMARY KEY,
    nuscenes_version  VARCHAR(32) NOT NULL,
    data_root_hint    TEXT,
    description       TEXT,
    imported_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (nuscenes_version)
);
```

若同一 `version` 存在多条（不同镜像路径），可改为 `UNIQUE (nuscenes_version, data_root_hint)` 或增加 `corpus_label`。

---

### 3.2 真值（样本级 + 框级）— **与推理运行独立**

```sql
CREATE TABLE ground_truth_sample (
    id               BIGSERIAL PRIMARY KEY,
    dataset_id       BIGINT NOT NULL REFERENCES nuscenes_dataset(dataset_id) ON DELETE RESTRICT,
    sample_token     VARCHAR(64) NOT NULL,
    scene_token      VARCHAR(64),
    timestamp_us     BIGINT,
    UNIQUE (dataset_id, sample_token)
);
CREATE INDEX idx_gt_sample_scene ON ground_truth_sample (dataset_id, scene_token);
```

```sql
CREATE TABLE ground_truth_box (
    id                  BIGSERIAL PRIMARY KEY,
    gt_sample_id        BIGINT NOT NULL REFERENCES ground_truth_sample(id) ON DELETE CASCADE,
    annotation_token    VARCHAR(64),
    class_name          VARCHAR(64) NOT NULL,
    translation_x       DOUBLE PRECISION NOT NULL,
    translation_y       DOUBLE PRECISION NOT NULL,
    translation_z       DOUBLE PRECISION NOT NULL,
    size_wlh_0          DOUBLE PRECISION NOT NULL,
    size_wlh_1          DOUBLE PRECISION NOT NULL,
    size_wlh_2          DOUBLE PRECISION NOT NULL,
    rotation_w          DOUBLE PRECISION NOT NULL,
    rotation_x          DOUBLE PRECISION NOT NULL,
    rotation_y          DOUBLE PRECISION NOT NULL,
    rotation_z          DOUBLE PRECISION NOT NULL,
    velocity_x          DOUBLE PRECISION,
    velocity_y          DOUBLE PRECISION,
    attribute_name      VARCHAR(128),
    num_pts             INTEGER,
    ego_dist            DOUBLE PRECISION
);
CREATE INDEX idx_gt_box_class ON ground_truth_box (class_name);
```

导入：对选定 `nuscenes_dataset` 全量写入；**不按 run 切片**。

---

### 3.3 实验 / 运行（单次推理）— **含结构化实验维度**

在 `notes` 之外增加常用结构化字段（**方案 A**，便于 BI 筛选）；可选 **方案 B** `run_tag` 见 §3.11。

```sql
CREATE TABLE experiment_run (
    run_id             BIGSERIAL PRIMARY KEY,
    gt_dataset_id      BIGINT NOT NULL REFERENCES nuscenes_dataset(dataset_id) ON DELETE RESTRICT,
    project_name       VARCHAR(128) NOT NULL DEFAULT 'AISAFETY_SparseBEV',
    config_path        TEXT,
    weights_path       TEXT,
    dataset_root       TEXT NOT NULL,
    eval_split         VARCHAR(32) NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    git_commit         VARCHAR(64),
    notes              TEXT,
    backbone           VARCHAR(64),
    input_resolution   VARCHAR(32),
    temporal_frames    INTEGER,
    tags               JSONB DEFAULT '[]'::jsonb,
    avg_latency_ms     DOUBLE PRECISION
);
```

| 字段 | 说明 |
|------|------|
| `backbone` | 如 `ResNet-50`、`VoVNet-99` |
| `input_resolution` | 如 `256x704` |
| `temporal_frames` | 如 SparseBEV `num_frames`（8） |
| `tags` | 灵活标签 JSON 数组，与 `run_tag` 二选一或并存 |
| `avg_latency_ms` | 可选；用于 §8 物化视图与性能看板 |

---

### 3.4 推理结果 — **显式指向 GT 样本**

```sql
CREATE TABLE prediction_sample (
    id               BIGSERIAL PRIMARY KEY,
    run_id           BIGINT NOT NULL REFERENCES experiment_run(run_id) ON DELETE CASCADE,
    gt_sample_id     BIGINT NOT NULL REFERENCES ground_truth_sample(id) ON DELETE RESTRICT,
    sample_token     VARCHAR(64) NOT NULL,
    UNIQUE (run_id, sample_token)
);
CREATE INDEX idx_pred_sample_gt ON prediction_sample (gt_sample_id);
CREATE INDEX idx_pred_sample_run ON prediction_sample (run_id);
```

```sql
CREATE TABLE prediction_box (
    id               BIGSERIAL PRIMARY KEY,
    pred_sample_id   BIGINT NOT NULL REFERENCES prediction_sample(id) ON DELETE CASCADE,
    class_name       VARCHAR(64) NOT NULL,
    score            DOUBLE PRECISION NOT NULL,
    translation_x    DOUBLE PRECISION NOT NULL,
    translation_y    DOUBLE PRECISION NOT NULL,
    translation_z    DOUBLE PRECISION NOT NULL,
    size_wlh_0       DOUBLE PRECISION NOT NULL,
    size_wlh_1       DOUBLE PRECISION NOT NULL,
    size_wlh_2       DOUBLE PRECISION NOT NULL,
    rotation_w       DOUBLE PRECISION NOT NULL,
    rotation_x       DOUBLE PRECISION NOT NULL,
    rotation_y       DOUBLE PRECISION NOT NULL,
    rotation_z       DOUBLE PRECISION NOT NULL,
    velocity_x       DOUBLE PRECISION,
    velocity_y       DOUBLE PRECISION,
    attribute_name   VARCHAR(128)
);
CREATE INDEX idx_pred_box_class ON prediction_box (class_name);
```

---

### 3.5 官方 / Safety 聚合指标

```sql
CREATE TABLE metrics_official (
    run_id      BIGINT PRIMARY KEY REFERENCES experiment_run(run_id) ON DELETE CASCADE,
    nd_score    DOUBLE PRECISION NOT NULL,
    mean_ap     DOUBLE PRECISION NOT NULL,
    mATE        DOUBLE PRECISION,
    mASE        DOUBLE PRECISION,
    mAOE        DOUBLE PRECISION,
    mAVE        DOUBLE PRECISION,
    mAAE        DOUBLE PRECISION,
    eval_time_s DOUBLE PRECISION,
    raw_json    JSONB
);
```

```sql
CREATE TABLE metrics_safety_critical (
    run_id       BIGINT PRIMARY KEY REFERENCES experiment_run(run_id) ON DELETE CASCADE,
    nd_score     DOUBLE PRECISION NOT NULL,
    mean_ap      DOUBLE PRECISION NOT NULL,
    mATE         DOUBLE PRECISION,
    mASE         DOUBLE PRECISION,
    mAOE         DOUBLE PRECISION,
    mAVE         DOUBLE PRECISION,
    mAAE         DOUBLE PRECISION,
    config_name  VARCHAR(128) NOT NULL DEFAULT 'detection_safety_critical',
    raw_json     JSONB
);
```

---

### 3.6 逐类别指标明细表（`metrics_per_class`）

将 per-class AP/误差从 `raw_json` **拆表**，便于可视化与 `GROUP BY class_name` 跨 run 对比。

```sql
CREATE TABLE metrics_per_class (
    id          BIGSERIAL PRIMARY KEY,
    run_id      BIGINT NOT NULL REFERENCES experiment_run(run_id) ON DELETE CASCADE,
    eval_type   VARCHAR(32) NOT NULL,
    class_name  VARCHAR(64) NOT NULL,
    ap          DOUBLE PRECISION,
    ate         DOUBLE PRECISION,
    ase         DOUBLE PRECISION,
    aoe         DOUBLE PRECISION,
    ave         DOUBLE PRECISION,
    aae         DOUBLE PRECISION,
    UNIQUE (run_id, eval_type, class_name)
);
CREATE INDEX idx_mpc_run_class ON metrics_per_class (run_id, class_name);
```

`eval_type`：`official` | `safety_critical`。数据可从 `metrics_details.json` / devkit 序列化结果解析填入。

---

### 3.7 逐距离段指标明细表（`metrics_per_distance`）

支撑 **AP–距离衰减**、Safety 场景近距配额分析。

```sql
CREATE TABLE metrics_per_distance (
    id            BIGSERIAL PRIMARY KEY,
    run_id        BIGINT NOT NULL REFERENCES experiment_run(run_id) ON DELETE CASCADE,
    eval_type     VARCHAR(32) NOT NULL,
    class_name    VARCHAR(64) NOT NULL,
    dist_bin_lo   DOUBLE PRECISION NOT NULL,
    dist_bin_hi   DOUBLE PRECISION NOT NULL,
    ap            DOUBLE PRECISION,
    ate           DOUBLE PRECISION,
    num_gt        INTEGER,
    num_pred      INTEGER,
    UNIQUE (run_id, eval_type, class_name, dist_bin_lo)
);
CREATE INDEX idx_mpd_run ON metrics_per_distance (run_id, eval_type);
```

若上游评测未直接输出距离分箱，可在应用层按 `ego_dist` 对 GT/预测聚类统计后写入（需与业务定义一致）。

---

### 3.8 混淆矩阵（聚合）

```sql
CREATE TABLE confusion_matrix_cell (
    id           BIGSERIAL PRIMARY KEY,
    run_id       BIGINT NOT NULL REFERENCES experiment_run(run_id) ON DELETE CASCADE,
    eval_type    VARCHAR(32) NOT NULL,
    dist_th_m    DOUBLE PRECISION,
    gt_class     VARCHAR(64) NOT NULL,
    pred_class   VARCHAR(64) NOT NULL,
    count        INTEGER NOT NULL,
    UNIQUE (run_id, eval_type, dist_th_m, gt_class, pred_class)
);
CREATE INDEX idx_cm_run ON confusion_matrix_cell (run_id, eval_type);
```

---

### 3.9 框级匹配明细表（`match_pair`）

混淆矩阵与 BEV 可视化的**证据链**：可下钻到每个 FP/FN/TP 框。

```sql
CREATE TABLE match_pair (
    id            BIGSERIAL PRIMARY KEY,
    run_id        BIGINT NOT NULL REFERENCES experiment_run(run_id) ON DELETE CASCADE,
    eval_type     VARCHAR(32) NOT NULL,
    dist_th_m     DOUBLE PRECISION,
    gt_box_id     BIGINT REFERENCES ground_truth_box(id) ON DELETE SET NULL,
    pred_box_id   BIGINT REFERENCES prediction_box(id) ON DELETE SET NULL,
    match_dist    DOUBLE PRECISION,
    gt_class      VARCHAR(64),
    pred_class    VARCHAR(64),
    is_tp         BOOLEAN NOT NULL
);
CREATE INDEX idx_match_run ON match_pair (run_id, eval_type);
CREATE INDEX idx_match_gt ON match_pair (gt_box_id);
CREATE INDEX idx_match_pred ON match_pair (pred_box_id);
```

语义：`gt_box_id` 空且 `pred_box_id` 非空 ≈ FP；反之 ≈ FN；二者非空且 `is_tp` ≈ TP 匹配对（具体判定以入库脚本为准）。

---

### 3.10 可选：`run_tag`（多对多结构化标签）

与 `experiment_run.tags` JSONB **可并存**；适合键值维度较多、需单独索引的场景。

```sql
CREATE TABLE run_tag (
    run_id   BIGINT NOT NULL REFERENCES experiment_run(run_id) ON DELETE CASCADE,
    tag_key  VARCHAR(64) NOT NULL,
    tag_val  VARCHAR(128) NOT NULL,
    PRIMARY KEY (run_id, tag_key)
);
```

---

## 4. 物化视图（跨 run 看板加速）

高频 **run 汇总** 可预聚合；以 PostgreSQL 为例（其他库用普通 VIEW 或定时任务表亦可）。

```sql
CREATE MATERIALIZED VIEW mv_run_summary AS
SELECT
    r.run_id,
    r.project_name,
    r.config_path,
    r.eval_split,
    r.created_at,
    r.backbone,
    r.input_resolution,
    r.temporal_frames,
    r.tags,
    r.avg_latency_ms,
    mo.nd_score   AS official_nds,
    mo.mean_ap    AS official_map,
    ms.nd_score   AS safety_nds,
    ms.mean_ap    AS safety_map,
    (SELECT COUNT(*) FROM prediction_sample ps WHERE ps.run_id = r.run_id) AS num_samples
FROM experiment_run r
LEFT JOIN metrics_official mo ON mo.run_id = r.run_id
LEFT JOIN metrics_safety_critical ms ON ms.run_id = r.run_id;
```

刷新策略：`REFRESH MATERIALIZED VIEW mv_run_summary`（每次批量导入 run 后执行）。

---

## 5. 与项目产物的映射

| 产物 | 映射表 |
|------|--------|
| nuScenes 全量标注 | `nuscenes_dataset` + `ground_truth_sample` + `ground_truth_box` |
| `results_nusc.json` | `prediction_sample` + `prediction_box` |
| official `metrics_summary.json` | `metrics_official`；per-class 解析入 `metrics_per_class` |
| safety `metrics_summary.json` | `metrics_safety_critical`；per-class 入 `metrics_per_class` |
| `metrics_details.json`（若保留） | 辅助填充 `metrics_per_class` / 距离分箱逻辑 |
| 匹配脚本 | `confusion_matrix_cell` + `match_pair` |
| 距离衰减自定义统计 | `metrics_per_distance` |

---

## 6. 实施顺序建议

1. 登记 `nuscenes_dataset` 并 **全量导入 GT**。  
2. 新建 `experiment_run`（含 backbone / 分辨率 / 帧数 / tags / latency 等）。  
3. 导入 `results_nusc.json` → `prediction_sample` / `prediction_box`。  
4. 写入 `metrics_official`、`metrics_safety_critical`。  
5. 解析详单写入 **`metrics_per_class`**；若有距离分箱则写入 **`metrics_per_distance`**。  
6. 跑匹配：写入 **`match_pair`**，再汇总 **`confusion_matrix_cell`**。  
7. **`REFRESH MATERIALIZED VIEW mv_run_summary`**（若使用）。

---

## 7. 混淆矩阵与 GT 的关系

固定 `run_id`，经 `prediction_sample.gt_sample_id` 对齐 GT；匹配粒度以 **`match_pair`** 为准再聚合到 **`confusion_matrix_cell`**。

---

## 8. 可选扩展（其余）

- **Split 注册表** `eval_split_sample(dataset_id, split_name, sample_token)`：校验预测样本是否属于声明的 `eval_split`。  
- **时间序列**：多次推理只增 `experiment_run` 及从属表；GT 不重复导入。
