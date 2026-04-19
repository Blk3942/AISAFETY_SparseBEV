# ERD generation prompts (from `sql_val_database_design.md`)

> **Note:** “ERD” means **Entity-Relationship Diagram**. The schema includes **analysis extensions** (per-class / per-distance metrics, `match_pair`, `run_tag`, and optional materialized view) per `docs/优化建议.md`.

**Core idea:** GT lives under `nuscenes_dataset` only. Each `experiment_run` references the corpus via `gt_dataset_id`. `prediction_sample.gt_sample_id` → `ground_truth_sample`. Extended run metadata (`backbone`, `input_resolution`, …) supports BI filters.

---

## 1. General (most diagram / ERD / Figma tools)

**Copy & paste:**

```text
Generate a clear ERD (Entity-Relationship Diagram). Chen or Crow’s Foot. Landscape, white background, print-friendly.

[Domain]
NuScenes 3D detection validation warehouse: full-dataset GT imported once per release; each eval job is an experiment_run with predictions and metrics. Include per-class and per-distance metric tables, box-level match_pair for confusion-matrix drill-down, optional run_tag, and optional mv_run_summary as a VIEW/MATERIALIZED VIEW (annotate as view, not a base table).

[Entities]
1. nuscenes_dataset (PK dataset_id): nuscenes_version, data_root_hint, imported_at, …

2. ground_truth_sample (PK id, FK dataset_id → nuscenes_dataset): sample_token, scene_token, timestamp_us. Unique (dataset_id, sample_token).

3. ground_truth_box (PK id, FK gt_sample_id → ground_truth_sample CASCADE): annotation_token, class_name, 3D pose/size/quaternion/velocity/attribute/num_pts/ego_dist.

4. experiment_run (PK run_id, FK gt_dataset_id → nuscenes_dataset): project_name, config_path, weights_path, dataset_root, eval_split, created_at, git_commit, notes, backbone, input_resolution, temporal_frames, tags JSONB, avg_latency_ms.

5. prediction_sample (PK id, FK run_id → experiment_run, FK gt_sample_id → ground_truth_sample): sample_token. Unique (run_id, sample_token).

6. prediction_box (PK id, FK pred_sample_id → prediction_sample CASCADE): class_name, score, 3D fields.

7. metrics_official (PK/FK run_id → experiment_run CASCADE): nd_score, mean_ap, m*m metrics, raw_json.

8. metrics_safety_critical (PK/FK run_id → experiment_run CASCADE): nd_score, mean_ap, errors, config_name, raw_json.

9. metrics_per_class (PK id, FK run_id → experiment_run CASCADE): eval_type official|safety_critical, class_name, ap, ate, ase, aoe, ave, aae. Unique (run_id, eval_type, class_name).

10. metrics_per_distance (PK id, FK run_id → experiment_run CASCADE): eval_type, class_name, dist_bin_lo, dist_bin_hi, ap, ate, num_gt, num_pred.

11. confusion_matrix_cell (PK id, FK run_id → experiment_run CASCADE): eval_type, dist_th_m, gt_class, pred_class, count.

12. match_pair (PK id, FK run_id → experiment_run CASCADE): eval_type, dist_th_m, gt_box_id NULLABLE FK→ground_truth_box, pred_box_id NULLABLE FK→prediction_box, match_dist, gt_class, pred_class, is_tp.

13. run_tag (PK run_id+tag_key, FK run_id → experiment_run CASCADE): tag_key, tag_val. Optional if using only experiment_run.tags JSONB—still draw if included.

[Cardinality]
- nuscenes_dataset 1:N ground_truth_sample; ground_truth_sample 1:N ground_truth_box
- nuscenes_dataset 1:N experiment_run
- experiment_run 1:N prediction_sample; ground_truth_sample 1:N prediction_sample (across runs)
- prediction_sample 1:N prediction_box
- experiment_run 1:0..1 metrics_official; 1:0..1 metrics_safety_critical
- experiment_run 1:N metrics_per_class; 1:N metrics_per_distance; 1:N confusion_matrix_cell; 1:N match_pair; 1:N run_tag

[Layout]
GT island left (nuscenes_dataset → ground_truth_sample → ground_truth_box). Center/right: experiment_run with branches to predictions, both metric aggregates, metrics_per_class, metrics_per_distance, confusion_matrix_cell, match_pair. Draw match_pair connecting to ground_truth_box and prediction_box with nullable FKs.

[Footnote]
Optional: dashed box “mv_run_summary” materialized view joining experiment_run + metrics_official + metrics_safety_critical + count(prediction_sample)—not a stored entity table.

Include all listed base tables (13). Mark PK/FK.
```

---

## 2. Mermaid `erDiagram`

**Copy & paste:**

```text
Output only one valid Mermaid erDiagram block (no prose).

Entities:
nuscenes_dataset, ground_truth_sample, ground_truth_box,
experiment_run (with backbone, input_resolution, temporal_frames, tags, avg_latency_ms),
prediction_sample, prediction_box,
metrics_official, metrics_safety_critical,
metrics_per_class (run_id, eval_type, class_name, ap…),
metrics_per_distance (run_id, eval_type, class_name, dist_bin_lo/hi…),
confusion_matrix_cell,
match_pair (nullable gt_box_id→ground_truth_box, nullable pred_box_id→prediction_box),
run_tag (composite PK run_id+tag_key).

Relationships:
nuscenes_dataset ||--o{ ground_truth_sample
ground_truth_sample ||--o{ ground_truth_box
nuscenes_dataset ||--o{ experiment_run
experiment_run ||--o{ prediction_sample
prediction_sample }o--|| ground_truth_sample
prediction_sample ||--o{ prediction_box
experiment_run ||--|| metrics_official
experiment_run ||--|| metrics_safety_critical
experiment_run ||--o{ metrics_per_class
experiment_run ||--o{ metrics_per_distance
experiment_run ||--o{ confusion_matrix_cell
experiment_run ||--o{ match_pair
match_pair }o--o| ground_truth_box
match_pair }o--o| prediction_box
experiment_run ||--o{ run_tag

English names; 5–8 attributes per entity where space allows.
```

---

## 3. dbdiagram.io / DBML

**Copy & paste:**

```text
DBML only. Include:

Table nuscenes_dataset { dataset_id bigint [pk, increment] }

Table ground_truth_sample {
  id bigint [pk, increment]
  dataset_id bigint [ref: > nuscenes_dataset.dataset_id]
  sample_token varchar
}

Table ground_truth_box {
  id bigint [pk, increment]
  gt_sample_id bigint [ref: > ground_truth_sample.id, delete: cascade]
  class_name varchar
}

Table experiment_run {
  run_id bigint [pk, increment]
  gt_dataset_id bigint [ref: > nuscenes_dataset.dataset_id]
  backbone varchar
  input_resolution varchar
  temporal_frames int
  tags json
  avg_latency_ms float
  eval_split varchar
}

Table prediction_sample {
  id bigint [pk, increment]
  run_id bigint [ref: > experiment_run.run_id]
  gt_sample_id bigint [ref: > ground_truth_sample.id]
  sample_token varchar
}

Table prediction_box {
  id bigint [pk, increment]
  pred_sample_id bigint [ref: > prediction_sample.id, delete: cascade]
  class_name varchar
  score float
}

Table metrics_official {
  run_id bigint [pk, ref: - experiment_run.run_id]
  nd_score float
  raw_json json
}

Table metrics_safety_critical {
  run_id bigint [pk, ref: - experiment_run.run_id]
  nd_score float
  raw_json json
}

Table metrics_per_class {
  id bigint [pk, increment]
  run_id bigint [ref: > experiment_run.run_id]
  eval_type varchar
  class_name varchar
}

Table metrics_per_distance {
  id bigint [pk, increment]
  run_id bigint [ref: > experiment_run.run_id]
  eval_type varchar
  class_name varchar
  dist_bin_lo float
  dist_bin_hi float
}

Table confusion_matrix_cell {
  id bigint [pk, increment]
  run_id bigint [ref: > experiment_run.run_id]
  eval_type varchar
  gt_class varchar
  pred_class varchar
}

Table match_pair {
  id bigint [pk, increment]
  run_id bigint [ref: > experiment_run.run_id]
  gt_box_id bigint [ref: > ground_truth_box.id]
  pred_box_id bigint [ref: > prediction_box.id]
  is_tp bool
}

Table run_tag {
  run_id bigint [ref: > experiment_run.run_id]
  tag_key varchar
  tag_val varchar
  indexes {
    (run_id, tag_key) [pk]
  }
}

GT tables must not reference experiment_run.
```

---

## 4. Short one-paragraph ERD prompt

**Copy & paste:**

```text
ERD for nuScenes detection validation: nuscenes_dataset feeds ground_truth_sample and ground_truth_box (no run_id). experiment_run references nuscenes_dataset via gt_dataset_id and includes backbone, input_resolution, temporal_frames, tags, avg_latency_ms. prediction_sample links run to ground_truth_sample via gt_sample_id; prediction_sample has many prediction_box. experiment_run has one-to-one metrics_official and metrics_safety_critical; one-to-many metrics_per_class, metrics_per_distance, confusion_matrix_cell, match_pair (nullable FKs to ground_truth_box and prediction_box), and optional run_tag. Optionally note materialized view mv_run_summary off diagram.
```

---

## 5. Negative / clarifying constraints

**Copy & paste:**

```text
Do not merge metrics_official with metrics_safety_critical or with metrics_per_class. Keep metrics_per_class and metrics_per_distance as separate tables. Do not attach run_id to ground_truth tables. Show match_pair with nullable gt_box_id and pred_box_id for FP/FN semantics. If draw mv_run_summary, label it as VIEW/MATERIALIZED VIEW, not as a physical table with FKs.
```
