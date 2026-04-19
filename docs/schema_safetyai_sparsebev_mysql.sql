-- safetyai_sparsebev schema (MySQL 8.x)
-- Derived from docs/sql_val_database_design.md — PostgreSQL types mapped to MySQL.

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

DROP DATABASE IF EXISTS safetyai_sparsebev;
CREATE DATABASE safetyai_sparsebev
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE safetyai_sparsebev;

-- ---------------------------------------------------------------------------
-- 3.1 nuscenes_dataset
-- ---------------------------------------------------------------------------
CREATE TABLE nuscenes_dataset (
    dataset_id        BIGINT AUTO_INCREMENT PRIMARY KEY,
    nuscenes_version  VARCHAR(32) NOT NULL,
    data_root_hint    TEXT,
    description       TEXT,
    imported_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_nuscenes_version (nuscenes_version)
) ENGINE=InnoDB;

-- ---------------------------------------------------------------------------
-- 3.2 ground_truth_sample / ground_truth_box
-- ---------------------------------------------------------------------------
CREATE TABLE ground_truth_sample (
    id               BIGINT AUTO_INCREMENT PRIMARY KEY,
    dataset_id       BIGINT NOT NULL,
    sample_token     VARCHAR(64) NOT NULL,
    scene_token      VARCHAR(64),
    timestamp_us     BIGINT,
    UNIQUE KEY uk_dataset_sample (dataset_id, sample_token),
    CONSTRAINT fk_gt_sample_dataset
      FOREIGN KEY (dataset_id) REFERENCES nuscenes_dataset(dataset_id)
      ON DELETE RESTRICT
) ENGINE=InnoDB;

CREATE INDEX idx_gt_sample_scene ON ground_truth_sample (dataset_id, scene_token);

CREATE TABLE ground_truth_box (
    id                  BIGINT AUTO_INCREMENT PRIMARY KEY,
    gt_sample_id        BIGINT NOT NULL,
    annotation_token    VARCHAR(64),
    class_name          VARCHAR(64) NOT NULL,
    translation_x       DOUBLE NOT NULL,
    translation_y       DOUBLE NOT NULL,
    translation_z       DOUBLE NOT NULL,
    size_wlh_0          DOUBLE NOT NULL,
    size_wlh_1          DOUBLE NOT NULL,
    size_wlh_2          DOUBLE NOT NULL,
    rotation_w          DOUBLE NOT NULL,
    rotation_x          DOUBLE NOT NULL,
    rotation_y          DOUBLE NOT NULL,
    rotation_z          DOUBLE NOT NULL,
    velocity_x          DOUBLE,
    velocity_y          DOUBLE,
    attribute_name      VARCHAR(128),
    num_pts             INT,
    ego_dist            DOUBLE,
    CONSTRAINT fk_gt_box_sample
      FOREIGN KEY (gt_sample_id) REFERENCES ground_truth_sample(id)
      ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE INDEX idx_gt_box_class ON ground_truth_box (class_name);

-- ---------------------------------------------------------------------------
-- 3.3 experiment_run
-- ---------------------------------------------------------------------------
CREATE TABLE experiment_run (
    run_id             BIGINT AUTO_INCREMENT PRIMARY KEY,
    gt_dataset_id      BIGINT NOT NULL,
    project_name       VARCHAR(128) NOT NULL DEFAULT 'AISAFETY_SparseBEV',
    config_path        TEXT,
    weights_path       TEXT,
    dataset_root       TEXT NOT NULL,
    eval_split         VARCHAR(32) NOT NULL,
    created_at         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    git_commit         VARCHAR(64),
    notes              TEXT,
    backbone           VARCHAR(64),
    input_resolution   VARCHAR(32),
    temporal_frames    INT,
    tags               JSON,
    avg_latency_ms     DOUBLE,
    CONSTRAINT fk_run_dataset
      FOREIGN KEY (gt_dataset_id) REFERENCES nuscenes_dataset(dataset_id)
      ON DELETE RESTRICT
) ENGINE=InnoDB;

-- ---------------------------------------------------------------------------
-- 3.4 prediction_sample / prediction_box
-- ---------------------------------------------------------------------------
CREATE TABLE prediction_sample (
    id               BIGINT AUTO_INCREMENT PRIMARY KEY,
    run_id           BIGINT NOT NULL,
    gt_sample_id     BIGINT NOT NULL,
    sample_token     VARCHAR(64) NOT NULL,
    UNIQUE KEY uk_run_sample (run_id, sample_token),
    CONSTRAINT fk_pred_sample_run
      FOREIGN KEY (run_id) REFERENCES experiment_run(run_id)
      ON DELETE CASCADE,
    CONSTRAINT fk_pred_sample_gt
      FOREIGN KEY (gt_sample_id) REFERENCES ground_truth_sample(id)
      ON DELETE RESTRICT
) ENGINE=InnoDB;

CREATE INDEX idx_pred_sample_gt ON prediction_sample (gt_sample_id);
CREATE INDEX idx_pred_sample_run ON prediction_sample (run_id);

CREATE TABLE prediction_box (
    id               BIGINT AUTO_INCREMENT PRIMARY KEY,
    pred_sample_id   BIGINT NOT NULL,
    class_name       VARCHAR(64) NOT NULL,
    score            DOUBLE NOT NULL,
    translation_x    DOUBLE NOT NULL,
    translation_y    DOUBLE NOT NULL,
    translation_z    DOUBLE NOT NULL,
    size_wlh_0       DOUBLE NOT NULL,
    size_wlh_1       DOUBLE NOT NULL,
    size_wlh_2       DOUBLE NOT NULL,
    rotation_w       DOUBLE NOT NULL,
    rotation_x       DOUBLE NOT NULL,
    rotation_y       DOUBLE NOT NULL,
    rotation_z       DOUBLE NOT NULL,
    velocity_x       DOUBLE,
    velocity_y       DOUBLE,
    attribute_name   VARCHAR(128),
    CONSTRAINT fk_pred_box_sample
      FOREIGN KEY (pred_sample_id) REFERENCES prediction_sample(id)
      ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE INDEX idx_pred_box_class ON prediction_box (class_name);

-- ---------------------------------------------------------------------------
-- 3.5 metrics_official / metrics_safety_critical
-- ---------------------------------------------------------------------------
CREATE TABLE metrics_official (
    run_id      BIGINT NOT NULL PRIMARY KEY,
    nd_score    DOUBLE NOT NULL,
    mean_ap     DOUBLE NOT NULL,
    mATE        DOUBLE,
    mASE        DOUBLE,
    mAOE        DOUBLE,
    mAVE        DOUBLE,
    mAAE        DOUBLE,
    eval_time_s DOUBLE,
    raw_json    JSON,
    CONSTRAINT fk_mo_run
      FOREIGN KEY (run_id) REFERENCES experiment_run(run_id)
      ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE TABLE metrics_safety_critical (
    run_id       BIGINT NOT NULL PRIMARY KEY,
    nd_score     DOUBLE NOT NULL,
    mean_ap      DOUBLE NOT NULL,
    mATE         DOUBLE,
    mASE         DOUBLE,
    mAOE         DOUBLE,
    mAVE         DOUBLE,
    mAAE         DOUBLE,
    config_name  VARCHAR(128) NOT NULL DEFAULT 'detection_safety_critical',
    raw_json     JSON,
    CONSTRAINT fk_msc_run
      FOREIGN KEY (run_id) REFERENCES experiment_run(run_id)
      ON DELETE CASCADE
) ENGINE=InnoDB;

-- ---------------------------------------------------------------------------
-- 3.6 / 3.7 / 3.8
-- ---------------------------------------------------------------------------
CREATE TABLE metrics_per_class (
    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    run_id      BIGINT NOT NULL,
    eval_type   VARCHAR(32) NOT NULL,
    class_name  VARCHAR(64) NOT NULL,
    ap          DOUBLE,
    ate         DOUBLE,
    ase         DOUBLE,
    aoe         DOUBLE,
    ave         DOUBLE,
    aae         DOUBLE,
    UNIQUE KEY uk_mpc (run_id, eval_type, class_name),
    CONSTRAINT fk_mpc_run
      FOREIGN KEY (run_id) REFERENCES experiment_run(run_id)
      ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE INDEX idx_mpc_run_class ON metrics_per_class (run_id, class_name);

CREATE TABLE metrics_per_distance (
    id            BIGINT AUTO_INCREMENT PRIMARY KEY,
    run_id        BIGINT NOT NULL,
    eval_type     VARCHAR(32) NOT NULL,
    class_name    VARCHAR(64) NOT NULL,
    dist_bin_lo   DOUBLE NOT NULL,
    dist_bin_hi   DOUBLE NOT NULL,
    ap            DOUBLE,
    ate           DOUBLE,
    num_gt        INT,
    num_pred      INT,
    UNIQUE KEY uk_mpd (run_id, eval_type, class_name, dist_bin_lo),
    CONSTRAINT fk_mpd_run
      FOREIGN KEY (run_id) REFERENCES experiment_run(run_id)
      ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE INDEX idx_mpd_run ON metrics_per_distance (run_id, eval_type);

CREATE TABLE confusion_matrix_cell (
    id           BIGINT AUTO_INCREMENT PRIMARY KEY,
    run_id       BIGINT NOT NULL,
    eval_type    VARCHAR(32) NOT NULL,
    dist_th_m    DOUBLE,
    gt_class     VARCHAR(64) NOT NULL,
    pred_class   VARCHAR(64) NOT NULL,
    count_val    INT NOT NULL,
    UNIQUE KEY uk_cm (run_id, eval_type, dist_th_m, gt_class, pred_class),
    CONSTRAINT fk_cm_run
      FOREIGN KEY (run_id) REFERENCES experiment_run(run_id)
      ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE INDEX idx_cm_run ON confusion_matrix_cell (run_id, eval_type);

-- ---------------------------------------------------------------------------
-- 3.9 match_pair (column `count` renamed to avoid MySQL reserved word issues in some tools)
-- ---------------------------------------------------------------------------
CREATE TABLE match_pair (
    id            BIGINT AUTO_INCREMENT PRIMARY KEY,
    run_id        BIGINT NOT NULL,
    eval_type     VARCHAR(32) NOT NULL,
    dist_th_m     DOUBLE,
    gt_box_id     BIGINT,
    pred_box_id   BIGINT,
    match_dist    DOUBLE,
    gt_class      VARCHAR(64),
    pred_class    VARCHAR(64),
    is_tp         TINYINT(1) NOT NULL,
    CONSTRAINT fk_mp_run
      FOREIGN KEY (run_id) REFERENCES experiment_run(run_id)
      ON DELETE CASCADE,
    CONSTRAINT fk_mp_gt_box
      FOREIGN KEY (gt_box_id) REFERENCES ground_truth_box(id)
      ON DELETE SET NULL,
    CONSTRAINT fk_mp_pred_box
      FOREIGN KEY (pred_box_id) REFERENCES prediction_box(id)
      ON DELETE SET NULL
) ENGINE=InnoDB;

CREATE INDEX idx_match_run ON match_pair (run_id, eval_type);
CREATE INDEX idx_match_gt ON match_pair (gt_box_id);
CREATE INDEX idx_match_pred ON match_pair (pred_box_id);

-- ---------------------------------------------------------------------------
-- 3.10 run_tag
-- ---------------------------------------------------------------------------
CREATE TABLE run_tag (
    run_id   BIGINT NOT NULL,
    tag_key  VARCHAR(64) NOT NULL,
    tag_val  VARCHAR(128) NOT NULL,
    PRIMARY KEY (run_id, tag_key),
    CONSTRAINT fk_rt_run
      FOREIGN KEY (run_id) REFERENCES experiment_run(run_id)
      ON DELETE CASCADE
) ENGINE=InnoDB;

SET FOREIGN_KEY_CHECKS = 1;

-- ---------------------------------------------------------------------------
-- View (MySQL has no MATERIALIZED VIEW — use regular VIEW; refresh = redefine N/A)
-- ---------------------------------------------------------------------------
CREATE VIEW v_run_summary AS
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
