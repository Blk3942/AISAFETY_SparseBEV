-- safetyai_sparsebev schema (MySQL 8.x)
-- Derived from docs/sql_val_database_design.md
-- Last updated: 2026-04 (v1.3 — added ground_truth_ego, nuscenes_scene,
--                         nuscenes_calibrated_sensor, nuscenes_sample_camera;
--                         added ego_dist + visibility_level to ground_truth_box)

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

DROP DATABASE IF EXISTS safetyai_sparsebev;
CREATE DATABASE safetyai_sparsebev
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE safetyai_sparsebev;

-- ---------------------------------------------------------------------------
-- nuscenes_dataset
-- ---------------------------------------------------------------------------
CREATE TABLE nuscenes_dataset (
    dataset_id        BIGINT AUTO_INCREMENT PRIMARY KEY,
    nuscenes_version  VARCHAR(32)  NOT NULL,
    data_root_hint    TEXT,
    description       TEXT,
    imported_at       TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_nuscenes_version (nuscenes_version)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ---------------------------------------------------------------------------
-- nuscenes_scene  (scene metadata: name, description, location, date)
-- ---------------------------------------------------------------------------
CREATE TABLE nuscenes_scene (
    scene_token   VARCHAR(64)  NOT NULL PRIMARY KEY,
    dataset_id    BIGINT       NOT NULL,
    name          VARCHAR(64),
    description   TEXT,
    nbr_samples   INT,
    log_location  VARCHAR(64)  COMMENT 'map region, e.g. singapore-onenorth / boston-seaport',
    log_date      VARCHAR(32)  COMMENT 'recording date, e.g. 2018-07-11',
    log_token     VARCHAR(64),
    CONSTRAINT fk_scene_dataset
        FOREIGN KEY (dataset_id) REFERENCES nuscenes_dataset(dataset_id)
        ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='nuScenes scene metadata (description, location, date)';

-- ---------------------------------------------------------------------------
-- nuscenes_calibrated_sensor  (camera intrinsics + extrinsics)
-- ---------------------------------------------------------------------------
CREATE TABLE nuscenes_calibrated_sensor (
    cs_token         VARCHAR(64)  NOT NULL PRIMARY KEY,
    sensor_token     VARCHAR(64),
    channel          VARCHAR(32)  COMMENT 'CAM_FRONT / CAM_BACK_LEFT ...',
    translation      JSON         COMMENT '[x,y,z] camera position in vehicle frame (m)',
    rotation         JSON         COMMENT '[w,x,y,z] vehicle->camera rotation quaternion',
    camera_intrinsic JSON         COMMENT '3x3 intrinsic matrix [[fx,0,cx],[0,fy,cy],[0,0,1]]'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='Camera calibration params (intrinsics+extrinsics) from nuScenes calibrated_sensor.json';

-- ---------------------------------------------------------------------------
-- ground_truth_sample
-- ---------------------------------------------------------------------------
CREATE TABLE ground_truth_sample (
    id            BIGINT AUTO_INCREMENT PRIMARY KEY,
    dataset_id    BIGINT       NOT NULL,
    sample_token  VARCHAR(64)  NOT NULL,
    scene_token   VARCHAR(64),
    timestamp_us  BIGINT,
    UNIQUE KEY uk_dataset_sample (dataset_id, sample_token),
    KEY idx_gt_sample_scene (dataset_id, scene_token),
    CONSTRAINT fk_gt_sample_dataset
        FOREIGN KEY (dataset_id) REFERENCES nuscenes_dataset(dataset_id)
        ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ---------------------------------------------------------------------------
-- nuscenes_sample_camera  (per-frame 6-camera image paths + calibration link)
-- ---------------------------------------------------------------------------
CREATE TABLE nuscenes_sample_camera (
    id               BIGINT       NOT NULL AUTO_INCREMENT PRIMARY KEY,
    gt_sample_id     BIGINT       NOT NULL,
    channel          VARCHAR(32)  NOT NULL  COMMENT 'CAM_FRONT / CAM_BACK ...',
    filename         VARCHAR(256) NOT NULL  COMMENT 'image path relative to dataroot',
    sd_token         VARCHAR(64)            COMMENT 'sample_data.token',
    cs_token         VARCHAR(64)            COMMENT '-> nuscenes_calibrated_sensor.cs_token',
    timestamp_us     BIGINT,
    width            SMALLINT,
    height           SMALLINT,
    CONSTRAINT fk_sc_gt   FOREIGN KEY (gt_sample_id) REFERENCES ground_truth_sample(id),
    CONSTRAINT fk_sc_cs   FOREIGN KEY (cs_token)     REFERENCES nuscenes_calibrated_sensor(cs_token),
    INDEX idx_sc_sample (gt_sample_id),
    INDEX idx_sc_channel (channel)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='Per-frame 6-camera image paths and calibration association';

-- ---------------------------------------------------------------------------
-- ground_truth_ego  (ego vehicle pose + derived speed, one row per sample)
-- ---------------------------------------------------------------------------
CREATE TABLE ground_truth_ego (
    id                BIGINT AUTO_INCREMENT PRIMARY KEY,
    gt_sample_id      BIGINT       NOT NULL,
    pos_x             DOUBLE       NOT NULL,
    pos_y             DOUBLE       NOT NULL,
    pos_z             DOUBLE       NOT NULL,
    rot_w             DOUBLE       NOT NULL,
    rot_x             DOUBLE       NOT NULL,
    rot_y             DOUBLE       NOT NULL,
    rot_z             DOUBLE       NOT NULL,
    yaw_rad           DOUBLE,
    speed_mps         DOUBLE,
    vx_mps            DOUBLE,
    vy_mps            DOUBLE,
    ego_timestamp_us  BIGINT,
    UNIQUE KEY uk_ego_sample (gt_sample_id),
    CONSTRAINT fk_ego_sample
        FOREIGN KEY (gt_sample_id) REFERENCES ground_truth_sample(id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='nuScenes ego vehicle pose and derived velocity (one row per sample)';

-- ---------------------------------------------------------------------------
-- ground_truth_box
-- ---------------------------------------------------------------------------
CREATE TABLE ground_truth_box (
    id               BIGINT AUTO_INCREMENT PRIMARY KEY,
    gt_sample_id     BIGINT       NOT NULL,
    annotation_token VARCHAR(64),
    class_name       VARCHAR(64)  NOT NULL,
    translation_x    DOUBLE       NOT NULL,
    translation_y    DOUBLE       NOT NULL,
    translation_z    DOUBLE       NOT NULL,
    size_wlh_0       DOUBLE       NOT NULL,
    size_wlh_1       DOUBLE       NOT NULL,
    size_wlh_2       DOUBLE       NOT NULL,
    rotation_w       DOUBLE       NOT NULL,
    rotation_x       DOUBLE       NOT NULL,
    rotation_y       DOUBLE       NOT NULL,
    rotation_z       DOUBLE       NOT NULL,
    velocity_x       DOUBLE,
    velocity_y       DOUBLE,
    attribute_name   VARCHAR(128),
    num_pts          INT,
    ego_dist         DOUBLE       COMMENT 'distance from ego vehicle center (m)',
    visibility_level TINYINT      COMMENT '1=<40% visible, 2=40~60%, 3=60~80%, 4=80~100% (nuScenes visibility_token)',
    KEY fk_gt_box_sample (gt_sample_id),
    KEY idx_gt_box_class (class_name),
    CONSTRAINT fk_gt_box_sample
        FOREIGN KEY (gt_sample_id) REFERENCES ground_truth_sample(id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ---------------------------------------------------------------------------
-- experiment_run
-- ---------------------------------------------------------------------------
CREATE TABLE experiment_run (
    run_id            BIGINT AUTO_INCREMENT PRIMARY KEY,
    gt_dataset_id     BIGINT       NOT NULL,
    project_name      VARCHAR(128) NOT NULL DEFAULT 'AISAFETY_SparseBEV',
    config_path       TEXT,
    weights_path      TEXT,
    dataset_root      TEXT         NOT NULL,
    eval_split        VARCHAR(32)  NOT NULL,
    created_at        TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    git_commit        VARCHAR(64),
    notes             TEXT,
    backbone          VARCHAR(64),
    input_resolution  VARCHAR(32),
    temporal_frames   INT,
    tags              JSON,
    avg_latency_ms    DOUBLE,
    KEY fk_run_dataset (gt_dataset_id),
    CONSTRAINT fk_run_dataset
        FOREIGN KEY (gt_dataset_id) REFERENCES nuscenes_dataset(dataset_id)
        ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ---------------------------------------------------------------------------
-- prediction_sample
-- ---------------------------------------------------------------------------
CREATE TABLE prediction_sample (
    id            BIGINT AUTO_INCREMENT PRIMARY KEY,
    run_id        BIGINT       NOT NULL,
    gt_sample_id  BIGINT       NOT NULL,
    sample_token  VARCHAR(64)  NOT NULL,
    UNIQUE KEY uk_run_sample (run_id, sample_token),
    KEY idx_pred_sample_gt  (gt_sample_id),
    KEY idx_pred_sample_run (run_id),
    CONSTRAINT fk_pred_sample_gt
        FOREIGN KEY (gt_sample_id) REFERENCES ground_truth_sample(id)
        ON DELETE RESTRICT,
    CONSTRAINT fk_pred_sample_run
        FOREIGN KEY (run_id) REFERENCES experiment_run(run_id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ---------------------------------------------------------------------------
-- prediction_box
-- ---------------------------------------------------------------------------
CREATE TABLE prediction_box (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    pred_sample_id  BIGINT       NOT NULL,
    class_name      VARCHAR(64)  NOT NULL,
    score           DOUBLE       NOT NULL,
    translation_x   DOUBLE       NOT NULL,
    translation_y   DOUBLE       NOT NULL,
    translation_z   DOUBLE       NOT NULL,
    size_wlh_0      DOUBLE       NOT NULL,
    size_wlh_1      DOUBLE       NOT NULL,
    size_wlh_2      DOUBLE       NOT NULL,
    rotation_w      DOUBLE       NOT NULL,
    rotation_x      DOUBLE       NOT NULL,
    rotation_y      DOUBLE       NOT NULL,
    rotation_z      DOUBLE       NOT NULL,
    velocity_x      DOUBLE,
    velocity_y      DOUBLE,
    attribute_name  VARCHAR(128),
    KEY fk_pred_box_sample (pred_sample_id),
    KEY idx_pred_box_class (class_name),
    CONSTRAINT fk_pred_box_sample
        FOREIGN KEY (pred_sample_id) REFERENCES prediction_sample(id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ---------------------------------------------------------------------------
-- metrics_official
-- ---------------------------------------------------------------------------
CREATE TABLE metrics_official (
    run_id       BIGINT  NOT NULL PRIMARY KEY,
    nd_score     DOUBLE  NOT NULL,
    mean_ap      DOUBLE  NOT NULL,
    mATE         DOUBLE,
    mASE         DOUBLE,
    mAOE         DOUBLE,
    mAVE         DOUBLE,
    mAAE         DOUBLE,
    eval_time_s  DOUBLE,
    raw_json     JSON,
    CONSTRAINT fk_mo_run
        FOREIGN KEY (run_id) REFERENCES experiment_run(run_id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ---------------------------------------------------------------------------
-- metrics_safety_critical
-- ---------------------------------------------------------------------------
CREATE TABLE metrics_safety_critical (
    run_id       BIGINT       NOT NULL PRIMARY KEY,
    nd_score     DOUBLE       NOT NULL,
    mean_ap      DOUBLE       NOT NULL,
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ---------------------------------------------------------------------------
-- metrics_per_class
-- ---------------------------------------------------------------------------
CREATE TABLE metrics_per_class (
    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    run_id      BIGINT       NOT NULL,
    eval_type   VARCHAR(32)  NOT NULL,
    class_name  VARCHAR(64)  NOT NULL,
    ap          DOUBLE,
    ate         DOUBLE,
    ase         DOUBLE,
    aoe         DOUBLE,
    ave         DOUBLE,
    aae         DOUBLE,
    UNIQUE KEY uk_mpc (run_id, eval_type, class_name),
    KEY idx_mpc_run_class (run_id, class_name),
    CONSTRAINT fk_mpc_run
        FOREIGN KEY (run_id) REFERENCES experiment_run(run_id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ---------------------------------------------------------------------------
-- metrics_per_distance
-- ---------------------------------------------------------------------------
CREATE TABLE metrics_per_distance (
    id           BIGINT AUTO_INCREMENT PRIMARY KEY,
    run_id       BIGINT       NOT NULL,
    eval_type    VARCHAR(32)  NOT NULL,
    class_name   VARCHAR(64)  NOT NULL,
    dist_bin_lo  DOUBLE       NOT NULL,
    dist_bin_hi  DOUBLE       NOT NULL,
    ap           DOUBLE,
    ate          DOUBLE,
    num_gt       INT,
    num_pred     INT,
    UNIQUE KEY uk_mpd (run_id, eval_type, class_name, dist_bin_lo),
    KEY idx_mpd_run (run_id, eval_type),
    CONSTRAINT fk_mpd_run
        FOREIGN KEY (run_id) REFERENCES experiment_run(run_id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ---------------------------------------------------------------------------
-- confusion_matrix_cell
-- ---------------------------------------------------------------------------
CREATE TABLE confusion_matrix_cell (
    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    run_id      BIGINT       NOT NULL,
    eval_type   VARCHAR(32)  NOT NULL,
    dist_th_m   DOUBLE,
    gt_class    VARCHAR(64)  NOT NULL,
    pred_class  VARCHAR(64)  NOT NULL,
    count_val   INT          NOT NULL,
    UNIQUE KEY uk_cm (run_id, eval_type, dist_th_m, gt_class, pred_class),
    KEY idx_cm_run (run_id, eval_type),
    CONSTRAINT fk_cm_run
        FOREIGN KEY (run_id) REFERENCES experiment_run(run_id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ---------------------------------------------------------------------------
-- match_pair
-- ---------------------------------------------------------------------------
CREATE TABLE match_pair (
    id           BIGINT AUTO_INCREMENT PRIMARY KEY,
    run_id       BIGINT       NOT NULL,
    eval_type    VARCHAR(32)  NOT NULL,
    dist_th_m    DOUBLE,
    gt_box_id    BIGINT,
    pred_box_id  BIGINT,
    match_dist   DOUBLE,
    gt_class     VARCHAR(64),
    pred_class   VARCHAR(64),
    is_tp        TINYINT(1)   NOT NULL,
    KEY idx_match_run  (run_id, eval_type),
    KEY idx_match_gt   (gt_box_id),
    KEY idx_match_pred (pred_box_id),
    CONSTRAINT fk_mp_run
        FOREIGN KEY (run_id)      REFERENCES experiment_run(run_id)  ON DELETE CASCADE,
    CONSTRAINT fk_mp_gt_box
        FOREIGN KEY (gt_box_id)   REFERENCES ground_truth_box(id)    ON DELETE SET NULL,
    CONSTRAINT fk_mp_pred_box
        FOREIGN KEY (pred_box_id) REFERENCES prediction_box(id)      ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ---------------------------------------------------------------------------
-- run_tag
-- ---------------------------------------------------------------------------
CREATE TABLE run_tag (
    run_id    BIGINT       NOT NULL,
    tag_key   VARCHAR(64)  NOT NULL,
    tag_val   VARCHAR(128) NOT NULL,
    PRIMARY KEY (run_id, tag_key),
    CONSTRAINT fk_rt_run
        FOREIGN KEY (run_id) REFERENCES experiment_run(run_id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ---------------------------------------------------------------------------
-- v_run_summary  (view)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_run_summary AS
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
    mo.nd_score  AS official_nds,
    mo.mean_ap   AS official_map,
    ms.nd_score  AS safety_nds,
    ms.mean_ap   AS safety_map,
    (SELECT COUNT(*) FROM prediction_sample ps WHERE ps.run_id = r.run_id) AS num_samples
FROM experiment_run r
LEFT JOIN metrics_official       mo ON mo.run_id = r.run_id
LEFT JOIN metrics_safety_critical ms ON ms.run_id = r.run_id;

SET FOREIGN_KEY_CHECKS = 1;
