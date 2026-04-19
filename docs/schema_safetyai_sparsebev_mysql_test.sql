-- Smoke test data for safetyai_sparsebev (run after schema_safetyai_sparsebev_mysql.sql)
USE safetyai_sparsebev;

INSERT INTO nuscenes_dataset (nuscenes_version, data_root_hint, description)
VALUES ('v1.0-mini', '/path/to/mini', 'test corpus');

SET @ds = LAST_INSERT_ID();

INSERT INTO ground_truth_sample (dataset_id, sample_token, scene_token, timestamp_us)
VALUES (@ds, 'sample_token_demo_001', 'scene-0103', 1531234567890123);

SET @gts = LAST_INSERT_ID();

INSERT INTO ground_truth_box (
  gt_sample_id, annotation_token, class_name,
  translation_x, translation_y, translation_z,
  size_wlh_0, size_wlh_1, size_wlh_2,
  rotation_w, rotation_x, rotation_y, rotation_z,
  velocity_x, velocity_y, attribute_name, num_pts, ego_dist
) VALUES (
  @gts, 'ann_demo', 'car',
  10.0, 5.0, -1.0,
  2.0, 4.5, 1.8,
  1.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 'vehicle.moving', 100, 12.0
);

SET @gtb = LAST_INSERT_ID();

INSERT INTO experiment_run (
  gt_dataset_id, project_name, config_path, weights_path, dataset_root, eval_split,
  backbone, input_resolution, temporal_frames, tags, avg_latency_ms, notes
) VALUES (
  @ds, 'AISAFETY_SparseBEV', 'configs/r50_nuimg_704x256_mini.py', 'checkpoints/r50.pth',
  '/data/nuscenes', 'mini_val',
  'ResNet-50', '256x704', 8, JSON_ARRAY('smoke_test'), 25.0, 'smoke test run'
);

SET @run = LAST_INSERT_ID();

INSERT INTO prediction_sample (run_id, gt_sample_id, sample_token)
VALUES (@run, @gts, 'sample_token_demo_001');

SET @ps = LAST_INSERT_ID();

INSERT INTO prediction_box (
  pred_sample_id, class_name, score,
  translation_x, translation_y, translation_z,
  size_wlh_0, size_wlh_1, size_wlh_2,
  rotation_w, rotation_x, rotation_y, rotation_z
) VALUES (
  @ps, 'car', 0.9,
  10.1, 5.1, -1.0,
  2.0, 4.5, 1.8,
  1.0, 0.0, 0.0, 0.0
);

SET @pb = LAST_INSERT_ID();

INSERT INTO metrics_official (run_id, nd_score, mean_ap, mATE, mASE, mAOE, mAVE, mAAE, eval_time_s, raw_json)
VALUES (@run, 0.4746, 0.4315, 0.6378, 0.4448, 0.5766, 0.4495, 0.3026, 2.3, JSON_OBJECT('source', 'smoke'));

INSERT INTO metrics_safety_critical (run_id, nd_score, mean_ap, mATE, raw_json)
VALUES (@run, 0.4276, 0.5110, 0.6131, JSON_OBJECT('source', 'smoke'));

INSERT INTO metrics_per_class (run_id, eval_type, class_name, ap, ate)
VALUES (@run, 'official', 'car', 0.707, 0.390);

INSERT INTO metrics_per_distance (run_id, eval_type, class_name, dist_bin_lo, dist_bin_hi, ap, num_gt, num_pred)
VALUES (@run, 'official', 'car', 0.0, 10.0, 0.65, 10, 12);

INSERT INTO confusion_matrix_cell (run_id, eval_type, dist_th_m, gt_class, pred_class, count_val)
VALUES (@run, 'official', 2.0, 'car', 'car', 42);

INSERT INTO match_pair (run_id, eval_type, dist_th_m, gt_box_id, pred_box_id, match_dist, gt_class, pred_class, is_tp)
VALUES (@run, 'official', 2.0, @gtb, @pb, 0.15, 'car', 'car', 1);

INSERT INTO run_tag (run_id, tag_key, tag_val)
VALUES (@run, 'env', 'ci-smoke');

SELECT 'experiment_run' AS t, COUNT(*) AS n FROM experiment_run
UNION ALL SELECT 'v_run_summary', COUNT(*) FROM v_run_summary;

SELECT run_id, official_nds, official_map, safety_nds, num_samples FROM v_run_summary;
