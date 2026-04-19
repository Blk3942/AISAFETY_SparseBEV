"""
将单次 val 结果（experiment_run + metrics_official + metrics_safety_critical
+ metrics_per_class）写入 safetyai_sparsebev 数据库。

用法：
    python3 tools/import_run_to_db.py \
        --official_json  submission/pts_bbox/metrics_summary.json \
        --safety_json    submission/pts_bbox/safety_critical_eval_out/metrics_summary.json \
        --config         configs/r50_nuimg_704x256.py \
        --weights        checkpoints/r50_nuimg_704x256.pth \
        --dataset_root   /root/autodl-tmp/NuScenes/ \
        --eval_split     val \
        --version        v1.0-trainval \
        --host 127.0.0.1 --port 13306 \
        --user remote1   --password 你的密码
"""
import argparse, json, math, subprocess
from pathlib import Path
import pymysql
import pymysql.cursors


def clean_json(obj):
    """递归将 NaN/Inf 替换为 None，避免 MySQL JSON 解析错误。"""
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json(v) for v in obj]
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--official_json',  default='submission/pts_bbox/metrics_summary.json')
    p.add_argument('--safety_json',    default='submission/pts_bbox/safety_critical_eval_out/metrics_summary.json')
    p.add_argument('--config',         default='configs/r50_nuimg_704x256.py')
    p.add_argument('--weights',        default='checkpoints/r50_nuimg_704x256.pth')
    p.add_argument('--dataset_root',   default='/root/autodl-tmp/NuScenes/')
    p.add_argument('--eval_split',     default='val')
    p.add_argument('--version',        default='v1.0-trainval')
    p.add_argument('--backbone',       default='ResNet-50')
    p.add_argument('--input_resolution', default='256x704')
    p.add_argument('--temporal_frames', type=int, default=8)
    p.add_argument('--notes',          default='')
    p.add_argument('--host',           default='127.0.0.1')
    p.add_argument('--port',           type=int, default=13306)
    p.add_argument('--user',           default='remote1')
    p.add_argument('--password',       default='')
    p.add_argument('--database',       default='safetyai_sparsebev')
    return p.parse_args()


def git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode().strip()[:16]
    except Exception:
        return None


def connect(args):
    return pymysql.connect(
        host=args.host, port=args.port,
        user=args.user, password=args.password,
        database=args.database,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )


def main():
    args = parse_args()

    off = json.load(open(args.official_json))
    saf = json.load(open(args.safety_json))

    conn = connect(args)
    cur  = conn.cursor()

    # ---- 查 dataset_id ----
    cur.execute('SELECT dataset_id FROM nuscenes_dataset WHERE nuscenes_version=%s',
                (args.version,))
    row = cur.fetchone()
    if row is None:
        cur.execute(
            'INSERT INTO nuscenes_dataset (nuscenes_version, data_root_hint) VALUES (%s, %s)',
            (args.version, args.dataset_root)
        )
        conn.commit()
        dataset_id = cur.lastrowid
        print(f'  新建 nuscenes_dataset, id={dataset_id}')
    else:
        dataset_id = row['dataset_id']
    print(f'  dataset_id = {dataset_id}')

    # ---- experiment_run ----
    cur.execute(
        '''INSERT INTO experiment_run
           (gt_dataset_id, project_name, config_path, weights_path,
            dataset_root, eval_split, git_commit, notes,
            backbone, input_resolution, temporal_frames, tags)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)''',
        (dataset_id, 'AISAFETY_SparseBEV',
         args.config, args.weights,
         args.dataset_root, args.eval_split,
         git_commit(), args.notes,
         args.backbone, args.input_resolution,
         args.temporal_frames,
         json.dumps(['full_val']))
    )
    conn.commit()
    run_id = cur.lastrowid
    print(f'  experiment_run  run_id = {run_id}')

    # ---- metrics_official ----
    off_tp = off.get('tp_errors', {})
    # raw_json: 仅保存可序列化的核心字段，避免 cfg 字段含 NaN 导致 MySQL JSON 解析错误
    off_raw = clean_json({k: v for k, v in off.items() if k not in ('cfg',)})
    cur.execute(
        '''INSERT INTO metrics_official
           (run_id, nd_score, mean_ap, mATE, mASE, mAOE, mAVE, mAAE, eval_time_s, raw_json)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)''',
        (run_id,
         off['nd_score'], off['mean_ap'],
         off_tp.get('trans_err'), off_tp.get('scale_err'),
         off_tp.get('orient_err'), off_tp.get('vel_err'), off_tp.get('attr_err'),
         off.get('eval_time'),
         json.dumps(off_raw))
    )
    conn.commit()
    print(f'  metrics_official 写入完成  NDS={off["nd_score"]:.4f}  mAP={off["mean_ap"]:.4f}')

    # ---- metrics_safety_critical ----
    saf_tp = saf.get('tp_errors', {})
    saf_raw = clean_json({k: v for k, v in saf.items() if k not in ('cfg',)})
    cur.execute(
        '''INSERT INTO metrics_safety_critical
           (run_id, nd_score, mean_ap, mATE, mASE, mAOE, mAVE, mAAE, config_name, raw_json)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)''',
        (run_id,
         saf['nd_score'], saf['mean_ap'],
         saf_tp.get('trans_err'), saf_tp.get('scale_err'),
         saf_tp.get('orient_err'), saf_tp.get('vel_err'), saf_tp.get('attr_err'),
         'detection_safety_critical',
         json.dumps(saf_raw))
    )
    conn.commit()
    print(f'  metrics_safety_critical  NDS={saf["nd_score"]:.4f}  mAP={saf["mean_ap"]:.4f}')

    def _f(v):
        """把 NaN/Inf 转成 None，MySQL 不接受这两个值。"""
        if v is None:
            return None
        try:
            return None if (math.isnan(v) or math.isinf(v)) else v
        except TypeError:
            return v

    # ---- metrics_per_class (official) ----
    per_class_rows = []
    label_tp = off.get('label_tp_errors', {})
    label_ap = off.get('label_aps', {})
    for cls, ap_dict in label_ap.items():
        vals = [v for v in ap_dict.values() if v is not None and not math.isnan(v)]
        ap = sum(vals) / len(vals) if vals else None
        tp  = label_tp.get(cls, {})
        per_class_rows.append((
            run_id, 'official', cls, _f(ap),
            _f(tp.get('trans_err')), _f(tp.get('scale_err')),
            _f(tp.get('orient_err')), _f(tp.get('vel_err')), _f(tp.get('attr_err'))
        ))
    if per_class_rows:
        cur.executemany(
            '''INSERT IGNORE INTO metrics_per_class
               (run_id, eval_type, class_name, ap, ate, ase, aoe, ave, aae)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)''',
            per_class_rows
        )
        conn.commit()
        print(f'  metrics_per_class (official)  {len(per_class_rows)} 条')

    # ---- metrics_per_class (safety) ----
    per_class_rows_s = []
    label_tp_s = saf.get('label_tp_errors', {})
    label_ap_s = saf.get('label_aps', {})
    for cls, ap_dict in label_ap_s.items():
        vals = [v for v in ap_dict.values() if v is not None and not math.isnan(v)]
        ap = sum(vals) / len(vals) if vals else None
        tp  = label_tp_s.get(cls, {})
        per_class_rows_s.append((
            run_id, 'safety_critical', cls, _f(ap),
            _f(tp.get('trans_err')), _f(tp.get('scale_err')),
            _f(tp.get('orient_err')), _f(tp.get('vel_err')), _f(tp.get('attr_err'))
        ))
    if per_class_rows_s:
        cur.executemany(
            '''INSERT IGNORE INTO metrics_per_class
               (run_id, eval_type, class_name, ap, ate, ase, aoe, ave, aae)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)''',
            per_class_rows_s
        )
        conn.commit()
        print(f'  metrics_per_class (safety)    {len(per_class_rows_s)} 条')

    # ---- v_run_summary 验证 ----
    cur.execute('SELECT * FROM v_run_summary WHERE run_id=%s', (run_id,))
    summary = cur.fetchone()
    print(f'\n=== 写入完成 ===')
    print(f'  run_id           = {run_id}')
    print(f'  official NDS     = {summary["official_nds"]:.4f}')
    print(f'  official mAP     = {summary["official_map"]:.4f}')
    print(f'  safety NDS       = {summary["safety_nds"]:.4f}')
    print(f'  safety mAP       = {summary["safety_map"]:.4f}')
    print(f'  num_samples      = {summary["num_samples"]}')

    cur.close()
    conn.close()


if __name__ == '__main__':
    main()
