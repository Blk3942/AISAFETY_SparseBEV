"""
导入 nuScenes GT 数据到 safetyai_sparsebev 数据库。
写入表：nuscenes_dataset, ground_truth_sample, ground_truth_box

用法（在远端服务器上，SSH 反向隧道 -R 13306:127.0.0.1:3306 保持开启时）：
    python3 tools/import_gt_to_db.py \
        --data_root /root/autodl-tmp/nuscense-full \
        --version v1.0-trainval \
        --host 127.0.0.1 \
        --port 13306 \
        --user remote \
        --password 你的密码 \
        --batch_size 2000
"""

import argparse
import json
import math
import time
from pathlib import Path

import pymysql
import pymysql.cursors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/root/autodl-tmp/nuscense-full")
    p.add_argument("--version",   default="v1.0-trainval",
                   help="v1.0-trainval | v1.0-mini | v1.0-test")
    p.add_argument("--host",     default="127.0.0.1")
    p.add_argument("--port",     type=int, default=13306)
    p.add_argument("--user",     default="remote")
    p.add_argument("--password", default="")
    p.add_argument("--database", default="safetyai_sparsebev")
    p.add_argument("--batch_size", type=int, default=2000,
                   help="每次批量 INSERT 的行数")
    p.add_argument("--skip_boxes", action="store_true",
                   help="只导 sample，不导 box（调试用）")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def connect(args):
    return pymysql.connect(
        host=args.host, port=args.port,
        user=args.user, password=args.password,
        database=args.database,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )


def batch_insert(cur, sql, rows, batch_size):
    """分批 executemany，避免单次传输过大。"""
    total = len(rows)
    for i in range(0, total, batch_size):
        chunk = rows[i: i + batch_size]
        cur.executemany(sql, chunk)
    return total


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    meta_dir = Path(args.data_root) / args.version

    print(f"[1/5] 读取 {meta_dir} ...")
    t0 = time.time()
    samples     = json.load(open(meta_dir / "sample.json"))
    annotations = json.load(open(meta_dir / "sample_annotation.json"))
    categories  = {c["token"]: c["name"]
                   for c in json.load(open(meta_dir / "category.json"))}
    instances   = {i["token"]: categories.get(i["category_token"], "unknown")
                   for i in json.load(open(meta_dir / "instance.json"))}
    attributes  = {a["token"]: a["name"]
                   for a in json.load(open(meta_dir / "attribute.json"))}
    print(f"    samples={len(samples)}, annotations={len(annotations)}, "
          f"耗时 {time.time()-t0:.1f}s")

    # sample_token -> scene_token, timestamp
    sample_map = {s["token"]: s for s in samples}

    print(f"[2/5] 连接数据库 {args.host}:{args.port} ...")
    conn = connect(args)
    cur  = conn.cursor()

    # ---- nuscenes_dataset 行 ----
    print(f"[3/5] 注册 nuscenes_dataset: {args.version} ...")
    cur.execute(
        "INSERT INTO nuscenes_dataset (nuscenes_version, data_root_hint, description) "
        "VALUES (%s, %s, %s) "
        "ON DUPLICATE KEY UPDATE data_root_hint=VALUES(data_root_hint)",
        (args.version, args.data_root,
         f"Full import from {args.data_root}/{args.version}")
    )
    conn.commit()
    cur.execute("SELECT dataset_id FROM nuscenes_dataset WHERE nuscenes_version=%s",
                (args.version,))
    dataset_id = cur.fetchone()["dataset_id"]
    print(f"    dataset_id = {dataset_id}")

    # ---- ground_truth_sample ----
    print(f"[4/5] 导入 ground_truth_sample ({len(samples)} 条) ...")
    t1 = time.time()

    # 先查已存在的 sample_token，避免重复
    cur.execute("SELECT sample_token FROM ground_truth_sample WHERE dataset_id=%s",
                (dataset_id,))
    existing_tokens = {r["sample_token"] for r in cur.fetchall()}
    print(f"    已存在 {len(existing_tokens)} 条，本次新增 "
          f"{len(samples) - len(existing_tokens)} 条")

    sample_rows = [
        (dataset_id, s["token"], s.get("scene_token", ""), s.get("timestamp"))
        for s in samples
        if s["token"] not in existing_tokens
    ]
    if sample_rows:
        batch_insert(
            cur,
            "INSERT IGNORE INTO ground_truth_sample "
            "(dataset_id, sample_token, scene_token, timestamp_us) "
            "VALUES (%s, %s, %s, %s)",
            sample_rows, args.batch_size
        )
        conn.commit()
    print(f"    完成，耗时 {time.time()-t1:.1f}s")

    # 建立 sample_token -> gt_sample_id 映射
    cur.execute("SELECT id, sample_token FROM ground_truth_sample WHERE dataset_id=%s",
                (dataset_id,))
    token_to_id = {r["sample_token"]: r["id"] for r in cur.fetchall()}

    # ---- ground_truth_box ----
    if args.skip_boxes:
        print("[5/5] --skip_boxes 已跳过 box 导入")
    else:
        print(f"[5/5] 导入 ground_truth_box ({len(annotations)} 条) ...")
        t2 = time.time()

        # 已存在的 annotation_token
        cur.execute("SELECT annotation_token FROM ground_truth_box b "
                    "JOIN ground_truth_sample s ON s.id=b.gt_sample_id "
                    "WHERE s.dataset_id=%s", (dataset_id,))
        existing_ann = {r["annotation_token"] for r in cur.fetchall()}
        print(f"    已存在 {len(existing_ann)} 条 box")

        box_rows = []
        skipped  = 0
        for ann in annotations:
            if ann["token"] in existing_ann:
                skipped += 1
                continue
            gt_id = token_to_id.get(ann["sample_token"])
            if gt_id is None:
                continue  # sample 不在当前 dataset
            t = ann["translation"]   # [x, y, z]
            sz = ann["size"]         # [w, l, h]
            r  = ann["rotation"]     # [w, x, y, z]
            vx = vy = None
            # velocity 字段在 annotation 里不直接存在（nuScenes 在 instance 里计算），此处留空
            attr_name = None
            if ann.get("attribute_tokens"):
                attr_name = attributes.get(ann["attribute_tokens"][0])
            class_name = instances.get(ann["instance_token"], "unknown")
            box_rows.append((
                gt_id,
                ann["token"],          # annotation_token
                class_name,
                t[0], t[1], t[2],      # translation
                sz[0], sz[1], sz[2],   # size wlh
                r[0], r[1], r[2], r[3],# rotation w,x,y,z
                vx, vy,
                attr_name,
                ann.get("num_lidar_pts"),
                None,                  # ego_dist: 需要 ego_pose 才能算，留 NULL
            ))

        print(f"    新增 {len(box_rows)} 条，跳过 {skipped} 条（已存在）")
        if box_rows:
            total_batches = math.ceil(len(box_rows) / args.batch_size)
            for i, start in enumerate(range(0, len(box_rows), args.batch_size)):
                chunk = box_rows[start: start + args.batch_size]
                cur.executemany(
                    "INSERT IGNORE INTO ground_truth_box "
                    "(gt_sample_id, annotation_token, class_name, "
                    " translation_x, translation_y, translation_z, "
                    " size_wlh_0, size_wlh_1, size_wlh_2, "
                    " rotation_w, rotation_x, rotation_y, rotation_z, "
                    " velocity_x, velocity_y, attribute_name, num_pts, ego_dist) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    chunk
                )
                conn.commit()
                if (i + 1) % 10 == 0 or (i + 1) == total_batches:
                    pct = (i + 1) / total_batches * 100
                    elapsed = time.time() - t2
                    print(f"    [{i+1}/{total_batches}] {pct:.0f}%  "
                          f"已写入 ~{(i+1)*args.batch_size} 条  "
                          f"耗时 {elapsed:.0f}s")
        print(f"    box 导入完成，总耗时 {time.time()-t2:.1f}s")

    # ---- 汇总 ----
    cur.execute("SELECT COUNT(*) AS n FROM ground_truth_sample WHERE dataset_id=%s",
                (dataset_id,))
    n_sample = cur.fetchone()["n"]
    cur.execute("SELECT COUNT(*) AS n FROM ground_truth_box b "
                "JOIN ground_truth_sample s ON s.id=b.gt_sample_id "
                "WHERE s.dataset_id=%s", (dataset_id,))
    n_box = cur.fetchone()["n"]
    print(f"\n完成！dataset_id={dataset_id}  "
          f"ground_truth_sample={n_sample}  ground_truth_box={n_box}")
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
