"""
回填 ground_truth_box.velocity_x / velocity_y。
nuScenes SDK 的 nusc.box_velocity(annotation_token) 返回全局坐标系 (vx, vy, vz)，
取前两分量写回 DB。
"""
import argparse
import math
import pymysql
import pymysql.cursors


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataroot", default=r"E:\Data\Nuscenes\Full")
    p.add_argument("--version", default="v1.0-trainval")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=3306)
    p.add_argument("--user", default="root")
    p.add_argument("--password", default="")
    p.add_argument("--database", default="safetyai_sparsebev")
    p.add_argument("--batch", type=int, default=5000)
    return p.parse_args()


def main():
    args = parse_args()

    print("加载 nuScenes SDK ...")
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    # annotation_token -> (vx, vy)；nan 表示无法计算（首/末帧）
    print("计算所有 annotation 速度 ...")
    vel_map: dict = {}
    for ann in nusc.sample_annotation:
        tok = ann["token"]
        try:
            v = nusc.box_velocity(tok)
            vx, vy = float(v[0]), float(v[1])
            if math.isnan(vx) or math.isnan(vy):
                vx = vy = None  # type: ignore
        except Exception:
            vx = vy = None  # type: ignore
        vel_map[tok] = (vx, vy)

    print(f"  共 {len(vel_map)} 条，有效速度 {sum(1 for v in vel_map.values() if v[0] is not None)} 条")

    conn = pymysql.connect(
        host=args.host, port=args.port,
        user=args.user, password=args.password,
        database=args.database, charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor, autocommit=False,
    )
    cur = conn.cursor()

    print("读取待更新 box ...")
    cur.execute(
        "SELECT id, annotation_token FROM ground_truth_box WHERE velocity_x IS NULL AND annotation_token IS NOT NULL"
    )
    rows = cur.fetchall()
    print(f"  待更新 {len(rows)} 条")

    updates = []
    for r in rows:
        v = vel_map.get(r["annotation_token"])
        if v is None:
            continue
        vx, vy = v
        if vx is None:
            continue
        updates.append((round(vx, 4), round(vy, 4), r["id"]))

    print(f"  可写入 {len(updates)} 条（其余首/末帧无速度，保持 NULL）")

    BATCH = args.batch
    for i in range(0, len(updates), BATCH):
        chunk = updates[i : i + BATCH]
        cur.executemany(
            "UPDATE ground_truth_box SET velocity_x=%s, velocity_y=%s WHERE id=%s",
            chunk,
        )
        conn.commit()
        pct = min(i + BATCH, len(updates))
        print(f"  {pct}/{len(updates)} 已写入")

    print("完成！")
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
