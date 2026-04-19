"""
导入相机标定参数和图片路径到数据库。

执行内容：
  Step 1: nuscenes_calibrated_sensor  —— 所有相机标定（内参+外参）
  Step 2: nuscenes_sample_camera      —— 每帧 6 路图片路径（约 20 万行）

用法：
    python3 tools/import_camera_to_db.py \
        --dataroot /root/autodl-tmp/NuScenes/ \
        --version  v1.0-trainval \
        --host 127.0.0.1 --port 13306 \
        --user remote1 --password 你的密码
"""
import argparse, json, time
import pymysql, pymysql.cursors


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataroot',   default='/root/autodl-tmp/NuScenes/')
    p.add_argument('--version',    default='v1.0-trainval')
    p.add_argument('--host',       default='127.0.0.1')
    p.add_argument('--port',       type=int, default=13306)
    p.add_argument('--user',       default='remote1')
    p.add_argument('--password',   default='')
    p.add_argument('--database',   default='safetyai_sparsebev')
    p.add_argument('--batch_size', type=int, default=2000)
    return p.parse_args()


def connect(args):
    return pymysql.connect(
        host=args.host, port=args.port,
        user=args.user, password=args.password,
        database=args.database,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Step 1: calibrated_sensor
# ---------------------------------------------------------------------------
def import_calibrated_sensors(dataroot, version, conn):
    print('\n[Step 1] 导入 nuscenes_calibrated_sensor ...')
    cur = conn.cursor()

    cur.execute('SELECT COUNT(*) AS n FROM nuscenes_calibrated_sensor')
    if cur.fetchone()['n'] > 0:
        print('  已存在，跳过')
        cur.close()
        return

    cs_list   = load_json(f'{dataroot}/{version}/calibrated_sensor.json')
    sensor_list = load_json(f'{dataroot}/{version}/sensor.json')
    # sensor_token -> channel (CAM_FRONT etc.)
    sensor_map = {s['token']: s['channel'] for s in sensor_list}

    cam_rows = []
    for cs in cs_list:
        channel = sensor_map.get(cs['sensor_token'], '')
        if not channel.startswith('CAM'):
            continue   # 只保留相机，跳过 LIDAR/RADAR
        cam_rows.append((
            cs['token'],
            cs['sensor_token'],
            channel,
            json.dumps(cs['translation']),
            json.dumps(cs['rotation']),
            json.dumps(cs['camera_intrinsic']) if cs['camera_intrinsic'] else None,
        ))

    cur.executemany(
        '''INSERT IGNORE INTO nuscenes_calibrated_sensor
           (cs_token, sensor_token, channel, translation, rotation, camera_intrinsic)
           VALUES (%s,%s,%s,%s,%s,%s)''',
        cam_rows
    )
    conn.commit()
    print(f'  写入 {len(cam_rows)} 条相机标定')

    # 统计
    cur.execute('SELECT channel, COUNT(*) AS n FROM nuscenes_calibrated_sensor GROUP BY channel ORDER BY channel')
    for r in cur.fetchall():
        print(f'    {r["channel"]:25s}: {r["n"]} 条标定')
    cur.close()


# ---------------------------------------------------------------------------
# Step 2: sample_camera（图片路径）
# ---------------------------------------------------------------------------
def import_sample_cameras(dataroot, version, conn, batch_size):
    print('\n[Step 2] 导入 nuscenes_sample_camera ...')
    cur = conn.cursor()

    cur.execute('SELECT COUNT(*) AS n FROM nuscenes_sample_camera')
    if cur.fetchone()['n'] > 0:
        print('  已存在，跳过')
        cur.close()
        return

    # 读取 sample_data，只保留关键帧相机
    sd_all = load_json(f'{dataroot}/{version}/sample_data.json')
    cam_sds = [sd for sd in sd_all
               if sd['is_key_frame'] and 'CAM' in sd['filename']]
    print(f'  关键帧相机 sample_data: {len(cam_sds)} 条')

    # sample_token → gt_sample_id
    cur.execute('SELECT id, sample_token FROM ground_truth_sample')
    token_map = {r['sample_token']: r['id'] for r in cur.fetchall()}

    # sensor_token → channel（via calibrated_sensor + sensor）
    cur.execute('SELECT cs_token, channel FROM nuscenes_calibrated_sensor')
    cs_channel = {r['cs_token']: r['channel'] for r in cur.fetchall()}

    rows = []
    skipped = 0
    t0 = time.time()

    for sd in cam_sds:
        gt_id = token_map.get(sd['sample_token'])
        if gt_id is None:
            skipped += 1
            continue
        channel = cs_channel.get(sd['calibrated_sensor_token'], '')
        rows.append((
            gt_id,
            channel,
            sd['filename'],
            sd['token'],
            sd['calibrated_sensor_token'],
            sd['timestamp'],
            sd.get('width'),
            sd.get('height'),
        ))

    print(f'  待写入: {len(rows)} 条，跳过（无 GT 对应）: {skipped} 条')

    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i+batch_size]
        cur.executemany(
            '''INSERT IGNORE INTO nuscenes_sample_camera
               (gt_sample_id, channel, filename, sd_token, cs_token,
                timestamp_us, width, height)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s)''',
            chunk
        )
        conn.commit()
        done = min(i + batch_size, len(rows))
        if done % 50000 == 0 or done == len(rows):
            print(f'  写入: {done}/{len(rows)}  {time.time()-t0:.1f}s')

    cur.close()


# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------
def verify(conn):
    print('\n[验证] 关联查询示例（sample token: 1e6f6e90ec654248926d03aaa42a120d）...')
    cur = conn.cursor()
    cur.execute('''
        SELECT sc.channel, sc.filename,
               cs.translation, cs.camera_intrinsic
        FROM ground_truth_sample gs
        JOIN nuscenes_sample_camera sc ON sc.gt_sample_id = gs.id
        JOIN nuscenes_calibrated_sensor cs ON cs.cs_token = sc.cs_token
        WHERE gs.sample_token = '1e6f6e90ec654248926d03aaa42a120d'
        ORDER BY sc.channel
    ''')
    rows = cur.fetchall()
    for r in rows:
        intr = json.loads(r['camera_intrinsic']) if r['camera_intrinsic'] else None
        fx   = intr[0][0] if intr else 'N/A'
        print(f"  {r['channel']:25s}  fx={fx:.1f}  {r['filename'].split('/')[-1]}")

    # 总量汇总
    cur.execute('SELECT COUNT(*) AS n FROM nuscenes_sample_camera')
    n_sc = cur.fetchone()['n']
    cur.execute('SELECT COUNT(*) AS n FROM nuscenes_calibrated_sensor')
    n_cs = cur.fetchone()['n']
    cur.execute('SELECT COUNT(DISTINCT gt_sample_id)*6 AS expected, COUNT(*) AS actual FROM nuscenes_sample_camera')
    r = cur.fetchone()
    print(f'\n=== 完成 ===')
    print(f'  nuscenes_calibrated_sensor : {n_cs} 条')
    print(f'  nuscenes_sample_camera     : {n_sc} 条  (期望 {r["expected"]})')

    cur.close()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    conn = connect(args)

    import_calibrated_sensors(args.dataroot, args.version, conn)
    import_sample_cameras(args.dataroot, args.version, conn, args.batch_size)
    verify(conn)

    conn.close()


if __name__ == '__main__':
    main()
