"""
快速回填 ground_truth_box.ego_dist：
  用临时表 + 单条 UPDATE JOIN，比逐行 UPDATE 快 100x+。
"""
import argparse, math, json
import pymysql, pymysql.cursors

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataroot', default='/root/autodl-tmp/NuScenes/')
    p.add_argument('--version',  default='v1.0-trainval')
    p.add_argument('--host',     default='127.0.0.1')
    p.add_argument('--port',     type=int, default=13306)
    p.add_argument('--user',     default='remote1')
    p.add_argument('--password', default='')
    p.add_argument('--database', default='safetyai_sparsebev')
    return p.parse_args()

def main():
    args = parse_args()
    print('读取 nuScenes ego_pose ...')
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    # sample_token -> ego_xy
    ego_map = {}
    for sample in nusc.sample:
        sd_token = sample['data'].get('LIDAR_TOP') or list(sample['data'].values())[0]
        sd = nusc.get('sample_data', sd_token)
        ep = nusc.get('ego_pose', sd['ego_pose_token'])
        ego_map[sample['token']] = ep['translation'][:2]
    print(f'  ego_map: {len(ego_map)} 条')

    conn = pymysql.connect(
        host=args.host, port=args.port,
        user=args.user, password=args.password,
        database=args.database, charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor, autocommit=False,
    )
    cur = conn.cursor()

    # 读取所有 NULL ego_dist 的 box
    print('读取待更新 box ...')
    cur.execute('''
        SELECT b.id, s.sample_token, b.translation_x, b.translation_y
        FROM ground_truth_box b
        JOIN ground_truth_sample s ON s.id = b.gt_sample_id
        WHERE b.ego_dist IS NULL
    ''')
    rows = cur.fetchall()
    print(f'  待更新 {len(rows)} 条')

    # 计算所有 ego_dist
    updates = []
    for r in rows:
        ego_xy = ego_map.get(r['sample_token'])
        if ego_xy is None: continue
        dx = r['translation_x'] - ego_xy[0]
        dy = r['translation_y'] - ego_xy[1]
        dist = round(math.sqrt(dx*dx + dy*dy), 4)
        updates.append((r['id'], dist))

    # 建临时表 + UPDATE JOIN（单次 SQL，速度极快）
    print('建临时表 + UPDATE JOIN ...')
    cur.execute('''
        CREATE TEMPORARY TABLE _ego_dist_tmp (
            box_id BIGINT PRIMARY KEY,
            dist   DOUBLE
        ) ENGINE=InnoDB
    ''')
    conn.commit()

    BATCH = 10000
    for i in range(0, len(updates), BATCH):
        chunk = updates[i:i+BATCH]
        cur.executemany('INSERT INTO _ego_dist_tmp (box_id, dist) VALUES (%s,%s)', chunk)
        conn.commit()
        pct = min(i+BATCH, len(updates))
        if pct % 100000 == 0 or pct == len(updates):
            print(f'  临时表: {pct}/{len(updates)}')

    print('执行 UPDATE JOIN ...')
    cur.execute('''
        UPDATE ground_truth_box b
        JOIN _ego_dist_tmp t ON t.box_id = b.id
        SET b.ego_dist = t.dist
    ''')
    conn.commit()
    affected = cur.rowcount
    print(f'  UPDATE 影响 {affected} 行')

    cur.execute('DROP TEMPORARY TABLE IF EXISTS _ego_dist_tmp')
    conn.commit()

    # 验证
    cur.execute('SELECT COUNT(*) AS n FROM ground_truth_box WHERE ego_dist IS NOT NULL')
    n_filled = cur.fetchone()['n']
    cur.execute('SELECT COUNT(*) AS n FROM ground_truth_box WHERE ego_dist IS NULL')
    n_null = cur.fetchone()['n']
    print(f'\nego_dist 已填: {n_filled}，仍为 NULL: {n_null}')
    cur.close(); conn.close()

if __name__ == '__main__':
    main()
