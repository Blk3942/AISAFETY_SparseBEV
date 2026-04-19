"""
导入 nuScenes scene 元数据并回填 ground_truth_box.visibility_level。

执行内容：
  1. nuscenes_scene 表：850 条场景（含描述、地点、日期）
  2. ground_truth_box.visibility_level：回填 1~4 级可见度

用法：
    python3 tools/import_scene_visibility.py \
        --dataroot /root/autodl-tmp/NuScenes/ \
        --version  v1.0-trainval \
        --host 127.0.0.1 --port 13306 \
        --user remote1 --password 你的密码
"""
import argparse, math, time
import pymysql, pymysql.cursors


def connect(args):
    return pymysql.connect(
        host=args.host, port=args.port,
        user=args.user, password=args.password,
        database=args.database,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataroot',   default='/root/autodl-tmp/NuScenes/')
    p.add_argument('--version',    default='v1.0-trainval')
    p.add_argument('--host',       default='127.0.0.1')
    p.add_argument('--port',       type=int, default=13306)
    p.add_argument('--user',       default='remote1')
    p.add_argument('--password',   default='')
    p.add_argument('--database',   default='safetyai_sparsebev')
    p.add_argument('--batch_size', type=int, default=5000)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 1: nuscenes_scene
# ---------------------------------------------------------------------------
def import_scenes(nusc, conn, dataset_id):
    print('\n[Step 1] 导入 nuscenes_scene ...')
    cur = conn.cursor()

    # 检查是否已存在
    cur.execute('SELECT COUNT(*) AS n FROM nuscenes_scene WHERE dataset_id=%s', (dataset_id,))
    if cur.fetchone()['n'] > 0:
        print('  已存在，跳过')
        cur.close()
        return

    # log_token -> location, date
    log_map = {l['token']: l for l in nusc.log}

    rows = []
    for sc in nusc.scene:
        log = log_map.get(sc['log_token'], {})
        rows.append((
            sc['token'],
            dataset_id,
            sc['name'],
            sc['description'],
            sc['nbr_samples'],
            log.get('location', None),
            log.get('date_captured', None),
            sc['log_token'],
        ))

    cur.executemany(
        '''INSERT IGNORE INTO nuscenes_scene
           (scene_token, dataset_id, name, description, nbr_samples,
            log_location, log_date, log_token)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s)''',
        rows
    )
    conn.commit()
    print(f'  写入 {len(rows)} 条 scene')

    # 统计地点分布
    cur.execute('''
        SELECT log_location, COUNT(*) AS n
        FROM nuscenes_scene WHERE dataset_id=%s
        GROUP BY log_location ORDER BY n DESC
    ''', (dataset_id,))
    for r in cur.fetchall():
        print(f'    {r["log_location"]:35s}: {r["n"]} 个场景')

    cur.close()


# ---------------------------------------------------------------------------
# Step 2: 回填 visibility_level
# ---------------------------------------------------------------------------
def fill_visibility(nusc, conn, batch_size):
    print('\n[Step 2] 回填 ground_truth_box.visibility_level ...')
    cur = conn.cursor()

    # 检查未填充数量
    cur.execute('SELECT COUNT(*) AS n FROM ground_truth_box WHERE visibility_level IS NULL')
    n_null = cur.fetchone()['n']
    if n_null == 0:
        print('  已全部填充，跳过')
        cur.close()
        return
    print(f'  待填充: {n_null} 条')

    # nuScenes visibility: token -> level (int 1~4，level 即 token 值本身)
    # visibility_token 取值为 "1","2","3","4"（直接对应等级）
    vis_map = {v['token']: int(v['token']) for v in nusc.visibility
               if v['token'].isdigit()}
    print(f'  visibility 等级映射: {vis_map}')

    # annotation_token -> visibility_token (来自 sample_annotation)
    print('  建立 annotation → visibility 映射 ...')
    ann_vis = {a['token']: vis_map.get(a['visibility_token'])
               for a in nusc.sample_annotation}

    # 读取所有 NULL visibility_level 的 box
    print('  读取待更新 box ...')
    cur.execute('''
        SELECT id, annotation_token
        FROM ground_truth_box
        WHERE visibility_level IS NULL
          AND annotation_token IS NOT NULL
    ''')
    rows = cur.fetchall()
    print(f'  有 annotation_token 的待更新: {len(rows)} 条')

    # 计算更新列表
    updates = []
    unknown = 0
    for r in rows:
        level = ann_vis.get(r['annotation_token'])
        if level is None:
            unknown += 1
            continue
        updates.append((r['id'], level))

    print(f'  可更新: {len(updates)} 条，无对应 token: {unknown} 条')

    # 临时表 + UPDATE JOIN（与 ego_dist 同样策略）
    t0 = time.time()
    cur.execute('''
        CREATE TEMPORARY TABLE _vis_tmp (
            box_id BIGINT PRIMARY KEY,
            vis    TINYINT
        ) ENGINE=InnoDB
    ''')
    conn.commit()

    for i in range(0, len(updates), batch_size):
        chunk = updates[i:i+batch_size]
        cur.executemany(
            'INSERT INTO _vis_tmp (box_id, vis) VALUES (%s,%s)',
            chunk
        )
        conn.commit()
        done = min(i + batch_size, len(updates))
        if done % 200000 == 0 or done == len(updates):
            print(f'  临时表写入: {done}/{len(updates)}  {time.time()-t0:.1f}s')

    print('  执行 UPDATE JOIN ...')
    cur.execute('''
        UPDATE ground_truth_box b
        JOIN _vis_tmp t ON t.box_id = b.id
        SET b.visibility_level = t.vis
    ''')
    conn.commit()
    affected = cur.rowcount
    print(f'  UPDATE 影响 {affected} 行，耗时 {time.time()-t0:.1f}s')

    cur.execute('DROP TEMPORARY TABLE IF EXISTS _vis_tmp')
    conn.commit()

    # 可见度分布统计
    cur.execute('''
        SELECT visibility_level,
               COUNT(*) AS n,
               ROUND(COUNT(*)*100.0/(SELECT COUNT(*) FROM ground_truth_box), 1) AS pct
        FROM ground_truth_box
        GROUP BY visibility_level
        ORDER BY visibility_level
    ''')
    print('\n  可见度分布:')
    level_desc = {1: '<40%', 2: '40~60%', 3: '60~80%', 4: '80~100%', None: 'NULL'}
    for r in cur.fetchall():
        desc = level_desc.get(r['visibility_level'], '?')
        print(f'    level {r["visibility_level"]} ({desc:8s}): '
              f'{r["n"]:>8,} 条  {r["pct"]}%')
    cur.close()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    print(f'加载 nuScenes {args.version} ...')
    t0 = time.time()
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    print(f'  完成，耗时 {time.time()-t0:.1f}s')

    conn = connect(args)
    cur  = conn.cursor()

    # 查 dataset_id
    cur.execute('SELECT dataset_id FROM nuscenes_dataset WHERE nuscenes_version=%s',
                (args.version,))
    row = cur.fetchone()
    if row is None:
        raise RuntimeError(f'nuscenes_dataset 中找不到 {args.version}，请先运行 import_run_to_db.py')
    dataset_id = row['dataset_id']
    print(f'dataset_id = {dataset_id}')
    cur.close()

    import_scenes(nusc, conn, dataset_id)
    fill_visibility(nusc, conn, args.batch_size)

    # 最终汇总
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) AS n FROM nuscenes_scene WHERE dataset_id=%s', (dataset_id,))
    n_scene = cur.fetchone()['n']
    cur.execute('SELECT COUNT(*) AS n FROM ground_truth_box WHERE visibility_level IS NOT NULL')
    n_vis = cur.fetchone()['n']
    cur.execute('SELECT COUNT(*) AS n FROM ground_truth_box')
    n_total = cur.fetchone()['n']
    print(f'\n=== 完成 ===')
    print(f'  nuscenes_scene        : {n_scene} 条')
    print(f'  visibility_level 已填 : {n_vis} / {n_total}')

    # 关联验证：scene → sample → box
    cur.execute('''
        SELECT sc.log_location, COUNT(DISTINCT s.id) AS samples, COUNT(b.id) AS boxes
        FROM nuscenes_scene sc
        JOIN ground_truth_sample s  ON s.scene_token = sc.scene_token
        JOIN ground_truth_box b     ON b.gt_sample_id = s.id
        GROUP BY sc.log_location
        ORDER BY samples DESC
    ''')
    print('\n  地点 → sample → box 关联验证:')
    for r in cur.fetchall():
        print(f'    {r["log_location"]:35s}: {r["samples"]:5} samples  {r["boxes"]:>8,} boxes')

    cur.close()
    conn.close()


if __name__ == '__main__':
    main()
