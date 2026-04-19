"""
从 nuScenes 解析自车位姿并导入 ground_truth_ego 表。

每条记录对应一个关键帧（sample），包含：
  - 自车全局位置 pos_x/y/z
  - 自车姿态四元数 rot_w/x/y/z
  - 从四元数解算的车头朝向 yaw_rad
  - 前后帧差分推算的速度 vx_mps / vy_mps / speed_mps
    （帧间间隔约 0.5s；首/尾帧用单向差分，中间帧用双向中心差分）

用法：
    python3 tools/import_ego_to_db.py \
        --dataroot /root/autodl-tmp/NuScenes/ \
        --version  v1.0-trainval \
        --host 127.0.0.1 --port 13306 \
        --user remote1 --password 你的密码
"""
import argparse
import math
import time
import pymysql
import pymysql.cursors


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def quat_to_yaw(w, x, y, z):
    """四元数 [w, x, y, z] → 车头朝向 yaw（rad），绕 Z 轴旋转角。"""
    return math.atan2(2.0 * (w * z + x * y),
                      1.0 - 2.0 * (y * y + z * z))


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
    p.add_argument('--dataroot', default='/root/autodl-tmp/NuScenes/')
    p.add_argument('--version',  default='v1.0-trainval')
    p.add_argument('--host',     default='127.0.0.1')
    p.add_argument('--port',     type=int, default=13306)
    p.add_argument('--user',     default='remote1')
    p.add_argument('--password', default='')
    p.add_argument('--database', default='safetyai_sparsebev')
    p.add_argument('--batch_size', type=int, default=500)
    return p.parse_args()


# ---------------------------------------------------------------------------
# 核心：解析 nuScenes 自车信息
# ---------------------------------------------------------------------------
def extract_ego_records(nusc):
    """
    返回 dict: sample_token -> ego_record
    ego_record 包含位置、旋转、yaw、推算速度。
    速度用中心差分（前后帧），首/尾帧用单向差分，孤立帧（无前后）速度置 None。
    """
    print('  建立 sample 链 ...')
    # 按 scene 遍历，维护有序的 prev/next 关系
    records = {}   # sample_token -> dict

    for scene in nusc.scene:
        # 收集本 scene 所有 sample（有序）
        chain = []
        token = scene['first_sample_token']
        while token:
            s = nusc.get('sample', token)
            chain.append(s)
            token = s['next']

        for i, s in enumerate(chain):
            # 获取 LIDAR_TOP（最优先）或 CAM_FRONT 的 ego_pose
            lidar_key = 'LIDAR_TOP'
            sensor_key = lidar_key if lidar_key in s['data'] else list(s['data'].keys())[0]
            sd = nusc.get('sample_data', s['data'][sensor_key])
            ep = nusc.get('ego_pose', sd['ego_pose_token'])

            t  = ep['translation']   # [x, y, z]
            r  = ep['rotation']      # [w, x, y, z]
            ts = ep['timestamp']     # us

            yaw = quat_to_yaw(r[0], r[1], r[2], r[3])

            # 速度推算
            vx = vy = speed = None

            def get_ep(s_other):
                sd_o = nusc.get('sample_data', s_other['data'][sensor_key])
                return nusc.get('ego_pose', sd_o['ego_pose_token'])

            if i == 0 and i + 1 < len(chain):
                # 首帧：前向差分
                ep_next = get_ep(chain[i + 1])
                dt = (ep_next['timestamp'] - ts) / 1e6
                if dt > 0:
                    vx = (ep_next['translation'][0] - t[0]) / dt
                    vy = (ep_next['translation'][1] - t[1]) / dt
            elif i == len(chain) - 1 and i > 0:
                # 尾帧：后向差分
                ep_prev = get_ep(chain[i - 1])
                dt = (ts - ep_prev['timestamp']) / 1e6
                if dt > 0:
                    vx = (t[0] - ep_prev['translation'][0]) / dt
                    vy = (t[1] - ep_prev['translation'][1]) / dt
            elif i > 0 and i + 1 < len(chain):
                # 中间帧：中心差分（更稳定）
                ep_prev = get_ep(chain[i - 1])
                ep_next = get_ep(chain[i + 1])
                dt = (ep_next['timestamp'] - ep_prev['timestamp']) / 1e6
                if dt > 0:
                    vx = (ep_next['translation'][0] - ep_prev['translation'][0]) / dt
                    vy = (ep_next['translation'][1] - ep_prev['translation'][1]) / dt

            if vx is not None and vy is not None:
                speed = math.sqrt(vx ** 2 + vy ** 2)

            records[s['token']] = dict(
                pos_x=t[0], pos_y=t[1], pos_z=t[2],
                rot_w=r[0], rot_x=r[1], rot_y=r[2], rot_z=r[3],
                yaw_rad=yaw,
                vx_mps=round(vx, 4) if vx is not None else None,
                vy_mps=round(vy, 4) if vy is not None else None,
                speed_mps=round(speed, 4) if speed is not None else None,
                ego_timestamp_us=ts,
            )

    return records


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    print(f'[1/3] 加载 nuScenes {args.version} ...')
    t0 = time.time()
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    print(f'  加载完成，耗时 {time.time()-t0:.1f}s')

    print(f'[2/3] 解析自车位姿与速度 ...')
    t1 = time.time()
    ego_records = extract_ego_records(nusc)
    print(f'  解析完成: {len(ego_records)} 条，耗时 {time.time()-t1:.1f}s')

    # 速度统计
    speeds = [v['speed_mps'] for v in ego_records.values() if v['speed_mps'] is not None]
    if speeds:
        import statistics
        print(f'  自车速度统计: mean={statistics.mean(speeds):.2f} m/s  '
              f'max={max(speeds):.2f} m/s  '
              f'静止(<0.5m/s): {sum(1 for s in speeds if s < 0.5)} 帧')

    print(f'[3/3] 写入数据库 ...')
    conn = connect(args)
    cur  = conn.cursor()

    # 检查是否已有数据
    cur.execute('SELECT COUNT(*) AS n FROM ground_truth_ego')
    existing = cur.fetchone()['n']
    if existing > 0:
        print(f'  已存在 {existing} 条，将跳过已有记录（INSERT IGNORE）')

    # 建立 sample_token -> gt_sample_id 映射
    cur.execute('SELECT id, sample_token FROM ground_truth_sample')
    token_to_id = {r['sample_token']: r['id'] for r in cur.fetchall()}

    # 组装插入行
    rows = []
    missing = 0
    for token, rec in ego_records.items():
        gt_id = token_to_id.get(token)
        if gt_id is None:
            missing += 1
            continue
        rows.append((
            gt_id,
            rec['pos_x'], rec['pos_y'], rec['pos_z'],
            rec['rot_w'], rec['rot_x'], rec['rot_y'], rec['rot_z'],
            rec['yaw_rad'],
            rec['speed_mps'], rec['vx_mps'], rec['vy_mps'],
            rec['ego_timestamp_us'],
        ))

    print(f'  待写入 {len(rows)} 条（{missing} 条 sample 不在 ground_truth_sample 中，跳过）')

    t2 = time.time()
    total = len(rows)
    for i in range(0, total, args.batch_size):
        chunk = rows[i:i + args.batch_size]
        cur.executemany(
            '''INSERT IGNORE INTO ground_truth_ego
               (gt_sample_id,
                pos_x, pos_y, pos_z,
                rot_w, rot_x, rot_y, rot_z,
                yaw_rad,
                speed_mps, vx_mps, vy_mps,
                ego_timestamp_us)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)''',
            chunk
        )
        conn.commit()
        done = min(i + args.batch_size, total)
        if done % 5000 == 0 or done == total:
            print(f'  写入进度: {done}/{total}  耗时 {time.time()-t2:.1f}s')

    # 验证
    cur.execute('SELECT COUNT(*) AS n FROM ground_truth_ego')
    n_final = cur.fetchone()['n']

    # 速度分布快速查询
    cur.execute('''
        SELECT
            ROUND(AVG(speed_mps), 2)  AS avg_speed,
            ROUND(MAX(speed_mps), 2)  AS max_speed,
            SUM(speed_mps < 0.5)      AS stationary_frames,
            SUM(speed_mps >= 0.5 AND speed_mps < 5)  AS slow_frames,
            SUM(speed_mps >= 5  AND speed_mps < 15)  AS medium_frames,
            SUM(speed_mps >= 15)                      AS fast_frames
        FROM ground_truth_ego
        WHERE speed_mps IS NOT NULL
    ''')
    stat = cur.fetchone()

    print(f'\n=== 导入完成 ===')
    print(f'  ground_truth_ego 总行数 : {n_final}')
    print(f'  自车速度统计:')
    print(f'    均值         : {stat["avg_speed"]} m/s')
    print(f'    最大值       : {stat["max_speed"]} m/s')
    print(f'    静止 (<0.5)  : {stat["stationary_frames"]} 帧')
    print(f'    低速 (0.5~5) : {stat["slow_frames"]} 帧')
    print(f'    中速 (5~15)  : {stat["medium_frames"]} 帧')
    print(f'    高速 (≥15)   : {stat["fast_frames"]} 帧')

    cur.close()
    conn.close()


if __name__ == '__main__':
    main()
