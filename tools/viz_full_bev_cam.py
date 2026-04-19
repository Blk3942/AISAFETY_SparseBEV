"""
完整 BEV + 相机投影 可视化（完全基于数据库）

布局：
  上半部分 —— GT Ground Truth   : 6路相机投影 + BEV
  下半部分 —— Prediction         : 6路相机投影 + BEV
  BEV 中标注 safety-critical 判定圆（默认 30m）

用法：
    python3 tools/viz_full_bev_cam.py \
        --sample_id 28061 \
        --dataroot /root/autodl-tmp/NuScenes \
        --score_thr 0.3 \
        --safety_range 30 \
        --out /tmp/viz_full_28061.png
"""
import argparse, json, math, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
import pymysql, pymysql.cursors


# ─────────────────────────── 全局配置 ────────────────────────────
CLASS_COLOR = {
    'ego':                              '#FFFFFF',
    'vehicle.car':                      '#4FC3F7',
    'vehicle.truck':                    '#1E88E5',
    'vehicle.bus.rigid':                '#0D47A1',
    'vehicle.bus.bendy':                '#0D47A1',
    'vehicle.trailer':                  '#5C6BC0',
    'vehicle.construction':             '#FF8F00',
    'vehicle.motorcycle':               '#AB47BC',
    'vehicle.bicycle':                  '#CE93D8',
    'human.pedestrian.adult':           '#EF5350',
    'human.pedestrian.child':           '#FF8A80',
    'movable_object.barrier':           '#78909C',
    'movable_object.trafficcone':       '#FF7043',
    'movable_object.pushable_pullable': '#A1887F',
    'movable_object.debris':            '#BCAAA4',
    'static_object.bicycle_rack':       '#90A4AE',
}
DEFAULT_COLOR = '#B0BEC5'

CAM_ROWS = [
    ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'],
    ['CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT'],
]
LEGEND_CLS = [
    ('ego',                        'ego'),
    ('vehicle.car',                'car'),
    ('vehicle.truck',              'truck'),
    ('vehicle.trailer',            'trailer'),
    ('vehicle.construction',       'construction'),
    ('vehicle.motorcycle',         'motorcycle'),
    ('vehicle.bicycle',            'bicycle'),
    ('human.pedestrian.adult',     'pedestrian'),
    ('movable_object.barrier',     'barrier'),
    ('movable_object.trafficcone', 'trafficcone'),
]
# 8 corners: 0-3 top face (z=+h/2), 4-7 bottom face (z=-h/2)
# front of box = +x direction → corners 0,1,4,5
BOX_EDGES       = [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4), (0,4),(1,5),(2,6),(3,7)]
FRONT_EDGE_SET  = frozenset([(0,1),(1,0),(0,4),(4,0),(1,5),(5,1),(4,5),(5,4)])
CLASS_ABBREV = {
    'vehicle.car':'car', 'vehicle.truck':'truck',
    'vehicle.bus.rigid':'bus', 'vehicle.bus.bendy':'bus',
    'vehicle.trailer':'trlr', 'vehicle.construction':'cons',
    'vehicle.motorcycle':'moto', 'vehicle.bicycle':'bike',
    'human.pedestrian.adult':'ped', 'human.pedestrian.child':'ped',
    'movable_object.barrier':'barr', 'movable_object.trafficcone':'cone',
}


def get_color(cls): return CLASS_COLOR.get(cls, DEFAULT_COLOR)


# ─────────────────────────── 数学工具 ────────────────────────────
def quat_to_rot(w, x, y, z):
    """四元数 [w,x,y,z] → 3×3 旋转矩阵"""
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)]
    ], dtype=np.float64)


def box_corners_global(tx, ty, tz, w, l, h, rw, rx, ry, rz):
    """计算 3D box 在全局坐标系的 8 个角点"""
    R = quat_to_rot(rw, rx, ry, rz)
    local = np.array([
        [ l/2,  w/2,  h/2], [ l/2, -w/2,  h/2],
        [-l/2, -w/2,  h/2], [-l/2,  w/2,  h/2],
        [ l/2,  w/2, -h/2], [ l/2, -w/2, -h/2],
        [-l/2, -w/2, -h/2], [-l/2,  w/2, -h/2],
    ], dtype=np.float64)
    return (R @ local.T).T + np.array([tx, ty, tz])


def project_to_cam(pts_global, ego, cam_t, cam_R_mat, K):
    """
    pts_global : (N, 3) 全局坐标系点
    ego        : dict {pos_x/y/z, rot_w/x/y/z}
    cam_t      : (3,) 相机在车体系的平移
    cam_R_mat  : (3,3) 相机在车体系的旋转矩阵
    K          : (3,3) 内参矩阵
    Returns: u (N,), v (N,), Z (N,)  —— Z 是相机坐标系深度
    """
    t_ego = np.array([ego['pos_x'], ego['pos_y'], ego['pos_z']])
    R_ego = quat_to_rot(ego['rot_w'], ego['rot_x'], ego['rot_y'], ego['rot_z'])

    # global → ego/vehicle frame
    p_ego = (R_ego.T @ (pts_global - t_ego).T).T
    # vehicle → camera frame
    p_cam = (cam_R_mat.T @ (p_ego - cam_t).T).T

    Z = p_cam[:, 2]
    Z_safe = np.where(np.abs(Z) > 1e-4, Z, 1e-4)
    u = K[0, 0] * p_cam[:, 0] / Z_safe + K[0, 2]
    v = K[1, 1] * p_cam[:, 1] / Z_safe + K[1, 2]
    return u, v, Z


# ─────────────────────────── 数据库读取 ────────────────────────────
def fetch_data(conn, sample_id, score_thr):
    cur = conn.cursor()

    cur.execute('''SELECT pos_x, pos_y, pos_z,
                          rot_w, rot_x, rot_y, rot_z,
                          yaw_rad, speed_mps
                   FROM ground_truth_ego WHERE gt_sample_id = %s''', (sample_id,))
    ego = cur.fetchone()

    cur.execute('''SELECT translation_x, translation_y, translation_z,
                          size_wlh_0, size_wlh_1, size_wlh_2,
                          rotation_w, rotation_x, rotation_y, rotation_z,
                          class_name, visibility_level, ego_dist
                   FROM ground_truth_box WHERE gt_sample_id = %s''', (sample_id,))
    gt_boxes = cur.fetchall()

    cur.execute('''SELECT pb.translation_x, pb.translation_y, pb.translation_z,
                          pb.size_wlh_0, pb.size_wlh_1, pb.size_wlh_2,
                          pb.rotation_w, pb.rotation_x, pb.rotation_y, pb.rotation_z,
                          pb.class_name, pb.score
                   FROM prediction_box pb
                   JOIN prediction_sample ps ON ps.id = pb.pred_sample_id
                   WHERE ps.gt_sample_id = %s AND pb.score >= %s''', (sample_id, score_thr))
    pred_boxes = cur.fetchall()

    cur.execute('''SELECT sc.channel, sc.filename,
                          cs.translation AS cam_t,
                          cs.rotation    AS cam_r,
                          cs.camera_intrinsic AS cam_k,
                          sc.width, sc.height
                   FROM nuscenes_sample_camera sc
                   JOIN nuscenes_calibrated_sensor cs ON cs.cs_token = sc.cs_token
                   WHERE sc.gt_sample_id = %s''', (sample_id,))
    cams = {}
    for r in cur.fetchall():
        cam_r_list = json.loads(r['cam_r'])   # [w,x,y,z]
        cams[r['channel']] = {
            'filename': r['filename'],
            't': np.array(json.loads(r['cam_t'])),
            'R': quat_to_rot(*cam_r_list),
            'K': np.array(json.loads(r['cam_k'])),
            'W': r['width']  or 1600,
            'H': r['height'] or 900,
        }

    cur.execute('''SELECT gs.sample_token, ns.name, ns.log_location, ns.log_date
                   FROM ground_truth_sample gs
                   LEFT JOIN nuscenes_scene ns ON ns.scene_token = gs.scene_token
                   WHERE gs.id = %s''', (sample_id,))
    meta = cur.fetchone()
    cur.close()
    return ego, gt_boxes, pred_boxes, cams, meta


# ─────────────────────────── 相机画图 ────────────────────────────
def draw_camera_view(ax, cam, boxes, ego, channel_label):
    img = mpimg.imread(cam['filepath'])
    W, H = cam['W'], cam['H']

    ax.imshow(img, extent=[0, W, H, 0], aspect='auto')
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_title(channel_label, color='#AAAACC', fontsize=7, pad=2)
    ax.axis('off')

    for b in boxes:
        corners = box_corners_global(
            b['translation_x'], b['translation_y'], b['translation_z'],
            b['size_wlh_0'], b['size_wlh_1'], b['size_wlh_2'],
            b['rotation_w'], b['rotation_x'], b['rotation_y'], b['rotation_z']
        )
        u, v, Z = project_to_cam(corners, ego, cam['t'], cam['R'], cam['K'])

        # 需要至少 4 个角点在相机前方且粗略在图像内（宽松范围）
        in_front = Z > 0.2
        in_image = (u > -300) & (u < W + 300) & (v > -300) & (v < H + 300)
        visible = in_front & in_image
        if visible.sum() < 3:
            continue

        col = get_color(b['class_name'])
        score_str = f'({b["score"]:.2f})' if 'score' in b else ''
        abbrev = CLASS_ABBREV.get(b['class_name'], b['class_name'].split('.')[-1][:5])

        for i, j in BOX_EDGES:
            if Z[i] > 0.2 and Z[j] > 0.2:
                is_front_edge = (i, j) in FRONT_EDGE_SET
                ax.plot([u[i], u[j]], [v[i], v[j]],
                        color=col,
                        linewidth=2.0 if is_front_edge else 1.0,
                        alpha=0.92,
                        solid_capstyle='round',
                        clip_on=True)

        # 标签：画在可见角点中心偏上方
        vu_vis = u[visible]
        vv_vis = v[visible]
        cx_px = vu_vis.mean()
        cy_px = vv_vis.min() - 4
        if 0 < cx_px < W and -50 < cy_px < H + 50:
            ax.text(cx_px, cy_px, f'{abbrev}{score_str}',
                    color=col, fontsize=5.5, ha='center', va='bottom',
                    fontweight='bold',
                    bbox=dict(facecolor='#00000060', edgecolor='none', pad=0.8))


# ─────────────────────────── BEV 画图 ────────────────────────────
def draw_bev(ax, boxes, ego, title, bev_range, safety_r, is_pred=False):
    ax.set_facecolor('#07070F')
    ax.set_xlim(-bev_range, bev_range)
    ax.set_ylim(-bev_range, bev_range)
    ax.set_aspect('equal')
    ax.set_title(title, color='#CCCCDD', fontsize=9, pad=6)

    # ── 距离圈 ──
    for r in range(10, bev_range + 1, 10):
        is_safety = (r == safety_r)
        c = plt.Circle((0, 0), r,
                        color='#FF3333' if is_safety else '#1E1E34',
                        fill=False,
                        linewidth=1.8 if is_safety else 0.7,
                        linestyle='-' if is_safety else '--',
                        zorder=2)
        ax.add_patch(c)
        label_col = '#FF5555' if is_safety else '#383860'
        ax.text(r * 0.707, r * 0.707, f'{r}m',
                color=label_col, fontsize=6.5, zorder=3, va='bottom')

    # safety critical 半透明填充区
    safety_fill = plt.Circle((0, 0), safety_r,
                              color='#FF2222', alpha=0.04, fill=True, zorder=1)
    ax.add_patch(safety_fill)
    ax.text(-safety_r + 1, safety_r - 3,
            f'Safety Critical ≤ {safety_r} m',
            color='#FF4444', fontsize=7, zorder=3, style='italic')

    # 轴线
    ax.axhline(0, color='#111130', lw=0.5)
    ax.axvline(0, color='#111130', lw=0.5)

    ex, ey = ego['pos_x'], ego['pos_y']
    yaw_ego = ego['yaw_rad'] or 0.0

    # ── 目标 box ──
    for b in boxes:
        rx = b['translation_x'] - ex
        ry = b['translation_y'] - ey
        if abs(rx) > bev_range or abs(ry) > bev_range:
            continue

        yaw = 2.0 * math.atan2(
            2.0 * (b['rotation_w'] * b['rotation_z'] + b['rotation_x'] * b['rotation_y']),
            1.0 - 2.0 * (b['rotation_y'] ** 2 + b['rotation_z'] ** 2)
        )
        col = get_color(b['class_name'])
        dist = math.hypot(rx, ry)
        # safety critical 范围内边框加粗
        lw = 1.8 if dist <= safety_r else 1.0
        alpha = 0.95 if dist <= safety_r else 0.75

        # 旋转矩形
        w, l = b['size_wlh_0'], b['size_wlh_1']
        hw, hl = w / 2, l / 2
        local_corners = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])
        c, s = math.cos(yaw), math.sin(yaw)
        R2 = np.array([[c, -s], [s, c]])
        rot_corners = local_corners @ R2.T
        xs = rot_corners[:, 0] + rx
        ys = rot_corners[:, 1] + ry
        poly = plt.Polygon(list(zip(xs, ys)), closed=True, fill=False,
                           edgecolor=col, linewidth=lw, alpha=alpha, zorder=3)
        ax.add_patch(poly)

        # 朝向箭头
        arrow_len = l * 0.55
        dx_arr = math.cos(yaw) * arrow_len
        dy_arr = math.sin(yaw) * arrow_len
        ax.annotate('', xy=(rx + dx_arr, ry + dy_arr), xytext=(rx, ry),
                    arrowprops=dict(arrowstyle='->', color=col,
                                   lw=0.9, mutation_scale=6),
                    zorder=4)

        # pred: score 标注
        if 'score' in b:
            ax.text(rx, ry + w * 0.55 + 0.3, f'{b["score"]:.2f}',
                    color=col, fontsize=4.5, ha='center', zorder=4)

    # ── 自车 ──
    hw_e, hl_e = 1.0, 2.25
    local_ego = np.array([[-hl_e, -hw_e], [hl_e, -hw_e], [hl_e, hw_e], [-hl_e, hw_e]])
    c, s = math.cos(yaw_ego), math.sin(yaw_ego)
    R_e = np.array([[c, -s], [s, c]])
    rot_ego = local_ego @ R_e.T
    ego_poly = plt.Polygon(rot_ego, closed=True,
                           facecolor='#FFFFFF', edgecolor='#FFFFFF',
                           linewidth=2.0, alpha=0.95, zorder=5)
    ax.add_patch(ego_poly)
    # 前进方向箭头（金色）
    ax.annotate('', xy=(math.cos(yaw_ego) * 3.0, math.sin(yaw_ego) * 3.0),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#FFD700',
                                lw=2.5, mutation_scale=14),
                zorder=6)

    # ── 图例 ──
    handles = [Line2D([0], [0], color=get_color(c), lw=2, label=lbl)
               for c, lbl in LEGEND_CLS]
    handles.append(Line2D([0], [0], color='#FF3333', lw=1.8, ls='-',
                           label=f'safety ≤{safety_r}m'))
    ax.legend(handles=handles, loc='lower right',
              facecolor='#0B0B18', edgecolor='#333355',
              labelcolor='#BBBBCC', fontsize=6.5,
              ncol=1, framealpha=0.92, borderpad=0.7, handlelength=1.5)

    ax.tick_params(colors='#444466', labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor('#222244')


# ─────────────────────────── 主函数 ────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample_id',    type=int,   default=28061)
    ap.add_argument('--score_thr',    type=float, default=0.30)
    ap.add_argument('--safety_range', type=int,   default=30)
    ap.add_argument('--bev_range',    type=int,   default=100)
    ap.add_argument('--dataroot',     default='/root/autodl-tmp/NuScenes')
    ap.add_argument('--host',         default='127.0.0.1')
    ap.add_argument('--port',         type=int,   default=13306)
    ap.add_argument('--user',         default='remote1')
    ap.add_argument('--password',     default='')
    ap.add_argument('--database',     default='safetyai_sparsebev')
    ap.add_argument('--out',          default='/tmp/viz_full.png')
    args = ap.parse_args()

    t0 = time.time()
    print('连接数据库并读取数据...')
    conn = pymysql.connect(
        host=args.host, port=args.port,
        user=args.user, password=args.password,
        database=args.database,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )
    ego, gt_boxes, pred_boxes, cams, meta = fetch_data(
        conn, args.sample_id, args.score_thr)
    conn.close()

    print(f'  GT boxes     : {len(gt_boxes)}')
    print(f'  Pred boxes   : {len(pred_boxes)} (score≥{args.score_thr})')
    print(f'  Cameras      : {sorted(cams.keys())}')
    print(f'  Scene        : {meta["name"]}  {meta["log_location"]}  {meta["log_date"]}')

    # 预先读取图片文件路径（加入 dataroot）
    for ch, c in cams.items():
        c['filepath'] = f'{args.dataroot}/{c["filename"]}'

    # ── 创建图 ──
    print('生成可视化...')
    fig = plt.figure(figsize=(30, 15), facecolor='#07070F')

    outer = GridSpec(2, 1, figure=fig, hspace=0.10,
                     top=0.94, bottom=0.03, left=0.01, right=0.99)

    section_cfg = [
        ('GT Ground Truth', gt_boxes, False),
        (f'Prediction  (score ≥ {args.score_thr:.2f})', pred_boxes, True),
    ]

    for sec_idx, (sec_label, boxes, is_pred) in enumerate(section_cfg):
        # 2行 × 4列：前3列相机，第4列BEV（跨两行）
        inner = GridSpecFromSubplotSpec(
            2, 4, subplot_spec=outer[sec_idx],
            wspace=0.015, hspace=0.015,
            width_ratios=[1.78, 1.78, 1.78, 2.0],
        )

        # 6 路相机
        for row_idx, cam_row in enumerate(CAM_ROWS):
            for col_idx, channel in enumerate(cam_row):
                ax = fig.add_subplot(inner[row_idx, col_idx])
                short = channel.replace('CAM_', '').replace('_', ' ')
                if channel in cams:
                    draw_camera_view(ax, cams[channel], boxes, ego, short)
                else:
                    ax.set_facecolor('#111122')
                    ax.text(0.5, 0.5, channel, ha='center', va='center',
                            color='#555577', transform=ax.transAxes, fontsize=8)
                    ax.axis('off')

        # BEV（跨两行）
        ax_bev = fig.add_subplot(inner[:, 3])
        draw_bev(ax_bev, boxes, ego, sec_label,
                 args.bev_range, args.safety_range, is_pred)

        # 分区标题
        fig.text(0.005, 0.96 - sec_idx * 0.49,
                 '▌ ' + ('Ground Truth' if sec_idx == 0 else f'Prediction  (score ≥ {args.score_thr:.2f})'),
                 color='#55FF88' if sec_idx == 0 else '#55AAFF',
                 fontsize=10, fontweight='bold',
                 transform=fig.transFigure)

    # 全图标题
    token_short = meta['sample_token'][:20] + '…'
    fig.suptitle(
        f'SparseBEV 检测可视化  |  sample {token_short}  '
        f'{meta["name"]}  {meta["log_location"]}  {meta["log_date"]}',
        color='#CCCCDD', fontsize=11, y=0.975,
    )

    plt.savefig(args.out, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'保存 → {args.out}  ({time.time()-t0:.1f}s)')


if __name__ == '__main__':
    main()
