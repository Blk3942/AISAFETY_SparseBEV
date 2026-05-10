#!/usr/bin/env python3
"""
Visualize nuScenes camera + BEV: GT vs prediction from a saved results_nusc.json.

This avoids rerunning model inference. It also supports overlaying a safety-critical GT
subset (filtered by safety_max_dist + class_range) for comparison.

PKL mode:
- If you pass --pkl_ann, the script will use nuscenes_infos_val_sweep.pkl to get:
  - camera image paths + intrinsics/extrinsics
  - GT boxes in lidar frame
- It will still use --results (results_nusc.json) for predictions in global frame.
- In PKL mode, BEV pointcloud rendering is disabled (no nuScenes JSON tables).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import box_in_image


classname_to_color = {
    'car': (255, 158, 0),
    'pedestrian': (0, 0, 230),
    'trailer': (255, 140, 0),
    'truck': (255, 99, 71),
    'bus': (255, 127, 80),
    'motorcycle': (255, 61, 99),
    'construction_vehicle': (233, 150, 70),
    'bicycle': (220, 20, 60),
    'barrier': (112, 128, 144),
    'traffic_cone': (47, 79, 79),
}

SAFETY_GT_COLOR = (180, 0, 255)  # magenta-ish


def category_to_detection_name(category_name: str):
    mapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck',
    }
    return mapping.get(category_name, None)


def _load_safety_cfg(path: str) -> Tuple[float, Dict[str, float]]:
    with open(path, 'r') as f:
        cfg = json.load(f)
    safety_max_dist = float(cfg.get('safety_max_dist', 30.0))
    class_range = cfg.get('class_range', {})
    class_range = {str(k): float(v) for k, v in class_range.items()}
    return safety_max_dist, class_range


def _filter_safety_lidar_boxes(bboxes_lidar: List[Box], safety_max_dist: float, class_range: Dict[str, float]) -> List[Box]:
    out = []
    for box in bboxes_lidar:
        dist = float(np.sqrt(box.center[0] ** 2 + box.center[1] ** 2))
        cls = getattr(box, 'name', '')
        max_d = float(class_range.get(cls, safety_max_dist))
        if dist < min(max_d, safety_max_dist):
            out.append(box)
    return out


def _safe_join(root: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    p = path.replace("\\", "/")
    for anchor in ("samples/", "sweeps/"):
        j = p.find(anchor)
        if j >= 0:
            p = p[j:]
            break
    p = p.replace("/", os.sep)
    return os.path.normpath(os.path.join(root, p))


def _quat_to_rot(q_wxyz: List[float]) -> np.ndarray:
    """nuScenes quaternion (w,x,y,z) -> 3x3 rotation matrix."""
    w, x, y, z = [float(v) for v in q_wxyz]
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return np.array(
        [
            [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=np.float32,
    )


def _project(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    x, y, z = points_cam[0], points_cam[1], points_cam[2]
    z = np.where(z == 0, 1e-6, z)
    u = (K[0, 0] * x / z) + K[0, 2]
    v = (K[1, 1] * y / z) + K[1, 2]
    return np.stack([u, v], axis=0)


def _lidar_to_cam(points_l: np.ndarray, R_c2l: np.ndarray, t_c2l: np.ndarray) -> np.ndarray:
    """sensor2lidar: x_l = R x_c + t => x_c = R^T (x_l - t)"""
    R_l2c = R_c2l.T
    return (R_l2c @ (points_l - t_c2l))


def _global_to_cam(points_g: np.ndarray, R_s2g: np.ndarray, t_s2g: np.ndarray) -> np.ndarray:
    """sensor2global: x_g = R x_s + t => x_s = R^T (x_g - t)"""
    R_g2s = R_s2g.T
    return (R_g2s @ (points_g - t_s2g))


def _rot_z(yaw: float) -> np.ndarray:
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _box_corners_center(x: float, y: float, z: float, dx: float, dy: float, dz: float, yaw: float) -> np.ndarray:
    """(3,8) corners around center."""
    x_c, y_c, z_c = dx / 2.0, dy / 2.0, dz / 2.0
    corners = np.array(
        [
            [x_c, y_c, z_c],
            [x_c, -y_c, z_c],
            [-x_c, -y_c, z_c],
            [-x_c, y_c, z_c],
            [x_c, y_c, -z_c],
            [x_c, -y_c, -z_c],
            [-x_c, -y_c, -z_c],
            [-x_c, y_c, -z_c],
        ],
        dtype=np.float32,
    ).T
    return (_rot_z(yaw) @ corners) + np.array([[x], [y], [z]], dtype=np.float32)


def _draw_edges(ax, pts2d: np.ndarray, rgb: Tuple[int, int, int], lw: float):
    c = np.array(rgb) / 255.0
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for a, b in edges:
        ax.plot([pts2d[0, a], pts2d[0, b]], [pts2d[1, a], pts2d[1, b]], color=c, linewidth=lw)


def _load_infos_pkl(pkl_path: str):
    import pickle
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)
    infos = d['infos'] if isinstance(d, dict) and 'infos' in d else d
    token_to_info = {info['token']: info for info in infos}
    return token_to_info


def _viz_pkl_mode(dataroot: str, token_to_info: dict, results_json: dict, sample_token: str, fig, gs, score_thr: float):
    """
    PKL mode visualization:
    - Top: GT (from pkl, lidar->cam projection)
    - Bottom: Pred (from results json, global->cam projection)
    - Right BEV panes are placeholders.
    """
    info = token_to_info.get(sample_token, None)
    if info is None:
        return False
    cams = info['cams']

    gt_boxes = np.array(info.get('gt_boxes', []), dtype=np.float32)  # (M,7) lidar
    gt_names = [str(x) for x in info.get('gt_names', [])]
    valid_flag = info.get('valid_flag', None)
    if valid_flag is None:
        valid_mask = np.ones((len(gt_names),), dtype=bool)
    else:
        valid_mask = np.array(valid_flag).astype(bool)

    pred_list = results_json.get('results', {}).get(sample_token, [])
    pred_list = [p for p in pred_list if float(p.get('detection_score', 0.0)) >= score_thr]

    cam_types = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT',
    ]

    for cam_id, cam_type in enumerate(cam_types):
        cam = cams[cam_type]
        img_path = _safe_join(dataroot, cam['data_path'])
        img = Image.open(img_path)

        K = np.array(cam['cam_intrinsic'], dtype=np.float32).reshape(3, 3)
        R_c2l = np.array(cam['sensor2lidar_rotation'], dtype=np.float32).reshape(3, 3)
        t_c2l = np.array(cam['sensor2lidar_translation'], dtype=np.float32).reshape(3, 1)
        R_s2g = np.array(cam['sensor2global_rotation'], dtype=np.float32).reshape(3, 3)
        t_s2g = np.array(cam['sensor2global_translation'], dtype=np.float32).reshape(3, 1)

        r = cam_id // 3
        c = cam_id % 3

        ax_gt = fig.add_subplot(gs[r, c])
        ax_gt.imshow(img)
        ax_gt.axis('off')
        ax_gt.set_title(cam_type if cam_id > 0 else f'Ground truth\n{cam_type}', fontsize=8)

        for b, name, ok in zip(gt_boxes, gt_names, valid_mask):
            if not ok:
                continue
            x, y, z, dx, dy, dz, yaw = [float(v) for v in b.tolist()]
            corners_l = _box_corners_center(x, y, z, dx, dy, dz, yaw)
            corners_c = _lidar_to_cam(corners_l, R_c2l=R_c2l, t_c2l=t_c2l)
            if float(np.mean(corners_c[2])) <= 0.5:
                continue
            pts2d = _project(corners_c, K)
            col = classname_to_color.get(name, (200, 200, 200))
            _draw_edges(ax_gt, pts2d, col, lw=1.8)

        ax_pr = fig.add_subplot(gs[r + 2, c])
        ax_pr.imshow(img)
        ax_pr.axis('off')
        ax_pr.set_title(cam_type if cam_id > 0 else f'Prediction\n{cam_type}', fontsize=8)

        for det in pred_list:
            name = str(det.get('detection_name', ''))
            center = np.array(det['translation'], dtype=np.float32).reshape(3, 1)
            w, l, h = [float(x) for x in det['size']]
            R = _quat_to_rot(det['rotation'])
            corners_local = _box_corners_center(0, 0, 0, l, w, h, 0.0)
            corners_g = (R @ corners_local) + center
            corners_c = _global_to_cam(corners_g, R_s2g=R_s2g, t_s2g=t_s2g)
            if float(np.mean(corners_c[2])) <= 0.5:
                continue
            pts2d = _project(corners_c, K)
            col = classname_to_color.get(name, (200, 200, 200))
            _draw_edges(ax_pr, pts2d, col, lw=1.2)

    ax0 = fig.add_subplot(gs[0:2, 3])
    ax0.axis('off')
    ax0.set_title('Ground truth\nBEV disabled (PKL mode)', fontsize=8)
    ax1 = fig.add_subplot(gs[2:4, 3])
    ax1.axis('off')
    ax1.set_title('Prediction\nBEV disabled (PKL mode)', fontsize=8)

    return True


def _global_box_to_lidar(nusc: NuScenes, sample_token: str, box_global: Box) -> Box:
    sample = nusc.get('sample', sample_token)
    lidar_sd_token = sample['data']['LIDAR_TOP']
    sd_record = nusc.get('sample_data', lidar_sd_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    b = box_global.copy()
    # global -> ego
    b.translate(-np.array(pose_record['translation']))
    b.rotate(Quaternion(pose_record['rotation']).inverse)
    # ego -> lidar sensor
    b.translate(-np.array(cs_record['translation']))
    b.rotate(Quaternion(cs_record['rotation']).inverse)
    return b


def _load_pred_boxes_lidar(nusc: NuScenes, results_json: dict, sample_token: str, score_thr: float) -> List[Box]:
    boxes = []
    for det in results_json['results'].get(sample_token, []):
        score = float(det.get('detection_score', 0.0))
        if score < score_thr:
            continue
        name = det.get('detection_name', 'unknown')
        q = Quaternion(det['rotation'])
        b_global = Box(
            center=det['translation'],
            size=det['size'],
            orientation=q,
            velocity=tuple(det.get('velocity', [0.0, 0.0])) + (0.0,),
            score=score,
            name=name,
        )
        boxes.append(_global_box_to_lidar(nusc, sample_token, b_global))
    return boxes


def _load_gt_boxes_lidar(nusc: NuScenes, sample_token: str) -> List[Box]:
    sample = nusc.get('sample', sample_token)
    boxes = []
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        name = category_to_detection_name(ann['category_name'])
        if name is None:
            continue
        b_global = Box(
            center=ann['translation'],
            size=ann['size'],
            orientation=Quaternion(ann['rotation']),
            name=name,
        )
        boxes.append(_global_box_to_lidar(nusc, sample_token, b_global))
    return boxes


def _viz_bbox_block(nusc: NuScenes, bboxes_lidar: List[Box], sample_token: str, fig, gs, row_start: int, block_title: str, linewidth=1.5, override_rgb=None):
    cam_types = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT',
    ]

    # lidar -> ego for camera rendering
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_sd = nusc.get('sample_data', lidar_token)
    lidar_cs = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    lidar2ego_rot = lidar_cs['rotation']
    lidar2ego_trans = lidar_cs['translation']

    for cam_id, cam_type in enumerate(cam_types):
        cam_sd_token = sample['data'][cam_type]
        sd_record = nusc.get('sample_data', cam_sd_token)
        cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        intrinsic = np.array(cs_record['camera_intrinsic'])
        img_path = nusc.get_sample_data_path(cam_sd_token)
        img_size = (sd_record['width'], sd_record['height'])

        r = row_start + cam_id // 3
        c = cam_id % 3
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(Image.open(img_path))

        for bbox in bboxes_lidar:
            bbox = bbox.copy()
            bbox.rotate(Quaternion(lidar2ego_rot))
            bbox.translate(np.array(lidar2ego_trans))
            bbox.translate(-np.array(cs_record['translation']))
            bbox.rotate(Quaternion(cs_record['rotation']).inverse)

            if box_in_image(bbox, intrinsic, img_size):
                if override_rgb is not None:
                    c_rgb = np.array(override_rgb) / 255.0
                else:
                    col = classname_to_color.get(bbox.name, (200, 200, 200))
                    c_rgb = np.array(col) / 255.0
                bbox.render(ax, view=intrinsic, normalize=True, colors=(c_rgb, c_rgb, c_rgb), linewidth=linewidth)

        ax.axis('off')
        title = cam_type if cam_id > 0 else f'{block_title}\n{cam_type}'
        ax.set_title(title, fontsize=8)
        ax.set_xlim(0, img_size[0])
        ax.set_ylim(img_size[1], 0)

    ax = fig.add_subplot(gs[row_start:row_start + 2, 3])
    nusc.explorer.render_sample_data(lidar_token, with_anns=False, ax=ax, verbose=False)
    ax.axis('off')
    ax.set_title(block_title + '\nLIDAR_TOP', fontsize=8)
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)

    pose_record = nusc.get('ego_pose', lidar_sd['ego_pose_token'])

    for bbox in bboxes_lidar:
        bbox = bbox.copy()
        bbox.rotate(Quaternion(lidar_cs['rotation']))
        bbox.translate(np.array(lidar_cs['translation']))
        bbox.rotate(Quaternion(pose_record['rotation']))
        yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
        bbox.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
        if override_rgb is not None:
            c_rgb = np.array(override_rgb) / 255.0
        else:
            col = classname_to_color.get(bbox.name, (200, 200, 200))
            c_rgb = np.array(col) / 255.0
        bbox.render(ax, view=np.eye(4), colors=(c_rgb, c_rgb, c_rgb), linewidth=linewidth)


def _create_block_axes(nusc: NuScenes, sample_token: str, fig, gs, row_start: int, block_title: str):
    """
    Create axes and draw backgrounds (images + lidar pointcloud) once.
    Returns a dict with camera axes and lidar axis, plus per-sensor cached transforms.
    """
    cam_types = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT',
    ]

    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_sd = nusc.get('sample_data', lidar_token)
    lidar_cs = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
    lidar2ego_rot = lidar_cs['rotation']
    lidar2ego_trans = lidar_cs['translation']

    cam_axes = {}
    cam_calibs = {}
    cam_intrinsics = {}
    cam_img_sizes = {}
    cam_sd_tokens = {}

    for cam_id, cam_type in enumerate(cam_types):
        cam_sd_token = sample['data'][cam_type]
        sd_record = nusc.get('sample_data', cam_sd_token)
        cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        intrinsic = np.array(cs_record['camera_intrinsic'])
        img_path = nusc.get_sample_data_path(cam_sd_token)
        img_size = (sd_record['width'], sd_record['height'])

        r = row_start + cam_id // 3
        c = cam_id % 3
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(Image.open(img_path))
        ax.axis('off')
        title = cam_type if cam_id > 0 else f'{block_title}\n{cam_type}'
        ax.set_title(title, fontsize=8)
        ax.set_xlim(0, img_size[0])
        ax.set_ylim(img_size[1], 0)

        cam_axes[cam_type] = ax
        cam_calibs[cam_type] = cs_record
        cam_intrinsics[cam_type] = intrinsic
        cam_img_sizes[cam_type] = img_size
        cam_sd_tokens[cam_type] = cam_sd_token

    lidar_ax = fig.add_subplot(gs[row_start:row_start + 2, 3])
    nusc.explorer.render_sample_data(lidar_token, with_anns=False, ax=lidar_ax, verbose=False)
    lidar_ax.axis('off')
    lidar_ax.set_title(block_title + '\nLIDAR_TOP', fontsize=8)
    lidar_ax.set_xlim(-40, 40)
    lidar_ax.set_ylim(-40, 40)

    pose_record = nusc.get('ego_pose', lidar_sd['ego_pose_token'])

    return {
        'cam_types': cam_types,
        'cam_axes': cam_axes,
        'lidar_ax': lidar_ax,
        'lidar_token': lidar_token,
        'lidar_sd': lidar_sd,
        'lidar_cs': lidar_cs,
        'pose_record': pose_record,
        'lidar2ego_rot': lidar2ego_rot,
        'lidar2ego_trans': lidar2ego_trans,
        'cam_calibs': cam_calibs,
        'cam_intrinsics': cam_intrinsics,
        'cam_img_sizes': cam_img_sizes,
    }


def _render_boxes_on_block(ctx, bboxes_lidar: List[Box], linewidth=1.5, override_rgb=None):
    """Render boxes onto pre-created axes (no background redraw)."""
    cam_types = ctx['cam_types']
    lidar2ego_rot = ctx['lidar2ego_rot']
    lidar2ego_trans = ctx['lidar2ego_trans']

    for cam_type in cam_types:
        ax = ctx['cam_axes'][cam_type]
        cs_record = ctx['cam_calibs'][cam_type]
        intrinsic = ctx['cam_intrinsics'][cam_type]
        img_size = ctx['cam_img_sizes'][cam_type]

        for bbox in bboxes_lidar:
            bb = bbox.copy()
            bb.rotate(Quaternion(lidar2ego_rot))
            bb.translate(np.array(lidar2ego_trans))
            bb.translate(-np.array(cs_record['translation']))
            bb.rotate(Quaternion(cs_record['rotation']).inverse)

            if box_in_image(bb, intrinsic, img_size):
                if override_rgb is not None:
                    c_rgb = np.array(override_rgb) / 255.0
                else:
                    col = classname_to_color.get(bb.name, (200, 200, 200))
                    c_rgb = np.array(col) / 255.0
                bb.render(ax, view=intrinsic, normalize=True, colors=(c_rgb, c_rgb, c_rgb), linewidth=linewidth)

    lidar_ax = ctx['lidar_ax']
    lidar_cs = ctx['lidar_cs']
    pose_record = ctx['pose_record']

    for bbox in bboxes_lidar:
        bb = bbox.copy()
        bb.rotate(Quaternion(lidar_cs['rotation']))
        bb.translate(np.array(lidar_cs['translation']))
        bb.rotate(Quaternion(pose_record['rotation']))
        yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
        bb.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
        if override_rgb is not None:
            c_rgb = np.array(override_rgb) / 255.0
        else:
            col = classname_to_color.get(bb.name, (200, 200, 200))
            c_rgb = np.array(col) / 255.0
        bb.render(lidar_ax, view=np.eye(4), colors=(c_rgb, c_rgb, c_rgb), linewidth=linewidth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True, help='results_nusc.json path')
    parser.add_argument('--dataroot', required=True)
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--out_dir', default='outputs/gt_pred_viz_rand100')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--score_threshold', type=float, default=0.3)
    parser.add_argument(
        '--pkl_ann',
        type=str,
        default=None,
        help='启用 PKL 模式：传入 nuscenes_infos_val_sweep.pkl 路径（适用于 dataroot 下无 v1.0-trainval 表）',
    )
    parser.add_argument('--viz_safety_gt', action='store_true')
    parser.add_argument(
        '--safety_cfg',
        type=str,
        default=os.path.join('safety_critical_eval', 'nuscenes', 'eval', 'detection', 'configs', 'detection_safety_critical.json'),
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.results, 'r') as f:
        results_json = json.load(f)

    sample_tokens = list(results_json.get('results', {}).keys())
    random.seed(args.seed)
    picked = random.sample(sample_tokens, k=min(args.num_samples, len(sample_tokens)))

    if args.viz_safety_gt:
        safety_max_dist, class_range = _load_safety_cfg(args.safety_cfg)
    else:
        safety_max_dist, class_range = None, None

    if args.pkl_ann:
        pkl_path = args.pkl_ann if os.path.isabs(args.pkl_ann) else os.path.join(args.dataroot, args.pkl_ann)
        token_to_info = _load_infos_pkl(pkl_path)
        for idx, sample_token in enumerate(picked):
            fig = plt.figure(figsize=(15.5, 10))
            gs = GridSpec(4, 4, figure=fig, hspace=0.15, wspace=0.05)
            ok = _viz_pkl_mode(args.dataroot, token_to_info, results_json, sample_token, fig, gs, args.score_threshold)
            if not ok:
                plt.close()
                continue
            fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.02, hspace=0.12, wspace=0.06)
            out_path = os.path.join(args.out_dir, f'gt_pred_{idx:04d}.jpg')
            plt.savefig(out_path, dpi=160, bbox_inches='tight')
            plt.close()
    else:
        nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
        for idx, sample_token in enumerate(picked):
            bboxes_gt = _load_gt_boxes_lidar(nusc, sample_token)
            bboxes_pred = _load_pred_boxes_lidar(nusc, results_json, sample_token, args.score_threshold)

            fig = plt.figure(figsize=(15.5, 10))
            gs = GridSpec(4, 4, figure=fig, hspace=0.15, wspace=0.05)

            ctx_gt = _create_block_axes(nusc, sample_token, fig, gs, row_start=0, block_title='Ground truth')
            _render_boxes_on_block(ctx_gt, bboxes_gt, linewidth=2.0, override_rgb=None)
            if args.viz_safety_gt and safety_max_dist is not None:
                bboxes_safety = _filter_safety_lidar_boxes(bboxes_gt, safety_max_dist, class_range or {})
                _render_boxes_on_block(ctx_gt, bboxes_safety, linewidth=2.2, override_rgb=SAFETY_GT_COLOR)

            ctx_pred = _create_block_axes(nusc, sample_token, fig, gs, row_start=2, block_title='Prediction')
            _render_boxes_on_block(ctx_pred, bboxes_pred, linewidth=1.5, override_rgb=None)

            fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.02, hspace=0.12, wspace=0.06)
            out_path = os.path.join(args.out_dir, f'gt_pred_{idx:04d}.jpg')
            plt.savefig(out_path, dpi=160, bbox_inches='tight')
            plt.close()


if __name__ == '__main__':
    main()

