#!/usr/bin/env python3
"""
Visualize 200 random val samples using:
- GT from nuscenes_infos_val_sweep.pkl
- Predictions from submission/pts_bbox/results_nusc.json
and overlay two prediction filters:
  - "official" (class_range only)
  - "safety" (class_range + safety_max_dist)

This script is designed for the dataset layout:
  E:\\Data\\Nuscenes\\Full\\
    samples\\...
    sweeps\\...
    nuscenes_infos_val_sweep.pkl
and does NOT require v1.0-trainval/*.json tables.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CAM_TYPES = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
]

COLOR_GT = (0, 255, 255)          # cyan
COLOR_OFFICIAL = (0, 255, 0)      # green
COLOR_SAFETY = (255, 60, 60)      # red

CLASSNAME_TO_COLOR = {  # RGB (fallback palette by class)
    "car": (255, 158, 0),
    "pedestrian": (0, 0, 230),
    "trailer": (255, 140, 0),
    "truck": (255, 99, 71),
    "bus": (255, 127, 80),
    "motorcycle": (255, 61, 99),
    "construction_vehicle": (233, 150, 70),
    "bicycle": (220, 20, 60),
    "barrier": (112, 128, 144),
    "traffic_cone": (47, 79, 79),
}

EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def safe_join(root: str, path: str) -> str:
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


def _rot_z(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def box_corners_lidar_center(x: float, y: float, z: float, dx: float, dy: float, dz: float, yaw: float) -> np.ndarray:
    x_c = dx / 2.0
    y_c = dy / 2.0
    z_c = dz / 2.0
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
    R = _rot_z(yaw)
    return (R @ corners) + np.array([[x], [y], [z]], dtype=np.float32)


def quat_to_rot(q: List[float]) -> np.ndarray:
    """
    nuScenes uses (w, x, y, z). Return 3x3 rotation matrix.
    """
    w, x, y, z = [float(v) for v in q]
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


def project(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    x = points_cam[0]
    y = points_cam[1]
    z = points_cam[2]
    z = np.where(z == 0, 1e-6, z)
    u = (K[0, 0] * x / z) + K[0, 2]
    v = (K[1, 1] * y / z) + K[1, 2]
    return np.stack([u, v], axis=0)


def draw_box(draw: ImageDraw.ImageDraw, pts2d: np.ndarray, color: Tuple[int, int, int], width: int) -> None:
    for a, b in EDGES:
        draw.line(
            [(float(pts2d[0, a]), float(pts2d[1, a])), (float(pts2d[0, b]), float(pts2d[1, b]))],
            fill=color,
            width=width,
        )


def global_to_cam(points_g: np.ndarray, R_s2g: np.ndarray, t_s2g: np.ndarray) -> np.ndarray:
    """
    points_g: (3,N), global frame.
    sensor2global: x_g = R_s2g x_s + t_s2g
    so x_s = R_s2g^T (x_g - t_s2g)
    """
    R_g2s = R_s2g.T
    return (R_g2s @ (points_g - t_s2g))


def lidar_to_cam(points_l: np.ndarray, R_c2l: np.ndarray, t_c2l: np.ndarray) -> np.ndarray:
    """
    sensor2lidar: x_l = R_c2l x_c + t_c2l => x_c = R_c2l^T (x_l - t_c2l)
    """
    R_l2c = R_c2l.T
    return (R_l2c @ (points_l - t_c2l))


def load_class_range_official(repo_root: str) -> Dict[str, float]:
    cfg_path = os.path.join(
        repo_root, "offline_nuscenes_eval", "nuscenes", "eval", "detection", "configs", "detection_cvpr_2019.json"
    )
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    return {str(k): float(v) for k, v in cfg["class_range"].items()}


def load_safety_cfg(repo_root: str) -> Tuple[float, Dict[str, float]]:
    cfg_path = os.path.join(
        repo_root,
        "safety_critical_eval",
        "nuscenes",
        "eval",
        "detection",
        "configs",
        "detection_safety_critical.json",
    )
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    safety_max_dist = float(cfg.get("safety_max_dist", 30.0))
    class_range = {str(k): float(v) for k, v in cfg.get("class_range", {}).items()}
    return safety_max_dist, class_range


def main():
    parser = argparse.ArgumentParser("Visualize 200 random samples: official vs safety-critical filtering")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root, e.g. E:\\Data\\Nuscenes\\Full")
    parser.add_argument(
        "--ann_pkl",
        type=str,
        default="nuscenes_infos_val_sweep.pkl",
        help="Val info pkl filename/path under data_root",
    )
    parser.add_argument(
        "--pred_json",
        type=str,
        default=os.path.join("submission", "pts_bbox", "results_nusc.json"),
        help="Prediction json (results_nusc.json). If relative, treated as repo-relative.",
    )
    parser.add_argument("--out_dir", type=str, default=os.path.join("outputs", "submission_viz200"))
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--score_thr", type=float, default=0.05, help="Prediction score threshold before filtering")
    parser.add_argument("--max_boxes", type=int, default=80, help="Max predicted boxes per view (after filtering)")
    args = parser.parse_args()

    repo_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    data_root = os.path.normpath(args.data_root)
    ann_path = args.ann_pkl if os.path.isabs(args.ann_pkl) else os.path.join(data_root, args.ann_pkl)
    pred_path = args.pred_json if os.path.isabs(args.pred_json) else os.path.join(repo_root, args.pred_json)

    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"ann pkl not found: {ann_path}")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"pred json not found: {pred_path}")

    os.makedirs(args.out_dir, exist_ok=True)

    with open(ann_path, "rb") as f:
        d = pickle.load(f)
    infos = d["infos"] if isinstance(d, dict) and "infos" in d else d
    n = len(infos)
    k = min(args.num_samples, n)
    random.seed(args.seed)
    indices = sorted(random.sample(range(n), k=k))

    # token -> info index
    token_to_info = {info["token"]: info for info in infos}

    # Load prediction json (big).
    with open(pred_path, "r") as f:
        pred_all = json.load(f)
    pred_by_token = pred_all.get("results", {})

    class_range_official = load_class_range_official(repo_root)
    safety_max_dist, class_range_safety = load_safety_cfg(repo_root)
    # prefer safety file's class_range if present (should match official defaults)
    class_range_used = class_range_safety or class_range_official

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for out_i, idx in enumerate(indices):
        info = infos[idx]
        token = info["token"]
        cams: Dict[str, dict] = info["cams"]

        ego_t = np.array(info["ego2global_translation"], dtype=np.float32).reshape(3, 1)

        gt_boxes = np.array(info.get("gt_boxes", []), dtype=np.float32)  # (M,7) lidar frame
        gt_names = [str(x) for x in info.get("gt_names", [])]
        valid_flag = info.get("valid_flag", None)
        if valid_flag is None:
            valid_mask = np.ones((len(gt_names),), dtype=bool)
        else:
            valid_mask = np.array(valid_flag).astype(bool)

        pred_list = pred_by_token.get(token, [])
        # pre-filter by score
        pred_list = [p for p in pred_list if float(p.get("detection_score", 0.0)) >= args.score_thr]

        # build official/safety filtered lists
        preds_official = []
        preds_safety = []
        for p in pred_list:
            name = str(p.get("detection_name", ""))
            if name == "":
                continue
            t = np.array(p["translation"], dtype=np.float32).reshape(3, 1)
            ego_dist = float(np.linalg.norm((t - ego_t)[:2]))
            max_d = float(class_range_used.get(name, safety_max_dist))
            if ego_dist < max_d:
                preds_official.append(p)
                if ego_dist < safety_max_dist:
                    preds_safety.append(p)

        # render 6 cams and compose 2x3
        imgs = []
        for cam_type in CAM_TYPES:
            cam = cams[cam_type]
            img_path = safe_join(data_root, cam["data_path"])
            img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img)

            K = np.array(cam["cam_intrinsic"], dtype=np.float32).reshape(3, 3)

            # lidar->cam for GT: sensor2lidar
            R_c2l = np.array(cam["sensor2lidar_rotation"], dtype=np.float32).reshape(3, 3)
            t_c2l = np.array(cam["sensor2lidar_translation"], dtype=np.float32).reshape(3, 1)

            # global->cam for preds: sensor2global
            R_s2g = np.array(cam["sensor2global_rotation"], dtype=np.float32).reshape(3, 3)
            t_s2g = np.array(cam["sensor2global_translation"], dtype=np.float32).reshape(3, 1)

            # draw GT
            for b, name, ok in zip(gt_boxes, gt_names, valid_mask):
                if not ok:
                    continue
                x, y, z, dx, dy, dz, yaw = [float(v) for v in b.tolist()]
                corners_l = box_corners_lidar_center(x, y, z, dx, dy, dz, yaw)
                corners_c = lidar_to_cam(corners_l, R_c2l=R_c2l, t_c2l=t_c2l)
                if float(np.mean(corners_c[2])) <= 0.5:
                    continue
                pts2d = project(corners_c, K)
                draw_box(draw, pts2d, color=COLOR_GT, width=2)

            # draw official preds (green)
            rendered = 0
            for p in preds_official:
                if rendered >= args.max_boxes:
                    break
                t = np.array(p["translation"], dtype=np.float32).reshape(3, 1)
                size = [float(x) for x in p["size"]]  # w,l,h in global
                rot = [float(x) for x in p["rotation"]]  # quat wxyz
                R = quat_to_rot(rot)
                w, l, h = size
                corners_local = box_corners_lidar_center(0, 0, 0, l, w, h, 0.0)  # reuse shape builder
                corners_g = (R @ corners_local) + t
                corners_c = global_to_cam(corners_g, R_s2g=R_s2g, t_s2g=t_s2g)
                if float(np.mean(corners_c[2])) <= 0.5:
                    continue
                pts2d = project(corners_c, K)
                draw_box(draw, pts2d, color=COLOR_OFFICIAL, width=2)
                rendered += 1

            # draw safety preds (red) thinner on top
            rendered = 0
            for p in preds_safety:
                if rendered >= args.max_boxes:
                    break
                t = np.array(p["translation"], dtype=np.float32).reshape(3, 1)
                size = [float(x) for x in p["size"]]
                rot = [float(x) for x in p["rotation"]]
                R = quat_to_rot(rot)
                w, l, h = size
                corners_local = box_corners_lidar_center(0, 0, 0, l, w, h, 0.0)
                corners_g = (R @ corners_local) + t
                corners_c = global_to_cam(corners_g, R_s2g=R_s2g, t_s2g=t_s2g)
                if float(np.mean(corners_c[2])) <= 0.5:
                    continue
                pts2d = project(corners_c, K)
                draw_box(draw, pts2d, color=COLOR_SAFETY, width=1)
                rendered += 1

            # header
            draw.rectangle([(0, 0), (img.size[0], 30)], fill=(0, 0, 0))
            draw.text((6, 6), f"{cam_type} | token={token[:8]} | idx={idx}", fill=(255, 255, 255), font=font)
            imgs.append(img)

        w = max(im.size[0] for im in imgs)
        h = max(im.size[1] for im in imgs)
        canvas = Image.new("RGB", (w * 3, h * 2), (15, 15, 15))
        for i, im in enumerate(imgs):
            r = i // 3
            c = i % 3
            canvas.paste(im, (c * w, r * h))

        # legend
        draw_c = ImageDraw.Draw(canvas)
        draw_c.rectangle([(0, 0), (540, 26)], fill=(0, 0, 0))
        draw_c.text((8, 5), "GT(cyan)  OFFICIAL pred(green)  SAFETY pred(red)", fill=(255, 255, 255), font=font)

        out_path = os.path.join(args.out_dir, f"sub_viz_{out_i:04d}_idx{idx}.jpg")
        canvas.save(out_path, quality=92)

    print(f"Saved {k} visualizations to: {args.out_dir}")


if __name__ == "__main__":
    main()

