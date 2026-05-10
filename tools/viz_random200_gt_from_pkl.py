#!/usr/bin/env python3
"""
Randomly visualize nuScenes val samples (GT) using only the *.pkl info file.

This is designed for datasets laid out like:
  E:\\Data\\Nuscenes\\Full\\
    samples\\CAM_...\\xxx.jpg
    sweeps\\...
    nuscenes_infos_val_sweep.pkl

It does NOT require v1.0-trainval/*.json tables.
"""

from __future__ import annotations

import argparse
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

CLASSNAME_TO_COLOR = {  # RGB
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


def _rot_z(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def box_corners_lidar(x: float, y: float, z: float, dx: float, dy: float, dz: float, yaw: float) -> np.ndarray:
    """
    Returns 8 corners in lidar frame (3, 8).
    Assumes box is centered at (x,y,z) with size (dx,dy,dz) and yaw around +Z.
    The exact z convention in pkl varies across pipelines; for visualization we treat (x,y,z) as box center.
    """
    # corners in box local frame
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
    ).T  # (3, 8)
    R = _rot_z(yaw)
    corners = (R @ corners) + np.array([[x], [y], [z]], dtype=np.float32)
    return corners


def lidar_to_cam_matrix(sensor2lidar_rotation: List[float], sensor2lidar_translation: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert cam->lidar extrinsics to lidar->cam.
    pkl stores sensor2lidar_{rotation,translation}: x_lidar = R * x_cam + t
    So x_cam = R^T * (x_lidar - t) = R^T * x_lidar + (-R^T t)
    """
    R_c2l = np.array(sensor2lidar_rotation, dtype=np.float32).reshape(3, 3)
    t_c2l = np.array(sensor2lidar_translation, dtype=np.float32).reshape(3, 1)
    R_l2c = R_c2l.T
    t_l2c = -R_l2c @ t_c2l
    return R_l2c, t_l2c


def project(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    points_cam: (3, N) in camera frame.
    Returns pixel coords (2, N).
    """
    x = points_cam[0]
    y = points_cam[1]
    z = points_cam[2]
    z = np.where(z == 0, 1e-6, z)
    u = (K[0, 0] * x / z) + K[0, 2]
    v = (K[1, 1] * y / z) + K[1, 2]
    return np.stack([u, v], axis=0)


EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # top
    (4, 5), (5, 6), (6, 7), (7, 4),  # bottom
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
]


def draw_box_3d(draw: ImageDraw.ImageDraw, pts2d: np.ndarray, color: Tuple[int, int, int], width: int = 2) -> None:
    for a, b in EDGES:
        xa, ya = float(pts2d[0, a]), float(pts2d[1, a])
        xb, yb = float(pts2d[0, b]), float(pts2d[1, b])
        draw.line([(xa, ya), (xb, yb)], fill=color, width=width)


def safe_join(root: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    # Many nuScenes info pkls store paths like "data/nuscenes/samples/..."
    # while local dataset root directly contains "samples/..." and "sweeps/...".
    p = path.replace("\\", "/")
    for anchor in ("samples/", "sweeps/"):
        j = p.find(anchor)
        if j >= 0:
            p = p[j:]
            break
    p = p.replace("/", os.sep)
    return os.path.normpath(os.path.join(root, p))


def main():
    parser = argparse.ArgumentParser("Random visualize 200 GT samples from nuscenes_infos_val_sweep.pkl")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root, e.g. E:\\Data\\Nuscenes\\Full")
    parser.add_argument("--ann_pkl", type=str, default="nuscenes_infos_val_sweep.pkl", help="Val info pkl filename/path")
    parser.add_argument("--out_dir", type=str, default="outputs/val200_gt_viz", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of random samples")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--max_boxes", type=int, default=100, help="Max GT boxes rendered per view (for speed)")
    parser.add_argument("--font_size", type=int, default=18)
    args = parser.parse_args()

    data_root = os.path.normpath(args.data_root)
    ann_pkl = args.ann_pkl
    ann_path = ann_pkl if os.path.isabs(ann_pkl) else os.path.join(data_root, ann_pkl)
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"ann pkl not found: {ann_path}")

    os.makedirs(args.out_dir, exist_ok=True)

    with open(ann_path, "rb") as f:
        d = pickle.load(f)
    infos = d["infos"] if isinstance(d, dict) and "infos" in d else d
    n = len(infos)
    if n == 0:
        raise RuntimeError("Empty infos in pkl")

    k = min(args.num_samples, n)
    random.seed(args.seed)
    indices = sorted(random.sample(range(n), k=k))

    try:
        font = ImageFont.truetype("arial.ttf", args.font_size)
    except Exception:
        font = ImageFont.load_default()

    for out_i, idx in enumerate(indices):
        info = infos[idx]
        cams: Dict[str, dict] = info["cams"]

        gt_boxes = np.array(info.get("gt_boxes", []), dtype=np.float32)  # (M,7)
        gt_names = [str(x) for x in info.get("gt_names", [])]
        valid_flag = info.get("valid_flag", None)
        if valid_flag is None:
            valid_mask = np.ones((len(gt_names),), dtype=bool)
        else:
            valid_mask = np.array(valid_flag).astype(bool)

        # simple 2x3 grid
        imgs = []
        for cam_type in CAM_TYPES:
            cam = cams[cam_type]
            img_path = safe_join(data_root, cam["data_path"])
            img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img)

            K = np.array(cam["cam_intrinsic"], dtype=np.float32).reshape(3, 3)
            R_l2c, t_l2c = lidar_to_cam_matrix(cam["sensor2lidar_rotation"], cam["sensor2lidar_translation"])

            rendered = 0
            for b, name, ok in zip(gt_boxes, gt_names, valid_mask):
                if not ok:
                    continue
                if rendered >= args.max_boxes:
                    break
                x, y, z, dx, dy, dz, yaw = [float(v) for v in b.tolist()]
                corners_l = box_corners_lidar(x, y, z, dx, dy, dz, yaw)  # (3,8)
                corners_c = (R_l2c @ corners_l) + t_l2c  # (3,8)
                # only render boxes mostly in front
                if float(np.mean(corners_c[2])) <= 0.5:
                    continue
                pts2d = project(corners_c, K)  # (2,8)
                color = CLASSNAME_TO_COLOR.get(name, (0, 255, 0))
                draw_box_3d(draw, pts2d, color=color, width=2)
                rendered += 1

            # title
            draw.rectangle([(0, 0), (img.size[0], 28)], fill=(0, 0, 0))
            draw.text((6, 4), f"{cam_type} | idx={idx}", fill=(255, 255, 255), font=font)
            imgs.append(img)

        # compose
        w = max(im.size[0] for im in imgs)
        h = max(im.size[1] for im in imgs)
        canvas = Image.new("RGB", (w * 3, h * 2), (20, 20, 20))
        for i, im in enumerate(imgs):
            r = i // 3
            c = i % 3
            canvas.paste(im, (c * w, r * h))

        out_path = os.path.join(args.out_dir, f"val_gt_{out_i:04d}_idx{idx}.jpg")
        canvas.save(out_path, quality=92)

    print(f"Saved {k} visualizations to: {args.out_dir}")


if __name__ == "__main__":
    main()

