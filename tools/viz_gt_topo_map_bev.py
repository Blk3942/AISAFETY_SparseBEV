#!/usr/bin/env python3
"""
GT-only visualization: six camera views + BEV with optional nuScenes map topology.

Compared to viz_full_bev_cam.py:
- No predictions (ground truth boxes only).
- Layout: left 2x3 cameras, single BEV on the right.
- BEV 底图：HD **basemap** PNG（maps/basemap 或 maps/expansion/basemap）+ **expansion 矢量** 栅格化层叠（默认约 38% basemap + 62% 拓扑）。
- 支持非标准布局：`maps/expansion/expansion/<map>.json` 与 `maps/expansion/basemap/<map>.png`（见 --map_dataroot / --basemap_dir）。
- GT merged into macros: Motor / VRU+Bicycle / Barrier.
- BEV：圆环距离为主参照；△ 为朝向，黄色箭头为平面速度（长度∝速率）；拓扑图层独立图例。

Depends: pymysql, matplotlib, numpy, pyquaternion, nuscenes (map_expansion), opencv-python (via nuscenes map).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymysql
import pymysql.cursors
from matplotlib.gridspec import GridSpecFromSubplotSpec, GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.image as mpimg

# nuScenes map (optional JSON under dataroot/maps/expansion/)
try:
    from nuscenes.map_expansion.map_api import NuScenesMap, locations as MAP_LOCATIONS

    HAS_MAP_EXPANSION = True
except ImportError:
    NuScenesMap = None  # type: ignore
    MAP_LOCATIONS = []
    HAS_MAP_EXPANSION = False

# patch_angle 与 nuScenes prediction/static_layers 一致（见文件后部 ego_patch_angle_deg）
try:
    from pyquaternion import Quaternion
    from nuscenes.eval.common.utils import quaternion_yaw
    from nuscenes.prediction.helper import angle_of_rotation

    HAS_MAP_ANGLE_PIPELINE = True
except ImportError:
    Quaternion = None  # type: ignore
    quaternion_yaw = None  # type: ignore

    angle_of_rotation = None  # type: ignore
    HAS_MAP_ANGLE_PIPELINE = False

# ── Module-level caches（批量渲染时避免重复加载）────────────────────────────────
_NUSC_CACHE: Dict[tuple, object] = {}      # (version, dataroot) → NuScenes
_MAP_OBJ_CACHE: Dict[str, object] = {}    # map_name → NuScenesMap
_EXPANSION_DISK_CACHE: Dict[str, Dict] = {}  # maps_root_base → avail dict


# ─────────────────────────── Camera layout & macro classes ───────────────────
CAM_ROWS = [
    ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
    ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"],
]

# nuScenes category -> macro for coloring
MACRO_MOTOR = "motor"
MACRO_VRU = "vru_ped_cycle"
MACRO_BARRIER = "barrier_macro"
MACRO_OTHER = "other_macro"

MACRO_COLOR = {
    MACRO_MOTOR: "#4FC3F7",
    MACRO_VRU: "#EF5350",
    MACRO_BARRIER: "#E040FB",   # 亮紫/品红，与灰色背景对比度明显
    MACRO_OTHER: "#B8860B",
}

MOTOR_CLASSES = frozenset(
    {
        "vehicle.car",
        "vehicle.truck",
        "vehicle.bus.rigid",
        "vehicle.bus.bendy",
        "vehicle.trailer",
        "vehicle.construction",
        "vehicle.motorcycle",
    }
)
VRU_CLASSES = frozenset({"human.pedestrian.adult", "human.pedestrian.child", "vehicle.bicycle"})
BARRIER_CLASSES = frozenset({"movable_object.barrier", "movable_object.trafficcone"})


def nusc_cls_to_macro(name: str) -> str:
    n = name or ""
    if n in MOTOR_CLASSES:
        return MACRO_MOTOR
    if n in VRU_CLASSES:
        return MACRO_VRU
    if n in BARRIER_CLASSES:
        return MACRO_BARRIER
    return MACRO_OTHER


# nuScenesMap.get_map_mask 图层顺序与配色（与 build_map_underlay_rgb 一致，用于 BEV 拓扑图例）
TOPO_LAYER_SPEC: List[Tuple[str, Tuple[float, float, float]]] = [
    ("Drivable area", (166 / 255, 206 / 255, 227 / 255)),
    ("Lane", (51 / 255, 160 / 255, 44 / 255)),
    ("Walkway", (227 / 255, 138 / 255, 138 / 255)),
    ("Ped crossing", (251 / 255, 180 / 255, 120 / 255)),
    ("Road divider", (175 / 255, 174 / 255, 255 / 255)),
    ("Lane divider", (106 / 255, 61 / 255, 154 / 255)),
]

# BEV：全局平面速度矢量 → map patch 平面（与 global_xyz_to_map_patch_xy 对位移的旋转一致，不含平移）
VEL_ARROW_COLOR = "#E8D060"
VEL_SCALE_M_PER_MS = 2.0  # 图上 1 m/s 对应若干米长
MAX_VEL_ARROW_M = 45.0

BOX_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]
FRONT_EDGE_SET = frozenset(
    [(0, 1), (1, 0), (0, 4), (4, 0), (1, 5), (5, 1), (4, 5), (5, 4)]
)


def quat_to_rot(w: float, x: float, y: float, z: float) -> np.ndarray:
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def yaw_from_box_quaternion(rw: float, rx: float, ry: float, rz: float) -> float:
    return math.atan2(
        2.0 * (rw * rz + rx * ry), 1.0 - 2.0 * (ry * ry + rz * rz)
    )


def nu_correct_yaw(yaw: float) -> float:
    """与 nuscenes.prediction.input_representation.static_layers.correct_yaw 相同。"""
    if yaw <= 0:
        return -np.pi - yaw
    return np.pi - yaw


def ego_patch_angle_deg(ego: Dict) -> float:
    """nuScenes get_map_mask 的 patch_angle（度），与 prediction StaticLayerRasterizer 一致。"""
    if HAS_MAP_ANGLE_PIPELINE and Quaternion is not None:
        q = Quaternion(
            float(ego["rot_w"]),
            float(ego["rot_x"]),
            float(ego["rot_y"]),
            float(ego["rot_z"]),
        )
        y = quaternion_yaw(q)
        y = nu_correct_yaw(y)
        return float(np.rad2deg(angle_of_rotation(y)))
    return math.degrees(
        yaw_from_box_quaternion(ego["rot_w"], ego["rot_x"], ego["rot_y"], ego["rot_z"])
    )


def box_corners_global(tx, ty, tz, w, l, h, rw, rx, ry, rz):
    """w,l,h = nuScenes size order; corners in global coords (N, 3)."""
    R = quat_to_rot(rw, rx, ry, rz)
    local = np.array(
        [
            [l / 2, w / 2, h / 2],
            [l / 2, -w / 2, h / 2],
            [-l / 2, -w / 2, h / 2],
            [-l / 2, w / 2, h / 2],
            [l / 2, w / 2, -h / 2],
            [l / 2, -w / 2, -h / 2],
            [-l / 2, -w / 2, -h / 2],
            [-l / 2, w / 2, -h / 2],
        ],
        dtype=np.float64,
    )
    return (R @ local.T).T + np.array([tx, ty, tz])


def ego_to_vehicle_points(ego: Dict, xyz_global: np.ndarray) -> np.ndarray:
    """Global -> ego vehicle axes (NuScenes: same convention as viz_full_bev_cam R_ego.T @ (p - t))."""
    t = np.array([ego["pos_x"], ego["pos_y"], ego["pos_z"]], dtype=np.float64)
    R_ego = quat_to_rot(ego["rot_w"], ego["rot_x"], ego["rot_y"], ego["rot_z"])
    return ((R_ego.T @ ((xyz_global - t).T)).T).astype(np.float64)


def global_xyz_to_map_patch_xy(xyz: np.ndarray, ego: Dict, patch_angle_deg: float) -> np.ndarray:
    """
    全局 XYZ → 与 NuScenesMap.get_map_mask(patch_box, patch_angle_deg, ...) 栅格一致的 2D patch 坐标（米）。
    这是 devkit 里对地图几何先做「绕 ego 的 -patch 旋转」再平移到 ego 原点后的坐标系；
    与 local_patch_to_global(lx, ly, ...) 互为逆变换，必须与 build_map_underlay_rgb / basemap 采样使用同一 patch_angle_deg。
    """
    arr = np.asarray(xyz, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 2:
        raise ValueError("global_xyz_to_map_patch_xy need at least x,y")
    ex, ey = float(ego["pos_x"]), float(ego["pos_y"])
    dx = arr[:, 0] - ex
    dy = arr[:, 1] - ey
    th = np.deg2rad(float(patch_angle_deg))
    c, s = np.cos(th), np.sin(th)
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    return np.stack([lx, ly], axis=-1)


def global_xy_vec_to_map_patch(vx: float, vy: float, patch_angle_deg: float) -> Tuple[float, float]:
    """全局 XY 矢量（如速度、朝向）旋转到 map patch XY，与位移变换使用的旋转矩阵一致。"""
    th = math.radians(float(patch_angle_deg))
    c, s = math.cos(th), math.sin(th)
    return c * vx + s * vy, -s * vx + c * vy


def camera_frustum_bev(
    cam_t: np.ndarray,
    cam_R: np.ndarray,
    K: np.ndarray,
    W: int,
    ego: Dict,
    patch_angle_deg: float,
    max_range: float = 70.0,
    n_pts: int = 28,
) -> Optional[np.ndarray]:
    """
    计算单个相机在 BEV map patch 平面内的视野扇区多边形（投影到地面 z=0 ego平面）。
    cam_t: (3,) 相机在 ego 系的平移；cam_R: (3,3) camera→ego 旋转；K: (3,3) 内参。
    返回 (M, 2) map patch 坐标，或 None（当没有有效地面交点时）。
    """
    fx, cx = float(K[0, 0]), float(K[0, 2])
    ang_l = math.atan2(-cx, fx)
    ang_r = math.atan2(float(W) - cx, fx)
    angles = np.linspace(ang_l, ang_r, n_pts)

    R_ego = quat_to_rot(ego["rot_w"], ego["rot_x"], ego["rot_y"], ego["rot_z"])
    t_ego_g = np.array([ego["pos_x"], ego["pos_y"], ego["pos_z"]])
    cam_h = float(cam_t[2])  # 相机在 ego 系的高度

    pts_g = []
    for a in angles:
        ray_cam = np.array([math.tan(a), 0.0, 1.0])
        ray_ego = cam_R @ ray_cam  # direction in ego frame
        # 与 ego z=0 平面相交：cam_t[2] + t * ray_ego[2] = 0
        if ray_ego[2] < -1e-4:
            t_param = min(-cam_h / ray_ego[2], max_range * 1.5)
        else:
            # 向上或平行——截断到水平最大距离
            horiz = math.hypot(ray_ego[0], ray_ego[1]) or 1e-6
            t_param = max_range / horiz
        pt_ego = cam_t + t_param * ray_ego
        pt_ego_z0 = np.array([pt_ego[0], pt_ego[1], 0.0])
        pts_g.append(R_ego @ pt_ego_z0 + t_ego_g)

    pts_arr = np.array(pts_g)
    # 相机原点（投影到 z=0）
    cam_ego_z0 = np.array([cam_t[0], cam_t[1], 0.0])
    cam_g = R_ego @ cam_ego_z0 + t_ego_g

    all_pts = np.vstack([cam_g.reshape(1, -1), pts_arr])
    return global_xyz_to_map_patch_xy(all_pts, ego, patch_angle_deg)


# 六个相机的视野扇区填充色与透明度
CAM_FRUSTUM_STYLE: Dict[str, Tuple[str, float]] = {
    "CAM_FRONT":       ("#60A8FF", 0.10),
    "CAM_FRONT_LEFT":  ("#A0C8FF", 0.08),
    "CAM_FRONT_RIGHT": ("#A0C8FF", 0.08),
    "CAM_BACK":        ("#FFB060", 0.10),
    "CAM_BACK_LEFT":   ("#FFD0A0", 0.08),
    "CAM_BACK_RIGHT":  ("#FFD0A0", 0.08),
}


def project_to_cam(
    pts_global: np.ndarray, ego: Dict, cam_t: np.ndarray, cam_R_mat: np.ndarray, K: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_ego = np.array([ego["pos_x"], ego["pos_y"], ego["pos_z"]])
    R_ego = quat_to_rot(ego["rot_w"], ego["rot_x"], ego["rot_y"], ego["rot_z"])
    p_ego = (R_ego.T @ (pts_global - t_ego).T).T
    p_cam = (cam_R_mat.T @ (p_ego - cam_t).T).T
    Z = p_cam[:, 2]
    Z_safe = np.where(np.abs(Z) > 1e-4, Z, 1e-4)
    u = K[0, 0] * p_cam[:, 0] / Z_safe + K[0, 2]
    v = K[1, 1] * p_cam[:, 1] / Z_safe + K[1, 2]
    return u, v, Z


# ─────────────────────────── Database ───────────────────
def mysql_connect(host: str, port: int, user: str, password: str, database: str):
    return pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )


def fetch_gt_cam_meta(conn, sample_id: int) -> Tuple[Optional[Dict], List, Dict, Dict]:
    cur = conn.cursor()
    cur.execute(
        """SELECT pos_x, pos_y, pos_z, rot_w, rot_x, rot_y, rot_z, yaw_rad, speed_mps
           FROM ground_truth_ego WHERE gt_sample_id=%s""",
        (sample_id,),
    )
    ego = cur.fetchone()

    cur.execute(
        """SELECT translation_x, translation_y, translation_z,
                  size_wlh_0, size_wlh_1, size_wlh_2,
                  rotation_w, rotation_x, rotation_y, rotation_z,
                  velocity_x, velocity_y,
                  class_name, visibility_level, ego_dist
           FROM ground_truth_box WHERE gt_sample_id=%s""",
        (sample_id,),
    )
    gt_boxes = cur.fetchall()

    cur.execute(
        """SELECT sc.channel, sc.filename, cs.translation AS cam_t, cs.rotation AS cam_r,
                  cs.camera_intrinsic AS cam_k, sc.width, sc.height
           FROM nuscenes_sample_camera sc
           JOIN nuscenes_calibrated_sensor cs ON cs.cs_token = sc.cs_token
           WHERE sc.gt_sample_id=%s""",
        (sample_id,),
    )
    cams = {}
    for r in cur.fetchall():
        cam_r_list = json.loads(r["cam_r"])
        cams[r["channel"]] = {
            "filename": r["filename"],
            "t": np.array(json.loads(r["cam_t"])),
            "R": quat_to_rot(*cam_r_list),
            "K": np.array(json.loads(r["cam_k"])),
            "W": r["width"] or 1600,
            "H": r["height"] or 900,
        }

    cur.execute(
        """SELECT gs.sample_token, ns.name, ns.log_location, ns.log_date
           FROM ground_truth_sample gs
           LEFT JOIN nuscenes_scene ns ON ns.scene_token = gs.scene_token
           WHERE gs.id=%s""",
        (sample_id,),
    )
    meta = cur.fetchone()
    cur.close()
    return ego, gt_boxes, cams, meta


def infer_map_location_nuscenes(dataroot: str, version: str, sample_token: Optional[str]) -> Optional[str]:
    if not sample_token:
        return None
    try:
        from nuscenes.nuscenes import NuScenes

        key = (version, dataroot)
        if key not in _NUSC_CACHE:
            _NUSC_CACHE[key] = NuScenes(version=version, dataroot=dataroot, verbose=False)
        nusc = _NUSC_CACHE[key]
        sc_tok = nusc.get("sample", sample_token)["scene_token"]
        log_tok = nusc.get("scene", sc_tok)["log_token"]
        loc = nusc.get("log", log_tok).get("location")
        return str(loc) if loc else None
    except Exception:
        return None


def resolve_map_name(meta: Dict, dataroot: str, version: str) -> Optional[str]:
    loc = meta.get("log_location") if meta else None
    if loc:
        ls = str(loc).strip().lower().replace("_", "-")
        for m in MAP_LOCATIONS:
            if m.lower() == ls:
                return m
    tk = meta.get("sample_token") if meta else None
    inferred = infer_map_location_nuscenes(dataroot, version, tk)
    if inferred:
        inl = inferred.lower().replace("_", "-")
        for m in MAP_LOCATIONS:
            if m.lower() == inl:
                return m
    return None


def expansion_maps_on_disk(maps_root_base: str) -> Dict[str, str]:
    """返回 {map_name: json路径}，支持 maps/expansion/*.json 与 maps/expansion/expansion/*.json。结果缓存。"""
    if maps_root_base in _EXPANSION_DISK_CACHE:
        return _EXPANSION_DISK_CACHE[maps_root_base]
    out: Dict[str, str] = {}
    base = Path(maps_root_base) / "maps" / "expansion"
    if not base.is_dir():
        _EXPANSION_DISK_CACHE[maps_root_base] = out
        return out
    for j in sorted(base.glob("*.json")):
        if j.stem in MAP_LOCATIONS:
            out[j.stem] = str(j)
    nested = base / "expansion"
    if nested.is_dir():
        for j in sorted(nested.glob("*.json")):
            if j.stem in MAP_LOCATIONS:
                out[j.stem] = str(j)
    _EXPANSION_DISK_CACHE[maps_root_base] = out
    return out


def resolve_expansion_json_path(maps_root_base: str, map_name: str) -> Optional[Path]:
    """优先嵌套目录 expansion/expansion（用户常见放置方式）。"""
    for rel in (
        Path("maps") / "expansion" / "expansion" / f"{map_name}.json",
        Path("maps") / "expansion" / f"{map_name}.json",
    ):
        p = Path(maps_root_base) / rel
        if p.is_file():
            return p
    return None


def resolve_basemap_png(maps_root_base: str, map_name: str, basemap_dir_override: Optional[str]) -> Optional[Path]:
    """查找 HD basemap PNG（与 BitMap basemap 同名）。"""
    candidates: List[Path] = []
    if basemap_dir_override:
        bo = Path(basemap_dir_override)
        if bo.is_dir():
            candidates.append(bo / f"{map_name}.png")
        elif bo.is_file():
            return bo
    root = Path(maps_root_base)
    candidates.extend(
        [
            root / "maps" / "expansion" / "basemap" / f"{map_name}.png",
            root / "maps" / "basemap" / f"{map_name}.png",
        ]
    )
    for p in candidates:
        if p.is_file():
            return p
    return None


def open_nuscenes_map_from_json(map_name: str, expansion_json: Path) -> "NuScenesMap":
    """
    NuScenesMap 仅接受 dataroot/maps/expansion/<map>.json；将任意路径的 json 复制到临时目录再加载。
    结果按 map_name 缓存，批量渲染时每个地图只加载一次。
    """
    cache_key = str(map_name)
    if cache_key in _MAP_OBJ_CACHE:
        return _MAP_OBJ_CACHE[cache_key]  # type: ignore[return-value]

    tmp = tempfile.mkdtemp(prefix="nusc_map_json_")
    try:
        dst_dir = Path(tmp) / "maps" / "expansion"
        dst_dir.mkdir(parents=True)
        shutil.copy2(expansion_json, dst_dir / f"{map_name}.json")
        obj = NuScenesMap(tmp, map_name)
        _MAP_OBJ_CACHE[cache_key] = obj
        return obj
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def local_patch_to_global(
    lx: np.ndarray, ly: np.ndarray, ego_x: float, ego_y: float, patch_angle_deg: float
) -> Tuple[np.ndarray, np.ndarray]:
    """与 get_map_mask 局部坐标到全局米制坐标一致（逆变换：R(+patch) @ local + ego）。"""
    th = np.deg2rad(np.asarray(patch_angle_deg, dtype=np.float64))
    c = np.cos(th)
    s = np.sin(th)
    gx = c * lx - s * ly + ego_x
    gy = s * lx + c * ly + ego_y
    return gx, gy


def raster_basemap_patch(
    basemap_rgb: np.ndarray,
    canvas_edge: Tuple[float, float],
    ego: Dict,
    half_span: float,
    out_hw: Tuple[int, int],
    patch_angle_deg: float,
) -> np.ndarray:
    """
    将整幅 basemap（与 nuScenes canvas_edge 对齐）采样到与拓扑 mask 相同的 ego patch 网格。
    basemap_rgb: Himg x Wimg x 3 uint8 或 float.
    """
    try:
        from scipy.ndimage import map_coordinates
    except ImportError:
        map_coordinates = None

    H, W = out_hw
    half_w = half_h = float(half_span)
    lx_a, rx = -half_w, half_w
    by_a, ty = -half_h, half_h
    ex, ey = float(ego["pos_x"]), float(ego["pos_y"])

    cols = np.linspace(lx_a + (rx - lx_a) / (2 * W), rx - (rx - lx_a) / (2 * W), W, dtype=np.float64)
    rows = np.linspace(by_a + (ty - by_a) / (2 * H), ty - (ty - by_a) / (2 * H), H, dtype=np.float64)
    LX, LY = np.meshgrid(cols, rows)
    gx, gy = local_patch_to_global(LX, LY, ex, ey, patch_angle_deg)

    cx_edge, cy_edge = float(canvas_edge[0]), float(canvas_edge[1])
    img_h, img_w = basemap_rgb.shape[0], basemap_rgb.shape[1]
    pu = gx / cx_edge * (img_w - 1)
    pv = (1.0 - gy / cy_edge) * (img_h - 1)

    if basemap_rgb.ndim == 2:
        basemap_rgb = np.stack([basemap_rgb, basemap_rgb, basemap_rgb], axis=-1)
    elif basemap_rgb.shape[2] == 4:
        basemap_rgb = basemap_rgb[:, :, :3]
    if basemap_rgb.dtype != np.float64:
        bm = basemap_rgb.astype(np.float64) / 255.0
    else:
        bm = np.clip(basemap_rgb, 0.0, 1.0)

    out = np.zeros((H, W, 3), dtype=np.float64)
    coords = np.stack([pv, pu])

    if map_coordinates is not None:
        for ch in range(min(3, bm.shape[2])):
            out[:, :, ch] = map_coordinates(bm[:, :, ch], coords, order=1, mode="constant", cval=0.0)
    else:
        # 极简双线性（无 scipy 时）
        pu_cl = np.clip(pu, 0, img_w - 1)
        pv_cl = np.clip(pv, 0, img_h - 1)
        i0 = np.floor(pv_cl).astype(np.int32)
        j0 = np.floor(pu_cl).astype(np.int32)
        i1 = np.clip(i0 + 1, 0, img_h - 1)
        j1 = np.clip(j0 + 1, 0, img_w - 1)
        wa = pv_cl - i0
        wb = pu_cl - j0
        for ch in range(min(3, bm.shape[2])):
            Ia = (1 - wb) * bm[i0, j0, ch] + wb * bm[i0, j1, ch]
            Ib = (1 - wb) * bm[i1, j0, ch] + wb * bm[i1, j1, ch]
            out[:, :, ch] = (1 - wa) * Ia + wa * Ib

    return np.clip(out, 0.0, 1.0)


def read_canvas_edge_from_json(json_path: Path) -> Optional[Tuple[float, float]]:
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
        ce = obj.get("canvas_edge")
        if ce and len(ce) >= 2:
            return float(ce[0]), float(ce[1])
    except Exception:
        pass
    return None


def pick_map_for_sample(
    meta: Dict, dataroot: str, version: str, maps_root_base: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    返回 (map_name, expansion_json_path)。
    优先级：log_location / NuScenes log → expansion 目录下单文件兜底。
    """
    avail = expansion_maps_on_disk(maps_root_base)
    if not avail:
        return None, None

    name = resolve_map_name(meta, dataroot, version)
    if name and name in avail:
        return name, avail[name]

    tk = meta.get("sample_token") if meta else None
    inferred = infer_map_location_nuscenes(dataroot, version, tk)
    if inferred:
        inl = inferred.lower().replace("_", "-")
        for m, path in avail.items():
            if m.lower() == inl:
                return m, path

    if len(avail) == 1:
        m = next(iter(avail.keys()))
        return m, avail[m]

    return None, None


def yaw_from_ego_pose(ego: Dict) -> float:
    """Global yaw around z (approx same convention as yaw_from_box_quaternion)."""
    return yaw_from_box_quaternion(ego["rot_w"], ego["rot_x"], ego["rot_y"], ego["rot_z"])


# ─────────────────────────── Draw ───────────────────
def draw_camera_view(ax, cam: Dict, boxes: List, ego: Dict):
    img = mpimg.imread(cam["filepath"])
    W, H = cam["W"], cam["H"]
    ax.imshow(img, extent=[0, W, H, 0], aspect="auto")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    for b in boxes:
        macro = nusc_cls_to_macro(str(b["class_name"]))
        col_hex = MACRO_COLOR[macro]

        corners = box_corners_global(
            b["translation_x"],
            b["translation_y"],
            b["translation_z"],
            b["size_wlh_0"],
            b["size_wlh_1"],
            b["size_wlh_2"],
            b["rotation_w"],
            b["rotation_x"],
            b["rotation_y"],
            b["rotation_z"],
        )
        u, v, Z = project_to_cam(corners, ego, cam["t"], cam["R"], cam["K"])
        in_front = Z > 0.2
        in_image = (u > -30) & (u < W + 30) & (v > -30) & (v < H + 30)
        visible = in_front & in_image
        if visible.sum() < 3:
            continue

        c_rgb = tuple(int(col_hex.strip("#")[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
        lw_main = 0.85 if macro != MACRO_OTHER else 0.65

        for i, j in BOX_EDGES:
            if Z[i] > 0.2 and Z[j] > 0.2:
                lw = lw_main + (0.22 if (i, j) in FRONT_EDGE_SET or (j, i) in FRONT_EDGE_SET else 0)
                ax.plot([u[i], u[j]], [v[i], v[j]], color=c_rgb, lw=lw, alpha=0.94)


def build_map_underlay_rgb(
    nusc_map: "NuScenesMap",
    ego: Dict,
    bev_range: float,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    patch_box = (float(ego["pos_x"]), float(ego["pos_y"]), 2 * bev_range, 2 * bev_range)
    layers = ["drivable_area", "lane", "walkway", "ped_crossing", "road_divider", "lane_divider"]
    px_m = 8
    canvas = (max(320, int(2 * bev_range * px_m)), max(320, int(2 * bev_range * px_m)))

    patch_angle_deg = ego_patch_angle_deg(ego)
    masks = np.asarray(nusc_map.get_map_mask(patch_box, patch_angle_deg, layers, canvas), dtype=np.float32)
    h, w = masks.shape[1], masks.shape[2]
    rgb = np.zeros((h, w, 3), dtype=np.float64)
    for i, (_, clr) in enumerate(TOPO_LAYER_SPEC):
        if i >= masks.shape[0]:
            break
        m = np.clip(masks[i], 0.0, 1.0)
        for c in range(3):
            rgb[:, :, c] += float(clr[c]) * m * 0.52
        rgb[:, :, :] = np.clip(rgb, 0.0, 1.0)
    # 不做 flipud：与 extent (-half_w,half_w)×(-half_h,half_h) 及 global_xyz_to_map_patch_xy 一致；
    # 旧版用车体坐标画 GT + flipud 会导致与地图行列/y 轴错位。
    half_w = patch_box[3] / 2.0
    half_h = patch_box[2] / 2.0
    extent_xy = (-half_w, half_w, -half_h, half_h)
    return rgb, extent_xy


def draw_polar_distance_guide(ax, r_max: float, ego_xy: Tuple[float, float]) -> None:
    """以自车为中心的等距圆环（主距离参照），带半径标注。"""
    step = max(10, int(r_max // 15)) if r_max > 50 else 10
    label_ang = math.pi / 5.0
    for r in range(step, int(r_max) + 1, step):
        rf = float(r)
        circ = plt.Circle(
            ego_xy,
            rf,
            color="#FF8C00",    # 橙色圆环，与蓝/绿拓扑底图及白色 ego 明显区分
            fill=False,
            ls="--",
            lw=1.1,
            alpha=0.88,
            zorder=1,
        )
        ax.add_patch(circ)
        tx = ego_xy[0] + (rf - step * 0.22) * math.cos(label_ang)
        ty = ego_xy[1] + (rf - step * 0.22) * math.sin(label_ang)
        ax.text(
            tx,
            ty,
            f"{r} m",
            fontsize=6.0,
            color="#FFB74D",
            ha="center",
            va="center",
            alpha=0.97,
            zorder=2,
            clip_on=True,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#07070F", edgecolor="none", alpha=0.65),
        )


# ─────────────────────────── Legend panel ─────────────────────────────────────
MACRO_LABEL = {
    MACRO_MOTOR: "Motor vehicles",
    MACRO_VRU: "Pedestrian / Bicycle",
    MACRO_BARRIER: "Barrier / Cone",
    MACRO_OTHER: "Other",
}


def _leg_title(ax, x: float, y: float, text: str, fs: float = 7.2) -> float:
    """在 axes 分数坐标 (x, y) 处写节标题，返回下一行 y。"""
    ax.text(x, y, text, transform=ax.transAxes, color="#AABBCC",
            fontsize=fs, fontweight="bold", ha="center", va="top")
    return y - 0.064


def _leg_row(ax, xp: float, xt: float, y: float, color: str, label: str, fs: float = 6.2) -> float:
    """在 axes 分数坐标中画色块 + 文字，xp=色块左边，xt=文字左边。"""
    rect = plt.Rectangle((xp, y - 0.024), 0.082, 0.030,
                          transform=ax.transAxes, facecolor=color,
                          edgecolor="#555", linewidth=0.5, clip_on=False)
    ax.add_patch(rect)
    ax.text(xt, y - 0.009, label, transform=ax.transAxes, color="#DDD",
            fontsize=fs, va="center", ha="left")
    return y - 0.053


def draw_legend_panel(ax, ego_speed_mps: float) -> None:
    """
    双列紧凑布局：
      左列 (0–0.48)  : Map Topology (6)
      右列 (0.52–1)  : Object Class (4) + Symbols (3)
      分隔线以下全宽  : Ego Speed + Speed Scale
    """
    ax.set_facecolor("#0a0a16")
    ax.axis("off")

    # ── 左列：Map Topology ────────────────────────────────────────────────────
    LX_CTR = 0.25          # 左列标题中心 x
    LX_P   = 0.03          # 左列色块左边 x
    LX_T   = 0.14          # 左列文字左边 x
    y_l = 0.97
    y_l = _leg_title(ax, LX_CTR, y_l, "Map Topology")
    for name, clr in TOPO_LAYER_SPEC:
        col_hex = "#{:02X}{:02X}{:02X}".format(
            int(clr[0] * 255), int(clr[1] * 255), int(clr[2] * 255)
        )
        y_l = _leg_row(ax, LX_P, LX_T, y_l, col_hex, name)

    # ── 右列：Object Class + Symbols ─────────────────────────────────────────
    RX_CTR = 0.76
    RX_P   = 0.54
    RX_T   = 0.65
    y_r = 0.97
    y_r = _leg_title(ax, RX_CTR, y_r, "Object Class")
    for macro, col in MACRO_COLOR.items():
        y_r = _leg_row(ax, RX_P, RX_T, y_r, col, MACRO_LABEL[macro])

    y_r -= 0.012
    y_r = _leg_title(ax, RX_CTR, y_r, "Symbols")

    # 三角（朝向）
    tx_c = RX_P + 0.038
    ty_s = y_r - 0.015
    tri = plt.Polygon(
        [[tx_c - 0.035, ty_s - 0.016], [tx_c + 0.035, ty_s - 0.016], [tx_c, ty_s + 0.016]],
        transform=ax.transAxes, facecolor="#AAA", edgecolor="#888", lw=0.5, clip_on=False,
    )
    ax.add_patch(tri)
    ax.text(RX_T, ty_s, "Heading", transform=ax.transAxes,
            color="#DDD", fontsize=6.2, va="center", ha="left")
    y_r -= 0.053

    # 黄箭头（速度）
    ax.annotate("", xy=(RX_P + 0.098, y_r - 0.013), xytext=(RX_P + 0.002, y_r - 0.013),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=VEL_ARROW_COLOR, lw=1.4, mutation_scale=7))
    ax.text(RX_T, y_r - 0.013, "Velocity", transform=ax.transAxes,
            color="#DDD", fontsize=6.2, va="center", ha="left")
    y_r -= 0.053

    # 白框（自车）
    rect_e = plt.Rectangle((RX_P, y_r - 0.020), 0.082, 0.028,
                            transform=ax.transAxes, facecolor="none",
                            edgecolor="#FFFFFF", linewidth=1.3, clip_on=False)
    ax.add_patch(rect_e)
    ax.text(RX_T, y_r - 0.006, "Ego vehicle", transform=ax.transAxes,
            color="#DDD", fontsize=6.2, va="center", ha="left")
    y_r -= 0.053

    # ── 分隔线（取两列中最低的 y）─────────────────────────────────────────────
    y_sep = min(y_l, y_r) - 0.020
    ax.plot([0.02, 0.98], [y_sep, y_sep], color="#334", linewidth=0.8,
            transform=ax.transAxes, clip_on=False)
    y = y_sep - 0.025

    # ── Ego Speed（全宽居中）─────────────────────────────────────────────────
    y = _leg_title(ax, 0.5, y, "Ego Speed", fs=7.2)
    ego_kmh = ego_speed_mps * 3.6
    ax.text(0.5, y - 0.004, f"{ego_speed_mps:.2f} m/s",
            transform=ax.transAxes, color="#FFFFFF",
            fontsize=10, fontweight="bold", ha="center", va="top")
    y -= 0.055
    ax.text(0.5, y, f"({ego_kmh:.1f} km/h)",
            transform=ax.transAxes, color="#AAA",
            fontsize=7.5, ha="center", va="top")
    y -= 0.050

    # ── Speed Scale（全宽）──────────────────────────────────────────────────
    ax.plot([0.02, 0.98], [y, y], color="#334", linewidth=0.8,
            transform=ax.transAxes, clip_on=False)
    y -= 0.022
    y = _leg_title(ax, 0.5, y, "Speed Scale", fs=7.0)
    ax.text(0.5, y, f"1 m/s = {VEL_SCALE_M_PER_MS:.0f} m on map",
            transform=ax.transAxes, color="#888", fontsize=6.0, ha="center", va="top")
    y -= 0.040

    ref_speeds_kmh = [10, 30, 60]
    max_ms = ref_speeds_kmh[-1] / 3.6
    x_a0, x_a1 = 0.04, 0.56   # 箭头起止（左端固定，右端为60km/h对应）
    for spd_kmh in ref_speeds_kmh:
        spd_ms = spd_kmh / 3.6
        xe = x_a0 + (x_a1 - x_a0) * (spd_ms / max_ms)
        ax.annotate("", xy=(xe, y - 0.013), xytext=(x_a0, y - 0.013),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=VEL_ARROW_COLOR,
                                    lw=1.3, mutation_scale=7))
        ax.text(xe + 0.03, y - 0.013,
                f"{spd_kmh} km/h ({spd_ms:.1f} m/s)",
                transform=ax.transAxes, color="#E8D060",
                fontsize=6.0, va="center", ha="left")
        y -= 0.048


def draw_bev(
    ax,
    gt_boxes: List,
    ego: Dict,
    bev_range: float,
    underlay_rgb: Optional[np.ndarray],
    map_extent_xy: Optional[Tuple[float, float, float, float]],
    patch_angle_deg: float,
    cams: Optional[Dict] = None,
) -> None:
    ax.set_facecolor("#07070F")
    ax.set_aspect("equal")
    ego_xy = (0.0, 0.0)

    lx, rx, by, ty = (-bev_range, bev_range, -bev_range, bev_range)
    if underlay_rgb is not None and map_extent_xy is not None:
        lx, rx, by, ty = map_extent_xy
        ax.imshow(
            underlay_rgb,
            origin="lower",
            extent=(lx, rx, by, ty),
            zorder=0,
            interpolation="bilinear",
            alpha=0.88,
        )

    ax.set_xlim(lx, rx)
    ax.set_ylim(by, ty)
    ax.grid(False)

    ring_max = min(float(bev_range), max(abs(lx), abs(rx), abs(by), abs(ty)))
    draw_polar_distance_guide(ax, ring_max, ego_xy)

    ax.set_xlabel("Map patch X (m)", color="#9aa", fontsize=7)
    ax.set_ylabel("Map patch Y (m)", color="#9aa", fontsize=7)
    ax.tick_params(colors="#778", labelsize=7)
    ax.set_title("BEV", color="#CCD", fontsize=9, pad=6)

    # ── 相机视野扇区（zorder=3，在底图之上、GT之下）─────────────────────────
    if cams:
        for cam_name, cam in cams.items():
            style = CAM_FRUSTUM_STYLE.get(cam_name)
            if style is None:
                continue
            col_f, alpha_f = style
            try:
                poly_pts = camera_frustum_bev(
                    cam["t"], cam["R"], cam["K"], int(cam["W"]),
                    ego, patch_angle_deg, max_range=min(bev_range * 0.9, 70.0),
                )
                if poly_pts is not None and len(poly_pts) >= 3:
                    fp = plt.Polygon(
                        poly_pts, closed=True,
                        facecolor=col_f, edgecolor=col_f,
                        lw=0.5, alpha=alpha_f, zorder=3,
                    )
                    ax.add_patch(fp)
            except Exception:
                pass

    # Ego 车体轮廓：车体系角点 → 全局 → map patch（与底图同一坐标系）
    hw_e, hl_e = 1.0, 2.25
    ego_corners_v = np.array(
        [[-hl_e, -hw_e, 0.0], [hl_e, -hw_e, 0.0], [hl_e, hw_e, 0.0], [-hl_e, hw_e, 0.0]], dtype=np.float64
    )
    R_ego = quat_to_rot(ego["rot_w"], ego["rot_x"], ego["rot_y"], ego["rot_z"])
    t_ego = np.array([ego["pos_x"], ego["pos_y"], ego["pos_z"]], dtype=np.float64)
    ego_corners_g = (R_ego @ ego_corners_v.T).T + t_ego
    ego_xy_map = global_xyz_to_map_patch_xy(ego_corners_g, ego, patch_angle_deg)
    ego_poly = plt.Polygon(
        ego_xy_map[:, :2], closed=True, facecolor="none", edgecolor="#FFFFFF", lw=2.2, alpha=0.98, zorder=9,
    )
    ax.add_patch(ego_poly)

    # 自车朝向（车头小箭头）+ 速度矢量
    yaw_ego = yaw_from_box_quaternion(ego["rot_w"], ego["rot_x"], ego["rot_y"], ego["rot_z"])
    ex_ux, ex_uy = global_xy_vec_to_map_patch(math.cos(yaw_ego), math.sin(yaw_ego), patch_angle_deg)
    en = math.hypot(ex_ux, ex_uy) or 1.0
    ex_ux, ex_uy = ex_ux / en, ex_uy / en
    # 车头朝向白色小箭头
    ax.annotate(
        "",
        xy=(hl_e * 0.85 * ex_ux, hl_e * 0.85 * ex_uy),
        xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="->", color="#FFFFFF", lw=1.6, mutation_scale=10, shrinkA=0, shrinkB=0),
        zorder=10,
    )

    # 自车速度矢量：沿自车朝向画黄色 quiver，长度 = speed × VEL_SCALE_M_PER_MS
    ego_speed = float(ego.get("speed_mps") or 0.0)
    if ego_speed >= 0.1:
        ego_alen = min(MAX_VEL_ARROW_M, ego_speed * VEL_SCALE_M_PER_MS)
        ego_dux, ego_duy = ego_alen * ex_ux, ego_alen * ex_uy
        ax.quiver(
            0.0, 0.0, ego_dux, ego_duy,
            color=VEL_ARROW_COLOR,
            angles="xy", scale_units="xy", scale=1,
            width=0.006, headwidth=4.5, headlength=5.5, headaxislength=4.5,
            alpha=0.98, zorder=11,
        )

    for b in gt_boxes:
        macro = nusc_cls_to_macro(str(b["class_name"]))
        col = MACRO_COLOR[macro]
        corners_g = box_corners_global(
            b["translation_x"],
            b["translation_y"],
            b["translation_z"],
            b["size_wlh_0"],
            b["size_wlh_1"],
            b["size_wlh_2"],
            b["rotation_w"],
            b["rotation_x"],
            b["rotation_y"],
            b["rotation_z"],
        )
        bot_g = corners_g[4:8]
        bot_map = global_xyz_to_map_patch_xy(bot_g, ego, patch_angle_deg)

        cen_map = global_xyz_to_map_patch_xy(
            np.array([[float(b["translation_x"]), float(b["translation_y"]), float(b["translation_z"])]]),
            ego,
            patch_angle_deg,
        )[0]
        bx, by = float(cen_map[0]), float(cen_map[1])

        lw = 0.78 if math.hypot(bx, by) <= 50 else 0.62
        poly = plt.Polygon(bot_map[:, :2].astype(float), closed=True, fill=False, edgecolor=col, linewidth=lw, alpha=0.95, zorder=5)
        ax.add_patch(poly)

        if quaternion_yaw is not None:
            yaw_b = float(
                quaternion_yaw(
                    Quaternion(float(b["rotation_w"]), float(b["rotation_x"]), float(b["rotation_y"]), float(b["rotation_z"]))
                )
            )
        else:
            yaw_b = yaw_from_box_quaternion(b["rotation_w"], b["rotation_x"], b["rotation_y"], b["rotation_z"])
        dir_gx, dir_gy = math.cos(yaw_b), math.sin(yaw_b)
        hx, hy = global_xy_vec_to_map_patch(dir_gx, dir_gy, patch_angle_deg)
        h_norm = math.hypot(hx, hy) or 1.0
        ux, uy = hx / h_norm, hy / h_norm

        sz_m = max(float(b["size_wlh_0"]), float(b["size_wlh_1"]))
        tri_len = min(3.8, max(0.55, 0.42 * sz_m))
        tip_x, tip_y = bx + tri_len * ux, by + tri_len * uy
        base_cx, base_cy = bx - 0.32 * tri_len * ux, by - 0.32 * tri_len * uy
        px, py = -uy, ux
        half_w = 0.36 * tri_len
        tri_xy = np.array(
            [
                [tip_x, tip_y],
                [base_cx + half_w * px, base_cy + half_w * py],
                [base_cx - half_w * px, base_cy - half_w * py],
            ],
            dtype=np.float64,
        )
        tri = plt.Polygon(tri_xy, closed=True, facecolor=col, edgecolor=col, lw=0.35, alpha=0.92, zorder=6)
        ax.add_patch(tri)

        _vx_raw = b.get("velocity_x")
        _vy_raw = b.get("velocity_y")
        try:
            vx_g = float(_vx_raw) if _vx_raw is not None else 0.0
            vy_g = float(_vy_raw) if _vy_raw is not None else 0.0
            if math.isnan(vx_g) or math.isnan(vy_g):
                vx_g = vy_g = 0.0
        except (TypeError, ValueError):
            vx_g = vy_g = 0.0
        speed = math.hypot(vx_g, vy_g)
        if speed >= 0.08:
            vpx, vpy = global_xy_vec_to_map_patch(vx_g, vy_g, patch_angle_deg)
            sp = math.hypot(vpx, vpy) or 1e-6
            alen = min(MAX_VEL_ARROW_M, speed * VEL_SCALE_M_PER_MS)
            dux, duy = alen * vpx / sp, alen * vpy / sp
            # 箭头从三角顶点出发，避免与朝向三角形重叠
            ax.quiver(
                tip_x, tip_y, dux, duy,
                color=VEL_ARROW_COLOR,
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.004,
                headwidth=4.5,
                headlength=5.5,
                headaxislength=4.5,
                alpha=0.95,
                zorder=7,
            )


def render_figure(conn, sample_id: int, args: argparse.Namespace) -> bool:
    ego, gt_boxes, cams, meta = fetch_gt_cam_meta(conn, sample_id)
    if ego is None or meta is None or len(cams) < 6:
        print(f"  [skip {sample_id}] missing ego/meta/cameras")
        return False

    dataroot = args.dataroot.rstrip("\\/")
    _map_root_opt = getattr(args, "map_dataroot", None)
    maps_root_base = (_map_root_opt or dataroot).rstrip("\\/")

    topo_rgb = None
    extent = None
    map_name: Optional[str] = args.map_name_override
    map_obj = None
    path_map_json: Optional[Path] = None

    if HAS_MAP_EXPANSION:
        if map_name and map_name in MAP_LOCATIONS:
            path_map_json = resolve_expansion_json_path(maps_root_base, map_name)
            if path_map_json is None:
                print(f"  Note: --map_name_override: no expansion JSON for map={map_name}")
                map_name = None

        if path_map_json is None:
            picked_name, picked_path = pick_map_for_sample(meta, dataroot, args.nuscenes_version, maps_root_base)
            if picked_name and picked_path:
                map_name = picked_name
                path_map_json = Path(picked_path)

        if path_map_json is not None and path_map_json.is_file():
            try:
                map_obj = open_nuscenes_map_from_json(str(map_name), path_map_json)
                topo_rgb, extent = build_map_underlay_rgb(map_obj, ego, args.bev_range)
                print(f"  Map expansion topology: {path_map_json}")
            except Exception as e:
                print(f"  Warning: failed to rasterize map topology ({e}); topology skipped.")
                topo_rgb, extent = None, None
                map_obj = None
        else:
            if expansion_maps_on_disk(maps_root_base):
                print(
                    "  Note: could not match scene to a map JSON "
                    f"(try --map_name_override). Search under: "
                    f"{Path(maps_root_base) / 'maps' / 'expansion'}"
                )
            else:
                print(f"  Note: no expansion JSON under maps/expansion — distance rings only.")
    else:
        print("  Note: nuScenes map_expansion not importable — distance rings only.")

    # HD basemap PNG（与 canvas_edge 对齐），采样到与拓扑相同的 ego patch
    basemap_patch = None
    patch_angle_deg = ego_patch_angle_deg(ego)
    if map_name:
        bmp_path = resolve_basemap_png(
            maps_root_base,
            map_name,
            getattr(args, "basemap_dir", None),
        )
        if bmp_path is not None:
            try:
                from PIL import Image

                bm_full = np.array(Image.open(bmp_path).convert("RGB"))
                ce = None
                if map_obj is not None:
                    ce = tuple(float(x) for x in map_obj.canvas_edge)
                elif path_map_json is not None:
                    ce = read_canvas_edge_from_json(path_map_json)
                if ce is not None:
                    out_hw = (
                        int(topo_rgb.shape[0]) if topo_rgb is not None else max(320, int(2 * args.bev_range * 8)),
                        int(topo_rgb.shape[1]) if topo_rgb is not None else max(320, int(2 * args.bev_range * 8)),
                    )
                    basemap_patch = raster_basemap_patch(
                        bm_full,
                        ce,
                        ego,
                        args.bev_range,
                        out_hw,
                        patch_angle_deg,
                    )
                    print(f"  Basemap raster: {bmp_path}")
            except Exception as e:
                print(f"  Warning: basemap overlay skipped ({e}).")

    underlay_rgb = None
    if topo_rgb is not None and basemap_patch is not None:
        underlay_rgb = np.clip(0.38 * basemap_patch + 0.62 * topo_rgb, 0.0, 1.0)
    elif topo_rgb is not None:
        underlay_rgb = topo_rgb
    elif basemap_patch is not None:
        underlay_rgb = basemap_patch
        if extent is None:
            half = float(args.bev_range)
            extent = (-half, half, -half, half)

    for ch in cams:
        cams[ch]["filepath"] = f"{dataroot}/{cams[ch]['filename']}"

    fig = plt.figure(figsize=(30.0, 8.8), facecolor="#07070F")
    # 三列：摄像头 | BEV | 图例面板（加宽，双列布局）
    outer = GridSpec(1, 3, figure=fig, width_ratios=[2.72, 1.05, 0.72], wspace=0.04)

    left = GridSpecFromSubplotSpec(
        2, 3, subplot_spec=outer[0], wspace=0.03, hspace=0.06,
    )

    order = (*CAM_ROWS[0], *CAM_ROWS[1])
    for ii, cam_name in enumerate(order):
        r, c = divmod(ii, 3)
        ax = fig.add_subplot(left[r, c])
        short = cam_name.replace("CAM_", "").replace("_", " ")
        if cam_name in cams:
            draw_camera_view(ax, cams[cam_name], gt_boxes, ego)
            ax.set_title(short, fontsize=8, pad=4, color="#BBC")

    ax_bev = fig.add_subplot(outer[1])
    draw_bev(ax_bev, gt_boxes, ego, args.bev_range, underlay_rgb, extent, patch_angle_deg, cams=cams)

    ax_leg = fig.add_subplot(outer[2])
    ego_speed = float(ego.get("speed_mps") or 0.0)
    draw_legend_panel(ax_leg, ego_speed)

    # ── 标题：主标题 + 场景信息副标题 ───────────────────────────────────────
    title_main = f"GT BEV Visualization  |  sample id = {sample_id}"
    loc_str  = meta.get("log_location") or ""
    date_str = str(meta.get("log_date") or "")
    tok_str  = str(meta.get("sample_token") or "")[:20]
    parts = []
    if map_name:
        parts.append(f"map: {map_name}")
    if loc_str:
        parts.append(f"location: {loc_str}")
    if date_str:
        parts.append(f"date: {date_str}")
    if tok_str:
        parts.append(f"token: {tok_str}...")
    subtitle = "   |   ".join(parts)
    fig.suptitle(title_main, color="#DDE", fontsize=11, y=0.993)
    if subtitle:
        fig.text(0.5, 0.968, subtitle, color="#7799AA", fontsize=7.5,
                 ha="center", va="top")

    plt.savefig(args.out, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def pick_random_id(conn, seed: int) -> int:
    cur = conn.cursor()
    cur.execute("SELECT id FROM ground_truth_sample")
    ids = [int(r["id"]) for r in cur.fetchall()]
    cur.close()
    if not ids:
        raise RuntimeError("ground_truth_sample empty")
    random.seed(seed)
    return random.choice(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_id", type=int, default=0, help="0 = random SQL id")
    ap.add_argument("--random_sample", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bev_range", type=float, default=80.0)
    ap.add_argument("--dataroot", default=r"E:\Data\Nuscenes\Full")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=3306)
    ap.add_argument("--user", default="root")
    ap.add_argument("--password", default="")
    ap.add_argument("--database", default="safetyai_sparsebev")
    ap.add_argument("--out", default=str(Path.home() / "Desktop" / "viz_gt_topo_map_demo.png"))
    ap.add_argument("--nuscenes_version", default="v1.0-trainval")
    ap.add_argument("--map_name_override", type=str, default=None, help='e.g. "boston-seaport"')
    ap.add_argument(
        "--map_dataroot",
        type=str,
        default=None,
        help='含 maps/expansion 的根目录（默认同 --dataroot）；支持嵌套 maps/expansion/expansion/*.json',
    )
    ap.add_argument(
        "--basemap_dir",
        type=str,
        default=None,
        help='basemap PNG 目录（默认尝试 maps/expansion/basemap 与 maps/basemap）',
    )

    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    conn = mysql_connect(args.host, args.port, args.user, args.password, args.database)
    try:
        t0 = time.time()
        if args.sample_id <= 0 or args.random_sample:
            sid = pick_random_id(conn, args.seed)
            print(f"Random sample id: {sid}")
        else:
            sid = args.sample_id
        ok = render_figure(conn, sid, args)
        if ok:
            print(f"Saved: {args.out}  ({time.time()-t0:.1f}s)")
        else:
            raise SystemExit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
