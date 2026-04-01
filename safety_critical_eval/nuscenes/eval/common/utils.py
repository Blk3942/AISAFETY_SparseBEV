# nuScenes dev-kit (subset).

from typing import Dict, List

import numpy as np
from pyquaternion import Quaternion

from nuscenes.eval.common.data_classes import EvalBox

DetectionBox = object  # typing workaround


def center_distance(gt_box: EvalBox, pred_box: EvalBox) -> float:
    """L2 distance between the box centers (xy only)."""
    return float(np.linalg.norm(np.array(pred_box.translation[:2]) - np.array(gt_box.translation[:2])))


def velocity_l2(gt_box: EvalBox, pred_box: EvalBox) -> float:
    """L2 distance between the velocity vectors (xy only)."""
    return float(np.linalg.norm(np.array(pred_box.velocity) - np.array(gt_box.velocity)))


def quaternion_yaw(q) -> float:
    """Yaw (radians) from quaternion in lidar/global frame.

    safety_critical_eval 的 box.rotation 可能是 list/tuple（wxyz），这里统一转成 Quaternion。
    """
    if not isinstance(q, Quaternion):
        q = Quaternion(q)
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))
    return float(np.arctan2(v[1], v[0]))


def angle_diff(x: float, y: float, period: float) -> float:
    """Signed smallest angle difference from y to x in (-pi, pi)."""
    diff = (x - y + period / 2) % period - period / 2
    if diff > np.pi:
        diff = diff - (2 * np.pi)
    return float(diff)


def yaw_diff(gt_box: EvalBox, eval_box: EvalBox, period: float = 2 * np.pi) -> float:
    """Abs yaw difference in [0, pi] using periodicity."""
    yaw_gt = quaternion_yaw(Quaternion(gt_box.rotation))
    yaw_est = quaternion_yaw(Quaternion(eval_box.rotation))
    return float(abs(angle_diff(yaw_gt, yaw_est, period)))


def attr_acc(gt_box: DetectionBox, pred_box: DetectionBox) -> float:
    """Attribute classification accuracy (0/1) or nan if GT has no attribute."""
    if getattr(gt_box, 'attribute_name', '') == '':
        return float('nan')
    return float(getattr(gt_box, 'attribute_name') == getattr(pred_box, 'attribute_name'))


def cummean(x: np.array) -> np.array:
    """NaN-aware cumulative mean."""
    if sum(np.isnan(x)) == len(x):
        return np.ones(len(x))
    sum_vals = np.nancumsum(x.astype(float))
    count_vals = np.cumsum(~np.isnan(x))
    return np.divide(sum_vals, count_vals, out=np.zeros_like(sum_vals), where=count_vals != 0)


def cumweighted_mean(x: np.array, w: np.array) -> np.array:
    """
    NaN-aware cumulative weighted mean.
    - NaN 的 x 会被忽略（对应权重也忽略）。
    - 若累计有效权重为 0，则该位置输出 1（与 devkit 对“无效误差”的处理保持同量纲兜底）。
    """
    assert x.shape == w.shape
    valid = ~np.isnan(x)
    wv = np.where(valid, w.astype(float), 0.0)
    xv = np.where(valid, x.astype(float), 0.0)
    num = np.cumsum(xv * wv)
    den = np.cumsum(wv)
    out = np.divide(num, den, out=np.ones_like(num), where=den != 0)
    return out


def ego_obj_yaw_diff_rad(ego_yaw: float, obj_yaw: float) -> float:
    """Abs yaw difference between ego heading and object yaw in [0, pi]."""
    return float(abs(angle_diff(obj_yaw, ego_yaw, period=2 * np.pi)))


def piecewise_weight(angle_rad: float, split_rad: float, w_small: float, w_large: float) -> float:
    """Two-bin weight by angle: [0, split] -> w_small, (split, pi] -> w_large."""
    if angle_rad <= split_rad:
        return float(w_small)
    return float(w_large)

