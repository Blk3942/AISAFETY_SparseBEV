# nuScenes dev-kit (detection) with safety-critical TP error modifications.

from typing import Callable

import numpy as np

from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.utils import (
    center_distance,
    yaw_diff,
    velocity_l2,
    attr_acc,
    cummean,
    cumweighted_mean,
    quaternion_yaw,
    ego_obj_yaw_diff_rad,
    piecewise_weight,
)
from nuscenes.eval.detection.data_classes import DetectionMetricData, DetectionConfig


def _safe_rel_err(a: float, b: float) -> float:
    """|a-b|/max(|b|,eps)"""
    eps = 1e-6
    return float(abs(a - b) / max(abs(b), eps))


def _scale_err_weighted(gt_box, pred_box, cfg: DetectionConfig) -> float:
    """
    Safety-critical scale error:
    weighted mean of relative errors on (w,l,h).
    """
    w_w = float(cfg.scale_dim_weights.get('w', 1.0))
    w_l = float(cfg.scale_dim_weights.get('l', 1.0))
    w_h = float(cfg.scale_dim_weights.get('h', 1.0))
    den = w_w + w_l + w_h
    if den <= 0:
        den = 1.0

    gt_w, gt_l, gt_h = [float(x) for x in gt_box.size]
    pr_w, pr_l, pr_h = [float(x) for x in pred_box.size]

    e_w = _safe_rel_err(pr_w, gt_w)
    e_l = _safe_rel_err(pr_l, gt_l)
    e_h = _safe_rel_err(pr_h, gt_h)
    return float((w_w * e_w + w_l * e_l + w_h * e_h) / den)


def _angle_weight_from_cfg(angle_rad: float, weighting_cfg: dict) -> float:
    split_deg = float(weighting_cfg.get('split_deg', 45.0))
    split_rad = float(np.deg2rad(split_deg))
    w_small = float(weighting_cfg.get('w_small', 1.0))
    w_large = float(weighting_cfg.get('w_large', 1.0))
    return piecewise_weight(angle_rad, split_rad, w_small=w_small, w_large=w_large)


def accumulate(
    gt_boxes: EvalBoxes,
    pred_boxes: EvalBoxes,
    class_name: str,
    dist_fcn: Callable,
    dist_th: float,
    cfg: DetectionConfig,
    verbose: bool = False,
) -> DetectionMetricData:
    """
    Same structure as official accumulate(), but:
    - scale_err: weighted per-dim relative error (w/l/h)
    - orient_err: multiplied by piecewise weight based on |yaw_obj - yaw_ego|
    - vel_err: multiplied by piecewise weight based on |yaw_obj - yaw_ego|
    - orient/vel use weighted cummean (to preserve meaning under per-match weights)
    """

    npos = len([1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name])
    if npos == 0:
        return DetectionMetricData.no_predictions()

    pred_boxes_list = [box for box in pred_boxes.all if box.detection_name == class_name]
    pred_confs = [box.detection_score for box in pred_boxes_list]
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    tp, fp, conf = [], [], []
    match_data = {
        'trans_err': [],
        'vel_err': [],
        'scale_err': [],
        'orient_err': [],
        'attr_err': [],
        'conf': [],
        # weights for weighted cumulative mean
        'orient_w': [],
        'vel_w': [],
    }

    taken = set()
    for ind in sortind:
        pred_box = pred_boxes_list[ind]
        min_dist = np.inf
        match_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes[pred_box.sample_token]):
            if gt_box.detection_name == class_name and (pred_box.sample_token, gt_idx) not in taken:
                this_distance = dist_fcn(gt_box, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        is_match = min_dist < dist_th
        if not is_match:
            tp.append(0)
            fp.append(1)
            conf.append(pred_box.detection_score)
            continue

        taken.add((pred_box.sample_token, match_gt_idx))
        tp.append(1)
        fp.append(0)
        conf.append(pred_box.detection_score)

        gt_box_match = gt_boxes[pred_box.sample_token][match_gt_idx]

        # base errors
        match_data['trans_err'].append(center_distance(gt_box_match, pred_box))
        match_data['vel_err'].append(velocity_l2(gt_box_match, pred_box))
        match_data['scale_err'].append(_scale_err_weighted(gt_box_match, pred_box, cfg))

        period = np.pi if class_name == 'barrier' else 2 * np.pi
        match_data['orient_err'].append(yaw_diff(gt_box_match, pred_box, period=period))
        match_data['attr_err'].append(1 - attr_acc(gt_box_match, pred_box))
        match_data['conf'].append(pred_box.detection_score)

        # angle between ego heading and object yaw (use GT yaw as "object yaw")
        ego_yaw = float(getattr(gt_box_match, 'ego_yaw', 0.0))
        obj_yaw = quaternion_yaw(gt_box_match.rotation)
        ang = ego_obj_yaw_diff_rad(ego_yaw=ego_yaw, obj_yaw=obj_yaw)

        match_data['orient_w'].append(_angle_weight_from_cfg(ang, cfg.orient_weighting))
        match_data['vel_w'].append(_angle_weight_from_cfg(ang, cfg.vel_weighting))

    if len(match_data['trans_err']) == 0:
        return DetectionMetricData.no_predictions()

    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    conf = np.array(conf)

    prec = tp / (fp + tp)
    rec = tp / float(npos)

    rec_interp = np.linspace(0, 1, DetectionMetricData.nelem)
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf_interp = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp

    # resample match data to align with conf curve
    conf_match = np.array(match_data['conf'])

    for key in ['trans_err', 'vel_err', 'scale_err', 'orient_err', 'attr_err']:
        x = np.array(match_data[key])
        if key == 'orient_err':
            w = np.array(match_data['orient_w'], dtype=float)
            tmp = cumweighted_mean(x * w, w)
        elif key == 'vel_err':
            w = np.array(match_data['vel_w'], dtype=float)
            tmp = cumweighted_mean(x * w, w)
        else:
            tmp = cummean(x)

        match_data[key] = np.interp(conf_interp[::-1], conf_match[::-1], tmp[::-1])[::-1]

    return DetectionMetricData(
        recall=rec,
        precision=prec,
        confidence=conf_interp,
        trans_err=match_data['trans_err'],
        vel_err=match_data['vel_err'],
        scale_err=match_data['scale_err'],
        orient_err=match_data['orient_err'],
        attr_err=match_data['attr_err'],
    )


def calc_ap(md: DetectionMetricData, min_recall: float, min_precision: float) -> float:
    assert 0 <= min_precision < 1
    assert 0 <= min_recall <= 1
    prec = np.copy(md.precision)
    prec = prec[round(100 * min_recall) + 1:]
    prec -= min_precision
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - min_precision)


def calc_tp(md: DetectionMetricData, min_recall: float, metric_name: str) -> float:
    first_ind = round(100 * min_recall) + 1
    last_ind = md.max_recall_ind
    if last_ind < first_ind:
        return 1.0
    return float(np.mean(getattr(md, metric_name)[first_ind:last_ind + 1]))

