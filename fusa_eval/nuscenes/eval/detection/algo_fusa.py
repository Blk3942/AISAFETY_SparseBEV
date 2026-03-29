# Weighted accumulate: GT mass and TP errors use per-GT weights (class × ego distance × attribute).
from typing import Callable

import numpy as np

from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.utils import (
    attr_acc,
    center_distance,
    scale_iou,
    velocity_l2,
    yaw_diff,
)
from nuscenes.eval.detection.data_classes import DetectionMetricData


def _cum_weighted_mean_nan(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Cumulative weighted mean; NaN values contribute zero weight."""
    v = values.astype(float)
    w = weights.astype(float)
    valid = ~np.isnan(v)
    vw = np.where(valid, v * w, 0.0)
    wv = np.where(valid, w, 0.0)
    cum_vw = np.cumsum(vw)
    cum_wv = np.cumsum(wv)
    out = np.divide(
        cum_vw,
        cum_wv,
        out=np.ones_like(cum_vw),
        where=cum_wv > 1e-12,
    )
    return out


def accumulate_weighted(
    gt_boxes: EvalBoxes,
    pred_boxes: EvalBoxes,
    class_name: str,
    dist_fcn: Callable,
    dist_th: float,
    weight_fn: Callable,
    verbose: bool = False,
) -> DetectionMetricData:
    """
    Same matching as official accumulate, but:
    - Positive mass npos = sum(weight_fn(gt)) over GT of this class
    - Recall / precision use cumulative matched weight vs npos and vs rank
    - TP error curves use cumulative weighted mean of per-match errors
    """
    npos = 0.0
    for gt_box in gt_boxes.all:
        if gt_box.detection_name == class_name:
            npos += weight_fn(gt_box)

    if verbose:
        n_gt = len([1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name])
        print(
            'Found weighted npos={} ({} GT) of class {} out of {} total across {} samples.'.format(
                npos, n_gt, class_name, len(gt_boxes.all), len(gt_boxes.sample_tokens)
            )
        )

    if npos <= 0:
        return DetectionMetricData.no_predictions()

    pred_boxes_list = [box for box in pred_boxes.all if box.detection_name == class_name]
    pred_confs = [box.detection_score for box in pred_boxes_list]

    if verbose:
        print(
            'Found {} PRED of class {} out of {} total across {} samples.'.format(
                len(pred_confs), class_name, len(pred_boxes.all), len(pred_boxes.sample_tokens)
            )
        )

    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    tp = []
    fp = []
    conf = []
    w_step = []

    match_data = {
        'trans_err': [],
        'vel_err': [],
        'scale_err': [],
        'orient_err': [],
        'attr_err': [],
        'conf': [],
        'match_w': [],
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

        if is_match:
            taken.add((pred_box.sample_token, match_gt_idx))
            tp.append(1)
            fp.append(0)
            conf.append(pred_box.detection_score)

            gt_box_match = gt_boxes[pred_box.sample_token][match_gt_idx]
            wm = weight_fn(gt_box_match)
            w_step.append(wm)

            period = np.pi if class_name == 'barrier' else 2 * np.pi
            te = center_distance(gt_box_match, pred_box)
            ve = velocity_l2(gt_box_match, pred_box)
            se = 1 - scale_iou(gt_box_match, pred_box)
            oe = yaw_diff(gt_box_match, pred_box, period=period)
            ae = 1 - attr_acc(gt_box_match, pred_box)

            match_data['trans_err'].append(te)
            match_data['vel_err'].append(ve)
            match_data['scale_err'].append(se)
            match_data['orient_err'].append(oe)
            match_data['attr_err'].append(ae)
            match_data['conf'].append(pred_box.detection_score)
            match_data['match_w'].append(wm)
        else:
            tp.append(0)
            fp.append(1)
            conf.append(pred_box.detection_score)
            w_step.append(0.0)

    if len(match_data['trans_err']) == 0:
        return DetectionMetricData.no_predictions()

    tp = np.cumsum(np.array(tp, dtype=float))
    fp = np.cumsum(np.array(fp, dtype=float))
    conf = np.array(conf, dtype=float)
    W = np.cumsum(np.array(w_step, dtype=float))

    # Precision 与官方一致（按预测条数计），避免 GT 权重>1 时出现 precision>1、mAP>1。
    # Recall 使用累积匹配权重 / 加权 GT 总质量 npos，体现距离/属性/类别重要性。
    prec = tp / np.maximum(fp + tp, 1e-12)
    rec = W / float(npos)

    rec_interp = np.linspace(0, 1, DetectionMetricData.nelem)
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp

    match_w = np.array(match_data['match_w'], dtype=float)
    for key in ('trans_err', 'vel_err', 'scale_err', 'orient_err', 'attr_err'):
        arr = np.array(match_data[key], dtype=float)
        tmp = _cum_weighted_mean_nan(arr, match_w)
        match_data[key] = np.interp(conf[::-1], match_data['conf'][::-1], tmp[::-1])[::-1]

    return DetectionMetricData(
        recall=rec,
        precision=prec,
        confidence=conf,
        trans_err=match_data['trans_err'],
        vel_err=match_data['vel_err'],
        scale_err=match_data['scale_err'],
        orient_err=match_data['orient_err'],
        attr_err=match_data['attr_err'],
    )
