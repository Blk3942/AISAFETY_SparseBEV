#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Optional

import numpy as np


def _ensure_safety_eval_path() -> None:
    root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'safety_critical_eval')
    )
    if os.path.isdir(root) and root not in sys.path:
        sys.path.insert(0, root)


_ensure_safety_eval_path()


def nds_from_summary(metrics_summary: Dict[str, Any]) -> float:
    cfg = metrics_summary.get('cfg') or {}
    w = float(cfg.get('mean_ap_weight', 5))
    mean_ap = float(metrics_summary['mean_ap'])
    tp_errors = metrics_summary['tp_errors']
    tp_metrics = ['trans_err', 'scale_err', 'orient_err', 'vel_err', 'attr_err']
    scores = []
    for k in tp_metrics:
        err = float(tp_errors[k])
        scores.append(max(0.0, 1.0 - err))
    num_tp = len(tp_metrics)
    total = w * mean_ap + float(np.sum(scores))
    return total / (w + num_tp)


def run_eval(
    result_path: str,
    dataroot: str,
    version: str = 'v1.0-trainval',
    eval_set: Optional[str] = None,
    output_dir: Optional[str] = None,
    config_name: str = 'detection_safety_critical',
    verbose: bool = True,
) -> Dict[str, Any]:
    from nuscenes import NuScenes
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.detection.evaluate import DetectionEval

    if eval_set is None:
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        if version not in eval_set_map:
            raise ValueError(f'请为 version={version} 显式传入 eval_set；已知映射: {eval_set_map}')
        eval_set = eval_set_map[version]

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(result_path)), 'safety_critical_eval_out')

    cfg = config_factory(config_name)
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=verbose)
    nusc_eval = DetectionEval(
        nusc,
        config=cfg,
        result_path=result_path,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=verbose,
    )
    metrics_summary = nusc_eval.main(plot_examples=0, render_curves=False)

    nds_check = nds_from_summary(metrics_summary)
    if verbose and abs(nds_check - float(metrics_summary['nd_score'])) > 1e-5:
        print('Warning: NDS 自检不一致', nds_check, metrics_summary['nd_score'])

    return metrics_summary


def main():
    parser = argparse.ArgumentParser(description='Safety-critical nuScenes detection evaluation (offline).')
    parser.add_argument('result_path', type=str, help='results_nusc.json 路径（nuScenes 提交格式）')
    parser.add_argument('--dataroot', type=str, required=True, help='nuScenes 根目录（含 v1.0-trainval 等）')
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='例如 v1.0-trainval, v1.0-mini')
    parser.add_argument('--eval-set', type=str, default=None, help='train/val/mini_val 等；默认自动映射')
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录（metrics_summary.json 等）')
    parser.add_argument('--config', type=str, default='detection_safety_critical', help='config_factory 名称')
    parser.add_argument('--quiet', action='store_true', help='减少打印')
    args = parser.parse_args()

    summary = run_eval(
        result_path=os.path.expanduser(args.result_path),
        dataroot=os.path.expanduser(args.dataroot),
        version=args.version,
        eval_set=args.eval_set,
        output_dir=os.path.expanduser(args.output_dir) if args.output_dir else None,
        config_name=args.config,
        verbose=not args.quiet,
    )

    if not args.quiet:
        print('mean_ap(weighted):', summary['mean_ap'])
        print('nd_score:', summary['nd_score'], '| 自检:', nds_from_summary(summary))


if __name__ == '__main__':
    main()

