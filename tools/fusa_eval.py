#!/usr/bin/env python3
# Copyright: logic documented below refers to nuScenes devkit & MMDet3D (external packages).
"""
FUSA 定制化评估入口（由 ``official_nuscenes_eval.py`` 完整复制）。

默认与官方 nuScenes 评测一致。传入 ``--fusa-weighting <json>`` 时启用可配置加权（见
``fusa_eval/configs/fusa_weighting.example.json``）：类别、自车距离分箱、属性、NDS 超参。

**仅** 将 ``fusa_eval/`` 加入 ``sys.path``；无加权基线可用 ``tools/official_nuscenes_eval.py``。

================================================================================
代码与逻辑溯源（便于你阅读上游实现）
================================================================================

1) NDS 的数学定义（加权平均）
   - 文件: nuscenes/eval/detection/data_classes.py
   - 类: DetectionMetrics
   - 属性: nd_score
   - 公式（与上游一致）::
       tp_scores[k] = max(0, 1 - tp_errors[k])   # k in trans_err, scale_err, ...
       NDS = (mean_ap_weight * mean_ap + sum(tp_scores)) / (mean_ap_weight + len(tp_scores))
   - 默认配置 detection_cvpr_2019.json 中 mean_ap_weight = 5
   - 文件: nuscenes/eval/detection/configs/detection_cvpr_2019.json

2) mAP / mATE / … 的计算流程
   - 文件: nuscenes/eval/detection/evaluate.py
   - 类: DetectionEval
   - evaluate() -> accumulate()（按类、按距离阈值）-> calc_ap / calc_tp
   - accumulate / calc_ap: nuscenes/eval/detection/algo.py

3) MMDet3D 如何把模型输出接到上述流程
   - 文件: mmdet3d/datasets/nuscenes_dataset.py
   - 方法: _evaluate_single() 使用 NuScenesEval(=DetectionEval)，读入 results_nusc.json，
     调用 main(render_curves=False)，再读取 metrics_summary.json 中的 nd_score、mAP 等。

4) 输入 JSON 格式
   - 由 NuScenesDataset._format_bbox 写出，字段含 meta + results[sample_token] -> [DetectionBox, ...]
   - 须与官方 load_prediction 兼容（见 nuscenes/eval/common/loaders.py）。

本脚本不复制上述算法，仅封装调用，保证与榜单/ val.py 一致。

默认将 ``fusa_eval/`` 置于 sys.path 最前（内嵌 nuScenes 检测评测与精简 ``NuScenes`` 加载器），
无需 pip 安装 ``nuscenes`` 即可完成评测（完全离线）。
================================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

import numpy as np


def _ensure_fusa_eval_path() -> None:
    """将 ``fusa_eval/`` 置于 sys.path 最前（仅 FUSA 评测包，不引用 offline_nuscenes_eval）。"""
    root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fusa_eval')
    )
    if not os.path.isdir(root):
        raise FileNotFoundError(
            '未找到 FUSA 评测目录: %s（请从 offline_nuscenes_eval 复制或保持仓库完整）' % root
        )
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_fusa_eval_path()


def nds_from_summary(metrics_summary: Dict[str, Any]) -> float:
    """
    用 metrics_summary.json 里的字段按官方公式重算 NDS，用于自检。
    与 nuscenes.eval.detection.data_classes.DetectionMetrics.nd_score 一致。
    """
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


def run_official_eval(
    result_path: str,
    dataroot: str,
    version: str = 'v1.0-trainval',
    eval_set: Optional[str] = None,
    output_dir: Optional[str] = None,
    config_name: str = 'detection_cvpr_2019',
    render_curves: bool = False,
    plot_examples: int = 0,
    verbose: bool = True,
    fusa_weighting_config: Optional[str] = None,
) -> Dict[str, Any]:
    """
    调用 nuscenes.eval.detection.evaluate.DetectionEval，与 MMDet3D 默认配置一致。
    """
    from nuscenes import NuScenes
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.detection.evaluate import DetectionEval

    if eval_set is None:
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        if version not in eval_set_map:
            raise ValueError(
                f'请为 version={version} 显式传入 eval_set；'
                f'已知映射: {eval_set_map}'
            )
        eval_set = eval_set_map[version]

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(result_path)), 'nuscenes_eval_out')

    cfg = config_factory(config_name)
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=verbose)
    nusc_eval = DetectionEval(
        nusc,
        config=cfg,
        result_path=result_path,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=verbose,
        fusa_weighting_config=fusa_weighting_config,
    )
    metrics_summary = nusc_eval.main(plot_examples=plot_examples, render_curves=render_curves)

    nds_check = nds_from_summary(metrics_summary)
    if verbose and abs(nds_check - float(metrics_summary['nd_score'])) > 1e-5:
        print('Warning: NDS 自检与 devkit 不一致', nds_check, metrics_summary['nd_score'])

    return metrics_summary


def main():
    parser = argparse.ArgumentParser(
        prog='fusa_eval.py',
        description='FUSA：nuScenes 本地 mAP / mATE… / NDS（与 official_nuscenes_eval.py 同逻辑，可在此文件定制）',
    )
    parser.add_argument('result_path', type=str, help='results_nusc.json 路径（MMDet3D 提交格式）')
    parser.add_argument('--dataroot', type=str, required=True, help='nuScenes 根目录（含 v1.0-trainval 等）')
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='例如 v1.0-trainval, v1.0-mini')
    parser.add_argument(
        '--eval-set',
        type=str,
        default=None,
        help='train / val / mini_val 等；默认按 MMDet3D：trainval->val, mini->mini_val',
    )
    parser.add_argument('--output-dir', type=str, default=None, help='metrics_summary.json 输出目录')
    parser.add_argument(
        '--config',
        type=str,
        default='detection_cvpr_2019',
        help='nuscenes config_factory 名称，须与训练评测一致',
    )
    parser.add_argument('--render-curves', action='store_true', help='是否绘制 PR/TP 曲线（较慢）')
    parser.add_argument('--plot-examples', type=int, default=0, help='可视化样本数量，默认 0')
    parser.add_argument('--quiet', action='store_true', help='减少打印')
    parser.add_argument(
        '--fusa-weighting',
        type=str,
        default=None,
        help='FUSA 加权配置 JSON（类别/自车距离/属性/NDS）；不设则与官方 nuScenes 一致',
    )
    args = parser.parse_args()

    fw = os.path.expanduser(args.fusa_weighting) if args.fusa_weighting else None

    summary = run_official_eval(
        result_path=os.path.expanduser(args.result_path),
        dataroot=os.path.expanduser(args.dataroot),
        version=args.version,
        eval_set=args.eval_set,
        output_dir=os.path.expanduser(args.output_dir) if args.output_dir else None,
        config_name=args.config,
        render_curves=args.render_curves,
        plot_examples=args.plot_examples,
        verbose=not args.quiet,
        fusa_weighting_config=fw,
    )

    out_json = os.path.join(
        args.output_dir or os.path.join(os.path.dirname(os.path.abspath(args.result_path)), 'nuscenes_eval_out'),
        'metrics_summary.json',
    )
    if not args.quiet:
        print('已写入:', out_json)
        print('NDS (nd_score):', summary['nd_score'], '| 自检:', nds_from_summary(summary))


if __name__ == '__main__':
    main()
