# DetectionEval for safety-critical config.

import json
import os
import random
import time
from typing import Tuple, Dict, Any

import numpy as np

from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, DetectionMetricDataList


class DetectionEval:
    def __init__(
        self,
        nusc: NuScenes,
        config: DetectionConfig,
        result_path: str,
        eval_set: str,
        output_dir: str = None,
        verbose: bool = True,
    ):
        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config

        assert os.path.exists(result_path), 'Error: The result file does not exist!'
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        if verbose:
            print('Initializing safety-critical nuScenes detection evaluation')

        self.pred_boxes, self.meta = load_prediction(
            self.result_path, self.cfg.max_boxes_per_sample, DetectionBox, verbose=verbose
        )
        self.gt_boxes = load_gt(self.nusc, self.eval_set, DetectionBox, verbose=verbose)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)

        if verbose:
            print('Filtering predictions (class_range + safety_max_dist)')
        self.pred_boxes = filter_eval_boxes(
            self.pred_boxes, self.cfg.class_range, global_max_dist=self.cfg.safety_max_dist, verbose=verbose
        )
        if verbose:
            print('Filtering ground truth annotations (class_range + safety_max_dist)')
        self.gt_boxes = filter_eval_boxes(
            self.gt_boxes, self.cfg.class_range, global_max_dist=self.cfg.safety_max_dist, verbose=verbose
        )

        self.sample_tokens = self.gt_boxes.sample_tokens

    def evaluate(self) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        start_time = time.time()

        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = DetectionMetricDataList()
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                md = accumulate(self.gt_boxes, self.pred_boxes, class_name, self.cfg.dist_fcn_callable, dist_th, cfg=self.cfg)
                metric_data_list.set(class_name, dist_th, md)

        if self.verbose:
            print('Calculating metrics...')
        metrics = DetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)

            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]
                if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        metrics.add_runtime(time.time() - start_time)
        return metrics, metric_data_list

    def main(self, plot_examples: int = 0, render_curves: bool = False) -> Dict[str, Any]:
        if plot_examples > 0:
            # Optional visualization intentionally not supported in this subset.
            random.seed(42)

        metrics, metric_data_list = self.evaluate()

        if self.verbose:
            print(f'Saving metrics to: {self.output_dir}')
        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = self.meta.copy()
        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(self.output_dir, 'metrics_details.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        if self.verbose:
            print('mean_ap (weighted): %.4f' % metrics_summary['mean_ap'])
            err_name_mapping = {
                'trans_err': 'mATE',
                'scale_err': 'mASE',
                'orient_err': 'mAOE',
                'vel_err': 'mAVE',
                'attr_err': 'mAAE'
            }
            for tp_name, tp_val in metrics_summary['tp_errors'].items():
                print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
            print('NDS: %.4f' % metrics_summary['nd_score'])
            print('Eval time: %.1fs' % metrics_summary['eval_time'])

        return metrics_summary


class NuScenesEval(DetectionEval):
    """Backward-compat alias."""

