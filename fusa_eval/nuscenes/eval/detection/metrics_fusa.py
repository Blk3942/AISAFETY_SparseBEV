# Class-level aggregation weights for mAP / TP errors / NDS (on top of per-GT weights in accumulate).
import numpy as np

from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics
from nuscenes.eval.detection.fusa_weighting import FusaWeightingSpec


class FusaDetectionMetrics(DetectionMetrics):
    def __init__(self, cfg: DetectionConfig, fusa: FusaWeightingSpec):
        super().__init__(cfg)
        self._fusa = fusa

    @property
    def mean_ap(self) -> float:
        names = list(self.cfg.class_names)
        ws = np.array([self._fusa.class_weight(n) for n in names], dtype=float)
        aps = np.array([self.mean_dist_aps[n] for n in names], dtype=float)
        s = ws.sum()
        if s <= 0:
            return float(np.mean(aps))
        return float((ws * aps).sum() / s)

    @property
    def tp_errors(self):
        errors = {}
        names = list(self.cfg.class_names)
        ws = np.array([self._fusa.class_weight(n) for n in names], dtype=float)
        for metric_name in TP_METRICS:
            vals = []
            wvals = []
            for i, detection_name in enumerate(names):
                v = self.get_label_tp(detection_name, metric_name)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                vals.append(float(v))
                wvals.append(float(ws[i]))
            if not vals:
                errors[metric_name] = float('nan')
            else:
                wv = np.array(wvals)
                vv = np.array(vals)
                errors[metric_name] = float((wv * vv).sum() / max(wv.sum(), 1e-12))
        return errors

    @property
    def tp_scores(self):
        scores = {}
        tp_errors = self.tp_errors
        mul = self._fusa.nds_tp_metric_multipliers or {}
        for metric_name in TP_METRICS:
            err = tp_errors[metric_name]
            if np.isnan(err):
                score = 0.0
            else:
                score = max(0.0, 1.0 - err)
            scores[metric_name] = score * float(mul.get(metric_name, 1.0))
        return scores

    @property
    def nd_score(self) -> float:
        w_map = self._fusa.nds_mean_ap_weight
        if w_map is None:
            w_map = float(self.cfg.mean_ap_weight)
        else:
            w_map = float(w_map)
        total = float(w_map * self.mean_ap + np.sum(list(self.tp_scores.values())))
        n_tp = len(TP_METRICS)
        return total / float(w_map + n_tp)

    def serialize(self):
        out = super().serialize()
        out['fusa_weighting'] = self._fusa.raw
        out['fusa_enabled'] = True
        return out
