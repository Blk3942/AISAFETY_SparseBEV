# nuScenes dev-kit (detection) with safety-critical extensions.

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

from nuscenes.eval.common.data_classes import MetricData, EvalBox
from nuscenes.eval.common.utils import center_distance
from nuscenes.eval.detection.constants import DETECTION_NAMES, ATTRIBUTE_NAMES, TP_METRICS


class DetectionConfig:
    """Detection evaluation settings + safety-critical extensions."""

    def __init__(
        self,
        class_range: Dict[str, int],
        dist_fcn: str,
        dist_ths: List[float],
        dist_th_tp: float,
        min_recall: float,
        min_precision: float,
        max_boxes_per_sample: int,
        mean_ap_weight: int,
        # safety extensions
        safety_max_dist: Optional[float] = None,
        class_ap_weights: Optional[Dict[str, float]] = None,
        scale_dim_weights: Optional[Dict[str, float]] = None,
        orient_weighting: Optional[Dict[str, float]] = None,
        vel_weighting: Optional[Dict[str, float]] = None,
    ):
        assert set(class_range.keys()) == set(DETECTION_NAMES), "Class count mismatch."
        assert dist_th_tp in dist_ths, "dist_th_tp must be in set of dist_ths."

        self.class_range = class_range
        self.dist_fcn = dist_fcn
        self.dist_ths = dist_ths
        self.dist_th_tp = dist_th_tp
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.max_boxes_per_sample = max_boxes_per_sample
        self.mean_ap_weight = mean_ap_weight

        self.class_names = self.class_range.keys()

        self.safety_max_dist = safety_max_dist
        self.class_ap_weights = class_ap_weights or {k: 1.0 for k in DETECTION_NAMES}
        self.scale_dim_weights = scale_dim_weights or {'w': 1.0, 'l': 1.0, 'h': 1.0}
        self.orient_weighting = orient_weighting or {'split_deg': 45.0, 'w_small': 1.0, 'w_large': 1.0}
        self.vel_weighting = vel_weighting or {'split_deg': 45.0, 'w_small': 1.0, 'w_large': 1.0}

        assert set(self.class_ap_weights.keys()) == set(DETECTION_NAMES), 'class_ap_weights must cover all classes.'
        for k in ['w', 'l', 'h']:
            assert k in self.scale_dim_weights

    def serialize(self) -> dict:
        return {
            'class_range': self.class_range,
            'dist_fcn': self.dist_fcn,
            'dist_ths': self.dist_ths,
            'dist_th_tp': self.dist_th_tp,
            'min_recall': self.min_recall,
            'min_precision': self.min_precision,
            'max_boxes_per_sample': self.max_boxes_per_sample,
            'mean_ap_weight': self.mean_ap_weight,
            'safety_max_dist': self.safety_max_dist,
            'class_ap_weights': self.class_ap_weights,
            'scale_dim_weights': self.scale_dim_weights,
            'orient_weighting': self.orient_weighting,
            'vel_weighting': self.vel_weighting,
        }

    @classmethod
    def deserialize(cls, content: dict):
        return cls(
            content['class_range'],
            content['dist_fcn'],
            content['dist_ths'],
            content['dist_th_tp'],
            content['min_recall'],
            content['min_precision'],
            content['max_boxes_per_sample'],
            content['mean_ap_weight'],
            safety_max_dist=content.get('safety_max_dist', None),
            class_ap_weights=content.get('class_ap_weights', None),
            scale_dim_weights=content.get('scale_dim_weights', None),
            orient_weighting=content.get('orient_weighting', None),
            vel_weighting=content.get('vel_weighting', None),
        )

    @property
    def dist_fcn_callable(self):
        if self.dist_fcn == 'center_distance':
            return center_distance
        raise Exception(f'Error: Unknown distance function {self.dist_fcn}!')


class DetectionMetricData(MetricData):
    nelem = 101

    def __init__(
        self,
        recall: np.array,
        precision: np.array,
        confidence: np.array,
        trans_err: np.array,
        vel_err: np.array,
        scale_err: np.array,
        orient_err: np.array,
        attr_err: np.array,
    ):
        assert len(recall) == self.nelem
        assert len(precision) == self.nelem
        assert len(confidence) == self.nelem
        assert len(trans_err) == self.nelem
        assert len(vel_err) == self.nelem
        assert len(scale_err) == self.nelem
        assert len(orient_err) == self.nelem
        assert len(attr_err) == self.nelem

        assert all(confidence == sorted(confidence, reverse=True))
        assert all(recall == sorted(recall))

        self.recall = recall
        self.precision = precision
        self.confidence = confidence
        self.trans_err = trans_err
        self.vel_err = vel_err
        self.scale_err = scale_err
        self.orient_err = orient_err
        self.attr_err = attr_err

    @property
    def max_recall_ind(self):
        non_zero = np.nonzero(self.confidence)[0]
        if len(non_zero) == 0:
            return 0
        return int(non_zero[-1])

    @property
    def max_recall(self):
        return float(self.recall[self.max_recall_ind])

    def serialize(self):
        return {
            'recall': self.recall.tolist(),
            'precision': self.precision.tolist(),
            'confidence': self.confidence.tolist(),
            'trans_err': self.trans_err.tolist(),
            'vel_err': self.vel_err.tolist(),
            'scale_err': self.scale_err.tolist(),
            'orient_err': self.orient_err.tolist(),
            'attr_err': self.attr_err.tolist(),
        }

    @classmethod
    def deserialize(cls, content: dict):
        return cls(
            recall=np.array(content['recall']),
            precision=np.array(content['precision']),
            confidence=np.array(content['confidence']),
            trans_err=np.array(content['trans_err']),
            vel_err=np.array(content['vel_err']),
            scale_err=np.array(content['scale_err']),
            orient_err=np.array(content['orient_err']),
            attr_err=np.array(content['attr_err']),
        )

    @classmethod
    def no_predictions(cls):
        return cls(
            recall=np.linspace(0, 1, cls.nelem),
            precision=np.zeros(cls.nelem),
            confidence=np.zeros(cls.nelem),
            trans_err=np.ones(cls.nelem),
            vel_err=np.ones(cls.nelem),
            scale_err=np.ones(cls.nelem),
            orient_err=np.ones(cls.nelem),
            attr_err=np.ones(cls.nelem),
        )


class DetectionMetrics:
    """Stores AP and TP metric results. Provides properties to summarize."""

    def __init__(self, cfg: DetectionConfig):
        self.cfg = cfg
        self._label_aps = defaultdict(lambda: defaultdict(float))
        self._label_tp_errors = defaultdict(lambda: defaultdict(float))
        self.eval_time = None

    def add_label_ap(self, detection_name: str, dist_th: float, ap: float) -> None:
        self._label_aps[detection_name][dist_th] = ap

    def get_label_ap(self, detection_name: str, dist_th: float) -> float:
        return float(self._label_aps[detection_name][dist_th])

    def add_label_tp(self, detection_name: str, metric_name: str, tp: float):
        self._label_tp_errors[detection_name][metric_name] = tp

    def get_label_tp(self, detection_name: str, metric_name: str) -> float:
        return float(self._label_tp_errors[detection_name][metric_name])

    def add_runtime(self, eval_time: float) -> None:
        self.eval_time = float(eval_time)

    @property
    def mean_dist_aps(self) -> Dict[str, float]:
        return {class_name: float(np.mean(list(d.values()))) for class_name, d in self._label_aps.items()}

    @property
    def mean_ap(self) -> float:
        """Safety-critical: weighted average over classes (weights from cfg.class_ap_weights)."""
        per_class = self.mean_dist_aps
        weights = self.cfg.class_ap_weights
        num = 0.0
        den = 0.0
        for cls_name, ap in per_class.items():
            w = float(weights.get(cls_name, 1.0))
            num += w * float(ap)
            den += w
        return float(num / den) if den > 0 else float(np.mean(list(per_class.values())))

    @property
    def tp_errors(self) -> Dict[str, float]:
        errors = {}
        for metric_name in TP_METRICS:
            class_errors = []
            for detection_name in self.cfg.class_names:
                class_errors.append(self.get_label_tp(detection_name, metric_name))
            errors[metric_name] = float(np.nanmean(class_errors))
        return errors

    @property
    def tp_scores(self) -> Dict[str, float]:
        scores = {}
        tp_errors = self.tp_errors
        for metric_name in TP_METRICS:
            score = 1.0 - tp_errors[metric_name]
            score = max(0.0, float(score))
            scores[metric_name] = score
        return scores

    @property
    def nd_score(self) -> float:
        total = float(self.cfg.mean_ap_weight * self.mean_ap + np.sum(list(self.tp_scores.values())))
        total = total / float(self.cfg.mean_ap_weight + len(self.tp_scores.keys()))
        return float(total)

    def serialize(self):
        return {
            'label_aps': self._label_aps,
            'mean_dist_aps': self.mean_dist_aps,
            'mean_ap': self.mean_ap,
            'label_tp_errors': self._label_tp_errors,
            'tp_errors': self.tp_errors,
            'tp_scores': self.tp_scores,
            'nd_score': self.nd_score,
            'eval_time': self.eval_time,
            'cfg': self.cfg.serialize(),
        }


class DetectionBox(EvalBox):
    """Data class used during detection evaluation. Can be a prediction or ground truth."""

    def __init__(
        self,
        sample_token: str = "",
        translation: Tuple[float, float, float] = (0, 0, 0),
        size: Tuple[float, float, float] = (0, 0, 0),
        rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
        velocity: Tuple[float, float] = (0, 0),
        ego_translation: Tuple[float, float, float] = (0, 0, 0),
        num_pts: int = -1,
        detection_name: str = 'car',
        detection_score: float = -1.0,
        attribute_name: str = '',
    ):
        super().__init__(sample_token, translation, size, rotation, velocity, ego_translation, num_pts)

        assert detection_name is not None and detection_name in DETECTION_NAMES
        assert attribute_name in ATTRIBUTE_NAMES or attribute_name == ''
        assert type(detection_score) == float
        assert not np.any(np.isnan(detection_score))

        self.detection_name = detection_name
        self.detection_score = detection_score
        self.attribute_name = attribute_name

    def serialize(self) -> dict:
        return {
            'sample_token': self.sample_token,
            'translation': self.translation,
            'size': self.size,
            'rotation': self.rotation,
            'velocity': self.velocity,
            'ego_translation': self.ego_translation,
            'num_pts': self.num_pts,
            'detection_name': self.detection_name,
            'detection_score': self.detection_score,
            'attribute_name': self.attribute_name,
        }

    @classmethod
    def deserialize(cls, content: dict):
        return cls(
            sample_token=content['sample_token'],
            translation=tuple(content['translation']),
            size=tuple(content['size']),
            rotation=tuple(content['rotation']),
            velocity=tuple(content['velocity']),
            ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content else tuple(content['ego_translation']),
            num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
            detection_name=content['detection_name'],
            detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
            attribute_name=content.get('attribute_name', ''),
        )


class DetectionMetricDataList:
    """Stores MetricData in a dict indexed by (name, match-distance)."""

    def __init__(self):
        self.md: Dict[Tuple[str, float], DetectionMetricData] = {}

    def __getitem__(self, key):
        return self.md[key]

    def set(self, detection_name: str, match_distance: float, data: DetectionMetricData):
        self.md[(detection_name, match_distance)] = data

    def serialize(self) -> dict:
        return {key[0] + ':' + str(key[1]): value.serialize() for key, value in self.md.items()}

