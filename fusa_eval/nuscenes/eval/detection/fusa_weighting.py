# FUSA: configurable weights for nuScenes detection evaluation (class / ego-distance / attribute).
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from nuscenes.eval.detection.constants import ATTRIBUTE_NAMES, DETECTION_NAMES


def _default_class_weights() -> Dict[str, float]:
    return {n: 1.0 for n in DETECTION_NAMES}


@dataclass
class FusaWeightingSpec:
    """Loaded from JSON; used by weighted accumulate and FusaDetectionMetrics."""

    raw: Dict[str, Any]
    class_weights: Dict[str, float]
    # Sorted (upper_bound_m, weight); last upper_bound_m is np.inf
    ego_distance_bins: List[Tuple[float, float]]
    attribute_default: float
    # detection_name -> {attribute_name or '_default' or '_empty': weight}
    attribute_per_class: Dict[str, Dict[str, float]]
    nds_mean_ap_weight: Optional[float] = None
    nds_tp_metric_multipliers: Dict[str, float] = field(default_factory=dict)

    def class_weight(self, detection_name: str) -> float:
        return float(self.class_weights.get(detection_name, 1.0))

    def ego_distance_weight(self, ego_dist_m: float) -> float:
        d = float(ego_dist_m)
        for upper, w in self.ego_distance_bins:
            if d < upper:
                return float(w)
        return float(self.ego_distance_bins[-1][1]) if self.ego_distance_bins else 1.0

    def attribute_weight(self, detection_name: str, attribute_name: str) -> float:
        per = self.attribute_per_class.get(detection_name)
        if not per:
            return self.attribute_default
        if not attribute_name:
            if '_empty' in per:
                return float(per['_empty'])
            return float(per.get('_default', self.attribute_default))
        if attribute_name in per:
            return float(per[attribute_name])
        return float(per.get('_default', self.attribute_default))

    def gt_weight(self, box) -> float:
        """Positive weight for one GT DetectionBox (after ego_translation / ego_dist are set)."""
        from nuscenes.eval.detection.data_classes import DetectionBox

        assert isinstance(box, DetectionBox)
        w = self.class_weight(box.detection_name)
        w *= self.ego_distance_weight(box.ego_dist)
        w *= self.attribute_weight(box.detection_name, box.attribute_name or '')
        return max(float(w), 1e-9)


def _parse_ego_distance_bins(data: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
    """Each item: {\"up_to_m\": number | null, \"weight\": number}. Bins are [prev, up_to_m)."""
    if not data:
        return [(np.inf, 1.0)]
    bins = []
    for item in data:
        u = item.get('up_to_m')
        upper = np.inf if u is None else float(u)
        w = float(item['weight'])
        bins.append((upper, w))
    bins.sort(key=lambda x: x[0])
    return bins


def _parse_attribute_weights(obj: Any) -> Tuple[float, Dict[str, Dict[str, float]]]:
    default = 1.0
    per_class: Dict[str, Dict[str, float]] = {}
    if not isinstance(obj, dict):
        return default, per_class
    default = float(obj.get('default', 1.0))
    raw_pc = obj.get('per_class', {}) or {}
    for cls, m in raw_pc.items():
        if cls not in DETECTION_NAMES:
            continue
        if not isinstance(m, dict):
            continue
        inner = {}
        for k, v in m.items():
            if k in ('_default', '_empty'):
                inner[k] = float(v)
                continue
            if k in ATTRIBUTE_NAMES:
                inner[k] = float(v)
        per_class[cls] = inner
    return default, per_class


def load_fusa_weighting_spec(path: str) -> FusaWeightingSpec:
    path = os.path.expanduser(path)
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    cw = _default_class_weights()
    for k, v in (raw.get('class_weights') or {}).items():
        if k in DETECTION_NAMES:
            cw[k] = float(v)

    ego_bins = _parse_ego_distance_bins(raw.get('ego_distance_bins') or [])

    attr_def, attr_pc = _parse_attribute_weights(raw.get('attribute_weights') or {})

    nds_map = raw.get('nds') or {}
    nds_maw = nds_map.get('mean_ap_weight')
    nds_maw = float(nds_maw) if nds_maw is not None else None
    tp_mul = {}
    for k, v in (nds_map.get('tp_metric_multipliers') or {}).items():
        if k in ('trans_err', 'scale_err', 'orient_err', 'vel_err', 'attr_err'):
            tp_mul[k] = float(v)

    return FusaWeightingSpec(
        raw=raw,
        class_weights=cw,
        ego_distance_bins=ego_bins,
        attribute_default=attr_def,
        attribute_per_class=attr_pc,
        nds_mean_ap_weight=nds_maw,
        nds_tp_metric_multipliers=tp_mul,
    )
