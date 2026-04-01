# Minimal config_factory for detection only.

import json
import os
from typing import Any

from nuscenes.eval.detection.data_classes import DetectionConfig


def config_factory(configuration_name: str) -> DetectionConfig:
    tokens = configuration_name.split('_')
    assert len(tokens) > 1, 'Error: Configuration name must have prefix "detection_"!'
    task = tokens[0]
    assert task == 'detection', f'Only detection supported, got {task}'

    this_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(this_dir, '..', task, 'configs', f'{configuration_name}.json')
    assert os.path.exists(cfg_path), f'Requested unknown configuration {configuration_name}'

    with open(cfg_path, 'r') as f:
        data: Any = json.load(f)
    return DetectionConfig.deserialize(data)

