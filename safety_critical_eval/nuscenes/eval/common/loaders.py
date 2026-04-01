# nuScenes dev-kit (subset) - detection only.

import json
from typing import Dict, Tuple, Optional

import numpy as np
import tqdm
from pyquaternion import Quaternion

from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.utils.splits import create_splits_scenes


def load_prediction(result_path: str, max_boxes_per_sample: int, box_cls, verbose: bool = False) -> Tuple[EvalBoxes, Dict]:
    with open(result_path) as f:
        data = json.load(f)
    assert 'results' in data, 'Error: No field `results` in result file.'

    all_results = EvalBoxes.deserialize(data['results'], box_cls)
    meta = data.get('meta', {})
    if verbose:
        print(f"Loaded results from {result_path}. Found detections for {len(all_results.sample_tokens)} samples.")

    for sample_token in all_results.sample_tokens:
        assert len(all_results.boxes[sample_token]) <= max_boxes_per_sample, \
            f"Error: Only <= {max_boxes_per_sample} boxes per sample allowed!"

    return all_results, meta


def load_gt(nusc: NuScenes, eval_split: str, box_cls, verbose: bool = False) -> EvalBoxes:
    assert box_cls == DetectionBox, 'This subset loader supports detection only.'

    attribute_map = {a['token']: a['name'] for a in nusc.attribute}
    if verbose:
        print(f'Loading annotations for {eval_split} split from nuScenes version: {nusc.version}')

    sample_tokens_all = [s['token'] for s in nusc.sample]
    assert len(sample_tokens_all) > 0, "Error: Database has no samples!"

    splits = create_splits_scenes()
    version = nusc.version
    if eval_split in {'train', 'val', 'train_detect'}:
        assert version.endswith('trainval'), f'Error: split {eval_split} not compatible with {version}'
    elif eval_split in {'mini_train', 'mini_val'}:
        assert version.endswith('mini'), f'Error: split {eval_split} not compatible with {version}'
    elif eval_split == 'test':
        assert version.endswith('test'), f'Error: split {eval_split} not compatible with {version}'
    else:
        raise ValueError(f'Error: Requested split {eval_split} not supported.')

    sample_tokens = []
    for sample_token in sample_tokens_all:
        scene_token = nusc.get('sample', sample_token)['scene_token']
        scene_record = nusc.get('scene', scene_token)
        if scene_record['name'] in splits[eval_split]:
            sample_tokens.append(sample_token)

    all_annotations = EvalBoxes()
    for sample_token in tqdm.tqdm(sample_tokens, leave=verbose):
        sample = nusc.get('sample', sample_token)
        sample_annotation_tokens = sample['anns']

        sample_boxes = []
        for sample_annotation_token in sample_annotation_tokens:
            sample_annotation = nusc.get('sample_annotation', sample_annotation_token)

            detection_name = category_to_detection_name(sample_annotation['category_name'])
            if detection_name is None:
                continue

            attr_tokens = sample_annotation['attribute_tokens']
            attr_count = len(attr_tokens)
            if attr_count == 0:
                attribute_name = ''
            elif attr_count == 1:
                attribute_name = attribute_map[attr_tokens[0]]
            else:
                raise Exception('Error: GT annotations must not have more than one attribute!')

            sample_boxes.append(
                box_cls(
                    sample_token=sample_token,
                    translation=sample_annotation['translation'],
                    size=sample_annotation['size'],
                    rotation=sample_annotation['rotation'],
                    velocity=nusc.box_velocity(sample_annotation['token'])[:2],
                    num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                    detection_name=detection_name,
                    detection_score=-1.0,
                    attribute_name=attribute_name,
                )
            )

        all_annotations.add_boxes(sample_token, sample_boxes)

    if verbose:
        print(f"Loaded ground truth annotations for {len(all_annotations.sample_tokens)} samples.")
    return all_annotations


def add_center_dist(nusc: NuScenes, eval_boxes: EvalBoxes):
    """
    Adds:
    - ego_translation (x,y,z) in ego frame (global - ego_pose translation; yaw not applied)
    - ego_yaw (radians) from ego_pose rotation
    """
    for sample_token in eval_boxes.sample_tokens:
        sample_rec = nusc.get('sample', sample_token)
        sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
        ego_yaw = quaternion_yaw(Quaternion(pose_record['rotation']))

        for box in eval_boxes[sample_token]:
            ego_translation = (
                box.translation[0] - pose_record['translation'][0],
                box.translation[1] - pose_record['translation'][1],
                box.translation[2] - pose_record['translation'][2],
            )
            box.ego_translation = ego_translation
            # dynamic attribute: used by safety-critical weighting.
            box.ego_yaw = float(ego_yaw)

    return eval_boxes


def filter_eval_boxes(
    eval_boxes: EvalBoxes,
    max_dist_by_class: Dict[str, float],
    global_max_dist: Optional[float] = None,
    verbose: bool = False,
) -> EvalBoxes:
    """
    Filters boxes by:
    - per-class max distance (official class_range)
    - optional global max distance (safety-critical radius X)
    - remove GT boxes with num_pts == 0 (pred boxes keep default -1)
    """
    total, after_dist, after_pts = 0, 0, 0
    for sample_token in eval_boxes.sample_tokens:
        total += len(eval_boxes[sample_token])
        filtered = []
        for box in eval_boxes[sample_token]:
            if box.ego_dist >= max_dist_by_class[box.detection_name]:
                continue
            if global_max_dist is not None and box.ego_dist >= global_max_dist:
                continue
            filtered.append(box)
        eval_boxes.boxes[sample_token] = filtered
        after_dist += len(filtered)

        eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if not box.num_pts == 0]
        after_pts += len(eval_boxes.boxes[sample_token])

    if verbose:
        print(f"=> Original number of boxes: {total}")
        print(f"=> After distance based filtering: {after_dist}")
        print(f"=> After points based filtering: {after_pts}")

    return eval_boxes

