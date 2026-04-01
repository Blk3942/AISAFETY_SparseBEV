# nuScenes dev-kit utilities (detection) - subset.

from typing import List, Optional


def category_to_detection_name(category_name: str) -> Optional[str]:
    detection_mapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck',
    }
    return detection_mapping.get(category_name, None)


def detection_name_to_rel_attributes(detection_name: str) -> List[str]:
    if detection_name in ['pedestrian']:
        return ['pedestrian.moving', 'pedestrian.sitting_lying_down', 'pedestrian.standing']
    if detection_name in ['bicycle', 'motorcycle']:
        return ['cycle.with_rider', 'cycle.without_rider']
    if detection_name in ['car', 'bus', 'construction_vehicle', 'trailer', 'truck']:
        return ['vehicle.moving', 'vehicle.parked', 'vehicle.stopped']
    if detection_name in ['barrier', 'traffic_cone']:
        return []
    raise ValueError(f'Error: {detection_name} is not a valid detection class.')

