# nuScenes dev-kit constants (detection).

DETECTION_NAMES = [
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle',
    'traffic_cone', 'barrier'
]

ATTRIBUTE_NAMES = [
    'pedestrian.moving', 'pedestrian.sitting_lying_down', 'pedestrian.standing',
    'cycle.with_rider', 'cycle.without_rider',
    'vehicle.moving', 'vehicle.parked', 'vehicle.stopped'
]

TP_METRICS = ['trans_err', 'scale_err', 'orient_err', 'vel_err', 'attr_err']

