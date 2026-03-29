"""Visualize nuScenes camera + BEV: ground truth (top) vs model prediction (bottom)."""
import os
import utils
import logging
import argparse
import importlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import box_in_image
from models.utils import VERSION


classname_to_color = {
    'car': (255, 158, 0),
    'pedestrian': (0, 0, 230),
    'trailer': (255, 140, 0),
    'truck': (255, 99, 71),
    'bus': (255, 127, 80),
    'motorcycle': (255, 61, 99),
    'construction_vehicle': (233, 150, 70),
    'bicycle': (220, 20, 60),
    'barrier': (112, 128, 144),
    'traffic_cone': (47, 79, 79),
}


def convert_to_nusc_box(
    bboxes, class_names, scores=None, labels=None, names=None,
    score_threshold=0.3, lift_center=False,
):
    results = []
    for q in range(bboxes.shape[0]):
        score = scores[q] if scores is not None else 1.0
        if score < score_threshold:
            continue

        label = int(labels[q]) if labels is not None else 0
        if names is not None:
            name = names[q]
        else:
            name = class_names[label] if 0 <= label < len(class_names) else class_names[-1]

        if name not in class_names:
            name = class_names[-1]

        bbox = bboxes[q].copy()
        if lift_center:
            bbox[2] += bbox[5] * 0.5

        orientation = Quaternion(axis=[0, 0, 1], radians=bbox[6])

        box = Box(
            center=[bbox[0], bbox[1], bbox[2]],
            size=[bbox[4], bbox[3], bbox[5]],
            orientation=orientation,
            score=score,
            label=label,
            velocity=(bbox[7], bbox[8], 0) if bboxes.shape[1] > 7 else (0, 0, 0),
            name=name,
        )
        results.append(box)

    return results


def viz_bbox_block(nusc, bboxes, data_info, fig, gs, row_start, block_title, linewidth=1.5):
    cam_types = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT',
    ]

    for cam_id, cam_type in enumerate(cam_types):
        sample_data_token = nusc.get('sample', data_info['token'])['data'][cam_type]

        sd_record = nusc.get('sample_data', sample_data_token)
        cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        intrinsic = np.array(cs_record['camera_intrinsic'])

        img_path = nusc.get_sample_data_path(sample_data_token)
        img_size = (sd_record['width'], sd_record['height'])

        r = row_start + cam_id // 3
        c = cam_id % 3
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(Image.open(img_path))

        for bbox in bboxes:
            bbox = bbox.copy()
            bbox.rotate(Quaternion(data_info['lidar2ego_rotation']))
            bbox.translate(np.array(data_info['lidar2ego_translation']))
            bbox.translate(-np.array(cs_record['translation']))
            bbox.rotate(Quaternion(cs_record['rotation']).inverse)

            if box_in_image(bbox, intrinsic, img_size):
                col = classname_to_color.get(bbox.name, (200, 200, 200))
                c_rgb = np.array(col) / 255.0
                bbox.render(ax, view=intrinsic, normalize=True, colors=(c_rgb, c_rgb, c_rgb), linewidth=linewidth)

        ax.axis('off')
        title = cam_type if cam_id > 0 else '%s\n%s' % (block_title, cam_type)
        ax.set_title(title, fontsize=8)
        ax.set_xlim(0, img_size[0])
        ax.set_ylim(img_size[1], 0)

    sample = nusc.get('sample', data_info['token'])
    lidar_data_token = sample['data']['LIDAR_TOP']

    ax = fig.add_subplot(gs[row_start:row_start + 2, 3])
    nusc.explorer.render_sample_data(lidar_data_token, with_anns=False, ax=ax, verbose=False)
    ax.axis('off')
    ax.set_title(block_title + '\nLIDAR_TOP', fontsize=8)
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)

    sd_record = nusc.get('sample_data', lidar_data_token)
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])

    for bbox in bboxes:
        bbox = bbox.copy()
        bbox.rotate(Quaternion(cs_record['rotation']))
        bbox.translate(np.array(cs_record['translation']))
        bbox.rotate(Quaternion(pose_record['rotation']))
        yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
        bbox.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)

        col = classname_to_color.get(bbox.name, (200, 200, 200))
        c_rgb = np.array(col) / 255.0
        bbox.render(ax, view=np.eye(4), colors=(c_rgb, c_rgb, c_rgb))


def main():
    parser = argparse.ArgumentParser(description='Visualize GT vs prediction on nuScenes cameras + BEV')
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--override', nargs='+', action=DictAction)
    parser.add_argument('--score_threshold', type=float, default=0.3)
    parser.add_argument('--max_samples', type=int, default=16, help='Max val frames to dump')
    parser.add_argument('--out_dir', type=str, default='outputs/gt_pred_viz')
    args = parser.parse_args()

    cfgs = Config.fromfile(args.config)
    if args.override is not None:
        cfgs.merge_from_dict(args.override)

    class_names = list(cfgs.class_names)

    importlib.import_module('models')
    importlib.import_module('loaders')

    from mmcv.utils.logging import logger_initialized
    logger_initialized['root'] = logging.Logger(__name__, logging.WARNING)
    logger_initialized['mmcv'] = logging.Logger(__name__, logging.WARNING)

    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1

    os.makedirs(args.out_dir, exist_ok=True)

    utils.init_logging(None, cfgs.debug)
    logging.info('Using GPU: %s' % torch.cuda.get_device_name(0))
    set_random_seed(0, deterministic=True)

    logging.info('Loading validation set from %s' % cfgs.data.val.data_root)
    val_dataset = build_dataset(cfgs.data.val)
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfgs.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=0,
    )

    logging.info('Creating model: %s' % cfgs.model.type)
    model = build_model(cfgs.model)
    model.cuda()
    model = MMDataParallel(model, [0])

    logging.info('Loading checkpoint from %s' % args.weights)
    checkpoint = load_checkpoint(
        model, args.weights, map_location='cuda', strict=True,
        logger=logging.Logger(__name__, logging.ERROR),
    )
    if 'version' in checkpoint:
        VERSION.name = checkpoint['version']

    ann_path = cfgs.data.val.ann_file
    if 'mini' in ann_path or 'v1.0-mini' in (cfgs.data.val.data_root or ''):
        nusc = NuScenes(version='v1.0-mini', dataroot=cfgs.data.val.data_root, verbose=False)
    else:
        nusc = NuScenes(version='v1.0-trainval', dataroot=cfgs.data.val.data_root, verbose=False)

    n_dump = min(args.max_samples, len(val_dataset))
    logging.info('Dumping %d samples to %s' % (n_dump, args.out_dir))

    for i, data in enumerate(val_loader):
        if i >= n_dump:
            break

        model.eval()
        with torch.no_grad():
            results = model(return_loss=False, rescale=True, **data)
            results = results[0]['pts_bbox']

        pred_tensor = results['boxes_3d'].tensor.numpy()
        if pred_tensor.shape[1] < 9:
            pad = np.zeros((pred_tensor.shape[0], 9 - pred_tensor.shape[1]))
            pred_tensor = np.concatenate([pred_tensor, pad], axis=1)

        bboxes_pred = convert_to_nusc_box(
            pred_tensor,
            class_names,
            scores=results['scores_3d'].numpy(),
            labels=results['labels_3d'].numpy(),
            score_threshold=args.score_threshold,
            lift_center=True,
        )

        ann = val_dataset.get_ann_info(i)
        gt_tensor = ann['gt_bboxes_3d'].tensor.numpy()
        gt_labels = ann['gt_labels_3d']
        valid = gt_labels >= 0
        gt_tensor = gt_tensor[valid]
        gt_labels = gt_labels[valid]
        if gt_tensor.shape[0] and gt_tensor.shape[1] < 9:
            pad = np.zeros((gt_tensor.shape[0], 9 - gt_tensor.shape[1]))
            gt_tensor = np.concatenate([gt_tensor, pad], axis=1)

        if gt_tensor.shape[0] > 0:
            bboxes_gt = convert_to_nusc_box(
                gt_tensor,
                class_names,
                scores=np.ones(gt_tensor.shape[0]),
                labels=gt_labels,
                score_threshold=0.0,
                lift_center=True,
            )
        else:
            bboxes_gt = []

        info = val_dataset.data_infos[i]
        fig = plt.figure(figsize=(15.5, 10))
        gs = GridSpec(4, 4, figure=fig, hspace=0.15, wspace=0.05)

        viz_bbox_block(nusc, bboxes_gt, info, fig, gs, row_start=0, block_title='Ground truth', linewidth=2.0)
        viz_bbox_block(nusc, bboxes_pred, info, fig, gs, row_start=2, block_title='Prediction', linewidth=1.5)

        fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.02, hspace=0.12, wspace=0.06)
        out_path = os.path.join(args.out_dir, 'gt_pred_%04d.jpg' % i)
        plt.savefig(out_path, dpi=160, bbox_inches='tight')
        plt.close()
        logging.info('Saved %s' % out_path)


if __name__ == '__main__':
    main()
