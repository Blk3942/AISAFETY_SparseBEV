_base_ = ['./r50_nuimg_704x256.py']

# v1.0-mini：先运行
#   python gen_sweep_info.py --data-root <Mini根目录> --version v1.0-mini
# 生成 nuscenes_infos_{train,val}_mini_sweep.pkl
dataset_root = '/root/autodl-tmp/nuScenes/Mini/'

data = dict(
    workers_per_gpu=4,
    train=dict(
        data_root=dataset_root,
        ann_file=dataset_root + 'nuscenes_infos_train_mini_sweep.pkl',
    ),
    val=dict(
        data_root=dataset_root,
        ann_file=dataset_root + 'nuscenes_infos_val_mini_sweep.pkl',
    ),
    test=dict(
        data_root=dataset_root,
        ann_file=dataset_root + 'nuscenes_infos_val_mini_sweep.pkl',
    ),
)
