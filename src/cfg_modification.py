import mmcv
import os.path as osp

from os import sep
from mmdet.apis import set_random_seed



def modify_cfg(cfg, params):
    from kitti_dataset import KittiTinyDataset

    DATA_ROOT = params['dataset']['root'] + sep + params['dataset']['name']

    cfg.custom_imports = dict(
        imports=['kitti_dataset'],
        allow_failed_imports=False)
    
    # Modify dataset type and path
    cfg.dataset_type = 'KittiTinyDataset'
    cfg.data_root = DATA_ROOT

    cfg.data.test.type = 'KittiTinyDataset'
    cfg.data.test.data_root = DATA_ROOT
    cfg.data.test.ann_file = 'train.txt'
    cfg.data.test.img_prefix = 'training/image_2'

    cfg.data.train.type = 'KittiTinyDataset'
    cfg.data.train.data_root = DATA_ROOT
    cfg.data.train.ann_file = 'train.txt'
    cfg.data.train.img_prefix = 'training/image_2'

    cfg.data.val.type = 'KittiTinyDataset'
    cfg.data.val.data_root = DATA_ROOT
    cfg.data.val.ann_file = 'val.txt'
    cfg.data.val.img_prefix = 'training/image_2'

    # modify num classes of the model in box head
    cfg.model.roi_head.bbox_head.num_classes = params['running']['num_classes']

    # If we need to finetune a model based on a pre-trained detector, we need to
    # use load_from to set the path of checkpoints.
    cfg.load_from = params['model']['checkpoint']  # 'checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco_20210526_095054-1f77628b.pth'

    # Set up working dir to save files and logs.
    cfg.work_dir =  params['running']['work_dir']
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # The original learning rate (LR) is set for 8-GPU training.
    # We divide it by 8 since we only use one GPU.
    cfg.optimizer.lr = float(params['running']['lr'])
    cfg.lr_config.warmup = None
    cfg.log_config.interval = 5  # images=50, bs=2 => it/epoch=25 => 5 log/epoch
    
    # We can also use tensorboard to log the training process
    cfg.log_config.hooks = [
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        ]
    
    cfg.runner.max_epochs = params['running']['max_epoch']
    
    # Change the evaluation metric since we use customized dataset.
    cfg.evaluation.metric = 'mAP'

    # We can set the evaluation interval to reduce the evaluation times
    cfg.evaluation.interval = 1

    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.checkpoint_config = dict(interval=1, filename_tmpl='model_epoch_{}.pth')

    # Set seed thus the results are more reproducible
    cfg.seed = params['running']['seed']
    set_random_seed(cfg.seed, deterministic=False)
    cfg.device = params['running']['device']
    cfg.gpu_ids = [0]  # range cant be saved as json
    return cfg


def set_mlflow_logger(cfg, params=None):
    # bad function
    flat_params = {}
    def get_all_items(params, prefix_key='', dlm='.'):
        if prefix_key != '':
            prefix_key += dlm
        for k in params:
            if isinstance(params[k], dict):
                get_all_items(
                    params[k], 
                    prefix_key=prefix_key + k
                    )
            else:
                flat_params[prefix_key + k] = params[k]

    if params is not None:        
        get_all_items(params)

    cfg.log_config.hooks.append(
        dict(
            type='MlflowLoggerHook', 
            exp_name=params['model']['exp_name'],
            log_model=False, 
            params=flat_params
            )
        )
    return cfg
