# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import yaml

import numpy as np
from easydict import EasyDict as edict


config = edict()

config.OUTPUT_DIR = 'output'
config.LOG_DIR = 'log'
config.DATA_DIR = ''
config.POSE_GT_PATH = ''
config.POSE_PRED_PATH = ''
config.GPUS = '0'
config.WORKERS = 16
config.PRINT_FREQ = 20

# Cudnn related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True


# common params for NETWORK
config.MODEL = edict()
config.MODEL.NUM_JOINTS = 16
config.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
config.MODEL.DECONV_WITH_BIAS = False
config.MODEL.NUM_DECONV_LAYERS = 3
config.MODEL.NUM_DECONV_FILTERS = [256, 256, 256]
config.MODEL.NUM_DECONV_KERNELS = [4, 4, 4]
config.MODEL.FINAL_CONV_KERNEL = 1
config.MODEL.TARGET_TYPE = 'gaussian'
config.MODEL.HEATMAP_SIZE = [64, 64]
config.MODEL.SIGMA = 2
config.MODEL.NUM_LAYERS = 50
config.MODEL.NAME = 'resnet50'
config.MODEL.STYLE = 'pytorch'
config.MODEL.HEAD = ''
config.MODEL.REFINE_LAYERS = 3
config.MODEL.DECODER_LAYERS = 4
config.MODEL.PRETRAINED_POSE = ''
config.MODEL.PRETRAINED_PAR = ''

config.EXTRA_POSE = edict()
config.EXTRA_POSE.PRETRAINED_LAYERS = []
config.EXTRA_POSE.FINAL_CONV_KERNEL= 1
config.EXTRA_POSE.STAGE2 = edict()
config.EXTRA_POSE.STAGE2.NUM_MODULES = 1
config.EXTRA_POSE.STAGE2.NUM_BRANCHES = 2
config.EXTRA_POSE.STAGE2.BLOCK = 'BASIC'
config.EXTRA_POSE.STAGE2.NUM_BLOCKS = []
config.EXTRA_POSE.STAGE2.NUM_CHANNELS = []
config.EXTRA_POSE.STAGE2.FUSE_METHOD = 'SUM'

config.EXTRA_POSE.STAGE3 = edict()
config.EXTRA_POSE.STAGE3.NUM_MODULES = 1
config.EXTRA_POSE.STAGE3.NUM_BRANCHES = 2
config.EXTRA_POSE.STAGE3.BLOCK = 'BASIC'
config.EXTRA_POSE.STAGE3.NUM_BLOCKS = []
config.EXTRA_POSE.STAGE3.NUM_CHANNELS = []
config.EXTRA_POSE.STAGE3.FUSE_METHOD = 'SUM'

config.EXTRA_POSE.STAGE4 = edict()
config.EXTRA_POSE.STAGE4.NUM_MODULES = 1
config.EXTRA_POSE.STAGE4.NUM_BRANCHES = 2
config.EXTRA_POSE.STAGE4.BLOCK = 'BASIC'
config.EXTRA_POSE.STAGE4.NUM_BLOCKS = []
config.EXTRA_POSE.STAGE4.NUM_CHANNELS = []
config.EXTRA_POSE.STAGE4.FUSE_METHOD = 'SUM'

config.EXTRA_PAR = edict()
# config.EXTRA_PAR.PRETRAINED = ''
config.EXTRA_PAR.FINAL_CONV_KERNEL= 1
config.EXTRA_PAR.STAGE1 = edict()
config.EXTRA_PAR.STAGE1.NUM_MODULES = 1
config.EXTRA_PAR.STAGE1.NUM_BRANCHES = 2
config.EXTRA_PAR.STAGE1.BLOCK = 'BOTTLENECK'
config.EXTRA_PAR.STAGE1.NUM_BLOCKS = []
config.EXTRA_PAR.STAGE1.NUM_CHANNELS = []
config.EXTRA_PAR.STAGE1.FUSE_METHOD = 'SUM'

config.EXTRA_PAR.STAGE2 = edict()
config.EXTRA_PAR.STAGE2.NUM_MODULES = 1
config.EXTRA_PAR.STAGE2.NUM_BRANCHES = 2
config.EXTRA_PAR.STAGE2.BLOCK = 'BASIC'
config.EXTRA_PAR.STAGE2.NUM_BLOCKS = []
config.EXTRA_PAR.STAGE2.NUM_CHANNELS = []
config.EXTRA_PAR.STAGE2.FUSE_METHOD = 'SUM'

config.EXTRA_PAR.STAGE3 = edict()
config.EXTRA_PAR.STAGE3.NUM_MODULES = 1
config.EXTRA_PAR.STAGE3.NUM_BRANCHES = 2
config.EXTRA_PAR.STAGE3.BLOCK = 'BASIC'
config.EXTRA_PAR.STAGE3.NUM_BLOCKS = []
config.EXTRA_PAR.STAGE3.NUM_CHANNELS = []
config.EXTRA_PAR.STAGE3.FUSE_METHOD = 'SUM'

config.EXTRA_PAR.STAGE4 = edict()
config.EXTRA_PAR.STAGE4.NUM_MODULES = 1
config.EXTRA_PAR.STAGE4.NUM_BRANCHES = 2
config.EXTRA_PAR.STAGE4.BLOCK = 'BASIC'
config.EXTRA_PAR.STAGE4.NUM_BLOCKS = []
config.EXTRA_PAR.STAGE4.NUM_CHANNELS = []
config.EXTRA_PAR.STAGE4.FUSE_METHOD = 'SUM'



config.LOSS = edict()
config.LOSS.USE_OHEM = False
config.LOSS.TOPK = 8
config.LOSS.USE_TARGET_WEIGHT = True
config.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False
config.LOSS.OHEMTHRES = 0.9
config.LOSS.OHEMKEEP = 100000

# DATASET related params
config.DATASET = edict()
config.DATASET.ROOT = ''
config.DATASET.DATASET = 'mpii'
config.DATASET.TRAIN_SET = 'train'
config.DATASET.TEST_SET = 'valid'
config.DATASET.DATA_FORMAT = 'jpg'
config.DATASET.HYBRID_JOINTS_TYPE = ''
config.DATASET.SELECT_DATA = False
config.DATASET.NUM_CLASSES = 19
config.DATASET.EXTRA_TRAIN_SET = ''
config.DATASET.TRAIN_IMROOT = ''
config.DATASET.VAL_IMROOT = ''
config.DATASET.TEST_IMROOT = ''
config.DATASET.TRAIN_SEGROOT = ''
config.DATASET.VAL_SEGROOT = ''
config.DATASET.NUM_JOINTS = 16




# training data augmentation
config.DATASET.FLIP = True
config.DATASET.SCALE_FACTOR = 0.25
config.DATASET.ROT_FACTOR = 30
config.DATASET.PROB_HALF_BODY = 0.0
config.DATASET.NUM_JOINTS_HALF_BODY = 8
config.DATASET.COLOR_RGB = False

# search
config.SEARCH = edict()
config.SEARCH.W_LR = 0.001
config.SEARCH.LR_FACTOR = 0.2
config.SEARCH.LR_STEP = [20, 40]
config.SEARCH.W_LR_MIN = 0.00001
config.SEARCH.MOMENTUM = 0.9
config.SEARCH.WEIGHT_DECAY = 0.0001
config.SEARCH.NESTEROV = False
config.SEARCH.INIT_EPOCHS = 20
config.SEARCH.EPOCHS = 60
config.SEARCH.BATCH_SIZE = 32
config.SEARCH.LAYERS = 10
config.SEARCH.INIT_CHANNELS = 32
config.SEARCH.RESUME = False
config.SEARCH.APLHA_LR = 0.0004
config.SEARCH.ALPHA_WEIGHT_DECAY = 0.0001
config.SEARCH.SEED = 2
config.SEARCH.W_GRADconfigLIP = 5
config.SEARCH.TRAIN_SET = 'train_train'
config.SEARCH.MINI_SET = 'train_a'
config.SEARCH.TEST_SET = 'train_valid'
config.SEARCH.NAME = 'mpii'
config.SEARCH.PATH = 'searchs'



# train
config.TRAIN = edict()

config.TRAIN.LR_FACTOR = 0.1
config.TRAIN.LR_STEP = [90, 110]
config.TRAIN.LR = 0.001

config.TRAIN.OPTIMIZER = 'adam'
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WD = 0.0001
config.TRAIN.NESTEROV = False
config.TRAIN.LAYERS = 10
config.TRAIN.INIT_CHANNELS = 32
config.TRAIN.GAMMA1 = 0.99
config.TRAIN.GAMMA2 = 0.0
config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.EPOCHS = 140
config.TRAIN.RESUME = False
config.TRAIN.CHECKPOINT = ''
config.TRAIN.BATCH_SIZE = 32
config.TRAIN.SHUFFLE = True

config.TRAIN.TRAIN_SET = 'train'
config.TRAIN.MINI_SET='mini'
config.TRAIN.TEST_SET = 'valid'
config.TRAIN.SAMPLE_SET='sample'
config.TRAIN.NAME = 'mpii'
config.TRAIN.PATH = 'augments'
config.TRAIN.GENOTYPE = None
config.TRAIN.IGNORE_LABEL = -1
config.TRAIN.SCALE_FACTOR = 16
config.TRAIN.NUM_SAMPLES = 0
config.TRAIN.FLIP = True

# testing
config.TEST = edict()

# size of images for each device
config.TEST.BATCH_SIZE = 32
# Test Model Epoch
config.TEST.FLIP_TEST = False
config.TEST.POST_PROCESS = True
config.TEST.SHIFT_HEATMAP = True
config.TEST.USE_GT_BBOX = False

#nms
config.TEST.IMAGE_THRE = 0.1
config.TEST.NMS_THRE = 0.6
config.TEST.SOFT_NMS = False
config.TEST.OKS_THRE = 0.5
config.TEST.IN_VIS_THRE = 0.0
config.TEST.COCO_BBOX_FILE = ''
config.TEST.BBOX_THRE = 1.0
config.TEST.MODEL_FILE = ''
config.TEST.NUM_SAMPLES = 0
config.TEST.SCALE_LIST = [1]
config.TEST.TEST_SET=''

#debug
config.DEBUG = edict()
config.DEBUG.DEBUG = False
config.DEBUG.SAVE_BATCH_IMAGES_GT = False
config.DEBUG.SAVE_BATCH_IMAGES_PRED = False
config.DEBUG.SAVE_HEATMAPS_GT = False
config.DEBUG.SAVE_HEATMAPS_PRED = False



def _update_dict(k, v):
    if k == 'DATASET':
        if 'MEAN' in v and v['MEAN']:
            v['MEAN'] = np.array([eval(x) if isinstance(x, str) else x
                                  for x in v['MEAN']])
        if 'STD' in v and v['STD']:
            v['STD'] = np.array([eval(x) if isinstance(x, str) else x
                                 for x in v['STD']])
    if k == 'MODEL':
        if 'EXTRA' in v and 'HEATMAP_SIZE' in v['EXTRA']:
            if isinstance(v['EXTRA']['HEATMAP_SIZE'], int):
                v['EXTRA']['HEATMAP_SIZE'] = np.array(
                    [v['EXTRA']['HEATMAP_SIZE'], v['EXTRA']['HEATMAP_SIZE']])
            else:
                v['EXTRA']['HEATMAP_SIZE'] = np.array(
                    v['EXTRA']['HEATMAP_SIZE'])
        if 'IMAGE_SIZE' in v:
            if isinstance(v['IMAGE_SIZE'], int):
                v['IMAGE_SIZE'] = np.array([v['IMAGE_SIZE'], v['IMAGE_SIZE']])
            else:
                v['IMAGE_SIZE'] = np.array(v['IMAGE_SIZE'])
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    expconfigonfig = None
    with open(config_file) as f:
        expconfigonfig = edict(yaml.load(f,Loader=yaml.FullLoader))
        for k, v in expconfigonfig.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))


def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)


def update_dir(model_dir, log_dir, data_dir):
    if model_dir:
        config.OUTPUT_DIR = model_dir

    if log_dir:
        config.LOG_DIR = log_dir

    if data_dir:
        config.DATA_DIR = data_dir

    config.DATASET.ROOT = os.path.join(
            config.DATA_DIR, config.DATASET.ROOT)

    config.TEST.COCO_BBOX_FILE = os.path.join(
            config.DATA_DIR, config.TEST.COCO_BBOX_FILE)

    config.MODEL.PRETRAINED = os.path.join(
            config.DATA_DIR, config.MODEL.PRETRAINED)


def get_model_name(cfg):
    name = cfg.MODEL.NAME
    full_name = cfg.MODEL.NAME
    extra = cfg.MODEL
    if name in ['pose_resnet']:
        name = '{model}_{num_layers}'.format(
            model=name,
            num_layers=extra.NUM_LAYERS)
        deconv_suffix = ''.join(
            'd{}'.format(num_filters)
            for num_filters in extra.NUM_DECONV_FILTERS)
        full_name = '{height}x{width}_{name}_{deconv_suffix}'.format(
            height=cfg.MODEL.IMAGE_SIZE[1],
            width=cfg.MODEL.IMAGE_SIZE[0],
            name=name,
            deconv_suffix=deconv_suffix)
    else:
        raise ValueError('Unkown model: {}'.format(cfg.MODEL))

    return name, full_name


if __name__ == '__main__':
    import sys
    genconfigonfig(sys.argv[1])
