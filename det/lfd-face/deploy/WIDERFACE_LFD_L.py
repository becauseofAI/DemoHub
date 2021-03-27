# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
import shutil
import os
import time
import torch
# from lfd.execution.utils import set_random_seed, set_cudnn_backend
from lfd.model.backbone import LFDResNet
from lfd.model.neck import SimpleNeck
from lfd.model.head import LFDHead
from lfd.model.losses import *
from lfd.model import *

# assert torch.cuda.is_available(), 'GPU training supported only!'

memo = 'WIDERFACE L' \
       'head: share, path merge, with GN' \
       'FL as classification loss, loss weight is set to 1.0' \
       'IoULoss as regression loss, distance_to_bbox_mode is set to sigmoid, loss weight is set to 1.0'


# all config parameters will be stored in config_dict
config_dict = dict()


'''
build model ----------------------------------------------------------------------------------------------
'''
def prepare_model():
    # input image channels: BGR--3, gray--1
    config_dict['num_input_channels'] = 3

    # classification_loss = CrossEntropyLoss(
    #     reduction='mean',
    #     loss_weight=1.0
    # )
    classification_loss = FocalLoss(
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        reduction='mean',
        loss_weight=1.0
    )

    regression_loss = IoULoss(
        eps=1e-6,
        reduction='mean',
        loss_weight=1.0
    )

    # number of classes
    config_dict['num_classes'] = 1
    config_dict['backbone_init_param_file_path'] = None  # if no pretrained weights, set to None
    lfd_backbone = LFDResNet(
        block_mode='faster',  # affect block type
        stem_mode='fast',  # affect stem type
        body_mode=None,  # affect body architecture
        input_channels=config_dict['num_input_channels'],
        stem_channels=64,
        body_architecture=[4, 2, 2, 1, 1],
        body_channels=[64, 64, 64, 128, 128],
        out_indices=((0, 3), (1, 1), (2, 1), (3, 0), (4, 0)),
        frozen_stages=-1,
        activation_cfg=dict(type='ReLU', inplace=True),
        norm_cfg=dict(type='BatchNorm2d'),
        init_with_weight_file=config_dict['backbone_init_param_file_path'],
        norm_eval=False
    )

    lfd_neck = SimpleNeck(
        num_neck_channels=128,
        num_input_channels_list=lfd_backbone.num_output_channels_list,
        num_input_strides_list=lfd_backbone.num_output_strides_list,
        norm_cfg=dict(type='BatchNorm2d'),
        activation_cfg=dict(type='ReLU', inplace=True)
    )

    lfd_head = LFDHead(
        num_classes=config_dict['num_classes'],
        num_heads=len(lfd_neck.num_output_strides_list),
        num_input_channels=128,
        num_head_channels=128,
        num_conv_layers=2,
        activation_cfg=dict(type='ReLU', inplace=True),
        norm_cfg=dict(type='GroupNorm', num_groups=16),
        share_head_flag=True,
        merge_path_flag=True,
        classification_loss_type=type(classification_loss).__name__,
        regression_loss_type=type(regression_loss).__name__
    )
    config_dict['detection_scales'] = ((4, 20), (20, 40), (40, 80), (80, 160), (160, 320))
    config_dict['model'] = LFD(
        backbone=lfd_backbone,
        neck=lfd_neck,
        head=lfd_head,
        num_classes=config_dict['num_classes'],
        regression_ranges=config_dict['detection_scales'],
        gray_range_factors=(0.9, 1.1),
        point_strides=lfd_neck.num_output_strides_list,
        classification_loss_func=classification_loss,
        regression_loss_func=regression_loss,
        distance_to_bbox_mode='sigmoid'
    )


if __name__ == '__main__':
    prepare_model()
