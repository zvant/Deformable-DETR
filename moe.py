# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import sys
import os
import argparse
import datetime
import json
import tqdm
import random
import time
import copy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset, build_dataset_scenes100
from engine import evaluate, train_one_epoch
from models import build_model, DeformableDETR

import contextlib
from detectron2.structures import BoxMode

from inference import inference_scenes100
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Intersections', 'scripts'))
# from evaluation import evaluate_masked

video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']


class DeformableDETRMoE(DeformableDETR):
    def forward(self, samples, video_id_batch):
        return super(DeformableDETRMoE, self).forward(samples)

    @classmethod
    def create_from_sup(cls, net, budget):
        if budget == 1:
            self.video_id_to_index = {v: 0 for v in video_id_list}
        elif budget == 10:
            self.video_id_to_index = {'001': 3, '003': 6, '005': 4, '006': 6, '007': 6, '008': 7, '009': 6, '011': 1, '012': 3, '013': 0, '014': 2, '015': 1, '016': 1, '017': 5, '019': 5, '020': 1, '023': 4, '025': 8, '027': 1, '034': 2, '036': 6, '039': 4, '040': 3, '043': 3, '044': 5, '046': 8, '048': 4, '049': 3, '050': 3, '051': 0, '053': 3, '054': 4, '055': 3, '056': 8, '058': 1, '059': 7, '060': 2, '066': 7, '067': 6, '068': 2, '069': 7, '070': 9, '071': 8, '073': 7, '074': 8, '075': 8, '076': 3, '077': 4, '080': 7, '085': 5, '086': 2, '087': 4, '088': 8, '090': 8, '091': 8, '092': 6, '093': 8, '094': 3, '095': 3, '098': 3, '099': 3, '105': 0, '108': 8, '110': 3, '112': 2, '114': 4, '115': 4, '116': 8, '117': 5, '118': 1, '125': 1, '127': 8, '128': 6, '129': 3, '130': 6, '131': 7, '132': 3, '135': 7, '136': 2, '141': 1, '146': 7, '148': 1, '149': 1, '150': 6, '152': 2, '154': 7, '156': 6, '158': 6, '159': 3, '160': 1, '161': 7, '164': 3, '167': 1, '169': 4, '170': 2, '171': 1, '172': 1, '175': 3, '178': 8, '179': 4}
        elif budget == 100:
            self.video_id_to_index = {v: i for i, v in enumerate(video_id_list)}
        else:
            raise NotImplementedError
        net.__class__ = cls
        return net


def train_moe(args):
    utils.init_distributed_mode(args)
    assert not args.masks
    print(args)
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model, criterion, postprocessors = build_model(args)

    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=True)
    model = DeformableDETRMoE.create_from_sup(model, args.budget)
    model.to(device)

    assert not args.distributed
    dataset_train = build_dataset_scenes100('train', 1.0, args)
    dataset_val = build_dataset_scenes100('val', 1.0, args)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size*3, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers*3, pin_memory=True)

    detections = {im['id']: im for im in copy.deepcopy(dataset_val.annotations)}
    for im in detections.values():
        im['file_name'] = os.path.basename(im['file_name'])
        im['annotations'] = []

    # just use a small LR for all modules
    if args.sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    print("Start training")
    for epoch in range(args.start_epoch, args.epochs):
        print('epoch %d/%d' % (epoch + 1, args.epochs))
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm, is_moe=True)
        lr_scheduler.step()
        utils.save_on_master({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
        }, os.path.join(args.output_dir, 'checkpoint_MoE_ep%d.pth' % (epoch + 1)))

        print('evaluating')
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            results_all = inference_scenes100(model, postprocessors, copy.deepcopy(detections), data_loader_val, device)
        print('videos average weighted: ' + '/'.join(map(lambda x: '%05.2f' % (x * 100), np.array(results_all['avg']['weighted']).mean(axis=0))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=1e-5, type=float)
    # parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    # parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    # parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    # parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')
    parser.add_argument('--sgd', action='store_true')
    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')
    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None, help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str, help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true', help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'), help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float, help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int, help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int, help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int, help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    # * Segmentation
    parser.add_argument('--masks', action='store_true', help="Train segmentation head if the flag is provided")
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false', help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float, help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float, help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float, help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='.', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    parser.add_argument('--opt', type=str)
    parser.add_argument('--input_scale', type=float, default=1.0)
    parser.add_argument('--budget', type=int, default=10)
    # parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.opt == 'train':
        train_moe(args)

'''
python moe.py --with_box_refine --two_stage --num_workers 2 --batch_size 2 --opt train --resume checkpoint.pth --epochs 2 --lr_drop 2
'''
