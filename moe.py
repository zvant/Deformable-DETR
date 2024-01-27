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
import math
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
from util.misc import NestedTensor, inverse_sigmoid
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset, build_dataset_scenes100
from engine import evaluate, train_one_epoch
from models import build_model, DeformableDETR
from models.backbone import Backbone, Joiner
from torchvision.models._utils import IntermediateLayerGetter

import contextlib
from inference import inference_scenes100
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Intersections', 'scripts'))
# from evaluation import evaluate_masked

video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']


class DeformableDETRMoE_B1(DeformableDETR):
    def forward(self, samples, video_id_batch):
        return super(DeformableDETRMoE_B1, self).forward(samples)


class DeformableDETRMoE(DeformableDETR):
    def forward(self, samples, video_id_batch):
        """The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries. Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as (center_x, center_y, height, width). These values are normalized in [0, 1], relative to the size of each individual image (disregarding possible padding). See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of dictionnaries containing the two above keys for each decoder layer.
        """
        for k in self.backbone_moe_keys:
            self.backbone[0].body[k].curr_video_id = video_id_batch
        for lvl in self.output_embed_levels:
            self.class_embed[lvl].curr_video_id = video_id_batch
            self.bbox_embed[lvl].curr_video_id = video_id_batch

        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = torch.nn.functional.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}

        for k in self.backbone_moe_keys:
            self.backbone[0].body[k].curr_video_id = None
        for lvl in self.output_embed_levels:
            self.class_embed[lvl].curr_video_id = None
            self.bbox_embed[lvl].curr_video_id = None
        return out

    @classmethod
    def create_from_sup(cls, net, budget):
        if budget == 1:
            print('still use single model')
            net.__class__ = DeformableDETRMoE_B1
            return net

        if budget == 10:
            net.video_id_to_index = {'001': 3, '003': 6, '005': 4, '006': 6, '007': 6, '008': 7, '009': 6, '011': 1, '012': 3, '013': 0, '014': 2, '015': 1, '016': 1, '017': 5, '019': 5, '020': 1, '023': 4, '025': 8, '027': 1, '034': 2, '036': 6, '039': 4, '040': 3, '043': 3, '044': 5, '046': 8, '048': 4, '049': 3, '050': 3, '051': 0, '053': 3, '054': 4, '055': 3, '056': 8, '058': 1, '059': 7, '060': 2, '066': 7, '067': 6, '068': 2, '069': 7, '070': 9, '071': 8, '073': 7, '074': 8, '075': 8, '076': 3, '077': 4, '080': 7, '085': 5, '086': 2, '087': 4, '088': 8, '090': 8, '091': 8, '092': 6, '093': 8, '094': 3, '095': 3, '098': 3, '099': 3, '105': 0, '108': 8, '110': 3, '112': 2, '114': 4, '115': 4, '116': 8, '117': 5, '118': 1, '125': 1, '127': 8, '128': 6, '129': 3, '130': 6, '131': 7, '132': 3, '135': 7, '136': 2, '141': 1, '146': 7, '148': 1, '149': 1, '150': 6, '152': 2, '154': 7, '156': 6, '158': 6, '159': 3, '160': 1, '161': 7, '164': 3, '167': 1, '169': 4, '170': 2, '171': 1, '172': 1, '175': 3, '178': 8, '179': 4}
        elif budget == 100:
            net.video_id_to_index = {v: i for i, v in enumerate(video_id_list)}
        else:
            raise NotImplementedError
        print('base model: %d parameters' % sum([p.numel() for p in net.parameters()]))
        assert isinstance(net.backbone, Joiner)
        assert len(net.backbone) == 2
        assert isinstance(net.backbone[0], Backbone)
        assert isinstance(net.backbone[0].body, IntermediateLayerGetter)
        net.backbone_moe_keys = ['conv1', 'layer1']
        for k in net.backbone_moe_keys:
            net.backbone[0].body[k] = MakeMoE(net.backbone[0].body[k], budget, net.video_id_to_index)
        net.output_embed_levels = [0, 1, 2, 3, 4, 5, 6]
        assert len(net.class_embed) == len(net.bbox_embed) == len(net.output_embed_levels)
        for lvl in net.output_embed_levels:
            net.class_embed[lvl] = MakeMoE(net.class_embed[lvl], budget, net.video_id_to_index)
            net.bbox_embed[lvl] = MakeMoE(net.bbox_embed[lvl], budget, net.video_id_to_index)
        print('MoE model: %d parameters' % sum([p.numel() for p in net.parameters()]))
        net.__class__ = cls
        return net


class MakeMoE(torch.nn.Module):
    def __init__(self, net: torch.nn.Module, budget: int, video_id_to_index: dict):
        super(MakeMoE, self).__init__()
        self.experts = torch.nn.ModuleList([copy.deepcopy(net) for _ in range(budget)])
        self.video_id_to_index = video_id_to_index

    def forward(self, x: torch.Tensor):
        assert len(x) == len(self.curr_video_id)
        out = [self.experts[self.video_id_to_index[self.curr_video_id[i]]](x[i : i + 1, :]) for i in range(0, len(self.curr_video_id))]
        return torch.cat(out, dim=0)


def train_moe(args):
    utils.init_distributed_mode(args)
    assert not args.masks
    print(args)
    device = torch.device(args.device)
    # fix the seed for reproducibility
    # seed = args.seed + utils.get_rank()
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
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


    eval_iter_list = list(range(0, args.iters, args.eval_iters))
    if max(eval_iter_list) < args.iters - 1000:
        eval_iter_list.append(args.iters - 2)
    eval_iter_list = set(eval_iter_list)
    print('evaluate & checkpoint at:', eval_iter_list)

    model.train()
    criterion.train()
    train_iter = iter(data_loader_train)

    for i in tqdm.tqdm(range(0, args.iters), ascii=True, desc='training'):
        if i in eval_iter_list:
            model.eval()
            with contextlib.redirect_stdout(open(os.devnull, 'w')):
                results_all = inference_scenes100(model, postprocessors, copy.deepcopy(detections), data_loader_val, device, is_moe=True)
            print('videos average weighted: ' + '/'.join(map(lambda x: '%05.2f' % (x * 100), np.array(results_all['avg']['weighted']).mean(axis=0))))
            utils.save_on_master({'model': model.state_dict()}, os.path.join(args.output_dir, 'checkpoint_MoE_budget%d_it%06d.pth' % (args.budget, i)))
            model.train()

        try:
            samples, targets = next(train_iter)
        except:
            print('end of dataset reached, restart')
            train_iter = iter(data_loader_train)
            continue

        video_id_batch = [int(t['video_id'][0]) if 'video_id' in t else 999 for t in targets]
        video_id_batch = [('%03d' % v) for v in video_id_batch]
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples, video_id_batch)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()


def eval_moe_scenes100(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)
    model, _, postprocessors = build_model(args)
    model = DeformableDETRMoE.create_from_sup(model, args.budget)
    model.to(device)
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    dataset_val = build_dataset_scenes100('val', args.input_scale, args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=True)
    detections = {im['id']: im for im in copy.deepcopy(dataset_val.annotations)}
    for im in detections.values():
        im['file_name'] = os.path.basename(im['file_name'])
        im['annotations'] = []
    results_all = inference_scenes100(model, postprocessors, detections, data_loader_val, device, is_moe=True)
    print('videos average weighted')
    for c in ['person', 'vehicle', 'overall', 'weighted']:
        print('%10s  ' % c, end='')
        print('/'.join(map(lambda x: '%05.2f' % (x * 100), np.array(results_all['avg'][c]).mean(axis=0))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-5, type=float)
    # parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    # parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    # parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    # parser.add_argument('--epochs', default=50, type=int)
    # parser.add_argument('--lr_drop', default=40, type=int)
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
    parser.add_argument('--pl_file', type=str, default='')
    parser.add_argument('--iters', default=6002, type=int)
    parser.add_argument('--eval_iters', default=3000, type=int)
    parser.add_argument('--budget', type=int, default=10)
    # parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.opt == 'train':
        train_moe(args)
    elif args.opt == 'eval_s100':
        eval_moe_scenes100(args)

'''
python moe.py --with_box_refine --two_stage --num_workers 3 --batch_size 3 --opt train --resume checkpoint.pth --output_dir outputs --budget 1 --iters 2002 --eval_iters 4000 --pl_file scenes100_pl_sample.json
python moe.py --with_box_refine --two_stage --num_workers 3 --batch_size 3 --opt train --resume checkpoint.pth --output_dir outputs --budget 1 --iters 200005 --eval_iters 20000

python moe.py --with_box_refine --two_stage --num_workers 10 --batch_size 10 --opt eval_s100 --resume outputs/checkpoint_MoE_budget10_it005998.pth --budget 10
'''
