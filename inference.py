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
from models import build_model

import contextlib
from detectron2.structures import BoxMode

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Intersections', 'scripts'))
from evaluation import evaluate_masked

video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']


def eval_coco(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=True)

    dataset_val = build_dataset(image_set='val', args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=True)
    base_ds = get_coco_api_from_dataset(dataset_val)
    test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir)


def eval_scenes100(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, _, postprocessors = build_model(args)
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
    results_all = inference_scenes100(model, postprocessors, detections, data_loader_val, device)
    with open('results_AP_base.json', 'w') as fp:
        json.dump(results_all, fp)


def inference_scenes100(model, postprocessors, detections, data_loader, device, is_moe=False):
    print('%d images' % len(detections))
    model.eval()
    with torch.no_grad():
        iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
        for samples, targets in tqdm.tqdm(data_loader, ascii=True, total=len(data_loader)):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            if is_moe:
                video_id_batch = [int(t['video_id'][0]) if 'video_id' in t else 999 for t in targets]
                video_id_batch = [('%03d' % v) for v in video_id_batch]
                outputs = model(samples, video_id_batch)
            else:
                outputs = model(samples)
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessors['bbox'](outputs, orig_target_sizes)
            if 'segm' in postprocessors.keys():
                target_sizes = torch.stack([t["size"] for t in targets], dim=0)
                results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
            res = {target['image_id'].item(): output for target, output in zip(targets, results)}
            for i in res:
                detections[i]['annotations'] = [{'bbox': list(map(float, b)), 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': int(l) - 1, 'score': float(s)} for s, l, b in zip(res[i]['scores'], res[i]['labels'], res[i]['boxes'])]

    results_all = {}
    results_avg = {}
    for video_id in video_id_list:
        detections_v = list(filter(lambda x: x['video_id'] == video_id, detections.values()))
        print(video_id, '%d images' % len(detections_v))
        detections_v = sorted(detections_v, key=lambda x: x['file_name'])
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            resuls_AP = evaluate_masked(video_id, detections_v)
        print(   '             %s' % '/'.join(resuls_AP['metrics']))
        for c in sorted(resuls_AP['results'].keys()):
            print('%10s  ' % c, end='')
            print('/'.join(map(lambda x: '%05.2f' % (x * 100), resuls_AP['results'][c])))
            if not c in results_avg:
                results_avg[c] = []
            results_avg[c].append(resuls_AP['results'][c])
        results_all[video_id] = resuls_AP
    print('videos average weighted')
    weighted = np.array(results_avg['weighted']).mean(axis=0)
    print('/'.join(map(lambda x: '%05.2f' % (x * 100), weighted)))
    results_all['avg'] = results_avg
    return results_all


def pseudo_label(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, _, postprocessors = build_model(args)
    model.to(device)
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()

    dataset_pl = build_dataset_scenes100('pl', args.input_scale, args)
    sampler = torch.utils.data.SequentialSampler(dataset_pl)
    data_loader = DataLoader(dataset_pl, args.batch_size, sampler=sampler, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=True)

    detections = {im['id']: im for im in copy.deepcopy(dataset_pl.annotations)}
    print('%d images' % len(detections))
    with torch.no_grad():
        iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
        for samples, targets in tqdm.tqdm(data_loader, ascii=True, total=len(data_loader)):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(samples)
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessors['bbox'](outputs, orig_target_sizes)
            if 'segm' in postprocessors.keys():
                target_sizes = torch.stack([t["size"] for t in targets], dim=0)
                results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
            res = {target['image_id'].item(): output for target, output in zip(targets, results)}
            for i in res:
                res[i]['scores'] = res[i]['scores'].detach().cpu().numpy()
                keep_indices = np.arange(0, len(res[i]['scores']))[res[i]['scores'] > args.refine_det_score_thres]
                detections[i]['annotations'] = [{'bbox': list(map(float, b)), 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': int(l) - 1, 'score': float(s)} for s, l, b in zip(res[i]['scores'][keep_indices], res[i]['labels'][keep_indices], res[i]['boxes'][keep_indices])]

    with open('scenes100_pl_x%.2f_s%.2f.json' % (args.input_scale, args.refine_det_score_thres), 'w') as fp:
        json.dump(detections, fp)


def bmeans_cluster(args):
    from sklearn.cluster import KMeans
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, _, postprocessors = build_model(args)
    model.to(device)
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()

    dataset_bmeans = build_dataset_scenes100('bmeans', args.input_scale, args)
    sampler = torch.utils.data.SequentialSampler(dataset_bmeans)
    data_loader = DataLoader(dataset_bmeans, args.batch_size, sampler=sampler, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=True)
    video_id_features = []
    video_features = []
    print('%d images' % len(dataset_bmeans.annotations))
    with torch.no_grad():
        for samples, targets in tqdm.tqdm(data_loader, ascii=True, total=len(data_loader)):
            video_id_features.extend(['%03d' % t['video_id'][0] for t in targets])
            samples = samples.to(device)
            features_all, _ = model.backbone(samples)
            video_features.append(torch.nn.functional.adaptive_avg_pool2d(features_all[1].decompose()[0], (7, 12)).view(len(targets), -1).detach().cpu())
    video_features = torch.cat(video_features, dim=0).detach().numpy()
    video_id_features = np.array(video_id_features)
    torch.cuda.empty_cache()

    print('running %s-Means for: %s %s' % (args.budget, video_features.shape, video_features.dtype))
    kmeans = KMeans(n_clusters=args.budget, random_state=0).fit(video_features)
    mapper = {'budget': args.budget, 'video_id_to_index': {}, 'used_indices': {}, 'un_used_indices': {b: True for b in range(0, args.budget)}}
    for v in video_id_list:
        cluster_ids = kmeans.labels_[video_id_features == v]
        i = np.argmax(np.bincount(cluster_ids))
        mapper['video_id_to_index'][v] = i
        mapper['used_indices'][i] = True
        if i in mapper['un_used_indices']:
            del mapper['un_used_indices'][i]
    print(mapper)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
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

    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
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
    parser.add_argument('--refine_det_score_thres', type=float, default=0.5, help='minimum detection score in pseudo annotation')
    # parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.opt == 'eval_coco':
        eval_coco(args)
    elif args.opt == 'eval_s100':
        eval_scenes100(args)
    elif args.opt == 'pl':
        pseudo_label(args)
    elif args.opt == 'bmeans':
        bmeans_cluster(args)

'''
python inference.py --with_box_refine --two_stage --num_workers 4 --batch_size 4 --opt eval_coco --coco_path ../MSCOCO2017 --resume checkpoint.pth
python inference.py --with_box_refine --two_stage --num_workers 4 --batch_size 4 --opt eval_s100 --resume checkpoint.pth
python inference.py --with_box_refine --two_stage --num_workers 6 --batch_size 4 --opt pl --resume checkpoint.pth --input_scale 1.25
python inference.py --with_box_refine --two_stage --num_workers 4 --batch_size 4 --opt bmeans --resume checkpoint.pth
'''
