# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import os
import json
from pathlib import Path
from PIL import Image

import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T


class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])
        video_id = torch.tensor([int(target['video_id'])]) if 'video_id' in target else torch.tensor([-1])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        target['video_id'] = video_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "images" / "train2017", root / "DeformableDERT" / 'train2017_cocostyle.json'),
        "val": (root / "images" / "val2017", root / "DeformableDERT" / 'val2017_cocostyle.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    return dataset


class CocoDetectionScenes100:
    video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']

    def __init__(self, split, transforms):
        from detectron2.structures import BoxMode
        if split == 'val':
            self.annotations = []
            for video_id in self.video_id_list:
                root_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'Intersections', 'images', 'annotated', video_id)
                img_dir = os.path.join(root_dir, 'unmasked')
                with open(os.path.join(root_dir, 'annotations.json'), 'r') as fp:
                    annotations_v = json.load(fp)
                for im in annotations_v:
                    im['video_id'] = video_id
                    im['file_name'] = os.path.join(img_dir, im['file_name'])
                self.annotations.extend(annotations_v)
            ann_id = 1
            for i, im in enumerate(self.annotations):
                im['id'] = i + 1
                im['image_id'] = im['id']
                for ann in im['annotations']:
                    assert ann['bbox_mode'] == BoxMode.XYXY_ABS
                    del ann['bbox_mode']
                    ann['category_id'] = ann['category_id'] + 1
                    ann['iscrowd'] = 0
                    x1, y1, x2, y2 = ann['bbox']
                    ann['bbox'] = [x1, y1, x2 - x1, y2 - y1]
                    ann['area'] = ann['bbox'][3] - ann['bbox'][2]
                    ann['segmentation'] = []
                    ann['image_id'] = im['id']
                    ann['id'] = ann_id
                    ann_id += 1

        elif split == 'pl' or split == 'bmeans': # unlabeled images for pseudo-labeling & BMeans
            self.annotations = []
            for video_id in self.video_id_list:
                root_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'Intersections', 'images', 'train_lmdb', video_id)
                img_dir = os.path.join(root_dir, 'jpegs')
                with open(os.path.join(root_dir, 'frames.json'), 'r') as fp:
                    frames = json.load(fp)
                ifilelist = frames['ifilelist'][:: (1000 if split == 'pl' else 3600)]
                annotations_v = []
                for f in ifilelist:
                    annotations_v.append({'video_id': video_id, 'width': frames['meta']['video']['W'], 'height': frames['meta']['video']['H'], 'file_name': os.path.join(img_dir, f), 'annotations': []})
                self.annotations.extend(annotations_v)
            for i, im in enumerate(self.annotations):
                im['id'] = i + 1
                im['image_id'] = im['id']

        elif split == 'train': # use pseudo labeled images
            with open(os.path.join(os.path.dirname(__file__), '..', 'scenes100_pl_1.25.json'), 'r') as fp:
                self.annotations = list(json.load(fp).values())
            ann_id = 1
            for i, im in enumerate(self.annotations):
                im['id'] = i + 1
                im['image_id'] = im['id']
                for ann in im['annotations']:
                    assert ann['bbox_mode'] == BoxMode.XYXY_ABS
                    del ann['bbox_mode']
                    ann['category_id'] = ann['category_id'] + 1
                    ann['iscrowd'] = 0
                    x1, y1, x2, y2 = ann['bbox']
                    ann['bbox'] = [x1, y1, x2 - x1, y2 - y1]
                    ann['area'] = ann['bbox'][3] - ann['bbox'][2]
                    ann['segmentation'] = []
                    ann['image_id'] = im['id']
                    ann['id'] = ann_id
                    ann_id += 1

        else:
            raise NotImplementedError

        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(False)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img = Image.open(self.annotations[idx]['file_name']).convert('RGB')
        target = self.annotations[idx]['annotations']
        target = {'image_id': self.annotations[idx]['id'], 'video_id': self.annotations[idx]['video_id'], 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def build_dataset_scenes100(split, scale_factor, args):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    if split == 'train':
        assert abs(scale_factor - 1.0) < 1e-5 # always use default in training
        tf = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])
    elif split == 'val' or split == 'pl' or split == 'bmeans':
        tf = T.Compose([
            T.RandomResize([int(800 * scale_factor)], max_size=int(1333 * scale_factor)),
            normalize,
        ])
    else:
        raise NotImplementedError
    dataset = CocoDetectionScenes100(split, tf)
    return dataset
