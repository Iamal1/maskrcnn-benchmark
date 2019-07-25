# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints

from torch.distributions.beta import Beta
from PIL import Image
import logging
logger = logging.getLogger("maskrcnn_benchmark.coco")
import numpy

min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms

    def __getitem__(self, idx):
    #     '''
    #     img is tensor now
    #     '''
    #     img_a, target_a, idx_a = self.get_one_item(idx)
    #     img_b, target_b, idx_b = self.get_one_item((idx+1) % len(self.ids))
    #     #merge them
    #     #merge img
    #     m = Beta(torch.tensor([1.5]), torch.tensor([1.5]))
    #     cof_a = m.sample()
    #     #cof_a = 0.5
    #     c,ha,wa = img_a.shape
    #     c,hb,wb = img_b.shape
    #     h,w = (max(ha,hb),max(wa,wb))
    #     img = img_a.new_zeros((c,h,w))
    #     img[:,:ha,:wa] = cof_a * img_a
    #     img[:,:hb,:wb] = (1-cof_a) * img_b

    #     #merge labels and masks
    #     boxes = torch.cat([target_a.bbox,target_b.bbox],dim=0)
    #     target = BoxList(boxes, (w,h), mode="xyxy")
        
    #     classes = torch.cat([target_a.get_field('labels'),target_b.get_field('labels')],dim=0)
    #     target.add_field("labels", classes)

    #     masks = target_a.get_field("masks").instances.polygons + target_b.get_field("masks").instances.polygons
    #     masks = SegmentationMask(masks, (w,h), mode='poly')
    #     target.add_field("masks", masks)

    #    # #add marks
    #    # marks = [1]*target_a.bbox.size(0) +  [0] * target_b.bbox.size(0)
    #    # target.add_field("marks", torch.tensor(marks))
    #     cofs = [cof_a]*target_a.bbox.size(0) +  [1-cof_a] * target_b.bbox.size(0)
    #     target.add_field('cofs',torch.tensor(cofs))
        
    #     return img, target, idx

    # def get_one_item(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        # NOTE masks should be [[[]]]
        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size, mode='poly')
        target.add_field("masks", masks)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
