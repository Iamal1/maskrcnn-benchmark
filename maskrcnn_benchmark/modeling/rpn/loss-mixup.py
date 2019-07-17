# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
This file contains specific functions for computing losses on the RPN
file
"""

import torch
from torch.nn import functional as F

from .utils import concat_box_prediction_layers

from ..balanced_positive_negative_sampler import BalancedPositiveNegativeSampler
from ..utils import cat

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist

import numpy

class RPNLossComputation(object):
    """
    This class computes the RPN loss.
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder,
                 generate_labels_func):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        # self.target_preparator = target_preparator
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.copied_fields = ['cofs']
        self.generate_labels_func = generate_labels_func
        self.discard_cases = ['not_visibility', 'between_thresholds']

    def match_targets_to_anchors(self, anchor, target, copied_fields=[]):
        match_quality_matrix = boxlist_iou(target, anchor)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # RPN doesn't need any fields from target
        # for creating the labels, so clear them all
        target = target.copy_with_fields(copied_fields)
        # get the targets corresponding GT for each anchor
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        
        return matched_targets

    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        cofs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            matched_targets = self.match_targets_to_anchors(
                anchors_per_image, targets_per_image, self.copied_fields
            )

            matched_idxs = matched_targets.get_field("matched_idxs")
            labels_per_image, cof_per_image = self.generate_labels_func(matched_targets)
            labels_per_image = labels_per_image.to(dtype=torch.float32)
            # Background (negative examples)
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0
            # set bg coefficient to 1.0
            cof_per_image[bg_indices] = 1
            # discard anchors that go out of the boundaries of the image
            if "not_visibility" in self.discard_cases:
                labels_per_image[~anchors_per_image.get_field("visibility")] = -1

            # discard indices that are between thresholds
            if "between_thresholds" in self.discard_cases:
                inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, anchors_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
            cofs.append(cof_per_image)
        return labels, regression_targets, cofs


    def __call__(self, anchors, objectness, box_regression, targets):
        """
        Arguments:
            anchors (list[BoxList])
            objectness (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor)
        """
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        
        labels, regression_targets, cofs = self.prepare_targets(anchors, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        
        batch_size = len(labels)
        
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)
        
        
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness, box_regression = \
                concat_box_prediction_layers(objectness, box_regression)

        objectness = objectness.squeeze()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        cofs = torch.cat(cofs, dim=0)
        sampled_cofs = cofs[sampled_inds]
        cofs_set = set(sampled_cofs.cpu().numpy())
        cof_masks={}
        sampled_cofs_np = sampled_cofs.cpu().numpy()
        for cof in cofs_set:
            if cof < 1:
                cof_mask = torch.from_numpy((sampled_cofs_np==cof).astype(numpy.uint8))
                cof_masks[cof]=cof_mask
            else:
                cof_mask_bg = torch.from_numpy((sampled_cofs_np==cof).astype(numpy.uint8))
        
        objectness_losses = []
        box_losses = []
        for cof, cof_mask in cof_masks.items():
            cof_mask_all = cof_mask | cof_mask_bg
            cof_pos_mask = torch.nonzero(cof_mask).squeeze(1)
            sampled_pos_inds_cof = sampled_pos_inds[cof_pos_mask]
            # contains pos and bg: sampled_inds_cof = sampled_inds[cof_mask_all]
            sampled_inds_cof = sampled_inds[cof_mask]
            
            objectness_loss_cof = F.binary_cross_entropy_with_logits(objectness[sampled_inds_cof], labels[sampled_inds_cof])
            objectness_losses.append(objectness_loss_cof*cof)
            box_loss_cof = get_masked_loss(box_regression, regression_targets, sampled_pos_inds_cof)
            box_losses.append(box_loss_cof*cof)
        #handle bg alone
        objectness_loss_bg = F.binary_cross_entropy_with_logits(objectness[sampled_inds[cof_mask_bg]], labels[sampled_inds[cof_mask_bg]], reduction='mean')
        #objectness_losses.append(objectness_loss_bg)    
        objectness_loss_fg = get_total_loss(objectness_losses) / batch_size
        bg_ratio = cof_mask_bg.sum().item()/sampled_cofs.numel()
        objectness_loss = (1 - bg_ratio) * objectness_loss_fg + bg_ratio * objectness_loss_bg
        
        box_loss = get_total_loss(box_losses) / batch_size
        
        #objectness_loss = F.binary_cross_entropy_with_logits(objectness[sampled_inds], labels[sampled_inds], weight=sampled_cofs)
        
        return objectness_loss, box_loss

# This function should be overwritten in RetinaNet
# overwrite for mixup
def generate_rpn_labels(matched_targets):
    matched_idxs = matched_targets.get_field("matched_idxs")
    labels_per_image = matched_idxs >= 0
    
    cof_per_image = matched_targets.get_field("cofs")
    return labels_per_image, cof_per_image

def get_total_loss(losses):
    loss = losses[0]
    for i in range(1,len(losses)):
        loss += losses[i]
    return loss

def get_masked_loss(box_regression, regression_targets, sampled_pos_inds):
    box_loss = smooth_l1_loss(
        box_regression[sampled_pos_inds],
        regression_targets[sampled_pos_inds],
        beta=1.0 / 9,
        size_average=True,
    )

    return box_loss


def make_rpn_loss_evaluator(cfg, box_coder):
    matcher = Matcher(
        cfg.MODEL.RPN.FG_IOU_THRESHOLD,
        cfg.MODEL.RPN.BG_IOU_THRESHOLD,
        allow_low_quality_matches=True,
    )

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION
    )

    loss_evaluator = RPNLossComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        generate_rpn_labels
    )
    return loss_evaluator