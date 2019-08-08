# Written by Roy Tseng
#
# Based on:
# --------------------------------------------------------
# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import os
import pycocotools.mask as mask_util

from maskrcnn_benchmark.utils.colormap import colormap

# Use a non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import ipdb

plt.rcParams['pdf.fonttype'] = 42  # For editing in Adobe Illustrator


_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)



def get_class_string(class_index, score, coco_demo):
    class_text = coco_demo.CATEGORIES[class_index] if coco_demo is not None else \
        'id{:d}'.format(class_index)
    return class_text +'({})'.format(class_index) + ' {:0.2f}'.format(score).lstrip('0')


def vis_one_image(
        im, im_name, output_dir, coco_demo, prefix='', thresh=0.9,
        dpi=200, box_alpha=0.0, show_class=False,
        ext='png'):
    """Visual debugging of detections."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # FIX A bug, input should be BGR format, but RGB for visualization
    boxes,masks,classes = coco_demo.detect_all_image(im[:,:,::-1])
    color_list = colormap(rgb=True) / 255

    fig = plt.figure(frameon=False)
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(im)

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    mask_color_id = 0
    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        if score < thresh:
            continue

        print(coco_demo.CATEGORIES[classes[i]], score)
        # show box (off by default, box_alpha=0.0)
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1],
                          fill=False, edgecolor='g',
                          linewidth=0.5, alpha=box_alpha))

        if show_class:
            ax.text(
                bbox[0], bbox[1] - 2,
                get_class_string(classes[i], score, coco_demo),
                fontsize=3,
                family='serif',
                bbox=dict(
                    facecolor='g', alpha=0.4, pad=0, edgecolor='none'),
                color='white')

        # show mask
        #HACK assume all not None
        if masks is not None:
            img = np.ones(im.shape)
            color_mask = color_list[mask_color_id % len(color_list), 0:3]
            mask_color_id += 1

            w_ratio = .4
            for c in range(3):
                color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
            for c in range(3):
                img[:, :, c] = color_mask[c]
            e = masks[:, :, i]
            
            # cv2.RETR_CCOMP: only find out-boundary and hole-boundary
            _, contour, hier = cv2.findContours(
                e.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

            for c in contour:
                polygon = Polygon(
                    c.reshape((-1, 2)),
                    fill=True, facecolor=color_mask,
                    edgecolor='w', linewidth=1.2,
                    alpha=0.5)
                ax.add_patch(polygon)

    output_name = os.path.splitext(os.path.basename(im_name))[0] + '.' + ext
    fig.savefig(os.path.join(output_dir, '{}_{}'.format(prefix, output_name)), dpi=dpi)
    plt.close('all')

def vis_one_image_2(
        im, im_name, output_dir, boxes, segms=None, keypoints=None, thresh=0.9,
        kp_thresh=2, dpi=200, box_alpha=0.0, dataset=None, show_class=False,
        ext='pdf'):
    """Visual debugging of detections. group by class"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if isinstance(boxes, list):
        boxes, segms, keypoints, classes = convert_from_cls_format(
            boxes, segms, keypoints)

    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
        return

    if segms is not None:
        masks = mask_util.decode(segms)

    color_list = colormap(rgb=True) / 255



    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    mask_color_id = 0
    bg=np.zeros(im.shape[:2],dtype=np.uint8)
    filter_list=['cat','dog','person']
    #filter_list=['cat']
    appear_list = []
    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        if score < thresh:
            continue

        print(dataset.classes[classes[i]], score)
        # show box (off by default, box_alpha=0.0)
        class_name = dataset.classes[classes[i]]
        appear_list.append(class_name)
        #skip unnece classes
        if not class_name in filter_list:
            continue
        if not os.path.exists(os.path.join(output_dir, class_name)):
            os.makedirs(os.path.join(output_dir, class_name))
        # show mask
        if segms is not None and len(segms) > i:
            img = np.ones(im.shape)
            e = masks[:, :, i]*255
            #ipdb.set_trace()
            #print(e.dtype)
            class_mask = e
            #class_mask = cv2.bitwise_and(im,im,mask=e.astype(np.uint8))
        else:
            class_mask = bg    
        #ipdb.set_trace()
        b_channel, g_channel, r_channel = cv2.split(im)
        alpha_channel = class_mask
        img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
        output_name = os.path.split(os.path.basename(im_name))[0] + '.' + ext
        save_path = os.path.join(output_dir, '{}/{}'.format(class_name,output_name))
        #RGB to BGR and save
        cv2.imwrite(save_path,img_BGRA)
    #last check
    for cls in filter_list:
        if cls not in appear_list:
            output_name = os.path.basename(im_name) + '.' + ext
            save_path = os.path.join(output_dir, '{}/{}'.format(cls,output_name))
            cv2.imwrite(save_path,bg)
