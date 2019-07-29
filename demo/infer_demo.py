# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2

from maskrcnn_benchmark.config import cfg as cfg

from predictor import COCODemo
from nc_predictor import NCDemo
import os
import time
import argparse
import numpy as np
from collections import defaultdict
#import ipdb

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.
      Args:
          filename (string): path to a file
      Returns:
          bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)
    
def get_imagelist_from_dir(dirpath):
    images = []
    for f in os.listdir(dirpath):
        if is_image_file(f):
            images.append(os.path.join(dirpath, f))
    return images
def vis_one_image(boxes, masks, classes, class_list=None, thresh=0.9):

    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
        return None


    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)
    # sorted_inds = np.argsort(-boxes[:, -1])

    obj_mask_dict = defaultdict(list)

    for i in sorted_inds:
        score = boxes[i, -1]
        if score < thresh:
            continue
        if classes[i] in class_list:
            e = masks[:, :, i]
            obj_mask_dict[classes[i]].append([e.astype(np.float), boxes[i, :4]])

    return obj_mask_dict
def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="../configs/panet/e2e_panet_R_50_FPN_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--ori-config-file",
        default="./configs/panet/e2e_panet_mdconv_X_101_32x8d_FPN_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=224,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="image_dir",
    )
    parser.add_argument(
        "--images",
        type=str,
        default='sample_imgs',
        help="images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='vis',
        help="output_dir",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--infer-ori",
        help="Modify model config options using the command-line",
        type=bool,
        default=False,
    )

    args = parser.parse_args()
    if not args.infer_ori:

        # load config from file and command-line arguments
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        cfg.freeze()

        # prepare object that handles inference plus adds predictions on top of image
        nc_demo = NCDemo(
            cfg,
            confidence_threshold=args.confidence_threshold,
            show_mask_heatmaps=args.show_mask_heatmaps,
            masks_per_dim=args.masks_per_dim,
            min_image_size=args.min_image_size,
        )
        uni_demo=nc_demo
    else:
        cfg.merge_from_file(args.ori_config_file)
        cfg.freeze()
        coco_demo = COCODemo(
            cfg,
            confidence_threshold=args.confidence_threshold,
            show_mask_heatmaps=args.show_mask_heatmaps,
            masks_per_dim=args.masks_per_dim,
            min_image_size=args.min_image_size,
        )
        uni_demo=coco_demo

    if args.image_dir:
        imglist = get_imagelist_from_dir(args.image_dir)
    else:
        imglist = [args.images]
    num_images = len(imglist)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for i in range(num_images):
        print('img', i)
        img = cv2.imread(imglist[i])
        assert img is not None
        start_time = time.time()
        #cls_boxes, cls_segms, cls_keyps = im_detect_all(maskRCNN, im, timers=timers)
        composite = uni_demo.run_on_opencv_image(img)
        print("Time: {:.2f} s / img".format(time.time() - start_time))
        if not args.infer_ori:
            cv2.imwrite(os.path.join(args.output_dir,'res_{}.png'.format(i)), composite)
            box,masks,classes = nc_demo.detect_all_image(img)
            class_list = [i for i in range(1,27)]
            obj_mask_dict = vis_one_image(box,masks,classes,class_list)
        else:
            cv2.imwrite(os.path.join(args.output_dir,'ori_res_{}.png'.format(i)), composite)

        #ipdb.set_trace()
        
if __name__ == "__main__":
    main()
