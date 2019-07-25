"""
change the predictor of mask and box head
to the needed size(num_classes)
"""
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer, change_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.layers import Conv2d

import cv2
from torch import nn

# config_file=config_file='/root/lzr/maskrcnn-benchmark-versa/configs/panet/e2e_panet_mdconv_X_101_32x8d_FPN_1x_sg.yaml'
# cfg.merge_from_file(config_file)
# cfg.freeze()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="/root/lzr/maskrcnn-benchmark-versa/configs/panet/e2e_panet_mdconv_X_101_32x8d_FPN_1x_sg.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--num_class",
        default=30,
        help="number of classes",
        type=int,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))


    # build model using default cfg---> ckpt model
    model = build_detection_model(cfg)
    # device = torch.device(cfg.MODEL.DEVICE)
    # model.to(device)

    #load model
    # FIXME not sure with opt and scheduler
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)
    arguments = {}
    arguments["iteration"] = 0
    output_dir = cfg.OUTPUT_DIR
    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(cfg, model, optimizer, scheduler, output_dir, save_to_disk)
    
    #load 
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)
    logger.info("arguments:\n{}".format(arguments))
    # surgery
    num_classes=args.num_class
    ## box_head
    box_num_inputs=cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
    model.roi_heads.box.predictor.cls_score = nn.Linear(box_num_inputs, num_classes)
    model.roi_heads.box.predictor.bbox_pred = nn.Linear(box_num_inputs, num_classes * 4)

    nn.init.normal_(model.roi_heads.box.predictor.cls_score.weight, mean=0, std=0.01)
    nn.init.constant_(model.roi_heads.box.predictor.cls_score.bias, 0)
    nn.init.normal_(model.roi_heads.box.predictor.bbox_pred.weight, mean=0, std=0.001)
    nn.init.constant_(model.roi_heads.box.predictor.bbox_pred.bias, 0)

    ## mask_head
    mask_num_inputs=cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[0]
    # num_classes=30
    model.roi_heads.mask.predictor.mask_fcn_logits=Conv2d(mask_num_inputs, num_classes, 1, 1, 0)
    nn.init.constant_(model.roi_heads.mask.predictor.mask_fcn_logits.bias, 0)
    nn.init.kaiming_normal_(model.roi_heads.mask.predictor.mask_fcn_logits.weight, mode="fan_out", nonlinearity="relu")
    
    # save model
    data = {}
    data["model"] = model.state_dict()
    if optimizer is not None:
        data["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        data["scheduler"] = scheduler.state_dict()
    data.update(arguments)
    #surgery on opt and sche
    #predi = change_optimizer(model,data)
    # NOTE not necessary this time, instead, we give up the saved opt and scheduler 
    # and create new for finetuning predictor 
    save_file = os.path.join(output_dir, "{}.pth".format("new_model"))
    logger.info("Saving checkpoint to {}".format(save_file))
    torch.save(data, save_file)


if __name__ == "__main__":
    main()