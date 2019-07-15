# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#add iou_feature to output for maskiou
from torch import nn
from torch.nn import functional as F

from ..box_head.roi_box_feature_extractors import ResNet50Conv5ROIFeatureExtractor
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.make_layers import make_conv3x3
import torch.nn.init as init
import torch

registry.ROI_MASK_FEATURE_EXTRACTORS.register(
    "ResNet50Conv5ROIFeatureExtractor", ResNet50Conv5ROIFeatureExtractor
)


@registry.ROI_MASK_FEATURE_EXTRACTORS.register("MaskRCNNFPNFeatureExtractor")
class MaskRCNNFPNFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(MaskRCNNFPNFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels
        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_MASK_HEAD.USE_GN
        layers = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS
        dilation = cfg.MODEL.ROI_MASK_HEAD.DILATION

        next_feature = input_size
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "mask_fcn{}".format(layer_idx)
            module = make_conv3x3(
                next_feature, layer_features,
                dilation=dilation, stride=1, use_gn=use_gn
            )
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)
        self.out_channels = layer_features

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)

        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))

        return x


@registry.ROI_MASK_FEATURE_EXTRACTORS.register("MaskRCNNFPN_adp_FeatureExtractor")
class MaskRCNNFPN_adp_FeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    adp adds another conv blocks as in fast_rcnn box heads.
    should only support adpative feature pooling
    not implemented...
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(MaskRCNNFPN_adp_FeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels
        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_MASK_HEAD.USE_GN
        layers = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS
        dilation = cfg.MODEL.ROI_MASK_HEAD.DILATION

        next_feature = input_size
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "mask_fcn{}".format(layer_idx)
            module = make_conv3x3(
                next_feature, layer_features,
                dilation=dilation, stride=1, use_gn=use_gn
            )
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)
        self.out_channels = layer_features

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)

        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))

        return x

@registry.ROI_MASK_FEATURE_EXTRACTORS.register("MaskRCNNFPN_adp_ff_FeatureExtractor")
class MaskRCNNFPN_adp_ff_FeatureExtractor(nn.Module):
    """
    Heads for MASKRCNNFPN for classification
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(MaskRCNNFPN_adp_ff_FeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
            panet = True
        )
        input_size = in_channels
        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_MASK_HEAD.USE_GN
        layers = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS
        dilation = cfg.MODEL.ROI_MASK_HEAD.DILATION

        next_feature = input_size
        layer_features = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[0]

        #first 2 conv layers are shared and remains the same, but the paper says 3 is best
        module_list = []
        for i in range(2):
            module_list.extend([
                make_conv3x3(next_feature, layer_features, dilation=dilation, stride=1, use_gn=use_gn, use_relu=True)
            ])
            next_feature = layer_features
        self.conv_fcn = nn.Sequential(*module_list)
        
        #this is for adaptive feature pooling, 
        self.mask_conv1 = nn.ModuleList()
        # num_levels = cfg.FPN.ROI_MAX_LEVEL - cfg.FPN.ROI_MIN_LEVEL + 1
        num_levels = 4
        for i in range(num_levels):
            self.mask_conv1.append(
                make_conv3x3(next_feature, layer_features, dilation=dilation, stride=1, use_gn=use_gn, use_relu=True),
            )

        self.mask_conv4 = make_conv3x3(next_feature, layer_features, dilation=dilation, stride=1, use_gn=use_gn, use_relu=True)

        self.mask_conv4_fc = make_conv3x3(next_feature, layer_features, dilation=dilation, stride=1, use_gn=use_gn, use_relu=True)

        self.mask_conv5_fc = make_conv3x3(next_feature, int(layer_features/2), dilation=dilation, stride=1, use_gn=use_gn, use_relu=True)

        self.mask_fc = nn.Sequential(
                nn.Linear(int(layer_features / 2) * (resolution) ** 2, cfg.MODEL.ROI_MASK_HEAD.RESOLUTION ** 2, bias=True),
                nn.ReLU(inplace=True))
        
        # upsample layer
        self.upconv = nn.ConvTranspose2d(layer_features, layer_features, 2, 2, 0)
        self.out_channels = layer_features
        #init_weights
        # make_conv3x3 has already done the init, default kaiming = MSRAFFill in panet.
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        #     if cfg.MRCNN.CONV_INIT == 'GaussianFill':
        #         init.normal_(m.weight, std=0.001)
        #     elif cfg.MRCNN.CONV_INIT == 'MSRAFill':
        #         mynn.init.MSRAFill(m.weight)
        #     else:
        #         raise ValueError
        #     if m.bias is not None:
        #         init.constant_(m.bias, 0)
        if isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                init.constant_(m.bias, 0)
        if isinstance(m, nn.Linear):
            init.normal_(m.weight, std=0.01)
            init.constant_(m.bias, 0)

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        #adaptive part
        for i in range(len(x)):
            x[i] = self.mask_conv1[i](x[i])
        for i in range(1, len(x)):
            x[0] = torch.max(x[0], x[i])
        x = x[0]
        roi_feature = x
        x = self.conv_fcn(x)

        batch_size = x.size(0)
        x_fcn = F.relu(self.upconv(self.mask_conv4(x)), inplace=True)
        x_ff = self.mask_fc(self.mask_conv5_fc(self.mask_conv4_fc(x)).view(batch_size, -1))

        return [x_fcn, x_ff], roi_feature




def make_roi_mask_feature_extractor(cfg, in_channels):
    func = registry.ROI_MASK_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)
