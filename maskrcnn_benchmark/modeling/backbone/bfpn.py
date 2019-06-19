import torch
import torch.nn.functional as F
from torch import nn
from ..make_layers import conv_with_kaiming_uniform, make_conv3x3
from ..non_local import NonLocalBlock2D

class BFP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 conv_cfg=None,
                 refine_level=2,
                 refine_type=None,
                 normalize=True,
                 activation=True):
        super(BFP, self).__init__()
        #use non_local for default, and only support this
        #assert refine_type in [None, 'conv', 'non_local']
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.normalize = normalize
        self.activation = activation
        self.with_bias = normalize is None
        self.conv_cfg = conv_cfg

        #extra levels is for retinanet
        #only op on backbone:c2-c5(0,1,2,3)
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.refine_level = refine_level
        # self.refine_type = refine_type

        self.ops = []
        self.rops = []
        for i in range(self.start_level, self.backbone_end_level):
            if i < self.refine_level:
                self.ops.append(F.adaptive_max_pool2d)
                self.rops.append(F.interpolate)
            else:
                self.ops.append(F.interpolate)
                self.rops.append(F.adaptive_max_pool2d)
        self.refine = NonLocalBlock2D(
            out_channels,
            conv_with_kaiming_uniform(use_gn=normalize,use_relu=activation),
            reduction=1
            )

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            self.extra_convs = nn.ModuleList()
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_conv = make_conv3x3(
                    in_channels,
                    out_channels,
                    stride=2,
                    padding=1,
                    use_gn=normalize,
                    use_relu=activation
                )
                # original: inplace=false?
                # extra_conv = ConvModule(
                #     in_channels,
                #     out_channels,
                #     3,
                #     stride=2,
                #     padding=1,
                #     normalize=normalize,
                #     bias=self.with_bias,
                #     activation=self.activation,
                #     inplace=False)
                self.extra_convs.append(extra_conv)

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        ops_params = [
            dict(output_size=inputs[self.refine_level].size()[2:])
            if i < self.refine_level else dict(
                size=inputs[self.refine_level].size()[2:], mode='nearest')
            for i in range(self.start_level, self.backbone_end_level)
        ]
        rops_params = [
            dict(size=inputs[i].size()[2:], mode='nearest') if
            i < self.refine_level else dict(output_size=inputs[i].size()[2:])
            for i in range(self.start_level, self.backbone_end_level)
        ]

        feats = [
            self.ops[i](inputs[i + self.start_level], **ops_params[i])
            for i in range(len(self.ops))
        ]

        bsf = sum(feats) / len(feats)
        # if self.refine_type is not None:
        bsf = self.refine(bsf)

        outs = [
            self.rops[i](bsf, **rops_params[i]) +
            inputs[i + self.start_level] for i in range(len(self.rops))
        ]

        used_backbone_levels = len(outs)
        # add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            # not used in our case
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.extra_convs[0](orig))
                else:
                    outs.append(self.extra_convs[0](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    # BUG: we should add relu before each extra conv
                    outs.append(self.extra_convs[i - used_backbone_levels](
                        outs[-1]))

        return tuple(outs)
