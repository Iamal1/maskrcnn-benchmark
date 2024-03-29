import torch
import torch.nn as nn
import torch.nn.functional as F

# from .make_layers import conv_with_kaiming_uniform as conv_block


class NonLocalBlock2D(nn.Module):

    def __init__(self,
                 in_channels,
                 conv_block,
                 reduction=2,
                 mode='embedded_gaussian'):
        super(NonLocalBlock2D, self).__init__()
        self.conv_block = conv_block
        self.in_channels = in_channels
        self.reduction = reduction
        self.inter_channels = in_channels // reduction
        self.mode = mode
        assert mode in ['embedded_gaussian']

        self.g = conv_block(
            self.in_channels,
            self.inter_channels,
            kernel_size=1)
        self.theta = conv_block(
            self.in_channels,
            self.inter_channels,
            kernel_size=1)
        self.phi = conv_block(
            self.in_channels,
            self.inter_channels,
            kernel_size=1)
        self.conv_mask = conv_block(
            self.inter_channels,
            self.in_channels,
            kernel_size=1)

    # DIFF: remove constant_init for conv_mask
    #     self.init_weights()

    # def init_weights(self):
    #     constant_init(self.conv_mask.conv, 0)

    def embedded_gaussian(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        map_t_p = torch.matmul(theta_x, phi_x)
        mask_t_p = F.softmax(map_t_p, dim=-1)

        map_ = torch.matmul(mask_t_p, g_x)
        map_ = map_.permute(0, 2, 1).contiguous()
        map_ = map_.view(batch_size, self.inter_channels, x.size(2), x.size(3))
        mask = self.conv_mask(map_)
        final = mask + x
        return final

    def forward(self, x):
        if self.mode == 'embedded_gaussian':
            output = self.embedded_gaussian(x)
        else:
            raise NotImplementedError("The code has not been implemented.")
        return output
