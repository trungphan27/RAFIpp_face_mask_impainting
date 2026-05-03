from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvINAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
        ]
        if activation:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=True):
        super().__init__()
        self.feature_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.gate_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.activation = nn.LeakyReLU(0.2, inplace=True) if activation else nn.Identity()

    def forward(self, x):
        feat = self.feature_conv(x)
        gate = torch.sigmoid(self.gate_conv(x))
        out = self.activation(feat) * gate
        out = self.norm(out)
        return out


class UpConvINAct(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvINAct(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, size=None):
        if size is None:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        else:
            x = F.interpolate(x, size=size, mode='nearest')
        return self.conv(x)


class UpGatedConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GatedConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, size=None):
        if size is None:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        else:
            x = F.interpolate(x, size=size, mode='nearest')
        return self.conv(x)


class DilatedBranch(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MCSAMBlock(nn.Module):
    def __init__(self, channels=512, reduction=4):
        super().__init__()
        branch_channels = channels // 4
        self.branch1 = DilatedBranch(channels, branch_channels, dilation=1)
        self.branch2 = DilatedBranch(channels, branch_channels, dilation=2)
        self.branch4 = DilatedBranch(channels, branch_channels, dilation=4)
        self.branch8 = DilatedBranch(channels, branch_channels, dilation=8)

        hidden = channels // reduction
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True)
        self.fuse = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        fd = torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch4(x),
            self.branch8(x),
        ], dim=1)

        z = F.adaptive_avg_pool2d(fd, output_size=1)
        wc = torch.sigmoid(self.channel_mlp(z))
        fc = fd * wc

        aavg = torch.mean(fc, dim=1, keepdim=True)
        amax, _ = torch.max(fc, dim=1, keepdim=True)
        ws = torch.sigmoid(self.spatial(torch.cat([aavg, amax], dim=1)))
        fs = fc * ws
        return x + self.fuse(fs)


class SkipAttentionGate(nn.Module):
    def __init__(self, enc_channels, dec_channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(enc_channels + dec_channels + 3, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, enc_feat, dec_feat, mask_map, boundary_map, conf_map):
        target_size = enc_feat.shape[-2:]
        dec_feat = F.interpolate(dec_feat, size=target_size, mode='nearest')
        mask_map = F.interpolate(mask_map, size=target_size, mode='bilinear', align_corners=False)
        boundary_map = F.interpolate(boundary_map, size=target_size, mode='bilinear', align_corners=False)
        conf_map = F.interpolate(conf_map, size=target_size, mode='bilinear', align_corners=False)
        alpha = self.gate(torch.cat([enc_feat, dec_feat, mask_map, boundary_map, conf_map], dim=1))
        gated = enc_feat * alpha
        return gated, alpha


class ResidualRefineBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvINAct(channels, channels, kernel_size=3, stride=1, padding=1),
            ConvINAct(channels, channels, kernel_size=3, stride=1, padding=1, activation=False),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))
