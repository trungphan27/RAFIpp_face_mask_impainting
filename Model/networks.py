from typing import Dict

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from .blocks import (
    ConvINAct,
    GatedConv2d,
    MCSAMBlock,
    ResidualRefineBlock,
    SkipAttentionGate,
    UpConvINAct,
    UpGatedConv,
)


class SegNetPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = ConvINAct(3, 64, stride=1)
        self.down1 = ConvINAct(64, 64, stride=2)
        self.e2 = ConvINAct(64, 128, stride=1)
        self.down2 = ConvINAct(128, 128, stride=2)
        self.e3 = ConvINAct(128, 256, stride=1)
        self.down3 = ConvINAct(256, 256, stride=2)
        self.e4 = ConvINAct(256, 512, stride=1)
        self.down4 = ConvINAct(512, 512, stride=2)
        self.bottleneck = ConvINAct(512, 1024, stride=1)

        self.up4 = UpConvINAct(1024, 512)
        self.dec4 = ConvINAct(1024, 512)
        self.up3 = UpConvINAct(512, 256)
        self.dec3 = ConvINAct(512, 256)
        self.up2 = UpConvINAct(256, 128)
        self.dec2 = ConvINAct(256, 128)
        self.up1 = UpConvINAct(128, 64)
        self.dec1 = ConvINAct(128, 64)

        self.mask_head = nn.Conv2d(64, 1, kernel_size=1)
        self.boundary_head = nn.Conv2d(64, 1, kernel_size=1)
        self.conf_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.e1(x)
        x = self.down1(e1)
        e2 = self.e2(x)
        x = self.down2(e2)
        e3 = self.e3(x)
        x = self.down3(e3)
        e4 = self.e4(x)
        x = self.down4(e4)
        x = self.bottleneck(x)

        x = self.up4(x, size=e4.shape[-2:])
        x = self.dec4(torch.cat([x, e4], dim=1))
        x = self.up3(x, size=e3.shape[-2:])
        x = self.dec3(torch.cat([x, e3], dim=1))
        x = self.up2(x, size=e2.shape[-2:])
        x = self.dec2(torch.cat([x, e2], dim=1))
        x = self.up1(x, size=e1.shape[-2:])
        fdec = self.dec1(torch.cat([x, e1], dim=1))

        mhat = torch.sigmoid(self.mask_head(fdec))
        bhat = torch.sigmoid(self.boundary_head(fdec))
        chat = torch.sigmoid(self.conf_head(fdec))
        return {
            'mask_pred': mhat,
            'boundary_pred': bhat,
            'confidence_pred': chat,
            'seg_feature': fdec,
        }


class RestoreNetPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = GatedConv2d(6, 64, stride=1)
        self.enc2 = GatedConv2d(64, 128, stride=2)
        self.enc3 = GatedConv2d(128, 256, stride=2)
        self.enc4 = GatedConv2d(256, 512, stride=2)

        self.mcsam1 = MCSAMBlock(512)
        self.mcsam2 = MCSAMBlock(512)
        self.mcsam3 = MCSAMBlock(512)
        self.alpha1 = nn.Parameter(torch.tensor(1.0))
        self.alpha2 = nn.Parameter(torch.tensor(1.0))
        self.alpha3 = nn.Parameter(torch.tensor(1.0))

        self.up_d4 = UpGatedConv(512, 256)
        self.skip3 = SkipAttentionGate(enc_channels=256, dec_channels=256)
        self.dec4 = GatedConv2d(512, 256)
        self.refine4 = ResidualRefineBlock(256)

        self.up_d3 = UpGatedConv(256, 128)
        self.skip2 = SkipAttentionGate(enc_channels=128, dec_channels=128)
        self.dec3 = GatedConv2d(256, 128)
        self.refine3 = ResidualRefineBlock(128)

        self.up_d2 = UpGatedConv(128, 64)
        self.skip1 = SkipAttentionGate(enc_channels=64, dec_channels=64)
        self.dec2 = GatedConv2d(128, 64)
        self.refine2 = ResidualRefineBlock(64)

        self.dec1 = GatedConv2d(64, 32, stride=1)
        self.refine1 = ResidualRefineBlock(32)
        self.to_rgb = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, image, mask_pred, boundary_pred, confidence_pred):
        x = torch.cat([image, mask_pred, boundary_pred, confidence_pred], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        z1 = self.mcsam1(e4)
        z2_prime = self.mcsam2(z1)
        z2 = z2_prime + self.alpha1 * z1
        z3_prime = self.mcsam3(z2)
        z3 = z3_prime + self.alpha2 * z2 + self.alpha3 * z1

        up4 = self.up_d4(z3, size=e3.shape[-2:])
        e3_gated, gate3 = self.skip3(e3, up4, mask_pred, boundary_pred, confidence_pred)
        d4 = self.dec4(torch.cat([up4, e3_gated], dim=1))
        d4 = self.refine4(d4)

        up3 = self.up_d3(d4, size=e2.shape[-2:])
        e2_gated, gate2 = self.skip2(e2, up3, mask_pred, boundary_pred, confidence_pred)
        d3 = self.dec3(torch.cat([up3, e2_gated], dim=1))
        d3 = self.refine3(d3)

        up2 = self.up_d2(d3, size=e1.shape[-2:])
        e1_gated, gate1 = self.skip1(e1, up2, mask_pred, boundary_pred, confidence_pred)
        d2 = self.dec2(torch.cat([up2, e1_gated], dim=1))
        d2 = self.refine2(d2)

        d1 = self.dec1(d2)
        d1 = self.refine1(d1)
        ir = torch.tanh(self.to_rgb(d1))
        isyn = (1.0 - mask_pred) * image + mask_pred * ir
        return {
            'restored': ir,
            'isyn': isyn,
            'gate1': gate1,
            'gate2': gate2,
            'gate3': gate3,
            'z3': z3,
        }


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=5):
        super().__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)),
        )

    def forward(self, x):
        return self.model(x)


class FeaturePatchDiscriminator(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)),
        )

    def forward(self, x):
        return self.model(x)


class VGG19FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_torchvision = False
        self.slices = None
        try:
            from torchvision import models
            try:
                weights = models.VGG19_Weights.IMAGENET1K_V1
                vgg = models.vgg19(weights=weights).features
            except Exception:
                try:
                    vgg = models.vgg19(weights=None).features
                except TypeError:
                    vgg = models.vgg19(pretrained=False).features
            self.slices = nn.ModuleDict({
                'relu1_2': nn.Sequential(*[vgg[i] for i in range(0, 4)]),
                'relu2_2': nn.Sequential(*[vgg[i] for i in range(4, 9)]),
                'relu3_4': nn.Sequential(*[vgg[i] for i in range(9, 18)]),
                'relu4_3': nn.Sequential(*[vgg[i] for i in range(18, 27)]),
                'relu5_2': nn.Sequential(*[vgg[i] for i in range(27, 34)]),
            })
            self.use_torchvision = True
        except Exception as exc:
            warnings.warn(
                'torchvision VGG19 could not be imported or initialized. Falling back to a small CNN feature extractor. '
                f'Underlying error: {exc}'
            )
            self.stage1 = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            )
            self.pool1 = nn.AvgPool2d(2)
            self.stage2 = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            )
            self.pool2 = nn.AvgPool2d(2)
            self.stage3 = nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            )
            self.pool3 = nn.AvgPool2d(2)
            self.stage4 = nn.Sequential(
                nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            )
            self.pool4 = nn.AvgPool2d(2)
            self.stage5 = nn.Sequential(
                nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            )
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = {}
        h = x
        if self.use_torchvision:
            h = self.slices['relu1_2'](h)
            out['relu1_2'] = h
            h = self.slices['relu2_2'](h)
            out['relu2_2'] = h
            h = self.slices['relu3_4'](h)
            out['relu3_4'] = h
            h = self.slices['relu4_3'](h)
            out['relu4_3'] = h
            h = self.slices['relu5_2'](h)
            out['relu5_2'] = h
            return out

        h = self.stage1(h)
        out['relu1_2'] = h
        h = self.pool1(h)
        h = self.stage2(h)
        out['relu2_2'] = h
        h = self.pool2(h)
        h = self.stage3(h)
        out['relu3_4'] = h
        h = self.pool3(h)
        h = self.stage4(h)
        out['relu4_3'] = h
        h = self.pool4(h)
        h = self.stage5(h)
        out['relu5_2'] = h
        return out


class IdentityEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.available = False
        self.encoder = None
        try:
            from facenet_pytorch import InceptionResnetV1
            self.encoder = InceptionResnetV1(pretrained='vggface2').eval()
            self.available = True
        except Exception as exc:
            warnings.warn(
                'facenet-pytorch is unavailable. Falling back to pooled RGB features for identity loss. '
                'Install facenet-pytorch for a stronger identity prior. '
                f'Underlying error: {exc}'
            )
            self.available = False
        if self.encoder is not None:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = (x.clamp(-1, 1) + 1.0) / 2.0
        if self.available:
            x = F.interpolate(x, size=(160, 160), mode='bilinear', align_corners=False)
            feat = self.encoder(x)
            return F.normalize(feat, dim=1)
        feat = F.adaptive_avg_pool2d(x, output_size=1).flatten(1)
        return F.normalize(feat, dim=1)


class RAFIpp(nn.Module):
    def __init__(self):
        super().__init__()
        self.segnet = SegNetPP()
        self.restore = RestoreNetPP()

    def forward(self, image):
        seg = self.segnet(image)
        restore = self.restore(image, seg['mask_pred'], seg['boundary_pred'], seg['confidence_pred'])
        return {**seg, **restore}
