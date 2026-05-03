from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import VGG19FeatureExtractor, IdentityEncoder


@dataclass
class LossConfig:
    lambda_m: float = 1.0
    lambda_b: float = 1.0
    lambda_c: float = 0.2
    lambda_dice: float = 1.0
    lambda_bdice: float = 1.0
    lambda_rec: float = 10.0
    lambda_ssim: float = 5.0
    lambda_perc: float = 1.0
    lambda_style: float = 100.0
    lambda_id: float = 2.0
    lambda_edge: float = 2.0
    lambda_adv: float = 0.1
    alpha_region: float = 3.0
    gamma_region: float = 2.0
    beta_conf: float = 5.0


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size

    def _window(self, channels: int, device):
        coords = torch.arange(self.window_size, dtype=torch.float32, device=device) - self.window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
        g = g / g.sum()
        window_1d = g.unsqueeze(1)
        window_2d = window_1d @ window_1d.t()
        return window_2d.expand(channels, 1, self.window_size, self.window_size).contiguous()

    def forward(self, x, y):
        x = (x.clamp(-1, 1) + 1.0) / 2.0
        y = (y.clamp(-1, 1) + 1.0) / 2.0
        channels = x.size(1)
        window = self._window(channels, x.device)
        padding = self.window_size // 2

        mu_x = F.conv2d(x, window, padding=padding, groups=channels)
        mu_y = F.conv2d(y, window, padding=padding, groups=channels)
        sigma_x = F.conv2d(x * x, window, padding=padding, groups=channels) - mu_x ** 2
        sigma_y = F.conv2d(y * y, window, padding=padding, groups=channels) - mu_y ** 2
        sigma_xy = F.conv2d(x * y, window, padding=padding, groups=channels) - mu_x * mu_y

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2) + 1e-8)
        return 1.0 - ssim_map.mean()


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    inter = torch.sum(pred * target, dim=(1, 2, 3))
    denom = torch.sum(pred, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3))
    return (1.0 - ((2.0 * inter + eps) / (denom + eps))).mean()


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    b, c, h, w = feat.shape
    feat = feat.view(b, c, h * w)
    gram = torch.bmm(feat, feat.transpose(1, 2)) / (c * h * w)
    return gram


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    x = (x.clamp(-1, 1) + 1.0) / 2.0
    if x.size(1) == 3:
        x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    gx = F.conv2d(x, kernel_x, padding=1)
    gy = F.conv2d(x, kernel_y, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


class LossFactory(nn.Module):
    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg
        self.bce = nn.BCELoss()
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.vgg = VGG19FeatureExtractor().eval()
        self.id_encoder = IdentityEncoder().eval()
        for module in [self.vgg, self.id_encoder]:
            for p in module.parameters():
                p.requires_grad = False

    def segmentation_losses(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mhat = outputs['mask_pred']
        bhat = outputs['boundary_pred']
        chat = outputs['confidence_pred']
        mask = batch['mask']
        boundary = batch['boundary']

        lmask = self.bce(mhat, mask) + self.cfg.lambda_dice * dice_loss(mhat, mask)
        lbdry = self.bce(bhat, boundary) + self.cfg.lambda_bdice * dice_loss(bhat, boundary)
        conf_target = torch.exp(-self.cfg.beta_conf * torch.abs(mhat.detach() - mask))
        lconf = self.l1(chat, conf_target)
        lseg = self.cfg.lambda_m * lmask + self.cfg.lambda_b * lbdry + self.cfg.lambda_c * lconf
        return {
            'loss_mask': lmask,
            'loss_boundary': lbdry,
            'loss_conf': lconf,
            'loss_seg': lseg,
        }

    def reconstruction_losses(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        isyn = outputs['isyn']
        gt = batch['gt']
        mask = batch['mask']
        boundary = batch['boundary']

        weight = 1.0 + self.cfg.alpha_region * mask + self.cfg.gamma_region * boundary
        lrec = torch.mean(torch.abs(weight * (isyn - gt)))
        lssim = self.ssim(mask * isyn, mask * gt)

        isyn_01 = (isyn.clamp(-1, 1) + 1.0) / 2.0
        gt_01 = (gt.clamp(-1, 1) + 1.0) / 2.0
        feat_syn = self.vgg(isyn_01)
        feat_gt = self.vgg(gt_01)
        lperc = (
            self.l1(feat_syn['relu1_2'], feat_gt['relu1_2']) +
            self.l1(feat_syn['relu2_2'], feat_gt['relu2_2']) +
            self.l1(feat_syn['relu3_4'], feat_gt['relu3_4']) +
            self.l1(feat_syn['relu4_3'], feat_gt['relu4_3'])
        )
        lstyle = (
            self.l1(gram_matrix(feat_syn['relu2_2']), gram_matrix(feat_gt['relu2_2'])) +
            self.l1(gram_matrix(feat_syn['relu3_4']), gram_matrix(feat_gt['relu3_4'])) +
            self.l1(gram_matrix(feat_syn['relu4_3']), gram_matrix(feat_gt['relu4_3'])) +
            self.l1(gram_matrix(feat_syn['relu5_2']), gram_matrix(feat_gt['relu5_2']))
        )
        id_syn = self.id_encoder(isyn)
        id_gt = self.id_encoder(gt)
        lid = 1.0 - torch.sum(id_syn * id_gt, dim=1).mean()
        ledge = self.l1(mask * sobel_edges(isyn), mask * sobel_edges(gt))
        return {
            'loss_rec': lrec,
            'loss_ssim': lssim,
            'loss_perc': lperc,
            'loss_style': lstyle,
            'loss_id': lid,
            'loss_edge': ledge,
            'vgg_fake_relu4_3': feat_syn['relu4_3'],
            'vgg_real_relu4_3': feat_gt['relu4_3'],
        }

    @staticmethod
    def discriminator_hinge(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()

    @staticmethod
    def generator_hinge(fake_logits: torch.Tensor) -> torch.Tensor:
        return -fake_logits.mean()

    def generator_total(self, seg_losses: Dict[str, torch.Tensor], rec_losses: Dict[str, torch.Tensor], adv_loss: torch.Tensor, include_seg: bool) -> Dict[str, torch.Tensor]:
        total = (
            self.cfg.lambda_rec * rec_losses['loss_rec'] +
            self.cfg.lambda_ssim * rec_losses['loss_ssim'] +
            self.cfg.lambda_perc * rec_losses['loss_perc'] +
            self.cfg.lambda_style * rec_losses['loss_style'] +
            self.cfg.lambda_id * rec_losses['loss_id'] +
            self.cfg.lambda_edge * rec_losses['loss_edge'] +
            self.cfg.lambda_adv * adv_loss
        )
        if include_seg:
            total = total + seg_losses['loss_seg']
        out = {**seg_losses, **{k: v for k, v in rec_losses.items() if not k.startswith('vgg_')}}
        out['loss_adv'] = adv_loss
        out['loss_total'] = total
        return out
