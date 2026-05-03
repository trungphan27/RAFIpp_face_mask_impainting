from typing import Dict

import torch
import torch.nn.functional as F


def denorm(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1.0) / 2.0


def l1_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def psnr_metric(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred = denorm(pred)
    target = denorm(target)
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    return (10.0 * torch.log10(1.0 / (mse + eps))).mean()


def _gaussian_window(window_size: int = 11, sigma: float = 1.5, channels: int = 3, device=None):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_1d = g.unsqueeze(1)
    window_2d = window_1d @ window_1d.t()
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


def ssim_metric(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    pred = denorm(pred)
    target = denorm(target)
    channels = pred.size(1)
    window = _gaussian_window(window_size, channels=channels, device=pred.device)
    padding = window_size // 2

    mu_x = F.conv2d(pred, window, padding=padding, groups=channels)
    mu_y = F.conv2d(target, window, padding=padding, groups=channels)
    sigma_x = F.conv2d(pred * pred, window, padding=padding, groups=channels) - mu_x ** 2
    sigma_y = F.conv2d(target * target, window, padding=padding, groups=channels) - mu_y ** 2
    sigma_xy = F.conv2d(pred * target, window, padding=padding, groups=channels) - mu_x * mu_y

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2) + 1e-8)
    return ssim_map.mean()


def dice_score(pred_mask: torch.Tensor, target_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = (pred_mask > 0.5).float()
    target = (target_mask > 0.5).float()
    inter = torch.sum(pred * target, dim=(1, 2, 3))
    denom = torch.sum(pred, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3))
    return ((2 * inter + eps) / (denom + eps)).mean()


def iou_score(pred_mask: torch.Tensor, target_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = (pred_mask > 0.5).float()
    target = (target_mask > 0.5).float()
    inter = torch.sum(pred * target, dim=(1, 2, 3))
    union = torch.sum((pred + target) > 0, dim=(1, 2, 3)).float()
    return ((inter + eps) / (union + eps)).mean()


def summarize_restoration(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    return {
        'l1': float(l1_metric(pred, target).item()),
        'psnr': float(psnr_metric(pred, target).item()),
        'ssim': float(ssim_metric(pred, target).item()),
    }
