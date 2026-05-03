from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().clamp(-1, 1)
    if tensor.dim() == 3 and tensor.size(0) in (1, 3):
        arr = tensor
    elif tensor.dim() == 2:
        arr = tensor.unsqueeze(0)
    else:
        raise ValueError(f'Unsupported tensor shape: {tuple(tensor.shape)}')

    if arr.size(0) == 1:
        arr = ((arr + 1.0) / 2.0 * 255.0).numpy().astype(np.uint8)[0]
        return arr
    arr = ((arr + 1.0) / 2.0 * 255.0).numpy().astype(np.uint8).transpose(1, 2, 0)
    return arr


def mask_to_uint8_image(mask_tensor: torch.Tensor) -> np.ndarray:
    mask = mask_tensor.detach().cpu().clamp(0, 1)
    if mask.dim() == 3 and mask.size(0) == 1:
        return (mask.numpy()[0] * 255.0).astype(np.uint8)
    raise ValueError(f'Unsupported mask tensor shape: {tuple(mask.shape)}')


def save_tensor_image(tensor: torch.Tensor, path: str, is_mask: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_mask:
        img = Image.fromarray(mask_to_uint8_image(tensor), mode='L')
    else:
        arr = tensor_to_uint8_image(tensor)
        img = Image.fromarray(arr) if arr.ndim == 2 else Image.fromarray(arr, mode='RGB')
    img.save(path)


def _make_grid(images: List[np.ndarray], nrow: int) -> np.ndarray:
    if not images:
        raise ValueError('No images to assemble.')
    h, w = images[0].shape[:2]
    c = 1 if images[0].ndim == 2 else images[0].shape[2]
    rows = int(np.ceil(len(images) / nrow))
    grid = np.zeros((rows * h, nrow * w, c), dtype=np.uint8)
    for idx, img in enumerate(images):
        r = idx // nrow
        col = idx % nrow
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
        grid[r * h:(r + 1) * h, col * w:(col + 1) * w] = img
    return grid


def save_training_grid(batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor], path: str, nrow: int = 4) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = []
    for sample in batch['masked']:
        images.append(tensor_to_uint8_image(sample))
    for sample in batch['mask']:
        images.append(np.repeat(mask_to_uint8_image(sample)[:, :, None], 3, axis=2))
    for sample in outputs['mask_pred']:
        images.append(np.repeat(mask_to_uint8_image(sample)[:, :, None], 3, axis=2))
    for sample in outputs['isyn']:
        images.append(tensor_to_uint8_image(sample))
    for sample in batch['gt']:
        images.append(tensor_to_uint8_image(sample))
    grid = _make_grid(images, nrow=nrow)
    Image.fromarray(grid, mode='RGB').save(path)
