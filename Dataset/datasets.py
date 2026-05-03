from pathlib import Path
from typing import Dict

import random
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset


class RAFIppCelebA(Dataset):
    def __init__(self, root, split='train', image_size=256, augment=False):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.augment = augment and split == 'train'

        split_file = self.root / 'splits' / f'{split}.txt'
        if not split_file.exists():
            raise FileNotFoundError(f'Split file not found: {split_file}')
        self.names = [line.strip() for line in split_file.read_text(encoding='utf-8').splitlines() if line.strip()]
        if not self.names:
            raise RuntimeError(f'No file names found in split: {split_file}')

    def __len__(self):
        return len(self.names)

    def _load_rgb(self, folder: str, name: str) -> Image.Image:
        path = self.root / folder / name
        return Image.open(path).convert('RGB').resize((self.image_size, self.image_size), Image.BICUBIC)

    def _load_mask(self, folder: str, name: str) -> Image.Image:
        path = self.root / folder / name
        return Image.open(path).convert('L').resize((self.image_size, self.image_size), Image.NEAREST)

    @staticmethod
    def _rgb_to_tensor(img: Image.Image) -> torch.Tensor:
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor * 2.0 - 1.0

    @staticmethod
    def _gray_to_tensor(img: Image.Image) -> torch.Tensor:
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    def _augment(self, gt, masked, mask, boundary):
        if random.random() < 0.5:
            gt = ImageOps.mirror(gt)
            masked = ImageOps.mirror(masked)
            mask = ImageOps.mirror(mask)
            boundary = ImageOps.mirror(boundary)
        return gt, masked, mask, boundary

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        name = self.names[idx]
        gt = self._load_rgb('gt', name)
        masked = self._load_rgb('masked', name)
        mask = self._load_mask('masks', name)
        boundary = self._load_mask('boundaries', name)

        if self.augment:
            gt, masked, mask, boundary = self._augment(gt, masked, mask, boundary)

        return {
            'name': name,
            'gt': self._rgb_to_tensor(gt),
            'masked': self._rgb_to_tensor(masked),
            'mask': self._gray_to_tensor(mask),
            'boundary': self._gray_to_tensor(boundary),
        }
