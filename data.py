"""
PyTorch dataset for LUMINNOUS database
"""
import os
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image


class Luminous(torch.utils.data.Dataset):
    """PyTorch dataset for LUMINNOUS database"""

    DATASET_NAME = "Luminous"
    IMG_DIRECTORY = "B-mode"
    MASK_DIRECTORY = "Masks"

    def __init__(
        self,
        root: str = ".data",
        combine_mask: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super(Luminous, self).__init__()
        self.root = root
        self.combine_mask = combine_mask
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms

        self.img_dir_path = os.path.join(
            root, self.DATASET_NAME, self.IMG_DIRECTORY
        )
        self.mask_dir_path = os.path.join(
            root, self.DATASET_NAME, self.MASK_DIRECTORY
        )
        self.masks, self.imgs = None, None

        if combine_mask:
            self.masks, self.imgs = [], []
            masks = list(sorted(os.listdir(self.mask_dir_path)))
            for i, _ in enumerate(masks):
                if i + 1 < len(masks) and masks[i][:masks[i].rfind('_')] == masks[i + 1][:masks[i + 1].rfind('_')]:
                    self.masks.append((masks[i], masks[i + 1]))
                elif masks[i][-5] == '2':
                    continue
                else:
                    self.masks.append(masks[i])
                self.imgs.append(self._mask_to_img(masks[i]))
        else:
            self.masks = list(sorted(os.listdir(self.mask_dir_path)))
            self.imgs = [
                self._mask_to_img(mask) for mask in self.masks
            ]
    
    def _mask_to_img(self, mask: str):
        return mask[:mask.rfind('_') + 1] + 'Bmode.tif'

    def __getitem__(self, index: int):
        img_path = os.path.join(self.img_dir_path, self.imgs[index])
        img = Image.open(img_path)

        if isinstance(self.masks[index], str):
            mask_path = os.path.join(self.mask_dir_path, self.masks[index])
            mask = Image.open(mask_path)
        else:
            mask1_path = os.path.join(self.mask_dir_path, self.masks[index][0])
            mask2_path = os.path.join(self.mask_dir_path, self.masks[index][1])
            mask1_np = np.array(Image.open(mask1_path))
            mask2_np = np.array(Image.open(mask2_path))
            mask = Image.fromarray(mask1_np + mask2_np, mode="L")

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.imgs)
        