"""
PyTorch dataset for LUMINNOUS database
"""
import os
from typing import Callable, Optional
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
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms

        self.img_dir_path = os.path.join(
            root, self.DATASET_NAME, self.IMG_DIRECTORY
        )
        self.mask_dir_path = os.path.join(
            root, self.DATASET_NAME, self.MASK_DIRECTORY
        )

        if combine_mask:
            # TODO: combine masks
            raise NotImplementedError()
        else:
            self.masks = list(sorted(os.listdir(self.mask_dir_path)))
            self.imgs = [
                mask[:mask.rfind('_') + 1] + 'Bmode.tif'
                for mask in self.masks
            ]

    def __getitem__(self, index: int):
        img_path = os.path.join(self.img_dir_path, self.imgs[index])
        mask_path = os.path.join(self.mask_dir_path, self.masks[index])

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.imgs)
        