import os
from typing import Callable, Optional
import torch
import torchvision
from PIL import Image


class Luminous(torch.utils.data.Dataset):
    """PyTorch dataset for Luminous ultrasound image database"""

    DATASET_NAME = "Luminous"
    IMG_DIRECTORY = "B-mode"
    MASK_DIRECTORY = "Masks"

    def __init__(self, root: str = ".data", transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        super(Luminous, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.imgs = list(
            sorted(os.listdir(os.path.join(root, self.DATASET_NAME, self.IMG_DIRECTORY))))
        self.masks = list(
            sorted(os.listdir(os.path.join(root, self.DATASET_NAME, self.MASK_DIRECTORY))))

    def __getitem__(self, index):
        img_path = os.path.join(
            self.root, self.DATASET_NAME, self.IMG_DIRECTORY, self.imgs[index])
        mask_path = os.path.join(
            self.root, self.DATASET_NAME, self.MASK_DIRECTORY, self.masks[index])

        img = Image.open(img_path)
        img = torchvision.transforms.functional.to_tensor(img)
        mask = Image.open(mask_path)
        mask = torchvision.transforms.functional.to_tensor(mask)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask

    def __len__(self):
        return len(self.imgs)
