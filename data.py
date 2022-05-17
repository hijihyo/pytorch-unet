import os
from typing import Callable, Optional
import torch
import torchvision
from PIL import Image


class SegmentationDataset(torch.utils.data.Dataset):
    """PyTorch template dataset for segmentation"""

    DATASET_NAME = None
    IMG_DIRECTORY = None
    MASK_DIRECTORY = None

    def __init__(self, root: str = ".data", transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        super(SegmentationDataset, self).__init__()
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


class ISBI2012(SegmentationDataset):
    """PyTorch dataset for ISBI 2012 EM segmentation challenge"""

    DATASET_NAME = "ISBI2012"
    IMG_DIRECTORY = "images"
    MASK_DIRECTORY = "labels"


class Luminous(SegmentationDataset):
    """PyTorch dataset for Luminous ultrasound image database"""

    DATASET_NAME = "Luminous"
    IMG_DIRECTORY = "B-mode"
    MASK_DIRECTORY = "Masks"
