"""
Utility methods
"""
import random
from typing import Optional

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch import nn, Tensor
from torch.utils.data import DataLoader, random_split
from torch.nn.modules.loss import _WeightedLoss
from ignite.metrics import DiceCoefficient
from ignite.metrics.confusion_matrix import ConfusionMatrix

from data import Luminous


def transforms(img, mask):
    """Randomly transforms an image and a mask at the same time"""
    if random.random() < 0.5:
        angle = random.randint(-30, 30)
        translate = (0.4 * random.random(), 0.4 * random.random())
        scale = 0.6 * random.random() + 0.8
        shear = (random.randint(-15, 15), random.randint(-15, 15))
        img = TF.affine(img, angle, translate, scale, shear)
        mask = TF.affine(mask, angle, translate, scale, shear)
    if random.random() < 0.5:
        brightness = 1 * random.random() + 0.5
        img = TF.adjust_brightness(img, brightness)
    if img.size()[-2:] != (624, 832):
        img = TF.resize(img, (624, 832))
        mask = TF.resize(mask, (624, 832))
    return img, mask


def collate_train_batch(batch):
    """Collates a batch in training data"""
    imgs, masks = [], []
    for img, mask in batch:
        img, mask = transforms(img, mask)
        imgs.append(img.unsqueeze(0))
        masks.append(mask)
    img_tensor = torch.cat(imgs)  # (B, 1, 624, 832)
    mask_tensor = torch.cat(masks).to(torch.long)  # (B, 624, 832)
    return img_tensor, mask_tensor


def collate_batch(batch):
    """Collates a batch"""
    imgs, masks = [], []
    for img, mask in batch:
        imgs.append(img.unsqueeze(0))
        masks.append(mask)
    img_tensor = torch.cat(imgs)  # (B, 1, 624, 832)
    mask_tensor = torch.cat(masks).to(torch.long)  # (B, 624, 832)
    return img_tensor, mask_tensor


def create_dataloaders(data_dir, combine_mask, split_ratio, batch_size):
    transform = T.Compose([
        T.ToTensor(),
        T.Pad((6, 5)),  # (614, 820) -> (624, 832)
    ])
    dataset = Luminous(
        root=data_dir,
        combine_mask=combine_mask,
        transform=transform,
        target_transform=transform
    )
    num_train = int(len(dataset) * split_ratio[0])
    num_val = int(len(dataset) * split_ratio[1])
    num_test = len(dataset) - num_train - num_val
    train_dataset, val_dataset, test_dataset = \
        random_split(dataset, (num_train, num_val, num_test))

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_train_batch)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=collate_batch)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=collate_batch)
    return train_dataloader, val_dataloader, test_dataloader


def kaiming_normal_initialize(module):
    """Initialize module parameters with kaiming initialization"""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        module.bias.data.zero_()


class SegmentationLoss(_WeightedLoss):
    """Combination of nn.CrossEntropyLoss and dice loss"""

    def __init__(self, weight: Optional[Tensor] = None,
                 size_average=None, ignore_index: int = -100, reduce=None,
                 reduction: str = 'mean', label_smoothing: float = 0.0, num_classes: int = 2) -> None:
        super(SegmentationLoss, self).__init__(
            weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.dice_coef = \
            DiceCoefficient(ConfusionMatrix(num_classes), ignore_index) \
            if ignore_index >= 0 else \
            DiceCoefficient(ConfusionMatrix(num_classes))

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Args:
            input: torch.Tensor, (C, H, W) or (B, C, H, W)
            target: torch.Tensor, (H, W) or (B, H, W)
        """
        ce_loss = F.cross_entropy(pred, target, weight=self.weight,
                                  ignore_index=self.ignore_index, reduction=self.reduction,
                                  label_smoothing=self.label_smoothing)
        self.dice_coef.update((pred, target))
        dice_loss = 1 - self.dice_coef.compute().mean()
        self.dice_coef.reset()
        return ce_loss + dice_loss.to(ce_loss.device)
