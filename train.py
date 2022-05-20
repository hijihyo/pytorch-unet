"""
Train U-Net with Luminous database
"""
import os
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, random_split

from data import Luminous
from model import UNet
from utils import evaluate, iterate_train


parser = argparse.ArgumentParser()

parser.add_argument("--batch-size", default=4, type=int,
                    help="The number of examples in one mini-batch")
parser.add_argument("--device", default="cuda", type=str,
                    help="Device to use; cpu or cuda")
parser.add_argument("--dropout", default=0.5, type=float,
                    help="The probability to dropout")
parser.add_argument("--learning-rate", default=1e-3,
                    type=float, help="Learning rate for Adam optimizer")
parser.add_argument("--num-epochs", default=20, type=int,
                    help="The number of epochs to train the model")
parser.add_argument("--early-stop", default=False, type=bool,
                    help="Whether to use early stopping")
parser.add_argument("--data-dir", default=".data", type=str,
                    help="The path to a directory where the data are stored")
parser.add_argument("--model-dir", default=".model", type=str,
                    help="The path to a directory to store model parameters")
parser.add_argument("--encoder-dir", default=".encoder", type=str,
                    help="The path to a directory to store encoder parameters")


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
    mask_tensor = torch.cat(masks)  # (B, 624, 832)
    return img_tensor, mask_tensor


def collate_batch(batch):
    """Collates a batch"""
    imgs, masks = [], []
    for img, mask in batch:
        imgs.append(img.unsqueeze(0))
        masks.append(mask)
    img_tensor = torch.cat(imgs)  # (B, 1, 624, 832)
    mask_tensor = torch.cat(masks)  # (B, 624, 832)
    return img_tensor, mask_tensor


def kaiming_normal_initialize(module):
    """Initialize module parameters with kaiming initialization"""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        module.bias.data.zero_()


if __name__ == "__main__":
    args = parser.parse_args()
    # TODO: implement early stopping
    assert args.early_stop is False, "Cannot use early stopping"

    assert os.path.exists(args.data_dir), "The data directory doesn't exist"
    assert os.path.exists(args.model_dir), "The model directory doesn't exist"
    assert os.path.exists(
        args.encoder_dir), "The encoder directory doesn't exist"

    device = args.device
    if device == "cuda":
        assert torch.cuda.is_available(), "Cannot use cuda"
    print('Current device:', device)

    # In order to script the transformations, please use torch.nn.Sequential instead of Compose.
    transform = T.Compose([
        T.ToTensor(),
        T.Pad((6, 5)),  # (614, 820) -> (624, 832)
    ])
    dataset = Luminous(
        root=args.data_dir,
        transform=transform,
        target_transform=transform
    )

    num_train = int(len(dataset) * 0.7)
    num_val = int(len(dataset) * 0.15)
    num_test = len(dataset) - num_train - num_val
    train_dataset, val_dataset, test_dataset = \
        random_split(dataset, (num_train, num_val, num_test))
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_train_batch)
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, collate_fn=collate_batch)
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, collate_fn=collate_batch)

    model = UNet(in_channels=1, num_classes=2, dropout=args.dropout)
    model = model.to(device)
    model.apply(kaiming_normal_initialize)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    train_history, val_history = \
        iterate_train(
            model, train_dataloader, val_dataloader, optimizer,
            loss_fn, device, num_epochs=args.num_epochs,
            model_dir=args.model_dir, encoder_dir=args.encoder_dir
        )

    print()
    test_loss_history, test_pa_history, test_iou_history = \
        evaluate(model, test_dataloader, loss_fn, device, desc="test")
    avg_test_loss = sum(test_loss_history) / len(test_dataloader)
    avg_test_pa = sum(test_pa_history) / len(test_dataloader)
    avg_test_iou = sum(test_iou_history) / len(test_dataloader)
    print(f"avg. test loss: {avg_test_loss:10.6f}")
    print(f"avg. test pixel acc.: {avg_test_pa:7.5f}")
    print(f"avg. test IoU.: {avg_test_iou:7.5f}")
