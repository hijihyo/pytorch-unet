"""
Train a U-Net model on Luminous database
"""
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
# import torchvision

from data import Luminous
from model import UNet
from utils import iterate_train


dataset = Luminous()
num_train = int(len(dataset) * 0.8)
num_val = len(dataset) - num_train
train_dataset, val_dataset = random_split(dataset, [num_train, num_val])


def collate_batch(batch):
    """Collate each batch"""
    img_list, mask_list = [], []
    for img, mask in batch:
        img = F.pad(img, (0, 0, 91, 91))  # (1, 796, 820)
        img = img[:, :796, :-24]  # (1, 796, 796)
        mask = F.pad(mask, (0, 0, 91, 91))  # (1, 796, 820)
        mask = mask[:, :796, :-24]  # (1, 796, 796)
        mask = mask[:, 92:-92, 92:-92]  # (1, 612, 612)
        img_list.append(img.unsqueeze(0))  # (1, 1, 796, 796)
        mask_list.append(mask)  # (1, 796, 796)
    img_tensor = torch.cat(img_list)
    mask_tensor = torch.cat(mask_list)
    return img_tensor, mask_tensor


BATCH_SIZE = 4
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_batch)
val_dataloader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_batch)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Current device:', DEVICE)

CHANNELS = [1, 64, 128, 256, 512, 1024]
NUM_CLASSES = 2  # 0 and 1
DROPOUT = 0.5  # used in the original paper
MODEL = UNet(CHANNELS, NUM_CLASSES, DROPOUT).to(DEVICE)
optimizer = optim.Adam(MODEL.parameters())
loss_fn = nn.CrossEntropyLoss()
print(MODEL)

iterate_train(MODEL, train_dataloader, val_dataloader,
              optimizer, loss_fn, DEVICE)
