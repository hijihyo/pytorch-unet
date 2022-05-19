"""
Train a U-Net model on Luminous database
"""
import random
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import nn, optim
from torch.utils.data import DataLoader

from data import Luminous
from model import UNet
from utils import evaluate, iterate_train


def custom_transforms(img, mask):
    if random.random() > 0.5:
        angle = random.randint(-30, 30)
        translate = (0.4 * random.random(), 0.4 * random.random())
        scale = 0.5 * random.random() + 0.5
        shear = (random.randint(-30, 30), random.randint(-30, 30))
        img = TF.affine(img, angle, translate, scale, shear)
        mask = TF.affine(mask, angle, translate, scale, shear)
    return img, mask


root = "/home/DATA/ksh/data"
# train_dataset = Luminous(root=root, split="train", transforms=custom_transforms)
train_dataset = Luminous(root=root, split="train")
val_dataset = Luminous(root=root, split="val")
test_dataset = Luminous(root=root, split="test")


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


BATCH_SIZE = 2 
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_batch)
val_dataloader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_batch)
test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_batch)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Current device:', DEVICE)


def init_xavier_uniform(module):
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.ConvTranspose2d):
        torch.nn.init.xavier_normal_(module.weight)
        module.bias.data.zero_()


CHANNELS = [1, 64, 128, 256, 512, 1024]
NUM_CLASSES = 2  # 0 and 1
DROPOUT = 0.5  # used in the original paper
model = UNet(CHANNELS, NUM_CLASSES, DROPOUT).to(DEVICE)
model.apply(init_xavier_uniform)
optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()
print(model)


train_history, val_history = iterate_train(
    model, train_dataloader, val_dataloader, optimizer, loss_fn, DEVICE, num_epochs=3)


print()
test_loss_history, test_pa_history, test_iou_history = \
    evaluate(model, test_dataloader, loss_fn, DEVICE, desc="test")
avg_test_loss = sum(test_loss_history) / len(test_dataloader)
avg_test_pa = sum(test_pa_history) / len(test_dataloader)
avg_test_iou = sum(test_iou_history) / len(test_dataloader)
print(f"avg. test loss: {avg_test_loss:10.6f}")
print(f"avg. test pixel acc.: {avg_test_pa:7.5f}")
print(f"avg. test IoU.: {avg_test_iou:7.5f}")
