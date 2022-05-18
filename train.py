"""
Train a U-Net model on Luminous database
"""
import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
# import torchvision
import matplotlib.pyplot as plt
from torchmetrics.functional import jaccard_index

from data import Luminous
from model import UNet
from utils import iterate_train, predict


def custom_transforms(img, mask):
    if random.random() > 0.5:
        angle = random.randint(-30, 30)
        translate = (0.4 * random.random(), 0.4 * random.random())
        scale = 0.5 * random.random() + 0.5
        shear = (random.randint(-30, 30), random.randint(-30, 30))
        img = TF.affine(img, angle, translate, scale, shear)
        mask = TF.affine(mask, angle, translate, scale, shear)
    return img, mask


train_dataset = Luminous(split="train", transforms=custom_transforms)
val_dataset = Luminous(split="val")
test_dataset = Luminous(split="test")


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
test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_batch)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Current device:', DEVICE)


def init_xavier_uniform(module):
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()


CHANNELS = [1, 64, 128, 256, 512, 1024]
NUM_CLASSES = 2  # 0 and 1
DROPOUT = 0.5  # used in the original paper
model = UNet(CHANNELS, NUM_CLASSES, DROPOUT).to(DEVICE)
model.apply(init_xavier_uniform)
optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()
print(model)

train_loss_history, val_loss_history = iterate_train(
    model, train_dataloader, val_dataloader, optimizer, loss_fn, DEVICE)
plt.plot(train_loss_history)
plt.title('Training loss history')
plt.show()


test_loss_history = []
test_pixel_acc = 0.
test_iou = 0.
for data in tqdm(test_dataloader, desc="test"):
    loss, pred = predict(model, data, loss_fn, DEVICE)
    test_loss_history.append(loss)
    test_pixel_acc += (pred.argmax(dim=1) == data[1].to(DEVICE, dtype=torch.long)).type(
        torch.float).sum().item()
    test_iou += jaccard_index(pred.argmax(dim=1),
                              data[1].to(DEVICE, dtype=torch.long), num_classes=2)
avg_test_loss = sum(test_loss_history) / len(test_dataloader)
test_pixel_acc /= len(test_dataloader)
test_iou /= len(test_dataloader)
print(f"avg. test loss: {avg_test_loss:10.6f}")
print(f"avg. test pixel acc.: {test_pixel_acc:7.5f}")
print(f"avg. test IoU.: {test_iou:7.5f}")
