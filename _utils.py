"""
Utility methods
"""
import os
import datetime
from pytz import timezone
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics.functional import jaccard_index


def predict_time(start_time, current_time, progress):
    """Predict the end time"""
    elapsed_time = current_time - start_time
    predicted_time = start_time + elapsed_time / progress
    return predicted_time


def dice_loss(pred: Tensor, target: Tensor, eps: float = 1e-6):
    """Computes dice loss between prediction and target (only for binary class"""
    assert pred.dim() == target.dim()  # (H, W) or (B, H, W)
    is_batched = pred.dim() == 3
    if not is_batched:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    dice = 0.
    for i, _ in enumerate(pred):
        inter = torch.sum(pred[i].view(-1) * target[i].view(-1))
        union = pred[i].sum() + target[i].sum()
        dice += (2 * inter + eps) / (union + eps)
    dice /= pred.size(0)
    return 1 - dice


def train(model, dataloader, optimizer, loss_fn, device: str):
    """Train the model for one epoch"""
    model.train()
    # batch_size = dataloader.batch_size
    # num_batches = len(dataloader)
    loss_history = []
    pixel_acc_history = []
    iou_history = []
    for img, mask in tqdm(dataloader, desc="  train"):
        img, mask = img.to(device), mask.to(device, dtype=torch.long)
        pred = model(img)
        loss = loss_fn(pred, mask) + dice_loss(F.softmax(pred), mask)
        # loss = loss_fn(pred, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        pixel_acc = (pred.argmax(dim=1) == mask).sum().item() / \
            (pred.size(0) * pred.size(2) * pred.size(3))
        pixel_acc_history.append(pixel_acc)
        iou = jaccard_index(pred.argmax(dim=1), mask, num_classes=2)
        iou_history.append(iou)
    return loss_history, pixel_acc_history, iou_history


def evaluate(
    model, dataloader, loss_fn, device: str, desc: str = "  validation"
):
    """Evaluate the model on dataset"""
    model.eval()
    # batch_size = dataloader.batch_size
    # num_batches = len(dataloader)
    loss_history = []
    pixel_acc_history = []
    iou_history = []
    with torch.no_grad():
        for img, mask in tqdm(dataloader, desc=desc):
            img, mask = img.to(device), mask.to(device, dtype=torch.long)
            pred = model(img)
            loss = loss_fn(pred, mask) + dice_loss(F.softmax(pred), mask)
            # loss = loss_fn(pred, mask)
            loss_history.append(loss.item())
            pixel_acc = (pred.argmax(dim=1) == mask).sum().item() / \
                (pred.size(0) * pred.size(2) * pred.size(3))
            pixel_acc_history.append(pixel_acc)
            iou = jaccard_index(pred.argmax(dim=1), mask, num_classes=2)
            iou_history.append(iou)
    return loss_history, pixel_acc_history, iou_history


def iterate_train(
    model, train_dataloader, val_dataloader, optimizer, loss_fn,
    device: str, num_epochs: int = 1, model_dir: str = ".model", encoder_dir: str = ".encoder"
):
    """Iterate training the model with validation"""
    train_history = {"loss": [], "pixel_acc": [], "iou": []}
    val_history = {"loss": [], "pixel_acc": [], "iou": []}
    start_time = datetime.datetime.now(timezone('Asia/Seoul'))
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}...")
        loss_history, pixel_acc_history, iou_history = train(model, train_dataloader,
                                                             optimizer, loss_fn, device)
        train_history["loss"] += loss_history
        train_history["pixel_acc"] += pixel_acc_history
        train_history["iou"] += iou_history
        avg_train_loss = sum(loss_history) / len(train_dataloader)
        avg_train_pa = sum(pixel_acc_history) / len(train_dataloader)
        avg_train_iou = sum(iou_history) / len(train_dataloader)

        loss_history, pixel_acc_history, iou_history = evaluate(
            model, val_dataloader, loss_fn, device)
        val_history["loss"] += loss_history
        val_history["pixel_acc"] += pixel_acc_history
        val_history["iou"] += iou_history
        avg_val_loss = sum(loss_history) / len(val_dataloader)
        avg_val_pa = sum(pixel_acc_history) / len(val_dataloader)
        avg_val_iou = sum(iou_history) / len(val_dataloader)

        print()
        print(f"  avg. training loss: {avg_train_loss:10.6f}")
        print(f"  avg. training pixel acc.: {avg_train_pa:10.6f}")
        print(f"  avg. training IoU: {avg_train_iou:10.6f}")

        print()
        print(f"  avg. validation loss: {avg_val_loss:10.6f}")
        print(f"  avg. validation pixel acc.: {avg_val_pa:10.6f}")
        print(f"  avg. validation IoU: {avg_val_iou:10.6f}")

        predicted_time = predict_time(
            start_time, datetime.datetime.now(timezone('Asia/Seoul')), (epoch / num_epochs))
        print()
        print("  expected end time:",
              predicted_time.strftime("%Y-%m-%d %H:%M:%S"))
        torch.save(model.state_dict(), os.path.join(
            model_dir, f'model-epoch{epoch}.pth'))
        torch.save(model.encoder.state_dict(), os.path.join(
            encoder_dir, f'encoder-epoch{epoch}.pth'))
        print()

    print('Done!')
    return train_history, val_history