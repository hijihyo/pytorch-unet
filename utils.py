"""
utility methods for pytorch-unet
"""
import datetime
from tqdm import tqdm
import torch


def predict_time(start_time, current_time, progress):
    """Predict the end time"""
    elapsed_time = current_time - start_time
    predicted_time = start_time + elapsed_time / progress
    return predicted_time


def train(model, dataloader, optimizer, loss_fn, device: str):
    """Train the model for one epoch"""
    model.train()
    # batch_size = dataloader.batch_size
    # num_batches = len(dataloader)
    loss_history = []
    for img, mask in tqdm(dataloader, desc="  training"):
        img, mask = img.to(device), mask.to(device, dtype=torch.long)
        pred = model(img)
        loss = loss_fn(pred, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
    return loss_history


def evaluate(model, dataloader, loss_fn, device: str):
    """Evaluate the model on dataset"""
    model.eval()
    # batch_size = dataloader.batch_size
    # num_batches = len(dataloader)
    loss_history = []
    with torch.no_grad():
        for img, mask in tqdm(dataloader, desc="  validation"):
            img, mask = img.to(device), mask.to(device, dtype=torch.long)
            pred = model(img)
            loss = loss_fn(pred, mask)
            loss_history.append(loss.item())
    return loss_history


def iterate_train(
    model, train_dataloader, val_dataloader, optimizer, loss_fn,
    device: str, num_epochs: int = 1, save_checkpoint: bool = True
):
    """Iterate training the model with validation"""
    train_loss_history = []
    val_loss_history = []
    start_time = datetime.datetime.now()
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}...")
        loss_history = train(model, train_dataloader,
                             optimizer, loss_fn, device)
        train_loss_history += loss_history
        avg_train_loss = sum(loss_history) / len(train_dataloader)
        loss_history = evaluate(model, val_dataloader, loss_fn, device)
        val_loss_history += loss_history
        avg_val_loss = sum(loss_history) / len(val_dataloader)
        print(f"  avg. training loss: {avg_train_loss:10.6f}")
        print(f"  avg. validation loss: {avg_val_loss:10.6f}")
        predicted_time = predict_time(
            start_time, datetime.datetime.now(), (epoch / num_epochs))
        print("  expected end time:",
                     predicted_time.strftime("%Y-%m-%d %H:%M:%S"))
        if save_checkpoint:
            torch.save(model.state_dict(), f'epoch{epoch}.pth')
    print('Done!')
    return train_loss_history, val_loss_history
