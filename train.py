"""
Train U-Net with Luminous database
"""
import argparse
import os
import torch
import torchvision.transforms as T

from torch import optim
from torch.utils.data import DataLoader, random_split
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import EarlyStopping
from ignite.metrics import Accuracy, Loss, IoU
from ignite.metrics.confusion_matrix import ConfusionMatrix

from data import Luminous
from model import UNet
from utils import collate_train_batch, collate_batch, kaiming_normal_initialize, SegmentationLoss

IN_CHANNELS = 1
NUM_CLASSES = 2
SPLIT_RATIO = (0.7, 0.15, 0.15)
LOG_INTERVAL = 10

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", default=2, type=int)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--dropout", default=0.5, type=float)
parser.add_argument("--learning-rate", default=1e-3)
parser.add_argument("--num-epochs", default=20, type=int)
parser.add_argument("--early-stop", default=False, type=bool)
parser.add_argument("--data-dir", default=".data", type=str)
parser.add_argument("--model-dir", default=".model", type=str)
parser.add_argument("--encoder-dir", default=".encoder", type=str)

if __name__ == "__main__":
    args = parser.parse_args()

    assert os.path.exists(args.data_dir)
    assert os.path.exists(args.model_dir)
    assert os.path.exists(args.encoder_dir)
    assert args.device != "cuda" or torch.cuda.is_available()
    print("Current device:", args.device)

    transform = T.Compose([
        T.ToTensor(),
        T.Pad((6, 5)),  # (614, 820) -> (624, 832)
    ])
    dataset = Luminous(root=args.data_dir, transform=transform,
                       target_transform=transform)
    num_train = int(len(dataset) * SPLIT_RATIO[0])
    num_val = int(len(dataset) * SPLIT_RATIO[1])
    num_test = len(dataset) - num_train - num_val
    train_dataset, val_dataset, test_dataset = \
        random_split(dataset, (num_train, num_val, num_test))

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_train_batch)
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, collate_fn=collate_batch)
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, collate_fn=collate_batch)

    model = UNet(IN_CHANNELS, NUM_CLASSES, dropout=args.dropout)
    model = model.to(args.device)
    model.apply(kaiming_normal_initialize)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = SegmentationLoss()

    trainer = create_supervised_trainer(
        model, optimizer, loss_fn, device=args.device)

    val_metrics = {
        "accuracy": Accuracy(),
        "loss": Loss(loss_fn),
        "IoU": IoU(ConfusionMatrix(NUM_CLASSES)),
    }
    evaluator = create_supervised_evaluator(
        model, metrics=val_metrics, device=args.device)

    @trainer.on(Events.ITERATION_COMPLETED(every=LOG_INTERVAL))
    def log_training_loss(_trainer):
        print(
            f"Epoch[{_trainer.state.epoch}] Loss: {_trainer.state.output:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(_trainer):
        evaluator.run(train_dataloader)
        metrics = evaluator.state.metrics
        print(
            f"Training Results - Epoch: {_trainer.state.epoch} Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f} Avg IoU: {metrics['IoU']:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(_trainer):
        evaluator.run(val_dataloader)
        metrics = evaluator.state.metrics
        print(
            f"Validation Results - Epoch: {_trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f} Avg IoU: {metrics['IoU']:.2f}")
    
    def score_function(engine):
        val_accuracy = engine.state.metrics['accuracy']
        return val_accuracy
    
    handler = EarlyStopping(patience=5, score_function=score_function, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, handler)

    trainer.run(train_dataloader, max_epochs=1)
