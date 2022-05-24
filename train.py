"""
Train U-Net with Luminous database
"""
import argparse
import os
import torch
import torchvision.transforms as T

from torch import optim
from torch.utils.data import DataLoader, random_split
from ignite.contrib.handlers import ProgressBar, global_step_from_engine
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Accuracy, Loss, IoU
from ignite.metrics.confusion_matrix import ConfusionMatrix

from data import Luminous
from model import UNet
from utils import collate_train_batch, collate_batch, kaiming_normal_initialize, SegmentationLoss

IN_CHANNELS = 1
NUM_CLASSES = 2
SPLIT_RATIO = (0.7, 0.15, 0.15)

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", default=2, type=int)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--dropout", default=0.5, type=float)
parser.add_argument("--learning-rate", default=1e-3)
parser.add_argument("--num-epochs", default=20, type=int)
parser.add_argument("--early-stop", dest='early_stop', action='store_true')
parser.set_defaults(early_stop=False)
parser.add_argument("--log-interval", default=50, type=int)
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
    loss_fn = SegmentationLoss(num_classes=NUM_CLASSES)

    trainer = create_supervised_trainer(
        model, optimizer, loss_fn, device=args.device)
    ProgressBar().attach(trainer, output_transform=lambda x: {'loss': x})

    val_metrics = {
        "accuracy": Accuracy(),
        "loss": Loss(loss_fn),
        "IoU": IoU(ConfusionMatrix(NUM_CLASSES)),
    }
    evaluator = create_supervised_evaluator(
        model, metrics=val_metrics, device=args.device)
    ProgressBar().attach(evaluator)

    @trainer.on(Events.EPOCH_STARTED)
    def log_epoch_no(_trainer):
        print(f"Epoch {_trainer.state.epoch}:")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(_trainer):
        evaluator.run(train_dataloader)
        metrics = evaluator.state.metrics
        print("\t[TRAIN RESULT]",
              f"avg. acc: {metrics['accuracy']:.5f}",
              f"avg. loss: {metrics['loss']:.5f}",
              f"avg. IoU: {metrics['IoU'].mean():.5f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(_trainer):
        evaluator.run(val_dataloader)
        metrics = evaluator.state.metrics
        print("\t[VAL. RESULT] ",
              f"avg. acc: {metrics['accuracy']:.5f}",
              f"avg. loss: {metrics['loss']:.5f}",
              f"avg. IoU: {metrics['IoU'].mean():.5f}")
    
    def score_function(engine):
        val_loss = engine.state.metrics['loss']
        return -val_loss
    
    if args.early_stop:
        handler = EarlyStopping(patience=5, score_function=score_function, trainer=trainer)
        evaluator.add_event_handler(Events.COMPLETED, handler)

    model_checkpoint = ModelCheckpoint(args.model_dir, n_saved=3, score_function=score_function, global_step_transform=global_step_from_engine(trainer))
    evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, { "model": model })

    encoder_checkpoint = ModelCheckpoint(args.encoder_dir, n_saved=3, score_function=score_function, global_step_transform=global_step_from_engine(trainer))
    evaluator.add_event_handler(Events.COMPLETED, encoder_checkpoint, { "encoder": model.encoder })

    trainer.run(train_dataloader, max_epochs=100)
