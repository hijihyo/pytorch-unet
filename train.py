"""
Train U-Net with Luminous database
"""
import argparse
import os
import torch

from torch import optim
from ignite.contrib.handlers import ProgressBar, global_step_from_engine
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping, ModelCheckpoint
from ignite.metrics import Accuracy, Loss, IoU
from ignite.metrics.confusion_matrix import ConfusionMatrix

from model import UNet
from utils import create_dataloaders, kaiming_normal_initialize, SegmentationLoss

IN_CHANNELS = 1
NUM_CLASSES = 2
SPLIT_RATIO = (0.7, 0.15, 0.15)

parser = argparse.ArgumentParser()

parser.add_argument("--data-dir", default=".data", type=str)
parser.add_argument("--model-dir", default=".model", type=str)
parser.add_argument("--encoder-dir", default=".encoder", type=str)
parser.add_argument("--checkpoint-dir", default=".checkpoint", type=str)
parser.add_argument("--device", default="cuda", type=str)

parser.add_argument("--combine-mask", dest='combine_mask', action='store_true')
parser.set_defaults(combine_mask=False)
parser.add_argument("--batch-size", default=2, type=int)

parser.add_argument("--dropout", default=0.5, type=float)
parser.add_argument("--learning-rate", default=1e-3)
parser.add_argument("--num-epochs", default=100, type=int)
parser.add_argument("--early-stop", dest='early_stop', action='store_true')
parser.set_defaults(early_stop=False)
parser.add_argument("--patience", default=10, type=int)
parser.add_argument("--resume-file", default=None, type=str)

if __name__ == "__main__":
    torch.manual_seed(0)
    args = parser.parse_args()

    assert os.path.exists(args.data_dir)
    assert os.path.exists(args.model_dir)
    assert os.path.exists(args.encoder_dir)
    assert os.path.exists(args.checkpoint_dir)
    assert args.resume_file is None or os.path.exists(args.resume_file)
    assert args.device != "cuda" or torch.cuda.is_available()

    train_dataloader, val_dataloader, test_dataloader = \
        create_dataloaders(args.data_dir, args.combine_mask,
                           SPLIT_RATIO, args.batch_size)

    model = UNet(IN_CHANNELS, NUM_CLASSES, dropout=args.dropout)
    model = model.to(args.device)
    model.apply(kaiming_normal_initialize)

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

    def score_function(_engine):
        val_loss = _engine.state.metrics['loss']
        return -val_loss

    early_stopper = None
    if args.early_stop:
        early_stopper = EarlyStopping(
            patience=args.patience, score_function=score_function, trainer=trainer)

    model_checkpoint = ModelCheckpoint(
        args.model_dir, n_saved=3, score_function=score_function, global_step_transform=global_step_from_engine(trainer), require_empty=False)
    encoder_checkpoint = ModelCheckpoint(
        args.encoder_dir, n_saved=3, score_function=score_function, global_step_transform=global_step_from_engine(trainer), require_empty=False)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(_trainer):
        evaluator.run(train_dataloader)
        metrics = evaluator.state.metrics
        print("  [TRAIN DATA]",
              f"avg. acc: {metrics['accuracy']:.5f} |",
              f"avg. loss: {metrics['loss']:.5f} |",
              f"avg. IoU: {metrics['IoU'].mean():.5f}")

        evaluator.run(val_dataloader)
        metrics = evaluator.state.metrics
        print("  [VALID DATA]",
              f"avg. acc: {metrics['accuracy']:.5f} |",
              f"avg. loss: {metrics['loss']:.5f} |",
              f"avg. IoU: {metrics['IoU'].mean():.5f}")

        if early_stopper is not None:
            early_stopper(evaluator)
            state_dict = early_stopper.state_dict()
            print("  [EARLY STOP]",
                  f"counter: {state_dict['counter']} |",
                  f"best score: {state_dict['best_score']:.5f}")

        model_checkpoint(evaluator, to_save={"model": model})
        encoder_checkpoint(evaluator, to_save={"encoder": model.encoder})

    if args.resume_file is not None:
        to_save = {"model": model, "optimizer": optimizer, "trainer": trainer}
        checkpoint = torch.load(args.resume_file, map_location=args.device)
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)
    to_save = {"model": model, "optimizer": optimizer, "trainer": trainer}
    checkpoint = Checkpoint(to_save, DiskSaver(args.checkpoint_dir, create_dir=True, require_empty=False), n_saved=1,
                                global_step_transform=global_step_from_engine(trainer))

    def log_test_results(_trainer):
        print("Completed:")
        evaluator.run(test_dataloader)
        metrics = evaluator.state.metrics
        print("  [TEST  DATA]",
              f"avg. acc: {metrics['accuracy']:.5f} |",
              f"avg. loss: {metrics['loss']:.5f} |",
              f"avg. IoU: {metrics['IoU'].mean():.5f}")

    trainer.add_event_handler(Events.COMPLETED, log_test_results)
    trainer.add_event_handler(Events.TERMINATE, log_test_results)
    trainer.add_event_handler(Events.COMPLETED, checkpoint)
    trainer.add_event_handler(Events.TERMINATE, checkpoint)

    if args.resume_file is None:
        print("Data is stored at", args.data_dir)
        print("Model will be stored at", args.model_dir)
        print("Encoder will be stored at", args.encoder_dir)
        print("Current device:", args.device)
        print()

        print("Number of (train, val, test) examples:", len(train_dataloader.dataset), len(
            val_dataloader.dataset), len(test_dataloader.dataset))
        print("Batch size:", args.batch_size)
        print("Dropout:", args.dropout)
        print("Learning rate:", args.learning_rate)
        print("Early stopping:",
              f"ON (patience: {args.patience})" if args.early_stop else "OFF")
        print(model)
        print()

    trainer.run(train_dataloader, max_epochs=args.num_epochs)
