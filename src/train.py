from argparse import ArgumentParser
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from config import TrainingConfig
from models.crnn_model import CRNNNetwork
from data.data_utils import load_datasets
from data.visualize_utils import display_batch
from utils.logger_utils import setup_logger
from utils.file_ops import ensure_unique_dir


def launch_training():
    parser = ArgumentParser(description="CRNN Training Pipeline")
    parser.add_argument("--train_folder", type=Path, default="./datasets/train")
    parser.add_argument("--validation_folder", type=Path, default="./datasets/val")
    parser.add_argument("--results_folder", type=Path, default="./results")
    parser.add_argument("--num_epochs", type=int, default=120)
    parser.add_argument("--device_type", default="cuda")
    parser.add_argument("--mean_vals", nargs="+", type=float, default=[0.4845])
    parser.add_argument("--std_vals", nargs="+", type=float, default=[0.1884])
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--alphabet_chars", default='ابپتشثجدزسصطعفقکگلمنوهی+۰۱۲۳۴۵۶۷۸۹')
    parser.add_argument("--preview_data", action="store_true")

    args = parser.parse_args()
    cfg = TrainingConfig()
    cfg.update_from_args(args)

    output_path = ensure_unique_dir(cfg.results_path)
    log = setup_logger("CRNN_Lightning", file_output=output_path / "train.log")

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=cfg.early_stop_patience)
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path,
        filename=cfg.best_model_name,
        monitor="val_loss",
        verbose=True
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer_engine = pl.Trainer(
        accelerator="gpu" if cfg.device == "cuda" else "cpu",
        devices=1,
        max_epochs=cfg.epochs,
        callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
        default_root_dir=output_path
    )

    model_net = CRNNNetwork(
        input_height=cfg.img_height,
        input_channels=cfg.channels,
        num_classes=cfg.num_classes,
        hidden_units=cfg.hidden_units,
        lstm_features=cfg.lstm_input_dim,
        learning_rate=cfg.lr,
        lr_reduce_factor=cfg.lr_reduce_factor,
        lr_patience=cfg.lr_patience,
        min_lr=cfg.min_lr
    )

    train_loader, val_loader = load_datasets(cfg)

    if args.preview_data:
        print("[INFO] Displaying sample batches from train_loader")
        display_batch(train_loader, mean=cfg.mean_vals, std=cfg.std_vals)
        print("[INFO] Displaying sample batches from val_loader")
        display_batch(val_loader, mean=cfg.mean_vals, std=cfg.std_vals)
        exit(0)

    trainer_engine.fit(model_net, train_loader, val_loader)


if __name__ == "__main__":
    launch_training()
