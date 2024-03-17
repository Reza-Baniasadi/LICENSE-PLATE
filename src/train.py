from argparse import ArgumentParser
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from config import Config
from models.crnn import LitCRNN
from data.dataloader import get_loaders
from data.visualize import visualize_data_loader
from utils.logger import get_logger
from utils.misc import mkdir_incremental


def main():
    parser = ArgumentParser()
    parser.add_argument("--train_directory", type=Path, default="./datasets/train")
    parser.add_argument("--val_directory", type=Path, default="./datasets/val")
    parser.add_argument("--output_dir", type=Path, default="./output")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mean", nargs="+", type=float, default=[0.4845])
    parser.add_argument("--std", nargs="+", type=float, default=[0.1884])
    parser.add_argument("--img_w", type=int, default=100)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--alphabets", default='ابپتشثجدزسصطعفقکگلمنوهی+۰۱۲۳۴۵۶۷۸۹')
    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()
    config = Config()
    config.update_config_param(args)

    output_dir = mkdir_incremental(config.output_dir)
    logger = get_logger("pytorch-lightning-crnn", log_path=output_dir / "log.log")

    early_stopping = EarlyStopping(monitor='val_loss', patience=config.early_stopping_patience)
    model_checkpoint = ModelCheckpoint(dirpath=output_dir, filename=config.file_name,
                                       monitor="val_loss", verbose=True)
    learning_rate_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        accelerator="gpu" if config.device == "cuda" else "cpu",
        devices=1,
        max_epochs=config.epochs,
        min_epochs=config.epochs // 10,
        callbacks=[early_stopping, model_checkpoint, learning_rate_monitor],
        default_root_dir=output_dir
    )

    model = LitCRNN(config.img_h, config.n_channels, config.n_classes,
                    config.n_hidden, config.lstm_input, config.lr,
                    config.lr_reduce_factor, config.lr_patience, config.min_lr)

    train_loader, val_loader = get_loaders(config)

    if args.visualize:
        print("[INFO] Visualizing train-loader")
        visualize_data_loader(train_loader, mean=config.mean, std=config.std)
        print("[INFO] Visualizing val-loader")
        visualize_data_loader(val_loader, mean=config.mean, std=config.std)
        exit(0)

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
