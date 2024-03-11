import os
import logging
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torchvision
import matplotlib.pyplot as plt


class Config:
    def __init__(self):
        self.output_dir = "./output"
        self.epochs = 100
        self.device = "cuda"
        self.mean = [0.4845]
        self.std = [0.1884]
        self.img_h = 32
        self.img_w = 100
        self.n_channels = 1
        self.n_classes = 50   
        self.n_hidden = 256
        self.lstm_input = 256
        self.lr = 1e-3
        self.lr_reduce_factor = 0.1
        self.lr_patience = 5
        self.min_lr = 1e-6
        self.early_stopping_patience = 10
        self.file_name = "best_model"
        self.batch_size = 128
        self.n_workers = 8

    def update_config_param(self, args):
        self.__dict__.update(vars(args))


    def mkdir_incremental(path: Path):
        path = Path(path)
        i = 0
        new_path = path
        while new_path.exists():
            i += 1
            new_path = path.parent / f"{path.name}_{i}"
        new_path.mkdir(parents=True, exist_ok=True)
        return new_path
    
    def get_logger(name, log_path=None):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if log_path:
            fh = logging.FileHandler(log_path)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        return logger
    

    def visualize_data_loader(loader, mean, std):
        imgs, labels = next(iter(loader))
        imgs = imgs[:8]
        imgs = imgs * torch.tensor(std).view(-1, 1, 1) + torch.tensor(mean).view(-1, 1, 1)

        grid = torchvision.utils.make_grid(imgs, nrow=4)
        plt.imshow(grid.permute(1, 2, 0))
        plt.show()

class LitCRNN(pl.LightningModule):
        def __init__(self, img_h, n_channels, n_classes, n_hidden, lstm_input,
                    lr, lr_reduce_factor, lr_patience, min_lr):
            super().__init__()
            self.save_hyperparameters()

            self.conv = nn.Sequential(
                nn.Conv2d(n_channels, 64, 3, 1, 1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            self.rnn = nn.LSTM(lstm_input, n_hidden, num_layers=2,
                            bidirectional=True, batch_first=True)
            self.fc = nn.Linear(n_hidden * 2, n_classes)
            self.lr = lr



        def forward(self, x):
            x = self.conv(x)  # [B, C, H, W]
            x = x.permute(0, 3, 1, 2)  # [B, W, C, H]
            B, W, C, H = x.size()
            x = x.reshape(B, W, C * H)
            x, _ = self.rnn(x)
            x = self.fc(x)
            return x


        def training_step(self, batch, batch_idx):
            imgs, labels = batch
            preds = self(imgs)
            
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), labels.view(-1))
            self.log("train_loss", loss)
            return loss
        

        def validation_step(self, batch, batch_idx):
            imgs, labels = batch
            preds = self(imgs)
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), labels.view(-1))
            self.log("val_loss", loss)
            return loss
        
        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=self.hparams.lr_reduce_factor,
                patience=self.hparams.lr_patience, min_lr=self.hparams.min_lr
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        

        def get_loaders(config):
            transform = transforms.Compose([
                transforms.Resize((config.img_h, config.img_w)),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.mean, std=config.std)
            ])

            train_dataset = datasets.ImageFolder(config.train_directory, transform=transform)
            val_dataset = datasets.ImageFolder(config.val_directory, transform=transform)

            train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                                    shuffle=True, num_workers=config.n_workers)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                                    shuffle=False, num_workers=config.n_workers)
            return train_loader, val_loader


# ---------------- Main ----------------
def main():
    parser = ArgumentParser()
    parser.add_argument("--train_directory", type=Path,
                        default="/home/ai/projects/vehicle-plate-recognition-training/recognition/datasets/train",
                        help="path to the dataset, default: ./dataset")
    parser.add_argument("--val_directory", type=Path,
                        default="/home/ai/projects/vehicle-plate-recognition-training/recognition/datasets/val",
                        help="path to the dataset, default: ./dataset")
    parser.add_argument("--output_dir", type=Path, default="./output",
                        help="path to the output directory, default: ./output")
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--device", default="cuda", help="training device, default is cuda")
    parser.add_argument("--mean", nargs="+", type=float, default=[0.4845], help="dataset channel-wise mean")
    parser.add_argument("--std", nargs="+", type=float, default=[0.1884], help="dataset channel-wise std")
    parser.add_argument("--img_w", type=int, default=100, help="dataset img width size")
    parser.add_argument("--n_workers", type=int, default=8, help="number of workers for dataloader")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--alphabets", default='ابپتشثجدزسصطعفقکگلمنوهی+۰۱۲۳۴۵۶۷۸۹',
                        help="alphabets used in the process")
    parser.add_argument("--visualize", action="store_true", help="Visualize data-loader")

    args = parser.parse_args()
    config = Config()
    config.update_config_param(args)

    output_dir = mkdir_incremental(config.output_dir)
    logger = get_logger("pytorch-lightning-image-classification", log_path=output_dir / "log.log")

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