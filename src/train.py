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