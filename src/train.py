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