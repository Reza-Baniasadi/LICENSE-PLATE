import warnings
warnings.filterwarnings("ignore")
from argparse import ArgumentParser
from pathlib import Path
import torch
import pytorch_lightning as pl
from deep_utils import mkdir_incremental, CRNNModelTorch, get_logger, TorchUtils, visualize_data_loader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import CRNNDataset
from settings import Config
from torch.nn import CTCLoss
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

torch.backends.cudnn.benchmark = True


class LitCRNN(pl.LightningModule):
    def __init__(self, img_h, n_channels, n_classes, n_hidden, lstm_input, lr, lr_reduce_factor, lr_patience, min_lr):
        super(LitCRNN, self).__init__()
        self.save_hyperparameters()
        self.model = CRNNModelTorch(img_h=self.hparams.img_h,
                                    n_channels=self.hparams.n_channels,
                                    n_classes=self.hparams.n_classes,
                                    n_hidden=self.hparams.n_hidden,
                                    lstm_input=self.hparams.lstm_input)
        self.model.apply(self.model.weights_init)
        self.criterion = CTCLoss(reduction='mean')


    def forward(self, x):
        logit = self.model(x)
        logit = torch.transpose(logit, 1, 0)
        return logit