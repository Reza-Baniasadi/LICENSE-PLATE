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
    

    def get_loss(self, batch):
        images, labels, labels_lengths = batch
        labels_lengths = labels_lengths.squeeze(1)
        batch_size = images.size(0)
        logits = self.model(images)
        input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
        loss = self.criterion(logits, labels, input_lengths, labels_lengths)
        return loss, batch_size
    
    @staticmethod
    def calculate_metrics(outputs):
        r_loss, size = 0, 0
        for row in outputs:
            r_loss += row["loss"]
            size += row["bs"]
        loss = r_loss / size
        return loss
    
    def test_step(self, batch):
        loss, bs = self.get_loss(batch)
        return {"loss": loss, "bs": bs}
    

    def training_step(self, batch):
        loss, bs = self.get_loss(batch)
        return {"loss": loss, "bs": bs}
    
    def validation_step(self, batch):
        loss, bs = self.get_loss(batch)
        return {"loss": loss, "bs": bs}
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=self.hparams.lr_reduce_factor,
                                      patience=self.hparams.lr_patience, verbose=True, min_lr=self.hparams.min_lr)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}