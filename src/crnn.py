import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


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
