import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class SequencePlateNet(pl.LightningModule):
    def __init__(self, height, channels, num_classes, hidden_dim, lstm_input_dim,
                 lr_value, reduce_factor, patience_count, min_lr_value):
        super().__init__()
        self.save_hyperparameters()

        self.feature_net = nn.Sequential(
            nn.Conv2d(channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2)
        )

        self.lstm_net = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.output_layer = nn.Linear(hidden_dim * 2, num_classes)
        self.lr_value = lr_value

    def forward(self, input_tensor):
        feat_map = self.feature_net(input_tensor)          # [B, C, H, W]
        feat_map = feat_map.permute(0, 3, 1, 2)           # [B, W, C, H]
        batch_size, width, channels, height = feat_map.shape
        seq_input = feat_map.reshape(batch_size, width, channels * height)
        lstm_out, _ = self.lstm_net(seq_input)
        predictions = self.output_layer(lstm_out)
        return predictions

    def step(self, batch, prefix="train"):
        imgs, labels = batch
        logits = self(imgs)
        loss_val = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.log(f"{prefix}_loss", loss_val, prog_bar=True)
        return loss_val

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr_value)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            factor=self.hparams.reduce_factor,
            patience=self.hparams.patience_count,
            min_lr=self.hparams.min_lr_value
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}
