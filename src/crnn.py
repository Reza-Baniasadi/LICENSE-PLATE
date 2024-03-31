import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class PlateSequenceModel(pl.LightningModule):
    def __init__(self, img_height, in_channels, num_labels, hidden_units, lstm_feat_dim,
                 learning_rate, reduce_lr_factor, patience_limit, min_learning_rate):
        super().__init__()
        self.save_hyperparameters()

        # Feature extractor
        self.encoder_block = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.ELU(),
            nn.MaxPool2d(2, 2)
        )

        # Sequential model
        self.sequence_model = nn.LSTM(
            input_size=lstm_feat_dim,
            hidden_size=hidden_units,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # Prediction layer
        self.prediction_head = nn.Linear(hidden_units * 2, num_labels)
        self.lr_val = learning_rate

    def forward_pass(self, images):
        features = self.encoder_block(images)        # [B, C, H, W]
        features = features.permute(0, 3, 1, 2)     # [B, W, C, H]
        B, W, C, H = features.shape
        sequence_input = features.reshape(B, W, C * H)
        seq_out, _ = self.sequence_model(sequence_input)
        output_logits = self.prediction_head(seq_out)
        return output_logits

    def compute_loss(self, batch_data, phase="train"):
        imgs, lbls = batch_data
        logits = self.forward_pass(imgs)
        loss_value = F.cross_entropy(logits.view(-1, logits.size(-1)), lbls.view(-1))
        self.log(f"{phase}_loss", loss_value, prog_bar=True)
        return loss_value

    def training_step(self, batch, batch_idx):
        return self.compute_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.compute_loss(batch, "val")

    def setup_optimizer_scheduler(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_val)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.hparams.reduce_lr_factor,
            patience=self.hparams.patience_limit,
            min_lr=self.hparams.min_learning_rate
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
