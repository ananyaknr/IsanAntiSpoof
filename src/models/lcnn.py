"""
LCNN (Light CNN) with Max Feature Map activation.
Wrapped as a PyTorch Lightning module.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig


class MaxFeatureMap(nn.Module):
    """MFM: split channels in half, take element-wise maximum."""
    def forward(self, x):
        assert x.size(1) % 2 == 0, "Channel dim must be even for MFM"
        half = x.size(1) // 2
        return torch.max(x[:, :half], x[:, half:])


class LCNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel, stride, padding)
        self.mfm  = MaxFeatureMap()
        self.bn   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn(self.mfm(self.conv(x)))


class LCNN(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        mc = cfg.model

        self.features = nn.Sequential(
            LCNNBlock(1,  32, kernel=5, padding=2),   # (B, 32, T, F)
            nn.MaxPool2d(2, 2),
            LCNNBlock(32, 48, kernel=1),
            LCNNBlock(48, 48, kernel=3, padding=1),
            nn.MaxPool2d(2, 2),
            LCNNBlock(48, 64, kernel=1),
            LCNNBlock(64, 64, kernel=3, padding=1),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            MaxFeatureMap(),           # channel halved → 64
            nn.Dropout(mc.dropout),
            nn.Linear(64, mc.num_classes),
        )

        self.criterion = nn.CrossEntropyLoss()
        self._best_dev_eer = float("inf")

    def forward(self, x):
        # x: (B, T, F) → add channel dim → (B, 1, T, F)
        x = x.unsqueeze(1)
        return self.classifier(self.features(x))

    def _step(self, batch, split: str):
        feats, labels = batch
        logits = self(feats)
        loss   = self.criterion(logits, labels)
        preds  = logits.argmax(dim=1)
        acc    = (preds == labels).float().mean()
        self.log(f"{split}/loss", loss, prog_bar=True)
        self.log(f"{split}/acc",  acc,  prog_bar=True)
        return loss, logits.softmax(dim=1)[:, 1], labels

    def training_step(self, batch, _):
        loss, _, _ = self._step(batch, "train")
        return loss

    def validation_step(self, batch, _):
        _, scores, labels = self._step(batch, "dev")
        return {"scores": scores.cpu(), "labels": labels.cpu()}

    def test_step(self, batch, _):
        _, scores, labels = self._step(batch, "eval")
        return {"scores": scores.cpu(), "labels": labels.cpu()}

    def configure_optimizers(self):
        mc = self.cfg.model
        opt = torch.optim.Adam(self.parameters(),
                               lr=mc.lr, weight_decay=mc.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=5)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "monitor": "dev/loss"}}
