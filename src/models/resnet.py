"""
ResNet-based countermeasure (E5 model ablation).
Wrapped as a PyTorch Lightning module.
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class ResNet(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        mc = cfg.model

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layers = nn.Sequential(
            *[ResBlock(32) for _ in range(sum(mc.layers))],
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, mc.num_classes),
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.classifier(self.layers(self.stem(x)))

    def _step(self, batch, split):
        feats, labels = batch
        logits = self(feats)
        loss   = self.criterion(logits, labels)
        acc    = (logits.argmax(1) == labels).float().mean()
        self.log(f"{split}/loss", loss, prog_bar=True)
        self.log(f"{split}/acc",  acc,  prog_bar=True)
        return loss, logits.softmax(1)[:, 1], labels

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
        return torch.optim.Adam(self.parameters(),
                                lr=mc.lr, weight_decay=mc.weight_decay)
