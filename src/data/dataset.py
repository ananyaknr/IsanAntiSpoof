"""
Dataset and LightningDataModule for Isan anti-spoofing.

AntiSpoofDataset:   reads protocol.txt, returns (feature_tensor, label)
AntiSpoofDataModule: wraps train/dev/eval splits for PyTorch Lightning
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
from pathlib import Path
from omegaconf import DictConfig


class AntiSpoofDataset(Dataset):
    LABEL_MAP = {"bonafide": 1, "spoof": 0}

    def __init__(self, protocol_path: str, feature_root: str,
                 feature_type: str, split: str,
                 active_datasets: list = None):
        """
        Args:
            protocol_path:   path to protocol.txt
            feature_root:    root of data/processed/
            feature_type:    'lfcc' | 'mfcc' | 'cqcc'
            split:           'train' | 'dev' | 'eval'
            active_datasets: filter to specific dataset sources (None = all)
        """
        self.samples = []
        feature_root = Path(feature_root)

        with open(protocol_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 6:
                    continue
                utt_id, dataset, _, label, s, _ = parts
                if s != split:
                    continue
                if active_datasets and dataset not in active_datasets:
                    continue
                feat_path = (feature_root / dataset / "features"
                             / feature_type / split / f"{utt_id}.npy")
                if feat_path.exists():
                    self.samples.append((feat_path, self.LABEL_MAP[label]))

        n_bona  = sum(1 for _, l in self.samples if l == 1)
        n_spoof = sum(1 for _, l in self.samples if l == 0)
        print(f"[Dataset] {split}/{feature_type}: "
              f"{len(self.samples)} samples  "
              f"(bonafide={n_bona}, spoof={n_spoof})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat_path, label = self.samples[idx]
        if feat_path.suffix == ".npz":
            feat = np.load(feat_path)["feat"]
        else:
            feat = np.load(feat_path)
        return torch.FloatTensor(feat), torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    """Pad variable-length sequences to longest in batch."""
    feats, labels = zip(*batch)
    feats_padded = pad_sequence(feats, batch_first=True, padding_value=0.0)
    return feats_padded, torch.stack(labels)


class AntiSpoofDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def _make_dataset(self, split: str) -> AntiSpoofDataset:
        active = getattr(self.cfg.experiment.dataset, split, None)
        return AntiSpoofDataset(
            protocol_path   = self.cfg.dataset.protocol_path,
            feature_root    = self.cfg.dataset.processed_root,
            feature_type    = self.cfg.feature.type,
            split           = split,
            active_datasets = list(active) if active else None,
        )

    def setup(self, stage=None):
        self.train_ds = self._make_dataset("train")
        self.dev_ds   = self._make_dataset("dev")
        self.eval_ds  = self._make_dataset("eval")

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size  = self.cfg.model.batch_size,
                          shuffle     = True,
                          collate_fn  = collate_fn,
                          num_workers = 4)

    def val_dataloader(self):
        return DataLoader(self.dev_ds,
                          batch_size  = self.cfg.model.batch_size,
                          shuffle     = False,
                          collate_fn  = collate_fn,
                          num_workers = 4)

    def test_dataloader(self):
        return DataLoader(self.eval_ds,
                          batch_size  = self.cfg.model.batch_size,
                          shuffle     = False,
                          collate_fn  = collate_fn,
                          num_workers = 4)
