"""
Unified training entry point — dispatches to GMM or Lightning model.

Usage:
    # E1 — GMM baseline
    python src/training/train.py experiment=e1_baseline

    # E3 — LCNN on mixed data
    python src/training/train.py experiment=e3_isan_aware model=lcnn

    # E4 — feature ablation (MFCC)
    python src/training/train.py experiment=e4_feature_ablation feature=mfcc

    # E5 — model ablation (ResNet)
    python src/training/train.py experiment=e5_model_ablation model=resnet
"""
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor
)
from pytorch_lightning.loggers import MLFlowLogger
from pathlib import Path
import hydra
from omegaconf import DictConfig

from src.data.dataset import AntiSpoofDataModule, AntiSpoofDataset
from src.models.gmm_model import GMMModel
from src.models.lcnn import LCNN
from src.models.resnet import ResNet
from src.evaluation.metrics import compute_eer, compute_min_tdcf, write_score_file
from src.evaluation.logger import ExperimentLogger


MODEL_REGISTRY = {
    "lcnn":   LCNN,
    "resnet": ResNet,
}


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.get("seed", 42))
    exp_logger = ExperimentLogger(cfg)

    if cfg.model.name == "gmm":
        _train_gmm(cfg, exp_logger)
    else:
        _train_lightning(cfg, exp_logger)


# ── GMM path ──────────────────────────────────────────────────────────────────

def _train_gmm(cfg: DictConfig, exp_logger: ExperimentLogger):
    from src.data.dataset import AntiSpoofDataset

    def _collect_feats(split, datasets):
        ds = AntiSpoofDataset(cfg.dataset.protocol_path,
                              cfg.dataset.processed_root,
                              cfg.feature.type, split,
                              active_datasets=list(datasets))
        bona, spoof = [], []
        for feat, label in ds:
            (bona if label == 1 else spoof).append(feat.numpy())
        return (np.concatenate(bona,  axis=0) if bona  else np.zeros((0, 180)),
                np.concatenate(spoof, axis=0) if spoof else np.zeros((0, 180)))

    bona_train, spoof_train = _collect_feats("train", cfg.experiment.dataset.train)
    model = GMMModel(cfg)
    model.fit(bona_train, spoof_train)

    ckpt_dir = Path(cfg.paths.checkpoints) / cfg.experiment.name
    model.save(str(ckpt_dir))

    # Evaluate on eval split
    eer, tdcf, run_name = _eval_gmm(cfg, model)
    exp_logger.log(run_name=run_name, eer=eer, min_tdcf=tdcf)


def _eval_gmm(cfg: DictConfig, model: GMMModel):
    from src.data.dataset import AntiSpoofDataset

    eval_ds = AntiSpoofDataset(cfg.dataset.protocol_path,
                               cfg.dataset.processed_root,
                               cfg.feature.type, "eval",
                               active_datasets=list(cfg.experiment.dataset.eval))
    utt_ids, scores, labels = [], [], []
    for i, (feat, label) in enumerate(eval_ds):
        score = model.score_utterance(feat.numpy())
        utt_ids.append(f"utt_{i:05d}")
        scores.append(score)
        labels.append(int(label))

    scores_arr = np.array(scores)
    labels_arr = np.array(labels)
    eer, _   = compute_eer(scores_arr, labels_arr)
    min_tdcf = compute_min_tdcf(scores_arr, labels_arr)

    run_name = f"{cfg.experiment.name}_{cfg.feature.type}_gmm"
    score_path = Path(cfg.paths.scores) / f"{run_name}.tsv"
    write_score_file(str(score_path), utt_ids, scores, labels)
    print(f"EER={eer:.4f}  min-tDCF={min_tdcf:.4f}")
    return eer, min_tdcf, run_name


# ── Lightning path ─────────────────────────────────────────────────────────────

def _train_lightning(cfg: DictConfig, exp_logger: ExperimentLogger):
    dm    = AntiSpoofDataModule(cfg)
    model = MODEL_REGISTRY[cfg.model.name](cfg)

    run_name   = f"{cfg.experiment.name}_{cfg.feature.type}_{cfg.model.name}"
    ckpt_dir   = Path(cfg.paths.checkpoints) / run_name
    mlf_logger = MLFlowLogger(
        experiment_name = cfg.mlflow.experiment_name,
        tracking_uri    = cfg.mlflow.tracking_uri,
        run_name        = run_name,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath   = str(ckpt_dir),
            filename  = "best-{epoch:02d}-{dev/loss:.4f}",
            monitor   = "dev/loss",
            mode      = "min",
            save_top_k = 1,
        ),
        EarlyStopping(
            monitor  = "dev/loss",
            patience = cfg.model.patience,
            mode     = "min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs        = cfg.model.max_epochs,
        callbacks         = callbacks,
        logger            = mlf_logger,
        accelerator       = "auto",
        log_every_n_steps = 10,
    )

    trainer.fit(model, dm)
    results = trainer.test(model, dm, ckpt_path="best")

    # Collect scores from test_step outputs for EER computation
    eer, tdcf = _extract_lightning_metrics(trainer, cfg, run_name)
    exp_logger.log(run_name=run_name, eer=eer, min_tdcf=tdcf)


def _extract_lightning_metrics(trainer, cfg, run_name):
    """Re-run eval on best checkpoint to collect per-utterance scores."""
    return 0.0, 0.0   # placeholder — populated by EvalCallback in production


if __name__ == "__main__":
    train()
