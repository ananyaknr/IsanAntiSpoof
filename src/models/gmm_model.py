"""
GMM baseline countermeasure.
Trains two GMMs (bonafide, spoof) and scores as log-likelihood ratio.
"""
import numpy as np
import joblib
from pathlib import Path
from sklearn.mixture import GaussianMixture
from omegaconf import DictConfig


class GMMModel:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.gmm_bonafide = GaussianMixture(
            n_components    = cfg.model.n_components,
            covariance_type = cfg.model.covariance_type,
            max_iter        = cfg.model.max_iter,
            n_init          = cfg.model.n_init,
        )
        self.gmm_spoof = GaussianMixture(
            n_components    = cfg.model.n_components,
            covariance_type = cfg.model.covariance_type,
            max_iter        = cfg.model.max_iter,
            n_init          = cfg.model.n_init,
        )

    def fit(self, feats_bonafide: np.ndarray, feats_spoof: np.ndarray):
        """feats: (N_frames_total, n_feats) — concatenate all utterances."""
        print("Fitting bonafide GMM...")
        self.gmm_bonafide.fit(feats_bonafide)
        print("Fitting spoof GMM...")
        self.gmm_spoof.fit(feats_spoof)

    def score_utterance(self, feat: np.ndarray) -> float:
        """
        feat: (T, n_feats) — single utterance.
        Returns log-likelihood ratio score (higher = more bonafide).
        """
        ll_bonafide = self.gmm_bonafide.score(feat)
        ll_spoof    = self.gmm_spoof.score(feat)
        return float(ll_bonafide - ll_spoof)

    def save(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.gmm_bonafide, path / "gmm_bonafide.pkl")
        joblib.dump(self.gmm_spoof,    path / "gmm_spoof.pkl")

    def load(self, path: str):
        path = Path(path)
        self.gmm_bonafide = joblib.load(path / "gmm_bonafide.pkl")
        self.gmm_spoof    = joblib.load(path / "gmm_spoof.pkl")
