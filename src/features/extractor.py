"""
Unified feature extractor — dispatches to LFCC / MFCC / CQCC
based on the Hydra feature config.

Usage:
    extractor = build_extractor(cfg.feature)
    feat = extractor(audio)   # → np.ndarray (T, n_coeffs*3)
"""
import numpy as np
import librosa
from scipy.fftpack import dct
from omegaconf import DictConfig


def build_extractor(feature_cfg: DictConfig):
    dispatch = {"lfcc": _extract_lfcc, "mfcc": _extract_mfcc, "cqcc": _extract_cqcc}
    if feature_cfg.type not in dispatch:
        raise ValueError(f"Unknown feature type: {feature_cfg.type}")
    def extractor(audio: np.ndarray) -> np.ndarray:
        return dispatch[feature_cfg.type](audio, feature_cfg)
    return extractor


# ── LFCC ──────────────────────────────────────────────────────────────────────

def _linear_filterbank(sr, n_fft, n_filter, fmin, fmax):
    freqs     = np.linspace(fmin, fmax, n_filter + 2)
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    fb = np.zeros((n_filter, len(fft_freqs)))
    for i in range(n_filter):
        low, center, high = freqs[i], freqs[i + 1], freqs[i + 2]
        fb[i] = np.maximum(0, np.minimum(
            (fft_freqs - low)   / (center - low   + 1e-8),
            (high - fft_freqs) / (high   - center + 1e-8),
        ))
    return fb


def _extract_lfcc(audio: np.ndarray, cfg: DictConfig) -> np.ndarray:
    stft = np.abs(librosa.stft(audio, n_fft=cfg.n_fft,
                               hop_length=cfg.hop_length,
                               win_length=cfg.win_length, window="hann"))
    fb   = _linear_filterbank(cfg.sr, cfg.n_fft, cfg.n_filter, cfg.fmin, cfg.fmax)
    base = dct(np.log(np.dot(fb, stft) + 1e-8),
               type=2, axis=0, norm="ortho")[:cfg.n_coeffs]
    return _with_deltas(base, cfg).T


# ── MFCC ──────────────────────────────────────────────────────────────────────

def _extract_mfcc(audio: np.ndarray, cfg: DictConfig) -> np.ndarray:
    base = librosa.feature.mfcc(y=audio, sr=cfg.sr, n_mfcc=cfg.n_coeffs,
                                 n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                                 win_length=cfg.win_length, n_mels=cfg.n_mels)
    return _with_deltas(base, cfg).T


# ── CQCC ──────────────────────────────────────────────────────────────────────

def _extract_cqcc(audio: np.ndarray, cfg: DictConfig) -> np.ndarray:
    cqt  = np.abs(librosa.cqt(audio, sr=cfg.sr, fmin=cfg.fmin,
                               n_bins=cfg.n_bins,
                               bins_per_octave=cfg.bins_per_octave,
                               hop_length=cfg.hop_length))
    base = dct(np.log(cqt + 1e-8), type=2, axis=0, norm="ortho")[:cfg.n_coeffs]
    return _with_deltas(base, cfg).T


# ── Shared helper ─────────────────────────────────────────────────────────────

def _with_deltas(base: np.ndarray, cfg: DictConfig) -> np.ndarray:
    parts = [base]
    if cfg.get("with_delta", True):
        parts.append(librosa.feature.delta(base, order=1))
    if cfg.get("with_delta_delta", True):
        parts.append(librosa.feature.delta(base, order=2))
    return np.concatenate(parts, axis=0)  # (n_coeffs * 3, T)
