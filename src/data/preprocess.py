"""
Audio preprocessing: load → resample → normalize → pad/trim.
Called by both build_protocol.py and extract_features.py.
"""
import librosa
import numpy as np


def preprocess_file(filepath: str, target_sr: int = 16000,
                    segment_duration: float = 4.0) -> np.ndarray:
    segment_samples = int(target_sr * segment_duration)
    audio, _ = librosa.load(filepath, sr=target_sr, mono=True)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    return _pad_or_trim(audio, segment_samples)


def _pad_or_trim(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) >= length:
        return audio[:length]
    repeats = (length // len(audio)) + 1
    return np.tile(audio, repeats)[:length]
