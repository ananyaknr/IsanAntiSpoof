"""
Batch feature extraction — reads protocol.txt, extracts all feature types.

Usage:
    python src/features/extract_all.py
    python src/features/extract_all.py feature.type=mfcc
"""
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig

from src.data.preprocess import preprocess_file
from src.features.extractor import build_extractor


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def extract_all(cfg: DictConfig) -> None:
    extractor   = build_extractor(cfg.feature)
    output_root = Path(cfg.dataset.processed_root)
    failed      = []

    with open(cfg.dataset.protocol_path) as f:
        lines = [l.strip().split("\t") for l in f
                 if l.strip() and not l.startswith("#")]

    total = len(lines)
    print(f"Extracting {cfg.feature.type} for {total} utterances...")

    for i, parts in enumerate(lines):
        if len(parts) < 6:
            continue
        utt_id, dataset, _, label, split, wav_path = parts

        out_dir  = output_root / dataset / "features" / cfg.feature.type / split
        out_path = out_dir / f"{utt_id}.npz"
        if out_path.exists():
            continue

        try:
            audio = preprocess_file(
                wav_path,
                target_sr        = cfg.dataset.target_sr,
                segment_duration = cfg.dataset.segment_duration,
            )
            feat = extractor(audio)
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_path, feat=feat)
        except Exception as e:
            failed.append((utt_id, str(e)))

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{total}")

    print(f"\nDone. Failed: {len(failed)}")
    if failed:
        fail_log = output_root / "failed.txt"
        with open(fail_log, "w") as f:
            for uid, err in failed:
                f.write(f"{uid}\t{err}\n")


if __name__ == "__main__":
    extract_all()
