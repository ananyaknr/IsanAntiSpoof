"""
Scan raw data directories and generate protocol.txt.

Usage:
    python src/data/build_protocol.py
    python src/data/build_protocol.py dataset.raw_root=data/raw

Protocol columns (tab-separated):
    utt_id  dataset  speaker_id  label  split  wav_path
"""
import random
from pathlib import Path
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def build_protocol(cfg: DictConfig) -> None:
    random.seed(cfg.dataset.random_seed)
    raw_root    = Path(cfg.dataset.raw_root)
    output_path = Path(cfg.dataset.protocol_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ratios = cfg.dataset.split_ratios
    entries = []

    for dataset_name, source_cfg in cfg.dataset.sources.items():
        folder = raw_root / dataset_name
        if not folder.exists():
            print(f"  [skip] {dataset_name} — not found at {folder}")
            continue

        # Special handling for asvspoof2019_la with protocol files
        if dataset_name == "asvspoof2019_la":
            protocols_dir = folder / "protocols"
            if protocols_dir.exists():
                # For E1 baseline, use E1_train_val.txt for train/dev, E1_test.txt for eval
                train_val_file = protocols_dir / "E1_train_val.txt"
                test_file = protocols_dir / "E1_test.txt"
                if train_val_file.exists() and test_file.exists():
                    # Read train_val and split into train and dev
                    train_val_entries = []
                    with open(train_val_file) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                speaker_id, file_id, _, _, label = parts
                                utt_id = f"{source_cfg.prefix}_{speaker_id}_{file_id}"
                                wav_path = folder / "flac" / f"{file_id}.flac"
                                train_val_entries.append((utt_id, speaker_id, label, str(wav_path)))
                    
                    random.shuffle(train_val_entries)
                    n_train_val = len(train_val_entries)
                    n_train = int(n_train_val * ratios.train / (ratios.train + ratios.dev))
                    n_dev = n_train_val - n_train
                    
                    for i, (utt_id, speaker_id, label, wav_path) in enumerate(train_val_entries):
                        split = "train" if i < n_train else "dev"
                        entries.append("\t".join([utt_id, dataset_name, speaker_id, label, split, wav_path]))
                    
                    # Read test
                    with open(test_file) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                speaker_id, file_id, _, _, label = parts
                                utt_id = f"{source_cfg.prefix}_{speaker_id}_{file_id}"
                                wav_path = folder / "flac" / f"{file_id}.flac"
                                entries.append("\t".join([utt_id, dataset_name, speaker_id, label, "eval", str(wav_path)]))
                    
                    print(f"  [ok] {dataset_name}: protocol-based")
                    continue
        
        # Default: scan for wav files
        wav_files = sorted(folder.glob("**/*.wav"))
        if not wav_files:
            wav_files = sorted(folder.glob("**/*.flac"))  # Try flac if no wav
        random.shuffle(wav_files)
        n       = len(wav_files)
        n_train = int(n * ratios.train)
        n_dev   = int(n * ratios.dev)
        n_eval  = n - n_train - n_dev

        for split_name, subset in [
            ("train", wav_files[:n_train]),
            ("dev",   wav_files[n_train: n_train + n_dev]),
            ("eval",  wav_files[n_train + n_dev:]),
        ]:
            for f in subset:
                utt_id = f"{source_cfg.prefix}_{f.stem}"
                entries.append("\t".join([
                    utt_id, dataset_name, f.stem,
                    source_cfg.label, split_name, str(f)
                ]))

        print(f"  [ok] {dataset_name}: {n} files")

    with open(output_path, "w") as out:
        out.write("\n".join(entries) + "\n")

    print(f"\nProtocol → {output_path}  ({len(entries)} entries)")


if __name__ == "__main__":
    build_protocol()
