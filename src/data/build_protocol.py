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

        wav_files = sorted(folder.glob("**/*.wav"))
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
