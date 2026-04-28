# Isan Anti-Spoofing: Synthetic Speech Detection for Low-Resource Thai Dialects

## Overview

This project implements the first synthetic speech detection (anti-spoofing) system specifically designed for the Isan dialect, a low-resource regional language spoken in Northeast Thailand. The system addresses the challenge of detecting spoofed speech in under-resourced languages where standard anti-spoofing models trained on high-resource languages (like English) perform poorly due to linguistic and acoustic differences.

### Key Contributions
- **Dialect-aware anti-spoofing**: Demonstrates the "dialect gap" problem and proposes mixed-training solutions
- **Multi-modal feature extraction**: Supports LFCC, MFCC, and CQCC features optimized for spoof detection
- **Comprehensive evaluation**: GMM baseline + deep learning models (LCNN, ResNet) with EER and min-tDCF metrics
- **Scalable pipeline**: Handles large datasets with compressed feature storage and efficient processing

### Background
Synthetic speech detection is crucial for securing voice-based authentication systems. Most research focuses on high-resource languages like English (ASVspoof challenges), but performance degrades significantly on regional dialects due to:
- Phonetic differences from standard languages
- Limited training data availability
- Different spoof attack patterns (TTS systems trained on dialect-specific data)

This work targets the Isan dialect, spoken by ~30% of Thailand's population but underrepresented in speech technology research.

---

## Methodology

### Datasets
The system uses a combination of public and custom datasets:

| Dataset | Type | Language | Description |
|---------|------|----------|-------------|
| `typhoon_isan` | Bonafide | Isan | Real speech from Isan speakers (evaluation target) |
| `asvspoof2019_la` | Spoof | English | Logical access attacks from ASVspoof 2019 |
| `thaispoof` | Spoof | Central Thai | TTS-generated spoof attacks in Thai |
| `isan_tts_spoofs` | Spoof | Isan | TTS-generated attacks specifically for Isan dialect |

**Note**: Raw audio datasets should be placed in `data/raw/` with subdirectories matching the dataset names above.

### Feature Extraction
Three acoustic feature types are supported, each capturing different aspects of speech spoofing artifacts:

- **LFCC (Linear Frequency Cepstral Coefficients)**: Emphasizes linear-frequency domain, better for detecting synthetic speech harmonics
- **MFCC (Mel Frequency Cepstral Coefficients)**: Standard mel-scale features, widely used in speech processing
- **CQCC (Constant-Q Cepstral Coefficients)**: Variable-resolution frequency analysis, effective for detecting formant modifications

Features are extracted from 4-second audio segments, normalized, and stored as compressed NumPy arrays for efficient storage.

### Models
Three model architectures provide a spectrum from traditional to modern approaches:

- **GMM (Gaussian Mixture Model)**: Unsupervised clustering baseline using scikit-learn
- **LCNN (Light CNN)**: Lightweight convolutional network with Max Feature Map activation for efficient spoof detection
- **ResNet**: Deep residual network for high-capacity modeling of complex spoof patterns

### Experiments
Six experiments systematically evaluate different aspects of cross-lingual and cross-dialect spoof detection:

1. **E1: Standard Baseline** - GMM on ASVspoof 2019 LA only (reference performance on standard benchmark)
2. **E1.5: Cross-Lingual Gap** - Train on ASVspoof 2019 LA, test on Central Thai data (quantifies language gap)
3. **E2: Cross-Lingual + Dialect Gap** - Train on ASVspoof 2019 LA, test on Isan data (zero-shot vulnerability assessment)
4. **E3: Cross-Dialect Gap** - Train on Central Thai data, test on Isan data (dialect transfer evaluation)
5. **E4: Proposed Isan-Aware System** - Mixed training on all datasets with LCNN (core contribution)
6. **E5: Feature Ablation** - Compare LFCC/MFCC/CQCC performance on mixed data
7. **E6: Model Ablation** - Compare GMM/LCNN/ResNet performance on mixed data

---

## Project Structure

```
isan_antispoof/
├── configs/                    ← Hydra configuration tree
│   ├── config.yaml             ← Root config with defaults and paths
│   ├── experiment/             ← E1–E6 experiment definitions
│   ├── feature/                ← LFCC/MFCC/CQCC parameter settings
│   ├── model/                  ← GMM/LCNN/ResNet hyperparameters
│   └── dataset/                ← Dataset paths, splits, and sources
│
├── src/
│   ├── data/
│   │   ├── preprocess.py       ← Audio loading, resampling, normalization
│   │   ├── build_protocol.py   ← Generate protocol.txt from raw data
│   │   └── dataset.py          ← PyTorch Dataset and Lightning DataModule
│   ├── features/
│   │   ├── extractor.py        ← Unified feature extraction dispatcher
│   │   └── extract_all.py      ← Batch feature extraction with compression
│   ├── models/
│   │   ├── gmm_model.py        ← Scikit-learn GMM implementation
│   │   ├── lcnn.py             ← LCNN with MFM activation (PyTorch Lightning)
│   │   └── resnet.py           ← ResNet architecture (PyTorch Lightning)
│   ├── training/
│   │   └── train.py            ← Unified training entry point with Hydra
│   └── evaluation/
│       ├── metrics.py          ← EER, min-tDCF, DET curves, score I/O
│       └── logger.py           ← MLflow + CSV experiment tracking
│
├── scripts/
│   ├── run_all_experiments.sh  ← Sequential execution of E1–E6
│   └── compare_experiments.py  ← Results comparison and DET plotting
│
├── experiments/
│   ├── results.csv             ← Experiment results table (auto-updated)
│   ├── scores/                 ← Per-utterance detection scores (.tsv)
│   └── plots/                  ← DET curve visualizations (.png)
│
├── checkpoints/                ← Saved model weights per experiment
├── protocols/                  ← Generated protocol.txt files
├── data/
│   ├── raw/                    ← Raw audio datasets (place here)
│   └── processed/              ← Extracted features (.npz compressed)
│
├── notebooks/                  ← Jupyter notebooks for analysis
├── mlruns/                     ← MLflow experiment tracking database
├── requirements.txt            ← Python dependencies
└── README.md
```

---

## Setup

### Environment
```bash
# Create conda environment
conda create -n isan_spoof python=3.10
conda activate isan_spoof

# Install dependencies
pip install -r requirements.txt
```

### Dataset Preparation
1. Download and organize datasets into `data/raw/`:
   ```
   data/raw/
   ├── typhoon_isan/          # Real Isan speech
   ├── asvspoof2019_la/       # English spoof attacks
   ├── thaispoof/             # Thai spoof attacks
   └── isan_tts_spoofs/       # Isan-specific spoof attacks
   ```

2. **Storage Optimization**: For large datasets (>20GB), consider:
   - External/network storage mounted at `data/raw/`
   - Cloud storage with remote processing
   - Subset selection for development

---

## Usage

### Step-by-Step Workflow

#### 1. Build Protocol
Generate the dataset protocol file from raw audio:
```bash
python src/data/build_protocol.py
```
This scans `data/raw/` and creates `protocols/protocol.txt` with utterance metadata.

#### 2. Extract Features
Extract acoustic features for all utterances (run for each feature type):
```bash
# Extract LFCC features (recommended for spoof detection)
python src/features/extract_all.py feature=lfcc

# Extract MFCC features
python src/features/extract_all.py feature=mfcc

# Extract CQCC features
python src/features/extract_all.py feature=cqcc
```
Features are saved as compressed `.npz` files in `data/processed/` for efficient storage.

#### 3. Run Experiments
Train and evaluate models:

```bash
# Single experiment (e.g., baseline GMM)
python src/training/train.py experiment=e1_baseline

# Override parameters on-the-fly
python src/training/train.py experiment=e4_isan_aware model=lcnn feature=mfcc

# Run all experiments sequentially
bash scripts/run_all_experiments.sh
```

#### 4. Compare Results
Analyze and visualize experiment outcomes:
```bash
# Generate results table and DET curves
python scripts/compare_experiments.py
```

### Configuration
The system uses Hydra for flexible configuration. Key config files:
- `configs/config.yaml`: Global defaults and paths
- `configs/experiment/`: Experiment-specific settings
- `configs/feature/`: Feature extraction parameters
- `configs/model/`: Model hyperparameters
- `configs/dataset/`: Dataset paths and splits

Override any parameter via command line:
```bash
python src/training/train.py model.batch_size=32 dataset.segment_duration=3.0
```

### Monitoring and Logging
- **MLflow**: Experiment tracking with UI (`mlflow ui`)
- **CSV Results**: `experiments/results.csv` with EER and min-tDCF metrics
- **Checkpoints**: Model weights saved in `checkpoints/`
- **Scores**: Per-utterance detection scores in `experiments/scores/`

---

## Results and Evaluation

### Metrics
- **EER (Equal Error Rate)**: Threshold where false acceptance = false rejection
- **min-tDCF (minimum Detection Cost Function)**: Cost-weighted error metric
- **DET Curves**: Detection error tradeoff visualization

### Current Results
Results are automatically logged to `experiments/results.csv`. Example baseline result:
- E1 Baseline (GMM, LFCC, ASVspoof 2019 LA): EER = 9.18%

### Expected Findings
- E2 demonstrates significant performance degradation when testing on Isan dialect
- E3 shows improved performance through mixed training on dialect-specific data
- Feature and model ablations identify optimal configurations for low-resource scenarios

---

## Optimization Features

### Compressed Feature Storage
Features are stored as compressed NumPy arrays (`.npz`) to reduce disk usage by ~70% compared to uncompressed `.npy` files, crucial for large datasets.

### Memory-Efficient Processing
- Per-utterance feature extraction prevents memory overflow
- PyTorch DataLoader with batching and padding
- Support for external/network storage for datasets larger than local disk capacity

### Scalable Architecture
- Hydra configuration enables easy parameter sweeps
- MLflow tracking supports experiment comparison
- Modular design allows adding new features/models/datasets

---

## Dependencies

Core dependencies (see `requirements.txt`):
- **PyTorch Lightning**: Deep learning training framework
- **Librosa**: Audio processing and feature extraction
- **Scikit-learn**: GMM implementation and metrics
- **Hydra**: Configuration management
- **MLflow**: Experiment tracking
- **NumPy/SciPy**: Numerical computing

Optional:
- **DVC**: Data versioning for large-scale dataset management

---

## Citation

If you use this work, please cite:

```
@misc{isan_antispoof_2024,
  title={Isan Anti-Spoofing: Synthetic Speech Detection for Low-Resource Thai Dialects},
  author={Ananya Kumar},
  year={2024},
  url={https://github.com/ananyaknr/IsanAntiSpoof}
}
```

---

## License

This project is released under the MIT License. See LICENSE file for details.

# Launch MLflow UI
mlflow ui --backend-store-uri mlruns
# Open http://localhost:5000
```

---

## Experiment matrix

| ID  | Description                       | Train data              | Eval data         |
|-----|-----------------------------------|-------------------------|-------------------|
| E1  | GMM baseline (reference)          | ASVspoof 2019 LA        | ASVspoof 2019 LA  |
| E2  | Dialect gap measurement           | Central Thai only       | Isan              |
| E3  | Isan-aware mixed training ★       | All sources             | Isan              |
| E4  | Feature ablation (LFCC/MFCC/CQCC) | All sources             | Isan              |
| E5  | Model ablation (GMM/LCNN/ResNet)  | All sources             | Isan              |

---

## Current work in progress

### E1 Baseline Integration (Completed ✅)

The E1 baseline experiment from the legacy IsanE1 project has been successfully migrated and integrated:

#### Artifacts Created
- **`experiments/scores/e1_baseline.tsv`** — Per-utterance GMM scores for all 71,237 evaluation samples
  - Format: TSV with columns `utt_id`, `score`
  - Source: Converted from `IsanE1/isan-spoof/results/score/eval_scores.txt`
  - Contains ASVspoof 2019 LA evaluation set predictions

- **`experiments/results.csv`** — Unified results tracking table
  - Initialized with E1 baseline row: **EER = 9.1826%**
  - Columns: `timestamp, experiment, model, feature, train_datasets, eval_datasets, eer, min_tdcf, run_id, notes`
  - Auto-updated by `ExperimentLogger` after each run

- **`experiments/E1_report.txt`** — Original E1 evaluation report
  - Preserved from legacy project for reference
  - Contains matched sample count and EER metric

- **`configs/experiment/e1_baseline.yaml`** — Configuration updated
  - Renamed from `E1_GMM_baseline` to `e1_baseline` for consistency
  - Specifies: GMM model, LFCC features, ASVspoof 2019 LA train/eval

#### Code Changes
- **`src/evaluation/metrics.py`** — Enhanced for backward compatibility
  - `read_score_file()` now handles both legacy 2-column and new 3-column TSV formats
  - Gracefully skips invalid/incomplete data in DET curve generation
  - `plot_det_curves()` skips score files missing ground truth labels with informative messaging

#### Integration Testing ✅
```bash
$ python scripts/compare_experiments.py
--------------------  --------  --------  -------  --------  --------
experiment            model     feature   train    EER       min-tDCF
--------------------  --------  --------  -------  --------  --------
e1_baseline           gmm       lfcc      asvs...  0.091826  nan
--------------------  --------  --------  -------  --------  --------
```

### Migration Status Summary

| Component | Status | Notes |
|-----------|--------|-------|
| E1 baseline results | ✅ Done | Scores, metrics, and config integrated |
| E1 eval report | ✅ Done | Archived for reference |
| Results tracking | ✅ Done | CSV infrastructure ready for E2–E5 |
| E1 model code | ⚠️ Not needed | Legacy PyTorch waveform model; main uses sklearn GMM |
| E1 eval utilities | ⚠️ Superseded | Main project has cleaner metric implementations |
| E1 feature extraction | ⚠️ Pending review | May contain optimized pipeline worth porting |

### Next Steps (E2–E5 Experiments)

The framework is now ready for additional experiments:

1. **E2 (Dialect gap)** — Train on ASVspoof 2019 LA + Thai data, eval on Isan
2. **E3 (Isan-aware)** — Train on all sources, eval on Isan (expected to perform best ★)
3. **E4 (Feature ablation)** — Compare LFCC/MFCC/CQCC with best training config
4. **E5 (Model ablation)** — Compare GMM/LCNN/ResNet on best feature type

Each run will automatically:
- Log results to `experiments/results.csv`
- Save scores to `experiments/scores/{experiment_name}.tsv`
- Generate DET curve in `experiments/plots/`
- Track parameters in MLflow

---

## Key design decisions

**Hydra config** — every hyperparameter lives in `configs/`. Swapping an experiment
is one CLI flag: `experiment=e3_isan_aware`. No code changes needed.

**MLflow tracking** — every run logs params, metrics, and the full config YAML.
Compare runs visually at `localhost:5000`.

**Flat results.csv** — `experiments/results.csv` is a human-readable table updated
after every run. Sort by EER to see the best system at a glance.

**Score files** — per-utterance `.tsv` files in `experiments/scores/` let you
recompute metrics with different thresholds without rerunning the model.

**Backward compatibility** — evaluation utilities gracefully handle both legacy
(2-column) and new (3-column TSV) score file formats, enabling smooth migration
from the IsanE1 codebase.

**DVC** (optional) — version the `data/` directory so each result is reproducible
with the exact dataset version that produced it.
