# Isan Anti-Spoofing

Synthetic speech detection for the Isan dialect (Northeast Thai).
First countermeasure system targeting a low-resource Thai regional language.

---

## Project structure

```
isan_antispoof/
├── configs/                    ← Hydra config tree
│   ├── config.yaml             ← root config (defaults + paths + mlflow)
│   ├── experiment/             ← E1–E5 experiment definitions
│   ├── feature/                ← lfcc / mfcc / cqcc parameters
│   ├── model/                  ← gmm / lcnn / resnet hyperparameters
│   └── dataset/                ← dataset paths, split ratios, sources
│
├── src/
│   ├── data/
│   │   ├── preprocess.py       ← load → resample → normalize → pad/trim
│   │   ├── build_protocol.py   ← scan raw/ → generate protocol.txt
│   │   └── dataset.py          ← AntiSpoofDataset + AntiSpoofDataModule
│   ├── features/
│   │   ├── extractor.py        ← build_extractor(cfg) → LFCC/MFCC/CQCC
│   │   └── extract_all.py      ← batch .npy extraction
│   ├── models/
│   │   ├── gmm_model.py        ← sklearn GMM baseline
│   │   ├── lcnn.py             ← LCNN + MFM (Lightning)
│   │   └── resnet.py           ← ResNet (Lightning)
│   ├── training/
│   │   └── train.py            ← unified entry point (Hydra)
│   └── evaluation/
│       ├── metrics.py          ← EER, min-tDCF, DET curve, score I/O
│       └── logger.py           ← MLflow + CSV dual logger
│
├── scripts/
│   ├── run_all_experiments.sh  ← run E1–E5 in sequence
│   └── compare_experiments.py ← print results table + DET plot
│
├── experiments/
│   ├── results.csv             ← one row per experiment run (auto-updated)
│   ├── scores/                 ← per-utterance score .tsv files
│   └── plots/                  ← DET curve PNGs
│
├── checkpoints/                ← saved model weights per experiment
├── protocols/                  ← protocol.txt (generated)
├── data/
│   ├── raw/                    ← drop Person A's datasets here
│   └── processed/              ← .npy features (generated)
├── notebooks/                  ← analysis notebooks
├── mlruns/                     ← MLflow run database
└── requirements.txt
```

---

## Setup

```bash
conda create -n isan_spoof python=3.10
conda activate isan_spoof
pip install -r requirements.txt
```

---

## Step-by-step workflow

### 1. Build protocol
```bash
python src/data/build_protocol.py
```

### 2. Extract features
```bash
# All three feature types
python src/features/extract_all.py feature=lfcc
python src/features/extract_all.py feature=mfcc
python src/features/extract_all.py feature=cqcc
```

### 3. Run experiments

```bash
# Single experiment
python src/training/train.py experiment=e1_baseline

# Override any param on the fly
python src/training/train.py experiment=e3_isan_aware model=lcnn feature=mfcc

# Run all E1–E5
bash scripts/run_all_experiments.sh
```

### 4. Compare results

```bash
# Print sorted results table + generate DET curve
python scripts/compare_experiments.py

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
