#!/usr/bin/env bash
# Run all experiments E1–E6 sequentially.
# Override any param with Hydra syntax, e.g.:
#   bash scripts/run_all_experiments.sh model.batch_size=64

set -e
EXTRA="${@}"

echo "=== E1: Standard Baseline ==="
python src/training/train.py experiment=e1_baseline model=gmm feature=lfcc $EXTRA

echo "=== E1.5: Cross-Lingual Gap ==="
python src/training/train.py experiment=e1_5_cross_lingual_gap model=gmm feature=lfcc $EXTRA

echo "=== E2: Cross-Lingual + Dialect Gap ==="
python src/training/train.py experiment=e2_dialect_gap model=gmm feature=lfcc $EXTRA

echo "=== E3: Cross-Dialect Gap ==="
python src/training/train.py experiment=e3_cross_dialect_gap model=lcnn feature=lfcc $EXTRA

echo "=== E4: Proposed Isan-Aware System ==="
python src/training/train.py experiment=e4_isan_aware model=lcnn feature=lfcc $EXTRA

echo "=== E5: Feature Ablation ==="
for FEAT in lfcc mfcc cqcc; do
  echo "--- E5 feature=$FEAT ---"
  python src/training/train.py experiment=e5_feature_ablation model=lcnn feature=$FEAT $EXTRA
done

echo "=== E6: Model Ablation ==="
for MODEL in gmm lcnn resnet; do
  echo "--- E6 model=$MODEL ---"
  python src/training/train.py experiment=e6_model_ablation model=$MODEL feature=lfcc $EXTRA
done

echo "All experiments complete. Results → experiments/results.csv"
echo "Launch MLflow UI: mlflow ui --backend-store-uri mlruns"
