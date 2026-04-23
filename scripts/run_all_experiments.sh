#!/usr/bin/env bash
# Run all 5 experiments sequentially.
# Override any param with Hydra syntax, e.g.:
#   bash scripts/run_all_experiments.sh model.batch_size=64

set -e
EXTRA="${@}"

echo "=== E1: GMM baseline ==="
python src/training/train.py experiment=e1_baseline model=gmm feature=lfcc $EXTRA

echo "=== E2: Dialect gap ==="
python src/training/train.py experiment=e2_dialect_gap model=gmm feature=lfcc $EXTRA

echo "=== E3: Isan-aware (core) ==="
python src/training/train.py experiment=e3_isan_aware model=lcnn feature=lfcc $EXTRA

echo "=== E4: Feature ablation ==="
for FEAT in lfcc mfcc cqcc; do
  echo "--- E4 feature=$FEAT ---"
  python src/training/train.py experiment=e4_feature_ablation model=lcnn feature=$FEAT $EXTRA
done

echo "=== E5: Model ablation ==="
for MODEL in gmm lcnn resnet; do
  echo "--- E5 model=$MODEL ---"
  python src/training/train.py experiment=e5_model_ablation model=$MODEL feature=lfcc $EXTRA
done

echo "All experiments complete. Results → experiments/results.csv"
echo "Launch MLflow UI: mlflow ui --backend-store-uri mlruns"
