"""
Print a sorted results table and generate a DET curve comparison plot.

Usage:
    python scripts/compare_experiments.py
    python scripts/compare_experiments.py --results experiments/results.csv
"""
import argparse
import csv
from pathlib import Path
from src.evaluation.metrics import plot_det_curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="experiments/results.csv")
    parser.add_argument("--scores",  default="experiments/scores")
    parser.add_argument("--plot",    default="experiments/plots/det_comparison.png")
    args = parser.parse_args()

    # ── Print results table ───────────────────────────────────────────────────
    rows = []
    with open(args.results) as f:
        rows = list(csv.DictReader(f))

    rows.sort(key=lambda r: float(r["eer"]))

    col_w = [20, 8, 8, 12, 8, 8]
    header = ["experiment", "model", "feature", "train_data", "EER", "min-tDCF"]
    sep    = "  ".join("-" * w for w in col_w)

    print("\n" + sep)
    print("  ".join(h.ljust(w) for h, w in zip(header, col_w)))
    print(sep)
    for r in rows:
        print("  ".join([
            r["experiment"][:col_w[0]].ljust(col_w[0]),
            r["model"].ljust(col_w[1]),
            r["feature"].ljust(col_w[2]),
            r["train_datasets"][:col_w[3]].ljust(col_w[3]),
            r["eer"].ljust(col_w[4]),
            r["min_tdcf"].ljust(col_w[5]),
        ]))
    print(sep + "\n")

    # ── DET curves ───────────────────────────────────────────────────────────
    score_dir = Path(args.scores)
    score_files = {
        p.stem: str(p) for p in sorted(score_dir.glob("*.tsv"))
    }
    if score_files:
        plot_det_curves(score_files, args.plot)
    else:
        print("No score files found yet. Run experiments first.")


if __name__ == "__main__":
    main()
