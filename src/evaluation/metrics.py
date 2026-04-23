"""
Evaluation metrics for anti-spoofing:
    - Equal Error Rate (EER)
    - minimum tandem Detection Cost Function (min-tDCF)
    - DET curve plotting
    - Score file I/O
"""
import numpy as np
import csv
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.optimize import brentq


# ── EER ───────────────────────────────────────────────────────────────────────

def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """
    Compute EER from score array and binary label array.

    Args:
        scores:  model scores (higher = more bonafide)
        labels:  1 = bonafide, 0 = spoof

    Returns:
        (eer, threshold)  where eer is in [0, 1]
    """
    scores = np.array(scores)
    labels = np.array(labels)

    thresholds = np.unique(scores)
    fars, frrs = [], []

    for thr in thresholds:
        preds    = (scores >= thr).astype(int)
        fp       = np.sum((preds == 1) & (labels == 0))
        fn       = np.sum((preds == 0) & (labels == 1))
        n_spoof  = np.sum(labels == 0)
        n_bona   = np.sum(labels == 1)
        fars.append(fp / n_spoof if n_spoof > 0 else 0.0)
        frrs.append(fn / n_bona  if n_bona  > 0 else 0.0)

    fars = np.array(fars)
    frrs = np.array(frrs)

    # Interpolate for the crossing point
    try:
        eer = brentq(interp1d(thresholds, frrs - fars), thresholds[0], thresholds[-1])
        threshold = eer
        eer_val = float(interp1d(thresholds, frrs)(eer))
    except Exception:
        idx     = np.argmin(np.abs(fars - frrs))
        eer_val = float((fars[idx] + frrs[idx]) / 2)
        threshold = float(thresholds[idx])

    return eer_val, threshold


# ── min-tDCF ──────────────────────────────────────────────────────────────────

def compute_min_tdcf(scores: np.ndarray, labels: np.ndarray,
                     p_spoof: float = 0.05,
                     c_miss: float = 1.0,
                     c_fa: float = 10.0) -> float:
    """
    ASVspoof 2019 min-tDCF (simplified, CM-only variant).
    Default cost parameters follow the ASVspoof 2019 evaluation plan.
    """
    scores = np.array(scores)
    labels = np.array(labels)
    thresholds = np.unique(scores)

    tdcfs = []
    n_bona  = np.sum(labels == 1)
    n_spoof = np.sum(labels == 0)

    for thr in thresholds:
        preds = (scores >= thr).astype(int)
        pmiss = np.sum((preds == 0) & (labels == 1)) / n_bona  if n_bona  > 0 else 0
        pfa   = np.sum((preds == 1) & (labels == 0)) / n_spoof if n_spoof > 0 else 0
        tdcf  = c_miss * pmiss * (1 - p_spoof) + c_fa * pfa * p_spoof
        tdcfs.append(tdcf)

    return float(np.min(tdcfs))


# ── Score file I/O ────────────────────────────────────────────────────────────

def write_score_file(path: str, utt_ids: list, scores: list, labels: list):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["utt_id", "score", "label"])
        for uid, s, l in zip(utt_ids, scores, labels):
            writer.writerow([uid, f"{s:.6f}", int(l)])


def read_score_file(path: str) -> tuple[list, np.ndarray, np.ndarray | None]:
    utt_ids, scores, labels = [], [], []
    with open(path) as f:
        reader = csv.reader(f, delimiter="\t")
        rows = [row for row in reader if row]

    if not rows:
        return utt_ids, np.array(scores), None

    first_row = rows[0]
    has_header = len(first_row) >= 2 and first_row[0].lower() == "utt_id" and first_row[1].lower() == "score"
    start_idx = 1 if has_header else 0

    for row in rows[start_idx:]:
        if len(row) < 2:
            continue
        utt_ids.append(row[0])
        scores.append(float(row[1]))
        if len(row) >= 3 and row[2] != "":
            labels.append(int(row[2]))

    if has_header and len(rows) > 1 and len(labels) != len(scores):
        labels = None
    if not has_header and len(labels) != len(scores):
        labels = None

    return utt_ids, np.array(scores), np.array(labels) if labels is not None else None


# ── DET curve ─────────────────────────────────────────────────────────────────

def plot_det_curves(score_files: dict, output_path: str):
    """
    score_files: {experiment_label: score_file_path}
    Saves a DET curve plot with one line per experiment.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    colors  = plt.cm.tab10(np.linspace(0, 1, len(score_files)))
    plotted = 0

    for (label, path), color in zip(score_files.items(), colors):
        _, scores, labels = read_score_file(path)
        if labels is None or len(labels) == 0:
            print(f"Skipping DET plot for '{label}': missing or incomplete label column in {path}")
            continue

        thresholds = np.unique(scores)
        n_bona  = np.sum(labels == 1)
        n_spoof = np.sum(labels == 0)
        if n_bona == 0 or n_spoof == 0:
            print(f"Skipping DET plot for '{label}': no bonafide/spoof labels found in {path}")
            continue

        fars, frrs = [], []
        for thr in thresholds:
            preds = (scores >= thr).astype(int)
            fars.append(np.sum((preds == 1) & (labels == 0)) / n_spoof)
            frrs.append(np.sum((preds == 0) & (labels == 1)) / n_bona)
        ax.plot(np.array(fars) * 100, np.array(frrs) * 100,
                label=label, color=color, linewidth=1.8)
        plotted += 1

    if plotted == 0:
        print("No valid score files with labels were found. DET curve not generated.")
        return

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("FAR (%)", fontsize=12)
    ax.set_ylabel("FRR (%)", fontsize=12)
    ax.set_title("DET Curves — Isan Anti-Spoofing", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close()
    print(f"DET curve saved → {output_path}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("FAR (%)", fontsize=12)
    ax.set_ylabel("FRR (%)", fontsize=12)
    ax.set_title("DET Curves — Isan Anti-Spoofing", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close()
    print(f"DET curve saved → {output_path}")
