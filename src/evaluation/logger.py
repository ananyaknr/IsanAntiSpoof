"""
Experiment logger — writes to both MLflow (full tracking) and a
flat results.csv (quick comparison table).

Usage:
    logger = ExperimentLogger(cfg)
    logger.log(run_name="E1_GMM_lfcc", eer=0.042, min_tdcf=0.118,
               feature="lfcc", model="gmm", datasets=["asvspoof2019_la"])
"""
import csv
import mlflow
from pathlib import Path
from datetime import datetime
from omegaconf import DictConfig, OmegaConf


class ExperimentLogger:
    def __init__(self, cfg: DictConfig):
        self.cfg         = cfg
        self.results_csv = Path(cfg.paths.results)
        self.results_csv.parent.mkdir(parents=True, exist_ok=True)

        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name)

        # Write CSV header if file is new
        if not self.results_csv.exists():
            with open(self.results_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "experiment", "model", "feature",
                    "train_datasets", "eval_datasets",
                    "eer", "min_tdcf", "run_id", "notes"
                ])

    def log(self, run_name: str, eer: float, min_tdcf: float,
            feature: str = None, model: str = None,
            train_datasets: list = None, eval_datasets: list = None,
            notes: str = "", extra_params: dict = None):

        cfg_dict    = OmegaConf.to_container(self.cfg, resolve=True)
        feature     = feature     or self.cfg.feature.type
        model       = model       or self.cfg.model.name
        train_dsets = train_datasets or list(self.cfg.experiment.dataset.train)
        eval_dsets  = eval_datasets  or list(self.cfg.experiment.dataset.eval)

        # ── MLflow ────────────────────────────────────────────────────────────
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params({
                "experiment":     self.cfg.experiment.name,
                "feature":        feature,
                "model":          model,
                "train_datasets": str(train_dsets),
                "eval_datasets":  str(eval_dsets),
                "seed":           self.cfg.get("seed", 42),
                **(extra_params or {}),
            })
            mlflow.log_metrics({"eer": eer, "min_tdcf": min_tdcf})
            mlflow.log_dict(cfg_dict, "config.yaml")

            run_id = run.info.run_id
            print(f"[MLflow] Run '{run_name}' — EER={eer:.4f}  "
                  f"min-tDCF={min_tdcf:.4f}  id={run_id[:8]}")

        # ── CSV ───────────────────────────────────────────────────────────────
        with open(self.results_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                run_name, model, feature,
                "|".join(train_dsets), "|".join(eval_dsets),
                f"{eer:.4f}", f"{min_tdcf:.4f}", run_id[:8], notes,
            ])

        return run_id
