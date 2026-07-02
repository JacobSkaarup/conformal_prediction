import json
import os
import time
import warnings

import optuna
from optuna.exceptions import ExperimentalWarning
from tqdm import tqdm

import src.conformal_prediction as cp
from src.data_loader import load_data
from src.model_tuning import lgb_tuner, xgb_tuner

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ExperimentalWarning, module="optuna.multi_objective")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_SEED = 89

train_end = "2025-08-31"
calibration_end = "2025-10-31"
test_end = "2025-12-31"

zones = ["dk1 down", "dk1 up", "dk2 down", "dk2 up"]
targets = ["volbids", "FRR", "CM"]

dst_path = os.path.join(os.getcwd(), "results", "model_summaries")
os.makedirs(dst_path, exist_ok=True)

job_idx = int(os.environ.get("LSB_JOBINDEX", 1))
model_name = "lgb" if job_idx == 1 else "xgb"

summary = {}

for zone in tqdm(zones, desc="Zone"):
    for target in tqdm(targets, desc="Target", leave=False):
        key = f"{zone} | {target}"

        X_train, X_calibration, X_test, y_train, y_calibration, y_test, X_scaler, y_scaler = cp.train_cal_test_date_split(
            *load_data(zone, target),
            train_end,
            calibration_end,
            test_end,
            scaled=True,
        )

        if model_name == "lgb":
            tuner = lgb_tuner(X_train=X_train, y_train=y_train, val_share=0.2, seed=RANDOM_SEED)
        else:
            tuner = xgb_tuner(X_train=X_train, y_train=y_train, val_share=0.2, seed=RANDOM_SEED)

        results = tuner.tune(n_trials_full=200, n_trials_k=100, progress_bar=False)

        summary[key] = {
            "optimal_features": results["optimal_features"],
            "optimal_params": results["best_params"],
            "holdout_mse": float(results["best_holdout_score"]),
            "best_k_union_input": int(results["best_k_union_input"]),
            "full_holdout_score": float(results["full_holdout_score"]),
            "quantile_alpha": float(results["quantile_alpha"]),
            "objective": results["objective"],
        }

job_file = os.path.join(dst_path, f"boosting_{model_name}_summary_job.json")
with open(job_file, "w") as f:
    json.dump(summary, f, indent=4)

