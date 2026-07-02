import json
import os
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

import src.conformal_prediction as cp
from src.data_loader import load_data
from src.model_tuning import TimeSeriesLassoTuner, TimeSeriesRidgeTuner

warnings.filterwarnings("ignore", category=ConvergenceWarning)

train_end = "2025-08-31"
calibration_end = "2025-10-31"
test_end = "2025-12-31"

dst_folder = os.path.join(os.getcwd(), "results", "model_summaries")
os.makedirs(dst_folder, exist_ok=True)

zones = ["dk1 down", "dk1 up", "dk2 down", "dk2 up"]
targets = ["volbids", "FRR", "CM"]

job_idx = int(os.environ.get("LSB_JOBINDEX", 1))
job_name = {1: "lasso", 2: "ridge_forward", 3: "ridge_backward"}.get(job_idx, "ridge_backward")

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

        if job_name == "lasso":
            tuner = TimeSeriesLassoTuner(X=X_train, y=y_train, alpha_grid=np.logspace(-4, 4, 50), val_share=0.2)
            optimal_features, optimal_alpha, holdout_mse, coefficients = tuner.fit_and_tune()
            summary[key] = {
                "optimal_features": optimal_features,
                "optimal_params": {"alpha": float(optimal_alpha) if optimal_alpha is not None else None},
                "holdout_mse": float(holdout_mse),
                "coefficients": coefficients,
            }

        elif job_name == "ridge_forward":
            tuner = TimeSeriesRidgeTuner(X=X_train, y=y_train, alpha_grid=np.logspace(-6, 5, 50), val_share=0.2)
            optimal_features, optimal_alpha, holdout_mse = tuner.forward_selection(tolerance=0.02)
            summary[key] = {
                "optimal_features": optimal_features,
                "optimal_params": {"alpha": float(optimal_alpha) if optimal_alpha is not None else None},
                "holdout_mse": float(holdout_mse),
            }

        else:
            tuner = TimeSeriesRidgeTuner(X=X_train, y=y_train, alpha_grid=np.logspace(-6, 5, 50), val_share=0.2)
            optimal_features, optimal_alpha, holdout_mse = tuner.backward_selection(tolerance=0.02)
            summary[key] = {
                "optimal_features": optimal_features,
                "optimal_params": {"alpha": float(optimal_alpha) if optimal_alpha is not None else None},
                "holdout_mse": float(holdout_mse),
            }

job_file = os.path.join(dst_folder, f"linear_{job_name}_summary.json")
with open(job_file, "w") as f:
    json.dump(summary, f, indent=4)
