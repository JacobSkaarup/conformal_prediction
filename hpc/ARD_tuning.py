import json
import os
import warnings
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ARDRegression
from tqdm.notebook import tqdm

import src.conformal_prediction as cp
from src.data_loader import load_data
import src.metrics as met
from src.model_tuning import TimeSeriesARDTuner

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Date thresholds (aligned with HPC tuning scripts)
train_end = "2025-08-31"
calibration_end = "2025-10-31"
test_end = "2025-12-31"

# Datasets to tune across
zones = ["dk1 down", "dk1 up", "dk2 down", "dk2 up"]
targets = ["volbids", "FRR", "CM"]

N_TRIALS = 120
COEF_THRESHOLD = 1.0e-8
SEED = 42


ard_summary = {}
prediction_dir = os.path.join(os.getcwd(), "predictions/ard_regression")
os.makedirs(prediction_dir, exist_ok=True)

for zone in tqdm(zones, desc="Zone", position=0):
    for target in tqdm(targets, desc="Target", leave=False, position=1):
        key = f"{zone} | {target}"
        file_key = key.replace(" | ", "_").replace(" ", "_")

        X_train, X_calibration, X_test, y_train, y_calibration, y_test, X_scaler, y_scaler = cp.train_cal_test_date_split(
            *load_data(zone, target),
            train_end,
            calibration_end,
            test_end,
            scaled=True,
        )
        X_native = pd.concat([X_train, X_calibration], axis=0)
        y_native = pd.concat([y_train, y_calibration], axis=0)

        X_cp = pd.concat([X_calibration, X_test], axis=0)
        y_cp = pd.concat([y_calibration, y_test], axis=0)

        dataset_seed = SEED 
        tuner = TimeSeriesARDTuner(
            X_train,
            y_train,
            coef_threshold=COEF_THRESHOLD,
            val_share=0.2,
            seed=dataset_seed,
        )
        results = tuner.fit_and_tune(n_trials=N_TRIALS, progress_bar=False)

        selected_features = results["selected_features"]
        

        model = ARDRegression(**results["best_params"])
        model.fit(X_native[selected_features], y_native)
        y_pred_native, y_std_native = model.predict(X_test[selected_features], return_std=True)
        native_preds = pd.DataFrame({"mean": y_pred_native, "std": y_std_native}, index=X_test.index)
        native_preds.to_parquet(os.path.join(prediction_dir, f"native_{file_key}.parquet"))

        native_crps = met.CRPS(y_true=y_test.values, mu=y_pred_native, sigma=y_std_native)
        ard_summary[key] = {
            "best_params": results["best_params"],
            "holdout nll": results["holdout_nll"],
            "crps": float(native_crps),
            "n_selected_features": len(selected_features),
            "selected_features": selected_features,
            "coefficients": results["coefficients"],
        }

        model.fit(X_train[selected_features], y_train)
        y_pred_cp, y_std_cp = model.predict(X_cp[selected_features], return_std=True)
        cp_preds = pd.DataFrame({"mean": y_pred_cp, "std": y_std_cp}, index=X_cp.index)
        cp_preds.to_parquet(os.path.join(prediction_dir, f"cp_{file_key}.parquet"))

dst_folder = os.path.join(os.getcwd(), "results/model_summaries")
os.makedirs(dst_folder, exist_ok=True)
out_path = os.path.join(dst_folder, "ard_regression_summary.json")

with open(out_path, "w") as f:
    json.dump(ard_summary, f, indent=4)
