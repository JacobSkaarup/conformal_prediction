import json
import os
import time
import numpy as np
import src.conformal_prediction as cp
from src.data_loader import load_data, remove_high_correlation
from src.tcn_pipeline import TCNForecaster
from src.tcn_tuning import (
    DEFAULT_POINT_INSTABILITY_MARKERS,
    align_holdout_predictions,
    cleanup_after_fit,
    make_instability_checker,
    run_two_phase_tuning,
    set_global_seed,
)

# --- Configuration & Initialization ---
train_end = "2025-08-31"
calibration_end = "2025-10-31"
test_end = "2025-12-31"
results_dir = os.path.join(os.getcwd(), "results/model_summaries/tcn_temp")
os.makedirs(results_dir, exist_ok=True)

# Define two-stage promotion parameters
N_TRIALS_PHASE_1 = 50
TOP_K_PROMOTION = 10
PRIMARY_SEED = 101
VERIFICATION_SEEDS = [202, 303, 25, 505]
STABILITY_WEIGHT = 0.2

# Target selection logic (Assuming LSB_JOBINDEX is handled as in original)
job_idx = int(os.environ.get('LSB_JOBINDEX', 1))
job_parameters = [[zone, target] for zone in ["dk1 down", "dk1 up", "dk2 down", "dk2 up"] for target in ["volbids", "FRR", "CM"]]
zone, target = job_parameters[job_idx - 1]
study_name = f"tcn_synchronous_tuning_{zone}_{target}"
study_storage_path = os.path.join(results_dir, f"{study_name}.db")

X_train, X_calibration, X_test, y_train, y_calibration, y_test, X_scaler, y_scaler = cp.train_cal_test_date_split(
    *load_data(zone, target), train_end, calibration_end, test_end, scaled=True
)
X_train, corr_cols = remove_high_correlation(X_train, threshold=0.90)
selected_features = list(X_train.columns)

guesses = [
    {
        "n_layers": 2,
        "n_filters": 8,
        "window_size": 60,
        "batch_size": 16,
        "epochs": 45,
        "kernel_size": 3,
        "dropout": 0.3252312085273195,
        "lr": 0.00360553866934869,
        "weight_decay": 8e-07,
        "patience": 3,
        "factor": 0.4043306539614822
    },
    {
        "n_layers": 2,
        "n_filters": 64,
        "window_size": 48,
        "batch_size": 32,
        "epochs": 50,
        "kernel_size": 3,
        "dropout": 0.2,
        "lr": 0.0009438952810126065,
        "weight_decay": 4.732551242713967e-06,
        "patience": 3,
        "factor": 0.6768770754842861
    },
    {
        "n_layers": 1,
        "n_filters": 32,
        "window_size": 60,
        "batch_size": 8,
        "epochs": 55,
        "kernel_size": 2,
        "dropout": 0.2,
        "lr": 0.0018032775846821658,
        "weight_decay": 7.1062212417212165e-06,
        "patience": 1,
        "factor": 0.4817642402507877
    },
    {
        "n_layers": 2,
        "n_filters": 16,
        "window_size": 36,
        "batch_size": 8,
        "epochs": 50,
        "kernel_size": 2,
        "dropout": 0.3240228249808733,
        "lr": 0.00034695916603302916,
        "weight_decay": 8e-07,
        "patience": 2,
        "factor": 0.528132336587877
    },
    {
        "window_size": 36,
        "batch_size": 8,
        "epochs": 70,
        "n_layers": 2,
        "n_filters": 8,
        "kernel_size": 2,
        "dropout": 0.4593979019797758,
        "lr": 0.0011439257943983722,
        "weight_decay": 3.230600727278847e-06,
        "patience": 1,
        "factor": 0.4945925038972909
    },
    {
        "window_size": 36,
        "batch_size": 8,
        "epochs": 45,
        "n_layers": 3,
        "n_filters": 64,
        "kernel_size": 2,
        "dropout": 0.2550213529560302,
        "lr": 0.0003287747413991121,
        "weight_decay": 3.7520558551242797e-06,
        "patience": 4,
        "factor": 0.4873687420594126
    }
]

best_payload, study = run_two_phase_tuning(
    task="point",
    X_train=X_train,
    y_train=y_train,
    study_name=study_name,
    study_storage_path=study_storage_path,
    n_trials_phase_1=N_TRIALS_PHASE_1,
    top_k_promotion=TOP_K_PROMOTION,
    primary_seed=PRIMARY_SEED,
    verification_seeds=VERIFICATION_SEEDS,
    stability_weight=STABILITY_WEIGHT,
    guesses=guesses,
    instability_checker=make_instability_checker(DEFAULT_POINT_INSTABILITY_MARKERS),
    catch_exceptions=(RuntimeError,),
)

best_params = best_payload["best_params"]
best_payload["selected_features"] = selected_features

if best_params is not None:
    print("\n--- Executing Final Holdout Evaluation ---")
    split_idx = int(len(X_train) * 0.8)
    holdout_predictions = []
    holdout_true = None

    for seed in VERIFICATION_SEEDS:
        set_global_seed(seed)
        try:
            holdout_forecaster = TCNForecaster(task="point", valshare=0.2, **best_params)
            holdout_forecaster.fit(X_train, y_train)
            preds, true = holdout_forecaster.predict(X_train, y_train)
            preds, true = align_holdout_predictions(preds, true, split_idx=split_idx, window_size=int(best_params["window_size"]))
            holdout_predictions.append(preds.to_numpy())
            holdout_true = true
        except Exception as e:
            print(f"Holdout evaluation failed for seed {seed}: {e}")
            holdout_predictions = []
            holdout_true = None
            break
        finally:
            cleanup_after_fit()

    if holdout_predictions and holdout_true is not None:
        avg_preds = np.mean(np.vstack(holdout_predictions), axis=0)
        best_payload["holdout_mse"] = float(np.mean((holdout_true.to_numpy() - avg_preds) ** 2))
        best_payload["holdout_metric_name"] = "mse"
    else:
        best_payload["holdout_mse"] = None
        best_payload["holdout_metric_name"] = "mse"


# Save back the complete comparable payload
with open(os.path.join(results_dir, f"tcn_point_best_params_{zone}_{target}.json"), "w") as f:
    json.dump(best_payload, f, indent=4)

# Await that all jobs have completed before merging the results into a combined summary file
if job_idx == len(job_parameters):  # Last job to finish will do the merging
    while True:
        all_files = os.listdir(results_dir)
        expected_files = [f"tcn_point_best_params_{param[0]}_{param[1]}.json" for param in job_parameters]
        if all(file in all_files for file in expected_files):
            combined_summary = {}
            for param in job_parameters:
                with open(os.path.join(results_dir, f"tcn_point_best_params_{param[0]}_{param[1]}.json"), "r") as f:
                    combined_summary[f"{param[0]} | {param[1]}"] = json.load(f)
            with open(os.path.join(os.getcwd(), "results/model_summaries/tcn_point_best_params_summary.json"), "w") as f:
                json.dump(combined_summary, f, indent=4)
            # Remove individual files to clean up
            for file in expected_files:
                os.remove(os.path.join(results_dir, file))
            break
        else:
            time.sleep(30)  # Check every 30 seconds