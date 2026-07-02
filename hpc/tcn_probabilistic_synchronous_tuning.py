import json
import os
import time
import numpy as np
import src.conformal_prediction as cp
from src.data_loader import load_data, remove_high_correlation
import src.metrics as met
from src.tcn_pipeline import TCNForecaster
from src.tcn_tuning import (
    DEFAULT_PROBABILISTIC_INSTABILITY_MARKERS,
    align_holdout_predictions,
    cleanup_after_fit,
    gaussian_nll,
    make_instability_checker,
    run_two_phase_tuning,
    set_global_seed,
)


# --- Configuration & Initialization ---
train_end = "2025-08-31"
calibration_end = "2025-10-31"
test_end = "2025-12-31"
results_dir = os.path.join(os.getcwd(), "results/model_summaries/tcn_prop_temp")
os.makedirs(results_dir, exist_ok=True)

# Define two-stage promotion parameters
N_TRIALS_PHASE_1 = 50
TOP_K_PROMOTION = 10
PRIMARY_SEED = 101
VERIFICATION_SEEDS = [202, 303, 25, 505]
STABILITY_WEIGHT = 0.2

# Target selection logic (LSB_JOBINDEX for cluster arrays)
job_idx = int(os.environ.get("LSB_JOBINDEX", 1))
job_parameters = [
    [zone, t]
    for zone in ["dk1 down", "dk1 up", "dk2 down", "dk2 up"]
    for t in ["volbids", "FRR", "CM"]
]
zone, target = job_parameters[job_idx - 1]
study_name = f"tcn_probabilistic_synchronous_tuning_{zone}_{target}"
study_storage_path = os.path.join(results_dir, f"{study_name}.db")

# Use the same load / split pattern as the point-tuning script for consistency
X_train, X_calibration, X_test, y_train, y_calibration, y_test, X_scaler, y_scaler = cp.train_cal_test_date_split(
    *load_data(zone, target), train_end, calibration_end, test_end, scaled=True
)
X_train, corr_cols = remove_high_correlation(X_train, threshold=0.90)
selected_features = list(X_train.columns)


guesses = [
    {
        "n_layers": 3,
        "n_filters": 64,
        "window_size": 60,
        "batch_size": 128,
        "epochs": 55,
        "kernel_size": 3,
        "dropout": 0.2481543276568569,
        "lr": 0.0012668363265474535,
        "weight_decay": 4.537041264656879e-05,
        "patience": 3,
        "factor": 0.6130230267382876,
    },
    {
        "n_layers": 2,
        "n_filters": 64,
        "window_size": 60,
        "batch_size": 64,
        "epochs": 80,
        "kernel_size": 3,
        "dropout": 0.29174980375688614,
        "lr": 0.0018063778197080188,
        "weight_decay": 1.6327810654127445e-05,
        "patience": 3,
        "factor": 0.5347134828491296,
    },
    {
        "n_layers": 3,
        "n_filters": 64,
        "window_size": 60,
        "batch_size": 64,
        "epochs": 45,
        "kernel_size": 3,
        "dropout": 0.26880386167081155,
        "lr": 0.0012016219311133535,
        "weight_decay": 5.0706359134358005e-06,
        "patience": 5,
        "factor": 0.6326801357658349,
    },
    {
        "n_layers": 3,
        "n_filters": 8,
        "window_size": 60,
        "batch_size": 64,
        "epochs": 65,
        "kernel_size": 3,
        "dropout": 0.28977783980701044,
        "lr": 0.00012557621355252724,
        "weight_decay": 4.169266243197284e-05,
        "patience": 6,
        "factor": 0.4,
    },
    {
        "n_layers": 3,
        "n_filters": 16,
        "window_size": 60,
        "batch_size": 64,
        "epochs": 50,
        "kernel_size": 3,
        "dropout": 0.2912340870774203,
        "lr": 0.0016526391090371751,
        "weight_decay": 3.300274712004716e-05,
        "patience": 8,
        "factor": 0.5217022263047776,
    },
]
best_payload, study = run_two_phase_tuning(
    task="probabilistic",
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
    instability_checker=make_instability_checker(DEFAULT_PROBABILISTIC_INSTABILITY_MARKERS),
    catch_exceptions=(RuntimeError, ValueError),
)

best_params = best_payload.get("best_params")
best_payload["selected_features"] = selected_features
if best_params is not None:
    print("\n--- Executing Final Holdout Evaluation (probabilistic) ---")
    split_idx = int(len(X_train) * 0.8)
    mu_predictions = []
    sigma_predictions = []
    holdout_true = None

    for seed in VERIFICATION_SEEDS:
        set_global_seed(seed)
        try:
            holdout_forecaster = TCNForecaster(task="probabilistic", valshare=0.2, **best_params)
            holdout_forecaster.fit(X_train, y_train)
            mu_pred, sigma_pred, true_vals = holdout_forecaster.predict(X_train, y_train)
            true_vals_full = true_vals
            mu_pred, true_vals = align_holdout_predictions(
                mu_pred,
                true_vals_full,
                split_idx=split_idx,
                window_size=int(best_params["window_size"]),
            )
            sigma_pred, _ = align_holdout_predictions(
                sigma_pred,
                true_vals_full,
                split_idx=split_idx,
                window_size=int(best_params["window_size"]),
            )
            mu_predictions.append(mu_pred.to_numpy())
            sigma_predictions.append(sigma_pred.to_numpy())
            holdout_true = true_vals
        except Exception as e:
            print(f"Holdout evaluation failed for seed {seed}: {e}")
            mu_predictions = []
            sigma_predictions = []
            holdout_true = None
            break
        finally:
            cleanup_after_fit()

    if mu_predictions and sigma_predictions and holdout_true is not None:
        mu_avg = np.mean(np.vstack(mu_predictions), axis=0)
        sigma_avg = np.mean(np.vstack(sigma_predictions), axis=0)
        holdout_y = holdout_true.to_numpy()
        best_payload["holdout_crps"] = float(met.CRPS(y_true=holdout_y, mu=mu_avg, sigma=sigma_avg))
        best_payload["holdout_nll"] = gaussian_nll(holdout_y, mu_avg, sigma_avg)
        best_payload["holdout_metric_name"] = "crps"
    else:
        best_payload["holdout_crps"] = None
        best_payload["holdout_nll"] = None
        best_payload["holdout_metric_name"] = "crps"



with open(os.path.join(results_dir, f"tcn_probabilistic_best_params_{zone}_{target}.json"), "w") as f:
    json.dump(best_payload, f, indent=4)



# Await that all jobs have completed before merging the results into a combined summary file
if job_idx == len(job_parameters):  # Last job to finish will do the merging
    while True:
        all_files = os.listdir(results_dir)
        expected_files = [f"tcn_probabilistic_best_params_{param[0]}_{param[1]}.json" for param in job_parameters]
        if all(file in all_files for file in expected_files):
            combined_summary = {}
            for param in job_parameters:
                with open(os.path.join(results_dir, f"tcn_probabilistic_best_params_{param[0]}_{param[1]}.json"), "r") as f:
                    combined_summary[f"{param[0]} | {param[1]}"] = json.load(f)
            with open(os.path.join(os.getcwd(), "results/model_summaries/tcn_probabilistic_best_params_summary.json"), "w") as f:
                json.dump(combined_summary, f, indent=4)
            # Remove individual files to clean up
            for file in expected_files:
                os.remove(os.path.join(results_dir, file))
            break
        else:
            time.sleep(30)  # Check every 30 seconds

            