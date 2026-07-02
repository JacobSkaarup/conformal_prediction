import os
import warnings

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
warnings.filterwarnings(
    "ignore",
    message="Found Intel OpenMP .* and LLVM OpenMP .* loaded at the same time.*",
    category=RuntimeWarning,
 )


import json

import lightgbm as lgb
import numpy as np
import optuna
import src.conformal_prediction as cp
import src.metrics as met
import xgboost as xgb
from optuna.samplers import TPESampler
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from src.data_loader import load_data
from tqdm.auto import tqdm

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# Set the date thresholds for splitting the data
train_end = "2025-08-31"
calibration_end = "2025-10-31"
test_end = "2025-12-31"
N_TRIALS = int(os.environ.get("N_TRIALS", 120))

summary_path = os.getcwd() + "/results/model_summaries/"

zones = ["dk1 down", "dk1 up", "dk2 down", "dk2 up"]
targets = ["volbids", "FRR", "CM"]
job_parameters = []
for zone in zones:
    for target in targets:
        job_parameters.append([zone, target])


def build_lgb(table, name):
    return lgb.LGBMRegressor(**dict(table[name]["best_params"]), objective="regression")


def build_knn(table, name):
    return KNeighborsRegressor(n_neighbors=table[name]["optimal_k"])


def build_lasso(table, name):
    return Lasso(alpha=table[name]["alpha"])


def build_ridge(table, name):
    return Ridge(alpha=table[name]["alpha"])


def build_xgb(table, name):
    return xgb.XGBRegressor(**dict(table[name]["best_params"]), objective="regression")


MODEL_JOBS = [
    {
        "model_name": "lgb",
        "summary_file": "lgb_summary.json",
        "model_builder": build_lgb,
    },
    {
        "model_name": "knn_forward",
        "summary_file": "knn_forward_selection_summary.json",
        "model_builder": build_knn,
    },
    {
        "model_name": "lasso_forward",
        "summary_file": "linear_forward_lasso.json",
        "model_builder": build_lasso,
    },
    {
        "model_name": "ridge_backward",
        "summary_file": "linear_backward_ridge.json",
        "model_builder": build_ridge,
    },
    {
        "model_name": "ridge_forward",
        "summary_file": "linear_forward_ridge.json",
        "model_builder": build_ridge,
    },
    {
        "model_name": "xgb",
        "summary_file": "xgb_summary.json",
        "model_builder": build_xgb,
    },
]


def SSCWC(intervals, y_val, alpha=0.1, eta=50, b=100):
    "Sigmoid Smoothed CWC"
    try:
        lower = np.quantile(intervals, alpha / 2, axis=1)
        upper = np.quantile(intervals, 1 - alpha / 2, axis = 1)
        picp = met.PICP(y_val.values, lower, upper)
        pinaw = met.PINAW(y_val.values, lower, upper)
        target = 1 - alpha
        cov_gap = picp - target
        return float(pinaw * (1 / (1 + np.exp(-b * cov_gap)) + np.exp(-eta * cov_gap)))
    except Exception:
        return float("inf")



def run_cps_tuning_for_table(results_file, model_builder, n_trials=200):
    with open(results_file, "r", encoding="utf-8") as f:
        table = json.load(f)

    cps_params = {}

    for job in tqdm(job_parameters, desc=f"Tuning CPS Parameters for {results_file}"):
        name = job[0] + " | " + job[1]
        feature, target = load_data(job[0], job[1])
        X_train, X_calibration, X_val, y_train, y_calibration, y_val, _, _ = cp.train_cal_test_date_split(
            feature, target, train_end, calibration_end, test_end, scaled=True
        )

        model = model_builder(table, name)
        feats = table[name]["optimal_features"]

        def objective(trial):
            # Create a fresh CPS per trial to avoid state leakage between trials
            cps = cp.CPS(model)

            cfg = {
                "k": trial.suggest_int("k", 2, 250, step=5),
                "mondrian_strategy": trial.suggest_categorical("mondrian_strategy", ["predictions", "difficulty", None]),
                "bins": trial.suggest_int("bins", 2, 5),
                "difficulty_model" : trial.suggest_categorical("difficulty_model", ["knn", "rf", None]),
            }
            try:
                cps.fit_calibrate(X_train.loc[:, feats], y_train, X_calibration.loc[:, feats], y_calibration, **cfg)
                intervals = cps.predict_cpds(X_val.loc[:, feats])
                # return SSCWC(intervals, y_val, alpha=0.1, eta=50, b=100)
                return met.CRPS(y_true=y_val.values, cpds=intervals)
            except (RuntimeError, ValueError, TypeError, FloatingPointError):
                return float("inf")

        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42, multivariate=True),
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=False,
            catch=(RuntimeError, ValueError, TypeError, FloatingPointError),
        )

        cps_params[name] = study.best_trial.params

    for key, value in cps_params.items():
        # Store all CPS tuning parameters under a single key for easy import/use
        table[key]["cps_params"] = value


    with open(results_file, "w") as f:
        json.dump(table, f, indent=4)

    return table


def get_job_idx():
    return int(os.environ.get("LSB_JOBINDEX", os.environ.get("JOB_IDX", 1)))


def main():
    total_jobs = len(MODEL_JOBS)
    job_idx = get_job_idx()

    if job_idx < 1 or job_idx > total_jobs:
        raise ValueError(f"LSB_JOBINDEX/JOB_IDX must be between 1 and {total_jobs}, got {job_idx}")

    job = MODEL_JOBS[job_idx - 1]
    results_file = summary_path + job["summary_file"]

    print(
        "Running CPS tuning | "
        f"job_idx={job_idx}/{total_jobs} | model={job['model_name']} | "
        f"n_trials={N_TRIALS} | summary={job['summary_file']}"
    )

    run_cps_tuning_for_table(
        results_file=results_file,
        model_builder=job["model_builder"],
        n_trials=N_TRIALS,
    )

    print(f"Completed CPS tuning for model={job['model_name']}")


if __name__ == "__main__":
    main()
