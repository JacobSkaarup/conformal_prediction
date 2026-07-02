from pathlib import Path
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import src.conformal_prediction as cp
from src.data_loader import load_data, remove_high_correlation
from src.tcn_pipeline import TCNForecaster


zones = ["dk1 down", "dk1 up", "dk2 down", "dk2 up"]
targets = ["volbids", "FRR", "CM"]
job_parameters = [(zone, target) for zone in zones for target in targets]
job_modes = ["cp", "native"]
total_jobs = len(job_parameters) * len(job_modes)

train_end = "2025-08-31"
calibration_end = "2025-10-31"
test_end = "2025-12-31"
N_ITERATIONS = 5
SEEDS = [101, 202, 303, 25, 505]

results_root = Path(os.getcwd())
predictions_root = results_root / "predictions"
predictions_root.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Switch around for better reproducibility at performance tradeoff
    torch.backends.cudnn.deterministic = False  # True
    torch.backends.cudnn.benchmark = True  # False


def safe_name(value):
    return value.replace(" ", "_")


def load_best_params():
    summary_file = results_root / "results" / "model_summaries" / "tcn_probabilistic_best_params_summary.json"
    with open(summary_file, "r") as f:
        summary = json.load(f)

    cfg = {}
    for key, value in summary.items():
        bp = value.get("best_params") or {}
        cfg[key] = {
            "best_params": dict(bp),
            "selected_features": value.get("selected_features"),
        }
    return cfg


def prepare_job_data(zd, t, feats=None):
    features, target = load_data(zd, t)
    X_train, X_calibration, X_val, y_train, y_calibration, y_val, _, _ = cp.train_cal_test_date_split(
        features,
        target,
        train_end,
        calibration_end,
        test_end,
        scaled=True,
    )

    # If tuning provided a feature subset, respect it; otherwise remove high-correlation columns as fallback
    if feats is None:
        X_train, _ = remove_high_correlation(X_train, threshold=0.90)
        feats = X_train.columns

    X_train, X_calibration, X_val = X_train[feats], X_calibration[feats], X_val[feats]

    X_train_cal = pd.concat([X_train, X_calibration], axis=0)
    y_train_cal = pd.concat([y_train, y_calibration], axis=0)
    X_full = pd.concat([X_train, X_calibration, X_val], axis=0)
    y_full = pd.concat([y_train, y_calibration, y_val], axis=0)

    return X_train, y_train, X_train_cal, y_train_cal, X_full, y_full, y_calibration, y_val


def average_predictions(fit_X, fit_y, X_full, y_full, cfg_local, split_idx, description):
    output_len = len(X_full) - cfg_local["window_size"]
    mu_iters = []
    sigma_iters = []
    true = None

    for i in tqdm(range(N_ITERATIONS), desc=description, leave=False):
        set_global_seed(SEEDS[i % len(SEEDS)])
        forecaster = TCNForecaster(task="probabilistic", valshare=0.2, **cfg_local)
        forecaster.fit(fit_X, fit_y)

        mu_iter, sigma_iter, true = forecaster.predict(X_full, y_full)
        mu_iters.append(mu_iter.to_numpy())
        sigma_iters.append(sigma_iter.to_numpy())

    mu_avg = np.mean(np.vstack(mu_iters), axis=0)
    sigma_avg = np.mean(np.vstack(sigma_iters), axis=0)
    mu_slice = mu_avg[-split_idx:]
    sigma_slice = sigma_avg[-split_idx:]
    true = true.iloc[-split_idx:]

    return pd.DataFrame(data={"mean": mu_slice, "std": sigma_slice, "true": true}, index=true.index)


def write_job_output(job_idx, zd, t, mode, job_df):
    out_dir = predictions_root / f"tcn_probabilistic"
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{mode}_{safe_name(zd)}_{safe_name(t)}.parquet"
    out_path = out_dir / filename
    job_df.to_parquet(out_path)
    return out_path




cfg = load_best_params()

# Each job invocation handles a single dataset+mode using LSB_JOBINDEX
job_idx = int(os.environ.get("LSB_JOBINDEX", os.environ.get("JOB_IDX", 1)))
if job_idx < 1 or job_idx > total_jobs:
    raise ValueError(f"LSB_JOBINDEX/JOB_IDX must be between 1 and {total_jobs}, got {job_idx}")

dataset_idx = (job_idx - 1) % len(job_parameters)
mode_idx = (job_idx - 1) // len(job_parameters)
zd, t = job_parameters[dataset_idx]
mode = job_modes[mode_idx]
print(
    "Running TCN probabilistic predictions | "
    f"job_idx={job_idx}/{total_jobs} | zone={zd} | target={t} | mode={mode}"
)

best_entry = cfg.get(f"{zd} | {t}", {})
cfg_local = dict(best_entry.get("best_params", {}) or {})
selected_feats = best_entry.get("selected_features")
int_params = ["n_layers", "n_filters", "window_size", "batch_size", "epochs", "kernel_size", "patience"]
for ip in int_params:
    if ip in cfg_local:
        cfg_local[ip] = int(cfg_local[ip])

X_train, y_train, X_train_cal, y_train_cal, X_full, y_full, y_calibration, y_val = prepare_job_data(zd, t, feats=selected_feats)

if mode == "cp":
    job_df = average_predictions(
        fit_X=X_train,
        fit_y=y_train,
        X_full=X_full,
        y_full=y_full,
        cfg_local=cfg_local,
        split_idx=len(y_calibration) + len(y_val),
        description=f"Processing {zd}_{t} (CP)",
    )
else:
    job_df = average_predictions(
        fit_X=X_train_cal,
        fit_y=y_train_cal,
        X_full=X_full,
        y_full=y_full,
        cfg_local=cfg_local,
        split_idx=len(y_val),
        description=f"Processing {zd}_{t} (Native)",
    )

out_path = write_job_output(job_idx, zd, t, mode, job_df)
print(f"Saved predictions to {out_path}")