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


job_parameters = [(zone, target) for zone in ["dk1 down", "dk1 up", "dk2 down", "dk2 up"] for target in ["volbids", "FRR", "CM"]]
total_jobs = len(job_parameters)
train_end = "2025-08-31"
calibration_end = "2025-10-31"
test_end = "2025-12-31"
N_ITERATIONS = 5
SEEDS = [101, 202, 303, 25, 505]

results_root = Path(os.getcwd())
predictions_root = results_root / "predictions"
legacy_root = predictions_root / "tcn_point"
temp_root = predictions_root / "temp_tcn_point"
predictions_root.mkdir(parents=True, exist_ok=True)
legacy_root.mkdir(parents=True, exist_ok=True)
temp_root.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def load_best_params():
    summary_file = results_root / "results" / "model_summaries" / "tcn_point_best_params_summary.json"
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

    if feats is None:
        X_train, _ = remove_high_correlation(X_train, threshold=0.90)
        feats = X_train.columns

    X_train, X_calibration, X_val = X_train[feats], X_calibration[feats], X_val[feats]
    X_train_cal = pd.concat([X_train, X_calibration], axis=0)
    y_train_cal = pd.concat([y_train, y_calibration], axis=0)
    X_full = pd.concat([X_train, X_calibration, X_val], axis=0)
    y_full = pd.concat([y_train, y_calibration, y_val], axis=0)

    return X_train, y_train, X_train_cal, y_train_cal, X_full, y_full, y_calibration, y_val


def average_point_predictions(fit_X, fit_y, X_full, y_full, cfg_local, split_idx, description):
    preds_iters = []
    true = None

    for i in tqdm(range(N_ITERATIONS), desc=description, leave=False):
        set_global_seed(SEEDS[i % len(SEEDS)])
        forecaster = TCNForecaster(task="point", valshare=0.2, **cfg_local)
        forecaster.fit(fit_X, fit_y)

        preds_iter, true = forecaster.predict(X_full, y_full)
        preds_iters.append(preds_iter.to_numpy())

    preds_avg = np.mean(np.vstack(preds_iters), axis=0)
    preds_std = np.std(np.vstack(preds_iters), axis=0)
    preds_slice = preds_avg[-split_idx:]
    std_slice = preds_std[-split_idx:]
    true = true.iloc[-split_idx:]

    return pd.DataFrame({"mean": preds_slice, "std": std_slice, "true": true}, index=true.index)


def write_job_outputs(job_idx, zone, target, cp_df, native_df):
    safe_zone = zone.replace(" ", "_")
    safe_target = target.replace(" ", "_")

    cp_path = legacy_root / f"cp_{safe_zone}_{safe_target}.parquet"
    native_path = legacy_root / f"native_{safe_zone}_{safe_target}.parquet"
    cp_df.to_parquet(cp_path)
    native_df.to_parquet(native_path)

    return cp_path, native_path


cfg = load_best_params()

job_idx = int(os.environ.get("LSB_JOBINDEX", os.environ.get("JOB_IDX", 1)))
if job_idx < 1 or job_idx > total_jobs:
    raise ValueError(f"LSB_JOBINDEX/JOB_IDX must be between 1 and {total_jobs}, got {job_idx}")

zone, target = job_parameters[job_idx - 1]
print(f"Processing {zone} | {target} for job_idx={job_idx}/{total_jobs}")

best_entry = cfg.get(f"{zone} | {target}", {})
cfg_local = dict(best_entry.get("best_params", {}) or {})
selected_feats = best_entry.get("selected_features")
int_params = ["n_layers", "n_filters", "window_size", "batch_size", "epochs", "kernel_size", "patience"]
for ip in int_params:
    if ip in cfg_local:
        cfg_local[ip] = int(cfg_local[ip])

X_train, y_train, X_train_cal, y_train_cal, X_full, y_full, y_calibration, y_val = prepare_job_data(
    zone,
    target,
    feats=selected_feats,
)

cp_df = average_point_predictions(
    fit_X=X_train,
    fit_y=y_train,
    X_full=X_full,
    y_full=y_full,
    cfg_local=cfg_local,
    split_idx=len(y_calibration) + len(y_val),
    description=f"Processing {zone}_{target} (CP)",
)

native_df = average_point_predictions(
    fit_X=X_train_cal,
    fit_y=y_train_cal,
    X_full=X_full,
    y_full=y_full,
    cfg_local=cfg_local,
    split_idx=len(y_val),
    description=f"Processing {zone}_{target} (Native)",
)

cp_path, native_path = write_job_outputs(job_idx, zone, target, cp_df, native_df)
print(f"Wrote {cp_path} and {native_path}")
