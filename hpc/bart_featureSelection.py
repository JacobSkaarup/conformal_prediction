import json
import os
import time
import warnings

import arviz as az
import numpy as np
import pymc_bart as pmb
from matplotlib import rcParams
from scipy.stats import pearsonr

import src.conformal_prediction as cp
import src.metrics as met
from src.data_loader import load_data
from src.models import BayesRegressor

warnings.simplefilter(action="ignore", category=FutureWarning)


def _configure_job_local_cache():
    """Set a per-job cache directory to avoid ArviZ cache race conditions on HPC."""
    if "XDG_CACHE_HOME" in os.environ and os.environ["XDG_CACHE_HOME"]:
        return

    user = os.environ.get("USER", "user")
    job_id = os.environ.get("LSB_JOBID", "local")
    job_idx = os.environ.get("LSB_JOBINDEX", "1")
    base_tmp = os.environ.get("TMPDIR", "/tmp")
    cache_dir = f"{base_tmp}/{user}/xdg_cache_{job_id}_{job_idx}"

    os.environ["XDG_CACHE_HOME"] = cache_dir
    os.makedirs(f"{cache_dir}/arviz", exist_ok=True)


_configure_job_local_cache()


RANDOM_SEED = 89
np.random.seed(RANDOM_SEED)


def get_lower_bound(vi_results):
    """Compute the lower HDI bound for pairwise draw-level R^2 references."""
    preds = vi_results["preds"]
    preds_all = vi_results["preds_all"]
    samples = preds.shape[1]

    r_2_ref = np.zeros(samples - 1)
    for j in range(samples - 1):
        r, _ = pearsonr(preds_all[j].flatten(), preds_all[j + 1].flatten())
        r_2_ref[j] = r**2

    ci_prob = rcParams.get("stats.ci_prob", 0.94)
    interval_bounds = az.hdi(r_2_ref, hdi_prob=ci_prob)
    return interval_bounds[0]


train_end = "2025-08-31"
calibration_end = "2025-10-31"
test_end = "2025-12-31"

zones = ["dk1 down", "dk1 up", "dk2 down", "dk2 up"]
targets = ["volbids", "FRR", "CM"]

dst_path = os.path.join(os.getcwd(), "results", "model_summaries")
os.makedirs(dst_path, exist_ok=True)

job_parameters = [(zone, target) for zone in zones for target in targets]
job_idx = int(os.environ.get("LSB_JOBINDEX", 1))
zone, target = job_parameters[job_idx - 1]

job_result = {}

X_train, X_calibration, X_test, y_train, y_calibration, y_test, X_scaler, y_scaler = cp.train_cal_test_date_split(
    *load_data(zone_direction=zone, target=target),
    train_end,
    calibration_end,
    test_end,
    scaled=True,
)

split_idx = int(len(X_train) * 0.8)
X_train_base = X_train.iloc[:split_idx].copy()
X_val_base = X_train.iloc[split_idx:].copy()
y_train_base = y_train.iloc[:split_idx].copy()
y_val_base = y_train.iloc[split_idx:].copy()

for distribution in ("normal", "studentT"):
    # Fit baseline model on the full feature set for comparison.
    baseline_model = BayesRegressor(seed=RANDOM_SEED, distribution=distribution)
    baseline_model.fit(X_train_base, y_train_base)

    baseline_samples = baseline_model.predict(X_val_base)
    initial_crps = float(met.CRPS(y_true=y_val_base, samples=baseline_samples))
    initial_val_mse = float(met.MSE(y_val_base, baseline_samples.mean(axis=1).values))

    var_inclusion = pmb.get_variable_inclusion(baseline_model.posterior, X=X_train_base)
    sorted_features = var_inclusion[1]
    inclusion_mask = var_inclusion[0] > 1 / len(X_train_base.columns)
    selected_features_inc = np.array(sorted_features)[inclusion_mask]

    vi_results = pmb.compute_variable_importance(baseline_model.posterior, bartrv=baseline_model.ww, X=X_val_base)
    lower_bound = get_lower_bound(vi_results)

    k = 0
    for i, score in enumerate(vi_results["r2_mean"]):
        if score >= lower_bound:
            k = i
            break

    selected_features_imp = [str(vi_results["labels"][0])] + [
        vi_results["labels"][i].split("+ ")[1] for i in range(1, k + 1)
    ]

    selected_features = sorted(set(selected_features_inc).union(set(selected_features_imp)))

    X_train_selected = X_train_base[selected_features]
    X_val_selected = X_val_base[selected_features]

    tuned_model = BayesRegressor(seed=RANDOM_SEED, distribution=distribution)
    tuned_model.fit(X_train_selected, y_train_base)

    val_samples = tuned_model.predict(X_val_selected)
    subset_crps = float(met.CRPS(y_true=y_val_base, samples=val_samples))
    holdout_mse = float(met.MSE(y_val_base, val_samples.mean(axis=1).values))

    key = f"{zone} | {target}"
    job_result.setdefault(key, {})[distribution] = {
        "optimal_features": selected_features,
        "optimal_params": {
            "distribution": distribution,
            "seed": RANDOM_SEED,
        },
        "holdout_mse": holdout_mse,
        "initial_crps": initial_crps,
        "subset_crps": subset_crps,
        "initial_val_mse": initial_val_mse,
        "n_features": len(selected_features),
    }

job_file = os.path.join(dst_path, f"bart_feature_selection_job_{job_idx}.json")
with open(job_file, "w") as f:
    json.dump(job_result, f, indent=4)

if job_idx == len(job_parameters):
    while True:
        all_files = os.listdir(dst_path)
        expected_files = [f"bart_feature_selection_job_{idx}.json" for idx in range(1, len(job_parameters) + 1)]
        if all(file_name in all_files for file_name in expected_files):
            combined_summary = {}
            for idx in range(1, len(job_parameters) + 1):
                with open(os.path.join(dst_path, f"bart_feature_selection_job_{idx}.json"), "r") as f:
                    partial = json.load(f)
                    for dataset_key, payload in partial.items():
                        if dataset_key not in combined_summary:
                            combined_summary[dataset_key] = {}
                        combined_summary[dataset_key].update(payload)

            with open(os.path.join(dst_path, "bart_feature_selection_summary.json"), "w") as f:
                json.dump(combined_summary, f, indent=4)

            for file_name in expected_files:
                os.remove(os.path.join(dst_path, file_name))
            break

        print("Waiting for all jobs to complete...")
        time.sleep(30)
