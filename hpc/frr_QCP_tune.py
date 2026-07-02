from pathlib import Path
import json
import os
import warnings

import numpy as np
import optuna
import pandas as pd
import src.conformal_prediction as cp
import src.metrics as met
from crepes.martingales import SimpleJumper
from src.conformal_prediction import conformalize_distribution
from src.data_loader import load_data
from tqdm import tqdm


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
warnings.filterwarnings(
    "ignore",
    message="Found Intel OpenMP .* and LLVM OpenMP .* loaded at the same time.*",
    category=RuntimeWarning,
)


MODEL_TYPES = ["bart", "tcn", "bayesian_ridge", "ard"]
ZONES = ["dk1 up", "dk1 down", "dk2 up", "dk2 down"]
TOTAL_JOBS = len(MODEL_TYPES) * len(ZONES)

TRAIN_END = pd.to_datetime("2025-08-31")
CALIBRATION_END = pd.to_datetime("2025-10-31")
TEST_END = pd.to_datetime("2025-12-31")

THRES = 1e3
ALPHA_MIN = 0.02
ALPHA_MAX = 0.95
N_SAMPLES = int(os.environ.get("N_SAMPLES", 10000))
N_PRED_DRAWS = int(os.environ.get("N_PRED_DRAWS", 2000))
DEFAULT_N_TRIALS = int(os.environ.get("N_TRIALS", 150))
OUTPUT_DIR = Path("results") / "frr_QCP_tuning"

ridge_summary = json.load(open("results/model_summaries/linear_forward_ridge.json", "r"))
ridge_summary = pd.DataFrame(ridge_summary).T

tcn_cp = pd.read_parquet("predictions/tcn_probabilistic_predictions_cp.parquet")
BR_cp = pd.read_parquet("predictions/bayesian_ridge/cp.parquet")
ARD_cp = pd.read_parquet("predictions/ard_regression/cp.parquet")


def _select_job(job_index):
    job_zero_based = job_index - 1
    if job_zero_based < 0 or job_zero_based >= TOTAL_JOBS:
        raise ValueError(f"job index must be between 1 and {TOTAL_JOBS}, got {job_index}")

    model_type = MODEL_TYPES[job_zero_based // len(ZONES)]
    zone = ZONES[job_zero_based % len(ZONES)]
    return model_type, zone


job_index = int(os.environ.get("LSB_JOBINDEX", os.environ.get("JOB_IDX", 1)))
model_type, zone = _select_job(job_index)
job_seed = 1000 + (job_index - 1)


def to_pvalue(y_val, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    y_val = np.asarray(y_val)
    P = [1]
    for i in range(1, len(y_val)):
        p = (np.sum(y_val[:i] < y_val[i]) + rng.uniform(0, 0.01) * np.sum(y_val[i] == y_val[:i])) / i
        P.append(p)
    return np.array(P)


def build_alphas(n_alphas):
    if n_alphas < 2:
        raise ValueError(f"n_alphas must be >= 2, got {n_alphas}")
    return np.linspace(ALPHA_MIN, ALPHA_MAX, n_alphas)


def suggest_cfg(trial):
    use_clustering = trial.suggest_categorical("use_clustering", [True, False])

    if use_clustering:
        clustering_method = trial.suggest_categorical("clustering_method", ["kmeans", "variance", "gaussian_mixture"])
        n_clusters = trial.suggest_int("n_clusters", 2, 6, step=1)
        blend_clusters = trial.suggest_categorical("blend_clusters", [True, False])
    else:
        clustering_method = None
        n_clusters = None
        blend_clusters = None

    return {
        "window_size": trial.suggest_categorical("window_size", [168, 336, 720]),
        "use_clustering": use_clustering,
        "clustering_method": clustering_method,
        "n_clusters": n_clusters,
        "asymmetric": trial.suggest_categorical("asymmetric", [True, False]),
        "weighted": trial.suggest_categorical("weighted", [True, False]),
        "blend_clusters": blend_clusters,
        "interpolator_kind": trial.suggest_categorical("interpolator_kind", ["linear", "cubic"]),
    }


def _load_prediction_frame(model_type, zone, y_calibration, y_validation, rng):
    if model_type == "bart":
        return pd.read_parquet(f"predictions/bart/bart_predictions_CP_all_Normal_{zone}_FRR.parquet")

    common_index = y_calibration.index.union(y_validation.index)

    if model_type == "tcn":
        mask = tcn_cp.columns.str.startswith(f"{zone} | FRR")
        preds = tcn_cp.loc[:, mask]
        mu, sigma = preds.iloc[:, 0], preds.iloc[:, 1]
        samples = rng.normal(loc=mu.values[:, None], scale=sigma.values[:, None], size=(len(mu), N_PRED_DRAWS))
        return pd.DataFrame(samples, index=common_index)

    if model_type == "bayesian_ridge":
        mask = BR_cp.columns.str.startswith(f"{zone} | FRR")
        preds = BR_cp.loc[:, mask]
        mu, sigma = preds.iloc[:, 0], preds.iloc[:, 1]
        samples = rng.normal(loc=mu.values[:, None], scale=sigma.values[:, None], size=(len(mu), N_PRED_DRAWS))
        return pd.DataFrame(samples, index=common_index)

    if model_type == "ard":
        mask = ARD_cp.columns.str.startswith(f"{zone} | FRR")
        preds = ARD_cp.loc[:, mask]
        mu, sigma = preds.iloc[:, 0], preds.iloc[:, 1]
        samples = rng.normal(loc=mu.values[:, None], scale=sigma.values[:, None], size=(len(mu), N_PRED_DRAWS))
        return pd.DataFrame(samples, index=common_index)

    raise ValueError(f"Unsupported model_type: {model_type}")


def prepare_job_context(model_type, zone):
    feature, target = load_data(zone, "FRR")
    _, X_calibration, X_validation, _, y_calibration, y_validation, _, y_scaler = cp.train_cal_test_date_split(
        feature,
        target,
        TRAIN_END,
        CALIBRATION_END,
        TEST_END,
        scaled=True,
    )

    rng = np.random.default_rng(job_seed)
    preds = _load_prediction_frame(model_type, zone, y_calibration, y_validation, rng)

    return {
        "X_calibration": X_calibration,
        "X_validation": X_validation,
        "y_calibration": y_calibration,
        "y_validation": y_validation,
        "y_validation_unscaled": y_scaler.inverse_scale(y_validation.values),
        "y_scaler": y_scaler,
        "preds": preds,
    }


JOB_CONTEXT = prepare_job_context(model_type, zone)


def sample_conformalized_distribution(preds, alphas, cfg, context=JOB_CONTEXT, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    conformal_cfg = {
        "use_clustering": cfg["use_clustering"],
        "clustering_method": cfg["clustering_method"],
        "n_clusters": cfg["n_clusters"],
        "asymmetric": cfg["asymmetric"],
        "weighted": cfg["weighted"],
        "blend_clusters": cfg["blend_clusters"],
        "interpolator_kind": cfg["interpolator_kind"],
    }

    X_cal = context["X_calibration"].iloc[-cfg["window_size"]:].copy()
    y_cal = context["y_calibration"].iloc[-cfg["window_size"]:].copy()
    X_val = context["X_validation"].copy()
    y_val = context["y_validation"].copy()
    y_scaler = context["y_scaler"]

    preds_cal = preds.loc[y_cal.index]
    preds_val = preds.loc[y_val.index]

    interpolators = []
    while True:
        p_values = to_pvalue(y_val, rng=rng)
        sj = SimpleJumper().apply(p_values)
        crossing = np.where(sj > THRES)[0]

        intp_local = conformalize_distribution(
            X_cal=X_cal,
            y_cal=y_cal,
            preds_cal=preds_cal,
            X_val=X_val,
            preds_val=preds_val,
            alphas=alphas,
            **conformal_cfg,
        )

        if crossing.size == 0:
            interpolators.extend(intp_local)
            break

        crossing_idx = int(np.ceil(crossing[0] / 24) * 24)
        crossing_idx = max(24, crossing_idx)

        if crossing_idx >= len(X_val):
            interpolators.extend(intp_local)
            break

        interpolators.extend(intp_local[:crossing_idx])

        # Move the first block from validation into calibration and keep a rolling window.
        X_cal = pd.concat([X_cal, X_val.iloc[:crossing_idx]], axis=0).iloc[-cfg["window_size"]:]
        y_cal = pd.concat([y_cal, y_val.iloc[:crossing_idx]], axis=0).iloc[-cfg["window_size"]:]
        X_val = X_val.iloc[crossing_idx:]
        y_val = y_val.iloc[crossing_idx:]
        preds_cal = preds.loc[y_cal.index]
        preds_val = preds.loc[y_val.index]

    frr_samples = np.zeros((len(interpolators), N_SAMPLES))
    for i, intp in enumerate(interpolators):
        u = rng.uniform(size=N_SAMPLES)
        frr_samples[i] = intp(u)

    return y_scaler.inverse_scale(frr_samples)


def safe_name(name):
    return name.replace(" ", "_").replace("|", "_").replace("/", "_")

def score_function(samples, true):
    return float(met.CRPS(y_true=np.asarray(true).reshape(-1), samples=samples))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    study_name = f"frr_QCP_tuning_{safe_name(model_type)}_{safe_name(zone)}"

    print(
        "Running Optuna FRR tuning | "
        f"job_index={job_index}/{TOTAL_JOBS} | model_type={model_type} | zone={zone} | n_trials={DEFAULT_N_TRIALS}"
    )

    def objective(trial):
        cfg = suggest_cfg(trial)
        n_alphas = trial.suggest_int("n_alphas", 5, 20, step=1)
        alphas = build_alphas(n_alphas)

        trial_rng = np.random.default_rng(job_seed + trial.number)
        samples = sample_conformalized_distribution(
            JOB_CONTEXT["preds"],
            alphas,
            cfg,
            context=JOB_CONTEXT,
            rng=trial_rng,
        )
        return score_function(samples, JOB_CONTEXT["y_validation_unscaled"])

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=DEFAULT_N_TRIALS)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    results_df = study.trials_dataframe()
    results_df.to_csv(OUTPUT_DIR / f"{study_name}.csv", index=False)

    summary = {
        "file": os.path.basename(__file__),
        "zone": zone,
        "model_type": model_type,
        "metric": "CRPS",
        "best_value": trial.value,
        "best_params": trial.params,
        "param_space": {
            "n_alphas": "int(5, 20)",
            "window_size": "cat([168, 336, 720])",
            "eta": "cat([10, 50, 100])",
            "use_clustering": "cat([True, False])",
            "clustering_method": "cat(['kmeans', 'variance', 'gaussian_mixture']) when use_clustering=True",
            "n_clusters": "int(2, 6) when use_clustering=True",
            "blend_clusters": "cat([True, False]) when use_clustering=True",
            "asymmetric": "cat([True, False])",
            "weighted": "cat([True, False])",
            "interpolator_kind": "cat(['linear', 'cubic'])",
        },
        "alpha_span": [ALPHA_MIN, ALPHA_MAX],
        "n_trials": DEFAULT_N_TRIALS,
    }

    summary_path = OUTPUT_DIR / f"summary_{study_name}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
