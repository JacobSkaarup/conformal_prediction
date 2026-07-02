import pandas as pd
from src.data_loader import load_data
import src.conformal_prediction as cp
import src.metrics as met
import optuna
from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def _safe_name(value):
    return str(value).replace(" ", "_")


def build_alphas(n_alphas, alpha_min=0.02, alpha_max=0.95):
    if n_alphas < 2:
        raise ValueError(f"n_alphas must be >= 2, got {n_alphas}")
    return np.linspace(alpha_min, alpha_max, n_alphas)


def choose_frr_params(trial):
    use_clustering = trial.suggest_categorical("use_clustering", [True, False])

    if use_clustering:
        clustering_method = trial.suggest_categorical(
            "clustering_method",
            ["kmeans", "gaussian_mixture", "variance"],
        )
        n_clusters = trial.suggest_int("n_clusters", 2, 6, step=1)
        blend_clusters = (
            trial.suggest_categorical("blend_clusters", [True, False])
            if clustering_method in ("kmeans", "gaussian_mixture")
            else False
        )
    else:
        clustering_method = "none"
        n_clusters = 1
        blend_clusters = False

    return {
        "use_clustering": use_clustering,
        "clustering_method": clustering_method,
        "n_clusters": n_clusters,
        "asymmetric": trial.suggest_categorical("asymmetric", [True, False]), # 
        "weighted": trial.suggest_categorical("weighted", [True, False]),
        "blend_clusters": blend_clusters,
    }


def main():
    train_end = "2025-08-31"
    calibration_end = "2025-10-31"
    test_end = "2025-12-31"

    n_samples = int(os.environ.get("N_SAMPLES", 10000))
    n_trials = int(os.environ.get("N_TRIALS", 50))

    zones = ["dk1 down", "dk2 down", "dk1 up", "dk2 up"]
    target = "FRR"
    total_jobs = len(zones)

    job_idx = int(os.environ.get("LSB_JOBINDEX", os.environ.get("JOB_IDX", 1)))
    if job_idx < 1 or job_idx > total_jobs:
        raise ValueError(f"LSB_JOBINDEX/JOB_IDX must be between 1 and {total_jobs}, got {job_idx}")

    zone = zones[job_idx - 1]

    print(
        "Running FRR Optuna tuning | "
        f"job_idx={job_idx}/{total_jobs} | zone={zone} | target={target} | "
        f"n_trials={n_trials} | n_samples={n_samples}"
    )

    results_folder = os.path.join(os.getcwd(), "results", "frr_crps_optuna")
    os.makedirs(results_folder, exist_ok=True)

    feature, target_series = load_data(zone, target)
    _, X_calibration, X_val, _, y_calibration, y_val, _, _ = cp.train_cal_test_date_split(
        feature,
        target_series,
        train_end,
        calibration_end,
        test_end,
        scaled=True,
    )

    preds_cp = pd.read_parquet(
        os.path.join(os.getcwd(), f"predictions/bart_predictions_CP_all_Normal_{zone}_{target}.parquet")
    )
    preds_cal = preds_cp.loc[y_calibration.index]
    preds_val = preds_cp.loc[y_val.index]

    def objective(trial):
        params = choose_frr_params(trial)
        n_alphas = trial.suggest_int("n_alphas", 3, 12, step=1)
        alphas = build_alphas(n_alphas)

        interpolators = cp.conformalize_distribution(
            X_cal=X_calibration,
            y_cal=y_calibration,
            preds_cal=preds_cal,
            X_val=X_val,
            preds_val=preds_val,
            alphas=alphas,
            use_clustering=params["use_clustering"],
            clustering_method=params["clustering_method"],
            n_clusters=params["n_clusters"],
            asymmetric=params["asymmetric"],
            weighted=params["weighted"],
            blend_clusters=params["blend_clusters"],
            return_type="interpolators",
            input_median=False,
            interpolator_kind="cubic",
        )

        rng = np.random.default_rng(120000 + job_idx * 10000 + trial.number)
        conformal_samples = np.zeros((len(y_val), n_samples), dtype=float)
        for i, dist_func in enumerate(interpolators):
            u = rng.uniform(0, 1, size=n_samples)
            conformal_samples[i, :] = dist_func(u)

        lower = np.percentile(conformal_samples, 5, axis=1)
        upper = np.percentile(conformal_samples, 95, axis=1)
        picp = met.PICP(y_val.values, lower, upper)
        mpiw = met.MPIW(y_val.values, lower, upper)
        crps = met.CRPS(y_true=y_val.values, samples=conformal_samples)

        trial.set_user_attr("zone", zone)
        trial.set_user_attr("target", target)
        trial.set_user_attr("PICP", float(picp))
        trial.set_user_attr("MPIW", float(mpiw))

        # Fail if missing coverage target
        # if picp < 0.8 if zones[job_idx - 1] in ("dk1 down", "dk2 down") else 0.9:
            # raise optuna.exceptions.TrialPruned(f"PICP {picp:.3f} below target")
        return crps

    study_name = f"frr_optuna_cubic_narrower_{_safe_name(zone)}"
    sampler = optuna.samplers.TPESampler(seed=7700 + job_idx)
    study = optuna.create_study(
        study_name=study_name,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
    )

    study.optimize(objective, n_trials=n_trials)

    trials_file = os.path.join(results_folder, f"{study_name}_trials.csv")
    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    trials_df.to_csv(trials_file, index=False)

    best_file = os.path.join(results_folder, f"{study_name}_best.json")
    best_payload = {
        "study_name": study_name,
        "Script name": os.path.basename(__file__),
        "zone": zone,
        "target": target,
        "best_crps": study.best_value,
        "best_params": study.best_params,
        "best_user_attrs": study.best_trial.user_attrs,
        "n_trials_total": len(study.trials),
        "n_trials_requested": n_trials,
        "n_samples": n_samples,
        "Tuning options": {
            "use_clustering": [True, False],
            "clustering_method": ["kmeans", "gaussian_mixture", "variance"],
            "n_clusters": [2, 6],
            "asymmetric": [True, False],
            "weighted": [True, False],
            "blend_clusters": [True, False],
            "n_alphas": [3, 12],
            "interpolator_kind": ["cubic"],
            "Alpha interval": [0.02, 0.95],
        },
    }

    with open(best_file, "w", encoding="utf-8") as fh:
        import json

        json.dump(best_payload, fh, indent=2)

    print(f"Saved Optuna trials to {trials_file}")
    print(f"Saved best configuration to {best_file}")


if __name__ == "__main__":
    main()