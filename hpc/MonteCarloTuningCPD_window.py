import json
import os
import numpy as np
import optuna
import pandas as pd
import src.conformal_prediction as cp
import src.metrics as met
from crepes import WrapRegressor
from crepes.martingales import SimpleJumper
from crepes.extras import DifficultyEstimator
from sklearn.linear_model import Ridge

from src.conformal_prediction import CPD_clustered
from src.data_loader import load_data


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
N_SAMPLES = 10000

def _safe_name(value):
    return str(value).replace(" ", "_")

def sample_cm_distributions(frr_samples, vol_samples, n_draws, rng):
    frr_sample = rng.choice(frr_samples, size=N_SAMPLES, replace=True, axis=1)
    vol_sample = rng.choice(vol_samples, size=N_SAMPLES, replace=True, axis=1)
    cm = frr_sample - vol_sample
    return cm


def upper_miscoverage(y_true, samples, alpha_upper):
    upper = np.quantile(samples, 1 - alpha_upper, axis=1)
    return float(np.mean(y_true > upper))


def build_volbids_distribution(zone, train_end, calibration_end, test_end, ridge_summary, c_threshold, window_size):
    X_train, X_calibration, X_validation, y_train, y_calibration, y_validation, _, y_scaler = cp.train_cal_test_date_split(
        *load_data(zone, "volbids"),
        train_end,
        calibration_end,
        test_end,
        scaled=True,
    )

    # Get zone specific model and features from summary
    zone_summary = ridge_summary[zone + " | volbids"]
    model = Ridge(**zone_summary["optimal_params"])
    feats = zone_summary["optimal_features"]

    wr = WrapRegressor(model)
    wr.fit(X_train.loc[:, feats], y_train.values)
    de = DifficultyEstimator()
    de.fit(X_train.loc[:, feats], y=y_train.values, k=25)

    # Cut calibration to window size for better martingale behavior
    X_cal = X_calibration.iloc[-window_size:].loc[:, feats].copy()
    y_cal = y_calibration.iloc[-window_size:].copy()
    X_val = X_validation.loc[:, feats].copy()
    y_val = y_validation.loc[:].copy()

    cdfs = []
    while True:
        wr.calibrate(X_cal, y_cal.values, cps=True, de=de)
        pred_cdf = wr.predict_cpds(X_val)

        p_values = wr.predict_p(X_val, y_val.values)
        sj = SimpleJumper().apply(p_values)
        crossing = np.where(sj > c_threshold)[0]

        if crossing.size == 0:
            cdfs.append(pred_cdf)
            break

        crossing_idx = int(np.ceil(crossing[0] / 24) * 24)
        if crossing_idx >= len(X_val):
            cdfs.append(pred_cdf)
            break

        cdfs.append(pred_cdf[:crossing_idx])
        X_cal = pd.concat([X_cal, X_val.iloc[:crossing_idx]], axis=0).iloc[-window_size:]
        y_cal = pd.concat([y_cal, y_val.iloc[:crossing_idx]], axis=0).iloc[-window_size:]
        X_val = X_val.iloc[crossing_idx:]
        y_val = y_val.iloc[crossing_idx:]

    cdf = np.concatenate(cdfs, axis=0)
    return y_scaler.inverse_scale(cdf)


def build_frr_distributions(X_calibration, X_val, y_calibration, y_val, y_scaler, preds_cp, params, n_samples, rng, window_size):
    y_eval = np.linspace(-4, 4, 250)
    preds_cal = preds_cp.loc[y_calibration.index]
    preds_val = preds_cp.loc[y_val.index]
    cpd = CPD_clustered(**params)
    cpd.calibrate(X_calibration.iloc[-window_size:], y_calibration.iloc[-window_size:], preds_cal.iloc[-window_size:], X_val)
    cpd.predict_conformal_cdf_batch(X_val, y_eval, preds_val)
    conformal_samples = cpd.sample_from_conformal_cdfs(y_eval, y_scaler, n_samples=n_samples)

    return y_val.index, conformal_samples



def choose_frr_params(trial):
    clustering_method = trial.suggest_categorical(
            "clustering_method",
            ["kmeans", "gaussian_mixture", "variance", None],
        )
    if clustering_method is None:
        n_clusters = 1
        blend_clusters = False
    else:
        n_clusters = trial.suggest_int("n_clusters", 2, 5, step=1)
        blend_clusters = (
            trial.suggest_categorical("blend_clusters", [True, False])
            if clustering_method in ("kmeans", "gaussian_mixture")
            else False
        )
    weighted = trial.suggest_categorical("weighted", [True, False])

   
    params = {
        "clustering_method": clustering_method,
        "n_clusters": n_clusters,
        "weighted": weighted,
        "blend_clusters": blend_clusters,
    }
    return params


def prepare_zone_context(zone, train_end, calibration_end, test_end):
    feature_frr, target_frr = load_data(zone, "FRR")
    _, X_calibration, X_val, _, y_calibration, y_val, _, y_scaler_frr = cp.train_cal_test_date_split(
        feature_frr,
        target_frr,
        train_end,
        calibration_end,
        test_end,
        scaled=True,
    )

    preds_cp = pd.read_parquet(f"predictions/bart/bart_predictions_CP_all_Normal_{zone}_FRR.parquet") # Using all features and normal distribution for CP predictions to have more room for improvement with tuning

    _, _, _, _, _, y_cm_val, _, _ = cp.train_cal_test_date_split(
        *load_data(zone, "CM"),
        train_end,
        calibration_end,
        test_end,
        scaled=False,
    )

    return {
        "X_calibration": X_calibration,
        "X_val": X_val,
        "y_calibration": y_calibration,
        "y_val": y_val,
        "y_cm_val": y_cm_val,
        "y_scaler_frr": y_scaler_frr,
        "preds_cp": preds_cp,
    }

def main():
    train_end = "2025-08-31"
    calibration_end = "2025-10-31"
    test_end = "2025-12-31"

    c_threshold = 1e3
    n_samples = int(os.environ.get("N_SAMPLES", 10000))
    n_trials = int(os.environ.get("N_TRIALS", 250))

    zones = ["dk1 up", "dk1 down", "dk2 up", "dk2 down"]
    total_jobs = len(zones)
    job_idx = int(os.environ.get("LSB_JOBINDEX", os.environ.get("JOB_IDX", 1)))
    if job_idx < 1 or job_idx > total_jobs:
        raise ValueError(f"LSB_JOBINDEX/JOB_IDX must be between 1 and {total_jobs}, got {job_idx}")

    zone = zones[job_idx - 1]
    zone_seed = 1000 + (job_idx - 1)

    print(
        "Running Optuna Monte Carlo tuning | "
        f"job_idx={job_idx}/{total_jobs} | zone={zone} | n_trials={n_trials} | n_samples={n_samples}"
    )

    with open("results/model_summaries/linear_ridge_forward_summary.json", "r") as f:
        ridge_summary = json.load(f)


    zone_context = prepare_zone_context(zone, train_end, calibration_end, test_end)
    


    def objective(trial):
        params = choose_frr_params(trial)
        rng = np.random.default_rng(89000 + zone_seed * 10000 + trial.number)
        job_idx = int(os.environ.get("LSB_JOBINDEX", os.environ.get("JOB_IDX", 1)))
        zone = zones[job_idx - 1]
        alpha_upper = 0.1 if zone.endswith("up") else 0.2

        
        vol_samples = build_volbids_distribution(
            zone,
            train_end,
            calibration_end,
            test_end,
            ridge_summary,
            c_threshold,
            window_size = trial.suggest_categorical("vol_window", [168, 336, 720, 1440])
        )

        y_idx, cp_frr = build_frr_distributions(
            X_calibration=zone_context["X_calibration"],
            X_val=zone_context["X_val"],
            y_calibration=zone_context["y_calibration"],
            y_val=zone_context["y_val"],
            y_scaler=zone_context["y_scaler_frr"],
            preds_cp=zone_context["preds_cp"],
            params=params,
            n_samples=n_samples,
            rng=rng,
            window_size=trial.suggest_categorical("frr_window", [168, 336, 720, 1440]),
        )

        

        cm_cp = sample_cm_distributions(cp_frr, vol_samples, n_samples, rng)
        y_true = zone_context["y_cm_val"].loc[y_idx].values

        cp_crps = met.CRPS(y_true=y_true, samples=cm_cp)
        cp_upper_miscoverage = upper_miscoverage(y_true, cm_cp, alpha_upper)
        

        trial.set_user_attr("zone", zone)
        trial.set_user_attr("alpha_upper", alpha_upper)
        trial.set_user_attr("cp_upper_miscoverage", cp_upper_miscoverage)
        trial.set_user_attr("cp_crps", cp_crps)

        penalty = 0
        if cp_upper_miscoverage > alpha_upper:
            print(f"Trial {trial.number}: Upper miscoverage {cp_upper_miscoverage:.3f} exceeds threshold {alpha_upper:.3f}, applying penalty")
            penalty = 1000*(cp_upper_miscoverage - alpha_upper)
        return cp_crps + penalty

    out_dir = os.path.join("results", "xFRR_CPD_clusters_bart_fixedQ_windows")
    os.makedirs(out_dir, exist_ok=True)

    zone_name = _safe_name(zone)
    study_name = f"mc_optuna_{zone_name}"
    

    sampler = optuna.samplers.TPESampler(seed=8900 + job_idx)
    study = optuna.create_study(
        study_name=study_name,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
    )

    study.optimize(objective, n_trials=n_trials)
    best_file = os.path.join(out_dir, f"{study_name}_best.json")
    best_payload = {
        "study_name": study_name,
        "Script name": os.path.basename(__file__),
        "zone": zone,
        "best_crps": study.best_trial.user_attrs["cp_crps"],
        "best_params": study.best_params,
        "best_user_attrs": study.best_trial.user_attrs,
        "n_trials_total": len(study.trials),
        "n_trials_requested": n_trials,
        "n_samples": n_samples,
        "Tuning options": {
            "clustering_method": ["kmeans", "gaussian_mixture", "variance", None],
            "n_clusters": [2, 5],
            "weighted": [True, False],
            "blend_clusters": [True, False],
            "vol_window": [168, 336, 720, 1440],
            "frr_window": [168, 336, 720, 1440],
        },
    }
    with open(best_file, "w", encoding="utf-8") as fh:
        json.dump(best_payload, fh, indent=2)

    print(f"Saved best configuration to {best_file}")


if __name__ == "__main__":
    main()