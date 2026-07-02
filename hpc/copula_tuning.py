import json
import os

import numpy as np
import optuna
import pandas as pd
import src.conformal_prediction as cp
import src.metrics as met
from crepes import WrapRegressor
from crepes.martingales import SimpleJumper
from scipy.stats import spearmanr, norm, t
from src.conformal_prediction import conformalize_distribution
from src.data_loader import load_data
from sklearn.linear_model import Ridge
from pathlib import Path



os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

def draw_student_t_copula(rho, nu, size):
    # 1. Draw from Multivariate Normal (the core dependency)
    cov = [[1, rho], [rho, 1]]
    z = np.random.multivariate_normal(mean=[0, 0], cov=cov, size=size)
    
    # 2. Draw the scaling factor (Chi-Square) for the tails
    # This creates the simultaneous extreme events
    w = np.random.chisquare(df=nu, size=(size, 1))
    
    # 3. Transform to Multivariate T-space
    z = z / np.sqrt(w / nu)
    
    # 4. Convert to Uniform [0, 1] using the T-distribution CDF
    # This is the "Copula" part
    u = t.cdf(z, df=nu)
    
    return u, z


def draw_frank_copula(theta, size):
    # In the theta -> 0 limit, Frank copula converges to independence.
    if np.isclose(theta, 0.0):
        u = np.random.uniform(0.0, 1.0, size=(size, 2))
        return u, u

    u1 = np.random.uniform(0.0, 1.0, size=size)
    w = np.random.uniform(0.0, 1.0, size=size)

    # Invert the conditional CDF V | U=u for the Frank copula.
    a = np.exp(-theta * u1)
    d = np.expm1(-theta)  # exp(-theta) - 1
    denom = a - w * (a - 1.0)
    b = 1.0 + (w * d) / denom

    b = np.clip(b, np.finfo(float).tiny, None)
    u2 = -np.log(b) / theta

    u = np.column_stack([u1, u2])
    u = np.clip(u, np.finfo(float).eps, 1.0 - np.finfo(float).eps)

    # Use logit-transformed uniforms as a latent visualization space.
    z = np.log(u / (1.0 - u))
    return u, z

def _safe_name(value):
    return str(value).replace(" ", "_")


def _spearman_to_gaussian_corr(rho_s):
    # Greiner’s Relation
    # For Gaussian copulas, latent Pearson correlation is not equal to Spearman rho.
    rho = 2.0 * np.sin(np.pi * float(rho_s) / 6.0)
    return float(np.clip(rho, -0.99, 0.99))


def randomized_pit_batch(sample_matrix, y_obs_vec, rng=None, atol=1e-12):
    rng = np.random.default_rng() if rng is None else rng
    sample_matrix = np.asarray(sample_matrix)
    y_obs_vec = np.asarray(y_obs_vec).reshape(-1)

    if sample_matrix.ndim != 2:
        raise ValueError("sample_matrix must have shape (n_obs, n_samples)")
    if sample_matrix.shape[0] != len(y_obs_vec):
        raise ValueError("sample_matrix rows must match len(y_obs_vec)")

    p_less = (sample_matrix < (y_obs_vec[:, None] - atol)).mean(axis=1)
    p_equal = np.isclose(sample_matrix, y_obs_vec[:, None], atol=atol, rtol=0.0).mean(axis=1)
    return p_less + rng.uniform(size=len(y_obs_vec)) * p_equal


def get_volbids_pit_cal(zone, train_end, calibration_end, test_end, ridge_summary, c_threshold, window_size):
    X_train, X_calibration, X_validation, y_train, y_calibration, y_validation, _, y_scaler = cp.train_cal_test_date_split(
        *load_data(zone, "volbids"), train_end, calibration_end, test_end, scaled=True
    )

    model = Ridge(alpha=ridge_summary.loc[zone + " | volbids", "alpha"])
    wr = WrapRegressor(model)
    wr.fit(X_train, y_train.values)

    rng = np.random.default_rng(42)
    X_hist = X_train.iloc[-window_size:].copy()
    y_hist = y_train.iloc[-window_size:].copy()
    X_hold = X_calibration.copy()
    y_hold = y_calibration.copy()

    pit_chunks = []
    while True:
        wr.calibrate(X_hist, y_hist.values, cps=True)
        pred_cdf_cal = wr.predict_cpds(X_hold)

        p_values_cal = wr.predict_p(X_hold, y_hold.values)
        sj_cal = SimpleJumper().apply(p_values_cal)
        crossing_cal = np.where(sj_cal > c_threshold)[0]

        if crossing_cal.size == 0:
            block_end = len(X_hold)
        else:
            block_end = int(np.ceil(crossing_cal[0] / 24) * 24)
            block_end = max(1, min(block_end, len(X_hold)))

        pred_block = pred_cdf_cal[:block_end]
        y_block = y_hold.iloc[:block_end].values
        pit_block = randomized_pit_batch(pred_block, y_block, rng=rng)
        pit_chunks.append(pit_block)

        if block_end >= len(X_hold):
            break

        X_hist = pd.concat([X_hist, X_hold.iloc[:block_end]], axis=0).iloc[-window_size:]
        y_hist = pd.concat([y_hist, y_hold.iloc[:block_end]], axis=0).iloc[-window_size:]
        X_hold = X_hold.iloc[block_end:]
        y_hold = y_hold.iloc[block_end:]

    pit_values_cal = np.concatenate(pit_chunks, axis=0)
    return pd.Series(pit_values_cal, index=y_calibration.index, name="pit")


def sample_cm_with_copula(frr_interpolators, vol_cdfs, n_draws, rng, copula_samples):
    n_obs = len(frr_interpolators)
    if len(vol_cdfs) != n_obs:
        raise ValueError(f"Length mismatch: len(vol_cdfs)={len(vol_cdfs)} != len(frr_interpolators)={n_obs}")

    copula_samples = np.asarray(copula_samples)
    if copula_samples.shape != (n_draws, 2):
        raise ValueError(f"copula_samples must have shape ({n_draws}, 2), got {copula_samples.shape}")

    cm = np.zeros((n_obs, n_draws), dtype=float)

    frr_samples = np.zeros((n_obs, n_draws))
    for i, dist_func in enumerate(frr_interpolators):
        u = copula_samples[:, 0]
        frr_samples[i, :] = dist_func(u)

    vol_samples = np.zeros((n_obs, n_draws))
    v = copula_samples[:, 1]
    for i in range(n_obs):
        support = vol_cdfs[i]
        n_support = len(support)
        idx = np.minimum((v * n_support).astype(int), n_support - 1)
        vol_samples[i, :] = support[idx]

    cm = frr_samples - vol_samples
    return cm


def upper_miscoverage(y_true, samples, alpha_upper):
    upper = np.quantile(samples, 1 - alpha_upper, axis=1)
    return float(np.mean(y_true > upper))


def objective_penalty(cp_upper_miscoverage, alpha_upper):
    excess = max(0.0, float(cp_upper_miscoverage) - float(alpha_upper))
    if excess <= 0:
        return 0.0

    # Keep the constraint strict while avoiding objective blow-ups that make trials indistinguishable.
    penalty = 25.0 * np.expm1(20.0 * excess)
    return float(min(penalty, 1e6))


def build_volbids_distribution(zone, train_end, calibration_end, test_end, ridge_summary, c_threshold, window_size):
    X_train, X_calibration, X_validation, y_train, y_calibration, y_validation, _, y_scaler = cp.train_cal_test_date_split(
        *load_data(zone, "volbids"),
        train_end,
        calibration_end,
        test_end,
        scaled=True,
    )

    model = Ridge(alpha=ridge_summary.loc[zone + " | volbids", "alpha"])
    wr = WrapRegressor(model)
    wr.fit(X_train, y_train.values)

    X_cal = X_calibration.iloc[-window_size:].copy()
    y_cal = y_calibration.iloc[-window_size:].copy()
    X_val = X_validation.copy()
    y_val = y_validation.copy()

    cdfs = []
    while True:
        wr.calibrate(X_cal, y_cal.values, cps=True)
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


def build_frr_interpolators(X_calibration, X_val, y_calibration, y_val, y_scaler, preds, alphas, params, n_samples, rng):
    preds_cal = preds.loc[y_calibration.index]
    preds_val = preds.loc[y_val.index]

    interpolators = conformalize_distribution(X_cal = X_calibration,
                                                        y_cal = y_calibration,
                                                        preds_cal = preds_cal,
                                                        X_val = X_val,
                                                        preds_val = preds_val,
                                                        alphas = alphas,
                                                        use_clustering = params["use_clustering"],
                                                        clustering_method=params["clustering_method"],
                                                        n_clusters=params["n_clusters"],
                                                        asymmetric=params["asymmetric"],
                                                        weighted=params["weighted"],
                                                        blend_clusters=params["blend_clusters"],
                                                        return_type="interpolators",
                                                        input_median=False,
                                                        )

    # Keep FRR and volbids in the same physical units before constructing CM = FRR - volbids.
    inv_scaled_interpolators = []
    for dist_func in interpolators:
        def _make_inv_scaled(base_func):
            def _inv_scaled(u):
                return np.asarray(y_scaler.inverse_scale(base_func(u)), dtype=float)
            return _inv_scaled
        inv_scaled_interpolators.append(_make_inv_scaled(dist_func))

    return y_val.index, inv_scaled_interpolators


def build_alphas(n_alphas, alpha_min=0.01, alpha_max=0.98):
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
        n_clusters = trial.suggest_int("n_clusters", 1, 6, step=1)
        blend_clusters = (
            trial.suggest_categorical("blend_clusters", [True, False])
            if clustering_method in ("kmeans", "gaussian_mixture")
            else False
        )
    else:
        clustering_method = "none"
        n_clusters = 1
        blend_clusters = False

    params = {
        "use_clustering": use_clustering,
        "clustering_method": clustering_method,
        "n_clusters": n_clusters,
        "asymmetric": trial.suggest_categorical("asymmetric", [True, False]),
        "weighted": trial.suggest_categorical("weighted", [True, False]),
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

    preds = pd.read_parquet(f"predictions/bart_predictions_CP_Normal_{zone}_FRR.parquet")

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
        "preds": preds,
    }



def main():
    root = Path(os.getcwd())
    result_dir = root / "results" 
    train_end = "2025-08-31"
    calibration_end = "2025-10-31"
    test_end = "2025-12-31"

    c_threshold = 1e3
    window_size = 720
    n_samples = int(os.environ.get("N_SAMPLES", 10000))
    n_trials = int(os.environ.get("N_TRIALS", 150))

    zones = ["dk1 up", "dk1 down", "dk2 up", "dk2 down"]
    total_jobs = len(zones)
    job_idx = int(os.environ.get("LSB_JOBINDEX", os.environ.get("JOB_IDX", 3)))
    if job_idx < 1 or job_idx > total_jobs:
        raise ValueError(f"LSB_JOBINDEX/JOB_IDX must be between 1 and {total_jobs}, got {job_idx}")

    zone = zones[job_idx - 1]
    zone_seed = 1000 + (job_idx - 1)

    print(
        "Running Optuna Copula CM sampling tuning | "
        f"job_idx={job_idx}/{total_jobs} | zone={zone} | n_trials={n_trials} | n_samples={n_samples}"
    )

    ridge_summary = json.load(open("results/linear_forward_ridge.json", "r"))
    ridge_summary = pd.DataFrame(ridge_summary).T

    zone_context = prepare_zone_context(zone, train_end, calibration_end, test_end)
    vol_samples = build_volbids_distribution(
        zone,
        train_end,
        calibration_end,
        test_end,
        ridge_summary,
        c_threshold,
        window_size,
    )

    # Get PIT values for copula calibration
    file_path = result_dir / "pit_FRR_calibration_grid" / "frr_crps_optuna_10-31" / f"pit_{_safe_name(zone)}.csv"
    frr_pit_cal = pd.read_csv(file_path, index_col=0, parse_dates=True)["pit"]
    
    frr_pit_cal.index = pd.to_datetime(frr_pit_cal.index, errors="coerce")
    volbids_pit_cal = get_volbids_pit_cal(zone, train_end, calibration_end, test_end, ridge_summary, c_threshold, window_size)
    volbids_pit_cal.index = pd.to_datetime(volbids_pit_cal.index, errors="coerce")

    pit_pair = pd.concat(
        [frr_pit_cal.rename("frr"),  volbids_pit_cal.rename("volbids")],
        axis=1, join="inner", ).dropna()
    
    if len(pit_pair) < 3:
        raise ValueError(
            f"Need at least 3 aligned PIT pairs to fit copula; got {len(pit_pair)} for zone {zone}."
        )

    rho_s, p_s = spearmanr(pit_pair["frr"].values, pit_pair["volbids"].values)
    if not np.isfinite(rho_s):
        rho_s = 0.0
    rho = _spearman_to_gaussian_corr(rho_s)

    alpha_upper = 0.1 if zone.endswith("up") else 0.2

    def objective(trial):
        params = choose_frr_params(trial)
        n_alphas = trial.suggest_int("n_alphas", 3, 12, step=1)
        alphas = build_alphas(n_alphas)

        rng = np.random.default_rng(89000 + zone_seed * 10000 + trial.number)

        # Create Gaussian copula samples
        z = rng.multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]], size=n_samples)
        copula_samples = norm.cdf(z)
        copula_samples = np.clip(copula_samples, np.finfo(float).eps, 1 - np.finfo(float).eps)

        y_idx, frr_interpolators = build_frr_interpolators(
            X_calibration=zone_context["X_calibration"],
            X_val=zone_context["X_val"],
            y_calibration=zone_context["y_calibration"],
            y_val=zone_context["y_val"],
            y_scaler=zone_context["y_scaler_frr"],
            preds=zone_context["preds"],
            alphas=alphas,
            params=params,
            n_samples=n_samples,
            rng=rng,
        )

        cm_cp = sample_cm_with_copula(frr_interpolators, vol_samples, n_samples, rng, copula_samples)
        y_true = zone_context["y_cm_val"].loc[y_idx].values

        cp_crps = met.CRPS(y_true=y_true, samples=cm_cp)
        cp_upper_miscoverage = upper_miscoverage(y_true, cm_cp, alpha_upper)
        

        trial.set_user_attr("zone", zone)
        trial.set_user_attr("alpha_upper", alpha_upper)
        trial.set_user_attr("cp_upper_miscoverage", cp_upper_miscoverage)
        trial.set_user_attr("copula_rho_spearman", float(rho_s))
        trial.set_user_attr("copula_rho_gaussian", float(rho))
        
        penalty = 0.0
        if cp_upper_miscoverage > alpha_upper:
            print(f"Trial {trial.number}: Upper miscoverage {cp_upper_miscoverage:.3f} exceeds threshold {alpha_upper:.3f}, applying penalty")
            penalty = objective_penalty(cp_upper_miscoverage, alpha_upper)
        trial.set_user_attr("penalty", float(penalty))
        trial.set_user_attr("cp_crps", float(cp_crps))
        return cp_crps + penalty

    out_dir = os.path.join("results", "xFRR_copula_cluster_param_optuna")
    os.makedirs(out_dir, exist_ok=True)

    zone_name = _safe_name(zone)
    study_name = f"copula_optuna_{zone_name}"

    sampler = optuna.samplers.TPESampler(seed=8900 + job_idx)
    study = optuna.create_study(
        study_name=study_name,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
    )

    study.optimize(objective, n_trials=n_trials)

    trials_file = os.path.join(out_dir, f"{study_name}_trials.csv")
    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    trials_df.to_csv(trials_file, index=False)

    best_file = os.path.join(out_dir, f"{study_name}_best.json")
    best_payload = {
        "study_name": study_name,
        "Script name": os.path.basename(__file__),
        "zone": zone,
        "best_crps": study.best_value,
        "best_params": study.best_params,
        "Spearman rho": float(rho_s),
        "Spearman p-value": float(p_s),
        "Gaussian copula rho": float(rho),
        "best_user_attrs": study.best_trial.user_attrs,
        "n_trials_total": len(study.trials),
        "n_trials_requested": n_trials,
        "n_samples": n_samples,
        "Tuning options": {
            "use_clustering": [True, False],
            "clustering_method": ["kmeans", "gaussian_mixture", "variance"],
            "n_clusters": [1, 6],
            "asymmetric": [True, False],
            "weighted": [True, False],
            "blend_clusters": [True, False],
            "n_alphas": [3, 12],
        },
    }
    with open(best_file, "w", encoding="utf-8") as fh:
        json.dump(best_payload, fh, indent=4)

    print(f"Saved Optuna trials to {trials_file}")
    print(f"Saved best configuration to {best_file}")


if __name__ == "__main__":
    main()
