import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import src.conformal_prediction as cp
from src.data_loader import load_data
from src.conformal_prediction import conformalize_distribution


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def _safe_name(value: str) -> str:
    return str(value).replace(" ", "_")


def randomized_pit_from_samples(samples, y_obs, rng=None, atol=1e-12):
    rng = np.random.default_rng() if rng is None else rng
    s = np.asarray(samples)

    p_less = np.mean(s < y_obs - atol)
    p_equal = np.mean(np.isclose(s, y_obs, atol=atol, rtol=0.0))
    return p_less + rng.uniform() * p_equal


def _resolve_prediction_path(root: Path, zone: str) -> Path:
    candidates = [
        root / f"predictions/bart_predictions_CP_Normal_{zone}_FRR.parquet",
        root / f"predictions/bart_predictions_CP_all_Normal_{zone}_FRR.parquet",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find FRR CP predictions. Checked: "
        + ", ".join(str(p) for p in candidates)
    )


def _resolve_params_path(root: Path, param_key: str, zone: str) -> Path:
    param_sources = {
        # "frr_crps_optuna": root / f"results/frr_crps_optuna/frr_optuna_{_safe_name(zone)}_best.json",
        # "copula_cluster_param_optuna": root
        # / f"results/xFRR_copula_cluster_param_optuna/copula_optuna_{_safe_name(zone)}_best.json",
        # "MC_cluster_param_optuna": root
        # / f"results/xFRR_MC_cluster_param_optuna/mc_optuna_{_safe_name(zone)}_best.json",
        # "uncluster": root / f"results/frr_crps_optuna/frr_optuna_uncluster_{_safe_name(zone)}_best.json",
        "narrow": root / f"results/frr_crps_optuna/frr_optuna_cubic_narrower_{_safe_name(zone)}_best.json",
    }

    try:
        params_path = param_sources[param_key]
    except KeyError as exc:
        raise ValueError(f"Unknown parameter source: {param_key}") from exc

    if not params_path.exists():
        raise FileNotFoundError(f"Missing tuned FRR params file: {params_path}")

    return params_path


def main():
    root = Path(os.getcwd())

    train_end = "2025-08-31"
    test_end = "2025-12-31"

    zones = ["dk1 up", "dk1 down", "dk2 up", "dk2 down"]
    parameter_sources = [
        # "frr_crps_optuna",
        # "copula_cluster_param_optuna",
        # "MC_cluster_param_optuna",
        # "uncluster",
        "narrow"
    ]
    calibration_end_dates = [
        ("2025-10-31", "10-31"),
        ("2025-11-07", "11-07"),
        ("2025-11-14", "11-14"),
        ("2025-11-21", "11-21"),
        ("2025-11-28", "11-28"),
    ]
    total_jobs = len(zones) * len(parameter_sources) * len(calibration_end_dates)

    job_idx = int(os.environ.get("LSB_JOBINDEX", os.environ.get("JOB_IDX", 1)))
    if job_idx < 1 or job_idx > total_jobs:
        raise ValueError(f"LSB_JOBINDEX/JOB_IDX must be between 1 and {total_jobs}, got {job_idx}")

    job_zero_based = job_idx - 1
    zone_idx = job_zero_based % len(zones)
    param_idx = (job_zero_based // len(zones)) % len(parameter_sources)
    calibration_idx = job_zero_based // (len(zones) * len(parameter_sources))

    zone = zones[zone_idx]
    param_key = parameter_sources[param_idx]
    calibration_end, calibration_tag = calibration_end_dates[calibration_idx]
    rng_seed = int(os.environ.get("RANDOM_SEED", 42))
    n_samples_pit = int(os.environ.get("N_SAMPLES_PIT", 4000))

    print(
        "Running FRR PIT calibration | "
        f"job_idx={job_idx}/{total_jobs} | zone={zone} | params={param_key} | "
        f"calibration_end={calibration_end} | "
        f"n_samples_pit={n_samples_pit} | seed={rng_seed}"
    )

    # Load split data.
    feature, target = load_data(zone, "FRR")
    _, X_calibration, _, _, y_calibration, _, _, _ = cp.train_cal_test_date_split(
        feature,
        target,
        train_end,
        calibration_end,
        test_end,
        scaled=True,
    )

    # Load precomputed FRR predictive samples and tuned conformal params.
    pred_path = _resolve_prediction_path(root, zone)
    preds_cp = pd.read_parquet(pred_path)
    preds_cal = preds_cp.loc[y_calibration.index]

    params_path = _resolve_params_path(root, param_key, zone)

    with open(params_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    zone_cfg = dict(cfg["best_params"])
    n_alpha = int(zone_cfg.pop("n_alphas"))
    # alphas = np.linspace(0.01, 0.98, n_alpha)
    alphas = np.linspace(0.05, 0.95, n_alpha) # Narrow version

    # LOO PIT computation on calibration period.
    rng = np.random.default_rng(rng_seed)
    n_cal = len(y_calibration)
    pit_values = np.zeros(n_cal, dtype=float)

    for i in tqdm(range(n_cal), desc=f"LOO PIT {zone}"):
        mask = np.ones(n_cal, dtype=bool)
        mask[i] = False

        X_cal_loo = X_calibration.iloc[mask]
        y_cal_loo = y_calibration.iloc[mask]
        preds_cal_loo = preds_cal.iloc[mask]

        X_target = X_calibration.iloc[[i]]
        preds_target = preds_cal.iloc[[i]]

        interpolator = conformalize_distribution(
            X_cal=X_cal_loo,
            y_cal=y_cal_loo,
            preds_cal=preds_cal_loo,
            X_val=X_target,
            preds_val=preds_target,
            alphas=alphas,
            return_type="interpolators",
            **zone_cfg,
            interpolator_kind = "cubic",
        )[0]

        u = rng.uniform(0.0, 1.0, size=n_samples_pit)
        sample_i = interpolator(u)
        y_obs_i = float(y_calibration.iloc[i])
        pit_values[i] = randomized_pit_from_samples(sample_i, y_obs_i, rng=rng)

    out_dir = root / "results" / "pit_FRR_calibration_grid" / f"{param_key}_{calibration_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_base = f"pit_{_safe_name(zone)}"
    out_csv = out_dir / f"{out_base}.csv"
    out_json = out_dir / f"{out_base}.json"

    pit_df = pd.DataFrame({"pit": pit_values}, index=y_calibration.index)
    pit_df.index.name = "timestamp"
    pit_df.to_csv(out_csv, index=True)

    summary = {
        "zone": zone,
        "Script name": os.path.basename(__file__),
        "param_key": param_key,
        "calibration_end": calibration_end,
        "calibration_tag": calibration_tag,
        "n_calibration": int(n_cal),
        "n_samples_pit": int(n_samples_pit),
        "rng_seed": int(rng_seed),
        "prediction_path": str(pred_path),
        "params_path": str(params_path),
        "output_csv": str(out_csv),
        "pit_mean": float(np.mean(pit_values)),
        "pit_std": float(np.std(pit_values)),
    }

    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Saved PIT values to {out_csv}")
    print(f"Saved summary to {out_json}")


if __name__ == "__main__":
    main()
