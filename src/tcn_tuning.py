import gc
import os
import random
from typing import Callable, Optional

import numpy as np
import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from src.tcn_pipeline import TCNForecaster


def generate_validation_split(n: int, val_share: float = 0.2):
    """Create a single chronological holdout split."""
    if not (0.0 < val_share < 1.0):
        raise ValueError("val_share must be in (0, 1).")

    split_idx = int(n * (1 - val_share))
    if split_idx < 1:
        raise ValueError("Not enough samples for the requested val_share.")
    if split_idx >= n:
        raise ValueError("Validation split would be empty.")

    return np.arange(0, split_idx), np.arange(split_idx, n)


DEFAULT_POINT_INSTABILITY_MARKERS = [
    "loss is",
    "diverging",
    "unable to find",
]

DEFAULT_PROBABILISTIC_INSTABILITY_MARKERS = [
    "loss is",
    "training is diverging",
    "unable to find an engine",
    "find was unable",
    "cuda error",
    "illegal memory access",
    "non-finite",
    "expected parameter loc",
    "constraint real",
]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def cleanup_after_fit() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def align_holdout_predictions(predictions, true_values, split_idx: int, window_size: int):
    """Align a full-series prediction output to the holdout slice used for scoring."""
    start_idx = max(split_idx - window_size, 0)
    return predictions.iloc[start_idx:], true_values.iloc[start_idx:]


def gaussian_nll(y_true, mu, sigma, eps: float = 1e-6) -> float:
    """Mean Gaussian NLL for normal predictive distributions."""
    y_true = np.asarray(y_true).reshape(-1)
    mu = np.asarray(mu).reshape(-1)
    sigma = np.asarray(sigma).reshape(-1)
    sigma = np.maximum(sigma, eps)
    nll = 0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(sigma) + ((y_true - mu) / sigma) ** 2)
    return float(np.mean(nll))


def sample_common_tcn_params(trial: optuna.Trial) -> dict:
    return {
        "window_size": trial.suggest_int("window_size", 24, 60, step=12),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128]),
        "epochs": trial.suggest_int("epochs", 45, 80, step=5),
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "n_filters": trial.suggest_categorical("n_filters", [8, 16, 32, 64]),
        "kernel_size": trial.suggest_categorical("kernel_size", [1, 2, 3]),
        "dropout": trial.suggest_float("dropout", 0.2, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True),
        "patience": trial.suggest_int("patience", 1, 8),
        "factor": trial.suggest_float("factor", 0.4, 0.7),
    }


def make_instability_checker(markers: list[str]) -> Callable[[Exception], bool]:
    markers_lower = tuple(marker.lower() for marker in markers)

    def checker(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(marker in msg for marker in markers_lower)

    return checker


def get_default_instability_checker(task: str) -> Callable[[Exception], bool]:
    if task == "probabilistic":
        return make_instability_checker(DEFAULT_PROBABILISTIC_INSTABILITY_MARKERS)
    return make_instability_checker(DEFAULT_POINT_INSTABILITY_MARKERS)


def _extract_loss(best_stats: dict) -> float:
    return float(best_stats.get("best_val_loss", min(best_stats["val_losses"])))


def run_two_phase_tuning(
    task: str,
    X_train,
    y_train,
    study_name: str,
    study_storage_path: Optional[str] = None,
    n_trials_phase_1: int = 50,
    top_k_promotion: int = 10,
    primary_seed: int = 101,
    verification_seeds: Optional[list[int]] = None,
    stability_weight: float = 0.2,
    guesses: Optional[list[dict]] = None,
    search_space_fn: Optional[Callable[[optuna.Trial], dict]] = None,
    instability_checker: Optional[Callable[[Exception], bool]] = None,
    catch_exceptions: tuple[type[Exception], ...] = (RuntimeError, ValueError),
    sampler_seed: int = 42,
    pruner_n_startup_trials: int = 8,
    pruner_n_warmup_steps: int = 12,
    pruner_interval_steps: int = 2,
    final_top_k: int = 3,
) -> tuple[dict, optuna.Study]:
    if verification_seeds is None:
        verification_seeds = [202, 303, 25, 505]
    if len(verification_seeds) < 2:
        raise ValueError("verification_seeds must include at least two seeds.")
    if search_space_fn is None:
        search_space_fn = sample_common_tcn_params
    if instability_checker is None:
        instability_checker = get_default_instability_checker(task)

    stage_2_seeds = list(verification_seeds[:2])
    stage_3_seeds = list(verification_seeds)

    def _fit_once(cfg: dict, seed: int, epoch_end_callback=None) -> float:
        set_global_seed(seed)
        try:
            forecaster = TCNForecaster(task=task, valshare=0.2, **cfg)
            forecaster.fit(X_train, y_train, epoch_end_callback=epoch_end_callback)
            return _extract_loss(forecaster.best_stats)
        finally:
            cleanup_after_fit()

    def _score_candidate(cfg: dict, initial_loss: float, seeds: list[int]) -> tuple[list[float], float, float, float]:
        losses = [float(initial_loss)]
        for seed in seeds:
            try:
                rep_loss = _fit_once(cfg, seed)
            except catch_exceptions as e:
                if instability_checker(e):
                    rep_loss = float("inf")
                else:
                    raise
            losses.append(float(rep_loss))

        mean_loss = float(np.mean(losses))
        std_loss = float(np.std(losses, ddof=0)) if np.isfinite(mean_loss) else float("inf")
        objective = mean_loss + stability_weight * std_loss
        return losses, mean_loss, std_loss, objective

    def phase_1_objective(trial: optuna.Trial) -> float:
        cfg = search_space_fn(trial)
        set_global_seed(primary_seed)

        def epoch_end_callback(epoch, val_loss):
            trial.report(float(val_loss), step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        try:
            forecaster = TCNForecaster(task=task, valshare=0.2, **cfg)
            forecaster.fit(X_train, y_train, epoch_end_callback=epoch_end_callback)
            rep_loss = _extract_loss(forecaster.best_stats)
        except catch_exceptions as e:
            if instability_checker(e):
                trial.set_user_attr("failure_reason", str(e))
                raise optuna.TrialPruned()
            raise
        finally:
            cleanup_after_fit()

        if not np.isfinite(rep_loss):
            trial.set_user_attr("failure_reason", "Non-finite loss")
            raise optuna.TrialPruned()
        return rep_loss

    storage = None
    if study_storage_path:
        storage_path = os.path.abspath(study_storage_path).replace("\\", "/")
        storage = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=TPESampler(seed=sampler_seed, multivariate=True),
        pruner=MedianPruner(
            n_startup_trials=pruner_n_startup_trials,
            n_warmup_steps=pruner_n_warmup_steps,
            interval_steps=pruner_interval_steps,
        ),
        storage=storage,
        load_if_exists=bool(storage),
    )

    if guesses and len(study.trials) == 0:
        for guess in guesses:
            study.enqueue_trial(guess)

    remaining_trials = max(n_trials_phase_1 - len(study.trials), 0)

    study.optimize(
        phase_1_objective,
        n_trials=remaining_trials,
        show_progress_bar=False,
        catch=catch_exceptions,
    )

    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    completed_trials.sort(key=lambda t: t.value)
    top_k_trials = completed_trials[:top_k_promotion]

    stage_2_results = []
    for rank, trial in enumerate(top_k_trials):
        losses, mean_loss, std_loss, objective = _score_candidate(trial.params, trial.value, stage_2_seeds)
        stage_2_results.append(
            {
                "trial": trial,
                "phase_1_rank": rank + 1,
                "losses": losses,
                "mean_loss": mean_loss,
                "std_loss": std_loss,
                "objective": objective,
            }
        )

    stage_2_results.sort(key=lambda item: item["objective"])
    top_stage_3_candidates = stage_2_results[:final_top_k]

    best_final_score = float("inf")
    best_payload = None

    for stage_2_rank, item in enumerate(top_stage_3_candidates):
        trial = item["trial"]
        losses, mean_loss, std_loss, objective = _score_candidate(trial.params, trial.value, stage_3_seeds)

        if objective < best_final_score:
            best_final_score = objective
            best_payload = {
                "best_params": trial.params,
                "best_objective": objective,
                "best_mean_loss": mean_loss,
                "best_std_loss": std_loss,
                "best_repeat_losses": losses,
                "stability_weight": stability_weight,
                "n_repeats": len(stage_3_seeds) + 1,
                "phase_1_rank": item["phase_1_rank"],
                "stage_2_rank": stage_2_rank + 1,
                "stage_2_mean_loss": item["mean_loss"],
                "stage_2_std_loss": item["std_loss"],
                "stage_2_objective": item["objective"],
                "stage_2_repeat_losses": item["losses"],
                "n_completed_phase_1_trials": len(completed_trials),
                "n_promotion_candidates": len(top_k_trials),
                "n_final_candidates": len(top_stage_3_candidates),
                "promotion_seeds": stage_2_seeds,
                "verification_seeds": stage_3_seeds,
            }

    if not best_payload:
        best_payload = {
            "best_params": None,
            "best_objective": None,
            "best_mean_loss": None,
            "best_std_loss": None,
            "best_repeat_losses": None,
            "stability_weight": stability_weight,
            "n_repeats": len(stage_3_seeds) + 1,
            "phase_1_rank": None,
            "stage_2_rank": None,
            "stage_2_mean_loss": None,
            "stage_2_std_loss": None,
            "stage_2_objective": None,
            "stage_2_repeat_losses": None,
            "n_completed_phase_1_trials": len(completed_trials),
            "n_promotion_candidates": len(top_k_trials),
            "n_final_candidates": len(top_stage_3_candidates),
            "promotion_seeds": stage_2_seeds,
            "verification_seeds": stage_3_seeds,
            "message": "All top K trials failed during multi-seed verification.",
        }

    return best_payload, study