import numpy as np
from sklearn.linear_model import ARDRegression, Lasso, Ridge, lasso_path
from sklearn.neighbors import KNeighborsRegressor
import src.metrics as met
import optuna
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import warnings
from optuna.exceptions import ExperimentalWarning
warnings.filterwarnings(
    "ignore",
    message=r".*Argument ``multivariate`` is an experimental feature.*",
    category=ExperimentalWarning,
)


def _pinball_loss(y_true, y_pred, alpha):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    err = y_true - y_pred
    return float(np.mean(np.maximum(alpha * err, (alpha - 1.0) * err)))


def _validation_score(y_true, y_pred, objective, quantile_alpha):
    if objective == "quantile":
        return _pinball_loss(y_true, y_pred, quantile_alpha)
    return met.MSE(y_true, y_pred)


def _gaussian_nll(y_true, mu, sigma, eps=1e-12):
    y_true = np.asarray(y_true).reshape(-1)
    mu = np.asarray(mu).reshape(-1)
    sigma = np.asarray(sigma).reshape(-1)
    sigma = np.clip(sigma, eps, None)
    var = sigma**2
    return float(0.5 * np.mean(np.log(2.0 * np.pi * var) + ((y_true - mu) ** 2) / var))

 
def _generate_validation_split(n, val_share=0.2):
    """Create a single chronological train/validation split."""
    if not (0.0 < val_share < 1.0):
        raise ValueError("val_share must be in (0, 1).")

    split_idx = int(n * (1.0 - val_share))
    if split_idx < 1 or split_idx >= n:
        raise ValueError("Not enough samples for the requested validation split.")

    return np.arange(0, split_idx), np.arange(split_idx, n)

############# LINEAR TUNER #############

class TimeSeriesRidgeTuner:
    """
    An optimized Forward and backward Feature Selection and Hyperparameter Tuner 
    designed specifically for time-series data using Ridge Regression.
    """
    def __init__(self, X, y, alpha_grid=np.logspace(-3, 4, 50), val_share=0.2):
        """
        Args:
            X (pd.DataFrame): Chronologically sorted training features.
            y (pd.Series or np.ndarray): Target variable matching X's timeline.
            alpha_grid (np.ndarray): Array of Ridge regularization penalties (alpha).
            val_share (float): Proportion of data dedicated to validation.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame to maintain feature naming context.")
            
        self.X = X
        self.y = np.asarray(y).flatten()
        self.feature_names = list(X.columns)
        self.alpha_grid = np.asarray(alpha_grid)
        self.val_share = val_share

        train_idx, val_idx = _generate_validation_split(len(self.X), val_share=self.val_share)
        self.X_train = self.X.iloc[train_idx]
        self.y_train = self.y[train_idx]
        self.X_val = self.X.iloc[val_idx]
        self.y_val = self.y[val_idx]
     

    def _evaluate_alpha_path_svd(self, X_tr, y_tr, X_val, y_val):
        """
        Analytically evaluates the entire alpha grid simultaneously using the SVD of X.
        Avoids iterative fitting loop over alpha.
        """
        # Center data to avoid regularizing the intercept term
        X_tr_mean = X_tr.mean(axis=0)
        y_tr_mean = y_tr.mean()
        
        X_tr_c = X_tr - X_tr_mean
        y_tr_c = y_tr - y_tr_mean
        X_val_c = X_val - X_tr_mean
        
        # Compute economy SVD: X = U * diag(S) * Vt
        U, S, Vt = np.linalg.svd(X_tr_c, full_matrices=False)
        
        # Pre-project target onto left singular vectors: U^T @ y
        Ut_y = U.T @ y_tr_c
        
        alphas_mses = np.empty(len(self.alpha_grid))
        
        # Vectorized/Analytical Ridge solution path loop
        for idx, alpha in enumerate(self.alpha_grid):
            # d_i = s_i / (s_i^2 + \alpha)
            d = S / (S**2 + alpha)
            # \beta = V @ diag(d) @ U^T @ y
            beta = Vt.T @ (d * Ut_y)
            
            preds = (X_val_c @ beta) + y_tr_mean
            alphas_mses[idx] = np.mean((y_val - preds) ** 2)
            
        return alphas_mses

    def _evaluate_subset(self, feature_subset):
        """Helper to compute the optimal alpha and minimum MSE for a given subset."""
        X_tr = self.X_train[feature_subset].values
        y_tr = self.y_train
        X_val = self.X_val[feature_subset].values
        y_val = self.y_val

        alphas_mses = self._evaluate_alpha_path_svd(X_tr, y_tr, X_val, y_val)
        optimal_alpha_idx = np.argmin(alphas_mses)

        return alphas_mses[optimal_alpha_idx], self.alpha_grid[optimal_alpha_idx]


    def forward_selection(self, tolerance=0.02):
        """
        Executes a decoupled forward selection algorithm.
        
        Args:
            tolerance (float): Minimum relative reduction in val MSE required to continue selection.
        """
        selected_features = []
        prev_best_val_mse = float('inf')
        
        p = len(self.feature_names)
        
        for step in range(p):
            best_candidate_feature = None
            best_candidate_alpha = None
            best_candidate_mse = float('inf')
            
            # Loop over unselected features
            for feature in self.feature_names:
                if feature in selected_features:
                    continue
                    
                candidate_set = selected_features + [feature]
                min_subset_mse, associated_alpha = self._evaluate_subset(candidate_set)
                
                # Track the best performing structural feature addition
                if min_subset_mse < best_candidate_mse:
                    best_candidate_mse = min_subset_mse
                    best_candidate_feature = feature
                    best_candidate_alpha = associated_alpha
            
            # Evaluate stopping condition based on structural step improvement
            if best_candidate_feature is not None:
                if prev_best_val_mse == float('inf'):
                    rel_improvement = 1.0
                else:
                    rel_improvement = (prev_best_val_mse - best_candidate_mse) / prev_best_val_mse
                
                if rel_improvement < tolerance:
                    break
                    
                selected_features.append(best_candidate_feature)
                prev_best_val_mse = best_candidate_mse
                
    
            else:
                break


        return selected_features, best_candidate_alpha, best_candidate_mse
    
    
    def backward_selection(self, tolerance=0.02):
        """
        Executes a decoupled backward elimination algorithm.
        
        Args:
            tolerance (float): Threshold for stopping. If removing a feature does not improve 
                               the model (or degrades val MSE by more than this relative share), 
                               elimination terminates.
        """
        selected_features = list(self.feature_names)
        
        # Establish baseline performance with all features included
        prev_best_val_mse, initial_alpha = self._evaluate_subset(selected_features)
        
        
        p = len(self.feature_names)
        
        for step in range(p - 1):  # Retain at least one feature
            best_candidate_feature = None
            best_candidate_alpha = None
            best_candidate_mse = float('inf')
            
            # Evaluate the impact of removing each current feature
            for feature in selected_features:
                candidate_set = [f for f in selected_features if f != feature]
                
                min_subset_mse, associated_alpha = self._evaluate_subset(candidate_set)
                
                # Identify the feature whose removal yields the lowest overall error
                if min_subset_mse < best_candidate_mse:
                    best_candidate_mse = min_subset_mse
                    best_candidate_feature = feature
                    best_candidate_alpha = associated_alpha
            
            # Check if removing the feature yields a valid relative improvement.
            # In backward selection, a positive rel_improvement means MSE decreased after dropping the feature.
            # If it increases (negative) or fails to beat the tolerance threshold, we stop dropping.
            rel_improvement = (prev_best_val_mse - best_candidate_mse) / prev_best_val_mse
            
            if rel_improvement < tolerance:
                break
                
            selected_features.remove(best_candidate_feature)
            prev_best_val_mse = best_candidate_mse
            
        return selected_features, best_candidate_alpha, best_candidate_mse


class TimeSeriesLassoTuner:
    """
    An ultra-fast time-series hyperparameter tuner and feature selector 
    that leverages the native geometric sparsity and warm-starts of the Lasso Path.
    """
    def __init__(self, X, y, alpha_grid=np.logspace(-4, 1, 50), val_share=0.2):
        """
        Args:
            X (pd.DataFrame): Feature matrix.
            y (array-like): Target values.
            alpha_grid (array-like): Grid of alpha values for the Lasso path.
            val_share (float): Proportion of data to use for validation.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame to track feature names.")
            
        self.X = X
        self.y = np.asarray(y).flatten()
        self.feature_names = list(X.columns)
        self.alpha_grid = np.sort(np.asarray(alpha_grid))[::-1] # Lasso path requires descending alphas
        self.val_share = val_share

        train_idx, val_idx = _generate_validation_split(len(self.X), val_share=self.val_share)
        self.X_train = self.X.iloc[train_idx]
        self.y_train = self.y[train_idx]
        self.X_val = self.X.iloc[val_idx]
        self.y_val = self.y[val_idx]

    def fit_and_tune(self):
        """
        Tunes alpha on the holdout split using coordinate descent paths,
        then fits a final model to extract selected features.

        Returns:
            selected_features (list): List of features with non-zero coefficients at the optimal alpha.
            optimal_alpha (float): The alpha value that minimizes validation MSE.
            best_mse (float): The minimum validation MSE achieved.
            feature_coefficients (dict): Mapping of selected features to their coefficients for interpretability.
        """
        holdout_mses = np.zeros(len(self.alpha_grid))

        X_tr = self.X_train.values
        y_tr = self.y_train
        X_val = self.X_val.values
        y_val = self.y_val

        # Computes the entire alpha path simultaneously using highly optimized warm-starts
        _, coef_path, _ = lasso_path(X_tr, y_tr, alphas=self.alpha_grid, max_iter=50000)

        # Calculate validation MSE for each alpha model along the path
        for idx in range(len(self.alpha_grid)):
            w = coef_path[:, idx]

            # Reconstruct the implicit intercept for centered data inside lasso_path
            intercept = np.mean(y_tr) - np.mean(X_tr, axis=0) @ w
            preds = (X_val @ w) + intercept
            holdout_mses[idx] = np.mean((y_val - preds) ** 2)

        best_alpha_idx = np.argmin(holdout_mses)
        optimal_alpha = self.alpha_grid[best_alpha_idx]
        best_holdout_mse = holdout_mses[best_alpha_idx]
        
        # Fit final model on the training split using the optimal alpha
        final_model = Lasso(alpha=optimal_alpha, max_iter=100000)
        final_model.fit(self.X_train, self.y_train)
        
        # Native feature selection: isolate features with non-zero weights
        non_zero_indices = np.where(final_model.coef_ != 0)[0]
        selected_features = [self.feature_names[i] for i in non_zero_indices]
        
        return (
            selected_features,
            optimal_alpha,
            best_holdout_mse,
            dict(zip(selected_features, final_model.coef_[non_zero_indices])) # Return selected features with their corresponding coefficients for interpretability
        )


class TimeSeriesARDTuner:
    """Time-series tuner for ARDRegression using a single holdout split and coefficient-based feature selection."""

    def __init__(self, X, y, val_share=0.2, coef_threshold=1.0e-8, seed=42, suggested_params=None):
        """
        Args:
            X (pd.DataFrame): Feature matrix.
            y (array-like): Target values.
            val_share (float): Proportion of data to use for validation.
            coef_threshold (float): Threshold for coefficient-based feature selection.
            seed (int): Random seed for reproducibility.
            suggested_params (dict): Dictionary of Optuna suggested hyperparameters.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame to preserve feature names.")

        self.X = X
        self.y = np.asarray(y).reshape(-1)
        self.feature_names = list(X.columns)
        self.val_share = val_share
        self.coef_threshold = coef_threshold
        self.seed = seed
        self.suggested_params = suggested_params

        train_idx, val_idx = _generate_validation_split(len(self.X), val_share=self.val_share)
        self.X_train = self.X.iloc[train_idx]
        self.y_train = self.y[train_idx]
        self.X_val = self.X.iloc[val_idx]
        self.y_val = self.y[val_idx]

    def _suggest_params(self, trial):
        if self.suggested_params:
            return self.suggested_params

        return {
            "max_iter": trial.suggest_int("max_iter", 200, 4000),
            "tol": trial.suggest_float("tol", 1e-6, 1e-2, log=True),
            "alpha_1": trial.suggest_float("alpha_1", 1e-8, 1e-2, log=True),
            "alpha_2": trial.suggest_float("alpha_2", 1e-8, 1e-2, log=True),
            "lambda_1": trial.suggest_float("lambda_1", 1e-8, 1e-2, log=True),
            "lambda_2": trial.suggest_float("lambda_2", 1e-8, 1e-2, log=True),
            "threshold_lambda": trial.suggest_float("threshold_lambda", 1e2, 1e6, log=True),
            "compute_score": False,
            "verbose": False,
        }

    def _evaluate_holdout_nll(self, feature_subset, params):
        feature_subset = list(feature_subset)

        X_tr = self.X_train[feature_subset]
        y_tr = self.y_train
        X_val = self.X_val[feature_subset]
        y_val = self.y_val

        model = ARDRegression(**params)
        try:
            model.fit(X_tr, y_tr)
            y_pred, y_std = model.predict(X_val, return_std=True)
            score = _gaussian_nll(y_val, y_pred, y_std)
        except Exception:
            return float("inf")

        if not np.isfinite(score):
            return float("inf")

        return float(score)

    def fit_and_tune(self, n_trials=120, progress_bar=True):
        """
         Tunes ARDRegression hyperparameters using Optuna and selects features based on coefficient magnitude.
        Args:
            n_trials (int): Number of Optuna trials for hyperparameter optimization.
            progress_bar (bool): Whether to display the Optuna progress bar.
         """
        sampler = optuna.samplers.TPESampler(seed=self.seed, multivariate=True)
        study = optuna.create_study(direction="minimize", sampler=sampler, study_name=f"ARD_Tuning_{self.seed}")

        def objective(trial):
            params = self._suggest_params(trial)
            return self._evaluate_holdout_nll(self.feature_names, params)

        study.optimize(objective, n_trials=n_trials, show_progress_bar=progress_bar)

        best_params = dict(study.best_params)
        best_params.update({"compute_score": False, "verbose": False})

        full_model = ARDRegression(**best_params)
        full_model.fit(self.X_train, self.y_train)

        coef_mask = np.abs(full_model.coef_) > self.coef_threshold
        selected_features = [
            feature for feature, keep in zip(self.feature_names, coef_mask) if keep
        ]
        if not selected_features:
            selected_features = [self.feature_names[int(np.argmax(np.abs(full_model.coef_)))]]

        selected_coefficients = {
            feature: float(full_model.coef_[self.feature_names.index(feature)])
            for feature in selected_features
        }

        selected_holdout_nll = self._evaluate_holdout_nll(selected_features, best_params)

        return {
            "best_params": best_params,
            "holdout_nll": float(selected_holdout_nll),
            "selected_features": selected_features,
            "coefficients": selected_coefficients,
        }

############# KNN TUNER #############

class knn_tuner:
    """Class for tuning KNN regression model using a single chronological validation split."""

    def __init__(self, X_train, y_train, val_share=0.2, k_grid=[1, 3, 5, 7, 9, 11, 13, 15, 20, 25, 30, 35, 40, 60, 80, 100]):
        """Initialize the tuner with a single chronological holdout split.
        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target variable.
            val_share (float): Proportion of training data to use for validation.
            k_grid (list): List of K values to evaluate for KNN regression.
        """
        n = len(X_train)
        split_idx = int(n * (1 - val_share))
        if split_idx < 1 or split_idx >= n:
            raise ValueError("val_share leaves no data for train or validation.")

        self.X_train, self.X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
        y_array = np.asarray(y_train).flatten()
        self.y_train, self.y_val = y_array[:split_idx], y_array[split_idx:]
        self.feature_names = list(X_train.columns)
        self.k_grid = list(k_grid)
        self.val_share = val_share

    def _evaluate_k_grid(self, candidate_features):
        mse_values = np.zeros(len(self.k_grid), dtype=float)

        X_tr = self.X_train[candidate_features]
        y_tr = self.y_train
        X_val = self.X_val[candidate_features]
        y_val = self.y_val

        for k_idx, k in enumerate(self.k_grid):
            knn = KNeighborsRegressor(n_neighbors=k)
            knn.fit(X_tr, y_tr)
            y_pred = knn.predict(X_val)
            mse_values[k_idx] = met.MSE(y_val, y_pred)

        best_k_index = np.argmin(mse_values)
        best_k = self.k_grid[best_k_index]
        return best_k, float(mse_values[best_k_index])


    def forward_selection(self, tolerance=0.02):
        """Perform forward feature selection using the holdout split to choose features and K.
        Args:
            tolerance (float): The relative improvement threshold to stop feature selection.
        Returns:
            optimal_features (list): The optimal set of features.
            optimal_k (int): The optimal K value for KNN regression.
            optimal_val_mse (float): The best validation MSE achieved with the optimal features and K.
        """
        selected_features = []
        prev_val_mse = float('inf')
        optimal_features = []
        optimal_k = None
        optimal_val_mse = float('inf')

        for i in range(len(self.feature_names)):
            best_feature = None
            best_feature_k = None
            best_feature_mse = float('inf')

            for feature in self.feature_names:
                if feature in selected_features:
                    continue

                candidate_features = selected_features + [feature]
                candidate_k, mse = self._evaluate_k_grid(candidate_features)
                if mse < best_feature_mse:
                    best_feature_mse = mse
                    best_feature = feature
                    best_feature_k = candidate_k

            if best_feature is None:
                break

            selected_features.append(best_feature)

            rel_improvement = (prev_val_mse - best_feature_mse) / prev_val_mse if prev_val_mse != 0 else 0
            optimal_features = selected_features.copy()
            optimal_k = best_feature_k
            optimal_val_mse = best_feature_mse
            if rel_improvement < tolerance:
                break
            prev_val_mse = best_feature_mse

        return optimal_features, optimal_k, optimal_val_mse

    def backward_selection(self, tolerance=0.02):
        """Perform backward feature selection using the holdout split to choose features and K.
        Args:
            tolerance (float): The relative improvement threshold to stop feature selection.
        Returns:
            optimal_features (list): The optimal set of features.
            optimal_k (int): The optimal K value for KNN regression.
            optimal_val_mse (float): The best validation MSE achieved with the optimal features and K.
        """
        prev_val_mse = float('inf')
        selected_features = list(self.feature_names)
        optimal_k, best_val_mse = self._evaluate_k_grid(selected_features)
        optimal_features = selected_features.copy()

        for i in range(len(self.feature_names)):
            best_feature = None
            iteration_best_k = optimal_k
            iteration_best_mse = float('inf')

            for feature in selected_features:
                candidate_features = [f for f in selected_features if f != feature]
                if not candidate_features:
                    continue

                candidate_k, mse = self._evaluate_k_grid(candidate_features)
                if mse < iteration_best_mse:
                    iteration_best_mse = mse
                    best_feature = feature
                    iteration_best_k = candidate_k

            if best_feature is None:
                break

            selected_features.remove(best_feature)
            optimal_k = iteration_best_k
            optimal_features = selected_features.copy()

            rel_improvement = (prev_val_mse - iteration_best_mse) / prev_val_mse if prev_val_mse != 0 else 0
            best_val_mse = iteration_best_mse
            if rel_improvement < tolerance:
                break
            prev_val_mse = iteration_best_mse


        return optimal_features, optimal_k, best_val_mse



############ Boosting tree tuner #############


class boosting_tuner_base:
    """Base class for tree-based boosting model tuners with union top-k feature selection on a single holdout split."""
    
    def __init__(self, X_train, y_train, seed=42, val_share=0.2):
        """Initialize the tuner with training data and a validation split.
        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target variable.
            val_share (float): Proportion of training data to use for validation.
            seed (int): Random seed for reproducibility.
        """
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame to preserve feature names.")

        n = len(X_train)
        split_idx = int(n * (1 - val_share))
        if split_idx < 1 or split_idx >= n:
            raise ValueError("val_share leaves no data for train or validation.")

        self.X_train, self.X_val = X_train.iloc[:split_idx].copy(), X_train.iloc[split_idx:].copy()
        y_array = np.asarray(y_train).flatten()
        self.y_train, self.y_val = y_array[:split_idx], y_array[split_idx:]
        self.feature_names = list(X_train.columns)
        self.seed = seed
        self.val_share = val_share

    def _enqueue_best_trials(self, target_study, source_studies, top_n=5):
        """Enqueue best trials from source studies to warm-start target study."""
        seen = set()
        for st in source_studies:
            completed = [t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE]
            completed = sorted(completed, key=lambda t: t.value)[:top_n]
            for t in completed:
                key = tuple(sorted(t.params.items()))
                if key in seen:
                    continue
                seen.add(key)
                target_study.enqueue_trial(t.params)

    def _topk_union_features(self, importance_df, k):
        """Extract union of top-k features by two importance metrics."""
        top_first = importance_df.nlargest(k, importance_df.columns[0]).index
        top_second = importance_df.nlargest(k, importance_df.columns[1]).index
        return top_first.union(top_second).tolist()

    def _suggest_params(self, trial, objective="mse", quantile_alpha=0.5):
        """Suggest hyperparameters for the model. Must be overridden by subclasses."""
        raise NotImplementedError

    def _tune_study(self, X_train, y_train, X_val, y_val, n_trials=50, 
                    warm_start_studies=None, warm_top_n=5, seed=42, 
                    objective="mse", quantile_alpha=0.5, progress_bar=True):
        """Tune hyperparameters using Optuna. Must be overridden by subclasses."""
        raise NotImplementedError

    def _fit_with_params(self, X_train, y_train, X_val, y_val, best_params, 
                         objective="mse", quantile_alpha=0.5):
        """Fit model with given parameters. Must be overridden by subclasses."""
        raise NotImplementedError

    def _get_importance(self, best_params):
        """Calculate feature importance. Must be overridden by subclasses."""
        raise NotImplementedError

    def tune(self, k_grid=(5, 10, 15, 20, 30, 40), n_trials_full=150, n_trials_k=80,
             warm_top_n=8, tolerance_ratio=0.01, objective="mse", quantile_alpha=0.5, progress_bar=True):
        """Perform full hyperparameter tuning with union top-k feature selection.
         Args:
            k_grid (tuple): Tuple of k values for top-k union feature selection.
            n_trials_full (int): Number of Optuna trials for tuning on all features.
            n_trials_k (int): Number of Optuna trials for tuning on each k-selected subset
            warm_top_n (int): Number of top trials to warm-start for each k iteration.
            tolerance_ratio (float): Relative tolerance for holdout score to prefer smaller feature sets.
            objective (str): Objective to optimize, either "mse" or "quantile".
            quantile_alpha (float): Alpha value for quantile regression if objective is "quantile".
            progress_bar (bool): Whether to show Optuna progress bars during tuning.

        Returns:
                dict: A dictionary containing tuning results and selected features.
        """
        if objective not in ("mse", "quantile"):
            raise ValueError("objective must be 'mse' or 'quantile'.")
        if not (0.0 < quantile_alpha < 1.0):
            raise ValueError("quantile_alpha must be in (0, 1).")
        
        # Step 1: tune on all features.
        full_study = self._tune_study(
            self.X_train, self.y_train, self.X_val, self.y_val,
            n_trials=n_trials_full,
            warm_start_studies=None,
            warm_top_n=warm_top_n,
            seed=self.seed,
            objective=objective,
            quantile_alpha=quantile_alpha,
            progress_bar=progress_bar,
        )
        
        full_params = full_study.best_params
        _, full_score = self._fit_with_params(
            self.X_train, self.y_train, self.X_val, self.y_val,
            full_params, objective=objective, quantile_alpha=quantile_alpha,
        )

        # Step 2: build ranking from importance metrics.
        importance_df = self._get_importance(full_params)

        # Step 3: for each k, use the union of the top-ranked features from both metrics, then re-tune.
        results = []
        prev_studies = [full_study]

        for k in k_grid:
            if k <= len(importance_df):
                feats = self._topk_union_features(importance_df, k)

                k_study = self._tune_study(
                    self.X_train[feats], self.y_train, self.X_val[feats], self.y_val,
                    n_trials=n_trials_k,
                    warm_start_studies=prev_studies,
                    warm_top_n=warm_top_n,
                    seed=self.seed + int(k),
                    objective=objective,
                    quantile_alpha=quantile_alpha,
                    progress_bar=progress_bar,
                )

                k_params = k_study.best_params
                _, k_score = self._fit_with_params(
                    self.X_train[feats], self.y_train, self.X_val[feats], self.y_val,
                    k_params, objective=objective, quantile_alpha=quantile_alpha,
                )

                results.append({
                    "k_union_input": k,
                    "holdout_score": k_score,
                    "features": feats,
                    "params": k_params,
                    "study": k_study,
                    "seed": self.seed + int(k),
                })

                prev_studies.append(k_study)

        if not results:
            raise ValueError("No candidate feature subsets were evaluated.")
        
        # Step 4: choose the best k based on holdout score, with tolerance for smaller feature sets when close.
        results_df = pd.DataFrame(
            [{"k_union_input": r["k_union_input"],
              "holdout_score": r["holdout_score"],
              "seed": r["seed"],
            } for r in results]
        ).sort_values("holdout_score")

        best_score = results_df.iloc[0]["holdout_score"]
        threshold = best_score * (1 + tolerance_ratio)

        candidates = [r for r in results if r["holdout_score"] <= threshold]
        chosen = (
            sorted(candidates, key=lambda r: (r["k_union_input"]))[0]
            if candidates
            else min(results, key=lambda r: r["holdout_score"])
        )

        return {
            "objective": objective,
            "quantile_alpha": quantile_alpha,
            "full_holdout_score": full_score,
            "importance_df": importance_df.sort_values(importance_df.columns[0], ascending=False),
            "k_results_df": results_df,
            "best_k_union_input": chosen["k_union_input"],
            "optimal_features": chosen["features"],
            "best_params": chosen["params"],
            "best_holdout_score": chosen["holdout_score"],
            "seed": chosen["seed"],
        }


class lgb_tuner(boosting_tuner_base):
    """Class for tuning LightGBM regression model using top-k union feature selection."""

    def _suggest_params(self, trial, objective="mse", quantile_alpha=0.5):
        params = {
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 250),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 150),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-5, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-5, 10.0, log=True),
            "n_estimators" : trial.suggest_int("n_estimators", 50, 2000, step=50),
        }
        if objective == "quantile":
            params["objective"] = "quantile"
            params["metric"] = "quantile"
            params["alpha"] = quantile_alpha
        else:
            params["objective"] = "regression"
            params["metric"] = "rmse"
        return params

    def _tune_study(self, X_train, y_train, X_val, y_val, n_trials=50,
                    warm_start_studies=None, warm_top_n=5, seed=42,
                    objective="mse", quantile_alpha=0.5, progress_bar=True):
        
        pruner = optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=20, interval_steps=5)
        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

        if warm_start_studies:
            self._enqueue_best_trials(study, warm_start_studies, top_n=warm_top_n)

        objective_mode = objective
        eval_metric = "quantile" if objective_mode == "quantile" else "rmse"

        def _optuna_pruning_callback(trial):
            def callback(env):
                for eval_result in env.evaluation_result_list:
                    data_name = eval_result[0]
                    eval_name = eval_result[1]
                    score = eval_result[2]
                    if data_name.startswith("valid") and eval_name == eval_metric:
                        trial.report(float(score), step=env.iteration)
                        if trial.should_prune():
                            raise optuna.TrialPruned(f"Trial pruned at iteration {env.iteration}.")
                        break
            return callback

        def objective_fn(trial):
            params = self._suggest_params(trial, objective=objective_mode, quantile_alpha=quantile_alpha)
            model = lgb.LGBMRegressor(random_state=self.seed, **params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(period=0),
                    _optuna_pruning_callback(trial),
                ],
            )

            preds = model.predict(X_val)
            holdout_score = _validation_score(y_val, preds, objective_mode, quantile_alpha)

            trial.set_user_attr("best_iter", int(model.best_iteration_ or 0))
            return float(holdout_score)

        study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=progress_bar)
        return study


    def _fit_with_params(self, X_train, y_train, X_val, y_val, best_params,
                         objective="mse", quantile_alpha=0.5):
        model = lgb.LGBMRegressor(random_state=self.seed, **best_params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        score = _validation_score(y_val, preds, objective, quantile_alpha)
        return model, score


    def _get_importance(self, best_params):
        gain_model = lgb.LGBMRegressor(random_state=self.seed, **best_params, importance_type="gain")
        gain_model.fit(self.X_train, self.y_train)
        gain = pd.Series(gain_model.feature_importances_, index=self.X_train.columns, name="gain")

        split_model = lgb.LGBMRegressor(random_state=self.seed, **best_params, importance_type="split")
        split_model.fit(self.X_train, self.y_train)
        split = pd.Series(split_model.feature_importances_, index=self.X_train.columns, name="split")

        imp = pd.concat([gain, split], axis=1).fillna(0.0)

        return imp


    def _topk_union_features(self, importance_df, k):
        top_gain = importance_df.nlargest(k, "gain").index
        top_split = importance_df.nlargest(k, "split").index
        return top_gain.union(top_split).tolist()


class xgb_tuner(boosting_tuner_base):
    """Class for tuning XGBoost regression model using union top-k feature selection."""

    def _suggest_params(self, trial, objective="mse", quantile_alpha=0.5):
        params = {
            "verbosity": 0,
            "booster": "gbtree",
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 14),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
            "gamma": trial.suggest_float("gamma", 1e-8, 5e-1, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-7, 1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 2000, step=50),
        }
        if objective == "quantile":
            params["objective"] = "reg:quantileerror"
            params["eval_metric"] = "quantile"
            params["quantile_alpha"] = quantile_alpha
        else:
            params["objective"] = "reg:squarederror"
            params["eval_metric"] = "rmse"
        return params

    def _tune_study(self, X_train, y_train, X_val, y_val, n_trials=50,
                    warm_start_studies=None, warm_top_n=5, seed=42,
                    objective="mse", quantile_alpha=0.5, progress_bar=True):
        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=20, interval_steps=5)
        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

        if warm_start_studies:
            self._enqueue_best_trials(study, warm_start_studies, top_n=warm_top_n)

        objective_mode = objective
        eval_metric = "quantile" if objective_mode == "quantile" else "rmse"

        def objective_fn(trial):
            params = self._suggest_params(trial, objective=objective_mode, quantile_alpha=quantile_alpha)
            num_boost_round = int(params.pop("n_estimators", 200))
            params["seed"] = self.seed

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dvalid = xgb.DMatrix(X_val, label=y_val)

            class _OptunaPruningCallback(xgb.callback.TrainingCallback):
                def after_iteration(self, model, epoch, evals_log):
                    valid_metrics = evals_log.get("valid", {})
                    metric_values = valid_metrics.get(eval_metric)
                    if metric_values:
                        score_at_step = float(metric_values[-1])
                        trial.report(score_at_step, step=epoch)
                        if trial.should_prune():
                            raise optuna.TrialPruned(f"Trial pruned at iteration {epoch}.")
                    return False

            booster = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=num_boost_round,
                evals=[(dvalid, "valid")],
                callbacks=[
                    xgb.callback.EarlyStopping(rounds=50, save_best=True),
                    _OptunaPruningCallback(),
                ],
                verbose_eval=False,
            )

            best_iter = booster.best_iteration if booster.best_iteration is not None else num_boost_round
            preds = booster.predict(dvalid, iteration_range=(0, best_iter + 1))
            holdout_score = _validation_score(y_val, preds, objective_mode, quantile_alpha)

            trial.set_user_attr("best_iter", int(num_boost_round))
            return float(holdout_score)

        study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=progress_bar)
        return study

    def _fit_with_params(self, X_train, y_train, X_val, y_val, best_params,
                         objective="mse", quantile_alpha=0.5):
        params = best_params.copy()
        num_boost_round = int(params.pop("n_estimators", 200))
        params["seed"] = self.seed

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dval, "valid")],
            callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)],
            verbose_eval=False,
        )

        best_iter = booster.best_iteration if booster.best_iteration is not None else num_boost_round
        preds = booster.predict(dval, iteration_range=(0, best_iter + 1))
        score = _validation_score(y_val, preds, objective, quantile_alpha)
        return booster, score

    def _get_importance(self, best_params):
        params = best_params.copy()
        num_boost_round = int(params.pop("n_estimators", 200))
        params["seed"] = self.seed

        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            verbose_eval=False,
        )

        gain_dict = booster.get_score(importance_type="gain")
        gain = pd.Series(
            [gain_dict.get(f, 0.0) for f in self.X_train.columns],
            index=self.X_train.columns,
            name="gain",
        )

        weight_dict = booster.get_score(importance_type="weight")
        weight = pd.Series(
            [weight_dict.get(f, 0) for f in self.X_train.columns],
            index=self.X_train.columns,
            name="weight",
        )

        imp = pd.concat([gain, weight], axis=1).fillna(0.0)
        return imp
