
import numpy as np
import pandas as pd
import src.metrics as met
from scipy.stats import gaussian_kde
from crepes import WrapRegressor
from crepes.extras import MondrianCategorizer, DifficultyEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from typing import Iterable, List, Union
from src.metrics import randomized_pit_batch
from crepes.extras import MondrianCategorizer, DifficultyEstimator
from crepes.martingales import SimpleJumper
from sklearn.linear_model import LogisticRegression

ArrayLike = Union[np.ndarray, List[float]]

def as_numpy_1d(values):
    """Convert Series/DataFrame/array-like to a 1D numpy array.
    Preserves numeric dtype and flattens to shape (n,).
    """
    if isinstance(values, (pd.Series, pd.DataFrame)):
        values = values.to_numpy()
    arr = np.asarray(values)
    return arr.reshape(-1)


def as_numpy_2d(values):
    """Convert Series/DataFrame/array-like to a 2D numpy array.
    If a 1D array is provided it is reshaped to (n, 1).
    """
    if isinstance(values, (pd.Series, pd.DataFrame)):
        values = values.to_numpy()
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr

class scaler():
    """Class for scaling and inverse scaling of data using mean and standard deviation."""
    def __init__(self, data):
        """Constructor for scaler class.
        :param data: pandas DataFrame or numpy array of data to be scaled. Rows of observations and variables in columns.
        """
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        # To avoid division by zero in case of constant features, we can set std to a small value (e.g., 1e-6) for those features.
        self.std = np.maximum(self.std, 1e-6)

    def scale(self, data):
        """Method for scaling data using mean and standard deviation.
        :param data: pandas DataFrame or numpy array of data to be scaled. Rows of observations and variables in columns.
        :return scaled_data: pandas DataFrame or numpy array of scaled data.
        """
        scaled_data = (data - self.mean) / self.std
        return scaled_data

    def inverse_scale(self, scaled_data):
        """Method for inverse scaling of data using mean and standard deviation.
        :param scaled_data: pandas DataFrame or numpy array of scaled data. Rows of observations and variables in columns.
        :return original_data: pandas DataFrame or numpy array of original data before scaling.
        """
        original_data = (scaled_data * self.std) + self.mean
        return original_data

def pad_cdfs(pred_cdf: Iterable[Union[ArrayLike, Iterable[ArrayLike]]],
             fill=np.nan,
             pad_side="right") -> np.ndarray:
    """
    Turn a nested structure of arrays or segments into a 2D array padded with `fill`.
    pred_cdf may be:
      - a flat iterable of 1D arrays/lists (one row per observation), or
      - an iterable of segments (each segment is an iterable of 1D arrays).
    Returns: 2D array shape (n_rows, max_len) with rows padded on the left or right.
    pad_side: 'right' (default) pads at row end, 'left' pads at row start.
    """
    rows: List[np.ndarray] = []
    for item in pred_cdf:
        # detect if item is a segment (iterable of arrays) or a single row
        if isinstance(item, (list, tuple, np.ndarray)) and len(item) > 0 and any(
            isinstance(x, (list, tuple, np.ndarray)) for x in item
        ):
            rows.extend([np.asarray(r, dtype=float) for r in item])
        else:
            rows.append(np.asarray(item, dtype=float))

    if len(rows) == 0:
        return np.empty((0, 0))

    max_len = max(r.size for r in rows)
    mat = np.full((len(rows), max_len), fill, dtype=float)

    for i, r in enumerate(rows):
        if pad_side == "right":
            mat[i, : r.size] = r
        elif pad_side == "left":
            mat[i, -r.size :] = r
        else:
            raise ValueError("pad_side must be 'right' or 'left'")

    return mat



def train_cal_test_split(X, y , train_size=0.7, cal_size=0.15, scaled=False):
    """Splits the data into training, calibration, and test sets based on specified proportions.
    Parameters:
    X (array-like): The feature matrix to be split.
    y (array-like): The target values to be split.
    train_size (float): Proportion of the dataset to be used for training (default is 0.7).
    cal_size (float): Proportion of the dataset to be used for calibration (default is 0.15).
    Returns:
    tuple: A tuple containing the training, calibration, and test sets.
    """
    n = len(X)
    
    train_end = int(train_size * n)
    cal_end = train_end + int(cal_size * n)
    
    train_X = X[:train_end]
    train_y = y[:train_end]
    cal_X = X[train_end:cal_end]
    cal_y = y[train_end:cal_end]
    test_X = X[cal_end:]
    test_y = y[cal_end:]

    scaler_X, scaler_y = None, None
    
    if scaled:
        scaler_X = scaler(train_X)
        scaler_y = scaler(train_y)

        train_X = scaler_X.scale(train_X)
        cal_X = scaler_X.scale(cal_X)
        test_X = scaler_X.scale(test_X)

        train_y = scaler_y.scale(train_y)
        cal_y = scaler_y.scale(cal_y)
        test_y = scaler_y.scale(test_y)

    return train_X, cal_X, test_X, train_y, cal_y, test_y, scaler_X, scaler_y

def train_cal_test_date_split(X, y , train_end, calibration_end, test_end, scaled=False):
    """Splits the data into training, calibration, and test sets based on specified dates.
    Parameters:
    X (DataFrame): The feature matrix to be split with a datetime index.
    y (DataFrame): The target values to be split with a datetime index.
    train_end (str): The end date for the training set in 'YYYY-MM-DD' format.
    calibration_end (str): The end date for the calibration set in 'YYYY-MM-DD' format.
    test_end (str): The end date for the test set in 'YYYY-MM-DD' format.
    Returns:
    tuple: A tuple containing the training, calibration, and test sets.
    """

    # Align split markers to the index timezone and treat dates as day endpoints.
    index_tz = getattr(getattr(X, "index", None), "tz", None)

    def _to_day_end(value):
        ts = pd.to_datetime(value)
        if index_tz is not None:
            if ts.tzinfo is None:
                ts = ts.tz_localize(index_tz)
            else:
                ts = ts.tz_convert(index_tz)
        elif ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        return ts.normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)

    train_end_dt = _to_day_end(train_end)
    calibration_end_dt = _to_day_end(calibration_end)
    test_end_dt = _to_day_end(test_end)

    cal_start_dt = train_end_dt + pd.Timedelta(nanoseconds=1)
    test_start_dt = calibration_end_dt + pd.Timedelta(nanoseconds=1)

    train_X = X.loc[:train_end_dt]
    train_y = y.loc[:train_end_dt]

    cal_X = X.loc[cal_start_dt:calibration_end_dt]
    cal_y = y.loc[cal_start_dt:calibration_end_dt]

    test_X = X.loc[test_start_dt:test_end_dt]
    test_y = y.loc[test_start_dt:test_end_dt]

    scaler_X, scaler_y = None, None

    if scaled:
        scaler_X = scaler(train_X)
        scaler_y = scaler(train_y)

        train_X = scaler_X.scale(train_X)
        cal_X = scaler_X.scale(cal_X)
        test_X = scaler_X.scale(test_X)

        train_y = scaler_y.scale(train_y)
        cal_y = scaler_y.scale(cal_y)
        test_y = scaler_y.scale(test_y)
        
    return train_X, cal_X, test_X, train_y, cal_y, test_y, scaler_X, scaler_y


class ConformalPredictor: # Classic
    """Conformal Predictor for regression tasks using absolute residuals as conformity scores."""
    def fit(self, y_cal, y_pred, alpha=0.05):
        """Fits the conformal predictor using the calibration data.
        Parameters:
        y_cal (array-like): The true values for the calibration set.
        y_pred (array-like): The predicted values for the calibration set.
        alpha (float): The significance level for the prediction intervals.
        """
        self.alpha = alpha

        if not isinstance(y_cal, np.ndarray):
            y_cal = y_cal.to_numpy()
        if not isinstance(y_pred, np.ndarray):
            y_pred = y_pred.to_numpy()

        self.scores = np.abs(y_cal - y_pred)
        n = len(y_cal)
        level = np.ceil((n+1)*(1-alpha))/n
        
        self.q = np.quantile(self.scores, level, method="higher")

    def conformalize(self, y_pred_test):
        """Generates conformal prediction intervals for the test data.
        Parameters:
        y_pred_test (array-like): The predicted values for the test set.
        Returns:
        tuple: A tuple containing the lower and upper bounds of the prediction intervals.
        """
                
        self.lower_bound = y_pred_test - self.q
        self.upper_bound = y_pred_test + self.q
        
        return self.lower_bound, self.upper_bound
    

class QuantileConformalPredictor:
    """Conformal Predictor for regression tasks using quantile predictions as conformity scores."""
    def fit(self, y_cal, y_lower, y_upper, alpha = 0.05):
        """Calculates the conformity scores for the calibration data using quantile predictions.
        Parameters:
        y_cal (array-like): The true values for the calibration set.
        y_lower (array-like): The lower quantile predictions for the calibration set.
        y_upper (array-like): The upper quantile predictions for the calibration set.
        alpha (float): The significance level for the prediction intervals.
        """

        self.alpha = alpha
        if not isinstance(y_cal, np.ndarray):
            y_cal = y_cal.to_numpy()
        if not isinstance(y_lower, np.ndarray):
            y_lower = y_lower.to_numpy()
        if not isinstance(y_upper, np.ndarray):
            y_upper = y_upper.to_numpy()
            
        self.scores = np.maximum(y_cal - y_upper, y_lower - y_cal)
        n = len(y_cal)
        level = np.ceil((n+1)*(1-alpha))/n
        self.q = np.quantile(self.scores, level, method="higher")

    def conformalize(self, y_lower_test, y_upper_test):
        """Generates quantile-based prediction intervals for the test data.
        Parameters:
        y_lower_test (array-like): The lower quantile predictions for the test set.
        y_upper_test (array-like): The upper quantile predictions for the test set.
        Returns:
        tuple: A tuple containing the lower and upper bounds of the prediction intervals.
        """

        self.lower_bound = y_lower_test - self.q
        self.upper_bound = y_upper_test + self.q
        return self.lower_bound, self.upper_bound
    


class BayesianConformalPredictor:
    """Conformal Predictor for regression tasks using Bayesian predictive distributions as conformity scores."""
    def __init__(self, sample_method="kde", subsample_size=1000, grid_size=256, grid_margin=0.1, random_state=None):
        """Initializes the Bayesian Conformal Predictor with a specified method for estimating the predictive distribution.
        Parameters:
        sample_method (str): The method to estimate the predictive distribution. Options are "kde", "subsample", or "normal_approx" (default is "kde").
        subsample_size (int): The number of samples to use for the "subsample" method (default is 1000).
        grid_size (int): Number of grid points used for numerical inversion in KDE modes.
        grid_margin (float): Relative margin around each sample range for KDE grid evaluation.
        random_state (int or None): Random seed for reproducible subsampling.
        """

        self.sample_method = sample_method
        self.subsample_size = subsample_size
        self.grid_size = grid_size
        self.grid_margin = grid_margin
        self.rng = np.random.default_rng(random_state)
        
        # Select the pdf strategy once during initialization
        if self.sample_method == "kde":
            self._get_pdf = gaussian_kde
        elif self.sample_method == "subsample":
            self._get_pdf = self._subsample_kde
        elif self.sample_method == "normal_approx":
            self._get_pdf = self._normal_approx
        else:
            raise ValueError("Invalid sample_method. Choose from 'kde', 'subsample', or 'normal_approx'.")

    def _subsample_kde(self, samples):
        if len(samples) > self.subsample_size:
            subsample = self.rng.choice(samples, size=self.subsample_size, replace=False)
        else:
            subsample = samples
        return gaussian_kde(subsample)

    def _normal_approx(self, samples):
        mean = np.mean(samples)
        std = np.std(samples)
        # return a callable that mimics gaussian_kde interface (callable for evaluation)
        # gaussian_kde(points) returns pdf values.
        def pdf_func(x):
            return np.array([(1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)])
        # We add an 'evaluate' method to match gaussian_kde's API if needed by predict
        pdf_func.evaluate = pdf_func 
        return pdf_func

    def fit(self, y_cal, distributions = None, mu = None, sigma = None, alpha=0.05):
        """Fits the Bayesian conformal predictor using the calibration data and predictive distributions.
        Parameters:
        y_cal (array-like): The true values for the calibration set.
        distributions (array-like): The predictive distributions for each sample in the calibration set.
        mu (array-like): The mean values for each sample in the calibration set.
        sigma (array-like): The standard deviation values for each sample in the calibration set.
        alpha (float): The significance level for the prediction intervals (default is 0.05).
        """
        
        if distributions is not None: # For BART or other methods that provide full predictive distributions, we can compute scores directly from the densities.
            if not isinstance(distributions, np.ndarray):
                distributions = distributions.to_numpy()

            if not isinstance(y_cal, np.ndarray):
                y_cal = y_cal.to_numpy()
            
            n, m = distributions.shape
            self.scores = np.zeros(n)

            if self.sample_method == "normal_approx":
                mu = np.mean(distributions, axis=1)
                sigma = np.std(distributions, axis=1)
                sigma = np.maximum(sigma, 1e-12)
                z = (y_cal - mu) / sigma
                pdf_at_true = np.exp(-0.5 * z * z) / (sigma * np.sqrt(2 * np.pi))
                self.scores = -pdf_at_true
            else:
                # Loop uses the pre-selected _get_pdf method
                for i in range(n):
                    density = self._get_pdf(distributions[i, :])
                    p_at_true = density.evaluate(y_cal[i])
                    self.scores[i] = -p_at_true[0]  # Negate to convert to conformity score (lower density = higher score)

            level = np.ceil((n + 1) * (1 - alpha)) / n
            self.q = np.quantile(self.scores, level, method="higher")

        elif mu is not None and sigma is not None: # For parametric models that provide mean and standard deviation
            # If mean and std are provided directly (e.g., from a parametric model), we can compute scores without distributions.
            sigma = np.maximum(sigma, 1e-12)
            z = (y_cal - mu) / sigma
            pdf_at_true = np.exp(-0.5 * z * z) / (sigma * np.sqrt(2 * np.pi))
            self.scores = -pdf_at_true
            n = len(y_cal)
            level = np.ceil((n + 1) * (1 - alpha)) / n
            self.q = np.quantile(self.scores, level, method="higher")
        else:
            raise ValueError("Either distributions or both mu and sigma must be provided for fitting.")

    def conformalize(self, distributions_test = None, mu_test = None, sigma_test = None):
        """Generates prediction intervals for the test data based on the predictive distributions.
        Parameters:
        distributions_test (array-like): The predictive samples for each test sample.
        mu_test (array-like): The mean values for each test sample.
        sigma_test (array-like): The standard deviation values for each test sample.

        Returns:
        tuple: A tuple containing the lower and upper bounds of the prediction intervals.
        """


        if distributions_test is not None:
            if not isinstance(distributions_test, np.ndarray):
                distributions_test = distributions_test.to_numpy()

            n = len(distributions_test)
            lower_interval = np.zeros(n)
            upper_interval = np.zeros(n)

            # Special casing for normal_approx is computationally faster (analytic solution)
            # so we might keep a specific branch or optimize it. 
            # But to be consistent with your request, we can use the generic numerical inversion for all:
            
            if self.sample_method == "normal_approx":
                # For Normal densities, f(x) >= threshold has a closed-form interval around mu.
                mu = np.mean(distributions_test, axis=1)
                sigma = np.std(distributions_test, axis=1)
                sigma = np.maximum(sigma, 1e-12)
                
                threshold_density = -self.q
                if threshold_density <= 0:
                    lower_interval = -np.inf * np.ones(n)
                    upper_interval = np.inf * np.ones(n)
                else:
                    ratio = threshold_density * sigma * np.sqrt(2 * np.pi)
                    ratio = np.maximum(ratio, 1e-300)
                    radius = np.zeros(n)
                    valid = ratio < 1.0
                    radius[valid] = sigma[valid] * np.sqrt(-2.0 * np.log(ratio[valid]))
                    lower_interval = mu - radius
                    upper_interval = mu + radius

            else:
                # Numerical inversion per sample with adaptive ranges avoids wasting grid work.
                for i in range(n):
                    samples_i = distributions_test[i, :]
                    local_min = samples_i.min()
                    local_max = samples_i.max()
                    local_span = max(local_max - local_min, 1e-12)
                    margin = local_span * self.grid_margin
                    x_eval = np.linspace(local_min - margin, local_max + margin, self.grid_size)

                    density = self._get_pdf(samples_i)
                    p_eval = density.evaluate(x_eval)
                    mask = p_eval >= -self.q
                    if np.any(mask):
                        idx = np.flatnonzero(mask)
                        lower_interval[i] = x_eval[idx[0]]
                        upper_interval[i] = x_eval[idx[-1]]
                    else:
                        # If no points satisfy the condition, set bounds to the mean
                        lower_interval[i] = samples_i.mean()
                        upper_interval[i] = samples_i.max()
        elif mu_test is not None and sigma_test is not None:
            sigma_test = np.maximum(sigma_test, 1e-12)
            threshold_density = -self.q
            if threshold_density <= 0:
                lower_interval = -np.inf * np.ones_like(mu_test)
                upper_interval = np.inf * np.ones_like(mu_test)
            else:
                ratio = threshold_density * sigma_test * np.sqrt(2 * np.pi)
                ratio = np.maximum(ratio, 1e-300)
                radius = np.zeros_like(mu_test)
                valid = ratio < 1.0
                radius[valid] = sigma_test[valid] * np.sqrt(-2.0 * np.log(ratio[valid]))
                lower_interval = mu_test - radius
                upper_interval = mu_test + radius
            

        self.lower_bound = lower_interval
        self.upper_bound = upper_interval
        return self.lower_bound, self.upper_bound

class CPS:
    """Conformal Predictive System (CPS) for regression tasks using a regression model and a conformal predictor."""
    def __init__(self, regression_model):
        """Initializes the CPS with a specified regression model.
        Parameters:
            regression_model: A regression model that has fit and predict methods (e.g., a scikit-learn regressor).
        """
        # self.regression_model = regression_model
        self.de = DifficultyEstimator()
        self.wrapedEstimator = WrapRegressor(regression_model)
        self.mc_diff = MondrianCategorizer()
        self.fitted = None

    def fit(self, X_train, y_train, difficulty_model = "knn", k=25):
        """Fits the regression model using the training data.
        Parameters:
            X_train (array-like): The feature matrix for the training set.
            y_train (array-like): The target values for the training set.
        """
        self.wrapedEstimator.fit(X_train, y_train)
        self.fitted = True

        if difficulty_model == "knn":
            # Difficulty estimates are std of targets using KNN in feature space between calibration and training data
            self.de.fit(X = X_train, y = y_train, k=k, scaler=True)
        elif difficulty_model == "rf":
            # Difficulty estimates are based on residual predictions from a Random Forest trained on the training data residuals
            preds = self.wrapedEstimator.predict(X_train)
            res = np.abs(preds - y_train)
            rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, res)
            self.de.fit(X_train, f=rf.predict, scaler=True)
        elif difficulty_model == None:
            # No difficulty estimation, just a single global category
            self.de = None
        else:
            raise ValueError("Invalid difficulty model. Choose from 'knn', 'rf' or None.")
        
    def calibrate(self, X_calibration, y_calibration, k=25, bins=8, mondrian_strategy='predictions'):
        """Calibrates the CPS using the calibration data and specified Mondrian categorization strategy.
        Parameters:
            X_calibration (array-like): The feature matrix for the calibration set.
            y_calibration (array-like): The target values for the calibration set.
            k (int): The number of nearest neighbors to consider for difficulty estimation (default is 25).
            bins (int): The number of bins for Mondrian categorization (default is 8).
            mondrian_strategy (str): The strategy for Mondrian categorization ('predictions', 'difficulty', or None).
        """
        if mondrian_strategy == 'predictions':
            # Mondrian categorization based on predicted values and difficulty scaling (normalization)
            self.mc_diff.fit(X_calibration, f=self.wrapedEstimator.predict, no_bins=bins)
            self.wrapedEstimator.calibrate(X_calibration, y_calibration, cps = True, mc = self.mc_diff, de = self.de)
        elif mondrian_strategy == 'difficulty':
            # Mondrian categorization based on residual estimates leading to difficulty-based categories
            self.mc_diff.fit(X_calibration, de=self.de, no_bins=bins)
            self.wrapedEstimator.calibrate(X_calibration, y_calibration, cps = True, mc = self.mc_diff, de = self.de)
        elif mondrian_strategy == None:
            # No Mondrian categorization, just a single global category
            self.wrapedEstimator.calibrate(X_calibration, y_calibration, cps = True, de = self.de)
        else:
            raise ValueError("Invalid Mondrian categorization strategy. Choose from 'difficulty', 'predictions' or None.")

    def predict(self, X):
        """Generates predictions for the given feature matrix using the fitted regression model.
        Parameters:
            X (array-like): The feature matrix for which to generate predictions.
        Returns:
            array-like: The predicted values for the input feature matrix.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction.")
        return self.wrapedEstimator.predict(X)

    def fit_calibrate(self, X_train, y_train, X_calibration, y_calibration, k=25, bins=8, mondrian_strategy='predictions', difficulty_model = "knn"):
        """Fits and calibrates the CPS using the training and calibration data. Fits only the first time and allows for multiple calibrations with different strategies without refitting the regression model.
        Parameters:
            X_train (array-like): The feature matrix for the training set.
            y_train (array-like): The target values for the training set.
            X_calibration (array-like): The feature matrix for the calibration set.
            y_calibration (array-like): The target values for the calibration set.
            k (int): The number of nearest neighbors to consider.
            bins (int): The number of bins for Mondrian categorization.
            mondrian_strategy (str): The strategy for Mondrian categorization ('predictions' or 'difficulty').
            difficulty_model (str): The model to use for estimating difficulty ('knn' or 'rf').
        """
        if not self.fitted:
            self.fit(X_train, y_train, difficulty_model, k)
        
        self.calibrate(X_calibration, y_calibration, k, bins, mondrian_strategy)
        

    def predict_cpds(self, X_test):
        """Generates conformal predictive distributions for the test data.
        Parameters:
            X_test (array-like): The feature matrix for the test set.
        Returns:
            array-like: The conformal predictive distributions for the test set.
        """
        return self.wrapedEstimator.predict_cpds(X_test)
    
    def predict_cps(self, X_test, lower_percentiles, higher_percentiles):
        """Generates conformal prediction intervals for the test data based on specified percentiles.
        Parameters:
            X_test (array-like): The feature matrix for the test set.
            lower_percentiles (array-like): The lower percentiles for the prediction intervals.
            higher_percentiles (array-like): The upper percentiles for the prediction intervals.
        Returns:
            tuple: A tuple containing the lower and upper bounds of the prediction intervals.
        """
        return self.wrapedEstimator.predict_cps(X = X_test, lower_percentiles = lower_percentiles, higher_percentiles = higher_percentiles)
    
    def predict_p(self, X_test, y_val):
        """Generates p-values for the test data based on the conformal predictive distributions.
        Parameters:
            X_test (array-like): The feature matrix for the test set.
            y_val (array-like): The true values for the test set.
        Returns:
            array-like: The p-values for the test set.
        """
        return self.wrapedEstimator.predict_p(X_test, y = y_val)



class covariateShiftCP():
    def __init__(self, regression_model):
        self.de = DifficultyEstimator()
        self.wrapedEstimator = WrapRegressor(regression_model)

    def fit(self, X_tr, y_train, X_ca, y_calibration):
        self.wrapedEstimator.fit(X_tr, y_train)
        res = np.abs(y_train.values - self.wrapedEstimator.predict(X_tr))
        self.de.fit(X = X_tr, residuals = res)
    
        self.wrapedEstimator.calibrate(X_ca, y_calibration.values, de=self.de)


    def conformalize(self, X_ca, X_te, alpha=0.1, inflation_factor=0.3):
        X_domain = np.vstack([X_ca, X_te.values])
        y_domain = np.concatenate([np.zeros(len(X_ca)), np.ones(len(X_te))])
        self.shift_detector = LogisticRegression(C=0.1).fit(X_domain, y_domain)

        sigmas_base = self.de.apply(X_te)

        # Get shift probabilities (0 to 1)
        shift_probs = self.shift_detector.predict_proba(X_te.values)[:, 1]

        # Define an inflation factor: 1.0 (no shift) to 5.0 (total shift)
        # This is a heuristic to widen intervals where we are "surprised" by X
        inflation = 1 + (shift_probs * inflation_factor) 
        sigmas_adapted = sigmas_base * inflation

        # 1. Get the base predictions
        y_hat = self.wrapedEstimator.predict(X_te)

        # 2. Get the calibrated "normalized residuals" (alphas) from the WrapRegressor
        # These were calculated as: |y_cal - y_hat_cal| / sigma_cal
        alphas = self.wrapedEstimator.cr.alphas

        level = np.ceil((len(alphas) + 1) * (1 - alpha)) / len(alphas)
        q_hat = np.quantile(alphas, level, method="higher")

        # 4. Generate Intervals manually using your Adapted Sigmas
        # Interval = Prediction +/- (Quantile * Sigma)
        lower_adapted = y_hat - (q_hat * sigmas_adapted)
        upper_adapted = y_hat + (q_hat * sigmas_adapted)

        return lower_adapted, upper_adapted
    

class covariateShiftCPS():
    def __init__(self, regression_model):
        self.wrapedEstimator = WrapRegressor(regression_model)

    def fit(self, X_tr, y_train, X_ca, y_calibration):
        self.wrapedEstimator.fit(X_tr, y_train)
        self.wrapedEstimator.calibrate(X_ca, y_calibration.values, cps = True)


    def conformalize(self, X_ca, X_te, alpha=0.1, inflation_factor=0.3):
        X_domain = np.vstack([X_ca, X_te.values])
        y_domain = np.concatenate([np.zeros(len(X_ca)), np.ones(len(X_te))])
        self.shift_detector = LogisticRegression(C=0.1).fit(X_domain, y_domain)

        # Get shift probabilities (0 to 1)
        shift_probs = self.shift_detector.predict_proba(X_te.values)[:, 1]

        # Define an inflation factor: 1.0 (no shift) to 5.0 (total shift)
        # This is a heuristic to widen intervals where we are "surprised" by X
        inflation = 1 + (shift_probs * inflation_factor) 


        cpds = self.wrapedEstimator.predict_cpds(X_te)
        cpds_adjusted = cpds * inflation[:, np.newaxis]  # Scale the predictive distributions by the inflation factor

        return cpds_adjusted


class ImportanceWeightedCPS:
    """Weighted conformal predictive system with externally provided density-ratio weights."""

    def __init__(self, regression_model, difficulty_model="rf", k=25, beta=0.01, scaler=True):
        self.wrapedEstimator = WrapRegressor(regression_model)
        self.de = DifficultyEstimator()
        self.difficulty_model = difficulty_model
        self.k = k
        self.beta = beta
        self.scaler = scaler
        self._fitted = False
        self._shift_eps = 1e-12

    @staticmethod
    def _as_numpy_1d(values):
        return as_numpy_1d(values)

    @staticmethod
    def _as_numpy_2d(values):
        return as_numpy_2d(values)

    @staticmethod
    def _weighted_quantile(sorted_values, sorted_weights, q):
        cumulative = np.cumsum(sorted_weights)
        idx = np.searchsorted(cumulative, q, side="left")
        idx = min(max(idx, 0), len(sorted_values) - 1)
        return sorted_values[idx]

    def _fit_difficulty(self, X_train, y_train):
        if self.difficulty_model == "knn":
            self.de.fit(X=X_train, y=y_train, k=self.k, scaler=self.scaler, beta=self.beta)
        elif self.difficulty_model == "rf":
            preds = self.wrapedEstimator.predict(X_train)
            res = np.abs(preds - y_train)
            rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, res)
            self.de.fit(X_train, f=rf.predict, scaler=self.scaler, beta=self.beta)
        else:
            raise ValueError("Invalid difficulty model. Choose from 'knn' or 'rf'.")

    def fit(self, X_train, y_train, X_calibration, y_calibration):
        """Fit point model, difficulty model, and store calibration scores for weighted CPS."""
        X_train_np = self._as_numpy_2d(X_train)
        X_cal_np = self._as_numpy_2d(X_calibration)
        y_train_np = self._as_numpy_1d(y_train)
        y_cal_np = self._as_numpy_1d(y_calibration)

        self.wrapedEstimator.fit(X_train_np, y_train_np)
        self._fit_difficulty(X_train_np, y_train_np)

        y_hat_cal = self._as_numpy_1d(self.wrapedEstimator.predict(X_cal_np))
        sigma_cal = np.maximum(self._as_numpy_1d(self.de.apply(X_cal_np)), self._shift_eps)

        # Signed normalized residuals define the support shifts for predictive distributions.
        self.alphas_cal = (y_cal_np - y_hat_cal) / sigma_cal
        self.X_calibration = X_cal_np
        self.y_calibration = y_cal_np
        self.sigma_cal = sigma_cal
        self._fitted = True
        return self

    def _prepare_calibration_weights(self, calibration_weights=None, weight_function=None):
        if not self._fitted:
            raise RuntimeError("Model must be fitted before calling predict_cpds")

        if calibration_weights is not None:
            weights = self._as_numpy_1d(calibration_weights)
        elif weight_function is not None:
            weights = self._as_numpy_1d(weight_function(self.X_calibration))
        else:
            weights = np.ones(len(self.alphas_cal))

        if len(weights) != len(self.alphas_cal):
            raise ValueError("calibration_weights must match the number of calibration samples")

        weights = np.maximum(weights, self._shift_eps)
        return weights / np.sum(weights)

    def predict_cpds(self, X_test, calibration_weights=None, weight_function=None, return_cdf=True):
        """Return weighted predictive distributions (support + masses) for each test object."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before calling predict_cpds")

        X_test_np = self._as_numpy_2d(X_test)
        y_hat_test = self._as_numpy_1d(self.wrapedEstimator.predict(X_test_np))
        sigma_test = np.maximum(self._as_numpy_1d(self.de.apply(X_test_np)), self._shift_eps)
        cal_weights = self._prepare_calibration_weights(calibration_weights, weight_function)

        cpds = []
        for i in range(len(y_hat_test)):
            support = y_hat_test[i] + sigma_test[i] * self.alphas_cal
            order = np.argsort(support)
            support_sorted = support[order]
            mass_sorted = cal_weights[order]
            distribution = {
                "support": support_sorted,
                "mass": mass_sorted,
            }
            if return_cdf:
                distribution["cdf"] = np.cumsum(mass_sorted)
            cpds.append(distribution)
        return cpds

    def predict_percentiles(self, X_test, percentiles, calibration_weights=None, weight_function=None):
        """Return weighted percentiles from the predictive distributions."""
        cpds = self.predict_cpds(
            X_test,
            calibration_weights=calibration_weights,
            weight_function=weight_function,
            return_cdf=False,
        )
        quantiles = np.array(percentiles, dtype=float) / 100.0
        out = np.zeros((len(cpds), len(quantiles)))
        for i, cpd in enumerate(cpds):
            values = cpd["support"]
            weights = cpd["mass"]
            for j, q in enumerate(quantiles):
                out[i, j] = self._weighted_quantile(values, weights, q)
        return out


class ClassifierWeightedCPS(ImportanceWeightedCPS):
    """Weighted CPS where density-ratio weights are estimated by a domain classifier."""

    def __init__(
        self,
        regression_model,
        difficulty_model="rf",
        k=25,
        beta=0.01,
        scaler=True,
        shift_C=0.1,
        clip_min=1e-3,
        clip_max=1e3,
    ):
        super().__init__(
            regression_model=regression_model,
            difficulty_model=difficulty_model,
            k=k,
            beta=beta,
            scaler=scaler,
        )
        self.shift_C = shift_C
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.shift_detector = None
        self.rho_target = None

    def fit_shift_detector(self, X_source, X_target):
        """Fit logistic domain classifier and store the target prior for ratio conversion."""
        X_source_np = self._as_numpy_2d(X_source)
        X_target_np = self._as_numpy_2d(X_target)
        X_domain = np.vstack([X_source_np, X_target_np])
        y_domain = np.concatenate([np.zeros(len(X_source_np)), np.ones(len(X_target_np))])

        self.rho_target = np.mean(y_domain)
        self.shift_detector = LogisticRegression(C=self.shift_C, max_iter=200, class_weight="balanced")
        self.shift_detector.fit(X_domain, y_domain)
        return self

    def _density_ratio(self, X):
        if self.shift_detector is None or self.rho_target is None:
            raise RuntimeError("Shift detector is not fitted. Call fit_shift_detector first.")

        X_np = self._as_numpy_2d(X)
        p_target = self.shift_detector.predict_proba(X_np)[:, 1]
        p_target = np.clip(p_target, self._shift_eps, 1 - self._shift_eps)
        ratio = ((1 - self.rho_target) / self.rho_target) * (p_target / (1 - p_target))
        return np.clip(ratio, self.clip_min, self.clip_max)

    def predict_cpds(self, X_test, return_cdf=True):
        """Return weighted predictive distributions using classifier-derived density-ratio weights."""
        cal_weights = self._density_ratio(self.X_calibration)
        return super().predict_cpds(X_test, calibration_weights=cal_weights, return_cdf=return_cdf)

    def predict_percentiles(self, X_test, percentiles):
        """Return weighted percentiles using classifier-derived density-ratio weights."""
        cal_weights = self._density_ratio(self.X_calibration)
        return super().predict_percentiles(X_test, percentiles, calibration_weights=cal_weights)
    


def conformalize_distribution(
    X_cal,
    y_cal,
    preds_cal,
    X_val,
    preds_val,
    alphas,
    use_clustering=False,
    clustering_method="kmeans",     # "kmeans", "variance", "gaussian_mixture"
    n_clusters=2,
    asymmetric=False,
    weighted=False,
    blend_clusters=True,            # soft blending for kmeans and gaussian_mixture
    return_type="interpolators",    # "interpolators" or "bounds"
    interpolator_kind="linear",            # "linear" or "cubic" for the inverse CDF interpolators
    random_state=42,
    input_median = False
):
    """
    Conformalize predictive distributions using calibration data, with optional clustering and weighting for covariate shift.
    Parameters:
        X_cal: Calibration features (DataFrame or array-like)
        y_cal: Calibration targets (Series or array-like)
        preds_cal: Calibration predictive distributions (DataFrame with quantiles or array-like)
        X_val: Validation features (DataFrame or array-like)
        preds_val: Validation predictive distributions (DataFrame with quantiles or array-like)
        alphas: Array-like of significance levels for the prediction intervals (e.g., [0.1, 0.05, 0.01])
        use_clustering: Whether to cluster calibration data for separate quantile estimation (default False)
        clustering_method: Method for clustering ('kmeans', 'variance', 'gaussian_mixture')
        n_clusters: Number of clusters to use if clustering is enabled (default 2)
        asymmetric: Whether to use separate quantiles for upper and lower tails (default False)
        weighted: Whether to apply importance weighting based on cluster membership (default False)
        blend_clusters: Whether to use soft blending of cluster quantiles for validation points (only for kmeans and gaussian_mixture, default True)
        return_type: Whether to return 'interpolators' for the inverse CDFs or 'bounds' for the prediction intervals (default 'interpolators')
        interpolator_kind: The kind of interpolation to use for the inverse CDFs ('linear' or 'cubic', default 'linear')
        random_state: Random seed for reproducibility in clustering (default 42)
        input_median: Whether to use the median of the predictive distributions as the point prediction for clustering and weighting (default False)
    
    Returns:
        If return_type is 'interpolators': A list of interpolator functions for each alpha level that take a value and return the corresponding quantile.
        If return_type is 'bounds': A dictionary with keys 'lower' and 'upper' containing the lower and upper bounds of the prediction intervals for each validation point and alpha level.
    """
    alphas = np.asarray(alphas, dtype=float)
    n_alpha = len(alphas)

    # Allow preds to be numpy arrays or DataFrames; ensure DataFrame for quantile operations
    if not hasattr(preds_cal, "quantile"):
        preds_cal = pd.DataFrame(preds_cal)
    if not hasattr(preds_val, "quantile"):
        preds_val = pd.DataFrame(preds_val)

    n_val = len(preds_val)

    # Ensure y_cal is 1d numpy
    y_cal_arr = as_numpy_1d(y_cal)

    # Basic shape checks
    if preds_cal.shape[0] != len(y_cal_arr):
        raise ValueError("preds_cal rows must match length of y_cal")
    if preds_val.shape[0] != n_val:
        raise ValueError("Invalid preds_val shape")

    if use_clustering and n_clusters < 1:
        raise ValueError("n_clusters must be >= 1")

    # ---------- Local helpers ----------
    def _weighted_quantile(scores_1d, weights_1d, alpha_tail):
        # Returns quantile at level (1 - alpha_tail), robust to edge cases.
        idx = np.argsort(scores_1d)
        s = scores_1d[idx]
        w = np.asarray(weights_1d, dtype=float)[idx]
        w = np.clip(w, 1e-12, None)
        cw = np.cumsum(w)
        cw /= cw[-1]
        target = np.clip(1.0 - alpha_tail, 0.0, 1.0 - 1e-12)
        pos = np.searchsorted(cw, target, side="left")
        pos = min(pos, len(s) - 1)
        return s[pos]

    def _covariate_weights(Xc, Xv):
        # Density ratio proxy via calibration-vs-validation classifier.
        Xc = np.asarray(Xc)
        Xv = np.asarray(Xv)
        X_combined = np.vstack([Xc, Xv])
        y_combined = np.hstack([np.zeros(len(Xc)), np.ones(len(Xv))])
        clf = LogisticRegression(max_iter=5000, solver="lbfgs", tol=1e-4, random_state=42).fit(X_combined, y_combined)
        p = clf.predict_proba(Xc)[:, 1]
        return np.clip(p / (1.0 - p + 1e-6), 0.1, 10.0)

    # ---------- Precompute quantiles and scores ----------
    q_low_cal = preds_cal.quantile(alphas / 2, axis=1).values       # (n_alpha, n_cal)
    q_high_cal = preds_cal.quantile(1 - alphas / 2, axis=1).values  # (n_alpha, n_cal)

    if asymmetric:
        # lower violation: y below lower quantile
        scores_low = q_low_cal - y_cal_arr
        # upper violation: y above upper quantile
        scores_high = y_cal_arr - q_high_cal
        alpha_tail = alphas / 2.0
    else:
        scores_sym = np.maximum(q_low_cal - y_cal_arr, y_cal_arr - q_high_cal)
        alpha_tail = alphas

    # ---------- Build clusters ----------
    if not use_clustering:
        n_clusters_eff = 1
        bin_indices = np.zeros(len(y_cal), dtype=int)
        val_bin_indices = np.zeros(n_val, dtype=int)
        val_cluster_weights = np.ones((n_val, 1))
    else:
        n_clusters_eff = n_clusters
        # Accept pandas or numpy for feature matrices
        Xc = as_numpy_2d(X_cal)
        Xv = as_numpy_2d(X_val)

        val_cluster_weights = np.zeros((n_val, n_clusters_eff))


        method = clustering_method.lower()
        if method == "kmeans":
            model = KMeans(n_clusters=n_clusters_eff, random_state=random_state).fit(Xc)
            bin_indices = model.predict(Xc)
            val_bin_indices = model.predict(Xv)
            
            
            # Soft weights from inverse distance
            d = model.transform(Xv)
            eps = 1e-6
            w_raw = 1.0 / (d + eps)    # shape (n_val, n_clusters)
            # per-row 99th-percentile cap (avoid single tiny-distance explosion)
            w_cap = np.quantile(w_raw, 0.99, axis=1, keepdims=True)
            # ensure a sensible minimum cap (fallback)
            min_cap = 1e3
            w = np.minimum(w_raw, np.maximum(w_cap, min_cap))
            # normalize safely
            denom = np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
            val_cluster_weights = w / denom

        elif method == "variance":
            cal_var = preds_cal.std(axis=1).values
            val_var = preds_val.std(axis=1).values
            edges = np.quantile(cal_var, np.linspace(0, 1, n_clusters_eff + 1))
            bin_indices = np.digitize(cal_var, edges[1:-1])
            val_bin_indices = np.digitize(val_var, edges[1:-1]) 
            val_cluster_weights[np.arange(n_val), val_bin_indices] = 1.0


        elif method == "gaussian_mixture":
            model = GaussianMixture(n_components=n_clusters_eff, random_state=random_state).fit(Xc)
            bin_indices = model.predict(Xc)
            val_bin_indices = model.predict(Xv)
            val_cluster_weights = model.predict_proba(Xv)
            val_cluster_weights /= val_cluster_weights.sum(axis=1, keepdims=True)


            
        else:
            raise ValueError(
                "Unsupported clustering_method. Use one of: "
                "kmeans, variance, agglomerative, gaussian_mixture"
            )

    # ---------- Calibration weights ----------
    if weighted:
        Xc_feats = as_numpy_2d(X_cal)
        Xv_feats = as_numpy_2d(X_val)
        cal_weights = _covariate_weights(Xc_feats, Xv_feats)
    else:
        cal_weights = np.ones(len(y_cal), dtype=float)

    # ---------- Quantile corrections per cluster ----------
    if asymmetric:
        q_corr_low = np.full((n_clusters_eff, n_alpha), np.nan, dtype=float)
        q_corr_high = np.full((n_clusters_eff, n_alpha), np.nan, dtype=float)
    else:
        q_corr_sym = np.full((n_clusters_eff, n_alpha), np.nan, dtype=float)

    for c in range(n_clusters_eff):
        mask = bin_indices == c
        n_c = int(mask.sum())
        if n_c == 0:
            continue

        if asymmetric:
            s_low_c = scores_low[:, mask]
            s_high_c = scores_high[:, mask]
        else:
            s_sym_c = scores_sym[:, mask]

        w_c = cal_weights[mask]

        if weighted:
            if asymmetric:
                q_corr_low[c] = np.array([
                    _weighted_quantile(s_low_c[i], w_c, alpha_tail[i]) for i in range(n_alpha)
                ])
                q_corr_high[c] = np.array([
                    _weighted_quantile(s_high_c[i], w_c, alpha_tail[i]) for i in range(n_alpha)
                ])
            else:
                q_corr_sym[c] = np.array([
                    _weighted_quantile(s_sym_c[i], w_c, alpha_tail[i]) for i in range(n_alpha)
                ])
        else:
            levels = np.clip(np.ceil((n_c + 1) * (1 - alpha_tail)) / n_c, 0, 1)
            if asymmetric:
                q_corr_low[c] = np.array([
                    np.quantile(s_low_c[i], levels[i], method="higher") for i in range(n_alpha)
                ])
                q_corr_high[c] = np.array([
                    np.quantile(s_high_c[i], levels[i], method="higher") for i in range(n_alpha)
                ])
            else:
                q_corr_sym[c] = np.array([
                    np.quantile(s_sym_c[i], levels[i], method="higher") for i in range(n_alpha)
                ])

    # Fallback for empty clusters: use global correction
    all_mask = np.ones(len(y_cal), dtype=bool)
    if asymmetric:
        s_low_all = scores_low[:, all_mask]
        s_high_all = scores_high[:, all_mask]
        w_all = cal_weights[all_mask]
        n_all = len(y_cal)

        if weighted:
            global_low = np.array([
                _weighted_quantile(s_low_all[i], w_all, alpha_tail[i]) for i in range(n_alpha)
            ])
            global_high = np.array([
                _weighted_quantile(s_high_all[i], w_all, alpha_tail[i]) for i in range(n_alpha)
            ])
        else:
            levels_all = np.clip(np.ceil((n_all + 1) * (1 - alpha_tail)) / n_all, 0, 1)
            global_low = np.array([
                np.quantile(s_low_all[i], levels_all[i], method="higher") for i in range(n_alpha)
            ])
            global_high = np.array([
                np.quantile(s_high_all[i], levels_all[i], method="higher") for i in range(n_alpha)
            ])

        for c in range(n_clusters_eff):
            if np.isnan(q_corr_low[c]).any():
                q_corr_low[c] = global_low
            if np.isnan(q_corr_high[c]).any():
                q_corr_high[c] = global_high
    else:
        s_sym_all = scores_sym[:, all_mask]
        w_all = cal_weights[all_mask]
        n_all = len(y_cal)

        if weighted:
            global_sym = np.array([
                _weighted_quantile(s_sym_all[i], w_all, alpha_tail[i]) for i in range(n_alpha)
            ])
        else:
            levels_all = np.clip(np.ceil((n_all + 1) * (1 - alpha_tail)) / n_all, 0, 1)
            global_sym = np.array([
                np.quantile(s_sym_all[i], levels_all[i], method="higher") for i in range(n_alpha)
            ])

        for c in range(n_clusters_eff):
            if np.isnan(q_corr_sym[c]).any():
                q_corr_sym[c] = global_sym

    # ---------- Apply corrections to validation quantiles ----------
    val_low_raw = preds_val.quantile(alphas / 2, axis=1).values       # (n_alpha, n_val)
    val_high_raw = preds_val.quantile(1 - alphas / 2, axis=1).values  # (n_alpha, n_val)
    val_median = preds_val.median(axis=1).values                       # (n_val,)

    soft_available = use_clustering and clustering_method.lower() in ("kmeans", "gaussian_mixture")
    use_soft = soft_available and blend_clusters

    if asymmetric:
        if use_soft:
            # (n_val, n_clusters) @ (n_clusters, n_alpha) -> (n_val, n_alpha)
            corr_low = (val_cluster_weights @ q_corr_low).T
            corr_high = (val_cluster_weights @ q_corr_high).T
        else:
            corr_low = np.zeros_like(val_low_raw)
            corr_high = np.zeros_like(val_high_raw)
            for c in range(n_clusters_eff):
                m = val_bin_indices == c
                if np.any(m):
                    corr_low[:, m] = q_corr_low[c][:, None]
                    corr_high[:, m] = q_corr_high[c][:, None]

        val_low_conf = val_low_raw - corr_low
        val_high_conf = val_high_raw + corr_high
    else:
        if use_soft:
            corr = (val_cluster_weights @ q_corr_sym).T
        else:
            corr = np.zeros_like(val_low_raw)
            for c in range(n_clusters_eff):
                m = val_bin_indices == c
                if np.any(m):
                    corr[:, m] = q_corr_sym[c][:, None]

        val_low_conf = val_low_raw - corr
        val_high_conf = val_high_raw + corr

    if return_type == "bounds":
        return val_low_conf, val_high_conf

    # ---------- Build inverse-CDF interpolators ----------
    if input_median:
        probs = np.concatenate([alphas / 2, [0.5], 1 - alphas / 2])
        all_values = np.vstack([val_low_conf, val_median, val_high_conf]).T
    else:
        probs = np.concatenate([alphas / 2, 1 - alphas / 2])
        all_values = np.vstack([val_low_conf, val_high_conf]).T

    sort_idx = np.argsort(probs)
    probs_sorted = probs[sort_idx]
    all_values = all_values[:, sort_idx]
    all_values = np.maximum.accumulate(all_values, axis=1)

    
    if interpolator_kind == "cubic":
        return [
            PchipInterpolator(x = probs_sorted, y = all_values[i])
            for i in range(n_val)
        ]
    elif interpolator_kind == "linear":
        return [
            interp1d(probs_sorted, all_values[i], kind="linear", fill_value="extrapolate")
            for i in range(n_val)
        ]
    elif interpolator_kind not in ("linear", "cubic"):
        raise ValueError("Unsupported interpolator_kind. Use 'linear' or 'cubic'.")


def martingale_CPS(model, window_size, X_train, y_train, X_calibration, y_calibration, X_validation, y_validation, feats, y_scaler=None,
                    de=False, mc_diff=False, mondrian_bins=10, return_pits=False, alpha = 0.1, thres = 1e3):
    
    rng = np.random.default_rng(42)
    wr = WrapRegressor(model)

    # Accept DataFrame or numpy arrays. If numpy arrays are provided, convert to pandas
    # because this function uses .loc/.iloc and column name indexing via `feats`.
    if not hasattr(X_train, "loc"):
        if not isinstance(feats, (list, tuple)) or not all(isinstance(f, int) for f in feats):
            raise ValueError("martingale_CPS: when passing numpy arrays for X_train, 'feats' must be a list of integer column indices.")
        X_train = pd.DataFrame(X_train)
    if not hasattr(X_calibration, "loc"):
        X_calibration = pd.DataFrame(X_calibration)
    if not hasattr(X_validation, "loc"):
        X_validation = pd.DataFrame(X_validation)

    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train)
    if isinstance(y_calibration, np.ndarray):
        y_calibration = pd.Series(y_calibration)
    if isinstance(y_validation, np.ndarray):
        y_validation = pd.Series(y_validation)

    if de:
        de = DifficultyEstimator()
    else:
        de = None
    if mc_diff:
        mc_diff = MondrianCategorizer()
    else:
        mc_diff = None

    wr.fit(X_train.loc[:, feats], y_train.values)
    de.fit(X_train.loc[:, feats], y=y_train.values, k=100)


    # -------------------------------------------------
    # 1) Calibration-phase rolling recalibration to get PITs
    # -------------------------------------------------
    if return_pits:
        X_hist = X_train.iloc[-window_size:].loc[:, feats].copy()
        y_hist = y_train.iloc[-window_size:].copy()
        X_hold = X_calibration.loc[:, feats].copy()
        y_hold = y_calibration.copy()

        pit_chunks = []

        while True:
            wr.calibrate(X_hist, y_hist.values, cps=True, de=de, mc=mc_diff)
            pred_cdf_cal = wr.predict_cpds(X_hold)

            p_values_cal = wr.predict_p(X_hold, y_hold.values)
            sj_cal = SimpleJumper().apply(p_values_cal)
            crossing_cal = np.where(sj_cal > thres)[0]

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
        pit_cal = pd.Series(pit_values_cal, index=y_calibration.index, name="pit")
    
    # Cut calibration to window size for better martingale behavior
    X_cal = X_calibration.iloc[-window_size:].copy()
    y_cal = y_calibration.iloc[-window_size:].copy()
    X_val = X_validation.copy()
    y_val = y_validation.copy()

    cdfs = []
    recalibration_points = []
    while True:
        if mc_diff is not None:
            mc_diff.fit(X_cal.loc[:, feats], de=de, no_bins=mondrian_bins)
        wr.calibrate(X_cal.loc[:, feats], y_cal.values, cps=True, de=de, mc=mc_diff)
        pred_cdf = wr.predict_cpds(X_val.loc[:, feats])

        p_values = wr.predict_p(X_val.loc[:, feats], y_val.values)
        sj = SimpleJumper().apply(p_values)
        crossing = np.where(sj > thres)[0]
        

        if crossing.size == 0:
            cdfs.append(pred_cdf)
            break
        else:
            crossing_idx = np.ceil(crossing[0]/24).astype(int)*24

            if crossing_idx >= len(X_val):
                cdfs.append(pred_cdf)
                break

            recalibration_points.append(crossing_idx)
            pred_cdf = pred_cdf[:crossing_idx]
            cdfs.append(pred_cdf)

            # Move the first :crossing samples from test to calibration and cut the calibration to the window size
            X_cal = pd.concat([X_cal, X_val.iloc[:crossing_idx]], axis=0).iloc[-window_size:]
            y_cal = pd.concat([y_cal, y_val.iloc[:crossing_idx]], axis=0).iloc[-window_size:]
            X_val = X_val.iloc[crossing_idx:]
            y_val = y_val.iloc[crossing_idx:]


    if mc_diff is not None:
        cpds_scaled = pad_cdfs(cdfs)
    else:
        cpds_scaled = np.concatenate(cdfs, axis=0)
    
    if y_scaler is None:
        distributions = cpds_scaled
    else:
        distributions = y_scaler.inverse_scale(cpds_scaled.copy())
        y_validation = y_scaler.inverse_scale(y_validation)

    lower = np.nanquantile(distributions, alpha / 2, axis=1)
    upper = np.nanquantile(distributions, 1 - alpha / 2, axis=1)
    recalibration_points = np.array(recalibration_points)

    picp = met.PICP(y_validation, lower, upper)
    mpiw = met.MPIW(y_validation, lower, upper)
    crps = met.CRPS(y_true=y_validation, cpds=distributions)
    median_recalibration_gap = np.median(recalibration_points)

    df_table = pd.DataFrame({
        "window size": window_size,
        "picp": picp,
        "mpiw": mpiw,
        "median_recalibration_gap": median_recalibration_gap,
        "crps": crps,
    }, index=[0])

    if return_pits:
        return distributions, df_table, pit_cal
    else:
        return distributions, df_table, recalibration_points
    


def sample_conformalized_distribution(preds, alphas, cfg, context, window_size = None, rng=None, n_samples=10000):
    rng = np.random.default_rng() if rng is None else rng

    if window_size is None:
        try: 
            window_size = cfg.pop("window_size")
        except KeyError:
            window_size = len(context["X_calibration"])

    X_cal = context["X_calibration"].iloc[-window_size:].copy()
    y_cal = context["y_calibration"].iloc[-window_size:].copy()
    X_val = context["X_validation"].copy()
    y_val = context["y_validation"].copy()
    y_scaler = context["y_scaler"]

    preds_cal = preds.loc[y_cal.index]
    preds_val = preds.loc[y_val.index]

    interpolators = conformalize_distribution(
                                                X_cal=X_cal,
                                                y_cal=y_cal,
                                                preds_cal=preds_cal,
                                                X_val=X_val,
                                                preds_val=preds_val,
                                                alphas=alphas,
                                                **cfg,
                                            )
    

    samples = np.zeros((len(interpolators), n_samples))
    for i, intp in enumerate(interpolators):
        u = rng.uniform(size=n_samples)
        samples[i] = intp(u)
    return y_scaler.inverse_scale(samples)



class CPD():
    def __init__(self, clustering_method=None, n_clusters=1, blend_clusters=True, weighted=False,
                 random_state=42, shift_C=0.1, clip_min=1e-3, clip_max=1e3):
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters
        self.blend_clusters = blend_clusters
        self.weighted = weighted
        self.random_state = random_state
        self.shift_C = shift_C
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.cluster_models = {}
        self.cluster_pits = []
        self.cluster_pit_cdfs = []
        self.clustering_model = None
        self.cluster_edges = None
        self.cdfs = None
        self._fitted = False

    @staticmethod
    def _as_numpy_2d(values):
        return as_numpy_2d(values)

    @staticmethod
    def _as_numpy_1d(values):
        return as_numpy_1d(values)

    @staticmethod
    def _weighted_quantile(values, weights, quantile):
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        order = np.argsort(values)
        values = values[order]
        weights = np.clip(weights[order], 1e-12, None)
        cumulative = np.cumsum(weights)
        cumulative /= cumulative[-1]
        quantile = np.clip(quantile, 0.0, 1.0 - 1e-12)
        index = np.searchsorted(cumulative, quantile, side="left")
        index = min(max(index, 0), len(values) - 1)
        return values[index]

    def _fit_cluster_model(self, X_cal, preds_cal):
        X_cal = self._as_numpy_2d(X_cal)
        preds_cal = self._as_numpy_2d(preds_cal)

        if self.clustering_method == "kmeans":
            self.clustering_model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
            cluster_labels = self.clustering_model.fit_predict(X_cal)
        elif self.clustering_method == "variance":
            cal_var = preds_cal.std(axis=1)
            self.cluster_edges = np.quantile(cal_var, np.linspace(0, 1, self.n_clusters + 1))
            cluster_labels = np.digitize(cal_var, self.cluster_edges[1:-1])
        elif self.clustering_method is None:
            cluster_labels = np.zeros(len(X_cal), dtype=int)
            self.n_clusters = 1
        elif self.clustering_method == "gaussian_mixture":
            self.clustering_model = GaussianMixture(n_components=self.n_clusters, random_state=self.random_state)
            self.clustering_model.fit(X_cal)
            cluster_labels = self.clustering_model.predict(X_cal)
        else:
            raise ValueError("Unsupported clustering_method. Use 'kmeans', 'variance', 'gaussian_mixture', or None.")

        return X_cal, preds_cal, cluster_labels

    def _predict_clusters(self, X, preds_val=None):
        X = self._as_numpy_2d(X)

        if self.clustering_method == "kmeans":
            return self.clustering_model.predict(X)
        if self.clustering_method == "variance":
            if preds_val is None:
                raise ValueError("preds_val must be provided when clustering_method='variance'.")
            var = self._last_pred_std(preds_val)
            return np.digitize(var, self.cluster_edges[1:-1])
        if self.clustering_method is None:
            return np.zeros(len(X), dtype=int)
        if self.clustering_method == "gaussian_mixture":
            return self.clustering_model.predict(X)
        raise ValueError("Unsupported clustering_method. Use 'kmeans', 'variance', 'gaussian_mixture', or None.")

    def _cluster_membership_weights(self, X_val, preds_val=None):
        X_val = self._as_numpy_2d(X_val)

        if self.n_clusters == 1 or self.clustering_method is None:
            return np.ones((len(X_val), 1), dtype=float)

        if self.clustering_method == "kmeans":
            distances = self.clustering_model.transform(X_val)
            weights = 1.0 / (distances + 1e-6)
            cap = np.quantile(weights, 0.99, axis=1, keepdims=True)
            weights = np.minimum(weights, np.maximum(cap, 1e3))
            weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
            return weights

        if self.clustering_method == "gaussian_mixture":
            weights = self.clustering_model.predict_proba(X_val)
            weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
            return weights

        weights = np.zeros((len(X_val), self.n_clusters), dtype=float)
        cluster_labels = self._predict_clusters(X_val, preds_val=preds_val)
        weights[np.arange(len(X_val)), cluster_labels] = 1.0
        return weights

    def _last_pred_std(self, preds):
        preds = self._as_numpy_2d(preds)
        return preds.std(axis=1)

    def _covariate_weights(self, X_cal, X_val):
        X_cal = self._as_numpy_2d(X_cal)
        X_val = self._as_numpy_2d(X_val)
        X_domain = np.vstack([X_cal, X_val])
        y_domain = np.concatenate([np.zeros(len(X_cal)), np.ones(len(X_val))])
        clf = LogisticRegression(C=self.shift_C, max_iter=500, class_weight="balanced")
        clf.fit(X_domain, y_domain)
        p_target = np.clip(clf.predict_proba(X_cal)[:, 1], 1e-12, 1 - 1e-12)
        ratio = p_target / (1.0 - p_target)
        return np.clip(ratio, self.clip_min, self.clip_max)

    def calibrate(self, X_cal, y_cal, preds_cal, X_val=None, weighted=None):
        X_cal = self._as_numpy_2d(X_cal)
        y_cal = self._as_numpy_1d(y_cal)
        preds_cal = self._as_numpy_2d(preds_cal)

        if weighted is None:
            weighted = self.weighted

        X_cal, preds_cal, cluster_labels = self._fit_cluster_model(X_cal, preds_cal)

        if weighted:
            if X_val is None:
                raise ValueError("X_val must be provided when weighted=True so covariate weights can be estimated.")
            cal_weights = self._covariate_weights(X_cal, X_val)
        else:
            cal_weights = np.ones(len(y_cal), dtype=float)

        cluster_pits = []
        cluster_pit_cdfs = []
        global_pits = np.mean(preds_cal <= y_cal[:, np.newaxis], axis=1)
        global_order = np.argsort(global_pits)
        global_support = global_pits[global_order]
        if weighted:
            global_weights = np.clip(cal_weights[global_order], 1e-12, None)
            global_weights /= np.sum(global_weights)
            global_cdf = np.cumsum(global_weights)
        else:
            global_cdf = np.linspace(1.0 / (len(global_support) + 1), len(global_support) / (len(global_support) + 1), len(global_support))

        for i in range(self.n_clusters):
            cluster_indices = np.where(cluster_labels == i)[0]
            if len(cluster_indices) == 0:
                cluster_pits.append(global_support)
                cluster_pit_cdfs.append(global_cdf)
                continue

            y_cal_cluster = y_cal[cluster_indices]
            preds_cal_cluster = preds_cal[cluster_indices]
            weights_cluster = cal_weights[cluster_indices]

            pit_scores_cluster = np.mean(preds_cal_cluster <= y_cal_cluster[:, np.newaxis], axis=1)
            pit_scores_cluster = np.asarray(pit_scores_cluster, dtype=float)
            order = np.argsort(pit_scores_cluster)
            sorted_pits = pit_scores_cluster[order]
            sorted_weights = np.clip(weights_cluster[order], 1e-12, None)
            sorted_weights /= np.sum(sorted_weights)
            cdf_y = np.cumsum(sorted_weights)

            cluster_pits.append(sorted_pits)
            cluster_pit_cdfs.append(cdf_y)

        self.cluster_pits = cluster_pits
        self.cluster_pit_cdfs = cluster_pit_cdfs
        self.cluster_weights = cal_weights
        self._fitted = True
        return self

    def _conformal_cdf_for_cluster(self, u_raw, cluster_index):
        return np.array([
            np.interp(u_row, self.cluster_pits[cluster_index], self.cluster_pit_cdfs[cluster_index], left=0.0, right=1.0)
            for u_row in u_raw
        ])

    def predict_conformal_cdf_batch(self, X_val, y_eval, preds_val, blend_clusters=None):
        if not self._fitted:
            raise RuntimeError("Model must be calibrated before calling predict_conformal_cdf_batch.")

        X_val = self._as_numpy_2d(X_val)
        y_eval = self._as_numpy_1d(y_eval)
        preds_val = self._as_numpy_2d(preds_val)

        if blend_clusters is None:
            blend_clusters = self.blend_clusters

        cluster_weights_val = self._cluster_membership_weights(X_val, preds_val)
        cdfs = np.zeros((X_val.shape[0], y_eval.shape[0]))

        if self.n_clusters == 1 or not blend_clusters:
            cluster_labels_val = self._predict_clusters(X_val, preds_val=preds_val)
            for i in range(self.n_clusters):
                cluster_indices_val = np.where(cluster_labels_val == i)[0]
                if len(cluster_indices_val) == 0:
                    continue
                preds_val_cluster = preds_val[cluster_indices_val]
                u_raw_cluster = np.mean(preds_val_cluster[:, np.newaxis, :] <= y_eval[np.newaxis, :, np.newaxis], axis=2)
                conformal_cdfs_cluster = self._conformal_cdf_for_cluster(u_raw_cluster, i)
                cdfs[cluster_indices_val] = conformal_cdfs_cluster
        else:
            u_raw = np.mean(preds_val[:, np.newaxis, :] <= y_eval[np.newaxis, :, np.newaxis], axis=2)
            for i in range(self.n_clusters):
                conformal_cdfs_cluster = self._conformal_cdf_for_cluster(u_raw, i)
                cdfs += cluster_weights_val[:, i][:, np.newaxis] * conformal_cdfs_cluster

        self.cdfs = np.clip(cdfs, 0.0, 1.0)
        return self.cdfs

    def sample_from_conformal_cdfs(self, y_eval, y_scaler=None, n_samples=1000, seed=None):
        if self.cdfs is None:
            raise RuntimeError("predict_conformal_cdf_batch must be called before sampling.")

        y_eval = self._as_numpy_1d(y_eval)
        n_test = self.cdfs.shape[0]
        rng = np.random.default_rng(seed)
        u_sim = rng.uniform(0.0, 1.0, size=(n_test, n_samples))
        samples_scaled = np.zeros((n_test, n_samples))

        for i in range(n_test):
            samples_scaled[i, :] = np.interp(u_sim[i, :], self.cdfs[i, :], y_eval)

        if y_scaler is not None:
            return y_scaler.inverse_scale(samples_scaled)
        return samples_scaled
