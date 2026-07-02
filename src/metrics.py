import numpy as np
from scipy.special import erf


# ---------- Helper functions ----------


def _as_1d(*arrays): # Helper function to ensure inputs are 1D numpy arrays and have the same length
    out = [np.asarray(a).reshape(-1) for a in arrays]
    n = out[0].shape[0]
    if any(a.shape[0] != n for a in out):
        raise ValueError("All inputs must have the same length.")
    return out


def _check_alpha(alpha: float): # Helper function to validate the alpha parameter
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")


def _norm_scale(y_true, method="range", eps=1e-8): # Helper function to compute the normalization scale for PINAW and PINMW
    y = np.asarray(y_true).reshape(-1)
    if method == "range":
        s = np.max(y) - np.min(y)
    elif method == "iqr":
        q75, q25 = np.percentile(y, [75, 25])
        s = q75 - q25
    elif method == "std":
        s = np.std(y)
    else:
        raise ValueError("method must be one of: 'range', 'iqr', 'std'")
    return max(float(s), eps)


# ---------- Point prediction metrics ----------

def MSE(y_true, y_pred):
    # MSE: Mean Squared Error (lower is better)
    y_true, y_pred = _as_1d(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def MAE(y_true, y_pred):
    # MAE: Mean Absolute Error (lower is better)
    y_true, y_pred = _as_1d(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def RMSE(y_true, y_pred):
    # RMSE: Root Mean Squared Error (lower is better)
    y_true, y_pred = _as_1d(y_true, y_pred)
    return float(np.sqrt(MSE(y_true, y_pred)))


def MedAE(y_true, y_pred):
    # MedAE: Median Absolute Error (lower is better)
    y_true, y_pred = _as_1d(y_true, y_pred)
    return float(np.median(np.abs(y_true - y_pred)))


def R2(y_true, y_pred, eps=1e-12):
    # R2: Coefficient of Determination (higher is better)
    y_true, y_pred = _as_1d(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + eps))


# ---------- Interval / conformal metrics ----------

def PICP(y_true, lower_bound, upper_bound):
    # PICP: Prediction Interval Coverage Probability (higher is better)
    y_true, lower_bound, upper_bound = _as_1d(y_true, lower_bound, upper_bound)
    return float(np.mean((y_true >= lower_bound) & (y_true <= upper_bound)))


def MPIW(y_true, lower_bound, upper_bound):
    # MPIW: Mean Prediction Interval Width (lower is better)
    _, lower_bound, upper_bound = _as_1d(y_true, lower_bound, upper_bound)
    return float(np.mean(upper_bound - lower_bound))


def MedPIW(y_true, lower_bound, upper_bound):
    # MedPIW: Median Prediction Interval Width (lower is better)
    _, lower_bound, upper_bound = _as_1d(y_true, lower_bound, upper_bound)
    return float(np.median(upper_bound - lower_bound))


def PINAW(y_true, lower_bound, upper_bound, norm="range"):
    # PINAW: Prediction Interval Normalized Average Width (lower is better)
    y_true, lower_bound, upper_bound = _as_1d(y_true, lower_bound, upper_bound)
    width = upper_bound - lower_bound
    return float(np.mean(width) / _norm_scale(y_true, method=norm))


def PINMW(y_true, lower_bound, upper_bound, norm="range"):
    # PINMW: Prediction Interval Normalized Median Width (lower is better)
    y_true, lower_bound, upper_bound = _as_1d(y_true, lower_bound, upper_bound)
    width = upper_bound - lower_bound
    return float(np.median(width) / _norm_scale(y_true, method=norm))


def CoverageGap(y_true, lower_bound, upper_bound, alpha=0.05):
    # CoverageGap: Absolute difference between PICP and target coverage (lower is better)
    _check_alpha(alpha)
    y_true, lower_bound, upper_bound = _as_1d(y_true, lower_bound, upper_bound)
    return float(abs(PICP(y_true, lower_bound, upper_bound) - (1.0 - alpha)))


def TailMiscoverage(y_true, lower_bound, upper_bound):
    # TailMiscoverage: Proportion of true values below lower bound and above upper bound (lower is better)
    y_true, lower_bound, upper_bound = _as_1d(y_true, lower_bound, upper_bound)
    lower_miss = float(np.mean(y_true < lower_bound))
    upper_miss = float(np.mean(y_true > upper_bound))
    return {"lower_miss": lower_miss, "upper_miss": upper_miss}


def CWC(y_true, lower_bound, upper_bound, alpha=0.1, eta=0.5, norm="range"):
    """
    Coverage Width-based Criterion (lower is better).

    One-sided penalty: penalize only under-coverage.
    """
    _check_alpha(alpha)
    picp = PICP(y_true, lower_bound, upper_bound)
    pinaw = PINAW(y_true, lower_bound, upper_bound, norm=norm)
    target = 1.0 - alpha

    if picp >= target:
        return float(pinaw)

    # Standard one-sided CWC form
    return float(pinaw * (1.0 + np.exp(-eta * (picp - target))))


def WinklerIntervalScore(y_true, lower_bound, upper_bound, alpha=0.05):
    """
    Mean Winkler interval score (lower is better).
    """
    _check_alpha(alpha)
    y_true, lower_bound, upper_bound = _as_1d(y_true, lower_bound, upper_bound)

    width = upper_bound - lower_bound
    below = (y_true < lower_bound).astype(float)
    above = (y_true > upper_bound).astype(float)

    score = (
        width
        + (2.0 / alpha) * (lower_bound - y_true) * below
        + (2.0 / alpha) * (y_true - upper_bound) * above
    )
    return float(np.mean(score))



def _normal_crps(y_true, mu, sigma, eps=1e-12):
    """Closed-form CRPS for a normal predictive distribution."""
    y_true, mu, sigma = _as_1d(y_true, mu, sigma)
    sigma = np.maximum(sigma, eps)
    z = (y_true - mu) / sigma
    phi = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    Phi = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    return sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))


def _crps_from_samples(y_true, samples):
    """Empirical CRPS from samples with equal weights."""
    y_true = np.asarray(y_true).reshape(-1)
    x = np.asarray(samples)
    if x.ndim != 2:
        raise ValueError("samples must be a 2D array with shape (n_obs, n_samples)")

    if x.shape[0] != y_true.shape[0] and x.shape[1] == y_true.shape[0]:
        x = x.T
    if x.shape[0] != y_true.shape[0]:
        raise ValueError("samples must have one row per observation")

    m = x.shape[1]
    if m < 2:
        raise ValueError("samples must contain at least 2 draws per observation")

    term1 = np.mean(np.abs(x - y_true[:, None]), axis=1)

    # Efficient formula for E|X-X'| under equal weights.
    x_sorted = np.sort(x, axis=1)
    coeff = (2.0 * np.arange(1, m + 1) - m - 1.0)
    e_xx = (2.0 / (m**2)) * (x_sorted * coeff).sum(axis=1)
    return term1 - 0.5 * e_xx


def _crps_from_discrete_cpds(y_true, cpds, eps=1e-12):
    """
    CRPS for discrete predictive distributions.

    Supported per-observation formats inside `cpds`:
    1) dict with keys:
       - `support` and `mass`, or
       - `support` and `cdf`
    2) array-like of support values only (e.g., CREPES CPS output), where
       each support point is assigned equal mass.
    """
    y_true = np.asarray(y_true).reshape(-1)
    if len(cpds) != y_true.shape[0]:
        raise ValueError("cpds must have the same length as y_true")

    out = np.zeros_like(y_true, dtype=float)
    for i, (y_i, cpd) in enumerate(zip(y_true, cpds)):
        if cpd is None:
            raise ValueError(f"cpd at index {i} is None")

        if isinstance(cpd, dict):
            if "support" not in cpd:
                raise ValueError("Each cpd dict must include a 'support' key")

            support = np.asarray(cpd["support"], dtype=object).reshape(-1)
            if support.size == 0:
                raise ValueError("cpd support cannot be empty")

            if "mass" in cpd:
                mass = np.asarray(cpd["mass"], dtype=float).reshape(-1)
            elif "cdf" in cpd:
                cdf = np.asarray(cpd["cdf"], dtype=float).reshape(-1)
                mass = np.diff(np.concatenate(([0.0], cdf)))
            else:
                raise ValueError("Each cpd dict must include either 'mass' or 'cdf'")
        else:
            # CREPES CPS-style: array of support values with implicit equal weights.
            support = np.asarray(cpd, dtype=object).reshape(-1)
            if support.size == 0:
                raise ValueError("cpd support cannot be empty")
            mass = np.full(support.shape[0], 1.0 / support.shape[0], dtype=float)

        # Some CPS outputs can contain None/non-finite support points (e.g., sparse bins).
        # Filter them out and keep masses aligned; if nothing remains, fail this CPD.
        def _to_float_or_nan(v):
            if v is None:
                return np.nan
            try:
                return float(v)
            except (TypeError, ValueError):
                return np.nan

        support_num = np.array([_to_float_or_nan(v) for v in support], dtype=float)
        finite_mask = np.isfinite(support_num)
        if not np.all(finite_mask):
            support_num = support_num[finite_mask]
            mass = np.asarray(mass, dtype=float).reshape(-1)[finite_mask]
        support = support_num

        if mass.shape[0] != support.shape[0]:
            raise ValueError("cpd support and mass/cdf must have the same length")
        if support.size == 0:
            raise ValueError("cpd support has no finite numeric values")

        # For numeric robustness, if the source was cdf and contains tiny negatives
        # after differencing, clip and renormalize below.
        order = np.argsort(support)
        x = support[order]
        w = np.maximum(mass[order], 0.0)
        s = w.sum()
        if s <= eps:
            raise ValueError("cpd mass must sum to a positive value")
        w = w / s

        # E|X-y|
        term1 = np.sum(w * np.abs(x - y_i))

        # E|X-X'| for weighted discrete distribution, O(k).
        cum_w = 0.0
        cum_xw = 0.0
        pair_sum = 0.0
        for xk, wk in zip(x, w):
            pair_sum += wk * (xk * cum_w - cum_xw)
            cum_w += wk
            cum_xw += wk * xk
        e_xx = 2.0 * pair_sum

        out[i] = term1 - 0.5 * e_xx

    return out


def CRPS(
    y_true,
    mu=None,
    sigma=None,
    samples=None,
    cpds=None,
    cdf_fn=None,
    x_grid=None,
):
    """
    Mean Continuous Ranked Probability Score (CRPS), lower is better.

    Supported forecast representations (choose exactly one):
    1) Normal predictive distribution: pass `mu` and `sigma`.
    2) Sample/draw-based forecast (e.g., BART): pass `samples` with shape
       (n_obs, n_samples) or (n_samples, n_obs).
     3) CPS/discrete predictive distributions: pass `cpds` as either:
         - list of dicts (`support` + `mass` or `cdf`), or
         - list of support arrays (implicit equal weights; CREPES CPS style).
    4) Generic CDF function: pass `cdf_fn` and `x_grid`.
       `cdf_fn` must accept (x_grid, i) and return CDF values for observation i.
    """
    y_true = np.asarray(y_true).reshape(-1)

    provided = [
        mu is not None or sigma is not None,
        samples is not None,
        cpds is not None,
        cdf_fn is not None,
    ]
    if sum(provided) != 1:
        raise ValueError(
            "Provide exactly one forecast representation: (mu and sigma), samples, cpds, or cdf_fn"
        )

    if mu is not None or sigma is not None:
        if mu is None or sigma is None:
            raise ValueError("Both mu and sigma must be provided together")
        crps_values = _normal_crps(y_true, mu, sigma)
        return float(np.mean(crps_values))

    if samples is not None:
        crps_values = _crps_from_samples(y_true, samples)
        return float(np.mean(crps_values))

    if cpds is not None:
        crps_values = _crps_from_discrete_cpds(y_true, cpds)
        return float(np.mean(crps_values))

    # Generic CDF route via numerical integration.
    if x_grid is None:
        raise ValueError("x_grid must be provided when using cdf_fn")
    x_grid = np.asarray(x_grid).reshape(-1)
    if x_grid.size < 2:
        raise ValueError("x_grid must contain at least two points")

    crps_values = np.zeros_like(y_true, dtype=float)
    for i, y_i in enumerate(y_true):
        F = np.asarray(cdf_fn(x_grid, i)).reshape(-1)
        if F.shape[0] != x_grid.shape[0]:
            raise ValueError("cdf_fn output must have same length as x_grid")
        H = (x_grid >= y_i).astype(float)
        crps_values[i] = np.trapz((F - H) ** 2, x_grid)

    return float(np.mean(crps_values))


def randomized_pit_from_samples(samples, y_obs, rng=None, atol=1e-12):
    rng = np.random.default_rng() if rng is None else rng
    s = np.asarray(samples)

    p_less = np.mean(s < y_obs - atol)
    p_equal = np.mean(np.isclose(s, y_obs, atol=atol, rtol=0.0))

    return p_less + rng.uniform() * p_equal


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


def to_pvalue(y_val):
    y_val = np.asarray(y_val)
    P = [1]
    for i in range(1, len(y_val)):
        p = (np.sum(y_val[:i] < y_val[i]) + np.random.uniform(0, 1) * np.sum(y_val[i] == y_val[:i])) / (i+1)
        P.append(p)
    return np.array(P)
