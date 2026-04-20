"""
Optimization model definitions for mean-variance, MVSK, and MCVaRSK portfolios.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import cvxpy as cp
import numpy as np
import yaml

SUPPORTED_MODELS = {"MV", "MVSK", "MCVARSK"}

# Keep optimization output clean during long batch runs.
# We suppress only cvxpy-originated user warnings (e.g., "solution may be inaccurate").
warnings.filterwarnings("ignore", category=UserWarning, module=r"cvxpy")


def _load_params() -> Dict[str, dict]:
    """
    Load configuration parameters from configs/params.yaml if available.

    Returns
    -------
    Dict[str, dict]
        Configuration dictionary, or an empty dict when the file is missing.
    """
    config_path = Path(__file__).resolve().parents[1] / "configs" / "params.yaml"
    if not config_path.exists():
        warnings.warn(f"Configuration file not found at {config_path}. Using defaults.")
        return {}

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def choose_solver(prefer: str = "ECOS") -> str:
    """
    Select an installed cvxpy solver, preferring the requested backend.

    Parameters
    ----------
    prefer : str, optional
        Solver name to prefer (default 'ECOS').

    Returns
    -------
    str
        Name of an installed solver suitable for cvxpy problems.

    Raises
    ------
    RuntimeError
        If no compatible solver is available in the environment.
    """
    available = {solver.upper() for solver in cp.installed_solvers()}
    candidates: Tuple[str, ...] = (prefer.upper(), "ECOS", "SCS")

    for candidate in candidates:
        if candidate in available:
            return candidate

    if available:
        return next(iter(available))

    raise RuntimeError("No cvxpy solvers are installed. Unable to solve the problem.")


def _portfolio_bounds(portfolio_cfg: Dict[str, object]) -> Tuple[Optional[float], Optional[float]]:
    """
    Resolve lower/upper weight bounds from the configuration dictionary.

    Parameters
    ----------
    portfolio_cfg : Dict[str, object]
        Portfolio configuration block.

    Returns
    -------
    Tuple[Optional[float], Optional[float]]
        Scalar lower and upper weight bounds (None if unbounded).
    """
    bounds = portfolio_cfg.get("weight_bounds", {})
    if isinstance(bounds, dict):
        default = bounds.get("default")
        if isinstance(default, (list, tuple)) and len(default) == 2:
            return float(default[0]), float(default[1])

    min_weight = portfolio_cfg.get("min_weight")
    max_weight = portfolio_cfg.get("max_weight")
    lower = float(min_weight) if min_weight is not None else None
    upper = float(max_weight) if max_weight is not None else None
    return lower, upper


def _linear_moment_term(values: Optional[Sequence[float]], weights: cp.Expression, label: str) -> cp.Expression:
    """
    Convert a moment input into a linear form compatible with the objective.

    Parameters
    ----------
    values : Optional[Sequence[float]]
        Moment values aligned with asset order, or scalar for portfolio-level input.
    weights : cp.Expression
        cvxpy variable representing asset weights.
    label : str
        Name of the moment for error diagnostics.

    Returns
    -------
    cp.Expression
        Linear expression representing the contribution of the moment.
    """
    if values is None:
        return cp.Constant(0.0)

    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return cp.Constant(float(arr))
    if arr.ndim == 1:
        if arr.size != weights.shape[0]:
            raise ValueError(f"{label} vector length {arr.size} does not match weights dimension {weights.shape[0]}.")
        return cp.sum(cp.multiply(arr, weights))

    raise ValueError(f"{label} input must be scalar or 1-D sequence.")


def _solver_sequence(solver_cfg: Dict[str, object]) -> Iterable[str]:
    """
    Build a sequence of solver preferences based on configuration.

    Parameters
    ----------
    solver_cfg : Dict[str, object]
        Solver configuration block.

    Returns
    -------
    Iterable[str]
        Sequence of solver names to attempt in order.
    """
    primary = str(solver_cfg.get("primary", "ECOS")).upper()
    secondary = solver_cfg.get("secondary")
    if secondary is not None:
        secondary = str(secondary).upper()

    sequence = [primary]
    if secondary and secondary not in sequence:
        sequence.append(secondary)
    for fallback in ("ECOS", "SCS"):
        if fallback not in sequence:
            sequence.append(fallback)
    return sequence


def solve_portfolio(
    model_name: str,
    mu: Sequence[float],
    sigma: Sequence[Sequence[float]],
    skew: Optional[Sequence[float]] = None,
    kurt: Optional[Sequence[float]] = None,
    cvar_series: Optional[Sequence[float]] = None,
    params: Optional[Dict[str, dict]] = None,
) -> np.ndarray:
    """
    Solve an optimization model for the requested portfolio formulation.

    Parameters
    ----------
    model_name : str
        Name of the optimization model ('MV', 'MVSK', 'MCVaRSK').
    mu : Sequence[float]
        Expected returns vector aligned with asset order.
    sigma : Sequence[Sequence[float]]
        Covariance matrix (Ledoit-Wolf or other estimator).
    skew : Optional[Sequence[float]], optional
        Asset-level skewness contributions (required for MVSK and MCVaRSK).
    kurt : Optional[Sequence[float]], optional
        Asset-level kurtosis contributions (required for MVSK and MCVaRSK).
    cvar_series : Optional[Sequence[float]], optional
        Asset-level CVaR contributions or scalar portfolio CVaR (required for MCVaRSK).
    params : Optional[Dict[str, dict]], optional
        Configuration dictionary providing objective weights and constraints.

    Returns
    -------
    np.ndarray
        Optimized asset weights summing to unity.

    Raises
    ------
    ValueError
        If inputs are inconsistent with the selected model.
    RuntimeError
        If the solver fails to achieve an optimal solution.
    """
    if model_name is None:
        raise ValueError("Model name must be provided.")

    model_key = model_name.upper()
    if model_key not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{model_name}'. Supported models: {sorted(SUPPORTED_MODELS)}.")

    mu_vec = np.asarray(mu, dtype=float).reshape(-1)
    sigma_mat = np.asarray(sigma, dtype=float)

    if sigma_mat.shape[0] != sigma_mat.shape[1]:
        raise ValueError("Covariance matrix must be square.")
    if sigma_mat.shape[0] != mu_vec.size:
        raise ValueError("Covariance matrix dimension must match the length of mu.")

    # Enhanced input validation
    if not np.isfinite(mu_vec).all():
        raise ValueError("Mean returns vector contains NaN or Inf values.")
    if not np.isfinite(sigma_mat).all():
        raise ValueError("Covariance matrix contains NaN or Inf values.")

    # Check for numerical issues in covariance matrix
    try:
        eigenvalues = np.linalg.eigvalsh(sigma_mat)
        min_eigenvalue = np.min(eigenvalues)
        if min_eigenvalue <= 0:
            # Add small regularization to ensure positive definiteness
            regularization = abs(min_eigenvalue) + 1e-8
            sigma_mat = sigma_mat + regularization * np.eye(sigma_mat.shape[0])
    except np.linalg.LinAlgError:
        raise ValueError("Covariance matrix is severely ill-conditioned.")

    # Validate moment inputs for higher-order models
    if model_key in {"MVSK", "MCVARSK"}:
        if skew is not None:
            skew_arr = np.asarray(skew, dtype=float).reshape(-1)
            if not np.isfinite(skew_arr).all():
                raise ValueError("Skewness vector contains NaN or Inf values.")
        if kurt is not None:
            kurt_arr = np.asarray(kurt, dtype=float).reshape(-1)
            if not np.isfinite(kurt_arr).all():
                raise ValueError("Kurtosis vector contains NaN or Inf values.")

    if model_key == "MCVARSK" and cvar_series is not None:
        cvar_arr = np.asarray(cvar_series, dtype=float).reshape(-1)
        if not np.isfinite(cvar_arr).all():
            raise ValueError("CVaR vector contains NaN or Inf values.")

    n_assets = mu_vec.size
    config = params or _load_params()
    portfolio_cfg = config.get("portfolio", {})
    models_cfg = config.get("models", {})
    objectives_cfg = models_cfg.get("objectives", {})
    solver_cfg = models_cfg.get("solver", {})
    solver_options = solver_cfg.get("options", {})

    lower_bound, upper_bound = _portfolio_bounds(portfolio_cfg)
    allow_short = bool(portfolio_cfg.get("allow_short", False))

    weights = cp.Variable(n_assets)
    constraints = [cp.sum(weights) == 1]

    if allow_short:
        if lower_bound is not None:
            constraints.append(weights >= lower_bound)
    else:
        lower_value = 0.0 if lower_bound is None else max(lower_bound, 0.0)
        constraints.append(weights >= lower_value)

    if upper_bound is not None:
        constraints.append(weights <= upper_bound)

    variance_term = cp.quad_form(weights, sigma_mat)
    return_term = cp.sum(cp.multiply(mu_vec, weights))

    model_cfg = objectives_cfg.get(model_key.lower(), {})

    lambda_variance = float(model_cfg.get("lambda_variance", 1.0))
    lambda_mean = float(model_cfg.get("lambda_mean", 1.0))

    if model_key == "MV":
        objective_expr = lambda_variance * variance_term - lambda_mean * return_term
    elif model_key == "MVSK":
        if skew is None or kurt is None:
            raise ValueError("Model 'MVSK' requires skew and kurt inputs.")
        skew_term = _linear_moment_term(skew, weights, "skewness")
        kurt_term = _linear_moment_term(kurt, weights, "kurtosis")
        lambda_skew = float(model_cfg.get("lambda_skew", 1.0))
        lambda_kurt = float(model_cfg.get("lambda_kurtosis", 1.0))
        objective_expr = (
            lambda_variance * variance_term
            - lambda_mean * return_term
            - lambda_skew * skew_term
            + lambda_kurt * kurt_term
        )
    else:  # MCVaRSK
        if skew is None or kurt is None:
            raise ValueError("Model 'MCVaRSK' requires skew and kurt inputs.")
        if cvar_series is None:
            raise ValueError("Model 'MCVaRSK' requires cvar_series input.")
        skew_term = _linear_moment_term(skew, weights, "skewness")
        kurt_term = _linear_moment_term(kurt, weights, "kurtosis")
        cvar_term = _linear_moment_term(cvar_series, weights, "cvar")
        lambda_skew = float(model_cfg.get("lambda_skew", 1.0))
        lambda_kurt = float(model_cfg.get("lambda_kurtosis", 1.0))
        lambda_cvar = float(model_cfg.get("lambda_cvar", 1.0))
        objective_expr = (
            lambda_cvar * cvar_term
            - lambda_mean * return_term
            - lambda_skew * skew_term
            + lambda_kurt * kurt_term
        )
    # Validate problem formulation before solving
    try:
        problem = cp.Problem(cp.Minimize(objective_expr), constraints)
    except Exception as exc:
        raise ValueError(f"Error constructing optimization problem: {exc}")

    # Check if problem is DCP (Disciplined Convex Programming)
    if not problem.is_dcp():
        raise ValueError(
            f"Problem is not DCP (Disciplined Convex Programming). "
            f"This usually indicates an issue with the objective formulation."
        )

    last_error: Optional[Exception] = None
    status: Optional[str] = None

    # Filter solver options to only include CVXPY-compatible parameters
    # Remove solver-specific parameters that might cause parsing errors
    cvxpy_options = {}
    if "max_iters" in solver_options:
        cvxpy_options["max_iters"] = solver_options["max_iters"]
    if "verbose" in solver_options:
        cvxpy_options["verbose"] = solver_options["verbose"]

    for solver_name in _solver_sequence(solver_cfg):
        try:
            selected_solver = choose_solver(solver_name)
        except RuntimeError as exc:
            last_error = exc
            continue

        try:
            # Try with filtered options first, then without options if it fails
            try:
                problem.solve(solver=selected_solver, **cvxpy_options)
            except Exception as solve_exc:
                # If options cause issues, try without them
                if cvxpy_options:
                    problem.solve(solver=selected_solver)
                else:
                    raise solve_exc

            status = problem.status
        except cp.DCPError as exc:
            # DCP violation error - this is a formulation problem
            raise ValueError(f"DCP violation in problem formulation: {exc}")
        except cp.SolverError as exc:
            last_error = exc
            status = None
            continue
        except Exception as exc:
            # Catch any other errors (including parsing errors)
            last_error = exc
            status = None
            continue

        if status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            break

    if status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or weights.value is None:
        error_msg = f"Solver failed to find an optimal solution. Status: {status}"
        if last_error:
            error_msg += f" Last error: {last_error}"
        raise RuntimeError(error_msg)

    result = np.asarray(weights.value, dtype=float).reshape(-1)

    # Final validation
    if not np.isfinite(result).all():
        raise RuntimeError("Solver returned non-finite weights.")

    return result


__all__ = [
    "choose_solver",
    "solve_portfolio",
]
