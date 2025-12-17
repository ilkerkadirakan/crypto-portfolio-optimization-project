"""
Rolling statistical moment calculations for cryptocurrency return series.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf

CVaR_ALPHA = 0.95
WINDOWS = {"1H": 4320, "1D": 180}


def _empirical_cvar(arr: np.ndarray, alpha: float = CVaR_ALPHA) -> float:
    """
    Compute empirical CVaR as the mean of the worst (1-alpha) quantile of returns.

    Parameters
    ----------
    arr : np.ndarray
        Array of returns from the rolling window.
    alpha : float, optional
        Confidence level for CVaR calculation (default 0.95).

    Returns
    -------
    float
        Average of the worst (1-alpha) fraction of returns, NaN if insufficient data.
    """
    clean = arr[~np.isnan(arr)]
    if clean.size == 0:
        return np.nan

    tail_count = max(int(np.ceil((1.0 - alpha) * clean.size)), 1)
    sorted_returns = np.sort(clean)
    return float(np.mean(sorted_returns[:tail_count]))


def _ledoit_variance(window_df: pd.DataFrame) -> np.ndarray:
    """
    Estimate per-asset variances using Ledoit-Wolf shrinkage on a window of returns.

    Parameters
    ----------
    window_df : pd.DataFrame
        Rolling window of returns with assets in columns.

    Returns
    -------
    np.ndarray
        Array of variance estimates aligned with the input columns.
    """
    valid_columns = [col for col in window_df.columns if window_df[col].notna().sum() >= 2]
    if not valid_columns:
        return np.full(window_df.shape[1], np.nan, dtype=float)

    clean_window = window_df[valid_columns].dropna(axis=0, how="any")
    if clean_window.shape[0] < 2:
        return np.full(window_df.shape[1], np.nan, dtype=float)

    estimator = LedoitWolf()
    estimator.fit(clean_window.values)
    diag_variance = np.diag(estimator.covariance_)

    full_variance = np.full(window_df.shape[1], np.nan, dtype=float)
    for col, value in zip(valid_columns, diag_variance, strict=False):
        full_variance[window_df.columns.get_loc(col)] = value
    return full_variance


def calc_moments(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Derive rolling statistical moments required by the optimization pipeline.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns indexed by datetime, frequency inferred from context.
    window : int
        Size of the rolling window expressed in number of observations.

    Returns
    -------
    pd.DataFrame
        Multi-index columns (moment, asset) containing mean, variance, skewness, kurtosis, CVaR.

    TODO
    ----
    - Extend outputs with cross-asset covariance matrices for downstream solvers.
    - Add performance optimizations for large hourly datasets if profiling indicates need.
    """
    if returns.empty:
        raise ValueError("Input returns DataFrame is empty; cannot compute moments.")

    data = (
        returns.sort_index()
        .loc[:, ~returns.columns.duplicated()]
        .astype(float)
    )

    rolling = data.rolling(window=window, min_periods=window)

    mean_df = rolling.mean()
    skew_df = rolling.apply(
        lambda arr: stats.skew(arr, nan_policy="omit", bias=False),
        raw=True,
    )
    kurt_df = rolling.apply(
        lambda arr: stats.kurtosis(arr, nan_policy="omit", bias=False, fisher=False),
        raw=True,
    )
    cvar_df = rolling.apply(
        lambda arr: _empirical_cvar(arr, alpha=CVaR_ALPHA),
        raw=True,
    )

    variance_records = []
    variance_index = []
    for i in range(window - 1, len(data)):
        window_slice = data.iloc[i - window + 1 : i + 1]
        variance_records.append(_ledoit_variance(window_slice))
        variance_index.append(data.index[i])

    variance_df = pd.DataFrame(
        variance_records,
        index=pd.Index(variance_index, name=data.index.name),
        columns=data.columns,
    )
    variance_df = variance_df.reindex(data.index)

    moments = {
        "mean": mean_df,
        "variance": variance_df,
        "skewness": skew_df,
        "kurtosis": kurt_df,
        "cvar": cvar_df,
    }
    return pd.concat(moments, axis=1)


def calc_all_moments(processed_dir: Path | None = None) -> Tuple[Path, Path]:
    """
    Execute rolling moment calculations for hourly and daily datasets.

    Parameters
    ----------
    processed_dir : Path | None, optional
        Directory containing the return parquet files; defaults to data/processed.

    Returns
    -------
    Tuple[Path, Path]
        Paths to the generated hourly and daily moment parquet files.

    TODO
    ----
    - Load configuration values (paths, alpha) from configs/params.yaml when available.
    - Introduce caching for repeated invocations once dataset size is confirmed.
    """
    project_root = Path(__file__).resolve().parents[1]
    base_dir = processed_dir or project_root / "data" / "processed"
    base_dir.mkdir(parents=True, exist_ok=True)

    return_paths: Dict[str, Path] = {
        "1H": base_dir / "returns_1h.parquet",
        "1D": base_dir / "returns_1d.parquet",
    }

    output_paths: Dict[str, Path] = {}
    for freq, input_path in return_paths.items():
        if not input_path.exists():
            raise FileNotFoundError(f"Missing returns file for {freq}: {input_path}")

        returns_df = pd.read_parquet(input_path)
        moments_df = calc_moments(returns_df, window=WINDOWS[freq])

        output_path = base_dir / f"moments_{freq.lower()}.parquet"
        moments_df.to_parquet(output_path)
        output_paths[freq] = output_path

        print(f"[moment_calc] {freq} moments shape: {moments_df.shape}")

    return output_paths["1H"], output_paths["1D"]


if __name__ == "__main__":
    calc_all_moments()
