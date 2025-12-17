"""
Machine learning forecasters for future portfolio moments and tail-risk metrics.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - LightGBM optional dependency
    lgb = None

TARGET_MOMENTS: Tuple[str, ...] = ("mean", "variance", "cvar")
FEATURE_MOMENTS: Tuple[str, ...] = ("mean", "variance", "cvar", "skewness", "kurtosis")
ESTIMATOR_KEY_MAP: Dict[str, str] = {
    "LightGBM": "lightgbm",
    "RandomForest": "random_forest",
}


def _load_params() -> Dict[str, dict]:
    """
    Load global configuration parameters from configs/params.yaml.

    Returns
    -------
    Dict[str, dict]
        Parsed configuration dictionary; empty dict if file missing.
    """
    config_path = Path(__file__).resolve().parents[1] / "configs" / "params.yaml"
    if not config_path.exists():
        warnings.warn(f"Configuration file not found at {config_path}. Using defaults.")
        return {}

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _build_model(model_name: str, params: Dict[str, object]) -> object:
    """
    Instantiate a regression model based on the provided name and parameters.

    Parameters
    ----------
    model_name : str
        Identifier for the regression model ('LightGBM' or 'RandomForest').
    params : Dict[str, object]
        Hyperparameters to inject into the estimator constructor.

    Returns
    -------
    object
        A scikit-learn compatible regressor instance.
    """
    if model_name == "RandomForest":
        return RandomForestRegressor(**params)

    if model_name == "LightGBM":
        if lgb is None:
            raise ImportError(
                "LightGBM is not installed. Install lightgbm to enable this estimator."
            )
        return lgb.LGBMRegressor(**params)

    raise ValueError(f"Unsupported model: {model_name}")


def _custom_walk_forward_splits(
    n_samples: int, train_windows: int, test_windows: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate walk-forward train/test splits for time-series validation.

    Parameters
    ----------
    n_samples : int
        Number of observations available for training.
    train_windows : int
        Size of the expanding training window (in observations).
    test_windows : int
        Size of the test window for each fold.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List of (train_indices, test_indices) pairs following walk-forward logic.
    """
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    start = train_windows
    while start + test_windows <= n_samples:
        train_idx = np.arange(start)
        test_idx = np.arange(start, start + test_windows)
        splits.append((train_idx.copy(), test_idx.copy()))
        start += test_windows
    return splits


def build_features_targets(
    moments_df: pd.DataFrame,
    lag_windows: int = 10,
    target_moments: Sequence[str] = TARGET_MOMENTS,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Assemble per-asset feature matrices and target vectors from historical moments.

    Parameters
    ----------
    moments_df : pd.DataFrame
        Multi-index column DataFrame with first level representing moment names
        and second level listing asset tickers.
    lag_windows : int, optional
        Number of lagged observations to include per feature moment (default 10).
    target_moments : Sequence[str], optional
        Moments to forecast in the downstream pipeline.

    Returns
    -------
    Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]
        Per-asset dictionaries mapping to feature matrices and target DataFrames.

    TODO
    ----
    - Extend feature set with cross-asset statistics once requirements are confirmed.
    - Persist engineered datasets for ML debugging when pipeline stabilization is needed.
    """
    if not isinstance(moments_df.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns with structure (moment, asset).")

    assets = sorted(moments_df.columns.get_level_values(1).unique())
    features: Dict[str, pd.DataFrame] = {}
    targets: Dict[str, pd.DataFrame] = {}

    for asset in assets:
        asset_frame = moments_df.xs(asset, level=1, axis=1).sort_index()

        missing_features = [moment for moment in FEATURE_MOMENTS if moment not in asset_frame.columns]
        if missing_features:
            warnings.warn(
                f"Asset {asset} missing required moments {missing_features}; skipping."
            )
            continue

        feature_columns: Dict[str, pd.Series] = {}
        for moment in FEATURE_MOMENTS:
            series = asset_frame[moment]
            for lag in range(1, lag_windows + 1):
                feature_columns[f"{moment}_lag_{lag}"] = series.shift(lag)

        asset_features = pd.DataFrame(feature_columns, index=asset_frame.index)
        asset_targets = asset_frame[list(target_moments)].copy()

        features[asset] = asset_features
        targets[asset] = asset_targets

    if not features:
        raise ValueError("No assets produced valid feature sets; check input data.")

    return features, targets


def _train_single_asset(
    asset: str,
    feature_df: pd.DataFrame,
    target_df: pd.DataFrame,
    models: Sequence[str],
    estimator_cfg: Dict[str, dict],
    walk_cfg: Dict[str, int],
    target_moments: Sequence[str],
) -> Tuple[str, pd.Series, pd.Timestamp | None]:
    """
    Fit enabled models for a single asset and produce averaged forecasts.

    Parameters
    ----------
    asset : str
        Asset ticker under evaluation.
    feature_df : pd.DataFrame
        Lagged feature matrix for the asset.
    target_df : pd.DataFrame
        Target DataFrame containing the future moments.
    models : Sequence[str]
        Names of estimators requested by the user.
    estimator_cfg : Dict[str, dict]
        Configuration block for estimator hyperparameters and enable flags.
    walk_cfg : Dict[str, int]
        Walk-forward configuration containing `train_windows` and `test_windows`.

    Returns
    -------
    Tuple[str, pd.Series, pd.Timestamp | None]
        Asset ticker, predicted moments series, and timestamp associated with features.
    """
    complete_feature_mask = feature_df.notna().all(axis=1)
    complete_target_mask = target_df[list(target_moments)].notna().all(axis=1)
    valid_idx = feature_df.index[complete_feature_mask & complete_target_mask]

    if len(valid_idx) <= 1:
        warnings.warn(f"Insufficient data to train asset {asset}; emitting NaNs.")
        return asset, pd.Series({moment: np.nan for moment in target_moments}), None

    train_idx = valid_idx[:-1]
    forecast_idx = valid_idx[-1]

    if len(train_idx) < walk_cfg.get("train_windows", 1):
        warnings.warn(
            f"Asset {asset} has only {len(train_idx)} training samples; "
            "predictions may be unstable."
        )

    X_train = feature_df.loc[train_idx]
    y_train = target_df.loc[train_idx, TARGET_MOMENTS]
    X_forecast = feature_df.loc[[forecast_idx]]

    predictions: Dict[str, float] = {}

    for target in target_moments:
        target_values = y_train[target]
        target_predictions: List[float] = []

        for model_name in models:
            cfg_key = ESTIMATOR_KEY_MAP.get(model_name, model_name.lower())
            model_info = estimator_cfg.get(cfg_key)
            if not model_info or not model_info.get("enabled", False):
                continue

            params = model_info.get("params", {})

            try:
                model = _build_model(model_name, params)
            except ImportError as exc:
                warnings.warn(str(exc))
                continue

            splits = _custom_walk_forward_splits(
                n_samples=len(X_train),
                train_windows=walk_cfg.get("train_windows", len(X_train)),
                test_windows=walk_cfg.get("test_windows", 1),
            )

            if splits:
                _ = Parallel(n_jobs=-1, prefer="threads")(
                    delayed(_fold_score)(
                        model_name, params, X_train, target_values, train_indices, test_indices
                    )
                    for train_indices, test_indices in splits
                )

            model.fit(X_train, target_values)
            target_predictions.append(float(model.predict(X_forecast)[0]))

        if target_predictions:
            predictions[target] = float(np.mean(target_predictions))
        else:
            predictions[target] = np.nan

    return asset, pd.Series(predictions, name=asset), forecast_idx


def _fold_score(
    model_name: str,
    params: Dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: Iterable[int],
    test_idx: Iterable[int],
) -> float:
    """
    Fit a model on a walk-forward split and return the mean squared error.

    Parameters
    ----------
    model_name : str
        Estimator name.
    params : Dict[str, object]
        Estimator parameters.
    X : pd.DataFrame
        Training feature matrix.
    y : pd.Series
        Training targets.
    train_idx : Iterable[int]
        Integer index positions for the training fold.
    test_idx : Iterable[int]
        Integer index positions for the validation fold.

    Returns
    -------
    float
        Mean squared error for the fold.
    """
    model = _build_model(model_name, params)
    X_train = X.iloc[list(train_idx)]
    y_train = y.iloc[list(train_idx)]
    X_test = X.iloc[list(test_idx)]
    y_test = y.iloc[list(test_idx)]

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return float(mean_squared_error(y_test, preds))


def _next_timestamp(index: pd.Index) -> pd.Timestamp | None:
    """
    Infer the next timestamp for forecasting based on historical spacing.

    Parameters
    ----------
    index : pd.Index
        Datetime index from the historical moment data.

    Returns
    -------
    pd.Timestamp | None
        Next timestamp if it can be inferred, otherwise None.
    """
    if not isinstance(index, pd.DatetimeIndex) or index.empty:
        return None

    inferred = pd.infer_freq(index)
    if inferred:
        return index[-1] + pd.tseries.frequencies.to_offset(inferred)

    if len(index) >= 2:
        delta = index[-1] - index[-2]
        if isinstance(delta, pd.Timedelta):
            return index[-1] + delta

    return None


def train_and_forecast(
    moments_df: pd.DataFrame,
    models: Sequence[str] = ("LightGBM", "RandomForest"),
    config: Dict[str, dict] | None = None,
) -> pd.DataFrame:
    """
    Train requested ML models and forecast next-period moments.

    Parameters
    ----------
    moments_df : pd.DataFrame
        Historical moments with MultiIndex columns (moment, asset).
    models : Sequence[str], optional
        Estimators to evaluate; defaults include LightGBM and RandomForest.
    config : Dict[str, dict] | None, optional
        Configuration dictionary providing ML hyperparameters and lags.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by the forecast timestamp with MultiIndex columns
        containing predicted mean, variance, and CVaR per asset.

    TODO
    ----
    - Blend model outputs using performance-weighted ensembles.
    - Persist cross-validation diagnostics for downstream model monitoring.
    """
    cfg = config or _load_params()
    ml_cfg = cfg.get("ml", {})
    lag_windows = ml_cfg.get("lag_windows", 10)
    target_moments = tuple(ml_cfg.get("targets", TARGET_MOMENTS))

    features, targets = build_features_targets(
        moments_df,
        lag_windows=lag_windows,
        target_moments=target_moments,
    )

    estimator_cfg = ml_cfg.get("estimators", {})
    walk_cfg = ml_cfg.get("walk_forward", {"train_windows": 60, "test_windows": 1})

    assets = sorted(features.keys())
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_train_single_asset)(
            asset,
            features[asset],
            targets[asset],
            models,
            estimator_cfg,
            walk_cfg,
            target_moments,
        )
        for asset in assets
    )

    predictions: Dict[Tuple[str, str], float] = {}

    for asset, series, _ in results:
        for moment in target_moments:
            predictions[(moment, asset)] = series.get(moment, np.nan)

    inferred_next = _next_timestamp(moments_df.index)
    if inferred_next is None:
        if isinstance(moments_df.index, pd.DatetimeIndex) and not moments_df.index.empty:
            inferred_next = moments_df.index[-1]
        else:
            inferred_next = pd.Timestamp(0)

    forecast_index = pd.Index([inferred_next])

    column_index = pd.MultiIndex.from_product(
        [target_moments, assets], names=["moment", "asset"]
    )

    forecast_df = pd.DataFrame(index=forecast_index, columns=column_index, dtype=float)
    for moment, asset in column_index:
        forecast_df[(moment, asset)] = predictions.get((moment, asset), np.nan)

    return forecast_df


def forecast_moments(processed_dir: Path | None = None) -> Tuple[Path, Path]:
    """
    Execute end-to-end ML forecasting for hourly and daily pipelines.

    Parameters
    ----------
    processed_dir : Path | None, optional
        Base directory containing processed datasets; defaults to data/processed.

    Returns
    -------
    Tuple[Path, Path]
        File paths to the saved hourly and daily forecast parquet files.

    TODO
    ----
    - Log runtime statistics and model diagnostics to support monitoring.
    """
    cfg = _load_params()
    project_root = Path(__file__).resolve().parents[1]
    base_dir = processed_dir or project_root / "data" / "processed"
    base_dir.mkdir(parents=True, exist_ok=True)

    frequencies = cfg.get("data", {}).get("frequencies", {"hourly": "1H", "daily": "1D"})

    output_paths: Dict[str, Path] = {}
    for label, freq_alias in frequencies.items():
        suffix = str(freq_alias).lower()
        moments_path = base_dir / f"moments_{suffix}.parquet"
        if not moments_path.exists():
            raise FileNotFoundError(f"Moments file not found for {label}: {moments_path}")

        moments_df = pd.read_parquet(moments_path)
        forecast_df = train_and_forecast(moments_df, config=cfg)

        output_path = base_dir / f"forecasted_moments_{suffix}.parquet"
        forecast_df.to_parquet(output_path)
        output_paths[label] = output_path

        print(f"[ml_forecast] {label} forecast shape: {forecast_df.shape}")

    hourly_path = output_paths.get("hourly")
    daily_path = output_paths.get("daily")
    return (
        hourly_path if hourly_path is not None else Path(),
        daily_path if daily_path is not None else Path(),
    )


if __name__ == "__main__":
    forecast_moments()
