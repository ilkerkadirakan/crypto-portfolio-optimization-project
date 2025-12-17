"""
ML-based portfolio weight learning (Student models).

This module implements direct weight learning where ML models learn to predict
portfolio weights by imitating and improving upon teacher (classical optimizer) weights.

NO moment forecasting - only weight prediction.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit

# Constants
LAG_WINDOWS = 10  # Number of past windows to use as features
TOP_K_ASSETS = {2, 3, 5}  # Valid portfolio sizes
MAX_WEIGHT = 0.30  # Maximum weight per asset


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has DatetimeIndex."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df


def _load_moments(processed_dir: Path, freq: str) -> pd.DataFrame:
    """Load pre-computed moment features."""
    freq_suffix = freq.lower()
    moments_path = processed_dir / f"moments_{freq_suffix}.parquet"

    if not moments_path.exists():
        raise FileNotFoundError(f"Moments file not found: {moments_path}")

    moments_df = pd.read_parquet(moments_path)
    return _ensure_datetime_index(moments_df)


def _load_returns(processed_dir: Path, freq: str) -> pd.DataFrame:
    """Load return series for raw return features."""
    freq_suffix = freq.lower()
    returns_path = processed_dir / f"returns_{freq_suffix}.parquet"

    if not returns_path.exists():
        raise FileNotFoundError(f"Returns file not found: {returns_path}")

    returns_df = pd.read_parquet(returns_path)
    return _ensure_datetime_index(returns_df)


def _extract_teacher_weights(teacher_results: pd.DataFrame) -> pd.DataFrame:
    """
    Extract weight columns from teacher backtest results.

    Returns DataFrame with columns: timestamp, combo, model, asset weights
    """
    weight_cols = [col for col in teacher_results.columns if col.startswith('weight_')]
    if not weight_cols:
        raise ValueError("No weight columns found in teacher results")

    # Extract weights with metadata
    weights_df = teacher_results[['timestamp', 'combo', 'model'] + weight_cols].copy()

    return weights_df


def _create_lagged_features(
    moments_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    asset_list: List[str],
    n_lags: int = LAG_WINDOWS
) -> pd.DataFrame:
    """
    Create lagged moment and return features for ML training.

    Features per asset:
    - lagged mean returns (n_lags windows)
    - lagged variance (n_lags windows)
    - lagged skewness (n_lags windows)
    - lagged kurtosis (n_lags windows)
    - lagged CVaR (n_lags windows)
    - lagged raw returns (n_lags windows)
    """
    features_list = []

    # Split moments into panels
    if isinstance(moments_df.columns, pd.MultiIndex):
        moment_names = moments_df.columns.get_level_values(0).unique()
        moment_panels = {m: moments_df.xs(m, level=0, axis=1) for m in moment_names}
    else:
        raise ValueError("Moments DataFrame should have MultiIndex columns")

    timestamps = moments_df.index

    for ts in timestamps:
        ts_idx = timestamps.get_loc(ts)

        if ts_idx < n_lags:
            continue  # Skip if not enough history

        row_features = {'timestamp': ts}

        for asset in asset_list:
            # Get lagged moments
            for lag in range(1, n_lags + 1):
                lag_idx = ts_idx - lag

                # Mean
                if 'mean' in moment_panels and asset in moment_panels['mean'].columns:
                    row_features[f'{asset}_mean_lag{lag}'] = moment_panels['mean'].iloc[lag_idx][asset]

                # Variance
                if 'variance' in moment_panels and asset in moment_panels['variance'].columns:
                    row_features[f'{asset}_var_lag{lag}'] = moment_panels['variance'].iloc[lag_idx][asset]

                # Skewness
                if 'skewness' in moment_panels and asset in moment_panels['skewness'].columns:
                    row_features[f'{asset}_skew_lag{lag}'] = moment_panels['skewness'].iloc[lag_idx][asset]

                # Kurtosis
                if 'kurtosis' in moment_panels and asset in moment_panels['kurtosis'].columns:
                    row_features[f'{asset}_kurt_lag{lag}'] = moment_panels['kurtosis'].iloc[lag_idx][asset]

                # CVaR
                if 'cvar' in moment_panels and asset in moment_panels['cvar'].columns:
                    row_features[f'{asset}_cvar_lag{lag}'] = moment_panels['cvar'].iloc[lag_idx][asset]

                # Raw returns
                if asset in returns_df.columns:
                    row_features[f'{asset}_return_lag{lag}'] = returns_df.iloc[lag_idx][asset]

        features_list.append(row_features)

    features_df = pd.DataFrame(features_list)
    features_df = features_df.set_index('timestamp')

    return features_df


def _apply_portfolio_constraints(
    raw_scores: np.ndarray,
    asset_list: List[str],
    combo_assets: List[str],
    k: int = 5
) -> np.ndarray:
    """
    Apply portfolio constraints to convert raw scores to valid weights.

    Steps:
    1. Softmax normalization
    2. Top-K mask (only combo assets)
    3. Weight cap (max 0.30 per asset)
    4. Final normalization
    """
    # Softmax normalization
    exp_scores = np.exp(raw_scores - np.max(raw_scores))
    weights = exp_scores / np.sum(exp_scores)

    # Top-K mask: zero out assets not in combo
    masked_weights = np.zeros_like(weights)
    for asset in combo_assets:
        if asset in asset_list:
            idx = asset_list.index(asset)
            masked_weights[idx] = weights[idx]

    # Apply weight cap
    masked_weights = np.clip(masked_weights, 0, MAX_WEIGHT)

    # Renormalize
    weight_sum = np.sum(masked_weights)
    if weight_sum > 0:
        masked_weights = masked_weights / weight_sum

    return masked_weights


def train_weight_models(
    teacher_results: pd.DataFrame,
    processed_dir: Path,
    freq: str
) -> None:
    """
    Train ML models to learn portfolio weights from teacher.

    This is the Student learning process.

    Parameters
    ----------
    teacher_results : pd.DataFrame
        Backtest results from teacher (classical optimizer)
    processed_dir : Path
        Directory containing processed data
    freq : str
        Frequency identifier (1H or 1D)
    """
    print(f"\n[ML-Weights] Training student models for freq={freq}")

    # Load moments and returns
    moments_df = _load_moments(processed_dir, freq)
    returns_df = _load_returns(processed_dir, freq)

    # Extract teacher weights
    teacher_weights = _extract_teacher_weights(teacher_results)
    weight_cols = [col for col in teacher_weights.columns if col.startswith('weight_')]
    asset_list = [col.replace('weight_', '') for col in weight_cols]

    print(f"[ML-Weights] Found {len(asset_list)} assets")
    print(f"[ML-Weights] Creating lagged features with {LAG_WINDOWS} lags...")

    # Create lagged features
    features_df = _create_lagged_features(moments_df, returns_df, asset_list, LAG_WINDOWS)

    print(f"[ML-Weights] Features shape: {features_df.shape}")

    # Merge features with teacher weights
    merged_df = features_df.join(teacher_weights.set_index('timestamp')[weight_cols], how='inner')

    if merged_df.empty:
        warnings.warn("[ML-Weights] No overlapping timestamps between features and teacher weights")
        return

    print(f"[ML-Weights] Training data shape: {merged_df.shape}")

    # Split features and targets
    X = merged_df.drop(columns=weight_cols).fillna(0)
    y = merged_df[weight_cols].fillna(0)

    print(f"[ML-Weights] Training {len(asset_list)} asset models...")

    # Train one model per asset
    models = {}
    for asset_col in weight_cols:
        asset_name = asset_col.replace('weight_', '')

        # Train with time series cross-validation
        model = LGBMRegressor(
            objective='regression',
            learning_rate=0.05,
            num_leaves=31,
            n_estimators=400,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )

        try:
            model.fit(X.values, y[asset_col].values)
            models[asset_name] = model
            print(f"[ML-Weights]   ✓ Trained model for {asset_name}")
        except Exception as exc:
            warnings.warn(f"[ML-Weights] Failed to train model for {asset_name}: {exc}")

    print(f"[ML-Weights] Successfully trained {len(models)} models")

    # Save models
    import joblib
    models_dir = processed_dir / "ml_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    models_path = models_dir / f"weight_models_{freq.lower()}.pkl"
    joblib.dump({
        'models': models,
        'asset_list': asset_list,
        'feature_cols': list(X.columns),
        'freq': freq
    }, models_path)

    print(f"[ML-Weights] Saved models to {models_path}")

    # Generate predictions for all data
    print(f"[ML-Weights] Generating weight predictions...")

    predictions = {}
    for asset_name, model in models.items():
        try:
            pred = model.predict(X.values)
            predictions[f'pred_weight_{asset_name}'] = pred
        except Exception as exc:
            warnings.warn(f"[ML-Weights] Failed to predict for {asset_name}: {exc}")

    pred_df = pd.DataFrame(predictions, index=X.index)

    # Save predictions
    pred_path = processed_dir / f"ml_predicted_weights_{freq.lower()}.parquet"
    pred_df.to_parquet(pred_path)

    print(f"[ML-Weights] Saved predictions to {pred_path}")
    print(f"[ML-Weights] Student training complete!\n")


def load_ml_weights(processed_dir: Path, freq: str) -> Tuple[Dict, pd.DataFrame]:
    """
    Load trained ML weight models and predictions.

    Returns
    -------
    models_dict : Dict
        Trained models and metadata
    predictions : pd.DataFrame
        Pre-computed weight predictions
    """
    import joblib

    models_dir = processed_dir / "ml_models"
    models_path = models_dir / f"weight_models_{freq.lower()}.pkl"

    if not models_path.exists():
        raise FileNotFoundError(f"ML models not found at {models_path}")

    models_dict = joblib.load(models_path)

    pred_path = processed_dir / f"ml_predicted_weights_{freq.lower()}.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"ML predictions not found at {pred_path}")

    predictions = pd.read_parquet(pred_path)

    return models_dict, predictions


__all__ = [
    'train_weight_models',
    'load_ml_weights',
]