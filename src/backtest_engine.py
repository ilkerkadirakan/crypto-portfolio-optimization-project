"""
Backtesting engine for rolling portfolio optimization and rebalancing analysis.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from sklearn.covariance import LedoitWolf

from .optim_models import solve_portfolio
from .combination_utils import cache_combinations

TRANSACTION_COST_BPS = 0.001  # 10 bps per leg
BASELINE_VERSION = {"baseline", "realized", "realised"}
ML_VERSION = {"ml", "forecast", "forecasted"}
def _ensure_utf8_stdout() -> None:
    try:
        import sys
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


@dataclass
class BacktestConfig:
    """
    Container for frequently accessed configuration settings.
    """

    window: int
    rebalance: str | int
    transaction_cost: float = TRANSACTION_COST_BPS




def _safe_loc(source: pd.DataFrame | pd.Series, ts) -> pd.Series:
    """
    Güvenli şekilde zaman indeksiyle alma:
    - Önce doğrudan deneme
    - Ardından ts'yi pd.Timestamp'e çevirip deneme
    - Son olarak reindex ile tek satırlık yeniden alma ve iloc ile döndürme
    Bu, iterable/array/series gibi yanlış anahtar türlerinde pandas'ın
    iterasyonuna takılmayı engeller.
    """
    if source is None:
        raise KeyError("Source is None")

    # 1) Doğrudan dene
    try:
        return source.loc[ts]
    except Exception:
        pass

    # 2) Timestamp dönüşümü ile dene
    try:
        ts_key = pd.Timestamp(ts)
    except Exception:
        # ts dönüştürülemezse hata fırlat
        raise

    try:
        return source.loc[ts_key]
    except Exception:
        pass

    # 3) Reindex ile tek satırlık alım (güvenli fallback)
    try:
        tmp = source.reindex([ts_key])
        if tmp.empty:
            raise KeyError(f"No entry for {ts_key}")
        # tmp bir DataFrame ise ilk (ve tek) satırı döndür
        if isinstance(tmp, pd.DataFrame):
            return tmp.iloc[0]
        return tmp.iloc[0]  # Series için de çalışır
    except Exception:
        raise


def _load_config() -> Dict[str, dict]:
    """
    Read the global parameter configuration from configs/params.yaml.
    """
    config_path = Path(__file__).resolve().parents[1] / "configs" / "params.yaml"
    if not config_path.exists():
        warnings.warn(f"Configuration file not found at {config_path}. Using defaults.")
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _normalize_freq(freq: str) -> str:
    """
    Normalize frequency identifiers to canonical forms ('1H' or '1D').
    """
    freq = freq.strip().lower()
    if freq in {"hourly", "1h", "h1"}:
        return "1H"
    if freq in {"daily", "1d", "d1"}:
        return "1D"
    raise ValueError(f"Unsupported frequency '{freq}'. Expected '1H' or '1D'.")


def _freq_suffix(freq: str) -> str:
    """
    Return a lowercase suffix used for file naming based on frequency.
    """
    return _normalize_freq(freq).lower()


def _get_backtest_config(config: Dict[str, dict], freq: str) -> BacktestConfig:
    """
    Create a BacktestConfig object with window and rebalance settings for a frequency.
    """
    norm = _normalize_freq(freq)
    windows_cfg = config.get("windows", {}).get("rolling", {})
    rebalance_cfg = config.get("rebalance", {})
    if norm == "1H":
        window = int(windows_cfg.get("hourly", 4320))
        rebalance = rebalance_cfg.get("hourly", {}).get("bars", 24)
    else:
        window = int(windows_cfg.get("daily", 180))
        rebalance = rebalance_cfg.get("daily", {}).get("rule", "W-MON")
    return BacktestConfig(window=window, rebalance=rebalance)


def _combo_label(combo: Union[str, int, Tuple, List, Dict[str, object]]) -> str:
    """
    Derive a human-readable label for an entry from combo_iterable.
    """
    if isinstance(combo, dict):
        for key in ("name", "label", "id"):
            if key in combo and combo[key]:
                return str(combo[key])
        return "_".join(f"{k}-{v}" for k, v in sorted(combo.items()))
    if isinstance(combo, (tuple, list)):
        return "_".join(str(item) for item in combo)
    return str(combo)


def _combo_assets(combo: Union[str, int, Tuple, List, Dict[str, object]]) -> Optional[List[str]]:
    """
    Extract an asset universe override from a combo definition if provided.
    """
    if isinstance(combo, dict):
        assets = combo.get("assets") or combo.get("tickers")
        if assets:
            return list(assets)
    # Interpret tuple/list directly as an asset subset when provided.
    if isinstance(combo, (tuple, list)) and combo:
        return [str(a) for a in combo]
    return None


def _combo_params(combo: Union[str, int, Tuple, List, Dict[str, object]]) -> Dict[str, dict]:
    """
    Extract parameter overrides from a combo entry.
    """
    if isinstance(combo, dict):
        params = combo.get("params")
        if isinstance(params, dict):
            return params
    return {}


def _merge_params(base: Dict[str, dict], overrides: Dict[str, dict]) -> Dict[str, dict]:
    """
    Create a nested copy of base parameters updated with overrides.
    """
    if not overrides:
        return base
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_params(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _split_moment_panels(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Split a MultiIndex moment DataFrame into per-moment DataFrames keyed by moment name.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Moment DataFrame is expected to use MultiIndex columns (moment, asset).")
    panels = {}
    for moment in df.columns.get_level_values(0).unique():
        panels[moment] = df.xs(moment, level=0, axis=1)
    return panels


def _rebalance_positions(index: pd.DatetimeIndex, cfg: BacktestConfig) -> List[int]:
    """
    Determine the positional indices at which rebalancing should occur.
    """
    if index.empty:
        return []

    if isinstance(cfg.rebalance, int):  # hourly cadence in bars
        step = max(int(cfg.rebalance), 1)
        start = cfg.window - 1 if cfg.window <= len(index) else 0
        positions = list(range(start, len(index), step))
    else:  # pandas offset alias (e.g., 'W-MON')
        rebalance_dates = index.to_series().resample(cfg.rebalance).last().dropna().values
        lookup = {ts: pos for pos, ts in enumerate(index)}
        positions = [lookup[ts] for ts in rebalance_dates if ts in lookup]
        if cfg.window - 1 < len(index) and (cfg.window - 1) not in positions:
            positions.insert(0, cfg.window - 1)
        positions = sorted(set(pos for pos in positions if pos < len(index)))
    return positions


def _sanitize_weights(weights: np.ndarray) -> np.ndarray:
    """
    Replace NaNs in weight vectors with zeros and renormalize if needed.
    """
    weights = np.asarray(weights, dtype=float)
    weights = np.nan_to_num(weights, nan=0.0)
    total = weights.sum()
    if total <= 0:
        return weights
    return weights / total


def _ledoit_covariance(window_returns: pd.DataFrame, assets: Sequence[str]) -> Optional[np.ndarray]:
    """
    Estimate a covariance matrix for the given assets using Ledoit-Wolf shrinkage.
    """
    subset = window_returns.loc[:, assets].dropna()
    if subset.empty or subset.shape[0] < 2:
        return None
    estimator = LedoitWolf()
    estimator.fit(subset.values)
    return estimator.covariance_


def _diag_covariance(variances: pd.Series) -> np.ndarray:
    """
    Construct a diagonal covariance matrix using per-asset variances.
    """
    filled = variances.fillna(variances.mean() if not variances.dropna().empty else 0.0)
    return np.diag(filled.values)


def _select_panels(moment_panels: Dict[str, pd.DataFrame], assets: Sequence[str]) -> Dict[str, pd.DataFrame]:
    """
    Restrict moment panels to the specified asset universe, reindexing as needed.
    """
    selected = {}
    for moment, panel in moment_panels.items():
        shared_assets = [asset for asset in assets if asset in panel.columns]
        selected[moment] = panel.reindex(columns=shared_assets)
    return selected


def _prepare_moment_inputs(
    ts: pd.Timestamp,
    assets: Sequence[str],
    panels: Dict[str, pd.DataFrame],
    fallback_panels: Dict[str, pd.DataFrame],
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Assemble the various moment series required for optimization at a timestamp.
    """
    def pick(moment: str, default: float = np.nan) -> pd.Series:
        source = panels.get(moment)
        if source is not None and ts in source.index:
            series = _safe_loc(source, ts)
        else:
            series = pd.Series(default, index=assets, dtype=float)
        if series.isna().all() and fallback_panels:
            fallback = fallback_panels.get(moment)
            if fallback is not None and ts in fallback.index:
                series = fallback.loc[ts].reindex(assets)
        return series.reindex(assets)

    mu = pick("mean", 0.0)
    variance = pick("variance", 0.0)
    skew = pick("skewness", 0.0)
    kurt = pick("kurtosis", 3.0)
    cvar = pick("cvar", 0.0)
    return mu, variance, skew, kurt, cvar


def _slugify(text: str) -> str:
    """
    Create a filesystem-friendly slug from an arbitrary string.
    """
    slug = "".join(ch if ch.isalnum() else "_" for ch in text.strip())
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_").lower() or "run"


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a DataFrame uses a DatetimeIndex by attempting to convert the index.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    try:
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise ValueError("DataFrame index could not be converted to datetime.") from exc


def _load_returns(freq: str, processed_dir: Path) -> pd.DataFrame:
    """
    Load the pre-computed returns parquet file for the requested frequency.
    """
    suffix = _freq_suffix(freq)
    returns_path = processed_dir / f"returns_{suffix}.parquet"
    if not returns_path.exists():
        raise FileNotFoundError(f"Returns file not found at {returns_path}")
    returns_df = pd.read_parquet(returns_path)
    return _ensure_datetime_index(returns_df).sort_index()


def _load_moments(freq: str, processed_dir: Path, version: str) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Load baseline and forecasted moment panels for the requested frequency.
    """
    suffix = _freq_suffix(freq)
    baseline_path = processed_dir / f"moments_{suffix}.parquet"
    forecast_path = processed_dir / f"forecasted_moments_{suffix}.parquet"

    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline moments file missing at {baseline_path}")

    baseline_panels = _split_moment_panels(_ensure_datetime_index(pd.read_parquet(baseline_path)))

    if forecast_path.exists():
        forecast_panels = _split_moment_panels(_ensure_datetime_index(pd.read_parquet(forecast_path)))
    else:
        if version.lower() in ML_VERSION:
            warnings.warn(f"Forecasted moments file missing at {forecast_path}; falling back to baseline moments.")
        forecast_panels = {}

    return baseline_panels, forecast_panels


def _load_ml_weights(freq: str, processed_dir: Path) -> pd.DataFrame | None:
    """
    Load ML-predicted portfolio weights for the requested frequency.

    Returns None if ML weights are not available.
    """
    suffix = _freq_suffix(freq)
    ml_weights_path = processed_dir / f"ml_predicted_weights_{suffix}.parquet"

    if not ml_weights_path.exists():
        return None

    ml_weights = pd.read_parquet(ml_weights_path)
    ml_weights = _ensure_datetime_index(ml_weights)
    return ml_weights


def _apply_asset_subset(df: pd.DataFrame, assets: Optional[List[str]]) -> pd.DataFrame:
    """
    Restrict a DataFrame to a subset of assets if specified.
    """
    if assets is None:
        return df
    available = [col for col in df.columns if col in assets]
    if not available:
        raise ValueError("Asset subset results in empty DataFrame.")
    return df.loc[:, available]


def _resample_hourly_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly performance series to daily returns using compounding.
    """
    if df.empty:
        return df.copy()
    daily = pd.DataFrame(index=pd.date_range(df.index.min().normalize(), df.index.max().normalize()))
    for column in ("gross_return", "net_return"):
        if column in df.columns:
            series = df[column]
            compounded = (1.0 + series).resample("1D").prod() - 1.0
            daily[column] = compounded
    return daily.dropna(how="all")


def run_backtest(
    freq: str,
    version: str,
    model_list: Sequence[str],
    combo_iterable: Iterable[Union[str, int, Tuple, List, Dict[str, object]]],
) -> pd.DataFrame:
    """
    Execute rolling portfolio backtests for the specified models and combinations.

    Parameters
    ----------
    freq : str
        Frequency identifier ('1H' or '1D').
    version : str
        Data version flag ('baseline' or 'ml').
    model_list : Sequence[str]
        Collection of optimization model names (e.g., ['MV', 'MVSK']).
    combo_iterable : Iterable
        Iterable of scenario definitions; each element may provide asset subsets
        and/or configuration overrides.

    Returns
    -------
    pd.DataFrame
        Concatenated performance records across all runs with per-timestamp results.
    """
    cfg = _load_config()
    freq_norm = _normalize_freq(freq)
    freq_suffix = _freq_suffix(freq_norm)
    version_norm = version.lower()
    processed_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    returns_df = _load_returns(freq_norm, processed_dir)
    returns_df = returns_df.sort_index()

    baseline_panels, forecast_panels = _load_moments(freq_norm, processed_dir, version_norm)
    cfg_bt = _get_backtest_config(cfg, freq_norm)

    # Load ML-predicted weights if version is ML
    ml_weights_df = None
    if version_norm in ML_VERSION:
        ml_weights_df = _load_ml_weights(freq_norm, processed_dir)
        if ml_weights_df is not None:
            print(f"[Backtest] Using ML-predicted weights for version={version_norm}")
        else:
            print(f"[Backtest] ML weights not available, falling back to baseline optimization")

    asset_list_all = list(returns_df.columns)
    simple_returns = (np.exp(returns_df) - 1.0).fillna(0.0)

    # Filter index to only include timestamps where moments are available
    # Check first valid moment timestamp
    first_valid_moment = baseline_panels.get("mean").dropna(how='all').index[0] if "mean" in baseline_panels and not baseline_panels["mean"].dropna(how='all').empty else None

    if first_valid_moment is not None:
        index_common = simple_returns.loc[first_valid_moment:].index
    else:
        index_common = simple_returns.index

    rebalance_positions = _rebalance_positions(index_common, cfg_bt)
    if not rebalance_positions:
        raise ValueError("No rebalance positions were generated; check window and data alignment.")

    all_runs: List[pd.DataFrame] = []
    project_root = Path(__file__).resolve().parents[1]
    results_root = project_root / "results" / "runs"
    results_root.mkdir(parents=True, exist_ok=True)

    # Prepare combinations: if not supplied, generate exhaustive combos (2,3,5)
    combo_list = list(combo_iterable) if combo_iterable else []
    if not combo_list:
        _ensure_utf8_stdout()
        try:
            combos_dict = cache_combinations()
            for size_key, combo_group in combos_dict.items():
                print(f"Group size {size_key}: {len(combo_group)} combinations")
            print("\u2705 Full combination generator ready")
            order = ["2", "3", "5"]
            combo_list = [c for key in order for c in combos_dict.get(key, [])]
        except Exception as exc:
            warnings.warn(f"Combination cache unavailable: {exc}. Proceeding with single full-universe run.")
            combo_list = [None]
    combo_iterable = combo_list

    for combo in combo_iterable:
        combo_label = _combo_label(combo)
        combo_assets = _combo_assets(combo) or asset_list_all
        params_override = _combo_params(combo)
        params = _merge_params(cfg, params_override) if params_override else cfg

        returns_subset = _apply_asset_subset(simple_returns, combo_assets)
        asset_list = list(returns_subset.columns)
        if not asset_list:
            warnings.warn(f"Combo '{combo_label}' resulted in empty asset universe; skipping.")
            continue

        baseline_selected = _select_panels(baseline_panels, asset_list)
        forecast_selected = _select_panels(forecast_panels, asset_list) if forecast_panels else {}

        for model_name in model_list:
            records: List[Dict[str, object]] = []

            prev_weights_full = np.zeros(len(asset_list), dtype=float)

            for pos_idx, start_pos in enumerate(rebalance_positions):
                ts = index_common[start_pos]
                end_pos = rebalance_positions[pos_idx + 1] if pos_idx + 1 < len(rebalance_positions) else len(index_common)
                period_index = index_common[start_pos:end_pos]

                if version_norm in ML_VERSION and forecast_selected:
                    mu_series, variance_series, skew_series, kurt_series, cvar_series = _prepare_moment_inputs(
                        ts, asset_list, forecast_selected, baseline_selected
                    )
                else:
                    mu_series, variance_series, skew_series, kurt_series, cvar_series = _prepare_moment_inputs(
                        ts, asset_list, baseline_selected, forecast_selected
                    )

                active_mask = (
                    mu_series.notna()
                    & variance_series.notna()
                    & skew_series.notna()
                    & kurt_series.notna()
                    & cvar_series.notna()
                )

                active_assets = [asset for asset, flag in zip(asset_list, active_mask) if flag]
                if not active_assets:
                    warnings.warn(f"No valid assets for optimization at {ts}; carrying previous weights.")
                    weights_full = prev_weights_full.copy()
                    turnover = 0.0
                    transaction_cost = 0.0
                else:
                    window_slice = returns_subset.iloc[max(0, start_pos - cfg_bt.window + 1):start_pos + 1]
                    covariance_matrix = _ledoit_covariance(window_slice, active_assets)
                    if covariance_matrix is None:
                        covariance_matrix = _diag_covariance(variance_series.loc[active_assets])

                    # Extract moment values for active assets
                    mu_vals = mu_series.loc[active_assets].values
                    skew_vals = skew_series.loc[active_assets].values
                    kurt_vals = kurt_series.loc[active_assets].values
                    cvar_vals = cvar_series.loc[active_assets].values

                    # Validate inputs before calling solver
                    input_valid = True
                    if not (np.isfinite(mu_vals).all() and np.isfinite(skew_vals).all() and
                            np.isfinite(kurt_vals).all() and np.isfinite(cvar_vals).all()):
                        input_valid = False
                    if not np.isfinite(covariance_matrix).all():
                        input_valid = False
                    if len(active_assets) < 2:
                        input_valid = False

                    if not input_valid:
                        # Track and warn about invalid inputs
                        failure_key = f"{combo_label}_{model_name}"
                        if not hasattr(run_backtest, '_failures'):
                            run_backtest._failures = {}
                        if failure_key not in run_backtest._failures:
                            run_backtest._failures[failure_key] = 0
                        run_backtest._failures[failure_key] += 1

                        if run_backtest._failures[failure_key] == 1:
                            warnings.warn(
                                f"Invalid inputs for model '{model_name}' combo '{combo_label}'. "
                                f"This warning will not repeat for this combination. "
                                f"Reason: NaN/Inf values or insufficient assets.",
                                UserWarning,
                                stacklevel=2
                            )
                        weights_full = prev_weights_full.copy()
                        turnover = 0.0
                        transaction_cost = 0.0
                    else:
                        try:
                            weights_active = solve_portfolio(
                                model_name=model_name,
                                mu=mu_vals,
                                sigma=covariance_matrix,
                                skew=skew_vals,
                                kurt=kurt_vals,
                                cvar_series=cvar_vals,
                                params=params,
                            )
                            weights_active = _sanitize_weights(weights_active)
                            weights_full = np.zeros(len(asset_list), dtype=float)
                            for asset, weight in zip(active_assets, weights_active):
                                weights_full[asset_list.index(asset)] = weight
                        except Exception as exc:
                            # Track failures per combo to avoid spam
                            failure_key = f"{combo_label}_{model_name}"
                            if not hasattr(run_backtest, '_failures'):
                                run_backtest._failures = {}

                            if failure_key not in run_backtest._failures:
                                run_backtest._failures[failure_key] = 0

                            run_backtest._failures[failure_key] += 1

                            # Only warn on first failure for each combo/model pair
                            if run_backtest._failures[failure_key] == 1:
                                warnings.warn(
                                    f"Optimization failed for model '{model_name}' combo '{combo_label}'. "
                                    f"This warning will not repeat for this combination. "
                                    f"Error: {exc}",
                                    UserWarning,
                                    stacklevel=2
                                )
                            weights_full = prev_weights_full.copy()

                        turnover = float(np.sum(np.abs(weights_full - prev_weights_full)))
                        transaction_cost = turnover * cfg_bt.transaction_cost

                for step_idx, period_ts in enumerate(period_index):
                    asset_returns = returns_subset.loc[period_ts, :].values
                    gross_return = float(np.dot(asset_returns, weights_full))
                    cost = transaction_cost if step_idx == 0 else 0.0
                    net_return = gross_return - cost
                    record = {
                        "timestamp": period_ts,
                        "freq": freq_norm,
                        "version": version_norm,
                        "model": model_name.upper(),
                        "combo": combo_label,
                        "gross_return": gross_return,
                        "transaction_cost": cost,
                        "net_return": net_return,
                        "turnover": turnover if step_idx == 0 else 0.0,
                        "rebalance": step_idx == 0,
                    }
                    for asset, weight in zip(asset_list, weights_full):
                        record[f"weight_{asset}"] = float(weight)
                    records.append(record)

                prev_weights_full = weights_full.copy()

            run_df = pd.DataFrame.from_records(records)
            if run_df.empty:
                continue
            run_df = run_df.sort_values("timestamp").set_index("timestamp")

            result_dir = results_root / f"{freq_suffix}_{model_name.lower()}_{version_norm}"
            result_dir.mkdir(parents=True, exist_ok=True)
            combo_slug = _slugify(combo_label)
            output_path = result_dir / f"{combo_slug}.parquet"
            run_df.to_parquet(output_path)

            if freq_norm == "1H":
                daily_df = _resample_hourly_to_daily(run_df[["gross_return", "net_return"]])
                if not daily_df.empty:
                    daily_df["model"] = model_name.upper()
                    daily_df["version"] = version_norm
                    daily_df["combo"] = combo_label
                    daily_dir = result_dir / "daily"
                    daily_dir.mkdir(parents=True, exist_ok=True)
                    daily_path = daily_dir / f"{combo_slug}.parquet"
                    daily_df.to_parquet(daily_path)

            all_runs.append(run_df.reset_index())

    if not all_runs:
        return pd.DataFrame()

    combined = pd.concat(all_runs, ignore_index=True)
    # Skip sorting - not needed for groupby and causes MemoryError on large datasets (46M+ rows)
    # combined = combined.sort_values(["combo", "model", "timestamp"])
    return combined


def _process_single_combo_model(args: Tuple) -> Optional[pd.DataFrame]:
    """
    Process a single combination-model pair for parallel execution.

    This function is designed to be called by multiprocessing.Pool, so all
    arguments are packed into a single tuple.

    Parameters
    ----------
    args : Tuple
        Packed arguments containing all necessary data for processing one combo-model pair.

    Returns
    -------
    pd.DataFrame or None
        Backtest results for this combo-model pair, or None if processing failed.
    """
    (
        combo, model_name, freq_norm, version_norm, returns_df, baseline_panels,
        forecast_panels, ml_weights_df, index_common, rebalance_positions, cfg_bt,
        cfg, asset_list_all, results_root, freq_suffix
    ) = args

    try:
        combo_label = _combo_label(combo)
        combo_assets = _combo_assets(combo) or asset_list_all
        params_override = _combo_params(combo)
        params = _merge_params(cfg, params_override) if params_override else cfg

        simple_returns = (np.exp(returns_df) - 1.0).fillna(0.0)
        returns_subset = _apply_asset_subset(simple_returns, combo_assets)
        asset_list = list(returns_subset.columns)

        if not asset_list:
            warnings.warn(f"Combo '{combo_label}' resulted in empty asset universe; skipping.")
            return None

        baseline_selected = _select_panels(baseline_panels, asset_list)
        forecast_selected = _select_panels(forecast_panels, asset_list) if forecast_panels else {}

        records: List[Dict[str, object]] = []
        prev_weights_full = np.zeros(len(asset_list), dtype=float)

        for pos_idx, start_pos in enumerate(rebalance_positions):
            ts = index_common[start_pos]
            end_pos = rebalance_positions[pos_idx + 1] if pos_idx + 1 < len(rebalance_positions) else len(index_common)
            period_index = index_common[start_pos:end_pos]

            if version_norm in ML_VERSION and forecast_selected:
                mu_series, variance_series, skew_series, kurt_series, cvar_series = _prepare_moment_inputs(
                    ts, asset_list, forecast_selected, baseline_selected
                )
            else:
                mu_series, variance_series, skew_series, kurt_series, cvar_series = _prepare_moment_inputs(
                    ts, asset_list, baseline_selected, forecast_selected
                )

            active_mask = (
                mu_series.notna()
                & variance_series.notna()
                & skew_series.notna()
                & kurt_series.notna()
                & cvar_series.notna()
            )

            active_assets = [asset for asset, flag in zip(asset_list, active_mask) if flag]
            if not active_assets:
                weights_full = prev_weights_full.copy()
                turnover = 0.0
                transaction_cost = 0.0
            else:
                window_slice = returns_subset.iloc[max(0, start_pos - cfg_bt.window + 1):start_pos + 1]
                covariance_matrix = _ledoit_covariance(window_slice, active_assets)
                if covariance_matrix is None:
                    covariance_matrix = _diag_covariance(variance_series.loc[active_assets])

                mu_vals = mu_series.loc[active_assets].values
                skew_vals = skew_series.loc[active_assets].values
                kurt_vals = kurt_series.loc[active_assets].values
                cvar_vals = cvar_series.loc[active_assets].values

                input_valid = True
                if not (np.isfinite(mu_vals).all() and np.isfinite(skew_vals).all() and
                        np.isfinite(kurt_vals).all() and np.isfinite(cvar_vals).all()):
                    input_valid = False
                if not np.isfinite(covariance_matrix).all():
                    input_valid = False
                if len(active_assets) < 2:
                    input_valid = False

                if not input_valid:
                    weights_full = prev_weights_full.copy()
                    turnover = 0.0
                    transaction_cost = 0.0
                else:
                    try:
                        weights_active = solve_portfolio(
                            model_name=model_name,
                            mu=mu_vals,
                            sigma=covariance_matrix,
                            skew=skew_vals,
                            kurt=kurt_vals,
                            cvar_series=cvar_vals,
                            params=params,
                        )
                        weights_active = _sanitize_weights(weights_active)
                        weights_full = np.zeros(len(asset_list), dtype=float)
                        for asset, weight in zip(active_assets, weights_active):
                            weights_full[asset_list.index(asset)] = weight
                    except Exception:
                        weights_full = prev_weights_full.copy()

                    turnover = float(np.sum(np.abs(weights_full - prev_weights_full)))
                    transaction_cost = turnover * cfg_bt.transaction_cost

            for step_idx, period_ts in enumerate(period_index):
                asset_returns = returns_subset.loc[period_ts, :].values
                gross_return = float(np.dot(asset_returns, weights_full))
                cost = transaction_cost if step_idx == 0 else 0.0
                net_return = gross_return - cost
                record = {
                    "timestamp": period_ts,
                    "freq": freq_norm,
                    "version": version_norm,
                    "model": model_name.upper(),
                    "combo": combo_label,
                    "gross_return": gross_return,
                    "transaction_cost": cost,
                    "net_return": net_return,
                    "turnover": turnover if step_idx == 0 else 0.0,
                    "rebalance": step_idx == 0,
                }
                for asset, weight in zip(asset_list, weights_full):
                    record[f"weight_{asset}"] = float(weight)
                records.append(record)

            prev_weights_full = weights_full.copy()

        run_df = pd.DataFrame.from_records(records)
        if run_df.empty:
            return None
        run_df = run_df.sort_values("timestamp").set_index("timestamp")

        result_dir = results_root / f"{freq_suffix}_{model_name.lower()}_{version_norm}"
        result_dir.mkdir(parents=True, exist_ok=True)
        combo_slug = _slugify(combo_label)
        output_path = result_dir / f"{combo_slug}.parquet"
        run_df.to_parquet(output_path)

        if freq_norm == "1H":
            daily_df = _resample_hourly_to_daily(run_df[["gross_return", "net_return"]])
            if not daily_df.empty:
                daily_df["model"] = model_name.upper()
                daily_df["version"] = version_norm
                daily_df["combo"] = combo_label
                daily_dir = result_dir / "daily"
                daily_dir.mkdir(parents=True, exist_ok=True)
                daily_path = daily_dir / f"{combo_slug}.parquet"
                daily_df.to_parquet(daily_path)

        return run_df.reset_index()

    except Exception as e:
        warnings.warn(f"Failed to process combo {_combo_label(combo)} with model {model_name}: {e}")
        return None


def run_backtest_parallel(
    freq: str,
    version: str,
    model_list: Sequence[str],
    combo_iterable: Iterable[Union[str, int, Tuple, List, Dict[str, object]]],
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Execute rolling portfolio backtests using parallel processing.

    This is a parallelized version of run_backtest that processes multiple
    combination-model pairs simultaneously using multiprocessing.

    Parameters
    ----------
    freq : str
        Frequency identifier ('1H' or '1D').
    version : str
        Data version flag ('baseline' or 'ml').
    model_list : Sequence[str]
        Collection of optimization model names (e.g., ['MV', 'MVSK']).
    combo_iterable : Iterable
        Iterable of scenario definitions.
    n_jobs : int, optional
        Number of parallel workers. -1 uses all available CPUs (default).

    Returns
    -------
    pd.DataFrame
        Concatenated performance records across all runs.
    """
    import multiprocessing as mp
    from functools import partial

    cfg = _load_config()
    freq_norm = _normalize_freq(freq)
    freq_suffix = _freq_suffix(freq_norm)
    version_norm = version.lower()
    processed_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    returns_df = _load_returns(freq_norm, processed_dir)
    returns_df = returns_df.sort_index()

    baseline_panels, forecast_panels = _load_moments(freq_norm, processed_dir, version_norm)
    cfg_bt = _get_backtest_config(cfg, freq_norm)

    ml_weights_df = None
    if version_norm in ML_VERSION:
        ml_weights_df = _load_ml_weights(freq_norm, processed_dir)
        if ml_weights_df is not None:
            print(f"[Backtest] Using ML-predicted weights for version={version_norm}")
        else:
            print(f"[Backtest] ML weights not available, falling back to baseline optimization")

    asset_list_all = list(returns_df.columns)

    first_valid_moment = baseline_panels.get("mean").dropna(how='all').index[0] if "mean" in baseline_panels and not baseline_panels["mean"].dropna(how='all').empty else None

    if first_valid_moment is not None:
        index_common = returns_df.loc[first_valid_moment:].index
    else:
        index_common = returns_df.index

    rebalance_positions = _rebalance_positions(index_common, cfg_bt)
    if not rebalance_positions:
        raise ValueError("No rebalance positions were generated; check window and data alignment.")

    project_root = Path(__file__).resolve().parents[1]
    results_root = project_root / "results" / "runs"
    results_root.mkdir(parents=True, exist_ok=True)

    combo_list = list(combo_iterable) if combo_iterable else []
    if not combo_list:
        _ensure_utf8_stdout()
        try:
            combos_dict = cache_combinations()
            for size_key, combo_group in combos_dict.items():
                print(f"Group size {size_key}: {len(combo_group)} combinations")
            print("\u2705 Full combination generator ready")
            order = ["2", "3", "5"]
            combo_list = [c for key in order for c in combos_dict.get(key, [])]
        except Exception as exc:
            warnings.warn(f"Combination cache unavailable: {exc}. Proceeding with single full-universe run.")
            combo_list = [None]

    # Create all (combo, model) pairs
    tasks = []
    for combo in combo_list:
        for model_name in model_list:
            task_args = (
                combo, model_name, freq_norm, version_norm, returns_df, baseline_panels,
                forecast_panels, ml_weights_df, index_common, rebalance_positions, cfg_bt,
                cfg, asset_list_all, results_root, freq_suffix
            )
            tasks.append(task_args)

    total_tasks = len(tasks)
    print(f"[Parallel Backtest] Processing {total_tasks} tasks ({len(combo_list)} combos × {len(model_list)} models)")

    # Determine number of workers
    if n_jobs == -1:
        n_workers = mp.cpu_count()
    else:
        n_workers = min(n_jobs, mp.cpu_count())

    print(f"[Parallel Backtest] Using {n_workers} worker processes")

    # Process in parallel
    all_runs: List[pd.DataFrame] = []
    with mp.Pool(processes=n_workers) as pool:
        # Use imap_unordered for progress tracking
        results = pool.imap_unordered(_process_single_combo_model, tasks)

        for i, result in enumerate(results, 1):
            if result is not None:
                all_runs.append(result)
            if i % 100 == 0 or i == total_tasks:
                print(f"[Parallel Backtest] Completed {i}/{total_tasks} tasks ({100*i/total_tasks:.1f}%)")

    if not all_runs:
        return pd.DataFrame()

    combined = pd.concat(all_runs, ignore_index=True)
    # Skip sorting - not needed for groupby and causes MemoryError on large datasets (46M+ rows)
    # combined = combined.sort_values(["combo", "model", "timestamp"])
    return combined


if __name__ == "__main__":  # pragma: no cover - convenience entrypoint
    CONFIG = _load_config()
    freq_default = CONFIG.get("data", {}).get("frequencies", {}).get("daily", "1D")
    run_backtest(freq=freq_default, version="baseline", model_list=["MV"], combo_iterable=[])










