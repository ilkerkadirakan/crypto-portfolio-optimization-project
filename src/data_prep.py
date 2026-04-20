# python
"""
Data preparation routines for generating aligned cryptocurrency return series.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat


def _matlab_datenum_to_datetime(values: np.ndarray) -> pd.DatetimeIndex:
    ordinal = np.floor(values).astype(int)
    fractional = values - ordinal
    base_dates = [dt.datetime.fromordinal(o) for o in ordinal]
    converted = [
        base + dt.timedelta(days=float(frac)) - dt.timedelta(days=366)
        for base, frac in zip(base_dates, fractional, strict=False)
    ]
    return pd.DatetimeIndex(converted)


def _to_datetime_index(raw_time: np.ndarray) -> pd.DatetimeIndex:
    flattened = np.asarray(raw_time).reshape(-1)

    if np.issubdtype(flattened.dtype, np.datetime64):
        return pd.to_datetime(flattened)

    try:
        converted = pd.to_datetime(flattened, utc=False, errors="raise")
        return converted
    except Exception:
        pass

    for unit in ("s", "ms", "us", "ns"):
        try:
            converted = pd.to_datetime(flattened, unit=unit, utc=False, errors="raise")
            if converted.notna().all():
                return converted
        except Exception:
            continue

    try:
        return _matlab_datenum_to_datetime(flattened.astype(float))
    except Exception as exc:
        raise ValueError("Unable to convert MATLAB timestamps to datetime.") from exc


def _infer_asset_name(stem: str) -> str:
    candidates = stem.replace("-", "_").split("_")
    token = candidates[0] if candidates else stem
    normalized = re.sub(r"\d+$", "", token.strip().upper())
    return normalized or token.strip().upper()


def _flatten_1d(arr) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype == object:
        try:
            return np.array(a.ravel().tolist(), dtype=float)
        except Exception:
            pass
    if a.ndim == 2 and 1 in a.shape:
        return a.ravel()
    if a.ndim >= 1:
        return a.reshape(-1)
    return np.array([float(a.item())])


def _extract_price_series(mat_payload: Dict[str, np.ndarray], file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (timestamps_array, prices_array).
    If only `price` exists, generate minute-based timestamps starting by file order (AAVEBTC1.mat -> Jan 2022).
    """

    time_keys: Iterable[str] = ("datetime", "timestamp", "time", "date", "dates")
    price_keys: Iterable[str] = ("price", "prices", "close", "adj_close", "value")

    keys = {k: v for k, v in mat_payload.items() if not k.startswith("__")}
    time_key = next((key for key in keys if any(tk in key.lower() for tk in time_keys)), None)
    price_key = next((key for key in keys if any(pk in key.lower() for pk in price_keys)), None)

    if price_key is None:
        raise KeyError(f"Could not locate price array in {file_path.name}.")

    price_array = _flatten_1d(keys[price_key]).astype(float)

    # Eğer zaman yoksa dosya adındaki sayıya göre 2022 yılından itibaren ay ata
    if time_key is None:
        # Örnek: AAVEBTC1.mat -> month_idx = 1 → 2022-01
        match = re.search(r"(\d+)$", file_path.stem)
        month_idx = int(match.group(1)) if match else 1

        # 36 ayın 12'sinde bir yılı artır
        year = 2022 + (month_idx - 1) // 12
        month = ((month_idx - 1) % 12) + 1

        # dakikalık timestamp dizisi oluştur
        n = price_array.size
        start = pd.Timestamp(year=year, month=month, day=1, hour=0, minute=1)
        index = pd.date_range(start=start, periods=n, freq="min", tz=None)

        return index.values, price_array

    # Eğer zaman key'i varsa — normal şekilde yükle
    time_array = _flatten_1d(keys[time_key]).astype(object)
    price_array = price_array[: time_array.shape[0]]  # güvenlik
    return time_array, price_array



def load_all_mat(data_dir: Path) -> Dict[str, pd.Series]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {data_dir}")

    asset_series: Dict[str, List[pd.Series]] = {}

    for mat_path in sorted(data_dir.glob("**/*.mat")):
        # use squeeze_me to simplify shapes and struct_as_record for compatibility
        payload = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
        try:
            timestamps, prices = _extract_price_series(payload, mat_path)
        except Exception as e:
            print(f"[skip] {mat_path.name}: {e}")
            continue

        # try to convert timestamps to DatetimeIndex; if not possible, keep integer index
        try:
            datetime_index = _to_datetime_index(timestamps)
            index = pd.DatetimeIndex(datetime_index)
        except Exception:
            index = pd.Index(np.arange(prices.size))

        series = pd.Series(prices, index=index, name=_infer_asset_name(mat_path.stem))
        series = series[~series.index.duplicated(keep="last")].sort_index()

        asset_series.setdefault(series.name, []).append(series)

    if not asset_series:
        raise FileNotFoundError(f"No .mat files discovered under {data_dir}")

    merged: Dict[str, pd.Series] = {}
    for asset, parts in asset_series.items():
        concatenated = pd.concat(parts).sort_index()
        concatenated = concatenated[~concatenated.index.duplicated(keep="last")]
        merged[asset] = concatenated.astype(float)

    return merged


def resample_and_log_returns(price_frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if price_frame.empty:
        return price_frame.copy()

    resampled = price_frame.resample(frequency).last().ffill(limit=1)
    log_returns = np.log(resampled / resampled.shift(1))
    log_returns = log_returns.replace([np.inf, -np.inf], np.nan)
    return log_returns.dropna(how="all")


def prepare_data(raw_data_dir: Path | None = None, processed_dir: Path | None = None) -> Tuple[Path, Path]:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = raw_data_dir or project_root / "data" / "raw"
    out_dir = processed_dir or project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    asset_prices = load_all_mat(raw_dir)
    price_frame = pd.concat(asset_prices.values(), axis=1)
    price_frame.columns = list(asset_prices.keys())
    price_frame = price_frame.sort_index()
    price_frame = price_frame[~price_frame.index.duplicated(keep="last")]

    returns_1h = resample_and_log_returns(price_frame, "1h")
    returns_1d = resample_and_log_returns(price_frame, "1D")

    path_1h = out_dir / "returns_1h.parquet"
    path_1d = out_dir / "returns_1d.parquet"

    returns_1h.to_parquet(path_1h)
    returns_1d.to_parquet(path_1d)

    print(f"[data_prep] 1H returns shape: {returns_1h.shape}")
    print(f"[data_prep] 1D returns shape: {returns_1d.shape}")

    return path_1h, path_1d


if __name__ == "__main__":
    prepare_data()
