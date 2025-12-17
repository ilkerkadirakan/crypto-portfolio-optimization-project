"""
Entry point module for orchestrating the ML-enhanced moment-based portfolio pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import yaml

from src import (
    backtest_engine,
    combination_utils,
    data_prep,
    metrics,
    ml_forecast,
    moment_calc,
    reporting,
)

DEFAULT_MODELS: Sequence[str] = ("MV", "MVSK", "MCVaRSK")
DEFAULT_VERSIONS: Sequence[str] = ("baseline", "ml")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments for pipeline orchestration.

    Parameters
    ----------
    argv : Sequence[str] | None
        Optional sequence of CLI arguments; defaults to sys.argv when None.

    Returns
    -------
    argparse.Namespace
        Parsed arguments accessible by attribute lookup.

    TODO
    ----
    - Surface fine-grained toggles for skipping specific pipeline stages.
    - Add CLI shortcuts for selecting predefined scenario combinations.
    """
    parser = argparse.ArgumentParser(
        description="Run the ML-enhanced moment-based portfolio optimization pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs") / "params.yaml",
        help="Path to the pipeline configuration file.",
    )
    parser.add_argument(
        "--frequencies",
        nargs="*",
        help="Optional list of frequencies to process (e.g. 1D 1H). Defaults to config values.",
    )
    parser.add_argument(
        "--versions",
        nargs="*",
        help="Optional list of data versions to backtest (baseline, ml). Defaults to both.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Optional list of optimization models (MV, MVSK, MCVaRSK). Defaults to all.",
    )
    return parser.parse_args(argv)


def _load_config(config_path: Path) -> Dict[str, object]:
    """
    Load YAML configuration parameters used throughout the pipeline.

    Parameters
    ----------
    config_path : Path
        Filesystem path to the params.yaml configuration file.

    Returns
    -------
    Dict[str, object]
        Parsed configuration dictionary; empty dict if file missing.

    TODO
    ----
    - Validate schema and provide user-friendly error messages for malformed files.
    - Introduce caching to avoid repeated disk reads during iterative development.
    """
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_project_paths(config: Dict[str, object], project_root: Path) -> Dict[str, Path]:
    """
    Resolve key project directories based on configuration defaults.

    Parameters
    ----------
    config : Dict[str, object]
        Runtime configuration dictionary.
    project_root : Path
        Base directory of the repository derived from main.py location.

    Returns
    -------
    Dict[str, Path]
        Mapping containing 'raw', 'processed', and 'results' directories.

    TODO
    ----
    - Support environment variable overrides for deployment-specific locations.
    - Add validation ensuring required directories exist before pipeline execution.
    """
    data_cfg = config.get("data", {}) if isinstance(config, dict) else {}
    raw_dir = project_root / data_cfg.get("raw_dir", "data/raw")
    processed_dir = project_root / data_cfg.get("processed_dir", "data/processed")
    results_dir = project_root / data_cfg.get("results_dir", "results")
    return {
        "raw": raw_dir,
        "processed": processed_dir,
        "results": results_dir,
    }


def _resolve_frequencies(config: Dict[str, object], explicit: Sequence[str] | None) -> List[str]:
    """
    Determine which frequencies should be processed by the pipeline.

    Parameters
    ----------
    config : Dict[str, object]
        Runtime configuration dictionary.
    explicit : Sequence[str] | None
        Optional user-specified frequency overrides.

    Returns
    -------
    List[str]
        Ordered list of normalized frequency strings (e.g. ['1D', '1H']).

    TODO
    ----
    - Validate that requested frequencies have corresponding data files.
    - Extend to support additional cadences beyond daily and hourly.
    """
    if explicit:
        normalized: List[str] = []
        for freq in explicit:
            canonical = str(freq).strip().upper()
            if canonical not in {"1H", "1D"}:
                raise ValueError(f"Unsupported frequency '{freq}'. Expected 1H or 1D.")
            if canonical not in normalized:
                normalized.append(canonical)
        return normalized

    freq_cfg = config.get("data", {}).get("frequencies", {}) if isinstance(config, dict) else {}
    order: List[str] = []
    for alias in ("daily", "hourly"):
        if alias in freq_cfg:
            candidate = str(freq_cfg[alias]).strip().upper()
            if candidate and candidate not in order:
                order.append(candidate)
    if not order:
        order = ["1D", "1H"]
    return order


def _normalize_versions(versions: Sequence[str] | None) -> List[str]:
    """
    Normalize version identifiers to canonical keywords consumed by backtests.

    Parameters
    ----------
    versions : Sequence[str] | None
        Optional user-specified version list; defaults to baseline and ml.

    Returns
    -------
    List[str]
        Lower-case list of version identifiers.

    TODO
    ----
    - Allow configuration-driven alias mapping beyond simple defaults.
    - Provide more descriptive guidance when unsupported versions are supplied.
    """
    if not versions:
        return [version for version in DEFAULT_VERSIONS]

    mapping = {
        "baseline": "baseline",
        "realized": "baseline",
        "realised": "baseline",
        "ml": "ml",
        "forecast": "ml",
        "forecasted": "ml",
    }
    normalized: List[str] = []
    for version in versions:
        key = str(version).strip().lower()
        if key not in mapping:
            raise ValueError(f"Unsupported version '{version}'. Expected one of {sorted(mapping)}.")
        normalized_value = mapping[key]
        if normalized_value not in normalized:
            normalized.append(normalized_value)
    return normalized


def _resolve_models(models: Sequence[str] | None) -> List[str]:
    """
    Normalize optimization model names to uppercase identifiers.

    Parameters
    ----------
    models : Sequence[str] | None
        Optional selection of model names; defaults to MV, MVSK, MCVaRSK.

    Returns
    -------
    List[str]
        Uppercase model names ready for backtest invocation.

    TODO
    ----
    - Validate against a dynamically discovered set of solver-ready models.
    - Provide aliases for common shorthand (e.g., 'mvsk' -> 'MVSK').
    """
    if not models:
        return [model for model in DEFAULT_MODELS]
    normalized: List[str] = []
    for model in models:
        canonical = str(model).strip().upper()
        if canonical not in {"MV", "MVSK", "MCVARSK"}:
            raise ValueError("Unsupported model '{0}'. Expected MV, MVSK, or MCVaRSK.".format(model))
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized


def _generate_cached_combinations(group_sizes: Sequence[int] | None = None) -> List[Tuple[str, ...]]:
    """
    Load or build cached asset combinations using the shared utility module.

    Parameters
    ----------
    group_sizes : Sequence[int] | None, optional
        Explicit combination sizes to generate. Defaults to the utility module's
        standard group sizes when omitted.

    Returns
    -------
    List[Tuple[str, ...]]
        Flattened list of asset tuples ready for downstream consumption.

    TODO
    ----
    - Allow asynchronous/pre-emptive cache warmup to hide generation latency.
    - Surface richer telemetry (e.g., unique count) for pipeline logging.
    """
    default_sizes = combination_utils.DEFAULT_GROUP_SIZES
    target_sizes = group_sizes or default_sizes
    try:
        target_tuple = tuple(int(size) for size in target_sizes)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid combination group sizes: {group_sizes}") from exc

    try:
        combos_by_size = combination_utils.cache_combinations(group_szes=target_tuple)
    except Exception as exc:  # pragma: no cover - defensive guard
        print(f"[main] Failed to generate cached combinations: {exc}")
        return []

    def _size_key(item: str) -> Tuple[int, str]:
        try:
            return (int(item), item)
        except ValueError:
            return (sys.maxsize, item)

    ordered_keys = sorted(combos_by_size.keys(), key=_size_key)
    flattened: List[Tuple[str, ...]] = []
    for key in ordered_keys:
        flattened.extend(combos_by_size.get(key, []))
    return flattened


def _resolve_combo_iterable(backtest_cfg: Dict[str, object]) -> List[object]:
    """
    Interpret backtest configuration entries into a combination iterable.

    Parameters
    ----------
    backtest_cfg : Dict[str, object]
        Backtest-specific settings pulled from the primary configuration file.

    Returns
    -------
    List[object]
        Scenario definitions compatible with ``backtest_engine.run_backtest``.

    TODO
    ----
    - Support referencing pre-defined scenario templates stored on disk.
    - Add validation for dict-based overrides (e.g., ensure assets exist).
    """
    combos_entry: object | None = None
    if isinstance(backtest_cfg, dict):
        combos_entry = backtest_cfg.get("combos") or backtest_cfg.get("scenarios")
    else:
        combos_entry = backtest_cfg

    if combos_entry is None:
        return _generate_cached_combinations()

    if isinstance(combos_entry, dict):
        mode = str(combos_entry.get("mode", "auto")).lower()
        group_sizes = combos_entry.get("group_sizes")
        if combos_entry.get("auto", False) or mode in {"auto", "cache", "default"}:
            return _generate_cached_combinations(group_sizes)
        manual = combos_entry.get("manual")
        if isinstance(manual, (list, tuple, set)):
            return list(manual)
        if "group_sizes" in combos_entry:
            return _generate_cached_combinations(group_sizes)
        return [combos_entry]

    if isinstance(combos_entry, (list, tuple, set)):
        return list(combos_entry)

    if isinstance(combos_entry, str):
        lowered = combos_entry.lower()
        if lowered in {"auto", "cache", "default", "generate"}:
            return _generate_cached_combinations()
        if lowered in {"full", "universe", "single"}:
            return [None]
        return [combos_entry]

    return [combos_entry]


def run_pipeline(
    config_path: Path,
    frequencies: Sequence[str] | None,
    versions: Sequence[str] | None,
    models: Sequence[str] | None,
) -> None:
    """
    Execute the full portfolio optimization workflow in sequential stages.

    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration file guiding the pipeline.
    frequencies : Sequence[str] | None
        Optional frequency overrides supplied via CLI.
    versions : Sequence[str] | None
        Optional version overrides supplied via CLI.
    models : Sequence[str] | None
        Optional model overrides supplied via CLI.

    Returns
    -------
    None
        This function performs actions for their side effects and prints progress.

    TODO
    ----
    - Emit structured logging for pipeline monitoring in production environments.
    - Parallelize frequency-specific backtests when resource constraints allow it.
    """
    project_root = Path(__file__).resolve().parent
    config = _load_config(config_path)
    paths = _resolve_project_paths(config, project_root)
    paths["processed"].mkdir(parents=True, exist_ok=True)
    paths["results"].mkdir(parents=True, exist_ok=True)

    freq_list = _resolve_frequencies(config, frequencies)
    version_list = _normalize_versions(versions)
    model_list = _resolve_models(models)

    backtest_cfg = config.get("backtest", {}) if isinstance(config, dict) else {}
    combos: List[object] = _resolve_combo_iterable(backtest_cfg)

    print("[main] Starting data preparation step.")
    data_prep.prepare_data(raw_data_dir=paths["raw"], processed_dir=paths["processed"])

    print("[main] Computing realized statistical moments.")
    moment_calc.calc_all_moments(processed_dir=paths["processed"])

    print("[main] Forecasting future moments via machine learning models.")
    ml_forecast.forecast_moments(processed_dir=paths["processed"])

    pipeline_results_dir = paths["results"] / "pipeline"
    pipeline_results_dir.mkdir(parents=True, exist_ok=True)

    run_outputs: List[pd.DataFrame] = []

    for freq in freq_list:
        for version in version_list:
            print(f"[main] Running backtest for freq={freq} version={version} models={model_list}.")
            results_df = backtest_engine.run_backtest(
                freq=freq,
                version=version,
                model_list=model_list,
                combo_iterable=combos,
            )
            if results_df.empty:
                print(f"[main] No backtest records generated for freq={freq} version={version}.")
                continue

            output_path = pipeline_results_dir / f"backtest_{freq.lower()}_{version}.parquet"
            results_df.to_parquet(output_path)
            print(f"[main] Saved backtest results to {output_path}")
            run_outputs.append(results_df)

    if run_outputs:
        combined_runs = pd.concat(run_outputs, ignore_index=True)
        print("[main] Summarizing portfolio performance metrics.")
        summary_df = pd.DataFrame()
        try:
            summary_df = metrics.summarize_portfolios(combined_runs)
        except Exception as exc:
            print(f"[main] Metric summarization failed: {exc}")
        else:
            summary_path = pipeline_results_dir / "summary_metrics.parquet"
            summary_df.to_parquet(summary_path)
            print(f"[main] Saved summary metrics to {summary_path}")

        print("[main] Generating analytical reports.")
        try:
            reporting.generate_all_reports(combined_runs)
        except Exception as exc:
            print(f"[main] Report generation failed: {exc}")


def main(argv: Sequence[str] | None = None) -> int:
    """
    CLI wrapper for executing the portfolio optimization pipeline.

    Parameters
    ----------
    argv : Sequence[str] | None
        Optional CLI argument list passed from the command line.

    Returns
    -------
    int
        Process exit code where zero denotes success.

    TODO
    ----
    - Capture and report detailed exception information for debugging.
    - Return non-zero codes for partial failures (e.g., missing LightGBM).
    """
    try:
        args = _parse_args(argv)
        run_pipeline(
            config_path=args.config,
            frequencies=args.frequencies,
            versions=args.versions,
            models=args.models,
        )
        return 0
    except Exception as exc:  # pragma: no cover - defensive guard for CLI usage
        print(f"[main] Pipeline execution failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
