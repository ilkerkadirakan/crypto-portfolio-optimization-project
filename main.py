"""
Entry point module for orchestrating the ML-enhanced moment-based portfolio pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import yaml

from src import (
    backtest_engine,
    combination_utils,
    data_prep,
    metrics,
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
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel workers for backtesting. -1 uses all CPUs (default).",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing (use sequential execution).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint if available.",
    )
    parser.add_argument(
        "--checkpoint-batch-size",
        type=int,
        default=500,
        help="Save checkpoint every N combinations (default: 500).",
    )
    parser.add_argument(
        "--skip-prep",
        action="store_true",
        help="Skip data preparation step if processed returns already exist.",
    )
    parser.add_argument(
        "--skip-moments",
        action="store_true",
        help="Skip moment calculation if moments parquet files already exist.",
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
        combos_by_size = combination_utils.cache_combinations(group_sizes=target_tuple)
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


def _get_checkpoint_path(results_dir: Path, freq: str, version: str) -> Path:
    """Get checkpoint file path."""
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"checkpoint_{version}_{freq.lower()}.parquet"


def _load_checkpoint(checkpoint_path: Path) -> pd.DataFrame:
    """Load checkpoint if exists."""
    if checkpoint_path.exists():
        print(f"[Checkpoint] ✓ Loading from {checkpoint_path}")
        df = pd.read_parquet(checkpoint_path)
        print(f"[Checkpoint] ✓ Found {len(df)} previously computed rows")
        return df
    return pd.DataFrame()


def _save_checkpoint(df: pd.DataFrame, checkpoint_path: Path) -> None:
    """Save checkpoint."""
    df.to_parquet(checkpoint_path)
    print(f"[Checkpoint] ✓ Saved {len(df)} rows")


def _get_completed_combos(checkpoint_df: pd.DataFrame) -> set:
    """Extract completed combo+model pairs from checkpoint."""
    if checkpoint_df.empty:
        return set()

    completed = set()
    for _, group in checkpoint_df.groupby(["combo", "model"]):
        combo = group["combo"].iloc[0]
        model = str(group["model"].iloc[0]).upper()
        completed.add((combo, model))

    return completed


# Enhanced checkpoint helpers (redefined for safety and backward compatibility)
def _build_run_signature(
    freq: str,
    checkpoint_scope: str,
    model_list: Sequence[str],
    combos: Sequence[object],
) -> str:
    """Build a stable signature to prevent cross-run checkpoint reuse."""
    combo_labels = [backtest_engine._combo_label(combo) for combo in combos]
    combo_labels_sorted = sorted(combo_labels)
    combo_hash = hashlib.sha256("\n".join(combo_labels_sorted).encode("utf-8")).hexdigest()
    payload = {
        "scope": checkpoint_scope.lower(),
        "freq": str(freq).upper(),
        "models": sorted(str(m).upper() for m in model_list),
        "combo_count": len(combo_labels_sorted),
        "combo_hash": combo_hash,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_checkpoint(checkpoint_path: Path, expected_signature: str | None = None) -> pd.DataFrame:
    """Load checkpoint if exists and optionally validate run signature."""
    if checkpoint_path.exists():
        print(f"[Checkpoint] Loading from {checkpoint_path}")
        try:
            df = pd.read_parquet(checkpoint_path)
        except Exception as exc:
            print(f"[Checkpoint] Failed to read checkpoint parquet ({exc}). Ignoring file.")
            return pd.DataFrame()

        if expected_signature is not None:
            sig_col = "_checkpoint_signature"
            if sig_col not in df.columns:
                print("[Checkpoint] Missing signature column in checkpoint. Ignoring old checkpoint.")
                return pd.DataFrame()
            checkpoint_signature = str(df[sig_col].iloc[0]) if not df.empty else ""
            if checkpoint_signature != expected_signature:
                print("[Checkpoint] Signature mismatch. Ignoring checkpoint from different run setup.")
                return pd.DataFrame()

        print(f"[Checkpoint] Found {len(df)} previously computed rows")
        return df
    return pd.DataFrame()


def _save_checkpoint(df: pd.DataFrame, checkpoint_path: Path, signature: str | None = None) -> None:
    """Save checkpoint atomically to reduce corruption risk on interruptions."""
    checkpoint_df = df.copy()
    if signature is not None:
        checkpoint_df["_checkpoint_signature"] = signature
    checkpoint_df["_checkpoint_saved_at"] = datetime.now(timezone.utc).isoformat()

    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    checkpoint_df.to_parquet(tmp_path)
    tmp_path.replace(checkpoint_path)
    print(f"[Checkpoint] Saved {len(checkpoint_df)} rows")


def _get_completed_combos(checkpoint_df: pd.DataFrame) -> set:
    """Extract completed combo+model pairs from checkpoint."""
    if checkpoint_df.empty:
        return set()
    required_cols = {"combo", "model"}
    if not required_cols.issubset(set(checkpoint_df.columns)):
        print("[Checkpoint] Missing required columns (combo/model). Ignoring checkpoint rows.")
        return set()

    completed = set()
    for _, group in checkpoint_df.groupby(["combo", "model"]):
        combo = group["combo"].iloc[0]
        model = str(group["model"].iloc[0]).upper()
        completed.add((combo, model))

    return completed


def run_pipeline(
    config_path: Path,
    frequencies: Sequence[str] | None,
    versions: Sequence[str] | None,
    models: Sequence[str] | None,
    n_jobs: int = -1,
    use_parallel: bool = True,
    resume: bool = False,
    checkpoint_batch_size: int = 500,
    skip_prep: bool = False,
    skip_moments: bool = False,
) -> None:
    """
    Execute the full portfolio optimization workflow with checkpoint support.

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
    n_jobs : int, optional
        Number of parallel workers. -1 uses all CPUs (default).
    use_parallel : bool, optional
        Whether to use parallel processing (default: True).
    resume : bool, optional
        Whether to resume from last checkpoint (default: False).
    checkpoint_batch_size : int, optional
        Save checkpoint every N combinations (default: 500).

    Returns
    -------
    None
        This function performs actions for their side effects and prints progress.

    TODO
    ----
    - Emit structured logging for pipeline monitoring in production environments.
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

    if skip_prep:
        print("[main] Skipping data preparation step (skip-prep enabled).")
    else:
        print("[main] Starting data preparation step.")
        data_prep.prepare_data(raw_data_dir=paths["raw"], processed_dir=paths["processed"])

    if skip_moments:
        print("[main] Skipping moment calculation (skip-moments enabled).")
    else:
        print("[main] Computing realized statistical moments.")
        moment_calc.calc_all_moments(processed_dir=paths["processed"])

    pipeline_results_dir = paths["results"] / "pipeline"
    pipeline_results_dir.mkdir(parents=True, exist_ok=True)

    run_outputs: List[pd.DataFrame] = []

    print("\n" + "="*80)
    print("DUAL-WINNER PORTFOLIO OPTIMIZATION FRAMEWORK")
    print("="*80)
    print(f"Testing {len(combos)} portfolio combinations")
    print(f"Models: {', '.join(model_list)}")
    print(f"Frequencies: {', '.join(freq_list)}")
    print("="*80)

    # STEP 1: Run TEACHER (Classical Optimizer) for ALL combinations
    print("\n" + "="*80)
    print("STEP 1: TEACHER PORTFOLIOS (Classical Optimization)")
    print("="*80)

    for freq in freq_list:
        print(f"\n[Teacher] Testing {len(combos)} combinations at freq={freq}")
        teacher_signature = _build_run_signature(
            freq=freq,
            checkpoint_scope="teacher",
            model_list=model_list,
            combos=combos,
        )

        # CHECKPOINT SYSTEM
        checkpoint_path = _get_checkpoint_path(paths["results"], freq, "teacher")

        if resume:
            checkpoint_df = _load_checkpoint(checkpoint_path, expected_signature=teacher_signature)
            completed_combos = _get_completed_combos(checkpoint_df)
            print(f"[Teacher] Found {len(completed_combos)} completed combo+model pairs")
        else:
            checkpoint_df = pd.DataFrame()
            completed_combos = set()

        # Filter remaining combinations per model
        remaining_by_model = {model: [] for model in model_list}
        for combo in combos:
            combo_str = backtest_engine._combo_label(combo)
            for model in model_list:
                if (combo_str, model) not in completed_combos:
                    remaining_by_model[model].append(combo)

        remaining_total = sum(len(items) for items in remaining_by_model.values())
        print(f"[Teacher] Remaining combo+model pairs to process: {remaining_total}")
        for model in model_list:
            print(f"[Teacher]   {model}: {len(remaining_by_model[model])}/{len(combos)} combos")

        if remaining_total == 0 and not checkpoint_df.empty:
            print("[Teacher] All combinations already completed (using checkpoint)")
            teacher_results = checkpoint_df
        else:
            # Process in batches with checkpoints
            all_teacher_results = [checkpoint_df] if not checkpoint_df.empty else []

            for model in model_list:
                remaining_combos = remaining_by_model[model]
                if not remaining_combos:
                    continue

                for batch_start in range(0, len(remaining_combos), checkpoint_batch_size):
                    batch_end = min(batch_start + checkpoint_batch_size, len(remaining_combos))
                    batch_combos = remaining_combos[batch_start:batch_end]

                    print(f"\n[Teacher] Processing {model} batch {batch_start//checkpoint_batch_size + 1}: "
                          f"combos {batch_start+1}-{batch_end}/{len(remaining_combos)}")

                    if use_parallel:
                        batch_results = backtest_engine.run_backtest_parallel(
                            freq=freq,
                            version="baseline",
                            model_list=[model],
                            combo_iterable=batch_combos,
                            n_jobs=n_jobs,
                        )
                    else:
                        batch_results = backtest_engine.run_backtest(
                            freq=freq,
                            version="baseline",
                            model_list=[model],
                            combo_iterable=batch_combos,
                        )

                    if not batch_results.empty:
                        all_teacher_results.append(batch_results)

                        # Save checkpoint
                        combined_df = pd.concat(all_teacher_results, ignore_index=True)
                        _save_checkpoint(combined_df, checkpoint_path, signature=teacher_signature)

            # Combine all results
            if all_teacher_results:
                teacher_results = pd.concat(all_teacher_results, ignore_index=True)
            else:
                teacher_results = pd.DataFrame()

        if teacher_results.empty:
            print(f"[Teacher] No results for freq={freq}. Skipping.")
            continue

        # Save teacher results
        teacher_path = pipeline_results_dir / f"teacher_{freq.lower()}.parquet"
        teacher_results.to_parquet(teacher_path)
        print(f"[Teacher] Saved results to {teacher_path}")
        run_outputs.append(teacher_results)

        # STEP 2: Generate TEACHER RANKING
        print(f"\nSTEP 2: Generating TEACHER RANKING for freq={freq}")

        import numpy as np
        teacher_grouped = teacher_results.groupby(['combo', 'model'])['net_return'].agg(['mean', 'std', 'count'])
        # Calculate annualized Sharpe ratio using 365-day convention
        daily_sharpe = teacher_grouped['mean'] / (teacher_grouped['std'] + 1e-10)
        teacher_grouped['sharpe'] = daily_sharpe * np.sqrt(365)
        teacher_grouped['annualized_return'] = teacher_grouped['mean'] * 365
        teacher_grouped['volatility'] = teacher_grouped['std'] * np.sqrt(365)

        teacher_ranking = teacher_grouped.sort_values('sharpe', ascending=False).reset_index()

        # Save teacher ranking
        teacher_ranking_path = pipeline_results_dir / f"teacher_ranking_{freq.lower()}.csv"
        teacher_ranking.to_csv(teacher_ranking_path, index=False)
        print(f"[Teacher] Saved ranking to {teacher_ranking_path}")

        # Display top 10 teacher portfolios
        print(f"\nTOP 10 TEACHER PORTFOLIOS ({freq}):")
        print("="*100)
        print(f"{'Rank':<6}{'Combo':<35}{'Model':<10}{'Sharpe':<10}{'Return':<12}{'Vol':<10}")
        print("-"*100)
        for idx, row in teacher_ranking.head(10).iterrows():
            print(f"{idx+1:<6}{row['combo'][:34]:<35}{row['model']:<10}{row['sharpe']:<10.4f}{row['annualized_return']:<12.2%}{row['volatility']:<10.2%}")
        print("="*100)

        # Find TEACHER WINNER
        teacher_winner = teacher_ranking.iloc[0]
        print(f"\nTEACHER WINNER ({freq}):")
        print(f"   Combo: {teacher_winner['combo']}")
        print(f"   Model: {teacher_winner['model']}")
        print(f"   Sharpe: {teacher_winner['sharpe']:.4f}")
        print(f"   Annual Return: {teacher_winner['annualized_return']:.2%}")
        print(f"   Volatility: {teacher_winner['volatility']:.2%}")

        # Save teacher winner
        teacher_winner_data = {
            'freq': freq,
            'combo': teacher_winner['combo'],
            'model': teacher_winner['model'],
            'sharpe': float(teacher_winner['sharpe']),
            'annualized_return': float(teacher_winner['annualized_return']),
            'volatility': float(teacher_winner['volatility']),
            'version': 'teacher'
        }
        import json
        winner_path = pipeline_results_dir / f"winner_teacher_{freq.lower()}.json"
        with open(winner_path, 'w') as f:
            json.dump(teacher_winner_data, f, indent=2)
        print(f"[Teacher] Saved winner to {winner_path}")

        # STEP 3: Train ML-WEIGHTS (Student) for ALL combinations
        if "ml" in version_list:
            print(f"\n" + "="*80)
            print("STEP 3: STUDENT PORTFOLIOS (ML Direct Weight Learning)")
            print("="*80)

            # Import ML weights module (we'll create this)
            try:
                from src import ml_weights
                print("[Student] Training ML models to learn portfolio weights...")
                ml_weights.train_weight_models(
                    teacher_results=teacher_results,
                    processed_dir=paths["processed"],
                    freq=freq,
                    use_ensemble=True,
                    model_types=['lgb', 'xgb', 'rf']  # Use ensemble of all models
                )
                print("[Student] ML weight models trained successfully")
                student_signature = _build_run_signature(
                    freq=freq,
                    checkpoint_scope="student",
                    model_list=model_list,
                    combos=combos,
                )

                # Run backtest with ML-predicted weights
                print(f"[Student] Running backtest with ML weights for {len(combos)} combinations...")

                # CHECKPOINT SYSTEM FOR STUDENT
                checkpoint_path_student = _get_checkpoint_path(paths["results"], freq, "student")

                if resume:
                    checkpoint_df_student = _load_checkpoint(
                        checkpoint_path_student,
                        expected_signature=student_signature,
                    )
                    completed_combos_student = _get_completed_combos(checkpoint_df_student)
                    print(f"[Student] Found {len(completed_combos_student)} completed combo+model pairs")
                else:
                    checkpoint_df_student = pd.DataFrame()
                    completed_combos_student = set()

                # Filter remaining combinations per model for student
                remaining_by_model_student = {model: [] for model in model_list}
                for combo in combos:
                    combo_str = backtest_engine._combo_label(combo)
                    for model in model_list:
                        if (combo_str, model) not in completed_combos_student:
                            remaining_by_model_student[model].append(combo)

                remaining_total_student = sum(len(items) for items in remaining_by_model_student.values())
                print(f"[Student] Remaining combo+model pairs to process: {remaining_total_student}")
                for model in model_list:
                    print(f"[Student]   {model}: {len(remaining_by_model_student[model])}/{len(combos)} combos")

                if remaining_total_student == 0 and not checkpoint_df_student.empty:
                    print("[Student] All combinations already completed (using checkpoint)")
                    student_results = checkpoint_df_student
                else:
                    # Process in batches with checkpoints
                    all_student_results = [checkpoint_df_student] if not checkpoint_df_student.empty else []

                    for model in model_list:
                        remaining_combos_student = remaining_by_model_student[model]
                        if not remaining_combos_student:
                            continue

                        for batch_start in range(0, len(remaining_combos_student), checkpoint_batch_size):
                            batch_end = min(batch_start + checkpoint_batch_size, len(remaining_combos_student))
                            batch_combos = remaining_combos_student[batch_start:batch_end]

                            print(f"\n[Student] Processing {model} batch {batch_start//checkpoint_batch_size + 1}: "
                                  f"combos {batch_start+1}-{batch_end}/{len(remaining_combos_student)}")

                            if use_parallel:
                                batch_results = backtest_engine.run_backtest_parallel(
                                    freq=freq,
                                    version="ml",
                                    model_list=[model],
                                    combo_iterable=batch_combos,
                                    n_jobs=n_jobs,
                                )
                            else:
                                batch_results = backtest_engine.run_backtest(
                                    freq=freq,
                                    version="ml",
                                    model_list=[model],
                                    combo_iterable=batch_combos,
                                )

                            if not batch_results.empty:
                                all_student_results.append(batch_results)

                                # Save checkpoint
                                combined_df_student = pd.concat(all_student_results, ignore_index=True)
                                _save_checkpoint(
                                    combined_df_student,
                                    checkpoint_path_student,
                                    signature=student_signature,
                                )

                    # Combine all results
                    if all_student_results:
                        student_results = pd.concat(all_student_results, ignore_index=True)
                    else:
                        student_results = pd.DataFrame()

                if not student_results.empty:
                    student_path = pipeline_results_dir / f"student_{freq.lower()}.parquet"
                    student_results.to_parquet(student_path)
                    print(f"[Student] Saved results to {student_path}")
                    run_outputs.append(student_results)

                    # STEP 4: Generate STUDENT RANKING
                    print(f"\n STEP 4: Generating STUDENT RANKING for freq={freq}")

                    student_grouped = student_results.groupby(['combo', 'model'])['net_return'].agg(['mean', 'std', 'count'])
                    # Calculate annualized Sharpe ratio using 365-day convention
                    daily_sharpe_student = student_grouped['mean'] / (student_grouped['std'] + 1e-10)
                    student_grouped['sharpe'] = daily_sharpe_student * np.sqrt(365)
                    student_grouped['annualized_return'] = student_grouped['mean'] * 365
                    student_grouped['volatility'] = student_grouped['std'] * np.sqrt(365)

                    student_ranking = student_grouped.sort_values('sharpe', ascending=False).reset_index()

                    # Save student ranking
                    student_ranking_path = pipeline_results_dir / f"student_ranking_{freq.lower()}.csv"
                    student_ranking.to_csv(student_ranking_path, index=False)
                    print(f"[Student] Saved ranking to {student_ranking_path}")

                    # Display top 10 student portfolios
                    print(f"\n TOP 10 STUDENT PORTFOLIOS ({freq}):")
                    print("="*100)
                    print(f"{'Rank':<6}{'Combo':<35}{'Model':<10}{'Sharpe':<10}{'Return':<12}{'Vol':<10}")
                    print("-"*100)
                    for idx, row in student_ranking.head(10).iterrows():
                        print(f"{idx+1:<6}{row['combo'][:34]:<35}{row['model']:<10}{row['sharpe']:<10.4f}{row['annualized_return']:<12.2%}{row['volatility']:<10.2%}")
                    print("="*100)

                    # Find STUDENT WINNER
                    student_winner = student_ranking.iloc[0]
                    print(f"\n STUDENT WINNER ({freq}):")
                    print(f"   Combo: {student_winner['combo']}")
                    print(f"   Model: {student_winner['model']}")
                    print(f"   Sharpe: {student_winner['sharpe']:.4f}")
                    print(f"   Annual Return: {student_winner['annualized_return']:.2%}")
                    print(f"   Volatility: {student_winner['volatility']:.2%}")

                    # Save student winner
                    student_winner_data = {
                        'freq': freq,
                        'combo': student_winner['combo'],
                        'model': student_winner['model'],
                        'sharpe': float(student_winner['sharpe']),
                        'annualized_return': float(student_winner['annualized_return']),
                        'volatility': float(student_winner['volatility']),
                        'version': 'student'
                    }
                    winner_path = pipeline_results_dir / f"winner_student_{freq.lower()}.json"
                    with open(winner_path, 'w') as f:
                        json.dump(student_winner_data, f, indent=2)
                    print(f"[Student] Saved winner to {winner_path}")

                    # STEP 5: COMPARE TEACHER vs STUDENT
                    print(f"\n" + "="*80)
                    print(f"  STEP 5: TEACHER vs STUDENT COMPARISON ({freq})")
                    print("="*80)

                    print(f"\n{'Metric':<25}{'Teacher':<20}{'Student':<20}{'Winner':<15}")
                    print("-"*80)
                    print(f"{'Sharpe Ratio':<25}{teacher_winner['sharpe']:<20.4f}{student_winner['sharpe']:<20.4f}{' ' + ('Teacher' if teacher_winner['sharpe'] > student_winner['sharpe'] else 'Student'):<15}")
                    print(f"{'Annual Return':<25}{teacher_winner['annualized_return']:<20.2%}{student_winner['annualized_return']:<20.2%}{' ' + ('Teacher' if teacher_winner['annualized_return'] > student_winner['annualized_return'] else 'Student'):<15}")
                    print(f"{'Volatility':<25}{teacher_winner['volatility']:<20.2%}{student_winner['volatility']:<20.2%}{' ' + ('Teacher' if teacher_winner['volatility'] < student_winner['volatility'] else 'Student'):<15}")
                    print("="*80)

                    # Save comparison
                    comparison_data = {
                        'freq': freq,
                        'teacher': teacher_winner_data,
                        'student': student_winner_data,
                        'winner': 'teacher' if teacher_winner['sharpe'] > student_winner['sharpe'] else 'student'
                    }
                    comparison_path = pipeline_results_dir / f"teacher_vs_student_{freq.lower()}.json"
                    with open(comparison_path, 'w') as f:
                        json.dump(comparison_data, f, indent=2)
                    print(f"\n Saved comparison to {comparison_path}")

                else:
                    print(f"[Student] No results generated")

            except ImportError as e:
                print(f"[Student] ML weights module not available: {e}")
                print("[Student] Skipping ML weight learning")

        print(f"\n Completed optimization pipeline for freq={freq}")
        print("="*80)

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
            n_jobs=args.n_jobs,
            use_parallel=not args.no_parallel,
            resume=args.resume,
            checkpoint_batch_size=args.checkpoint_batch_size,
            skip_prep=args.skip_prep,
            skip_moments=args.skip_moments,
        )
        return 0
    except Exception as exc:  # pragma: no cover - defensive guard for CLI usage
        print(f"[main] Pipeline execution failed: {exc}")
        return 1



if __name__ == "__main__":
    sys.exit(main())
