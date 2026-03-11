"""
Run only the Student (ML weight learning) using existing Teacher results.
No need to recalculate all combinations.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from src import ml_weights, backtest_engine, combination_utils
import json
import warnings
from collections import defaultdict
import hashlib
import datetime as dt

# Suppress cvxpy and sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='cvxpy')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


def _get_checkpoint_path(results_dir: Path, freq: str) -> Path:
    """Get checkpoint file path for student."""
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"checkpoint_student_{freq.lower()}.parquet"


def _load_checkpoint(checkpoint_path: Path) -> pd.DataFrame:
    """Load checkpoint if exists, supporting both parquet and CSV formats."""
    # Try parquet first
    if checkpoint_path.exists():
        try:
            print(f"[Checkpoint] ✓ Loading from {checkpoint_path}")
            df = pd.read_parquet(checkpoint_path)
            print(f"[Checkpoint] ✓ Found {len(df)} previously computed rows")
            return df
        except Exception as e:
            print(f"[Checkpoint] Failed to load parquet: {e}")

    # Try CSV alternative
    csv_path = checkpoint_path.with_suffix('.csv')
    if csv_path.exists():
        try:
            print(f"[Checkpoint] ✓ Loading from CSV: {csv_path}")
            df = pd.read_csv(csv_path)
            print(f"[Checkpoint] ✓ Found {len(df)} previously computed rows from CSV")
            return df
        except Exception as e:
            print(f"[Checkpoint] Failed to load CSV: {e}")

    return pd.DataFrame()


def _save_checkpoint(df: pd.DataFrame, checkpoint_path: Path) -> None:
    """Save checkpoint with memory-efficient chunked writing."""
    try:
        # Try normal save first
        if len(df) < 100000:  # For smaller datasets, use normal save
            df.to_parquet(checkpoint_path)
        else:
            # For large datasets, use chunked writing
            _save_large_parquet(df, checkpoint_path)
        print(f"[Checkpoint] ✓ Saved {len(df)} rows")
    except Exception as e:
        print(f"[Checkpoint] Error saving checkpoint: {e}")
        # Try alternative formats
        try:
            csv_path = checkpoint_path.with_suffix('.csv')
            df.to_csv(csv_path, index=False)
            print(f"[Checkpoint] ✓ Saved as CSV: {csv_path}")
        except Exception as csv_e:
            print(f"[Checkpoint] Failed to save as CSV: {csv_e}")


def _save_large_parquet(df: pd.DataFrame, file_path: Path, chunk_size: int = 50000) -> None:
    """Save large DataFrame to parquet using chunked approach."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Convert DataFrame to Arrow table in chunks
    total_rows = len(df)
    print(f"[SaveChunk] Saving {total_rows} rows in chunks of {chunk_size}")

    schema = None
    writer = None
    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")

    try:
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk_df = df.iloc[start_idx:end_idx].copy()

            # Convert chunk to Arrow table
            table = pa.Table.from_pandas(chunk_df, preserve_index=True)

            if schema is None:
                schema = table.schema
                writer = pq.ParquetWriter(tmp_path, schema)

            writer.write_table(table)

            if start_idx % (chunk_size * 5) == 0:  # Progress every 5 chunks
                print(f"[SaveChunk] Progress: {end_idx}/{total_rows} rows ({end_idx/total_rows:.1%})")

    finally:
        if writer:
            writer.close()
    tmp_path.replace(file_path)

    print(f"[SaveChunk] ✓ Completed chunked save to {file_path}")


def _hash_file(path: Path) -> str:
    """Compute a SHA256 hash for a file without loading it all into memory."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _archive_file(src: Path, dest_dir: Path, stem_suffix: str) -> None:
    """Copy a file into the attempt archive directory with a suffix."""
    if not src.exists():
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{src.stem}_{stem_suffix}{src.suffix}"
    dest.write_bytes(src.read_bytes())


def _combo_to_str(combo) -> str:
    """Normalize combo representations into the backtest label format."""
    if isinstance(combo, dict):
        for key in ("name", "label", "id"):
            if key in combo and combo[key]:
                return str(combo[key])
        return "_".join(f"{k}-{v}" for k, v in sorted(combo.items()))
    if isinstance(combo, (list, tuple)):
        return "_".join(str(item) for item in combo)
    return str(combo)


def _get_completed_combos(checkpoint_df: pd.DataFrame) -> set:
    """Extract completed combo+model pairs from checkpoint."""
    if checkpoint_df.empty:
        return set()

    completed = set()
    # Checkpoint'teki 'combo' sütunu zaten string formatında (örn: "BTC_ETH_BNB")
    # Model adlarını normalize et (büyük harfe çevir)
    for combo_str, model in checkpoint_df[['combo', 'model']].drop_duplicates().values:
        model_normalized = str(model).upper()  # Model adını normalize et
        completed.add((combo_str, model_normalized))

    print(f"[Checkpoint] Loaded {len(completed)} completed (combo, model) pairs")
    if completed and len(completed) <= 5:
        print(f"[Checkpoint] Sample (first 5): {list(completed)[:5]}")

    return completed


def _build_run_signature(
    freq: str,
    version_flag: str,
    model_list: list[str],
    combos: list,
) -> str:
    combo_labels = sorted(_combo_to_str(combo) for combo in combos)
    combo_hash = hashlib.sha256("\n".join(combo_labels).encode("utf-8")).hexdigest()
    payload = {
        "freq": str(freq).upper(),
        "version": str(version_flag).lower(),
        "models": sorted(str(m).upper() for m in model_list),
        "combo_count": len(combo_labels),
        "combo_hash": combo_hash,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_checkpoint(checkpoint_path: Path, expected_signature: str | None = None) -> pd.DataFrame:
    """Load checkpoint if exists, supporting both parquet and CSV formats."""
    if checkpoint_path.exists():
        try:
            print(f"[Checkpoint] Loading from {checkpoint_path}")
            df = pd.read_parquet(checkpoint_path)
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
        except Exception as e:
            print(f"[Checkpoint] Failed to load parquet: {e}")

    csv_path = checkpoint_path.with_suffix('.csv')
    if csv_path.exists():
        try:
            print(f"[Checkpoint] Loading from CSV: {csv_path}")
            df = pd.read_csv(csv_path)
            if expected_signature is not None and "_checkpoint_signature" in df.columns and not df.empty:
                checkpoint_signature = str(df["_checkpoint_signature"].iloc[0])
                if checkpoint_signature != expected_signature:
                    print("[Checkpoint] CSV signature mismatch. Ignoring checkpoint.")
                    return pd.DataFrame()
            print(f"[Checkpoint] Found {len(df)} previously computed rows from CSV")
            return df
        except Exception as e:
            print(f"[Checkpoint] Failed to load CSV: {e}")

    return pd.DataFrame()


def _save_checkpoint(df: pd.DataFrame, checkpoint_path: Path, signature: str | None = None) -> None:
    """Save checkpoint with atomic parquet write and CSV fallback."""
    checkpoint_df = df.copy()
    if signature is not None:
        checkpoint_df["_checkpoint_signature"] = signature
    checkpoint_df["_checkpoint_saved_at"] = dt.datetime.utcnow().isoformat()
    try:
        if len(checkpoint_df) < 100000:
            tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
            checkpoint_df.to_parquet(tmp_path)
            tmp_path.replace(checkpoint_path)
        else:
            _save_large_parquet(checkpoint_df, checkpoint_path)
        print(f"[Checkpoint] Saved {len(checkpoint_df)} rows")
    except Exception as e:
        print(f"[Checkpoint] Error saving checkpoint: {e}")
        try:
            csv_path = checkpoint_path.with_suffix('.csv')
            checkpoint_df.to_csv(csv_path, index=False)
            print(f"[Checkpoint] Saved as CSV: {csv_path}")
        except Exception as csv_e:
            print(f"[Checkpoint] Failed to save as CSV: {csv_e}")


def _get_rebalance_timestamps(freq: str, processed_dir: Path) -> list[pd.Timestamp]:
    cfg = backtest_engine._load_config()
    cfg_bt = backtest_engine._get_backtest_config(cfg, freq)
    returns_df = backtest_engine._load_returns(freq, processed_dir).sort_index()
    baseline_panels, _ = backtest_engine._load_moments(freq, processed_dir, "ml")
    first_valid = baseline_panels.get("mean").dropna(how="all").index[0]
    index_common = returns_df.loc[first_valid:].index
    positions = backtest_engine._rebalance_positions(index_common, cfg_bt)
    return [index_common[pos] for pos in positions]


def _run_student_walk_forward(
    teacher_results: pd.DataFrame,
    combos: list,
    freq: str,
    processed_dir: Path,
    pipeline_results_dir: Path,
    model_list: list[str],
    use_ensemble: bool,
    model_types: list,
    label: str,
    top_k_teachers: int,
    same_asset_count: bool,
    n_lags: int,
    noise_std: float,
    noise_samples: int,
    xgb_multi_output: bool,
    softmax_temp: float,
    wf_train_windows: int,
    wf_test_windows: int,
    wf_max_folds: int | None,
) -> None:
    timestamps = _get_rebalance_timestamps(freq, processed_dir)
    if len(timestamps) < wf_train_windows + wf_test_windows:
        print("[Student] Not enough rebalance windows for walk-forward")
        return

    folds = list(range(wf_train_windows, len(timestamps) - wf_test_windows + 1))
    if wf_max_folds:
        folds = folds[-wf_max_folds:]

    wf_results = []
    for fold_idx, i in enumerate(folds, 1):
        train_start = timestamps[i - wf_train_windows]
        train_end = timestamps[i - 1]
        test_start = timestamps[i]
        test_end = timestamps[i + wf_test_windows - 1]

        print(f"\n[WF] Fold {fold_idx}/{len(folds)} train={train_start.date()}..{train_end.date()} "
              f"test={test_start.date()}..{test_end.date()}")

        train_subset = teacher_results[
            (teacher_results["timestamp"] >= train_start)
            & (teacher_results["timestamp"] <= train_end)
        ]
        if train_subset.empty:
            print("[WF] Skipping fold: no teacher data")
            continue

        ml_weights.train_weight_models(
            teacher_results=train_subset,
            processed_dir=processed_dir,
            freq=freq,
            use_ensemble=use_ensemble,
            model_types=model_types,
            use_multi_output_xgb=xgb_multi_output,
            softmax_temp=softmax_temp,
            top_k_teachers=top_k_teachers,
            same_asset_count=same_asset_count,
            n_lags=n_lags,
            noise_std=noise_std,
            noise_samples=noise_samples,
        )

        fold_results = backtest_engine.run_backtest_parallel(
            freq=freq,
            version="ml",
            model_list=model_list,
            combo_iterable=combos,
            n_jobs=-1,
            start_ts=test_start,
            end_ts=test_end,
        )
        if fold_results.empty:
            continue
        fold_results = fold_results.assign(
            wf_fold=fold_idx,
            wf_train_start=train_start,
            wf_train_end=train_end,
            wf_test_start=test_start,
            wf_test_end=test_end,
        )
        wf_results.append(fold_results)

    if not wf_results:
        print("[WF] No results generated")
        return

    student_results = pd.concat(wf_results, ignore_index=True)
    student_path = pipeline_results_dir / f"student_wf_{freq.lower()}.parquet"
    print(f"[WF] Saving {len(student_results)} results to {student_path}")
    _save_large_parquet(student_results, student_path)

    grouped = student_results.groupby(['combo', 'model'])['net_return'].agg(['mean', 'std', 'count'])
    daily_sharpe = grouped['mean'] / (grouped['std'] + 1e-10)
    grouped['sharpe'] = daily_sharpe * np.sqrt(365)
    grouped['annualized_return'] = grouped['mean'] * 365
    grouped['volatility'] = grouped['std'] * np.sqrt(365)

    ranking = grouped.sort_values('sharpe', ascending=False).reset_index()
    ranking_path = pipeline_results_dir / f"student_ranking_wf_{freq.lower()}.csv"
    ranking.to_csv(ranking_path, index=False)
    print(f"[WF] Saved ranking to {ranking_path}")

    winner = ranking.iloc[0]
    winner_data = {
        'freq': freq,
        'combo': winner['combo'],
        'model': winner['model'],
        'sharpe': float(winner['sharpe']),
        'annualized_return': float(winner['annualized_return']),
        'volatility': float(winner['volatility']),
        'version': 'student',
        'walk_forward': True,
        'train_windows': wf_train_windows,
        'test_windows': wf_test_windows,
        'folds': len(folds),
    }
    winner_path = pipeline_results_dir / f"winner_student_wf_{freq.lower()}.json"
    with open(winner_path, 'w') as f:
        json.dump(winner_data, f, indent=2)
    print(f"[WF] Saved winner to {winner_path}")

    teacher_results = teacher_results.copy()
    teacher_results["timestamp"] = pd.to_datetime(teacher_results["timestamp"])
    oos_ts = student_results["timestamp"].unique()
    teacher_oos = teacher_results[teacher_results["timestamp"].isin(oos_ts)]
    teacher_grouped = teacher_oos.groupby(['combo', 'model'])['net_return'].agg(['mean', 'std', 'count'])
    daily_sharpe = teacher_grouped['mean'] / (teacher_grouped['std'] + 1e-10)
    teacher_grouped['sharpe'] = daily_sharpe * np.sqrt(365)
    teacher_grouped['annualized_return'] = teacher_grouped['mean'] * 365
    teacher_grouped['volatility'] = teacher_grouped['std'] * np.sqrt(365)
    teacher_ranking = teacher_grouped.sort_values('sharpe', ascending=False).reset_index()
    teacher_winner = teacher_ranking.iloc[0]

    comparison = {
        'freq': freq,
        'walk_forward': True,
        'teacher': {
            'combo': teacher_winner['combo'],
            'model': teacher_winner['model'],
            'sharpe': float(teacher_winner['sharpe']),
            'annualized_return': float(teacher_winner['annualized_return']),
            'volatility': float(teacher_winner['volatility']),
        },
        'student': {
            'combo': winner['combo'],
            'model': winner['model'],
            'sharpe': float(winner['sharpe']),
            'annualized_return': float(winner['annualized_return']),
            'volatility': float(winner['volatility']),
        },
    }
    comparison_path = pipeline_results_dir / f"teacher_vs_student_wf_{freq.lower()}.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"[WF] Saved comparison to {comparison_path}")


def run_student_only(
    freq="1D",
    checkpoint_batch_size=500,
    model_choice="ensemble",
    combo_limit=None,
    combo_sizes=None,
    top_k_teachers=1,
    same_asset_count=False,
    disable_checkpoint=True,
    model_list=None,
    n_lags=15,
    noise_std=0.0,
    noise_samples=0,
    xgb_multi_output=False,
    softmax_temp=1.0,
    limit_to_predicted_combos=False,
    ml_onfly=False,
    oos_split=0.0,
    walk_forward=False,
    wf_train_windows=None,
    wf_test_windows=None,
    wf_max_folds=None,
):
    """
    Train and evaluate Student models using pre-computed Teacher results.

    Parameters
    ----------
    freq : str
        Frequency to process (1D or 1H)
    """
    project_root = Path(__file__).resolve().parent
    processed_dir = project_root / "data" / "processed"
    pipeline_results_dir = project_root / "results" / "pipeline"

    # Load existing teacher results
    teacher_path = pipeline_results_dir / f"teacher_{freq.lower()}.parquet"

    if not teacher_path.exists():
        print(f"[Error] Teacher results not found at {teacher_path}")
        print("[Error] Please run the full pipeline first with: python main.py --frequencies 1D")
        return

    print(f"[Student-Only] Loading teacher results from {teacher_path}")
    teacher_results = pd.read_parquet(teacher_path)
    print(f"[Student-Only] Loaded {len(teacher_results)} teacher backtest rows")
    teacher_results = teacher_results.copy()
    teacher_results["timestamp"] = pd.to_datetime(teacher_results["timestamp"])

    # Load combinations
    combos_by_size = combination_utils.cache_combinations()
    combos = []
    if combo_sizes:
        size_keys = [str(size) for size in combo_sizes]
    else:
        size_keys = sorted(combos_by_size.keys())
    for size_key in size_keys:
        combos.extend(combos_by_size.get(size_key, []))

    if combo_limit is not None:
        combos = combos[:max(combo_limit, 0)]

    print(f"[Student-Only] Found {len(combos)} portfolio combinations")

    from src import ml_weights
    if model_choice == "ensemble":
        use_ensemble = True
        model_types = ["lgb", "xgb", "rf"]
        label = "ensemble"
    elif model_choice == "xgb":
        use_ensemble = False
        model_types = ["xgb"]
        label = "XGBoost"
    elif model_choice == "cat":
        use_ensemble = False
        model_types = ["cat"]
        label = "CatBoost"
    else:
        use_ensemble = False
        model_types = ["lgb"]
        label = "LightGBM"

    if walk_forward:
        cfg = backtest_engine._load_config()
        wf_cfg = cfg.get("ml", {}).get("walk_forward", {})
        wf_train_windows = wf_train_windows or int(wf_cfg.get("train_windows", 60))
        wf_test_windows = wf_test_windows or int(wf_cfg.get("test_windows", 1))
        _run_student_walk_forward(
            teacher_results=teacher_results,
            combos=combos,
            freq=freq,
            processed_dir=processed_dir,
            pipeline_results_dir=pipeline_results_dir,
            model_list=model_list or ["MV", "MVSK", "MCVaRSK"],
            use_ensemble=use_ensemble,
            model_types=model_types,
            label=label,
            top_k_teachers=top_k_teachers,
            same_asset_count=same_asset_count,
            n_lags=n_lags,
            noise_std=noise_std,
            noise_samples=noise_samples,
            xgb_multi_output=xgb_multi_output,
            softmax_temp=softmax_temp,
            wf_train_windows=wf_train_windows,
            wf_test_windows=wf_test_windows,
            wf_max_folds=wf_max_folds,
        )
        return

    # STEP 1: Train Student ML models
    print("\n" + "="*80)
    print("TRAINING STUDENT (ML Weight Learning)")
    print("="*80)

    print(f"[Student] Training ML {label} models to learn portfolio weights...")
    ml_weights.train_weight_models(
        teacher_results=teacher_results,
        processed_dir=processed_dir,
        freq=freq,
        use_ensemble=use_ensemble,
        model_types=model_types,
        use_multi_output_xgb=xgb_multi_output,
        softmax_temp=softmax_temp,
        top_k_teachers=top_k_teachers,
        same_asset_count=same_asset_count,
        n_lags=n_lags,
        noise_std=noise_std,
        noise_samples=noise_samples,
    )
    print(f"[Student] ML {label} weight models trained successfully")

    # Archive ML artifacts with a hash so experiments are reproducible
    timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    ml_pred_path = processed_dir / f"ml_predicted_weights_{freq.lower()}.parquet"
    ml_model_path = processed_dir / "ml_models" / f"weight_models_{freq.lower()}.pkl"
    attempt_hash = _hash_file(ml_pred_path) if ml_pred_path.exists() else "missing"
    attempt_suffix = f"{timestamp}_{attempt_hash[:8]}"
    attempts_dir = (project_root / "results" / "attempts" / f"student_{freq.lower()}")
    _archive_file(ml_pred_path, attempts_dir, attempt_suffix)
    _archive_file(ml_model_path, attempts_dir, attempt_suffix)
    print(f"[Student] Archived ML artifacts with id={attempt_suffix}")

    if limit_to_predicted_combos:
        if not ml_pred_path.exists():
            raise FileNotFoundError(f"ML predictions not found at {ml_pred_path}")
        pred_df = pd.read_parquet(ml_pred_path, columns=["combo"])
        predicted_combos = set(pred_df["combo"].dropna().unique())
        if not predicted_combos:
            raise ValueError("ML predictions contain no combos; cannot limit student run.")
        combos = [c for c in combos if _combo_to_str(c) in predicted_combos]
        print(f"[Student] Limiting to ML-predicted combos: {len(combos)}")

    # STEP 2: Run Student backtest
    print("\n" + "="*80)
    print("BACKTESTING STUDENT PORTFOLIOS")
    print("="*80)

    model_list = model_list or ["MV", "MVSK", "MCVaRSK"]
    version_flag = "ml_onfly" if ml_onfly else "ml"
    run_signature = _build_run_signature(
        freq=freq,
        version_flag=version_flag,
        model_list=model_list,
        combos=combos,
    )
    results_root = project_root / "results"

    print(f"[Student] Running backtest with ML weights for {len(combos)} combinations...")
    print(f"[Student] Using parallel processing with checkpoint support...")

    # CHECKPOINT SYSTEM
    checkpoint_path = _get_checkpoint_path(results_root, freq)
    if disable_checkpoint:
        print("[Student] Checkpointing disabled for this run")
        checkpoint_df = pd.DataFrame()
        completed_combos = set()
    else:
        checkpoint_df = _load_checkpoint(checkpoint_path, expected_signature=run_signature)
        completed_combos = _get_completed_combos(checkpoint_df)

    if completed_combos:
        print(f"[Student] Found {len(completed_combos)} completed combo+model pairs")

    # Filter remaining combinations
    remaining_combos = []
    debug_count = 0
    for combo in combos:
        # Combo formatını normalize et (backtest_engine._combo_label ile aynı formatta)
        combo_str = _combo_to_str(combo)

        # Her model için kontrol et - model adlarını normalize et
        models_needed = [m for m in model_list if (combo_str, m.upper()) not in completed_combos]

        if models_needed:
            remaining_combos.append(combo)
            # Debug için ilk 3 örneği göster
            if debug_count < 3:
                print(f"[Debug] Combo '{combo_str}' needs models: {models_needed}")
                # Hangi modellerin checkpoint'te olduğunu da göster
                models_completed = [m for m in model_list if (combo_str, m.upper()) in completed_combos]
                print(f"[Debug]   Already has: {models_completed}")
                debug_count += 1
        elif debug_count < 3:
            # İlk birkaç tamamlanmış combo'yu da göster
            print(f"[Debug] Combo '{combo_str}' already completed for all models")
            debug_count += 1

    print(f"[Student] Remaining combinations to process: {len(remaining_combos)}/{len(combos)}")

    if not remaining_combos and not checkpoint_df.empty:
        print(f"[Student] All combinations already completed (using checkpoint)")
        student_results = checkpoint_df
    else:
        # Process in batches with checkpoints
        all_student_results = [checkpoint_df] if not checkpoint_df.empty else []

        for batch_start in range(0, len(remaining_combos), checkpoint_batch_size):
            batch_end = min(batch_start + checkpoint_batch_size, len(remaining_combos))
            batch_combos = remaining_combos[batch_start:batch_end]

            print(f"\n[Student] Processing batch {batch_start//checkpoint_batch_size + 1}: "
                  f"combos {batch_start+1}-{batch_end}/{len(remaining_combos)}")

            batch_results = backtest_engine.run_backtest_parallel(
                freq=freq,
                version=version_flag,
                model_list=model_list,
                combo_iterable=batch_combos,
                n_jobs=-1,
            )

            if not batch_results.empty:
                print(f"[Debug] Batch produced {len(batch_results)} rows, "
                      f"{batch_results['combo'].nunique()} unique combos, "
                      f"{batch_results['model'].nunique()} unique models")

                all_student_results.append(batch_results)

                if not disable_checkpoint:
                    # Save checkpoint
                    combined_df = pd.concat(all_student_results, ignore_index=True)
                    _save_checkpoint(combined_df, checkpoint_path, signature=run_signature)
            else:
                print(f"[Warning] Batch {batch_start//checkpoint_batch_size + 1} produced no results")

        # Combine all results
        if all_student_results:
            student_results = pd.concat(all_student_results, ignore_index=True)
        else:
            student_results = pd.DataFrame()

    if student_results.empty:
        print("[Student] No results generated")
        return

    # Handle final student results efficiently
    student_path = pipeline_results_dir / f"student_{freq.lower()}.parquet"

    # If using only checkpoint data, skip expensive save operation
    if len(remaining_combos) == 0 and not checkpoint_df.empty and not disable_checkpoint:
        print(f"[Student] All data from checkpoint ({len(student_results)} rows)")
        print(f"[Student] Skipping save to avoid memory issues - using checkpoint data for analysis")
        # Create a note file about the location
        note_path = student_path.with_suffix('.txt')
        with open(note_path, 'w') as f:
            f.write(f"Student results are in checkpoint file: {checkpoint_path}\n")
            f.write(f"Total rows: {len(student_results)}\n")
            f.write(f"Generated at: {pd.Timestamp.now()}\n")
        print(f"[Student] ✓ Created reference note: {note_path}")
    else:
        # Save new combined results with memory-efficient approach
        print(f"[Student] Saving {len(student_results)} results to {student_path}")

        try:
            # Try chunked save for large DataFrames
            if len(student_results) > 100000:
                print(f"[Student] Large dataset detected ({len(student_results)} rows), using chunked save...")
                _save_large_parquet(student_results, student_path)
            else:
                student_results.to_parquet(student_path)
            print(f"[Student] ✓ Saved results to {student_path}")
        except Exception as e:
            print(f"[Student] Error saving parquet: {e}")
            # Fallback to CSV
            csv_path = student_path.with_suffix('.csv')
            print(f"[Student] Falling back to CSV: {csv_path}")
            student_results.to_csv(csv_path, index=False)
            print(f"[Student] ✓ Saved results to CSV: {csv_path}")

    # STEP 3: Generate Student ranking
    print("\n" + "="*80)
    print("GENERATING STUDENT RANKING")
    print("="*80)

    student_grouped = student_results.groupby(['combo', 'model'])['net_return'].agg(['mean', 'std', 'count'])
    # Calculate annualized Sharpe ratio
    daily_sharpe = student_grouped['mean'] / (student_grouped['std'] + 1e-10)
    student_grouped['sharpe'] = daily_sharpe * np.sqrt(365)
    student_grouped['annualized_return'] = student_grouped['mean'] * 365
    student_grouped['volatility'] = student_grouped['std'] * np.sqrt(365)

    student_ranking = student_grouped.sort_values('sharpe', ascending=False).reset_index()

    # Save student ranking
    student_ranking_path = pipeline_results_dir / f"student_ranking_{freq.lower()}.csv"
    student_ranking.to_csv(student_ranking_path, index=False)
    print(f"[Student] Saved ranking to {student_ranking_path}")

    # Display top 10
    print(f"\nTOP 10 STUDENT PORTFOLIOS ({freq}):")
    print("="*100)
    print(f"{'Rank':<6}{'Combo':<35}{'Model':<10}{'Sharpe':<10}{'Return':<12}{'Vol':<10}")
    print("-"*100)
    for idx, row in student_ranking.head(10).iterrows():
        print(f"{idx+1:<6}{row['combo'][:34]:<35}{row['model']:<10}{row['sharpe']:<10.4f}{row['annualized_return']:<12.2%}{row['volatility']:<10.2%}")
    print("="*100)

    # Student winner
    student_winner = student_ranking.iloc[0]
    print(f"\nSTUDENT WINNER ({freq}):")
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

    # Archive ranking + winner with the same attempt suffix
    _archive_file(student_ranking_path, attempts_dir, attempt_suffix)
    _archive_file(winner_path, attempts_dir, attempt_suffix)

    # Optional holdout (OOS) evaluation on the tail of the sample
    if oos_split and 0 < float(oos_split) < 1:
        split_ratio = float(oos_split)
        student_results = student_results.copy()
        student_results["timestamp"] = pd.to_datetime(student_results["timestamp"])
        teacher_results_local = teacher_results.copy()
        teacher_results_local["timestamp"] = pd.to_datetime(teacher_results_local["timestamp"])

        unique_ts = pd.Index(student_results["timestamp"]).unique().sort_values()
        cutoff_idx = max(1, int(len(unique_ts) * (1 - split_ratio)))
        cutoff_ts = unique_ts[cutoff_idx - 1]

        oos_student = student_results[student_results["timestamp"] > cutoff_ts]
        oos_teacher = teacher_results_local[teacher_results_local["timestamp"] > cutoff_ts]

        if not oos_student.empty:
            oos_grouped = oos_student.groupby(['combo', 'model'])['net_return'].agg(['mean', 'std', 'count'])
            daily_sharpe = oos_grouped['mean'] / (oos_grouped['std'] + 1e-10)
            oos_grouped['sharpe'] = daily_sharpe * np.sqrt(365)
            oos_grouped['annualized_return'] = oos_grouped['mean'] * 365
            oos_grouped['volatility'] = oos_grouped['std'] * np.sqrt(365)
            oos_ranking = oos_grouped.sort_values('sharpe', ascending=False).reset_index()

            oos_ranking_path = pipeline_results_dir / f"student_ranking_oos_{freq.lower()}.csv"
            oos_ranking.to_csv(oos_ranking_path, index=False)
            print(f"[Student] Saved OOS ranking to {oos_ranking_path}")

            oos_winner = oos_ranking.iloc[0]
            oos_winner_data = {
                'freq': freq,
                'combo': oos_winner['combo'],
                'model': oos_winner['model'],
                'sharpe': float(oos_winner['sharpe']),
                'annualized_return': float(oos_winner['annualized_return']),
                'volatility': float(oos_winner['volatility']),
                'version': 'student',
                'cutoff': str(cutoff_ts),
                'oos_split': split_ratio,
            }
            oos_winner_path = pipeline_results_dir / f"winner_student_oos_{freq.lower()}.json"
            with open(oos_winner_path, 'w') as f:
                json.dump(oos_winner_data, f, indent=2)
            print(f"[Student] Saved OOS winner to {oos_winner_path}")

            teacher_grouped = oos_teacher.groupby(['combo', 'model'])['net_return'].agg(['mean', 'std', 'count'])
            daily_sharpe = teacher_grouped['mean'] / (teacher_grouped['std'] + 1e-10)
            teacher_grouped['sharpe'] = daily_sharpe * np.sqrt(365)
            teacher_grouped['annualized_return'] = teacher_grouped['mean'] * 365
            teacher_grouped['volatility'] = teacher_grouped['std'] * np.sqrt(365)
            teacher_ranking = teacher_grouped.sort_values('sharpe', ascending=False).reset_index()
            teacher_oos_winner = teacher_ranking.iloc[0]

            comparison_oos = {
                'freq': freq,
                'cutoff': str(cutoff_ts),
                'oos_split': split_ratio,
                'teacher': {
                    'combo': teacher_oos_winner['combo'],
                    'model': teacher_oos_winner['model'],
                    'sharpe': float(teacher_oos_winner['sharpe']),
                    'annualized_return': float(teacher_oos_winner['annualized_return']),
                    'volatility': float(teacher_oos_winner['volatility']),
                },
                'student': {
                    'combo': oos_winner['combo'],
                    'model': oos_winner['model'],
                    'sharpe': float(oos_winner['sharpe']),
                    'annualized_return': float(oos_winner['annualized_return']),
                    'volatility': float(oos_winner['volatility']),
                },
            }
            comparison_oos_path = pipeline_results_dir / f"teacher_vs_student_oos_{freq.lower()}.json"
            with open(comparison_oos_path, 'w') as f:
                json.dump(comparison_oos, f, indent=2)
            print(f"[Student] Saved OOS comparison to {comparison_oos_path}")

    # STEP 4: Compare Teacher vs Student
    print("\n" + "="*80)
    print("TEACHER vs STUDENT COMPARISON")
    print("="*80)

    # Load teacher winner
    teacher_winner_path = pipeline_results_dir / f"winner_teacher_{freq.lower()}.json"
    with open(teacher_winner_path, 'r') as f:
        teacher_winner_data = json.load(f)

    print(f"\n{'Metric':<25}{'Teacher':<20}{'Student':<20}{'Winner':<15}")
    print("-"*80)
    print(f"{'Sharpe Ratio':<25}{teacher_winner_data['sharpe']:<20.4f}{student_winner['sharpe']:<20.4f}{' ' + ('Teacher' if teacher_winner_data['sharpe'] > student_winner['sharpe'] else 'Student'):<15}")
    print(f"{'Annual Return':<25}{teacher_winner_data['annualized_return']:<20.2%}{student_winner['annualized_return']:<20.2%}{' ' + ('Teacher' if teacher_winner_data['annualized_return'] > student_winner['annualized_return'] else 'Student'):<15}")
    print(f"{'Volatility':<25}{teacher_winner_data['volatility']:<20.2%}{student_winner['volatility']:<20.2%}{' ' + ('Teacher' if teacher_winner_data['volatility'] < student_winner['volatility'] else 'Student'):<15}")
    print("="*80)

    # Save comparison
    comparison_data = {
        'freq': freq,
        'teacher': teacher_winner_data,
        'student': student_winner_data,
        'winner': 'teacher' if teacher_winner_data['sharpe'] > student_winner['sharpe'] else 'student'
    }
    comparison_path = pipeline_results_dir / f"teacher_vs_student_{freq.lower()}.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    print(f"\nSaved comparison to {comparison_path}")

    print("\n✅ Student training and evaluation complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run only the Student (ML weight learning) using existing Teacher results."
    )
    parser.add_argument(
        "--freqs",
        nargs="*",
        default=["1D"],
        help="Frequencies to process (e.g. 1D 1H). Default: 1D.",
    )
    parser.add_argument(
        "--model",
        choices=["xgb", "lgb", "cat", "ensemble"],
        default="ensemble",
        help="Student model choice. Default: ensemble.",
    )
    parser.add_argument(
        "--combo-limit",
        type=int,
        default=None,
        help="Limit the number of combinations for a fast test.",
    )
    parser.add_argument(
        "--combo-sizes",
        nargs="*",
        type=int,
        default=None,
        help="Subset combination sizes (e.g. 2 3 5). Defaults to all sizes.",
    )
    parser.add_argument(
        "--top-k-teachers",
        type=int,
        default=1,
        help="Number of top teacher portfolios to use for training (by Sharpe).",
    )
    parser.add_argument(
        "--same-asset-count",
        action="store_true",
        help="Restrict top teachers to the same asset count as the best portfolio.",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpoint loading/saving for this run.",
    )
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="Enable checkpoint loading/saving for this run.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Backtest model list (e.g. MVSK or MV MVSK). Default: all.",
    )
    parser.add_argument(
        "--n-lags",
        type=int,
        default=15,
        help="Number of lag windows for feature generation.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Stddev of Gaussian noise added to teacher weights for augmentation.",
    )
    parser.add_argument(
        "--noise-samples",
        type=int,
        default=0,
        help="Number of noisy copies of the training targets to add.",
    )
    parser.add_argument(
        "--xgb-multi-output",
        action="store_true",
        help="Use a single multi-output XGBoost model instead of per-asset models.",
    )
    parser.add_argument(
        "--softmax-temp",
        type=float,
        default=1.0,
        help="Softmax temperature for predicted weights (higher = more uniform).",
    )
    parser.add_argument(
        "--limit-ml-combos",
        action="store_true",
        help="Restrict student backtest to combos with ML predictions.",
    )
    parser.add_argument(
        "--ml-onfly",
        action="store_true",
        help="Use on-the-fly ML inference during backtest (no precomputed weights).",
    )
    parser.add_argument(
        "--oos-split",
        type=float,
        default=0.0,
        help="Holdout split ratio for OOS evaluation (e.g. 0.3 uses last 30%).",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Enable walk-forward evaluation over rebalance windows.",
    )
    parser.add_argument(
        "--wf-train-windows",
        type=int,
        default=None,
        help="Walk-forward train window count (rebalance steps).",
    )
    parser.add_argument(
        "--wf-test-windows",
        type=int,
        default=None,
        help="Walk-forward test window count (rebalance steps).",
    )
    parser.add_argument(
        "--wf-max-folds",
        type=int,
        default=None,
        help="Limit number of walk-forward folds (use last N).",
    )
    args = parser.parse_args()

    for freq in args.freqs:
        print("\n" + "="*80)
        print(f"RUNNING STUDENT FOR {freq}")
        print("="*80)

        run_student_only(
            freq=freq,
            checkpoint_batch_size=500,
            model_choice=args.model,
            combo_limit=args.combo_limit,
            combo_sizes=args.combo_sizes,
            top_k_teachers=args.top_k_teachers,
            same_asset_count=args.same_asset_count,
            disable_checkpoint=not args.checkpoint,
            model_list=[m.upper() for m in args.models] if args.models else None,
            n_lags=args.n_lags,
            noise_std=args.noise_std,
            noise_samples=args.noise_samples,
            xgb_multi_output=args.xgb_multi_output,
            softmax_temp=args.softmax_temp,
            limit_to_predicted_combos=args.limit_ml_combos,
            ml_onfly=args.ml_onfly,
            oos_split=args.oos_split,
            walk_forward=args.walk_forward,
            wf_train_windows=args.wf_train_windows,
            wf_test_windows=args.wf_test_windows,
            wf_max_folds=args.wf_max_folds,
        )
