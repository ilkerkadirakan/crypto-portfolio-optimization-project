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

    try:
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk_df = df.iloc[start_idx:end_idx].copy()

            # Convert chunk to Arrow table
            table = pa.Table.from_pandas(chunk_df, preserve_index=True)

            if schema is None:
                schema = table.schema
                writer = pq.ParquetWriter(file_path, schema)

            writer.write_table(table)

            if start_idx % (chunk_size * 5) == 0:  # Progress every 5 chunks
                print(f"[SaveChunk] Progress: {end_idx}/{total_rows} rows ({end_idx/total_rows:.1%})")

    finally:
        if writer:
            writer.close()

    print(f"[SaveChunk] ✓ Completed chunked save to {file_path}")


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


def run_student_only(freq="1D", checkpoint_batch_size=500):
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

    # Load combinations
    combos_by_size = combination_utils.cache_combinations()
    combos = []
    for size_key in sorted(combos_by_size.keys()):
        combos.extend(combos_by_size[size_key])

    print(f"[Student-Only] Found {len(combos)} portfolio combinations")

    # STEP 1: Train Student ML models
    print("\n" + "="*80)
    print("TRAINING STUDENT (ML Weight Learning)")
    print("="*80)

    from src import ml_weights
    print("[Student] Training ML ensemble models to learn portfolio weights...")
    ml_weights.train_weight_models(
        teacher_results=teacher_results,
        processed_dir=processed_dir,
        freq=freq,
        use_ensemble=True,  # Enable ensemble for better performance
        model_types=['lgb', 'xgb', 'rf']  # Use all available models
    )
    print("[Student] ML ensemble weight models trained successfully")

    # STEP 2: Run Student backtest
    print("\n" + "="*80)
    print("BACKTESTING STUDENT PORTFOLIOS")
    print("="*80)

    model_list = ["MV", "MVSK", "MCVaRSK"]

    print(f"[Student] Running backtest with ML weights for {len(combos)} combinations...")
    print(f"[Student] Using parallel processing with checkpoint support...")

    # CHECKPOINT SYSTEM
    checkpoint_path = _get_checkpoint_path(pipeline_results_dir, freq)
    checkpoint_df = _load_checkpoint(checkpoint_path)
    completed_combos = _get_completed_combos(checkpoint_df)

    if completed_combos:
        print(f"[Student] Found {len(completed_combos)} completed combo+model pairs")

    # Filter remaining combinations
    remaining_combos = []
    debug_count = 0
    for combo in combos:
        # Combo formatını normalize et (backtest_engine._combo_label ile aynı formatta)
        if isinstance(combo, dict):
            # Dict formatı için
            for key in ("name", "label", "id"):
                if key in combo and combo[key]:
                    combo_str = str(combo[key])
                    break
            else:
                combo_str = "_".join(f"{k}-{v}" for k, v in sorted(combo.items()))
        elif isinstance(combo, (list, tuple)):
            # List/tuple için string'e çevir
            combo_str = "_".join(str(item) for item in combo)
        else:
            combo_str = str(combo)

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
                version="ml",
                model_list=model_list,
                combo_iterable=batch_combos,
                n_jobs=-1,
            )

            if not batch_results.empty:
                print(f"[Debug] Batch produced {len(batch_results)} rows, "
                      f"{batch_results['combo'].nunique()} unique combos, "
                      f"{batch_results['model'].nunique()} unique models")

                all_student_results.append(batch_results)

                # Save checkpoint
                combined_df = pd.concat(all_student_results, ignore_index=True)
                _save_checkpoint(combined_df, checkpoint_path)
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
    if len(remaining_combos) == 0 and not checkpoint_df.empty:
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
    student_grouped['sharpe'] = daily_sharpe * np.sqrt(252)  # Annualized
    student_grouped['annualized_return'] = student_grouped['mean'] * 252
    student_grouped['volatility'] = student_grouped['std'] * np.sqrt(252)

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
    import sys

    # Support both 1D and 1H
    if len(sys.argv) > 1:
        freqs = sys.argv[1:]
    else:
        freqs = ["1D"]  # Default: process both frequencies

    for freq in freqs:
        print("\n" + "="*80)
        print(f"RUNNING STUDENT FOR {freq}")
        print("="*80)
        run_student_only(freq)
