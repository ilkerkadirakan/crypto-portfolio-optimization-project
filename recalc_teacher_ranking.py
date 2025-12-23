"""
Recalculate Teacher ranking with CORRECT annualized Sharpe ratio.
Uses existing teacher backtest results (no need to re-run backtests).
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json

def recalculate_teacher_ranking(freq="1D"):
    """
    Recalculate teacher ranking with correct annualized Sharpe.

    Parameters
    ----------
    freq : str
        Frequency to process (1D or 1H)
    """
    project_root = Path(__file__).resolve().parent
    pipeline_results_dir = project_root / "results" / "pipeline"

    # Load existing teacher results
    teacher_path = pipeline_results_dir / f"teacher_{freq.lower()}.parquet"

    if not teacher_path.exists():
        print(f"[Error] Teacher results not found at {teacher_path}")
        return

    print(f"[Recalc] Loading teacher results from {teacher_path}")
    teacher_results = pd.read_parquet(teacher_path)
    print(f"[Recalc] Loaded {len(teacher_results)} teacher backtest rows")

    # Recalculate ranking with CORRECT annualized Sharpe
    print(f"[Recalc] Calculating CORRECT annualized Sharpe ratio...")

    teacher_grouped = teacher_results.groupby(['combo', 'model'])['net_return'].agg(['mean', 'std', 'count'])

    # CORRECT formula: annualized Sharpe = (mean / std) * sqrt(252)
    daily_sharpe = teacher_grouped['mean'] / (teacher_grouped['std'] + 1e-10)
    teacher_grouped['sharpe'] = daily_sharpe * np.sqrt(252)  # Annualized Sharpe
    teacher_grouped['annualized_return'] = teacher_grouped['mean'] * 252
    teacher_grouped['volatility'] = teacher_grouped['std'] * np.sqrt(252)

    teacher_ranking = teacher_grouped.sort_values('sharpe', ascending=False).reset_index()

    # Save corrected teacher ranking
    teacher_ranking_path = pipeline_results_dir / f"teacher_ranking_{freq.lower()}.csv"
    teacher_ranking.to_csv(teacher_ranking_path, index=False)
    print(f"[Recalc] ✅ Saved CORRECTED ranking to {teacher_ranking_path}")

    # Display top 10 teacher portfolios
    print(f"\nTOP 10 TEACHER PORTFOLIOS ({freq}) - CORRECTED:")
    print("="*100)
    print(f"{'Rank':<6}{'Combo':<35}{'Model':<10}{'Sharpe':<10}{'Return':<12}{'Vol':<10}")
    print("-"*100)
    for idx, row in teacher_ranking.head(10).iterrows():
        print(f"{idx+1:<6}{row['combo'][:34]:<35}{row['model']:<10}{row['sharpe']:<10.4f}{row['annualized_return']:<12.2%}{row['volatility']:<10.2%}")
    print("="*100)

    # Find TEACHER WINNER (corrected)
    teacher_winner = teacher_ranking.iloc[0]
    print(f"\nTEACHER WINNER ({freq}) - CORRECTED:")
    print(f"   Combo: {teacher_winner['combo']}")
    print(f"   Model: {teacher_winner['model']}")
    print(f"   Sharpe: {teacher_winner['sharpe']:.4f}")
    print(f"   Annual Return: {teacher_winner['annualized_return']:.2%}")
    print(f"   Volatility: {teacher_winner['volatility']:.2%}")

    # Save corrected teacher winner
    teacher_winner_data = {
        'freq': freq,
        'combo': teacher_winner['combo'],
        'model': teacher_winner['model'],
        'sharpe': float(teacher_winner['sharpe']),
        'annualized_return': float(teacher_winner['annualized_return']),
        'volatility': float(teacher_winner['volatility']),
        'version': 'teacher'
    }
    winner_path = pipeline_results_dir / f"winner_teacher_{freq.lower()}.json"
    with open(winner_path, 'w') as f:
        json.dump(teacher_winner_data, f, indent=2)
    print(f"[Recalc] ✅ Saved CORRECTED winner to {winner_path}")

    print(f"\n✅ Teacher ranking recalculated with CORRECT Sharpe for {freq}!")


if __name__ == "__main__":
    import sys

    # Support both 1D and 1H
    if len(sys.argv) > 1:
        freqs = sys.argv[1:]
    else:
        freqs = ["1D", "1H"]  # Default: process both frequencies

    for freq in freqs:
        print("\n" + "="*80)
        print(f"RECALCULATING TEACHER RANKING FOR {freq}")
        print("="*80)
        recalculate_teacher_ranking(freq)
