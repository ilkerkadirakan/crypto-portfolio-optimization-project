"""
Reporting utilities for summarizing portfolio performance and diagnostics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import summarize_portfolios
import json

plt.switch_backend("Agg")


def _output_dirs() -> Tuple[Path, Path]:
    """
    Create and return paths to the tables and figures output directories.
    """
    project_root = Path(__file__).resolve().parents[1]
    tables_dir = project_root / "results" / "tables"
    figs_dir = project_root / "results" / "figs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir, figs_dir


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the timestamp column exists and is parsed as datetime.
    """
    if "timestamp" not in df.columns:
        raise ValueError("df_runs must contain a 'timestamp' column.")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp")


def _annualize_stats(returns: pd.Series, freq: str) -> Tuple[float, float]:
    """
    Compute annualized return and volatility given a return series and frequency.
    """
    returns = returns.dropna()
    if returns.empty:
        return np.nan, np.nan

    freq = freq.upper()
    if freq == "1D":
        periods_per_year = 365
        mean = (1.0 + returns).prod() ** (periods_per_year / len(returns)) - 1.0
        vol = returns.std(ddof=1) * np.sqrt(periods_per_year)
        return mean, vol

    if freq == "1H":
        periods_per_day = 24
        periods_per_year = periods_per_day * 365
        compounded = (1.0 + returns).prod() ** (periods_per_year / len(returns)) - 1.0
        vol = returns.std(ddof=1) * np.sqrt(periods_per_year)
        return compounded, vol

    raise ValueError(f"Unsupported frequency: {freq}")


def _mean_series(group: pd.DataFrame) -> pd.Series:
    """
    Aggregate net returns to a single series by averaging across combos/models.
    """
    pivot = (
        group.pivot_table(
            index="timestamp",
            values="net_return",
            columns=["model", "combo"],
            aggfunc="mean",
        )
    )
    mean_series = pivot.mean(axis=1).dropna()
    return mean_series


def _daily_series(series: pd.Series) -> pd.Series:
    """
    Convert an intraday return series to daily compounded returns.
    """
    if series.empty:
        return series
    compounded = (1.0 + series).resample("1D").prod() - 1.0
    return compounded.dropna()


def _rolling_sharpe(series: pd.Series, window: int = 60) -> pd.Series:
    """
    Compute rolling Sharpe ratio on a daily return series.
    """
    if series.empty:
        return series

    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std(ddof=1)
    sharpe = (rolling_mean / rolling_std) * np.sqrt(365)
    return sharpe.replace([np.inf, -np.inf], np.nan)


def _drawdown_series(series: pd.Series) -> pd.Series:
    """
    Calculate drawdown series from daily returns.
    """
    if series.empty:
        return series
    cumulative = (1.0 + series).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative / running_max) - 1.0
    return drawdown


def _plot_frontier(frontier_df: pd.DataFrame, figs_dir: Path) -> None:
    """
    Plot a scatter frontier comparing hourly and daily strategies.
    """
    if frontier_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"1H": "#1f77b4", "1D": "#ff7f0e"}
    for freq, subset in frontier_df.groupby("freq"):
        ax.scatter(
            subset["vol"],
            subset["return"],
            label=freq,
            alpha=0.7,
            s=60,
            color=colors.get(freq, None),
        )
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.set_title("Efficient Frontier: 1H vs 1D")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig_path = figs_dir / "frontier_1h_vs_1d.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


def _plot_rolling_sharpe(rolling_sharpes: Dict[str, pd.Series], figs_dir: Path) -> None:
    """
    Plot rolling Sharpe ratios for each frequency.
    """
    if not rolling_sharpes:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for freq, series in rolling_sharpes.items():
        if series.empty:
            continue
        ax.plot(series.index, series.values, label=freq)
    ax.set_title("60-Day Rolling Sharpe Ratio")
    ax.set_ylabel("Sharpe Ratio")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig_path = figs_dir / "rolling_sharpe.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


def _plot_drawdown(drawdowns: Dict[str, pd.Series], figs_dir: Path) -> None:
    """
    Plot drawdown curves for each frequency.
    """
    if not drawdowns:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for freq, series in drawdowns.items():
        if series.empty:
            continue
        ax.plot(series.index, series.values, label=freq)
    ax.set_title("Cumulative Drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig_path = figs_dir / "drawdown.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


def _plot_teacher_vs_student(figs_dir: Path) -> None:
    """
    Plot Teacher vs Student comparison for all frequencies.

    Reads winner JSON files and creates comparison bar charts.
    """
    project_root = Path(__file__).resolve().parents[1]
    pipeline_dir = project_root / "results" / "pipeline"

    if not pipeline_dir.exists():
        return

    # Load comparison files
    comparisons = {}
    for freq in ["1d", "1h"]:
        comp_file = pipeline_dir / f"teacher_vs_student_{freq}.json"
        if comp_file.exists():
            with open(comp_file, 'r') as f:
                comparisons[freq.upper()] = json.load(f)

    if not comparisons:
        return

    # Create subplot for each frequency
    n_freqs = len(comparisons)
    fig, axes = plt.subplots(1, n_freqs, figsize=(6 * n_freqs, 5))
    if n_freqs == 1:
        axes = [axes]

    for idx, (freq, data) in enumerate(comparisons.items()):
        ax = axes[idx]

        teacher = data.get('teacher', {})
        student = data.get('student', {})

        metrics = ['sharpe', 'annualized_return', 'volatility']
        labels = ['Sharpe', 'Annual Return', 'Volatility']

        teacher_vals = [teacher.get(m, 0) for m in metrics]
        student_vals = [student.get(m, 0) for m in metrics]

        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax.bar(x - width/2, teacher_vals, width, label='Teacher', color='#1f77b4', alpha=0.8)
        bars2 = ax.bar(x + width/2, student_vals, width, label='Student', color='#ff7f0e', alpha=0.8)

        ax.set_ylabel('Value')
        ax.set_title(f'Teacher vs Student ({freq})')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.legend()
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    fig_path = figs_dir / "teacher_vs_student_comparison.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"[Reporting] Saved Teacher vs Student comparison to {fig_path}")


def _plot_ranking_comparison(figs_dir: Path) -> None:
    """
    Plot top 10 portfolios for Teacher and Student side by side.
    """
    project_root = Path(__file__).resolve().parents[1]
    pipeline_dir = project_root / "results" / "pipeline"

    if not pipeline_dir.exists():
        return

    # Choose one frequency (1D) for detailed comparison
    teacher_ranking_file = pipeline_dir / "teacher_ranking_1d.csv"
    student_ranking_file = pipeline_dir / "student_ranking_1d.csv"

    if not teacher_ranking_file.exists() or not student_ranking_file.exists():
        return

    teacher_df = pd.read_csv(teacher_ranking_file).head(10)
    student_df = pd.read_csv(student_ranking_file).head(10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Teacher Top 10
    teacher_combos = [c[:25] + '...' if len(c) > 25 else c for c in teacher_df['combo'].values]
    teacher_sharpe = teacher_df['sharpe'].values

    ax1.barh(range(len(teacher_sharpe)), teacher_sharpe, color='#1f77b4', alpha=0.7)
    ax1.set_yticks(range(len(teacher_combos)))
    ax1.set_yticklabels(teacher_combos, fontsize=9)
    ax1.set_xlabel('Sharpe Ratio')
    ax1.set_title('TOP 10 Teacher Portfolios (1D)', fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, axis='x', linestyle='--', alpha=0.3)

    # Add value labels
    for i, v in enumerate(teacher_sharpe):
        ax1.text(v, i, f' {v:.4f}', va='center', fontsize=8)

    # Student Top 10
    student_combos = [c[:25] + '...' if len(c) > 25 else c for c in student_df['combo'].values]
    student_sharpe = student_df['sharpe'].values

    ax2.barh(range(len(student_sharpe)), student_sharpe, color='#ff7f0e', alpha=0.7)
    ax2.set_yticks(range(len(student_combos)))
    ax2.set_yticklabels(student_combos, fontsize=9)
    ax2.set_xlabel('Sharpe Ratio')
    ax2.set_title('TOP 10 Student Portfolios (1D)', fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(True, axis='x', linestyle='--', alpha=0.3)

    # Add value labels
    for i, v in enumerate(student_sharpe):
        ax2.text(v, i, f' {v:.4f}', va='center', fontsize=8)

    fig.tight_layout()
    fig_path = figs_dir / "top10_teacher_vs_student.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"[Reporting] Saved Top 10 ranking comparison to {fig_path}")


def _plot_winner_highlights(figs_dir: Path) -> None:
    """
    Create a summary visualization highlighting the final winners.
    """
    project_root = Path(__file__).resolve().parents[1]
    pipeline_dir = project_root / "results" / "pipeline"

    if not pipeline_dir.exists():
        return

    # Load all winners
    winners = {}
    for freq in ["1d", "1h"]:
        for version in ["teacher", "student"]:
            winner_file = pipeline_dir / f"winner_{version}_{freq}.json"
            if winner_file.exists():
                with open(winner_file, 'r') as f:
                    winners[f"{version}_{freq}"] = json.load(f)

    if not winners:
        return

    # Create summary figure
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle('🏆 DUAL-WINNER PORTFOLIO OPTIMIZATION RESULTS',
                 fontsize=16, fontweight='bold', y=0.98)

    colors = {'teacher': '#1f77b4', 'student': '#ff7f0e'}

    for idx, freq in enumerate(['1d', '1h']):
        freq_upper = freq.upper()

        # Create subplot
        ax = fig.add_subplot(gs[idx, :])
        ax.axis('off')

        # Header
        ax.text(0.5, 0.95, f'Frequency: {freq_upper}',
               ha='center', fontsize=14, fontweight='bold',
               transform=ax.transAxes)

        y_pos = 0.80
        for version in ['teacher', 'student']:
            key = f"{version}_{freq}"
            if key not in winners:
                continue

            data = winners[key]

            # Version title
            ax.text(0.5, y_pos, f'👑 {version.upper()} WINNER',
                   ha='center', fontsize=12, fontweight='bold',
                   color=colors[version], transform=ax.transAxes)

            y_pos -= 0.10

            # Details
            details = [
                f"Combo: {data.get('combo', 'N/A')}",
                f"Model: {data.get('model', 'N/A')}",
                f"Sharpe: {data.get('sharpe', 0):.4f}",
                f"Annual Return: {data.get('annualized_return', 0):.2%}",
                f"Volatility: {data.get('volatility', 0):.2%}"
            ]

            for detail in details:
                ax.text(0.5, y_pos, detail, ha='center', fontsize=10,
                       family='monospace', transform=ax.transAxes)
                y_pos -= 0.06

            y_pos -= 0.05

        # Add separator
        if idx == 0:
            ax.axhline(y=0.05, color='gray', linestyle='--',
                      linewidth=1, transform=ax.transAxes)

    fig_path = figs_dir / "winner_summary.png"
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[Reporting] Saved winner summary to {fig_path}")


def generate_all_reports(df_runs: pd.DataFrame) -> None:
    """
    Create summary tables and figures from backtest run outputs.

    Includes both traditional metrics and Dual-Winner visualizations.
    """
    tables_dir, figs_dir = _output_dirs()

    if df_runs.empty:
        summary_path = tables_dir / "summary_all.csv"
        top_path = tables_dir / "top20_by_sharpe.csv"
        pd.DataFrame().to_csv(summary_path, index=True)
        pd.DataFrame().to_csv(top_path, index=True)
        return

    df_runs = _ensure_datetime(df_runs)

    summary_df = summarize_portfolios(df_runs)
    summary_path = tables_dir / "summary_all.csv"
    summary_df.to_csv(summary_path)

    top20 = summary_df.sort_values("sharpe", ascending=False).head(20)
    top_path = tables_dir / "top20_by_sharpe.csv"
    top20.to_csv(top_path)

    frontier_records = []
    rolling_sharpes: Dict[str, pd.Series] = {}
    drawdown_series_map: Dict[str, pd.Series] = {}

    for freq, freq_group in df_runs.groupby("freq"):
        freq_upper = str(freq).upper()
        mean_series = _mean_series(freq_group)
        if freq_upper == "1H":
            daily_series = _daily_series(mean_series)
        else:
            daily_series = mean_series

        if not daily_series.empty:
            rolling_sharpes[freq_upper] = _rolling_sharpe(daily_series)
            drawdown_series_map[freq_upper] = _drawdown_series(daily_series)

        for (model, combo), group in freq_group.groupby(["model", "combo"]):
            returns = group.set_index("timestamp")["net_return"].dropna()
            ann_return, ann_vol = _annualize_stats(returns, freq_upper)
            frontier_records.append(
                {
                    "freq": freq_upper,
                    "model": model,
                    "combo": combo,
                    "return": ann_return,
                    "vol": ann_vol,
                }
            )

    frontier_df = pd.DataFrame(frontier_records)

    # Traditional plots
    _plot_frontier(frontier_df, figs_dir)
    _plot_rolling_sharpe(rolling_sharpes, figs_dir)
    _plot_drawdown(drawdown_series_map, figs_dir)

    # Dual-Winner visualizations
    print("\n[Reporting] Generating Dual-Winner visualizations...")
    _plot_teacher_vs_student(figs_dir)
    _plot_ranking_comparison(figs_dir)
    _plot_winner_highlights(figs_dir)
    print("[Reporting] All reports generated successfully!")


__all__ = [
    "generate_all_reports",
]


if __name__ == "__main__":
    # Test dual-winner visualizations
    project_root = Path(__file__).resolve().parents[1]
    figs_dir = project_root / "results" / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    _plot_teacher_vs_student(figs_dir)
    _plot_ranking_comparison(figs_dir)
    _plot_winner_highlights(figs_dir)
    print("Test visualizations generated!")
