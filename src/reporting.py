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
        periods_per_year = 252
        mean = (1.0 + returns).prod() ** (periods_per_year / len(returns)) - 1.0
        vol = returns.std(ddof=1) * np.sqrt(periods_per_year)
        return mean, vol

    if freq == "1H":
        periods_per_day = 24
        periods_per_year = periods_per_day * 252
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
    sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
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


def generate_all_reports(df_runs: pd.DataFrame) -> None:
    """
    Create summary tables and figures from backtest run outputs.
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
    _plot_frontier(frontier_df, figs_dir)
    _plot_rolling_sharpe(rolling_sharpes, figs_dir)
    _plot_drawdown(drawdown_series_map, figs_dir)


__all__ = [
    "generate_all_reports",
]
