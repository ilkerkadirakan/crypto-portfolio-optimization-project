"""
Performance metric calculations for the optimized cryptocurrency portfolios.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import yaml

CVaR_ALPHA = 0.95
ANNUALIZATION_FACTOR = 252


def _load_params() -> Dict[str, dict]:
    """
    Load global configuration parameters to source CVaR alpha if present.
    """
    config_path = Path(__file__).resolve().parents[1] / "configs" / "params.yaml"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_alpha(config: Dict[str, dict]) -> float:
    """
    Resolve the CVaR alpha value from the configuration, defaulting to 0.95.
    """
    portfolio_cfg = config.get("portfolio", {})
    alpha = portfolio_cfg.get("cvar_alpha", CVaR_ALPHA)
    try:
        return max(0.0, min(1.0, float(alpha)))
    except (TypeError, ValueError):
        return CVaR_ALPHA


def _annualize_sharpe(mean_return: float, std_return: float) -> float:
    """
    Annualize the Sharpe ratio given mean and standard deviation of returns.
    """
    if std_return == 0:
        return np.nan
    daily_sharpe = mean_return / std_return
    return daily_sharpe * np.sqrt(ANNUALIZATION_FACTOR)


def _sortino_ratio(returns: pd.Series) -> float:
    """
    Compute the Sortino ratio using downside deviation.
    """
    mean_return = returns.mean()
    negative_returns = returns[returns < 0]
    if negative_returns.empty:
        return np.inf
    downside_std = negative_returns.std(ddof=1)
    if downside_std == 0:
        return np.inf
    daily_sortino = mean_return / downside_std
    return daily_sortino * np.sqrt(ANNUALIZATION_FACTOR)


def _max_drawdown(returns: pd.Series) -> float:
    """
    Calculate the maximum drawdown of a return series.
    """
    cumulative = (1.0 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdowns = 1.0 - cumulative / running_max
    return drawdowns.max()


def _empirical_cvar(returns: pd.Series, alpha: float) -> float:
    """
    Compute empirical CVaR by averaging the worst (1-alpha) fraction of returns.
    """
    if returns.empty:
        return np.nan
    sorted_losses = returns.sort_values()
    cutoff = int(np.ceil((1.0 - alpha) * len(sorted_losses)))
    cutoff = max(cutoff, 1)
    tail = sorted_losses.iloc[:cutoff]
    return tail.mean()


def compute_metrics(returns: pd.Series, costs: Optional[pd.Series] = None) -> Dict[str, float]:
    """
    Compute core performance metrics from a time series of portfolio returns.

    Parameters
    ----------
    returns : pd.Series
        Portfolio return series (daily or aggregated) indexed by datetime.
    costs : Optional[pd.Series], optional
        Transaction cost series aligned with the returns; if provided, net returns
        are computed as returns minus costs.

    Returns
    -------
    Dict[str, float]
        Dictionary of performance metrics including Sharpe, Sortino, drawdown, CVaR,
        turnover, and cost-adjusted Sharpe (when costs provided).
    """
    config = _load_params()
    alpha = _resolve_alpha(config)

    returns = returns.dropna()
    if returns.empty:
        return {
            "sharpe": np.nan,
            "sortino": np.nan,
            "max_drawdown": np.nan,
            "cvar": np.nan,
            "turnover": np.nan,
            "cost_adjusted_sharpe": np.nan,
        }

    net_returns = returns.copy()
    turnover = np.nan
    if costs is not None:
        aligned_costs = costs.reindex(returns.index).fillna(0.0)
        net_returns = returns - aligned_costs
        turnover = aligned_costs.sum()

    mean_return = returns.mean()
    std_return = returns.std(ddof=1)
    sharpe = _annualize_sharpe(mean_return, std_return)
    sortino = _sortino_ratio(returns)
    max_dd = _max_drawdown(net_returns)
    cvar = _empirical_cvar(net_returns, alpha)

    cost_sharpe = _annualize_sharpe(net_returns.mean(), net_returns.std(ddof=1))

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "cvar": cvar,
        "turnover": turnover,
        "cost_adjusted_sharpe": cost_sharpe,
    }


def _extract_runs(df_runs: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Split run-level results into grouped DataFrames for metric calculation.
    """
    required_cols = {"timestamp", "model", "combo", "net_return"}
    missing = required_cols - set(df_runs.columns)
    if missing:
        raise ValueError(f"df_runs missing required columns: {missing}")
    df_runs = df_runs.copy()
    df_runs["timestamp"] = pd.to_datetime(df_runs["timestamp"])
    df_runs = df_runs.sort_values("timestamp")
    groups = {}
    for (model, combo), group in df_runs.groupby(["model", "combo"]):
        groups[f"{model}::{combo}"] = group.set_index("timestamp")
    return groups


def summarize_portfolios(df_runs: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate performance metrics across all run combinations.

    Parameters
    ----------
    df_runs : pd.DataFrame
        Combined backtest output as returned by run_backtest.

    Returns
    -------
    pd.DataFrame
        Table of performance metrics indexed by (model, combo).
    """
    if df_runs.empty:
        return pd.DataFrame()

    groups = _extract_runs(df_runs)
    summary_records = []
    for key, group in groups.items():
        model, combo = key.split("::", maxsplit=1)
        returns = group["net_return"]
        costs = group.get("transaction_cost")
        metrics = compute_metrics(returns, costs)
        metrics.update({"model": model, "combo": combo})
        summary_records.append(metrics)

    summary_df = pd.DataFrame(summary_records)
    summary_df = summary_df.set_index(["model", "combo"]).sort_index()
    summary_df = summary_df[
        ["sharpe", "cost_adjusted_sharpe", "sortino", "max_drawdown", "cvar", "turnover"]
    ]
    return summary_df


__all__ = [
    "compute_metrics",
    "summarize_portfolios",
]
