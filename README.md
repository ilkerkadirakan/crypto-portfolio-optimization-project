# crypto_portfolio_moments

## 1. Overview

`crypto_portfolio_moments` provides an end-to-end, ML-enhanced moment-based portfolio optimization framework tailored to a 20-asset cryptocurrency universe. The platform extends the classical Markowitz mean-variance model to incorporate third and fourth moments (skewness, kurtosis) and Conditional Value-at-Risk (CVaR). Both daily (1D) and hourly (1H) cadences share a 180-day lookback window (4320 hours), ensuring alignment across data preparation, modeling, and backtesting workflows.

## 2. Features

- Dual operating modes: Baseline (realized historical moments) and ML-Enhanced (LightGBM and RandomForest forecasts via walk-forward validation).
- Multi-objective optimizers covering MV, MVSK, and MCVaRSK formulations with configurable risk aversion weights and long-only constraints.
- Unified 180-day rolling window for feature engineering across 1D and 1H series.
- Rolling backtesting engine with weekly (1D) and 24-bar (1H) rebalances, transaction cost modeling, and turnover tracking.
- Analytics suite producing parquet, CSV, and PNG artifacts for quantitative diagnostics and presentation-ready reporting.
- Centralized configuration through `configs/params.yaml` for reproducible experiments and parameter management.

## 3. Project Structure

```
crypto_portfolio_moments/
|-- agents.md
|-- README.md
|-- requirements.txt
|-- configs/
|   |-- params.yaml
|   `-- assets.yaml
|-- data/
|   |-- raw/
|   `-- processed/
|-- results/
|   |-- pipeline/
|   |-- runs/
|   |-- tables/
|   `-- figs/
`-- src/
    |-- data_prep.py
    |-- moment_calc.py
    |-- ml_forecast.py
    |-- optim_models.py
    |-- backtest_engine.py
    |-- metrics.py
    |-- reporting.py
    `-- utils.py
```

## 4. Methods

- **Data Preparation:** Raw `.mat` price files are merged, synchronized, and resampled to 1H and 1D frequencies, yielding log returns backed by 180 observations.
- **Moment Estimation:** Rolling computations supply mean, Ledoit-Wolf variance, skewness, kurtosis, and empirical CVaR (alpha = 0.95) for every asset-frequency pair.
- **ML Forecasting:** LightGBM and RandomForest regressors train with walk-forward splits to predict next-period mean, variance, and CVaR moments.
- **Optimization Models:** cvxpy solvers implement MV, MVSK, and MCVaRSK objectives with configurable lambda weights, long-only bounds, and optional weight caps.
- **Backtesting:** Weekly (1D) and 24-bar (1H) rebalances deploy both baseline and ML-enhanced moment inputs, recording turnover and transaction costs.
- **Reporting & Analytics:** Metric summaries, efficient frontiers, rolling Sharpe ratios, and drawdown analyses enable both academic evaluation and practitioner decision support.

## 5. How to Run

1. Install dependencies (see Section 7).
2. Place provider `.mat` files inside `data/raw/`.
3. Review and adjust `configs/params.yaml` for data locations, model hyperparameters, and solver settings.
4. Execute the full pipeline:

```bash
python main.py
```

Run individual modules for incremental validation:

```bash
python -m src.data_prep
python -m src.moment_calc
python -m src.ml_forecast
```

Launch targeted backtests via CLI overrides:

```bash
python main.py --frequencies 1D 1H --versions baseline ml --models MV MVSK MCVaRSK
```

## 6. Expected Outputs

- **Processed Data:** `data/processed/returns_1h.parquet`, `returns_1d.parquet`, and frequency-specific moment and forecast parquet files.
- **Backtest Runs:** Time-series performance records per frequency/model/version stored in `results/runs/`.
- **Pipeline Summaries:** Combined backtest tables and metrics archived within `results/pipeline/`.
- **Analytics:** CSV summaries (for example, `summary_all.csv`, `top20_by_sharpe.csv`) and PNG figures (efficient frontier, rolling Sharpe, drawdown) emitted to `results/tables/` and `results/figs/`.

## 7. System Requirements & Dependencies

- **Python:** Version 3.10 or newer is recommended for cvxpy and LightGBM compatibility.
- **Dependencies:** Install with `pip install -r requirements.txt`. Core libraries include numpy, pandas, scipy, cvxpy, scikit-learn, lightgbm, matplotlib, seaborn, joblib, tqdm, and PyYAML.
- **Hardware:** Multi-core CPU with at least 16 GB RAM is recommended for hourly ML training and optimization workloads.

## 8. Authors & License

- **Authors:** Kadir & Batuhan (project integrators) with LLM agent collaboration (Codex, ChatGPT).
- **License:** Refer to the repository documentation or contact the maintainers for licensing terms.
