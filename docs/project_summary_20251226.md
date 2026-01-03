# Project Summary Report (2025-12-26)

## Purpose
Develop a crypto portfolio strategy using MVSK-based optimization and supervised ML weight learning (teacher-student). Goal: improve Sharpe vs classical optimizer while keeping risk under control.

## Data and Preprocessing
- Source: 36 months of high-frequency crypto prices (20 assets).
- Resampled to 1D and 1H; main runs focus on 1D.
- Log returns computed.
- Rolling window (daily): 180 days.
- Rebalance: weekly (W-MON).

## Feature Engineering
- Rolling moments per asset: mean, variance (Ledoit-Wolf covariance), skewness, kurtosis, CVaR.
- Lagged features: n_lags = 5 for current main run (moments + returns).

## Teacher (Classical Optimization)
- Models: MV, MVSK, MCVaRSK (cvxpy).
- Full combination coverage: 2/3/5-asset combos (16,834 total).
- Ranking by annualized Sharpe (365).
- Teacher winner (1D, MVSK):
  - Combo: AVAXBTC_ETHBTC_LTCBTC_SOLBTC_STXBTC
  - Sharpe: 1.1351
  - Annual Return: 61.29%
  - Volatility: 53.99%

## Student (ML Weight Learning)
- Model: XGBoost multi-output (single model predicts all asset weights).
- Target: teacher optimal weights.
- Combo-conditional features: combo_has_<asset> indicators so weights vary by combo.
- Softmax normalization applied on predictions.

## Key Experiments and Outcomes
1) Single XGB (per-asset)
- Repeatedly plateaued around Sharpe ~0.6917.

2) Multi-output XGB (non-combo-conditional)
- In-sample Sharpe inflated (1.6+) but same weights across combos (invalid diversity). Fixed later.

3) CatBoost
- Top-1 teacher caused many assets to be constant; training failed on many assets.
- Top-300 teacher full run still returned ~0.6917.

4) Combo-conditional XGB (full run, OOS split 0.3)
- In-sample student winner:
  - Combo: ETCBTC_ICPBTC_LTCBTC_SOLBTC_STXBTC
  - Sharpe: 1.1532
  - Annual Return: 65.49%
  - Volatility: 56.79%
- OOS (last 30%) comparison:
  - Student: AAVEBTC_DOGEBTC_XRPBTC (MVSK)
    - Sharpe: 2.1237
    - Annual Return: 106.49%
    - Volatility: 50.14%
  - Teacher: AAVEBTC_XRPBTC (MCVARSK)
    - Sharpe: 2.1189
    - Annual Return: 137.30%
    - Volatility: 64.80%

5) Combo-conditional XGB (full run, OOS split 0.25)
- OOS (last 25%) comparison:
  - Student: AAVEBTC_DOGEBTC_XRPBTC (MVSK)
    - Sharpe: 3.0994
    - Annual Return: 156.69%
    - Volatility: 50.56%
    - Note: OOS döneminde ağırlıklar 1/3-1/3-1/3 sabit kaldı (rebalance boyunca değişmedi).
  - Teacher: AAVEBTC_XRPBTC (MCVARSK)
    - Sharpe: 2.9262
    - Annual Return: 195.12%
    - Volatility: 66.68%
    - Note: OOS döneminde ağırlıklar ~0.50/0.50 sabit kaldı (çok düşük varyans).

6) ML-only combos (limit-ml-combos, top-300 teacher train set)
- ML tahmini olan combo sayısı: 212 (tüm 16,834 yerine).
- In-sample student winner:
  - Combo: BCHBTC_DOGEBTC_SOLBTC_STXBTC_XRPBTC
  - Sharpe: 0.6441
  - Annual Return: 26.92%
  - Volatility: 41.79%
- OOS (%25) student winner:
  - Combo: AAVEBTC_BNBBTC_LTCBTC_SOLBTC_XRPBTC
  - Sharpe: 1.7069
  - Annual Return: 70.20%
  - Volatility: 41.13%
- Not: Önceki yüksek OOS skorların bir kısmı ML olmayan fallback optimizasyondan geliyordu. Bu koşu tamamen ML tahmini olan combo’larla sınırlıdır.

7) Softmax temp=0.7 (full run)
- In-sample dropped to Sharpe 0.7922.
- OOS winner unchanged; OOS Sharpe still ~2.1237.

## OOS Stability Checks (quick)
- OOS split 0.2: top Sharpe ~3.6124
- OOS split 0.4: top Sharpe ~1.4295
- Indicates OOS sensitivity to split; stability is not fully established.

## Current Best Interpretation
- Combo-conditional XGB is the most valid setup (weights vary per combo).
- In-sample improvement is modest (Student ~1.1532 vs Teacher 1.1351).
- OOS results are strong but sensitive to split; should be reported with caution.

## Artifacts (latest)
- Student results: results/pipeline/student_1d.parquet
- Student ranking: results/pipeline/student_ranking_1d.csv
- Student winner: results/pipeline/winner_student_1d.json
- OOS student ranking: results/pipeline/student_ranking_oos_1d.csv
- OOS student winner: results/pipeline/winner_student_oos_1d.json
- OOS comparison: results/pipeline/teacher_vs_student_oos_1d.json
- OOS 0.30 backup: results/runs/oos_0_30_backup

## Limitations
- OOS stability not fully validated (split sensitivity high).
- OOS %25 döneminde Student ve Teacher ağırlıkları düşük varyans gösterdi; sinyal zayıflığı veya dönemsel simetri olasılığı var.
- PGP not implemented (mentioned in proposal but not required for current runs).
- Walk-forward validation not completed due to runtime.

## Suggested Next Step
- Limited walk-forward (last 50-100 rebalances) to verify OOS stability.
