# Crypto Portfolio Optimization Project

This repository contains the reproducible code for a teacher-student portfolio optimization study in cryptocurrency markets.

The study combines classical portfolio optimization models with a machine learning allocator. Classical optimizers are used as **Teacher** models to generate portfolio weights. A multi-output XGBoost model is then trained as the **Student** model to approximate these allocation decisions under a strict out-of-sample evaluation protocol.

## Research Setting

- Asset universe: 20 BTC-quoted crypto pairs
- Portfolio universe: 16,834 combinations from 2-, 3-, and 5-asset portfolios
- Main frequency: daily (`1D`)
- Teacher models: `MV`, `MVSK`, `MCVaRSK`
- Student model: multi-output `XGBoost`
- Main evaluation: strict chronological OOS split
- Final OOS split: 25%
- Transaction cost: 10 bps
- Annualization basis: 365 days

## Method Summary

The pipeline has two main stages.

1. **Teacher optimization**

   The Teacher stage evaluates portfolio combinations using classical optimization models. These models produce long-only, fully invested portfolio weights under risk-return constraints. Teacher results are ranked by Sharpe ratio.

2. **Student distillation**

   The Student model learns portfolio weights from selected Teacher outputs instead of predicting prices or returns directly. The final student experiment uses training-only portfolio selection, a two-stage candidate filtering procedure, and strict OOS evaluation to reduce look-ahead bias.

The goal is not to guarantee that the Student always outperforms the Teacher. The goal is to test whether the Student can learn a financially meaningful approximation of the Teacher allocation behavior with lower computational cost.

## Repository Structure

```text
configs/                 Experiment parameters and asset universe
data/                    Raw and processed data directories
src/                     Data preparation, optimization, backtesting, metrics, and ML code
main.py                  Full teacher-student pipeline
run_student_only.py      Student-only experiments using existing teacher outputs
recalc_teacher_ranking.py
                         Recalculate teacher ranking tables
requirements.txt         Python dependencies
```

Generated experiment outputs are intentionally kept outside version control:

```text
results/
report_bundle_*/
data/processed/
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Main Commands

Run the full teacher-student pipeline:

```bash
python main.py --frequencies 1D --models MV MVSK MCVaRSK --resume
```

Run the final strict-OOS student configuration from existing Teacher outputs:

```bash
python run_student_only.py --freqs 1D --oos-split 0.25 --top-k-teachers 50 --top-combos 200 --same-asset-count --model xgb --xgb-multi-output --ml-onfly --models MVSK --two-stage --stage1-candidate-multiplier 2 --stage1-train-frac 0.80 --stage1-max-rows 120000 --xgb-learning-rate 0.05 --xgb-max-depth 4 --xgb-n-estimators 700 --xgb-subsample 0.9 --xgb-colsample-bytree 0.9 --xgb-min-child-weight 3 --xgb-reg-alpha 0.0 --xgb-reg-lambda 1.0 --no-checkpoint
```

Recalculate teacher rankings without rerunning the full teacher backtest:

```bash
python recalc_teacher_ranking.py 1D
```

## Final Experiment Configuration

The final reported Student configuration uses:

```text
frequency: 1D
oos_split: 0.25
top_k_teachers: 50
top_combos: 200
same_asset_count: true
two_stage: true
stage1_candidate_multiplier: 2
stage1_train_frac: 0.80
stage1_max_rows: 120000
model: xgb
xgb_multi_output: true
xgb_learning_rate: 0.05
xgb_max_depth: 4
xgb_n_estimators: 700
xgb_subsample: 0.9
xgb_colsample_bytree: 0.9
xgb_min_child_weight: 3
xgb_reg_alpha: 0.0
xgb_reg_lambda: 1.0
teacher_rank_weighting: false
teacher_sharpe_weighting: false
ml_onfly: true
```

## Key Output Files

Typical pipeline outputs are written under `results/pipeline/`:

```text
teacher_1d.parquet
teacher_ranking_1d.csv
winner_teacher_1d.json
student_1d.parquet
student_ranking_1d.csv
winner_student_1d.json
teacher_vs_student_1d.json
```

Strict OOS runs are archived under:

```text
results/pipeline/oos_runs/
```

Each archived OOS run contains metadata, winner files, rankings, model artifacts, and the corresponding Teacher-vs-Student comparison.

## Notes

- Raw data and generated outputs are not intended to be committed.
- The final article text and thesis files are kept outside this repository.
- The repository is intended to preserve the executable research pipeline, not every exploratory report or intermediate experiment artifact.
