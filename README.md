# Crypto Portfolio Optimization Project

This repository contains the reproducible code for a teacher-student portfolio optimization study in cryptocurrency markets.

The project compares classical portfolio optimizers with a student model that learns portfolio weights from teacher outputs. The main empirical setting uses 20 BTC-quoted crypto assets, rolling higher-moment features, long-only portfolio constraints, and strict out-of-sample evaluation.

## Repository Structure

```text
configs/       Experiment configuration and asset universe
data/          Raw and processed data files
src/           Data preparation, optimization, backtesting, metrics, and ML code
main.py        Full teacher-student pipeline
run_student_only.py
               Student-only experiments using existing teacher outputs
recalc_teacher_ranking.py
               Recalculate teacher ranking tables
```

Local run outputs are intentionally ignored by Git:

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

Run the full pipeline:

```bash
python main.py --frequencies 1D --models MV MVSK MCVaRSK --resume
```

Run only the student stage from existing teacher results:

```bash
python run_student_only.py --freqs 1D --oos-split 0.25 --top-k-teachers 50 --top-combos 200 --same-asset-count --model xgb --xgb-multi-output --ml-onfly --models MVSK --two-stage --stage1-candidate-multiplier 2 --stage1-train-frac 0.80 --stage1-max-rows 120000 --xgb-learning-rate 0.05 --xgb-max-depth 4 --xgb-n-estimators 700 --xgb-subsample 0.9 --xgb-colsample-bytree 0.9 --xgb-min-child-weight 3 --xgb-reg-alpha 0.0 --xgb-reg-lambda 1.0 --no-checkpoint
```

Recalculate teacher rankings:

```bash
python recalc_teacher_ranking.py 1D
```

## Notes

- Generated result bundles should stay outside version control.
- The repository keeps reproducible code and concise documentation; large experiment outputs are regenerated from the pipeline.
