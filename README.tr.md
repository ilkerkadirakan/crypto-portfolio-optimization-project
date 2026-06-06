# Kripto Portfoy Optimizasyonu Projesi

Bu repo, kripto para piyasalarinda teacher-student portfoy optimizasyonu calismasinin tekrar uretilebilir kodlarini icerir.

Calisma, klasik portfoy optimizasyon modelleri ile makine ogrenmesi tabanli bir portfoy agirlik tahmin modelini birlestirir. Klasik optimizasyon modelleri **Teacher** olarak kullanilir ve portfoy agirliklari uretir. Multi-output XGBoost modeli ise **Student** olarak egitilir ve Teacher tarafindan uretilen agirlik kararlarini yaklasik olarak ogrenmeye calisir.

## Calisma Kurulumu

- Varlik evreni: BTC bazli 20 kripto para cifti
- Portfoy evreni: 2, 3 ve 5 varlikli portfoylerden uretilen 16,834 kombinasyon
- Ana frekans: gunluk (`1D`)
- Teacher modelleri: `MV`, `MVSK`, `MCVaRSK`
- Student modeli: multi-output `XGBoost`
- Ana degerlendirme: strict chronological OOS split
- Final OOS ayrimi: %25
- Islem maliyeti: 10 bps
- Yilliklandirma: 365 gun

## Yontem Ozeti

Pipeline iki ana asamadan olusur.

1. **Teacher optimizasyonu**

   Teacher asamasinda klasik optimizasyon modelleri genis portfoy kombinasyonu evreni uzerinde calistirilir. Bu modeller long-only ve tam yatirim kisitlari altinda portfoy agirliklari uretir. Teacher sonuclari Sharpe oranina gore siralanir.

2. **Student distillation**

   Student modeli fiyat veya getiri tahmin etmek yerine secilmis Teacher ciktilarindan portfoy agirliklarini ogrenir. Final deneyde yalnizca egitim donemi bilgisiyle portfoy secimi yapilir, iki asamali aday filtreleme uygulanir ve strict OOS protokoluyle degerlendirme yapilir.

Amac Student modelinin Teacher modelini her zaman gecmesi degildir. Amac, Student modelinin Teacher tahsis davranisinin finansal olarak anlamli bir yaklasimini daha dusuk hesaplama maliyetiyle uretip uretemedigini test etmektir.

## Repo Yapisi

```text
configs/                 Deney parametreleri ve varlik evreni
data/                    Ham ve islenmis veri klasorleri
src/                     Veri hazirlama, optimizasyon, backtest, metrik ve ML kodlari
main.py                  Tam teacher-student pipeline
run_student_only.py      Mevcut Teacher ciktilariyla Student deneyleri
recalc_teacher_ranking.py
                         Teacher ranking tablolarini yeniden hesaplama
requirements.txt         Python bagimliliklari
```

Uretilen deney ciktilari Git disinda tutulur:

```text
results/
report_bundle_*/
data/processed/
```

## Kurulum

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Temel Komutlar

Tam teacher-student pipeline:

```bash
python main.py --frequencies 1D --models MV MVSK MCVaRSK --resume
```

Mevcut Teacher sonuclarindan final strict-OOS Student kosusu:

```bash
python run_student_only.py --freqs 1D --oos-split 0.25 --top-k-teachers 50 --top-combos 200 --same-asset-count --model xgb --xgb-multi-output --ml-onfly --models MVSK --two-stage --stage1-candidate-multiplier 2 --stage1-train-frac 0.80 --stage1-max-rows 120000 --xgb-learning-rate 0.05 --xgb-max-depth 4 --xgb-n-estimators 700 --xgb-subsample 0.9 --xgb-colsample-bytree 0.9 --xgb-min-child-weight 3 --xgb-reg-alpha 0.0 --xgb-reg-lambda 1.0 --no-checkpoint
```

Teacher ranking tablosunu yeniden hesaplama:

```bash
python recalc_teacher_ranking.py 1D
```

## Final Deney Parametreleri

Final raporlanan Student kosusu su parametrelerle uretilmistir:

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

## Temel Cikti Dosyalari

Tipik pipeline ciktilari `results/pipeline/` altina yazilir:

```text
teacher_1d.parquet
teacher_ranking_1d.csv
winner_teacher_1d.json
student_1d.parquet
student_ranking_1d.csv
winner_student_1d.json
teacher_vs_student_1d.json
```

Strict OOS kosulari su klasorde arsivlenir:

```text
results/pipeline/oos_runs/
```

Her OOS run klasoru metadata, winner dosyalari, ranking dosyalari, model artefaktlari ve Teacher-vs-Student karsilastirmasini icerir.

## Notlar

- Ham veri ve uretilen sonuc dosyalari commit edilmemelidir.
- Final tez/metin dosyalari bu repo disinda tutulur.
- Repo, calistirilabilir arastirma pipeline'ini saklamak icindir; her ara rapor veya kesif deneyi repo icine alinmaz.
