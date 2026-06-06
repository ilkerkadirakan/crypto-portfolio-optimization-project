# Kripto Portfoy Optimizasyonu Projesi

Bu repo, kripto para piyasalarinda teacher-student portfoy optimizasyonu calismasinin tekrar uretilebilir kodlarini icerir.

Calisma, klasik portfoy optimizasyon modelleri ile teacher ciktilarindan portfoy agirliklari ogrenen student modeli karsilastirir. Ana deney kurulumu 20 BTC bazli kripto varlik, rolling yuksek moment ozellikleri, long-only portfoy kisitlari ve strict out-of-sample degerlendirme uzerine kuruludur.

## Repo Yapisi

```text
configs/       Deney ayarlari ve varlik evreni
data/          Ham ve islenmis veri dosyalari
src/           Veri hazirlama, optimizasyon, backtest, metrik ve ML kodlari
main.py        Tam teacher-student pipeline
run_student_only.py
               Mevcut teacher ciktilariyla student deneyleri
recalc_teacher_ranking.py
               Teacher ranking tablolarini yeniden hesaplama
```

Lokal calisma ciktilari Git disinda tutulur:

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

Tam pipeline:

```bash
python main.py --frequencies 1D --models MV MVSK MCVaRSK --resume
```

Sadece student asamasi:

```bash
python run_student_only.py --freqs 1D --oos-split 0.25 --top-k-teachers 50 --top-combos 200 --same-asset-count --model xgb --xgb-multi-output --ml-onfly --models MVSK --two-stage --stage1-candidate-multiplier 2 --stage1-train-frac 0.80 --stage1-max-rows 120000 --xgb-learning-rate 0.05 --xgb-max-depth 4 --xgb-n-estimators 700 --xgb-subsample 0.9 --xgb-colsample-bytree 0.9 --xgb-min-child-weight 3 --xgb-reg-alpha 0.0 --xgb-reg-lambda 1.0 --no-checkpoint
```

Teacher ranking yeniden hesaplama:

```bash
python recalc_teacher_ranking.py 1D
```

## Notlar

- Uretilen sonuc bundle'lari versiyon kontrolune alinmaz.
- Repo, tekrar uretilebilir kodu ve kisa dokumantasyonu tutar; buyuk deney ciktilari pipeline ile yeniden uretilir.
