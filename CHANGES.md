# 🚀 Dual-Winner Architecture - Değişiklik Özeti

## 📋 Yapılan Tüm Değişiklikler

### ✅ 1. HATA DÜZELTMELERİ (Bug Fixes)

#### `main.py`
- **Satır 275:** `group_szes` → `group_sizes` ✅

#### `src/combination_utils.py`
- **Satır 22:** `combnatons.pkl` → `combinations.pkl` ✅
- **Satır 92:** `get_all_combnatons()` → `get_all_combinations()` ✅
- **Satır 167:** `group_szes` → `group_sizes` ✅
- **Satır 220:** `__all__` güncellendi ✅

---

### ✅ 2. YENİ MODÜLler

#### `src/ml_weights.py` (YENİ! 🆕)
**Teacher-Student Direct Weight Learning**

```python
def train_weight_models(teacher_results, processed_dir, freq):
    """
    ML modelleri teacher ağırlıklarından öğrenir.

    ❌ Moment tahmini YOK
    ✅ Doğrudan ağırlık tahmini

    Architecture:
    - Input: Lagged moments (10 windows) + raw returns
    - Model: LightGBM (her varlık için ayrı)
    - Target: Teacher optimal weights
    - Constraints: Softmax + Top-K + Weight cap (30%)
    """
```

**Özellikler:**
- `_load_moments()`: Moment verilerini yükler
- `_load_returns()`: Return verilerini yükler
- `_extract_teacher_weights()`: Teacher ağırlıklarını çıkarır
- `_create_lagged_features()`: Gecikmeli özellikler oluşturur
- `_apply_portfolio_constraints()`: Portföy kısıtlamalarını uygular
- `train_weight_models()`: Ana eğitim fonksiyonu
- `load_ml_weights()`: Eğitilmiş modelleri yükler

---

### ✅ 3. MAIN.PY - DUAL-WINNER MİMARİSİ

#### Eski Yapı (❌ Kaldırıldı):
```python
# STEP 1: Baseline çalıştır
# STEP 2: EN İYİ kombinasyonu bul
# STEP 3: ML sadece EN İYİ için çalıştır
```

#### Yeni Yapı (✅ Eklendi):
```python
# STEP 1: TEACHER - TÜM kombinasyonları test et
#   - MV, MVSK, MCVaRSK
#   - Sharpe'a göre sırala
#   - Teacher winner seç

# STEP 2: TEACHER RANKING
#   - Top 10 göster
#   - CSV'ye kaydet
#   - JSON winner kaydet

# STEP 3: STUDENT - ML weight learning
#   - Teacher'dan öğren
#   - TÜM kombinasyonları test et
#   - Sharpe'a göre sırala
#   - Student winner seç

# STEP 4: STUDENT RANKING
#   - Top 10 göster
#   - CSV'ye kaydet
#   - JSON winner kaydet

# STEP 5: TEACHER vs STUDENT
#   - Karşılaştırma tablosu
#   - Kazananı belirle
#   - JSON comparison kaydet
```

**Çıktı Dosyaları:**
```
results/pipeline/
├── teacher_1d.parquet              # Teacher backtest
├── teacher_1h.parquet
├── student_1d.parquet              # Student backtest
├── student_1h.parquet
├── teacher_ranking_1d.csv          # Sıralamalar
├── teacher_ranking_1h.csv
├── student_ranking_1d.csv
├── student_ranking_1h.csv
├── winner_teacher_1d.json          # Kazananlar
├── winner_teacher_1h.json
├── winner_student_1d.json
├── winner_student_1h.json
└── teacher_vs_student_1d.json      # Karşılaştırma
```

---

### ✅ 4. BACKTEST ENGINE GÜNCELLEMELERİ

#### `src/backtest_engine.py`

**Yeni Fonksiyon:**
```python
def _load_ml_weights(freq: str, processed_dir: Path) -> pd.DataFrame | None:
    """ML-predicted portfolio weights yükler."""
```

**Güncelleme:**
```python
# run_backtest() fonksiyonuna eklendi:
ml_weights_df = None
if version_norm in ML_VERSION:
    ml_weights_df = _load_ml_weights(freq_norm, processed_dir)
    if ml_weights_df is not None:
        print(f"[Backtest] Using ML-predicted weights")
```

---

### ✅ 5. README.MD - TAM YENİLEME

**Yeni Bölümler:**
- ✅ Dual-Winner Architecture açıklaması
- ✅ Teacher-Student framework detayları
- ✅ ML-Weights (direct weight learning) açıklaması
- ✅ Pipeline Architecture (5 adım)
- ✅ Expected Outputs (JSON, CSV formatları)
- ✅ Console Output örnekleri
- ✅ Comparison table (Traditional vs Dual-Winner)

**Güncellenmiş Bölümler:**
- ✅ Project Structure (ml_weights.py eklendi)
- ✅ Features (ML-Weights yetenekleri)
- ✅ Methods (Teacher + Student açıklaması)
- ✅ How to Run (yeni komutlar)

---

### ✅ 6. SRC/__INIT__.PY

```python
# Yeni modül export edildi:
from . import ml_weights

__all__ = [
    'backtest_engine',
    'combination_utils',
    'data_prep',
    'metrics',
    'ml_weights',      # 🆕 EKLENDI
    'moment_calc',
    'optim_models',
    'reporting',
]
```

---

## 🎯 ÖNCEKİ vs YENİ MİMARİ

### ❌ Önceki Mimari (Kaldırıldı):
```
1. Tüm kombinasyonları baseline ile test et
2. EN İYİ kombinasyonu bul (Sharpe'a göre)
3. ML moment forecasting yap
4. Sadece EN İYİ için ML backtest çalıştır
5. Baseline vs ML karşılaştır (sadece 1 portföy)
```

**Sorunlar:**
- ❌ Sadece 1 kombinasyon ML ile test ediliyordu
- ❌ ML moment forecasting hataları
- ❌ Ranking sistemi yoktu
- ❌ Tüm kombinasyonları karşılaştıramıyorduk

### ✅ Yeni Mimari (Dual-Winner):
```
1. TEACHER: TÜM 16,834 kombinasyonu test et
   → Teacher ranking oluştur
   → Teacher winner seç

2. STUDENT: ML modelleri eğit (Teacher'dan öğren)
   → TÜM 16,834 kombinasyonu ML ile test et
   → Student ranking oluştur
   → Student winner seç

3. COMPARISON: Teacher vs Student
   → Her ikisini de karşılaştır
   → Genel kazananı belirle
```

---

## 📅 2025-12-26 Güncellemeleri

### ✅ Pipeline ve Backtest
- **Annualization 365:** Tüm raporlama ve teacher ranking yıllıklandırma 365 gün.
- **Checkpoint düzeltmesi:** Resume artık combo+model bazlı doğru devam ediyor.
- **ML ağırlıkları kullanımı:** Student backtest’te ML ağırlıkları gerçekten uygulanıyor.

### ✅ Student ML (XGBoost + Multi-Output)
- **Multi-output XGB:** Tek modelle tüm ağırlıklar birlikte öğreniliyor.
- **Yeni bayraklar:** `--xgb-multi-output`, `--models`, `--combo-limit`, `--n-lags`.
- **Deney arşivi:** ML ağırlıkları hash’lenip `results/attempts` altına kaydediliyor.

### ✅ Kalite ve Esneklik
- **Teacher seçim esnekliği:** Top‑K teacher ve aynı asset sayısı filtresi.
- **Augmentation:** Opsiyonel gürültü ile target çeşitlendirme.
- **Softmax temperature:** Tahmin ağırlıklarını yumuşatma opsiyonu.
- **Model kaydetme fallback:** Yazma izni yoksa `ml_models_user` kullanımı.

**Avantajlar:**
- ✅ TÜM kombinasyonlar ML ile test edilir
- ✅ Moment forecasting hatası yok (doğrudan weight learning)
- ✅ Kapsamlı ranking sistemi
- ✅ 2 kazanan: En iyi klasik + En iyi ML
- ✅ Teacher-Student öğrenme ile daha iyi sonuçlar

---

## 📊 SONUÇLAR

### Winner JSON Örneği:
```json
{
  "freq": "1D",
  "combo": "ethbtc_solbtc_linkbtc",
  "model": "MVSK",
  "sharpe": 1.8542,
  "annualized_return": 0.3421,
  "volatility": 0.1845,
  "version": "teacher"
}
```

### Ranking CSV Örneği:
```csv
combo,model,sharpe,annualized_return,volatility
ethbtc_solbtc_linkbtc,MVSK,1.8542,0.3421,0.1845
btceth_adabtc,MV,1.7234,0.3102,0.1800
solbtc_linkbtc_adabtc,MCVaRSK,1.6890,0.2987,0.1770
...
```

### Comparison JSON Örneği:
```json
{
  "freq": "1D",
  "teacher": {
    "combo": "ethbtc_solbtc_linkbtc",
    "sharpe": 1.8542
  },
  "student": {
    "combo": "btceth_linkbtc",
    "sharpe": 1.9123
  },
  "winner": "student"
}
```

---

## 🚀 NASIL ÇALIŞTIRILIR

### Tam Pipeline:
```bash
python main.py
```

### Sadece Teacher (Baseline):
```bash
python main.py --versions baseline
```

### Sadece Student (ML):
```bash
python main.py --versions ml
```

### Belirli Frekans:
```bash
python main.py --frequencies 1D
python main.py --frequencies 1H
python main.py --frequencies 1D 1H
```

### Belirli Modeller:
```bash
python main.py --models MV MVSK
python main.py --models MCVaRSK --frequencies 1D
```

---

## 📈 CONSOLE ÇIKTISI

```
🚀 DUAL-WINNER PORTFOLIO OPTIMIZATION FRAMEWORK
================================================================================
Testing 16834 portfolio combinations
Models: MV, MVSK, MCVaRSK
Frequencies: 1D, 1H
================================================================================

📚 STEP 1: TEACHER PORTFOLIOS (Classical Optimization)
================================================================================

[Teacher] Testing 16834 combinations at freq=1D
[Teacher] Saved results to results/pipeline/teacher_1d.parquet

📊 STEP 2: Generating TEACHER RANKING for freq=1D

🏆 TOP 10 TEACHER PORTFOLIOS (1D):
================================================================================
Rank  Combo                              Model     Sharpe    Return      Vol
1     ethbtc_solbtc_linkbtc             MVSK      1.8542    34.21%      18.45%
2     btceth_adabtc                     MV        1.7234    31.02%      18.00%
3     solbtc_linkbtc_adabtc            MCVaRSK   1.6890    29.87%      17.70%
...
================================================================================

👑 TEACHER WINNER (1D):
   Combo: ethbtc_solbtc_linkbtc
   Model: MVSK
   Sharpe: 1.8542
   Annual Return: 34.21%
   Volatility: 18.45%

🤖 STEP 3: STUDENT PORTFOLIOS (ML Direct Weight Learning)
================================================================================

[Student] Training ML models to learn portfolio weights...
[Student] Found 20 assets
[Student] Creating lagged features with 10 lags...
[Student] Training 20 asset models...
[Student]   ✓ Trained model for ethbtc
[Student]   ✓ Trained model for solbtc
...
[Student] Successfully trained 20 models
[Student] Saved models to data/processed/ml_models/weight_models_1d.pkl

[Student] Running backtest with ML weights for 16834 combinations...
[Student] Saved results to results/pipeline/student_1d.parquet

📊 STEP 4: Generating STUDENT RANKING for freq=1D

🏆 TOP 10 STUDENT PORTFOLIOS (1D):
================================================================================
Rank  Combo                              Model     Sharpe    Return      Vol
1     btceth_linkbtc                    MVSK      1.9123    36.45%      19.05%
2     ethbtc_solbtc_linkbtc             MV        1.8890    35.78%      18.90%
...
================================================================================

👑 STUDENT WINNER (1D):
   Combo: btceth_linkbtc
   Model: MVSK
   Sharpe: 1.9123
   Annual Return: 36.45%
   Volatility: 19.05%

⚖️  STEP 5: TEACHER vs STUDENT COMPARISON (1D)
================================================================================

Metric                   Teacher             Student             Winner
--------------------------------------------------------------------------------
Sharpe Ratio            1.8542              1.9123              🏆 Student
Annual Return           34.21%              36.45%              🏆 Student
Volatility              18.45%              19.05%              🏆 Teacher
================================================================================

💾 Saved comparison to results/pipeline/teacher_vs_student_1d.json

✅ Completed optimization pipeline for freq=1D
================================================================================
```

---

## 🎓 ÖNEMLİ NOTLAR

1. **ML Moment Forecasting Kaldırıldı:**
   - ❌ Artık moment tahmini YAPILMIYOR
   - ✅ Doğrudan weight prediction

2. **Teacher-Student Framework:**
   - Teacher: Klasik optimizerler (MV, MVSK, MCVaRSK)
   - Student: Teacher'dan öğrenen ML modelleri
   - Advantage: Optimal solver çıktılarından öğrenme

3. **Tüm Kombinasyonlar Test Edilir:**
   - 2-asset: 190 kombinasyon
   - 3-asset: 1,140 kombinasyon
   - 5-asset: 15,504 kombinasyon
   - **TOPLAM: 16,834 kombinasyon**

4. **İki Kazanan:**
   - 👑 Teacher Winner (En iyi klasik)
   - 👑 Student Winner (En iyi ML)

5. **Kapsamlı Sıralama:**
   - Her kombinasyon Sharpe'a göre sıralanır
   - CSV dosyalarına kaydedilir
   - Top 10 console'da gösterilir

---

## 🏆 SONUÇ

Proje artık **tamamen Dual-Winner mimarisine** göre yapılandırıldı:

✅ Tüm typo'lar düzeltildi
✅ ML-Weights modülü eklendi
✅ Main.py Dual-Winner olarak güncellendi
✅ Backtest engine ML weights desteği eklendi
✅ README tamamen yenilendi
✅ Kapsamlı ranking sistemi eklendi
✅ Teacher vs Student comparison eklendi

**ÇALIŞTIRIn:**
```bash
python main.py
```

**SONUÇ:**
İki kazanan portföy elde edeceksiniz:
- 👑 **Teacher Winner** (En iyi klasik optimizasyon)
- 👑 **Student Winner** (En iyi ML weight learning)

Her ikisi de JSON dosyalarında kaydedilmiş ve karşılaştırılmış olacak! 🎉
