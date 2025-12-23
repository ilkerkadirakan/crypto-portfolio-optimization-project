# crypto_portfolio_moments

## 1. Genel Bakış

`crypto_portfolio_moments` projesi, kripto para portföyleri için **İkili-Kazanan Portföy Optimizasyon Çerçevesi** (Dual-Winner Portfolio Optimization Framework) uygular. Sistem, klasik yüksek-moment optimizasyonu (MV, MVSK, MCVaRSK) ile makine öğrenmesi tabanlı doğrudan ağırlık tahminini birleştirerek 20 varlıklı bir evrende en iyi performans gösteren portföyleri bulur.

**Temel Yenilik:** Geleneksel ML yaklaşımlarından farklı olarak, modellerimiz momentleri tahmin etmek yerine portföy ağırlıklarını doğrudan öğrenir ve klasik optimizatör çıktılarını taklit ederek iyileştirir (Öğretmen-Öğrenci öğrenme).

Hem günlük (1D) hem de saatlik (1H) frekanslarda birleşik 180 günlük geriye bakış penceresi (saatlik veri için 4320 saat) kullanılır, tüm pipeline aşamalarında tutarlılık sağlanır.

## 2. Özellikler

### **İkili-Kazanan Mimarisi**
- **Öğretmen Portföyleri (Klasik):** Yüksek-moment risk modellemeli MV, MVSK ve MCVaRSK optimizatörleri
- **Öğrenci Portföyleri (ML-Ağırlıkları):** Öğretmen çıktılarından doğrudan optimal ağırlıkları öğrenen LightGBM modelleri
- **Sıralama Sistemi:** Tüm portföy kombinasyonlarının Sharpe oranına göre kapsamlı sıralaması
- **Kazanan Seçimi:** En iyi öğretmen ve en iyi öğrenci portföylerinin otomatik belirlenmesi

### **Temel Yetenekler**
- **Kapsamlı kombinasyon testi:** 2-varlık, 3-varlık ve 5-varlık portföyleri (toplam 16.834 kombinasyon)
- **Yüksek-moment optimizasyonu:** Ortalama-Varyans-Çarpıklık-Basıklık (MVSK) ve CVaR tabanlı modeller
- **Doğrudan ML ağırlık öğrenme:** Moment tahmini yok, öğretmen-öğrenci çerçevesiyle saf ağırlık tahmini
- **Paralel işleme:** Çok çekirdekli işlemcilerle dramatik hızlanma için multiprocessing desteği (8-32x daha hızlı)
- **Yuvarlanma backtesting:** İşlem maliyetleri (10 bps) ve devir hızı takibi
- **Kapsamlı metrikler:** Sharpe, Sortino, Maksimum Düşüş, CVaR
- **Bulut-hazır:** AWS, GCP ve Hetzner dağıtımı için optimize edilmiş
- **Merkezi YAML yapılandırması:** Tekrar üretilebilir deneyler

## 3. Proje Yapısı

```
crypto_portfolio_moments/
├── main.py                      # Pipeline düzenleyici (İkili-Kazanan çerçevesi)
├── README.md                    # İngilizce dokümantasyon
├── README.tr.md                 # Türkçe dokümantasyon
├── requirements.txt
├── configs/
│   ├── params.yaml             # Pipeline yapılandırması
│   └── assets.yaml             # Varlık evreni (20 kripto para)
├── data/
│   ├── raw/                    # Ham .mat fiyat dosyaları (36 ay, dakikalık)
│   ├── processed/              # İşlenmiş getiriler ve momentler
│   │   ├── returns_1h.parquet
│   │   ├── returns_1d.parquet
│   │   ├── moments_1h.parquet
│   │   ├── moments_1d.parquet
│   │   ├── ml_predicted_weights_1h.parquet
│   │   └── ml_predicted_weights_1d.parquet
│   └── meta/
│       └── combinations.pkl    # Önbelleğe alınmış portföy kombinasyonları
├── results/
│   ├── pipeline/               # Ana sonuçlar dizini
│   │   ├── teacher_1h.parquet       # Öğretmen backtest sonuçları (saatlik)
│   │   ├── teacher_1d.parquet       # Öğretmen backtest sonuçları (günlük)
│   │   ├── student_1h.parquet       # Öğrenci backtest sonuçları (saatlik)
│   │   ├── student_1d.parquet       # Öğrenci backtest sonuçları (günlük)
│   │   ├── teacher_ranking_1h.csv   # Öğretmen portföy sıralamaları
│   │   ├── teacher_ranking_1d.csv
│   │   ├── student_ranking_1h.csv   # Öğrenci portföy sıralamaları
│   │   ├── student_ranking_1d.csv
│   │   ├── winner_teacher_1h.json   # En iyi öğretmen portföyü
│   │   ├── winner_teacher_1d.json
│   │   ├── winner_student_1h.json   # En iyi öğrenci portföyü
│   │   ├── winner_student_1d.json
│   │   └── teacher_vs_student_*.json # Performans karşılaştırması
│   ├── runs/                   # Bireysel çalıştırma sonuçları
│   ├── tables/                 # Özet metrikler
│   └── figs/                   # Görselleştirmeler
└── src/
    ├── __init__.py
    ├── data_prep.py            # Veri yükleme ve ön işleme
    ├── moment_calc.py          # Yüksek-moment hesaplama
    ├── ml_weights.py           # ML doğrudan ağırlık öğrenme (Öğretmen-Öğrenci)
    ├── optim_models.py         # Klasik portföy optimizatörleri (MV, MVSK, MCVaRSK)
    ├── backtest_engine.py      # Yuvarlanma backtest motoru (paralel işleme destekli)
    ├── combination_utils.py    # Portföy kombinasyon üreteci
    ├── metrics.py              # Performans metrikleri
    └── reporting.py            # Görselleştirme ve raporlama
```

## 4. Pipeline Mimarisi

### **ADIM 1: Veri Hazırlama**
Ham `.mat` fiyat dosyaları (36 ay, dakikalık):
- 20 varlık üzerinde birleştirilir ve senkronize edilir
- 1H ve 1D frekanslarına yeniden örneklenir
- Log getirilerine dönüştürülür

### **ADIM 2: Moment Hesaplama**
Yuvarlanma 180 günlük pencereler hesaplar:
- Ortalama (μ)
- Ledoit-Wolf kovaryans (Σ)
- Çarpıklık
- Basıklık
- CVaR (95. yüzdelik)

### **ADIM 3: Öğretmen Portföyleri (Klasik Optimizasyon)**
TÜM 16.834 kombinasyon için:
- MV, MVSK, MCVaRSK optimizatörleri çalıştırılır
- Yuvarlanma yeniden dengeleme ile backtest yapılır
- Sharpe oranına göre sıralama üretilir
- **ÖĞRETMEN KAZANAN** seçilir

### **ADIM 4: Öğrenci Portföyleri (ML Ağırlık Öğrenme)**
- Portföy ağırlıklarını tahmin etmek için LightGBM modelleri eğitilir
- **Girdi özellikleri:** Gecikmeli momentler (ortalama, varyans, çarpıklık, basıklık, CVaR) + ham getiriler
- **Hedef:** Öğretmen optimal ağırlıkları
- **Kısıtlamalar:** Softmax, Top-K maskesi, ağırlık üst sınırı (%30)
- TÜM kombinasyonlar ML-tahminli ağırlıklarla backtest edilir
- Sharpe oranına göre sıralama üretilir
- **ÖĞRENCİ KAZANAN** seçilir

### **ADIM 5: İkili-Kazanan Karşılaştırması**
Öğretmen ve Öğrenci karşılaştırılır:
- Sharpe Oranı
- Yıllıklandırılmış Getiri
- Volatilite
- Maksimum Düşüş

## 5. Yöntemler

### **Klasik Optimizasyon (Öğretmen)**
- **MV (Ortalama-Varyans):** Klasik Markowitz optimizasyonu
  ```
  min_w  w'Σw - λ_μ μ'w
  ```
- **MVSK (Ortalama-Varyans-Çarpıklık-Basıklık):** Yüksek-moment uzantısı
  ```
  min_w  w'Σw - λ_μ μ'w - λ_s S'w + λ_k K'w
  ```
- **MCVaRSK (CVaR-tabanlı):** Koşullu Riske Maruz Değer kullanan risk ölçüsü
  ```
  min_w  CVaR_α(w) - λ_μ μ'w - λ_s S'w + λ_k K'w
  ```

### **ML Ağırlık Öğrenme (Öğrenci)**
**Mimari:** Öğretmen-Öğrenci Distilasyonu
- Her varlık için bir LightGBM regresörü
- **Özellikler (varlık başına, 10 gecikme):**
  - Gecikmeli ortalama getiriler
  - Gecikmeli varyans
  - Gecikmeli çarpıklık
  - Gecikmeli basıklık
  - Gecikmeli CVaR
  - Gecikmeli ham getiriler
- **Hedef:** Öğretmen optimal ağırlıkları w_T
- **Son işleme:**
  1. Softmax normalizasyonu
  2. Top-K maskesi (sadece combo varlıkları)
  3. Ağırlık üst sınırı (varlık başına maksimum %30)
  4. Son normalizasyon (toplam = 1)

### **Backtesting**
- **Yeniden dengeleme:** Haftalık (1D) / 24-saat (1H)
- **İşlem maliyetleri:** İşlem başına 10 baz puan
- **Devir hızı takibi:** Pozisyon değişikliklerinin tam muhasebesi
- **Performans metrikleri:** Sharpe, Sortino, Maks Düşüş, CVaR

## 6. Nasıl Çalıştırılır

### Hızlı Başlangıç
```bash
# Bağımlılıkları yükle
pip install -r requirements.txt

# Tam pipeline'ı çalıştır (paralel, tüm CPU'lar)
python main.py
```

### Paralel İşleme (YENİ)
Pipeline artık çalışma süresini dramatik şekilde azaltmak için multiprocessing kullanarak **paralel işlemeyi** destekliyor:

```bash
# Tüm mevcut CPU'ları kullan (varsayılan)
python main.py

# Worker sayısını belirt
python main.py --n-jobs 16

# Paralel işlemeyi devre dışı bırak (sıralı)
python main.py --no-parallel
```

**Performans Karşılaştırması:**
| Worker Sayısı | 16.834 Kombinasyon × 3 Model | Tahmini Süre |
|---------------|------------------------------|--------------|
| 1 (sıralı) | 50.502 optimizasyon | 140-280 saat (5-12 gün) |
| 8 çekirdek | Paralel işleme | 17-35 saat |
| 16 çekirdek | Paralel işleme | 9-18 saat |
| 32 çekirdek | Paralel işleme | 5-9 saat |

### Belirli Frekansları Çalıştır
```bash
python main.py --frequencies 1D
python main.py --frequencies 1H
python main.py --frequencies 1D 1H
```

### Belirli Modelleri Çalıştır
```bash
python main.py --models MV MVSK MCVaRSK
python main.py --models MVSK --frequencies 1D
```

### Sadece Öğretmen veya Öğrenci
```bash
# Sadece öğretmen (baseline)
python main.py --versions baseline

# Sadece öğrenci (ML)
python main.py --versions ml
```

### Kombine Örnekler
```bash
# Hızlı test: sadece baseline, 1D, tek model, 8 worker
python main.py --versions baseline --frequencies 1D --models MV --n-jobs 8

# Üretim: tüm kombinasyonlar, 16 worker
python main.py --n-jobs 16

# Bulut dağıtımı: paralelleştirmeyi maksimize et
python main.py --n-jobs 32
```

## 7. Beklenen Çıktılar

### **Kazanan Dosyaları (JSON)**
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

### **Sıralamalar (CSV)**
Sharpe oranına göre sıralanmış en iyi portföyler:
```
combo,model,sharpe,annualized_return,volatility
ethbtc_solbtc_linkbtc,MVSK,1.8542,0.3421,0.1845
btceth_adabtc,MV,1.7234,0.3102,0.1800
...
```

### **Karşılaştırma (JSON)**
```json
{
  "freq": "1D",
  "teacher": {...},
  "student": {...},
  "winner": "student"
}
```

### **Konsol Çıktısı**
```
================================================================================
İKİLİ-KAZANAN PORTFÖY OPTİMİZASYON ÇERÇEVESİ
================================================================================
16834 portföy kombinasyonu test ediliyor
Modeller: MV, MVSK, MCVaRSK
Frekanslar: 1D, 1H
================================================================================

================================================================================
ADIM 1: ÖĞRETMEN PORTFÖYLER (Klasik Optimizasyon)
================================================================================
[Teacher] 2 worker ile paralel işleme kullanılıyor
[Parallel Backtest] 50502 görev işleniyor (16834 kombinasyon × 3 model)
[Parallel Backtest] 2 worker process kullanılıyor
[Parallel Backtest] 100/50502 görev tamamlandı (0.2%)
...

EN İYİ 10 ÖĞRETMEN PORTFÖYÜ (1D):
================================================================================
Sıra  Kombinasyon                        Model     Sharpe    Getiri      Vol
1     ethbtc_solbtc_linkbtc             MVSK      1.8542    34.21%      18.45%
2     btceth_adabtc                     MV        1.7234    31.02%      18.00%
...

ÖĞRETMEN KAZANAN (1D):
   Kombinasyon: ethbtc_solbtc_linkbtc
   Model: MVSK
   Sharpe: 1.8542
   Yıllık Getiri: 34.21%
   Volatilite: 18.45%
```

## 8. Sistem Gereksinimleri

### Yerel Makine
- **Python:** 3.10 veya daha yeni
- **RAM:** Minimum 16 GB (tam kombinasyonlar için 32 GB önerilir)
- **CPU:** Çok çekirdekli (paralel işleme için 8-16 çekirdek önerilir)
- **Depolama:** Veri ve sonuçlar için 10 GB

### Temel Bağımlılıklar
- `numpy`, `pandas`, `scipy`
- `cvxpy` (konveks optimizasyon)
- `lightgbm` (ML modelleri)
- `scikit-learn` (ön işleme)
- `matplotlib`, `seaborn` (görselleştirme)
- `PyYAML`, `joblib`, `tqdm`

### Bulut Dağıtımı

Büyük ölçekli deneyler için bulut dağıtımı önerilir. Pipeline CPU-yoğundur (GPU'ya bağımlı değildir).

#### Önerilen Bulut Sağlayıcıları

**1. Hetzner Cloud (En Uygun Maliyetli)**
- **CCX33:** 8 vCPU, 32GB RAM → €47/ay (~$50)
- **CCX63:** 16 vCPU, 64GB RAM → €138/ay (~$150)
- En uygun: Uzun süreli deneyler, maliyet-hassas projeler

**2. AWS EC2**
- **c7i.4xlarge:** 16 vCPU, 32GB RAM → $0.68/saat
- **c7i.8xlarge:** 32 vCPU, 64GB RAM → $1.36/saat
- **Spot instance'lar:** %60-70 indirim → $0.20-0.40/saat
- En uygun: Esnek hesaplama, kısa patlamalar

**3. Google Cloud Compute**
- **c3-standard-22:** 22 vCPU, 88GB RAM → $1.05/saat
- **Preemptible instance'lar:** %80 indirim → $0.21/saat
- En uygun: Yüksek çekirdek sayısı, toplu işleme

#### Bulut Kurulum Örneği (AWS)
```bash
# 1. EC2 instance başlat (Ubuntu 22.04, c7i.4xlarge)
# 2. Bağımlılıkları yükle
sudo apt update
sudo apt install -y python3.10 python3-pip git

# 3. Repository'yi klonla
git clone <repo-url>
cd crypto_portfolio_moments

# 4. Python bağımlılıklarını yükle
pip3 install -r requirements.txt

# 5. Veriyi yükle (eğer repo'da değilse)
# scp, rsync veya S3 sync kullan

# 6. Pipeline'ı 16 worker ile çalıştır
python3 main.py --n-jobs 16

# 7. Sonuçları indir
# scp -r results/ yerel-makine:~/results/
```

#### Maliyet Tahmini (AWS Spot Instance'lar)
| Kombinasyon Boyutu | Çalışma Süresi (16 çekirdek) | Maliyet (spot @ $0.20/saat) |
|--------------------|-------------------------------|----------------------------|
| Sadece 2-varlık (190) | ~1 saat | $0.20 |
| 2 + 3-varlık (1.330) | ~7 saat | $1.40 |
| Tam (16.834) | ~18 saat | $3.60 |

#### Git LFS Gerekli Değil
- **Veri dosyaları (.mat)** `.gitignore`'da (toplam 71 MB)
- **Sonuçlar (.parquet)** `.gitignore`'da
- Sadece **kod ve yapılandırmalar** Git'te
- Veriyi ayrıca scp, rsync veya bulut depolama ile yükleyin

## 9. Yapılandırma

`configs/params.yaml` dosyasını düzenleyerek özelleştirin:
- **Veri yolları:** raw_dir, processed_dir, results_dir
- **Pencere boyutları:** yuvarlanma penceresi (180 gün)
- **Yeniden dengeleme:** frekans ve kurallar
- **Model parametreleri:** lambda ağırlıkları, kısıtlamalar
- **ML ayarları:** n_estimators, learning_rate, vb.

### Kombinasyon Sayısını Azaltma

Eğer 16.834 kombinasyon çok fazlaysa, `configs/params.yaml` dosyasına şunu ekleyin:

```yaml
backtest:
  combos:
    mode: auto
    group_sizes: [2]  # Sadece 2'li kombinasyonlar (190 adet)
```

veya

```yaml
backtest:
  combos:
    mode: auto
    group_sizes: [2, 3]  # 2'li ve 3'lü (toplam 1.330 adet)
```

## 10. Yazarlar ve Lisans

- **Yazarlar:** Kadir & Batuhan (proje entegratörleri) ile LLM agent işbirliği
- **Çerçeve:** Öğretmen-Öğrenci ML ile İkili-Kazanan Portföy Optimizasyonu
- **Lisans:** Repository dokümantasyonuna bakın veya proje sahipleriyle iletişime geçin

---

## Geleneksel Yaklaşımlardan Temel Farklar

| Yön | Geleneksel ML | Bizim Yaklaşımımız (İkili-Kazanan) |
|-----|---------------|-----------------------------------|
| ML Hedefi | Momentleri tahmin et (μ, σ, CVaR) | Ağırlıkları doğrudan tahmin et |
| Mimari | Tek aşamalı | Öğretmen-Öğrenci distilasyonu |
| Baseline | Bir referans portföy | Tüm kombinasyonlar test edilir |
| Çıktı | En iyi portföy | İki kazanan (Öğretmen + Öğrenci) |
| Karşılaştırma | ML vs Baseline | Kapsamlı sıralama sistemi |

**Yenilik:** Doğrudan ağırlık öğrenme, moment tahmin hatalarını ortadan kaldırır ve optimal çözücü çıktılarından öğrenir.

---

## Sık Sorulan Sorular

### 1. Bilgisayarım yeterli mi?
**Minimum:** 16GB RAM, 8 çekirdek CPU (test için yeterli)
**Önerilen:** 32GB RAM, 16+ çekirdek CPU (tüm kombinasyonlar için)
**Bulut:** AWS/Hetzner 16-32 çekirdek instance (~18 saat, $3-10)

### 2. Çalışma süresi ne kadar?
- **Yerel (8 çekirdek):** ~24-35 saat
- **Yerel (16 çekirdek):** ~12-18 saat
- **Bulut (32 çekirdek):** ~5-9 saat

### 3. GPU gerekli mi?
Hayır! Bu CPU-yoğun bir iştir. CVXPY optimizasyon ve LightGBM CPU kullanır.

### 4. Kombinasyon sayısını nasıl azaltırım?
`configs/params.yaml`'da `group_sizes: [2]` veya `group_sizes: [2, 3]` ayarlayın.

### 5. Paralel işleme çalışmıyor mu?
```bash
# Worker sayısını kontrol edin
python main.py --n-jobs 8  # 8 worker

# Paralel işlemeyi devre dışı bırakın
python main.py --no-parallel
```

### 6. Sonuçları nerede bulabilirim?
- **Ana sonuçlar:** `results/pipeline/`
- **Sıralamalar:** `results/pipeline/teacher_ranking_1d.csv`
- **Kazananlar:** `results/pipeline/winner_teacher_1d.json`
- **Karşılaştırma:** `results/pipeline/teacher_vs_student_1d.json`

---

## İletişim ve Destek

Sorularınız veya sorunlarınız için:
- GitHub Issues: `<repo-url>/issues`
- Yazarlar: Kadir & Batuhan

**Başarılı optimizasyonlar! 🚀**