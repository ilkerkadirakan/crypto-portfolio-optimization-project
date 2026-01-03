# Teknik Rapor (Aşırı Detaylı)

## 0. Özet
Bu rapor, kripto portföy optimizasyonu projesinin teknik mimarisini uçtan uca, veri akışından model eğitimine ve backtest raporlamasına kadar detaylandırır. Amaç; klasik MV/MVSK/MCVaRSK optimizasyonunu “Teacher” olarak kullanıp, bu çıktıları denetimli öğrenmeyle taklit eden “Student” modelini eğitmek ve performansı Sharpe, getiri ve volatilite üzerinden kıyaslamaktır.

---

## 1. Veri Katmanı
### 1.1 Veri Kaynağı
- 36 aylık yüksek frekanslı kripto fiyat serileri (20 varlık).
- Ham veri `.mat`/ham formatlardan projeye aktarılır.

### 1.2 Yeniden Örnekleme
- Ham seriler 1H ve 1D frekansına indirgenir.
- Analizin ana çıktıları 1D üzerinden üretilir.

### 1.3 Getiri Hesabı
- Log getiriler: `r_t = log(P_t / P_{t-1})`
- Uygulamada `returns_1d.parquet` ve `returns_1h.parquet` saklanır.

---

## 2. Moment Hesaplama
### 2.1 Rolling Pencere
- 1D için 180 günlük rolling pencere.
- 1H için 4320 saat (yaklaşık 180 gün).

### 2.2 Hesaplanan Momentler
- Ortalama (μ)
- Varyans ve kovaryans (Ledoit-Wolf shrinkage)
- Çarpıklık (skewness)
- Basıklık (kurtosis)
- CVaR (α=0.95)

### 2.3 Çıktılar
- `moments_1d.parquet`, `moments_1h.parquet`
- Momentler, Teacher ve Student için ortak giriş kaynağıdır.

---

## 3. Teacher (Klasik Optimizasyon)
### 3.1 Modeller
- MV (Mean-Variance)
- MVSK (Mean-Variance-Skewness-Kurtosis)
- MCVaRSK (CVaR tabanlı)

### 3.2 Amaç Fonksiyonları
- MV: `min w'Σw - λ_μ μ'w`
- MVSK: `min w'Σw - λ_μ μ'w - λ_s S'w + λ_k K'w`
- MCVaRSK: `min λ_cvar CVaR(w) - λ_μ μ'w - λ_s S'w + λ_k K'w`

### 3.3 Kısıtlar
- Ağırlık toplamı = 1
- Long-only (varsayılan)
- Ağırlık üst sınırı (config: max_weight = 0.50)

### 3.4 Çözümleyici
- cvxpy
- Solver öncelikleri: ECOS → SCS

### 3.5 Kombinasyon Kapsamı
- 2, 3 ve 5 varlık kombinasyonları
- Toplam 16,834 combo

### 3.6 Teacher Çıktıları
- `teacher_1d.parquet`
- `teacher_ranking_1d.csv`
- `winner_teacher_1d.json`

---

## 4. Student (Denetimli Öğrenme)
### 4.1 Amaç
Teacher optimal ağırlıklarını tahmin ederek portföy optimizasyonunu ML ile hızlandırmak ve potansiyel performans artışı sağlamak.

### 4.2 Özellik Seti
- Lag’lı momentler: mean, variance, skewness, kurtosis, CVaR
- Lag’lı getiriler
- n_lags = 5 (deneyde ana parametre)

### 4.3 Model
- Multi-output XGBoost
- Tek model tüm asset ağırlıklarını aynı anda tahmin eder

### 4.4 Combo-Conditional Tasarım
- Combo bilgisi `combo_has_<asset>` özellikleriyle modele verilir
- Böylece her kombinasyon için farklı ağırlık üretimi sağlanır

### 4.5 Öğrenme Hedefi
- Teacher optimal ağırlıkları (doğrudan tahmin)

### 4.6 Eğitim Verisi Seçimi
- Teacher sıralamasından Top-K (300) portföy
- same-asset-count: teacher winner ile aynı varlık sayısına sahip kombinasyonlar

### 4.7 Çıktılar
- `ml_predicted_weights_1d.parquet`
- `student_1d.parquet`
- `student_ranking_1d.csv`
- `winner_student_1d.json`

---

## 5. Backtest Motoru
### 5.1 Rebalance
- Haftalık (W-MON)

### 5.2 İşlem Maliyetleri
- Transaction cost: 10 bps

### 5.3 ML Backtest
- ML tahmin ağırlıkları kullanılır
- Combo-conditional ağırlıklar zamana ve combo’ya bağlıdır

---

## 6. Karşılaştırma ve Metrikler
### 6.1 Performans Metrikleri
- Sharpe (annualized, 365)
- Annual return
- Volatility

### 6.2 In-sample Örnek Sonuç
Student (1D, MVSK):
- Sharpe: 1.1532
- Annual Return: 65.49%
- Volatility: 56.79%
Teacher (1D, MVSK):
- Sharpe: 1.1351

### 6.3 OOS Holdout (0.30)
- Student Sharpe: 2.1237
- Teacher Sharpe: 2.1189

### 6.4 OOS Holdout (0.25)
- Student Sharpe: 3.0994
- Teacher Sharpe: 2.9262
- Not: OOS %25 döneminde Student ağırlıkları 1/3-1/3-1/3 sabit, Teacher ağırlıkları ~0.50/0.50 sabit kaldı (çok düşük varyans).

### 6.5 ML-Only Kombolar (limit-ml-combos)
- ML tahmini olan combo sayısı: 212.
- Student in-sample Sharpe: 0.6441 (winner: BCHBTC_DOGEBTC_SOLBTC_STXBTC_XRPBTC, MVSK).
- Student OOS (%25) Sharpe: 1.7069 (winner: AAVEBTC_BNBBTC_LTCBTC_SOLBTC_XRPBTC, MVSK).
- Not: Bu koşuda yalnızca ML tahmini olan combo’lar kullanıldığı için fallback optimizasyon yoktur; önceki yüksek OOS skorların bir kısmı fallback etkisi içerebilir.

---

## 7. PGP Neden Kullanılmadı?
- PGP ayrı bir modelleme paradigmasıdır; goal-deviation minimizasyonu ister.
- Projenin önceliği teacher-student ML yaklaşımını tamamlamak olduğu için PGP bu sürümde uygulanmadı.
- PGP ileride eklenebilir (geliştirme önerisi).
 - Teknik zorluk: PGP’de hedef sapmaları için uygun ölçekleme ve hedef seviyeleri gerekir; bu seviyeler yüksek momentler (skew/kurt) için stabil kalibrasyon ister.
 - Ek olarak PGP, mevcut MVSK çözümünden farklı bir amaç fonksiyonu kurar; bu da öğretmen-öğrenci boru hattını yeniden hizalamayı gerektirir.
 - Zaman kısıtı altında, PGP’nin sağlam parametrizasyonu ve doğrulaması ayrı bir çalışma alanı olduğundan bu sürümde kapsam dışı bırakılmıştır.

---

## 8. Deneysel Süreç ve Düzeltmeler
- ML ağırlıkları backtestte başlangıçta kullanılmıyordu → düzeltildi.
- Multi-output XGB’nin combo bağımsız ağırlık üretmesi sorunu → combo-conditional fix ile giderildi.
- OOS split sonucunda stabilite ihtiyacı → walk-forward önerildi.

---

## 9. Geliştirme Önerileri
- Walk-forward doğrulama (train/test penceresi ileri kaydırma).
- 1H frekansında aynı pipeline’ın uygulanması.
- PGP entegrasyonu.
- Lag ve target seçimi hassasiyet analizi.

---

## 10. Dosya Haritası
- Teacher: `results/pipeline/teacher_1d.parquet`
- Student: `results/pipeline/student_1d.parquet`
- Rankings: `results/pipeline/student_ranking_1d.csv`
- Winners: `results/pipeline/winner_student_1d.json`
- OOS: `results/pipeline/teacher_vs_student_oos_1d.json`

---

## 11. Kapanış
Bu proje, MVSK tabanlı klasik optimizasyon ile denetimli öğrenmenin birlikte kullanılabileceğini ve combo-conditional ağırlık tahmininin daha gerçekçi sonuçlar ürettiğini göstermiştir. In-sample ve OOS sonuçları umut verici olmakla birlikte, daha kapsamlı genelleme testleri (walk-forward) önerilmektedir.

## Ek: Kazanan Portföy Ağırlıkları
- Ayrıntılı ağırlık tabloları: `docs/weights_summary.md`
