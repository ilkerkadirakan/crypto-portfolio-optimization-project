# Teknik Rapor (Detaylı)

## 1. Problem Tanımı
Kripto para getirileri yüksek volatiliteye sahiptir ve dağılım özellikleri çoğunlukla normal dağılım varsayımını ihlal eder. Bu nedenle sadece ortalama ve varyansa dayanan klasik Mean-Variance (MV) portföy optimizasyonu, kuyruk risklerini ve asimetrik getiri yapısını tam yansıtamaz. Çarpıklık (skewness) ve basıklık (kurtosis) gibi yüksek momentlerin modele dahil edilmesi, risk ölçümünü daha gerçekçi hale getirir. Bu çalışma, MVSK ve CVaR tabanlı klasik optimizasyon sonuçlarının denetimli öğrenme ile taklit edilip geliştirilmesi ve bunun portföy performansına etkisinin değerlendirilmesi problemine odaklanır.

## 2. Amaç
- MV/MVSK/MCVaRSK tabanlı klasik optimizasyon (Teacher) ile optimal ağırlıklar üretmek.
- Bu ağırlıkları denetimli öğrenme (Student) ile tahmin ederek performansı artırmak.
- Tüm kombinasyonlarda backtest yaparak Student ve Teacher performansını kıyaslamak.
- OOS (holdout %30) test ile genelleme kabiliyetini ölçmek.

## 3. Veri ve Ön İşleme
- Veri: 36 aylık yüksek frekanslı kripto fiyat serileri (20 varlık).
- Frekanslar: 1H ve 1D; analizde 1D esas alındı.
- Getiriler: log dönüşümlü getiriler (log returns).
- Rebalance: haftalık (W-MON).

## 4. Pencere Seçimi (180 Gün)
Pencere uzunluğu istatistiksel stabilite ve rejim adaptasyonu arasında denge sağlar:
- 60–90 gün: parametre kestirimleri gürültülü ve oynak olur.
- 365+ gün: rejim değişimlerine yavaş tepki verir.
- 180 gün: hem yeterli örnekleme hem de adaptasyon açısından dengeli bir tercih.

Bu nedenle 180 günlük rolling pencere kullanılmıştır.

## 5. Moment Hesaplama
Her varlık için rolling pencere üzerinden şu momentler hesaplanmıştır:
- Ortalama (μ)
- Varyans (Σ, Ledoit-Wolf shrinkage)
- Çarpıklık (skewness)
- Basıklık (kurtosis)
- CVaR (α = 0.95)

## 6. Teacher Model (Klasik Optimizasyon)
- Modeller: MV, MVSK, MCVaRSK (cvxpy ile çözülen konveks optimizasyon).
- Kombinasyonlar: 2/3/5 varlık (toplam 16,834 combo).
- Backtest: haftalık rebalance ile tam kombinasyon kapsamı.

**Teacher Winner (1D, MVSK)**
- Combo: AVAXBTC_ETHBTC_LTCBTC_SOLBTC_STXBTC
- Sharpe: 1.1351
- Annual Return: 61.29%
- Volatility: 53.99%

## 7. Student Model (Denetimli Öğrenme)
### 7.1. Model Yapısı
- Model: Multi-output XGBoost (tek model tüm ağırlıkları tahmin eder).
- Hedef: Teacher optimal ağırlıkları.
- Özellikler: lag’lı momentler + lag’lı getiriler (n_lags = 5).
- Combo-conditional: combo_has_<asset> göstergeleri ile her kombinasyon için farklı ağırlık üretimi.

### 7.2. Eğitim Seti
- Teacher ranking’den top-K (300) kombinasyon kullanıldı.
- same-asset-count filtresi ile aynı varlık sayısındaki kombinasyonlar seçildi.

### 7.3. PGP Neden Kullanılmadı?
PGP (Polynomial Goal Programming), çoklu hedef sapmaları minimizasyonuna dayanır ve MVSK gibi tek amaç fonksiyonundan farklıdır. Bu projede:
- Mevcut çözüm altyapısı MV/MVSK/MCVaRSK üzerine kuruluydu.
- PGP entegrasyonu ek hedef sapmaları ve ölçekleme parametreleri gerektirir.
- Öncelik, teacher-student ML yaklaşımının uygulanabilirliğini görmekti.

Bu nedenle PGP bu aşamada uygulanmamış, ileride eklenebilecek bir geliştirme olarak bırakılmıştır.

## 8. Backtest Sonuçları
### 8.1. In-Sample (Full Combo)
**Student Winner (1D, MVSK)**
- Combo: ETCBTC_ICPBTC_LTCBTC_SOLBTC_STXBTC
- Sharpe: 1.1532
- Annual Return: 65.49%
- Volatility: 56.79%

Student, Teacher’a kıyasla **sınırlı ama pozitif** iyileşme sağlamıştır.

### 8.2. OOS (Holdout %30)
Cutoff: 2024-04-02

**Student OOS Winner**
- Combo: AAVEBTC_DOGEBTC_XRPBTC
- Sharpe: 2.1237
- Annual Return: 106.49%
- Volatility: 50.14%

**Teacher OOS Winner**
- Combo: AAVEBTC_XRPBTC
- Sharpe: 2.1189
- Annual Return: 137.30%
- Volatility: 64.80%

Sonuçlar, Student’ın Sharpe açısından hafif üstün olduğunu, Teacher’ın ise daha yüksek getiri ürettiğini göstermektedir.

## 9. Değerlendirme
- Combo-conditional ML yaklaşımı, aynı ağırlıkların tüm kombinasyonlara uygulanması hatasını çözdü.
- Multi-output XGBoost, MVSK optimizasyonunu taklit edebilir ve bazı durumlarda Sharpe artışı sağlar.
- OOS %30 split sonuçları, Student’ın risk-getiri dengesinde avantaj sağlayabileceğini göstermektedir.

## 10. Geliştirme Önerileri
- Walk-forward doğrulama ile genelleme testi (train/test pencereleri ilerletilerek).
- Farklı lag sayıları ile parametre hassasiyet analizi.
- 1H (saatlik) frekansta aynı pipeline’ın uygulanması ve 1D sonuçlarıyla karşılaştırmalı analiz.
- PGP entegrasyonu ve MVSK-PGP karşılaştırması (gelecek çalışma).

## 11. Sonuç
Bu çalışma, kripto portföy optimizasyonunda MVSK temelli teacher‑student yaklaşımının uygulanabilirliğini göstermiştir. In-sample’da Student performansı Teacher’a yakın veya üstün, OOS’da ise Sharpe bakımından rekabetçi bulunmuştur. Böylece MVSK + denetimli öğrenme çerçevesi, volatil piyasalarda uygulanabilir bir alternatif olarak değerlendirilebilir.
