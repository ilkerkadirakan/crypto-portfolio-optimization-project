# 2209-A Sonuç Raporu Taslağı

## 1. Giriş
Bu proje, kripto para piyasalarının yüksek volatilite ve asimetrik getiri dağılımlarını dikkate alarak portföy optimizasyonunu geliştirmeyi amaçlamaktadır. Geleneksel Ortalama‑Varyans (MV) yaklaşımının yetersiz kaldığı durumlarda, Çarpıklık ve Basıklık gibi yüksek momentleri içeren MVSK çerçevesi daha gerçekçi bir risk ölçütlendirmesi sunar. Çalışmada MV, MVSK ve MCVaRSK modelleri klasik optimizasyon (Teacher) olarak ele alınmış; bu çıktılar denetimli öğrenme (Student) ile tahmin edilerek performans karşılaştırması yapılmıştır.

## 2. Rapor Dönemlerinde Yapılan Çalışmalar
- **Veri hazırlama:** 36 aylık yüksek frekanslı kripto fiyat verileri 1H ve 1D frekanslarına indirgenmiş, log getiriler hesaplanmıştır.
- **Moment hesaplama:** 180 günlük rolling pencereyle ortalama, varyans (Ledoit‑Wolf), çarpıklık, basıklık ve CVaR momentleri üretilmiştir.
- **Teacher optimizasyonu:** MV, MVSK ve MCVaRSK modelleri ile tüm 2/3/5 varlık kombinasyonları (16.834 combo) optimize edilerek backtest yapılmıştır. Teacher sıralaması Sharpe oranına göre oluşturulmuştur.
- **Student modeli:** Teacher optimal ağırlıklarını hedefleyen multi‑output XGBoost modeli eğitilmiş, combo‑conditional özellikler kullanılarak her kombinasyon için ağırlık üretimi sağlanmıştır.
- **Backtest ve karşılaştırma:** Student ağırlıkları ile tüm kombinasyonlar backtest edilerek Student‑Teacher kıyaslaması yapılmıştır.
- **OOS (Holdout %30):** Genelleme performansı için son %30 dönem test olarak ayrılmıştır.
- **Raporlama ve görselleştirme:** Öğretmen‑öğrenci karşılaştırmaları, kazanan portföy ağırlıkları ve performans grafikleri üretilmiştir.

## 3. Sonuç
- In‑sample sonuçlarda Student, Teacher’a yakın veya sınırlı bir iyileşme göstermiştir.
- OOS (%30 holdout) değerlendirmede Student Sharpe oranı Teacher’dan hafif yüksek bulunmuştur; Teacher ise daha yüksek yıllık getiri üretmiştir.
- Combo‑conditional yaklaşım, her kombinasyon için farklı ağırlık üretimini sağlayarak daha gerçekçi bir yapı sunmuştur.

## 4. Çıktılar (Yayınlar, Sunumlar v.b.)
- Şu aşamada proje kapsamında resmi yayın veya konferans sunumu yapılmamıştır.
- Üretilen çıktılar:
  - Detaylı teknik rapor ve özet rapor
  - Öğretmen‑öğrenci karşılaştırma tabloları
  - Kazanan portföy ağırlık raporları
  - Performans ve Sharpe grafikleri

## 5. Proje ile İlgili Harcama Kalemleri
- Proje kapsamında herhangi bir bütçe kullanılmamış, harcama yapılmamıştır.
