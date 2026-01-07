# ML Model Denemeleri Raporu (TR)

## 1. Kapsam ve Amac
Bu rapor, Student (ML) tarafinda denenen tum model yaklasimlarini, elde edilen bulgulari ve pratik sorunlari ozetler. Amac, hangi modelin ne kadar tutarli sonuc verdigini ve hangi sorunlarla karsilasildigini net sekilde belgelemektir.

## 2. Denenen Modeller ve Yaklasimlar

### 2.1. Single XGBoost (Per-Asset)
- Her varlik icin ayri model ile agirlik tahmini.
- In-sample performans tekrar tekrar Sharpe ~0.6917 civarinda plato yapti.
- Avantaj: Hızli ve basit.
- Dezavantaj: Teacher seviyesini yakalayamadi.

### 2.2. Multi-Output XGBoost (Tek Model, Tum Agirliklar)
- Tek model tum varlik agirliklarini ayni anda tahmin etti.
- Combo-conditional ozellikler yokken ayni agirliklar tum kombinasyonlara uygulandi (gecersiz cesitlilik).
- Bu nedenle bu surum dogrudan raporlama icin uygun degildi; sonradan combo-conditional fix eklendi.

### 2.3. Combo-Conditional XGBoost (Full Run)
- Combo bilgisi, "combo_has_<asset>" indikatorleri ile modele verildi.
- In-sample Student winner:
  - Combo: ETCBTC_ICPBTC_LTCBTC_SOLBTC_STXBTC
  - Sharpe: 1.1532
- OOS (holdout %30):
  - Student: Sharpe 2.1237
  - Teacher: Sharpe 2.1189
- OOS (holdout %25):
  - Student: Sharpe 3.0994
  - Teacher: Sharpe 2.9262
- Not: Bu OOS kosularinda fallback optimizasyonun etkisi oldugu icin sonradan ML-only test gerekli goruldu.

### 2.4. ML-Only Kombolar (limit-ml-combos)
- ML tahmini uretilmis combo sayisi: 212 (tumu degil).
- In-sample Student winner:
  - Combo: BCHBTC_DOGEBTC_SOLBTC_STXBTC_XRPBTC
  - Sharpe: 0.6441
- OOS (%25) Student winner:
  - Combo: AAVEBTC_BNBBTC_LTCBTC_SOLBTC_XRPBTC
  - Sharpe: 1.7069
- Not: Bu kosu fallback optimizasyonu kullanmadigi icin gercek ML performansini gosterir.

### 2.5. CatBoost
- CatBoost ile deneme yapildi.
- Teacher hedef agirliklari bazi varliklarda sabit oldugu icin egitimde sorunlar yasandi.
- Full run performansi, single XGB seviyesinin altinda veya benzer seviyede kaldi.

### 2.6. Ensemble (LGBM + XGB + RF)
- Ensemble secenegi kodda mevcut.
- Bu kosu icin tam full run raporu cikartilmadi.
- Not: Dilersen yeniden calistirip bu rapora net metrik ekleyebiliriz.

## 3. Genel Degerlendirme
- Kombinasyon bazli farklilik uretmeyen modeller rapor icin kullanisli degil.
- Combo-conditional XGB en dogru mimari olarak one cikti.
- ML-only kosuda performans dusuk kaldi; bu, modelin teacher agirliklarini tam taklit edemedigini gosterir.
- OOS skorlarin bir kismi fallback optimizasyon etkisi tasiyordu; bu nedenle ML-only raporu kritik oldu.

## 4. Sonraki Adimlar
- On-the-fly ML (combo + zaman bazli dogrudan tahmin) ile tam kapsama test.
- Ensemble kosusunu tam run ile tamamlayip net metrik ekleme.
- Feature seti ve lag sayisini genisleterek underfit riski azaltma.
