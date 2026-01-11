# Soru-Cevap Raporu (Detayli)

Bu belge, rapor uzerindeki 6 soruya ayrintili ve teknik olarak savunulabilir yanitlari icerir.

---

## 1) ML-ONLY neden 212 kombinasyon? Bu bir oran mi?
Hayir, bir oran degil. ML-ONLY kosu, yalnizca **ML tahmini uretilmis kombinasyonlar** uzerinden calisir. Bu kombinasyonlar, egitim setini olustururken secilen top-K teacher kombinasyonlarindan gelir.

Bu projede egitim seti su sekilde sinirlandi:
- `top_k=300`: Teacher siralamasinda en iyi 300 combo alindi.
- `same_asset_count=True`: Teacher winner ile ayni varlik sayisina sahip combo'lar secildi.

Bu iki filtre (ve benzersiz kombolara indirgeme) sonucunda **212 benzersiz combo** kaldi. ML-only kosu, fallback optimizasyonu tamamen kapattigi icin **yalnizca bu 212 combo** uzerinden sonuc uretir.

Sonuc: 212 sayisi bir oran degil, **egitim filtresi + benzersiz kombinasyon sayisinin sonucu**dur.

---

## 2) Neden haftalik rebalance? Gunluk neden degil?
Haftalik rebalance secimi iki temel nedenle tercih edildi:

1. **Islem maliyeti kontrolu**  
   Gunluk rebalance daha sik al-sat anlamina gelir; bu da net getiriyi islem maliyeti nedeniyle dusurebilir.

2. **Gurultuyu azaltma**  
   Kripto getirileri cok oynak oldugu icin gunluk sinyallerde noise daha yuksektir. Haftalik rebalance, sinyali biraz daha stabilize eder.

Bu secim bir "trade-off"tur. Gunluk rebalance daha hizli tepki verir ama **maliyet + oynaklik artar**. Bilimsel olarak haftalik pencere, performans ve maliyet dengesi icin makul bir secimdir.  
Istenirse gunluk veya 2-gunluk denemeler ayrica karsilastirmaya eklenebilir.

---

## 3) Return hesaplarken islem maliyeti dikkate alindi mi?
Evet. Backtest motoru net getiriyi hesaplarken **transaction cost** uygular:
- `transaction_cost = turnover * 10 bps`
- Net getiri = `gross_return - transaction_cost`

Bu nedenle rapordaki getiri ve Sharpe metrikleri **net getiridir**, islem maliyeti dahildir.

---

## 4) Kumulatif return grafiklerinin farki neden buyuk?
Iki temel neden var:

1. **Farkli portfoy secimleri**  
   Teacher ve Student winner kombolari farklidir; bu nedenle kumulatif getiri egileri dogal olarak ayrisir.

2. **ML-only vs Full Combo farki**  
   ML-only kosu, fallback optimizasyon kullanmadigi icin daha dusuk ve "gercek ML" performansi gosterir.  
   Full combo kosuda ise ML tahmini olmayan combo'lar fallback optimizasyona duser ve performansi yukseltebilir.

Ek olarak, farkli OOS split tarihleri (0.25 vs 0.30) **baslangic noktalarini ve trendi** degistirir.  
Bu nedenle grafiklerin ayni olmasi beklenmez; farklar metodoloji farkindan kaynaklanir.

---

## 5) XGBoost kararlarini anlayabiliyor muyuz? (Neden %40 agirlik?)
XGBoost, kararlarin temelinde hangi ozelliklerin etkili oldugunu gosterir, ancak:
- **Tam ekonomik aciklama** vermez,
- **Karar gerekcesini anlatan kural** uretmez.

Yine de iki seviyede yorum yapilabilir:
1. **Feature importance / SHAP analizi**  
   Hangi momentlerin (mean/var/skew/kurt/cvar) veya lag'larin agirliklari belirlemede etkili oldugu gorulebilir.
2. **Karar profili**  
   Ornegin, yuksek momentum + dusuk volatilite olan varliklarin daha yuksek agirlik alabilmesi gibi kaliplar tespit edilebilir.

Sonuc: **kismen yorumlanabilir**, ama %40 agirlik verme sebebi tamamen "aciklanabilir" degildir.

---

## 6) Piyasa rejimine gore ayrisma yapilabilir mi?
Evet, bu cok mantikli bir sonraki asamadir.  
Olası yaklasim:
- Volatiliteye gore **high-vol / low-vol** rejimleri belirlenir,
- Her rejim icin **ayri model veya parametre seti** kullanilir,
- Rejim degistiginde model secimi degisir.

Bu, modelin **piyasa kosullarina adaptasyonunu** artirabilir ve daha istikrarlı performans uretmesini saglayabilir.  
Raporun "gelecek calisma" kisminda oneri olarak verilebilir.
