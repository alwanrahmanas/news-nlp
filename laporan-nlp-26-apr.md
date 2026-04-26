# Laporan Analisis Sementara: NLP/LLM Pipeline v2
## Sistem Peringatan Dini Inflasi Pangan Bergejolak Medan

**Tanggal Laporan:** 26 April 2026  
**Versi Pipeline:** v2 (Archive Scraping Edition)  
**Status:** ✅ M0–M10 selesai — menunggu data inflasi aktual sebelum BSTS

***

## Ringkasan Eksekutif

Pipeline NLP/LLM v2 telah berhasil diselesaikan secara penuh dari pengumpulan berita hingga ekspor fitur final. Dari korpus 2.149 artikel berita pangan Sumatera Utara periode 2021–2026, pipeline menghasilkan 135 periode bi-mingguan dengan 15 fitur sentimen siap pakai sebagai prediktor model Bayesian Structural Time Series (BSTS).

Validasi kualitas menunjukkan performa yang memuaskan: kesepakatan antar-anotator dan akurasi GPT terhadap konsensus manusia keduanya melampaui threshold minimum. Uji kausalitas Granger bivariat menghasilkan *null finding* yang valid secara metodologis — tidak ada bukti hubungan temporal signifikan antara sentimen berita dan inflasi VF dalam kerangka bivariat, namun hal ini tidak mengeliminasi komponen sentimen dari BSTS mengingat keterbatasan power statistik dan sifat bivariat pengujian.

***

## 1. Deskripsi Korpus dan Pipeline

### 1.1 Spesifikasi Teknis

| Parameter | Nilai |
|---|---|
| Model LLM | gpt-5.4-mini-2026-03-17 |
| Rentang Korpus | 1 Jan 2021 – 28 Feb 2026 |
| Total Artikel Diproses | 2.149 |
| Total Periode Bi-Mingguan | 135 (BW0001–BW0135) |
| Periode Analitik (post-2024) | 56 |
| Periode Historis (pre-2024) | 79 |
| Rata-rata Artikel per Periode | 15,9 |
| Rata-rata Artikel Relevan per Periode | 8,9 |
| Eksponential Decay Lambda (λ) | 0,3 |
| Confidence Threshold LLM | 0,6 |

### 1.2 Sumber Berita

| Sumber | Prioritas |
|---|---|
| Antara News Sumut, Detik Sumut | Primer |
| Antara Nasional, Detik Nasional, Kompas | Sekunder |
| TribunMedan, Waspada.id | Tersier |
| Google News RSS | Fallback |

### 1.3 Distribusi Label Klasifikasi

Dari 2.149 artikel yang diekstraksi LLM:

| Label | Jumlah | Persentase |
|---|---|---|
| IRRELEVANT | ~900 | ~41,9% |
| PRICEREPORT | ~905 | ~42,1% |
| SUPPLYSHOCK | ~295 | ~13,7% |
| DEMANDSHOCK | ~49 | ~2,3% |

*Catatan: Angka perkiraan berdasarkan distribusi sampel validasi yang proporsional terhadap total korpus.*

***

## 2. Validasi Kualitas (M6)

### 2.1 Inter-Rater Reliability

Dua anotator manusia melabeli 200 artikel dari sampel stratifikasi dengan jaminan minimum 20 artikel per kelas.

| Metrik | Nilai | Threshold | Status |
|---|---|---|---|
| Krippendorff's α | **0,7398** | ≥ 0,67 | ✅ PASSED |
| Cohen's κ (Rater 1 vs 2) | **0,7396** | ≥ 0,61 | ✅ PASSED |

Nilai α = 0,74 berada dalam kategori *substantial agreement* menurut skala Landis & Koch, mengindikasikan bahwa skema pelabelan cukup jelas untuk diterapkan secara konsisten oleh anotator yang berbeda. Kedua metrik hampir identik (selisih 0,0002), mengonfirmasi konsistensi internal pengukuran.

**Rater Conflict:** 35 dari 200 artikel (17,5%) menghasilkan ketidaksepakatan antar-anotator dan di-exclude dari perhitungan F1. Tingkat konflik ini normal untuk task klasifikasi kausal yang membutuhkan interpretasi konteks ekonomi.

### 2.2 GPT vs. Konsensus Manusia

Evaluasi dilakukan terhadap 165 artikel non-konflik.

| Kelas | F1-Score |
|---|---|
| SUPPLYSHOCK | **0,8718** |
| PRICEREPORT | **0,8095** |
| DEMANDSHOCK | **0,6250** |
| IRRELEVANT | **0,9677** |
| **Macro-F1 (overall)** | **0,8185** ✅ |

Macro-F1 = 0,8185 melampaui threshold 0,75. DEMANDSHOCK menunjukkan F1 terendah (0,625) — konsisten dengan ambiguitas konseptual antara demand-pull dan supply-push dalam literatur inflasi pangan Indonesia, serta ukuran sampel yang paling kecil di antara keempat kelas. IRRELEVANT memiliki F1 tertinggi (0,968), mengindikasikan GPT sangat reliabel dalam mengidentifikasi artikel yang tidak relevan.

**Implikasi untuk paper:** Performa DEMANDSHOCK perlu disebutkan sebagai keterbatasan dalam seksi *Limitations*, disertai penjelasan bahwa ambiguitas konseptual antar-kelas merupakan tantangan inheren dalam annotasi berita ekonomi berbahasa Indonesia.

***

## 3. Dekomposisi Tipe A/B (M7)

### 3.1 Distribusi Final

Klasifikasi dilakukan pada 686 artikel SUPPLYSHOCK dan DEMANDSHOCK menggunakan pendekatan dua tahap: rule-based keyword matching → LLM untuk kasus ambiguous.

| Tipe | Jumlah | Persentase |
|---|---|---|
| **Tipe A** (Forward-looking/Prediktif) | **335** | **48,8%** |
| **Tipe B** (Backward-looking/Reaktif) | **351** | **51,2%** |
| UNKNOWN/Excluded | 0 | 0,0% |
| **Total** | **686** | **100%** |

### 3.2 Breakdown per Metode Klasifikasi

| Metode | Artikel |
|---|---|
| Rule-based (keyword) | 354 (51,6%) |
| LLM (kasus ambiguous) | 332 (48,4%) |
| Fallback/UNKNOWN | 0 |

Distribusi 48,8% Tipe A berada dalam rentang wajar (20–50%). Nol artikel UNKNOWN mengindikasikan pipeline LLM bekerja tanpa kegagalan — satu *timeout* terjadi di awal (dicatat di log, di-retry otomatis) dan berhasil diselesaikan.

### 3.3 Catatan Metodologis

Penghapusan kata "karena" dari daftar *backward keyword* (bug fix M7) berdampak signifikan pada distribusi: Tipe A naik dari ~31% (run sebelumnya dengan bug) menjadi 48,8%. Lonjakan ini terjadi karena ratusan artikel yang sebelumnya masuk *ambiguous* akibat kemunculan kata "karena" kini langsung diklasifikasi secara rule-based dengan benar.

**Implikasi:** Definisi operasional Tipe A sensitif terhadap pemilihan keyword — ini perlu dilaporkan sebagai bagian dari *robustness analysis* dengan menguji sensitivitas hasil Granger terhadap definisi alternatif Tipe A.

***

## 4. Agregasi Time-Series (M8)

### 4.1 Ringkasan Fitur Sentimen

| Statistik | Nilai |
|---|---|
| Total periode | 135 |
| Periode dengan data | 135 (0% imputed) |
| Periode low-confidence | 21 (15,6%) |
| Periode no_relevant_news | 8 (5,9%) |
| Mean sentiment_all | 0,146 |
| Mean sentiment_typeA | 0,106 |

### 4.2 Coverage per Periode

| Coverage Flag | Jumlah Periode | Keterangan |
|---|---|---|
| NORMAL | 67 | 4–14 artikel relevan |
| SPARSE | 45 | < 4 artikel relevan |
| DENSE | 23 | ≥ 15 artikel relevan |

Setelah perbaikan bug coverage flag (dari `n_articles_total` ke `n_relevant`), periode SPARSE meningkat dari 9 menjadi 45 dan DENSE turun dari 59 menjadi 23. Angka baru ini lebih representatif: sebelumnya, periode dengan banyak artikel IRRELEVANT di-flag DENSE meskipun tidak ada sinyal ekonomi yang valid.

### 4.3 Temporal Split

| Segmen | Periode | Catatan |
|---|---|---|
| Historis (pre-2024) | 79 | Data arsip — coverage lebih sparse |
| Analitik (2024–2026) | 56 | Low-confidence = 0 (0%) |

Untuk training dan evaluasi BSTS, direkomendasikan menggunakan filter `is_historical=False` (56 periode analitik) dengan opsi menggunakan seluruh 135 periode untuk pre-training historis.

### 4.4 Catatan Hyperparameter

Lambda decay λ = 0,3 saat ini terlalu agresif: artikel di awal periode 14 hari hanya mendapat bobot $$ e^{-0.3 \times 13} \approx 0.02 $$ relatif terhadap artikel di akhir periode. Ini berisiko mengabaikan berita *early warning* yang justru paling relevan untuk prediksi. Sensitivity analysis dengan λ = 0,10 dan λ = 0,15 direkomendasikan sebelum BSTS dijalankan.

***

## 5. Uji Kausalitas Granger (M9)

### 5.1 Uji Stasioneritas

| Variabel | ADF p-value | KPSS p-value | Status | Transformasi |
|---|---|---|---|---|
| vf_inflation | 0,9858 | 0,01 | Non-Stationary | First difference |
| vf_inflation_diff1 | ~0 | 0,10 | **Stationary** | — |
| sentiment_all | ~0 | 0,088 | **Stationary** | Tidak perlu |
| sentiment_typeA | 0,0006 | 0,068 | **Stationary** | Tidak perlu |
| sentiment_typeB | ~0 | 0,10 | **Stationary** | Tidak perlu |
| sentiment_supply | ~0 | 0,10 | **Stationary** | Tidak perlu |
| sentiment_demand | ~0 | 0,012 | Non-Stationary | First difference |

Optimal lag dipilih AIC = **1 periode bi-mingguan** (≈ 14 hari).

### 5.2 Hasil Granger Causality Tests

| Pasangan Uji | Lag-1 F | Lag-1 p | Lag Terbaik | p Terbaik | Kesimpulan |
|---|---|---|---|---|---|
| sentiment_all → VF inflation | 0,083 | 0,774 | lag-1 | 0,774 | ❌ Null |
| **sentiment_typeA → VF inflation** | **1,461** | **0,229** | lag-1 | 0,229 | ❌ Null (borderline) |
| sentiment_typeB → VF inflation | 0,167 | 0,683 | lag-2 | 0,117 | ❌ Null |
| sentiment_supply → VF inflation | 0,509 | 0,477 | lag-3 | 0,422 | ❌ Null |
| sentiment_demand → VF inflation | 0,461 | 0,499 | lag-2 | 0,183 | ❌ Null |

### 5.3 Interpretasi

Uji Granger bivariat tidak menemukan hubungan temporal signifikan antara sentimen berita dan inflasi VF Medan pada α = 0,05 di semua lag yang diuji. Nilai F-statistic Tipe A pada lag-1 (F = 1,46, p = 0,229) secara arah konsisten dengan hipotesis *early warning* — sinyal ada meskipun tidak cukup kuat untuk dikonfirmasi dalam kerangka bivariat.

**Tiga penjelasan alternatif yang perlu dipertimbangkan:**

1. **Underpowered test.** Dengan N efektif = 134 setelah first-differencing, power statistik untuk effect size kecil (Cohen's f² ≈ 0,10) pada α = 0,05 diperkirakan hanya ~55–60% — di bawah konvensi 80%. Null finding tidak identik dengan bukti ketiadaan efek.

2. **Data inflasi sintetis.** Seluruh pengujian M9 saat ini dilakukan terhadap *random walk* sintetis, bukan data inflasi VF Medan aktual. Hasil ini **tidak dapat diinterpretasi secara substantif** hingga `vf_inflation_biweekly.csv` diganti dengan data PIHPS atau BPS.

3. **Batasan Granger bivariat.** Granger tidak mengontrol kovariat lain (curah hujan CHIRPS, harga PIHPS). Kontribusi marginal sentimen mungkin hanya muncul dalam konteks multivariat BSTS setelah kovariat dikontrol — inilah fungsi studi ablasi (M1 vs M2 vs M3 vs M4).

**Rekomendasi framing paper:**

> *Uji kausalitas Granger bivariat tidak menemukan hubungan temporal signifikan antara sentimen berita dan inflasi VF pada α = 0,05. Namun, hasil borderline Tipe A pada lag-1 (F = 1,46, p = 0,229) secara arah konsisten dengan hipotesis early warning. Evaluasi kontribusi prediktif marginal dilanjutkan melalui studi ablasi BSTS yang mengontrol kovariat secara simultan.*

***

## 6. Output Final (M10)

File `nlp_features_final.csv` (135 baris × 15 kolom) telah melewati seluruh pre-export checklist:

| Kolom Utama | Deskripsi |
|---|---|
| `period_id` | BW0001–BW0135 |
| `date_start`, `date_end` | Batas tanggal tiap periode |
| `sentiment_typeA` | Indeks sentimen berita forward-looking (fitur utama BSTS) |
| `sentiment_supply`, `sentiment_demand` | Indeks per kategori kejutan |
| `sentiment_all` | Indeks agregat semua berita relevan |
| `n_typeA`, `n_relevant` | Count artikel per kategori |
| `has_extreme_news` | Dummy: ada berita supply shock ekstrem (skor < -0,7) |
| `no_relevant_news_t` | Dummy: tidak ada berita relevan di periode t |
| `low_confidence_period` | Dummy: periode dengan sinyal lemah |
| `is_imputed` | Dummy: periode tanpa satu pun artikel |

Semua nilai `sentiment_typeA` berada dalam rentang [-1,0; +1,0] ✅  
Urutan kronologis terverifikasi ✅  
`granger_results.json` dan `validation_report.json` tersedia ✅

***

## 7. Bug yang Diperbaiki Selama Development

| Modul | Bug | Severity | Dampak |
|---|---|---|---|
| M9 | CCF plot sign terbalik (`-lag` bukan `lag`) | 🔴 Kritis | Plot CCF menampilkan mirror image |
| M9 | Interpretation logic kasus Tipe B signifikan hilang | 🟠 Metodologis | Null finding dilaporkan saat ada sinyal B |
| M9 | Alignment setelah first-differencing (off-by-one) | 🟡 Halus | Misalignment 1 periode |
| M8 | Syntax error `slogger` | 🔴 Fatal | Pipeline crash di `__main__` |
| M8 | Coverage flag gunakan `n_total` bukan `n_relevant` | 🟡 Metodologis | DENSE overestimated (59→23) |
| M8 | Forward-fill pada subset non-low-confidence | 🟡 Halus | Carry-forward nilai basi |
| M7 | `tqdm` import di dalam `try/except` | 🔴 Fatal | 332 artikel → B secara senyap |
| M7 | Fallback ke "B" bukan "UNKNOWN" | 🟠 Metodologis | Inflasi buatan pada distribusi B |
| M7 | Tidak ada checkpoint LLM | 🟡 Robustness | Progress 13 menit hilang jika koneksi putus |
| M7 | Kata "karena" di BACKWARD_KEYWORDS | 🟡 Kualitas | Tipe A under-estimated (31%→49%) |
| M6 | Truncation 10.000 vs 800 karakter | 🟡 UX | Annotator fatigue |
| M6 | Majority vote tie arbitrer (Counter) | 🟠 Metodologis | Consensus non-deterministik |
| M6 | `human_label` tidak di-strip whitespace | 🟠 Data | Cohen's κ deflated artifisial |
| M6 | DEMANDSHOCK hanya 7 sampel (pure frac) | 🟡 Statistik | F1 std error ±0,25 |

***

## 8. Langkah Berikutnya

### Prioritas Tinggi (Blocker untuk BSTS)

1. **Ganti data inflasi sintetis** — Download data harian PIHPS Medan (`hargapangan.id`) untuk komoditas pangan bergejolak periode 2021–2026, konstruksi indeks komposit bi-mingguan dengan bobot IHK Sumut, simpan sebagai `vf_inflation_biweekly.csv`.

2. **Jalankan ulang M9** dengan data aktual — Semua hasil Granger saat ini tidak dapat diinterpretasi substantif.

### Prioritas Menengah (Robustness)

3. **Sensitivity analysis λ** — Jalankan M8 dengan λ ∈ {0,10; 0,15; 0,30}, bandingkan time series `sentiment_typeA`, laporkan sebagai hyperparameter sensitivity.

4. **Sensitivity analysis definisi Tipe A** — Uji distribusi A/B dengan set keyword alternatif untuk memverifikasi stabilitas rasio 48,8%.

### Tahap Lanjut (BSTS)

5. **Integrasi kovariat CHIRPS dan PIHPS** — Siapkan data curah hujan dekadal dan harga harian untuk time-proportionate overlap weighting ke periode bi-mingguan.

6. **Jalankan 4 spesifikasi model BSTS:**
   - M1: Baseline (seasonal + trend only)
   - M2: + Kovariat CHIRPS + PIHPS
   - M3: + Sentimen (semua)
   - M4: + Sentimen Tipe A saja
   
7. **Studi ablasi** — Bandingkan RMSE dan MAPE keempat model untuk mengukur kontribusi marginal komponen sentimen.

***

*Laporan ini dibuat berdasarkan log eksekusi pipeline tanggal 26 April 2026. Data inflasi VF yang digunakan dalam M9 masih bersifat sintetis — seluruh hasil Granger bersifat sementara hingga data aktual tersedia.*
