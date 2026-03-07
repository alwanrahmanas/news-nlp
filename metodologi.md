# METODOLOGI PENELITIAN
## Sistem Peringatan Dini Triple-Modal Bayesian untuk Inflasi Pangan Bergejolak
### Kota Medan, Sumatera Utara, Indonesia

---

## 1. OVERVIEW PENELITIAN

| Item | Detail |
|---|---|
| **Judul** | Sistem Peringatan Dini Triple-Modal Bayesian untuk Inflasi Pangan Bergejolak |
| **Studi kasus** | Kota Medan, Sumatera Utara (populasi ~2.5 juta) |
| **Pendekatan** | Design science (proof-of-concept), bukan deployment operasional |
| **Periode data** | Januari 2021 – Februari 2026 (62 bulan, ~120 periode bi-mingguan) |
| **Frekuensi analisis** | Bi-mingguan (26 periode/tahun) |
| **Framework utama** | Bayesian Structural Time Series (BSTS) + Prior Spike-and-Slab |
| **Target variabel** | Inflasi Volatile Foods (VF) bi-mingguan Kota Medan |

---

## 2. VARIABEL TARGET

### 2.1 Konstruksi Inflasi VF Bi-Mingguan
- **Sumber primer**: Subindeks Volatile Foods BPS Kota Medan (bulanan, 12 komoditas)
- **Komoditas VF Inti** (8 komoditas, tersedia di PIHPS Pedagang Besar Kota Medan):
  - **Hortikultura rantai pasok Karo**: Cabai Merah, Cabai Rawit, Bawang Merah, Bawang Putih
  - **Protein hewani**: Daging Ayam, Telur Ayam, Daging Sapi
  - **Bahan pokok**: Beras (Kualitas Medium I)
- **Komoditas relevan sinyal CHIRPS** (tidak tersedia di PIHPS Pedagang Besar, namun
  dimasukkan sebagai keyword scraping berita karena dominan berasal dari Karo):
  - Tomat, Kentang
- **Catatan eksklusi**: Minyak Goreng dan Gula Pasir dikecualikan dari indeks VF karena
  termasuk *administered price* (diatur HET pemerintah), bukan volatile food murni.
  Kacang Kedelai dikecualikan karena didominasi impor (supply chain tidak berkaitan
  dengan rantai pasok Karo–Medan).
- **Disagregasi temporal** (bulanan → bi-mingguan):
  1. Bangun proksi harga grosir harian dari PIHPS **Pedagang Besar** Kota Medan
     (rata-rata geometrik berbobot IHK) — dipilih karena harga grosir mendahului
     harga eceran 1–3 hari, memberikan lead time tambahan untuk sistem peringatan.
  2. Agregasi ke rata-rata bi-mingguan
  3. Kalibrasi ke benchmark bulanan BPS via **prosedur Chow-Lin**
  4. Hitung perubahan persentase bi-mingguan sebagai `y_t`
- **Total observasi target**: N = ~120 periode (ukuran sampel kecil, small-sample regime)

---

## 3. SUMBER DATA (TIGA MODALITAS)

### 3.1 Modalitas 1 — Data Presipitasi Satelit CHIRPS
- **Sumber**: Climate Hazards Group InfraRed Precipitation with Station data (USGS FTP)
- **Resolusi spasial**: 0.05° (~5km), frekuensi: dekadal (10 hari)
- **Area target**: Kabupaten Karo, Sumatera Utara (sentra produksi utama hortikultura Medan)
  - Karo ≈ 45% pasokan cabai merah Medan
  - ~75 sel grid dalam batas administratif Karo (shapefile GADM)
- **Pipeline agregasi spasial**:
  1. Clipping raster ke batas administratif Karo
  2. Pembobotan lahan pertanian (Copernicus land cover 100m)
  3. Hitung weighted mean precipitation (statistik zonal)
  4. Agregasi dekadal → bi-mingguan via weighted-overlap averaging
- **Struktur lag**: L0, L1, L2, L3 (0–42 hari), plus dummy curah hujan ekstrem
  (`rain_extreme_t = 1` jika > persentil ke-85 historis)
- **Validasi**: Korelasi silang dengan rain gauge BMKG Kabanjahe
  (Stasiun 96195, elevasi 1.310m), target r ≥ 0.75

### 3.2 Modalitas 2 — Sentimen Berita Berbasis LLM

#### 3.2.1 Korpus Berita
- **Korpus berita**: 777 artikel (setelah pembersihan), periode Januari 2021 – Februari 2026
- **Sumber scraping** (7 sumber, dikembangkan via direct archive scraping + RSS fallback):

| Sumber | Artikel | Proporsi | Tipe |
|---|---|---|---|
| Google News RSS | 241 | 31.0% | Fallback (terkini) |
| Detik.com / Detik Sumut | 375 | 48.3% | Direct archive |
| Antara News / Antara Sumut | 161 | 20.7% | Direct archive |
| Kompas.com | — | — | Direct archive |
| TribunMedan / Tribunnews | — | — | Direct archive |
| Waspada.id | — | — | Direct archive (regional) |

> **Catatan**: Angka awal berdasarkan hasil pilot scraping dari 3 sumber pertama.
> Angka final akan diperbarui setelah pipeline 7 sumber berjalan penuh.

- **Strategi scraping**:
  - **Primary keywords** (16 keyword, geo-anchored ke Medan/Karo/Sumut):
    - Cabai: "cabai merah Medan", "cabai rawit Medan", "pasokan cabai Karo", "harga cabai Sumut"
    - Bawang: "bawang merah Medan", "bawang putih Medan", "pasokan bawang Karo"
    - Hortikultura Karo: "harga tomat Medan", "harga kentang Medan", "pasokan tomat Karo",
      "pasokan kentang Karo"
    - Protein: "harga ayam Medan", "harga telur Medan", "daging sapi Medan"
    - Beras: "harga beras Medan", "stok beras Sumut"
  - **Secondary keywords** (11 keyword, kontekstual):
    - "banjir Karo", "cuaca Berastagi", "inflasi pangan Medan", "TPID Medan",
      "gagal panen Sumut", "kelangkaan cabai", "permintaan Ramadan Medan", dll.
  - **Geo-filter** wajib: artikel harus menyebut minimal 1 dari 20+ term geografis
    (medan, karo, berastagi, sumut, deli serdang, pasar sambu, pasar aksara, dll.)
    agar lolos ke korpus utama. Artikel yang ditolak disimpan ke `corpus_rejected.jsonl`
    untuk audit.
  - **Length filter**: 500–30.000 karakter (~80–5.000 kata)

#### 3.2.2 Pipeline Ekstraksi LLM
- **Model**: GPT-5-mini (OpenAI, 2025) — dipilih karena kemampuan reasoning yang lebih
  baik dibanding GPT-4o-mini untuk klasifikasi zero-shot Bahasa Indonesia.
  > *Update dari proposal awal yang menggunakan GPT-4o-mini.*
- **Klasifikasi zero-shot** → 4 label:
  - `SUPPLYSHOCK`: gangguan produksi/distribusi (364 artikel, 46.8%)
  - `DEMANDSHOCK`: lonjakan konsumsi (39 artikel, 5.0%)
  - `PRICEREPORT`: observasi harga reaktif (150 artikel, 19.3%)
  - `IRRELEVANT`: tidak relevan (224 artikel, 28.8%)
- **Skor sentimen**: rentang [-1, +1]
- **Confidence score** per artikel (mean = 0.866)

#### 3.2.3 Validasi Kualitas
- 3 annotator manusia independen, sampel 200 artikel terstratifikasi
- Target Cohen's κ ≥ 0.7 dan F1-macro ≥ 0.75
- **Dekomposisi berita** (100 sampel manual):
  - **Tipe A**: prediktif, supply-driven (forward-looking) → kandidat Granger causality
  - **Tipe B**: reaktif, price-driven (backward-looking) → hanya korelasi kontemporer

#### 3.2.4 Agregasi Bi-Mingguan
- Indeks sentimen tertimbang confidence dengan fungsi peluruhan eksponensial (λ = 0.3):
  - `NSSSUP_t`: indeks sentimen supply shock
  - `NSSDMD_t`: indeks sentimen demand shock
- **Penanganan periode kosong**:
  - Jika `n_relevant = 0` → imputasi **nilai netral (0)**
  - Dummy `no_relevant_news_t = 1{n_relevant=0}` sebagai kovariat kontrol

### 3.3 Modalitas 3 — Harga Grosir PIHPS Frekuensi Tinggi
- **Sumber**: Pusat Informasi Harga Pangan Strategis (Kemendag), via MOU dengan
  Bank Indonesia Provinsi Sumut (MOU BI-Sumut-02/2026)
- **Jenis pasar**: **Pedagang Besar** (bukan Pasar Tradisional/Modern) — dipilih karena:
  - Harga grosir mendahului harga eceran 1–3 hari (tambahan lead time)
  - Intervensi TPID (buffer stock release, koordinasi impor) beroperasi di level grosir
  - Sinyal lebih tajam terhadap guncangan pasokan dibanding harga eceran
- **Cakupan**: Harga harian, 8 komoditas VF inti, semua pasar grosir Kota Medan
- **Konstruksi indeks**:
  - Rata-rata geometrik berbobot IHK
  - Agregasi ke bi-mingguan → hitung perubahan persentase
  - Lag yang diuji: L1 dan L2
- **Quality control**: Interpolasi linier untuk ~5% data harian hilang, outlier flag jika
  perubahan harian > 50%, cek konsistensi korelasi antar pasar

### 3.4 Indikator Musiman (Kontrol)
- **Dummy Ramadan/Idul Fitri**: biner, 2 minggu pra-Lebaran termasuk (permintaan +20–40%)
- **Dummy Natal/Tahun Baru**: 15 Desember – 7 Januari

---

## 4. SPESIFIKASI MODEL BSTS

### 4.1 Formulasi State-Space

**Persamaan observasi:**
```
y_t = μ_t + γ_t + β'X_t + ε_t,    ε_t ~ N(0, σ²_ε)
```

**Persamaan transisi state (local linear trend):**
```
μ_t+1 = μ_t + δ_t + η_t,          η_t ~ N(0, σ²_η)
δ_t+1 = δ_t + ζ_t,                ζ_t ~ N(0, σ²_ζ)
```

**Komponen musiman:**
```
γ_t: komponen musiman, siklus S = 26 periode (bi-mingguan per tahun)
```

**Notasi:**
- `y_t` : inflasi VF bi-mingguan (target)
- `μ_t` : tren linier lokal (local linear trend)
- `γ_t` : komponen musiman (S = 26 periode)
- `X_t` : matriks prediktor (lag curah hujan, sentimen berita, harga grosir, dummy musiman)
- `β`   : vektor koefisien (diestimasi dengan prior Spike-and-Slab)
- `ε_t` : error observasi

### 4.2 Prior Spike-and-Slab

- **Tujuan**: seleksi variabel otomatis dalam setting small-sample (N ≈ 120, p > 0.05·N)
- **Mekanisme**: prior menempatkan massa diskrit pada nol (spike) dan massa difus kontinu
  pada nilai non-nol (slab)
- **Prior probabilitas inklusi**: `π_j ~ Uniform(0,1)` atau informatif
- **Varians slab**: dikalibrasi terhadap ekspektasi signal-to-noise (Scott & Varian, 2014)
- **Varians lain**: `σ² ~ InverseGamma(0.001, 0.001)` (weakly informative)
- **Implementasi**: paket `bsts` di R (Scott, 2024)

### 4.3 Inferensi Posterior (MCMC)

- **Algoritma**: Gibbs sampler + Forward-Filtering Backward-Sampling (FFBS) untuk state laten
- **Iterasi**: 60.000 total (10.000 burn-in + 50.000 produksi, thinning setiap ke-5)
- **Chains**: 3 rantai paralel
- **Diagnostik konvergensi**: Gelman-Rubin statistic (target R̂ < 1.1) dan Effective
  Sample Size (ESS) untuk parameter utama

### 4.4 Posterior Inclusion Probability (PIP)

Interpretasi PIP mengikuti threshold Scott & Varian (2013):
- `PIP ≥ 0.5` : bukti **kuat** (variabel dimasukkan ke model)
- `0.1 ≤ PIP < 0.5` : bukti **moderat**
- `PIP < 0.1` : bukti **lemah** (di-spike / diabaikan)

---

## 5. ANALISIS SENSITIVITAS PRIOR

- **Variasi parameter**: 15 konfigurasi (variasikan π dan σ²_slab dalam rentang yang masuk akal)
- **Kriteria robustness** (minimal 3 dari 3 harus terpenuhi):
  1. Korelasi peringkat Spearman urutan PIP: target ρ ≥ 0.9
  2. Stabilitas akurasi peramalan: variasi MAE ≤ 10%
  3. Stabilitas koefisien 5 prediktor teratas: rasio range/mean ≤ 0.2
- **Konsekuensi gagal**: derajat klaim diturunkan dari "conclusive" menjadi "suggestive"

---

## 6. STUDI ABLASI (M1–M5)

### 6.1 Spesifikasi Lima Model Bersarang

| Model | Komponen | Tujuan |
|---|---|---|
| **M1** | Tren + musiman + dummy hari raya | Baseline naif |
| **M2** | M1 + lag harga grosir PIHPS | Nilai tambah harga frekuensi tinggi |
| **M3** | M2 + indeks sentimen berita LLM | Nilai tambah berita |
| **M4** | M3 + curah hujan satelit CHIRPS | Model lengkap triple-modal |
| **M5** | M2 + CHIRPS tanpa berita | Alternatif: satelit tanpa berita |

### 6.2 Hipotesis Ablasi

| Perbandingan | Hipotesis | Target |
|---|---|---|
| M2 vs M1 | PIHPS mengurangi MAE | 20–25% |
| M3 vs M2 | Berita mengurangi MAE tambahan | 10–15% |
| M4 vs M3 | Satelit mengurangi MAE tambahan | 5–10% |
| M4 vs M1 | Sistem penuh mengurangi MAE total | 25–30% |

**Transparansi hasil nol**: Jika kontribusi satelit/berita < 5% MAE dan tidak signifikan
secara statistik, simpulkan investasi tidak terbenarkan biayanya → rekomendasikan fokus
ke PIHPS saja.

### 6.3 Prosedur Evaluasi Out-of-Sample

- **Metode**: Expanding window cross-validation
- **Training**: periode 1 hingga t
- **Forecast**: 2 periode ke depan (4 minggu)
- **Evaluasi window**: periode 79–104 (Januari–Desember 2024)
- **Metrik**:
  - MAE (Mean Absolute Error)
  - MAPE (Mean Absolute Percentage Error)
  - CRPS (Continuous Ranked Probability Score) — kualitas distribusi prediktif
  - PIT uniformity test (kalibrasi probabilistik)
- **Uji signifikansi**: Diebold-Mariano test dengan koreksi Bonferroni (5 perbandingan)
- **Benchmark**: SARIMA naif, AR(4), ETS (auto), heuristik "alert jika kenaikan PIHPS > 10%"

---

## 7. VALIDASI KAUSALITAS GRANGER

### 7.1 Uji Granger Causality
- **Model**: VAR bivariat
  ```
  H0: variabel X tidak Granger-menyebabkan y_t
  ```
- **Uji**: F-test, lag optimal via AIC
- **Dijalankan terpisah untuk**:
  1. Sentimen berita agregat
  2. Sentimen Tipe A (prediktif, supply-driven)
  3. Sentimen Tipe B (reaktif, price-driven)
  4. Curah hujan satelit CHIRPS
  5. Harga grosir PIHPS

### 7.2 Cross-Correlation Analysis
- Hitung `Corr(X_{t+k}, y_t)` untuk lag k = -4 hingga +4
- Puncak korelasi pada lag negatif (k < 0) = sinyal prediktif X terhadap y

### 7.3 Hipotesis Kausal yang Diuji
- Berita agregat: Granger marginal (p = 0.05–0.10, karena kontaminasi Tipe B)
- Tipe A supply-driven: Granger robust (p < 0.05, lag 1–2)
- Tipe B price-driven: hanya korelasi kontemporer, lag prediktif tidak signifikan
- CHIRPS: Granger signifikan pada lag 2–4 (10–21 hari)

---

## 8. SISTEM ALERT OPERASIONAL (MULTI-TIER)

### 8.1 Definisi Threshold

Berdasarkan distribusi prediktif posterior `P(y_{t+1} > θ | data_t)`:

| Level | Trigger | Ambang batas (θ) | Tindakan Rekomendasi |
|---|---|---|---|
| 🔴 KRITIS | `P(y > θ_kritis)` ≥ 0.40 | Persentil ke-85 inflasi historis 2021–2023 | Lepas stok penyangga / impor darurat segera |
| 🟠 PERINGATAN | `P(y > θ_peringatan)` ≥ 0.50 | Persentil ke-75 | Aktivasi operasi siaga + intensifikasi pemantauan |
| 🟡 WASPADA | `P(y > θ_waspada)` ≥ 0.60 | Persentil ke-60 | Peningkatan surveilans + rapat koordinasi stakeholder |
| 🟢 NORMAL | Di bawah semua threshold | — | Surveilans rutin |

> **Catatan**: Threshold probabilitas 0.40–0.60 adalah titik awal (berdasarkan FEWS NET).
> Implementasi operasional memerlukan kalibrasi lokal bersama TPID Kota Medan.

### 8.2 Fungsi Loss Asimetris
- **Basis**: biaya kesejahteraan akibat krisis tidak terdeteksi >> biaya alarm palsu
- Misclassification cost krisis: ~Rp 675 miliar (dampak daya beli 2% inflasi)
- Biaya intervensi preventif: Rp 50–100 juta per episode
- **Implikasi**: sensitivitas tinggi diutamakan (toleransi false positive rate ≤ 25–35%)

### 8.3 Back-Testing Retrospektif 2024
- **Periode uji**: Januari–Desember 2024 (12 episode inflasi tinggi)
- **Episode yang divalidasi**: Ramadan 2024, Lebaran 2024, banjir Karo Februari 2024,
  pemogokan transportasi
- **Target performa**:
  - True Positive Rate (TPR): ≥ 70–80%
  - False Positive Rate (FPR): ≤ 25–35%
  - Median lead time: 16–20 hari
- **Optimasi threshold**: Analisis kurva ROC (Youden index atau cost-weighted utility)

---

## 9. PENANGANAN DATA KHUSUS

### 9.1 Coverage Berita & Sparse Periods
- Berita di 2021–2022 secara natural lebih sparse (arsip digital kurang ter-index)
- **Tidak dianggap missing y_t**: target (inflasi VF) tetap tersedia dari PIHPS+BPS
- **Imputasi kovariat berita**: `NSSSUP_t = 0, NSSDMD_t = 0` jika `n_relevant = 0`
- **Dummy coverage**: `no_relevant_news_t = 1{n_relevant=0}` — membedakan "netral karena
  tenang" vs "netral karena coverage buruk"
- **Coverage quality per periode**: simpan `n_total`, `n_relevant`, `n_supplyshock`,
  `n_demandshock` sebagai diagnostik

### 9.2 Class Imbalance pada Corpus Berita
- DEMANDSHOCK hanya 39 artikel (5%) — tidak dipakai untuk training classifier
- Label digunakan sebagai **fitur agregat per periode**, bukan untuk klasifikasi individual
- Spike-slab akan menseleksi apakah komponen DEMANDSHOCK informatif atau tidak

### 9.3 Temporal Split
- **Training**: periode 1–78 (Januari 2021 – Desember 2023)
- **Validation/Test**: periode 79–104 (Januari–Desember 2024)
- **JANGAN** gunakan random split — ini data time series, random split menyebabkan
  data leakage

### 9.4 Forward-Looking vs Reactive
- **Tipe A (forward-looking)**: supply-driven, prediktif, cocok untuk Granger test
  dan fitur BSTS
- **Tipe B (reactive/historical)**: price-driven, korelasi kontemporer, tidak prediktif
- Dekomposisi manual pada 100 sampel untuk validasi

---

## 10. STRUKTUR FITUR LENGKAP (MATRIKS X_t)

| Kelompok | Variabel | Lag | Keterangan |
|---|---|---|---|
| **PIHPS** | `pihps_L1`, `pihps_L2` | L1, L2 | Perubahan persen harga grosir Pedagang Besar |
| **CHIRPS** | `rain_L0`, `rain_L1`, `rain_L2`, `rain_L3` | L0–L3 | Curah hujan Karo (mm/periode) |
| **CHIRPS** | `rain_extreme_t` | L0–L2 | Dummy > persentil ke-85 historis |
| **Berita** | `NSSSUP_t` | L1, L2 | Indeks sentimen supply shock |
| **Berita** | `NSSDMD_t` | L1, L2 | Indeks sentimen demand shock |
| **Berita** | `no_relevant_news_t` | t | Dummy coverage: 1 jika tidak ada artikel relevan |
| **Musiman** | `ramadan_t`, `lebaran_t` | t | Dummy periode Ramadan/Lebaran |
| **Musiman** | `nataru_t` | t | Dummy Natal–Tahun Baru (15 Des–7 Jan) |

> **Seleksi variabel**: Semua fitur masuk ke dalam spike-slab secara bersamaan.
> PIP menentukan fitur mana yang dimasukkan ke model akhir.

---

## 11. STACK TEKNOLOGI

| Komponen | Tools |
|---|---|
| **Pemodelan BSTS** | R, paket `bsts` (Scott, 2024) |
| **Agregasi spasial CHIRPS** | Python, `rasterio`, `geopandas`, `rasterstats` |
| **Scraping berita** | Python, `httpx`, `BeautifulSoup`, `trafilatura` |
| **Ekstraksi sentimen** | Python, OpenAI API (GPT-5-mini, 2025) |
| **Pipeline data** | Python (`pandas`, `numpy`) |
| **Visualisasi** | R (`ggplot2`), Python (`matplotlib`, `plotly`) |
| **Validasi & evaluasi** | R (`scoringRules` untuk CRPS, `MCS` untuk DM test) |

---

## 12. STRUKTUR OUTPUT YANG DIHARAPKAN

```
output/
├── 01_data/
│   ├── chirps_biweekly_karo.csv          # Curah hujan bi-mingguan Karo
│   ├── pihps_biweekly_medan.csv          # Harga grosir bi-mingguan Medan
│   ├── news_sentiment_biweekly.csv       # Indeks sentimen bi-mingguan
│   └── vf_inflation_biweekly.csv         # Target: inflasi VF bi-mingguan
│
├── 02_validation/
│   ├── granger_results.csv               # Hasil uji Granger per variabel
│   ├── prior_sensitivity_15configs.csv   # Hasil 15 konfigurasi prior
│   └── llm_human_agreement.csv           # Cohen's κ, F1-score
│
├── 03_models/
│   ├── M1_baseline/
│   ├── M2_pihps/
│   ├── M3_news/
│   ├── M4_full/
│   └── M5_satellite/
│
├── 04_ablation/
│   ├── expanding_window_forecasts.csv    # Forecast per window per model
│   ├── mae_mape_crps_comparison.csv      # Perbandingan metrik semua model
│   └── diebold_mariano_results.csv       # Hasil DM test dengan Bonferroni
│
└── 05_alert_system/
    ├── backtesting_2024_results.csv      # TPR, FPR, lead time per episode
    ├── roc_curve_thresholds.csv          # Analisis ROC
    └── alert_timeline_2024.png           # Visualisasi alert vs realisasi
```

---

## 13. REFERENSI KUNCI

- Scott, S.L. & Varian, H.R. (2014). *Predicting the present with Bayesian structural time series.*
- George, E.I. & McCulloch, R.E. (1993). *Variable selection via Gibbs sampling.*
- Kwon, S. et al. (2025). *LLM-extracted news sentiment and food inflation forecasting.*
- Funk, C. et al. (2015). *The climate hazards infrared precipitation with stations (CHIRPS) dataset.*
- Depaoli, S. & van de Schoot, R. (2017). *Improving transparency and replication in Bayesian statistics.*
- Cho, S. et al. (2025). *LLM-Bayesian state space model (LBS) for multimodal time series.*
- Diebold, F.X. & Mariano, R.S. (1995). *Comparing predictive accuracy.*
- Chow, G.C. & Lin, A.L. (1971). *Best linear unbiased interpolation, distribution, and extrapolation of time series.*

---

## 14. CHANGELOG METODOLOGI

| Versi | Tanggal | Perubahan |
|---|---|---|
| 1.0 | Feb 2026 | Draft awal (proposal) |
| 1.1 | Mar 2026 | Tambah Tomat & Kentang ke keyword scraping (rantai pasok Karo) |
| 1.1 | Mar 2026 | Eksklusi eksplisit Minyak Goreng, Gula Pasir, Kacang Kedelai dari indeks VF |
| 1.1 | Mar 2026 | Spesifikasi PIHPS Pedagang Besar (bukan Pasar Tradisional) + justifikasi |
| 1.1 | Mar 2026 | Sumber scraping diperluas dari 3 → 7 (tambah Waspada.id, Kompas, TribunMedan) |
| 1.1 | Mar 2026 | Geo-filter wajib ditambahkan ke pipeline scraping (20+ term geografis) |
| 1.1 | Mar 2026 | Length filter: 500–30.000 karakter (koreksi dari 100–5.000 yang keliru) |
| 1.1 | Mar 2026 | Model LLM diperbarui: GPT-4o-mini → GPT-5-mini (reasoning lebih kuat) |

---

*File ini adalah panduan implementasi teknis yang hidup (living document).*
*Versi: 1.1 | Tanggal: Maret 2026*
*Sumber: Proposal Penelitian + Iterasi Implementasi Pipeline*
