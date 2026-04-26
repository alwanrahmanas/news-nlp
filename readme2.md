# 🗞️ NLP/LLM Pipeline v2 — Inflasi Pangan Bergejolak Medan

[
[
[
[

> **Komponen NLP** dari sistem *Multimodal Bayesian Early Warning for Volatile Food Inflation* (Medan, Sumatera Utara). Pipeline ini mengekstraksi sinyal sentimen berita dari arsip media daring Indonesia (2021–2026) sebagai salah satu modalitas dalam kerangka **Bayesian Structural Time Series (BSTS) triple-modal**.

***

## Daftar Isi

- [Gambaran Umum](#gambaran-umum)
- [Perbedaan vs v1](#perbedaan-vs-v1)
- [Arsitektur Pipeline](#arsitektur-pipeline)
- [Sumber Berita](#sumber-berita)
- [Persyaratan Sistem](#persyaratan-sistem)
- [Instalasi](#instalasi)
- [Konfigurasi](#konfigurasi)
- [Menjalankan Pipeline](#menjalankan-pipeline)
- [Output & Artefak](#output--artefak)
- [Status Validasi](#status-validasi)
- [Troubleshooting](#troubleshooting)
- [Referensi](#referensi)

***

## Gambaran Umum

Pipeline ini memproses corpus berita berbahasa Indonesia untuk menghasilkan **indeks sentimen bi-mingguan** yang merepresentasikan tekanan rantai pasok pangan di Medan. Sinyal ini kemudian digunakan sebagai prediktor leading dalam model BSTS bersama data curah hujan satelit CHIRPS dan harga grosir PIHPS.

```
Arsip Media Online
  (2021–2026)
       │
       ▼
 ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
 │  M1 Scrape  │───▶│ M2 Prepro.  │───▶│ M3 Prompt   │
 │  7 sumber   │    │  + Dedup.   │    │  Design     │
 └─────────────┘    └─────────────┘    └─────────────┘
                                              │
       ┌───────────────────────────────────── ▼
       │                              ┌─────────────┐
       │                              │  M4 LLM     │
       │                              │  Ekstraksi  │
       │                              └─────────────┘
       │                                     │
       ▼                                     ▼
 ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
 │  M6 Human   │◀───│  M5 Parsing │◀───│  M5 QC      │
 │  Validasi   │    │  & Schema   │    │  Checking   │
 └─────────────┘    └─────────────┘    └─────────────┘
       │
       ▼
 ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
 │ M7 Tipe A/B │───▶│ M8 Agregasi │───▶│ M9 Granger  │
 │ Dekomposisi │    │ Bi-mingguan │    │  Causality  │
 └─────────────┘    └─────────────┘    └─────────────┘
                                              │
                                              ▼
                                       ┌─────────────┐
                                       │ M10 Feature │
                                       │   Export    │
                                       └─────────────┘
                                              │
                                              ▼
                                  sentiment_features.csv
                                  (→ BSTS Model)
```

***

## Perbedaan vs v1

| Aspek | v1 (Google News RSS) | v2 (Archive Scraping) |
|---|---|---|
| **Sumber data** | Google News RSS fallback | Scraping langsung ke arsip Antara, Detik, Kompas, Tribun |
| **Jangkauan historis** | ~1 bulan terakhir | **5+ tahun (2021–2026)** |
| **Coverage** | ~33 artikel setelah filter | 2.000–5.000+ artikel |
| **Resumable** | ❌ | ✅ Checkpoint per-query |
| **Rate limiting** | Sederhana (global) | Per-domain, configurable |
| **Geo filter** | 6 term | 25+ variasi geo Sumut |
| **Type A/B decomp.** | Tidak ada | ✅ Rule-based + LLM fallback |
| **Granger test** | Tidak ada | ✅ Multi-variabel, auto-stationarity |

***

## Arsitektur Pipeline

Pipeline terdiri dari **11 modul** yang berjalan secara sekuensial. Setiap modul menghasilkan artefak yang menjadi input modul berikutnya.

| Modul | Nama | Input | Output | Catatan |
|---|---|---|---|---|
| **M0** | Setup & Struktur | — | `config.yaml`, struktur direktori | Dijalankan sekali |
| **M1** | Koleksi Berita | Keywords, config | `data/raw/*.jsonl` | ⏳ Bisa berjam-jam |
| **M2** | Preprocessing | Raw articles | `data/processed/articles_clean.parquet` | Deduplikasi, geo-filter |
| **M3** | Desain Prompt | — | `prompts/` | Template few-shot |
| **M4** | Ekstraksi LLM | Clean articles | `data/processed/extractions_raw.jsonl` | 💰 Membutuhkan API key |
| **M5** | Parsing & QC | Raw extractions | `data/processed/extractions_clean.parquet` | Validasi schema |
| **M6** | Validasi Manusia | Sample 200 artikel | `data/validation/validation_report.json` | ⏸ Pipeline pause di sini |
| **M7** | Dekomposisi A/B | Clean extractions | `data/processed/typeAB_labels.parquet` | Rule-based + LLM opsional |
| **M8** | Agregasi | Labels + sentiment | `data/processed/sentiment_features.csv` | Bi-mingguan, decay λ=0.3 |
| **M9** | Granger Test | Features + inflasi | `outputs/granger_results.json` | Auto stationarity check |
| **M10** | Feature Export | All processed | `outputs/final_features.csv` | Siap untuk BSTS |

### Label Klasifikasi

Setiap artikel diklasifikasikan ke dalam salah satu dari empat label:

| Label | Deskripsi | Relevansi |
|---|---|---|
| `SUPPLYSHOCK` | Gangguan produksi atau distribusi rantai pasok | ⭐ Prediktif (Tipe A) |
| `DEMANDSHOCK` | Lonjakan konsumsi (Ramadan, hari raya) | Prediktif terbatas |
| `PRICEREPORT` | Observasi harga reaktif, tanpa konten forward-looking | Hanya kontemporer |
| `IRRELEVANT` | Konten non-relevan | Difilter |

***

## Sumber Berita

| Sumber | URL Arsip | Prioritas | Keterangan |
|---|---|---|---|
| Antara Sumut | `sumut.antaranews.com/search` | 🟢 Primer | Berita regional resmi |
| Detik Sumut | `detik.com/sumut/search` | 🟢 Primer | Cakupan Sumut luas |
| Antara Nasional | `antaranews.com/search` | 🟢 Primer | Wire service nasional |
| Detik Nasional | `detik.com/search` | 🟡 Sekunder | Nasional, selektif |
| Kompas | `search.kompas.com/search` | 🟡 Sekunder | Kualitas editorial tinggi |
| TribunMedan | `tribunnews.com/search` | 🟠 Tersier | Sangat lokal |
| Waspada.id | `waspada.id/search` | 🟠 Tersier | Lokal Sumut |
| Google News RSS | `news.google.com/rss` | ⚪ Fallback | Jika sumber lain gagal |

***

## Persyaratan Sistem

- **Python** ≥ 3.10
- **RAM** ≥ 4 GB (disarankan 8 GB untuk corpus besar)
- **Koneksi internet** stabil untuk M1 (scraping) dan M4 (LLM)
- **OpenAI API key** aktif dengan saldo cukup

### Estimasi Biaya API (M4)

| Ukuran Corpus | Estimasi Artikel | Estimasi Biaya (GPT-5.4-mini) |
|---|---|---|
| Kecil | ~1.000 artikel | ~USD 2–5 |
| Normal | ~2.000–3.000 artikel | ~USD 5–15 |
| Besar | ~5.000 artikel | ~USD 15–30 |

> Biaya aktual tergantung panjang artikel dan jumlah token yang diproses.

***

## Instalasi

```bash
# 1. Clone atau masuk ke direktori proyek
cd nlp_pipeline_v2

# 2. Buat dan aktifkan virtual environment
python -m venv .venv

# Linux/Mac
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (CMD)
.venv\Scripts\activate.bat

# 3. Install dependencies
pip install -r requirements.txt

# 4. Salin dan isi konfigurasi API key
cp .env.example .env
# Edit .env dan isi: OPENAI_API_KEY=sk-...
```

***

## Konfigurasi

Seluruh konfigurasi pipeline terpusat di `config.yaml`. Parameter utama:

```yaml
# config.yaml (ringkasan parameter kritis)

pipeline:
  project: nlp_pipeline_v2_inflasi_medan
  corpus_period: ["2021-01-01", "2026-02-28"]

llm:
  model: gpt-5.4-mini-2026-03-17
  max_tokens: 10000
  batch_size: 20
  confidence_threshold: 0.6        # Threshold confidence score LLM

labels:
  - SUPPLYSHOCK
  - DEMANDSHOCK
  - PRICEREPORT
  - IRRELEVANT

aggregation:
  frequency: biweekly              # Frekuensi agregasi
  decay_lambda: 0.3                # Parameter peluruhan eksponensial

granger:
  max_lag: 4                       # Maksimum lag yang diuji
  alpha: 0.05                      # Tingkat signifikansi

scraper:
  sources: 8
  keywords:
    primary: 16                    # Kata kunci utama
    secondary: 11                  # Kata kunci sekunder
```

***

## Menjalankan Pipeline

### Opsi 1: Jalankan Penuh (Rekomendasi untuk pertama kali)

```bash
python run_pipeline.py --auto-confirm
```

### Opsi 2: Jalankan Modul Tertentu

```bash
# Hanya scraping (M0 + M1)
python run_pipeline.py --only m0 m1

# Mulai dari M2 (setelah scraping selesai)
python run_pipeline.py --from m2 --auto-confirm

# Jalankan satu modul spesifik
python src/m9_granger.py
```

### Opsi 3: Jalankan di Background (VPS/Server)

```bash
nohup python run_pipeline.py --auto-confirm > pipeline.log 2>&1 &

# Pantau progress
tail -f logs/pipeline.log

# Cek apakah masih berjalan
ps aux | grep run_pipeline
```

### ⏸ Pause Wajib di M6

Pipeline akan **otomatis berhenti** setelah M5 selesai dan menunggu anotasi manusia. Proses anotasi:

1. Buka file `data/validation/sample_200.jsonl`
2. Anotasi 200 artikel secara independen oleh ≥3 anotator
3. Simpan hasil ke format yang diharapkan
4. Jalankan ulang pipeline dari M6:

```bash
python run_pipeline.py --from m6 --auto-confirm
```

### Fitur Checkpoint (Resume M1)

Jika koneksi terputus saat scraping, **cukup jalankan ulang** tanpa menghapus apapun:

```bash
python run_pipeline.py --only m1
```

Pipeline melanjutkan dari query yang terakhir berhasil. Progress tersimpan di `data/raw/checkpoints/`.

> ⚠️ **Catatan**: M1 (scraping 8 sumber × 27 keyword × 30 halaman) membutuhkan **beberapa jam**. Disarankan jalankan di server/VPS dengan koneksi stabil.

***

## Output & Artefak

Setelah pipeline selesai, artefak utama tersimpan di:

```
outputs/
├── final_features.csv           # ← Input utama untuk model BSTS
├── granger_results.json         # Hasil uji kausalitas Granger
├── granger_plots/
│   ├── ccf_sentiment_all.png    # Cross-correlation function plot
│   ├── ccf_sentiment_typea.png
│   ├── ccf_sentiment_typeb.png
│   ├── ccf_sentiment_supply.png
│   └── ccf_sentiment_demand.png
data/
├── processed/
│   ├── sentiment_features.csv   # 135 periode bi-mingguan × 22 fitur
│   ├── typeAB_labels.parquet    # Dekomposisi Tipe A/B (2160 artikel)
│   └── extractions_clean.parquet
└── validation/
    └── validation_report.json   # Metrik validasi human vs GPT
```

### Skema `sentiment_features.csv` (22 kolom)

| Kolom | Deskripsi |
|---|---|
| `period_id` | Kode periode bi-mingguan (BW0001–BW0135) |
| `date_start`, `date_end` | Rentang tanggal periode |
| `sentiment_all` | Indeks sentimen seluruh artikel relevan |
| `sentiment_typeA` | Indeks sentimen Tipe A (supply-driven) |
| `sentiment_typeB` | Indeks sentimen Tipe B (price-driven) |
| `sentiment_supply` | Rata-rata sentimen label SUPPLYSHOCK |
| `sentiment_demand` | Rata-rata sentimen label DEMANDSHOCK |
| `n_articles` | Total artikel pada periode |
| `n_relevant` | Artikel relevan (bukan IRRELEVANT) |
| `coverage_flag` | `NORMAL` / `DENSE` / `SPARSE` |
| `low_confidence_period` | Boolean — flag periode sinyal rendah |
| `no_relevant_news_t` | Boolean — tidak ada artikel relevan |
| *(+10 kolom lainnya)* | Fitur tambahan dan metadata |

***

## Status Validasi

Hasil terkini dari run terakhir (2026-04-26):

### M6 — Human Validation

| Metrik | Nilai | Target | Status |
|---|---|---|---|
| Krippendorff's α (inter-rater) | 0.6468 | ≥ 0.70 | ⚠️ Marginal |
| Cohen's κ (rater1 vs rater2) | 0.6467 | ≥ 0.70 | ⚠️ Marginal |
| GPT vs Human macro-F1 | **0.7778** | ≥ 0.70 | ✅ Lolos |
| F1 — SUPPLYSHOCK | **0.8408** | ≥ 0.70 | ✅ Kuat |
| F1 — DEMANDSHOCK | 0.6400 | ≥ 0.70 | ❌ Di bawah target |
| F1 — PRICEREPORT | 0.7037 | ≥ 0.70 | ✅ Marginal |
| F1 — IRRELEVANT | **0.9268** | ≥ 0.70 | ✅ Sangat kuat |

> 💡 Rendahnya κ inter-rater dan F1 DEMANDSHOCK mengindikasikan batas semantik antara `DEMANDSHOCK` dan `PRICEREPORT` masih ambigu bagi anotator. Pertimbangkan perbaikan annotation guideline.

### M7 — Type A/B Decomposition

| Label | Jumlah | Proporsi | Catatan |
|---|---|---|---|
| Tipe A (supply-driven) | 178 | 16.7% | ⚠️ Di bawah 20% |
| Tipe B (price-driven) | 887 | 83.3% | Termasuk 291 ambiguous → B |
| **Total** | **1.065** | 100% | SUPPLY + DEMAND saja |

> ⚠️ Proporsi Tipe A rendah karena LLM dinonaktifkan untuk kasus ambiguous (291 artikel di-default ke B). Aktifkan LLM di M7 untuk klasifikasi yang lebih akurat.

### M9 — Granger Causality

| Variabel | Lag Terbaik | F-stat | p-value | Interpretasi |
|---|---|---|---|---|
| Semua berita → VF | lag-2 | 2.731 | 0.069 | ⚠️ Borderline (p<0.10) |
| Tipe A → VF | lag-1 | 0.003 | 0.958 | ❌ Tidak signifikan |
| Tipe B → VF | lag-2 | 1.568 | 0.213 | ❌ Tidak signifikan |
| SUPPLYSHOCK → VF | lag-3 | 2.500 | 0.063 | ⚠️ Borderline |
| DEMANDSHOCK → VF | lag-1 | 0.701 | 0.404 | ❌ Tidak signifikan |

> 📝 Hasil Granger yang tidak signifikan untuk Tipe A sebagian besar disebabkan oleh densitas sinyal rendah (rata-rata ~1.3 artikel Tipe A/periode). Ini adalah **valid null finding** yang akan dilaporkan transparan dalam paper.

***

## Troubleshooting

**Scraping terlalu lambat atau sering timeout**
```yaml
# config.yaml — kurangi concurrency atau tambah delay
scraper:
  delay_per_request: 3.0       # detik
  max_retries: 5
  timeout: 30
```

**M4 error: quota exceeded**
Periksa saldo OpenAI API. Kurangi `batch_size` di `config.yaml` atau gunakan model lebih kecil.

**M7: Tipe A terlalu sedikit (< 20%)**
Aktifkan LLM untuk klasifikasi kasus ambiguous:
```yaml
# config.yaml
type_ab:
  use_llm_for_ambiguous: true   # Default: false
```

**M9: inflasi NON-STATIONARY**
Pipeline menangani ini otomatis dengan first-difference. Jika ingin menguji level series, gunakan VAR dengan `trend='ct'`.

**Pipeline crash di tengah M1**
Jalankan ulang — checkpoint system akan melanjutkan dari query terakhir:
```bash
python run_pipeline.py --only m1
```

***

## Referensi

Penelitian ini merupakan bagian dari:

> **"A Multimodal Bayesian Framework for Early Warning of Volatile Food Inflation: Integrating Satellite Rainfall, LLM-Extracted News Sentiment, and High-Frequency Prices in Small-Sample Settings"**
> Universitas Sumatera Utara / [Institusi], 2026.

Metodologi NLP mengadaptasi kerangka dekomposisi sentimen dari:
- Kwon et al. (2025) — *LLM-based macroeconomic sentiment decomposition*
- Landis & Koch (1977) — *Inter-rater reliability thresholds (κ)*
- Granger (1969) — *Testing for causality*

***
***
Tambahan penjelasan mengenai pelabelan (berikan di C:\Users\US3R\OneDrive\Dokumen\data-science\Project\sumateranomics\nlp\nlp_pipeline_v2\data\validation\sample_200_for_annotation.xlsx)

Berikut adalah rancangan aturan pelabelan yang menggunakan kerangka kausalitas ekonomi makro untuk memisahkan pergeseran kurva penawaran, kurva permintaan, evaluasi statistik, dan distorsi data.

1. Kategori IRRELEVANT (Filter Kebisingan dan Distorsi)
Kategori ini adalah garis pertahanan pertama. Sebuah teks wajib dilabeli IRRELEVANT jika memenuhi minimal satu dari kondisi berikut:

Kebocoran Spasial: Kejadian, harga, atau gangguan logistik berlokasi di luar Provinsi Sumatera Utara. Data dari wilayah lain berisiko merusak pemetaan rantai pasok lokal.

Distorsi Harga Intervensi: Informasi utama memuat harga subsidi, diskon ritel modern, atau operasi pasar murah. Angka ini tidak mencerminkan ekuilibrium pasar organik dan akan memicu prediksi deflasi palsu pada model.

Retorika Politik dan Regulasi Normatif: Pernyataan pejabat mengenai imbauan, rencana, atau fokus pengendalian inflasi yang tidak disertai tindakan fisik atau data volume riil di lapangan.

Penyebutan Non Kausal: Teks memuat nama komoditas pangan bergejolak namun dalam konteks resep masakan, ulasan restoran, pakan hewan peliharaan, atau kurban individu.

2. Kategori PRICEREPORT (Evaluasi Retrospektif dan Reaktif)
Label ini digunakan khusus untuk teks yang memotret kondisi yang sudah terjadi, tanpa menyajikan variabel prediktif (indikator awal) untuk masa depan.

Laporan Harga Murni: Teks hanya menyatakan harga komoditas naik, turun, atau stabil pada titik waktu tertentu tanpa penjelasan kausalitas fisik yang mendasari pergerakan tersebut.

Rilis Statistik Resmi: Publikasi Indeks Harga Konsumen (IHK) atau Nilai Tukar Petani (NTP) dari otoritas statistik. Ini adalah rekapitulasi data historis bulan sebelumnya.

Evaluasi Stok Pasif: Pernyataan birokrasi bahwa ketersediaan barang "aman" atau "terkendali" tanpa adanya injeksi volume pasokan baru berskala masif ke dalam pasar.

3. Kategori SUPPLYSHOCK (Pergeseran Kurva Penawaran)
Label ini mengidentifikasi gangguan atau injeksi pada sisi ketersediaan fisik barang. Teks harus secara eksplisit memuat variabel fundamental yang mengubah volume komoditas:

Disrupsi Produksi dan Logistik: Gagal panen, penyakit hama, cuaca ekstrem, infrastruktur distribusi terputus, pungutan liar logistik, atau penjarahan fasilitas penyimpanan.

Momentum Produksi Mayor: Datangnya musim panen raya yang mengubah ekspektasi volume pasokan lokal secara drastis.

Injeksi Volume Intervensi: Penyaluran Cadangan Beras Pemerintah (CBP) atau program Stabilisasi Pasokan dan Harga Pangan (SPHP) oleh Bulog dalam skala ribuan ton. Fokusnya adalah pada penambahan ketersediaan barang secara fisik di pasar, bukan pada harga jual subsidi.

4. Kategori DEMANDSHOCK (Pergeseran Kurva Permintaan)
Aturan untuk kategori ini memerlukan pembuktian kausalitas yang sangat ketat untuk menghindari kerancuan dengan PRICEREPORT.

Bukti Penarikan Volume Tiba Tiba: Tidak dibenarkan melabeli teks sebagai DEMANDSHOCK hanya karena narasi "harga naik akibat Ramadhan". Teks wajib menyajikan informasi utama berupa perilaku konsumen yang menguras stok pasar. Contoh valid: "stok distributor menipis karena aksi borong warga" atau "pembelian massal oleh entitas politik".

Kejutan Permintaan Struktural: Adanya program institusional berskala besar (seperti Makan Bergizi Gratis) yang menciptakan serapan agregat baru secara mendadak di luar pola konsumsi organik masyarakat.

Siklus Kalender Eksplisit: Hanya berlaku jika teks secara langsung menyatakan bahwa kelangkaan atau kenaikan pesanan di tingkat produsen dipicu oleh persiapan menjelang hari besar keagamaan.

Aturan Penengah (Tie Breaker Rule)
Jika sebuah teks memuat laporan harga (PRICEREPORT) namun menyertakan informasi utama mengenai jembatan putus atau panen raya, maka label SUPPLYSHOCK wajib diprioritaskan. Variabel kausalitas selalu memiliki hierarki informasi yang lebih tinggi dibandingkan dengan sekadar angka harga reaktif, karena variabel tersebut menyediakan daya prediktif bagi model Bayesian Structural Time Series.
***
<p align="center">
  <sub>Dibuat untuk keperluan riset akademik. Data berita digunakan sesuai ketentuan fair use.</sub>
</p>