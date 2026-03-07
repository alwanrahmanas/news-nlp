# NLP/LLM Pipeline v2 — Inflasi Pangan Bergejolak Medan
## (Archive Scraping Edition)

Pipeline NLP/LLM untuk **Sistem Peringatan Dini Triple-Modal Bayesian** — mengintegrasikan sentimen berita, citra satelit, dan harga dalam Bayesian Structural Time Series (BSTS).

### Perbedaan vs v1

| Aspek | v1 (Google News RSS) | v2 (Archive Scraping) |
|---|---|---|
| **Sumber data** | Google News RSS fallback | Scraping langsung ke arsip Antara, Detik, Kompas, Tribun |
| **Jangkauan historis** | Hanya ~1 bulan terakhir | 5+ tahun (2021–2026) |
| **Coverage** | ~33 artikel setelah filter | Target **500–2000+** artikel |
| **Resumable** | Tidak | Ya — checkpoint system |
| **Rate limiting** | Sederhana | Per-domain, configurable |
| **Geo filter** | 6 term | 25+ variasi geo Sumut |

### Arsitektur Pipeline

Pipeline terdiri dari 11 modul utama:

* **M0**: Setup & Struktur Proyek
* **M1**: Koleksi Berita (**Archive Scraping** — multi-source)
* **M2**: Preprocessing & Deduplication (geo filter diperluas)
* **M3**: Desain Prompt & Schema LLM
* **M4**: Ekstraksi LLM menggunakan GPT-4o-mini
* **M5**: Parsing & Quality Control (QC) Ekstraksi
* **M6**: Validasi Manusia & Inter-Rater Reliability
* **M7**: Dekomposisi Sinyal Tipe A / Tipe B
* **M8**: Agregasi Time-Series Bi-Mingguan
* **M9**: Uji Kausalitas Granger
* **M10**: Final Feature Export

### Sumber Berita yang Discrape

| Sumber | URL Arsip | Prioritas |
|---|---|---|
| Antara Sumut | `sumut.antaranews.com/search` | 1 (Primer) |
| Detik Sumut | `detik.com/sumut/search` | 1 (Primer) |
| Antara Nasional | `antaranews.com/search` | 1 (Primer) |
| Detik Nasional | `detik.com/search` | 2 (Sekunder) |
| Kompas | `search.kompas.com/search` | 2 (Sekunder) |
| TribunMedan | `tribunnews.com/search` | 3 (Tersier) |
| Google News RSS | `news.google.com/rss` | 99 (Fallback) |

### Setup & Menjalankan

```bash
# 1. Virtual Environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ATAU
.venv\Scripts\activate     # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Konfigurasi API key
# Edit file .env dan isi OPENAI_API_KEY
nano .env

# 4. Jalankan pipeline
python run_pipeline.py --auto-confirm

# 5. Jalankan hanya scraping (M0+M1)
python run_pipeline.py --only m0 m1

# 6. Jalankan dari M2 (setelah scraping selesai)
python run_pipeline.py --from m2 --auto-confirm

# 7. Jalankan di background (VPS)
nohup python run_pipeline.py --auto-confirm > pipeline.log 2>&1 &
tail -f logs/pipeline.log
```

### Fitur Checkpoint (Resume)

Jika koneksi terputus saat scraping (M1), jalankan ulang:

```bash
python run_pipeline.py --only m1
```

Pipeline akan otomatis melanjutkan dari query yang belum selesai. Progress tersimpan di `data/raw/checkpoints/`.

### Catatan Penting

1. **M1** membutuhkan koneksi internet stabil. Scraping 7 sumber × 28 keyword × 30 halaman bisa memakan waktu **beberapa jam**.
2. **M4** membutuhkan OpenAI API key yang valid. Biaya tergantung jumlah artikel.
3. **M6** membutuhkan anotasi manusia — pipeline akan pause di sini.
4. Seluruh konfigurasi ada di `config.yaml`.
