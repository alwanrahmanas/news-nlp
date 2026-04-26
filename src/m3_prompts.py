"""
M3 — DESAIN PROMPT & SCHEMA OUTPUT
====================================
File: src/m3_prompts.py + file teks di prompts/

Tugas:
  - Mendefinisikan schema JSON output LLM
  - Membuat dan menyimpan prompt utama (main classification)
  - Membuat dan menyimpan prompt Type A/B decomposition
  - Fungsi build_prompt() untuk inject artikel ke template
"""

import json
import logging
from pathlib import Path

from jsonschema import validate, ValidationError

from m0_setup import ROOT_DIR, get_cfg_and_logger

logger = logging.getLogger("nlp_pipeline.m3")


# ── Schema Definition ────────────────────────────────────────

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning_chain": {
            "type": "object",
            "description": "Step-by-step Chain-of-Thought reasoning trace",
            "properties": {
                "step1_commodities": {"type": "string"},
                "step2_geography": {"type": "string"},
                "step3_event_type": {"type": "string"},
                "step4_temporal": {"type": "string"},
                "step5_sentiment": {"type": "string"},
            },
            "required": ["step1_commodities", "step2_geography",
                          "step3_event_type", "step4_temporal", "step5_sentiment"]
        },
        "label": {
            "type": "string",
            "enum": ["SUPPLYSHOCK", "DEMANDSHOCK", "PRICEREPORT", "IRRELEVANT"]
        },
        "sentiment_score": {
            "type": "number",
            "minimum": -1.0,
            "maximum": 1.0
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0
        },
        "commodities": {
            "type": "array",
            "items": {"type": "string"}
        },
        "supply_location": {
            "type": ["string", "null"]
        },
        "is_forward_looking": {
            "type": "boolean"
        },
        "rationale": {
            "type": "string",
            "maxLength": 1000
        }
    },
    "required": ["reasoning_chain", "label", "sentiment_score", "confidence",
                  "commodities", "supply_location", "is_forward_looking", "rationale"]
}

TYPEAB_SCHEMA = {
    "type": "object",
    "properties": {
        "type_ab": {
            "type": "string",
            "enum": ["A", "B"]
        },
        "ab_confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0
        },
        "temporal_signal": {
            "type": "string"
        }
    },
    "required": ["type_ab", "ab_confidence", "temporal_signal"]
}


# ── Prompt Templates ─────────────────────────────────────────

PROMPT_MAIN = """Kamu adalah analis senior ketahanan pangan Indonesia yang mengklasifikasi artikel berita
untuk model prediktif inflasi pangan di Medan, Sumatera Utara.

═══════════════════════════════════════════════
CHAIN-OF-THOUGHT: Analisis artikel STEP-BY-STEP
═══════════════════════════════════════════════

Sebelum memberi label, WAJIB lakukan analisis bertahap berikut:

STEP 1 — IDENTIFIKASI KOMODITAS:
  Temukan semua komoditas pangan yang disebut dalam artikel.
  Komoditas target utama (volatile food):
    cabai merah, cabai rawit, bawang merah, bawang putih,
    telur ayam, daging ayam, gula pasir, minyak goreng,
    beras, daging sapi, tomat, kentang, kacang kedelai
  Jika TIDAK ADA komoditas pangan yang disebut → pertimbangkan IRRELEVANT.

STEP 2 — CEK RELEVANSI GEOGRAFIS:
  Apakah artikel berkaitan dengan wilayah Sumatera Utara / Medan / sekitarnya?
  Lokasi relevan:
    - Kota Medan, Pasar Induk Medan, Pajak Sambu, Pajak Aksara
    - Kabupaten Karo, Berastagi (sentra hortikultura hulu)
    - Deli Serdang, Binjai, Tebing Tinggi, Siantar, Toba
    - "Sumatera Utara" / "Sumut" secara umum
  Jika artikel hanya membahas daerah LAIN (Jawa, Kalimantan, dll)
  TANPA keterkaitan pasokan ke Sumut → pertimbangkan IRRELEVANT.

STEP 3 — TENTUKAN JENIS PERISTIWA:
  (a) SUPPLY-SIDE EVENT: Ada gangguan/ancaman/risiko pada RANTAI PASOKAN?
      Contoh: banjir, longsor, kekeringan, erupsi gunung, gagal panen,
      hama/penyakit tanaman, kerusakan jalan distribusi, kenaikan biaya
      pupuk/BBM, larangan impor, kelangkaan stok di produsen.
      → Jika YA → kandidat SUPPLYSHOCK

  (b) DEMAND-SIDE EVENT: Ada lonjakan/tekanan PERMINTAAN abnormal?
      Contoh: menjelang Ramadhan/Lebaran/Natal/Tahun Baru, operasi pasar,
      panic buying, distribusi bansos massal, permintaan ekspor mendadak.
      → Jika YA → kandidat DEMANDSHOCK

  (c) PRICE MOVEMENT REPORTING: Artikel MELAPORKAN pergerakan harga
      yang sudah terjadi TANPA menjelaskan faktor supply/demand ke depan?
      Contoh: "Harga cabai naik 20% di Pasar Sambu", "daftar harga hari
      ini", "update harga komoditas pekan ini".
      → Jika YA → kandidat PRICEREPORT

  (d) OTHER: Tidak masuk kategori di atas (politik, resep, opini umum).
      → IRRELEVANT

STEP 4 — TENTUKAN ORIENTASI TEMPORAL:
  - FORWARD-LOOKING (is_forward_looking: true):
    Artikel MENDAHULUI dampak harga. Peristiwa baru terjadi/akan terjadi,
    harga BELUM terpengaruh tapi BERPOTENSI naik.
    Kata kunci: "dikhawatirkan", "berpotensi", "diprediksi", "waspada",
    "antisipasi", "jika berlanjut", "terancam", "membayangi", "ancaman",
    "tanda-tanda", "diperkirakan", "bisa menyebabkan"
    ATURAN KUNCI: Jika bencana/gangguan terjadi di HULU (Karo/Berastagi)
    dan dampak ke HILIR (Medan) belum dirasakan → OTOMATIS forward-looking
    (karena ada lag waktu distribusi).

  - BACKWARD-LOOKING (is_forward_looking: false):
    Artikel MELAPORKAN kondisi yang SUDAH terjadi. Harga SUDAH bergerak.
    Kata kunci: "akibat", "karena", "disebabkan", "setelah", "pasca",
    "sudah naik", "tercatat", "melonjak sejak", "pekan lalu"

STEP 5 — TENTUKAN SENTIMEN:
  Skor -1.0 sampai +1.0:
   +0.7 sampai +1.0 : tekanan inflasi sangat kuat (harga naik tajam/pasokan kritis)
   +0.3 sampai +0.7 : tekanan inflasi moderat
   -0.3 sampai +0.3 : netral/campuran
   -0.7 sampai -0.3 : tekanan deflasi moderat (harga turun/pasokan surplus)
   -1.0 sampai -0.7 : tekanan deflasi sangat kuat

═══════════════════════════════════════════════
CONTOH KLASIFIKASI (Few-Shot)
═══════════════════════════════════════════════

CONTOH 1 — SUPPLYSHOCK:
  Judul: "Banjir Landa Karo, Ribuan Hektare Lahan Cabai Terendam"
  → Label: SUPPLYSHOCK, sentiment: +0.8, is_forward_looking: true
  → Rationale: "Banjir di Karo merusak lahan cabai. Dampak belum terasa
     di Medan tapi pasokan akan terganggu 1-2 minggu ke depan."

CONTOH 2 — DEMANDSHOCK:
  Judul: "Jelang Ramadhan, Warga Medan Ramai Belanja Kebutuhan Pokok"
  → Label: DEMANDSHOCK, sentiment: +0.6, is_forward_looking: true
  → Rationale: "Lonjakan permintaan musiman menjelang Ramadhan
     berpotensi menekan harga naik untuk cabai, ayam, dan telur."

CONTOH 3 — PRICEREPORT:
  Judul: "Harga Cabai di Pasar Sambu Medan Tembus Rp120 Ribu per Kg"
  → Label: PRICEREPORT, sentiment: +0.7, is_forward_looking: false
  → Rationale: "Melaporkan harga cabai yang sudah naik tinggi di pasar.
     Tidak ada informasi prediktif tentang supply/demand ke depan."

CONTOH 4 — IRRELEVANT:
  Judul: "Resep Sambal Cabai Merah Khas Medan yang Menggugah Selera"
  → Label: IRRELEVANT, sentiment: 0.0, is_forward_looking: false
  → Rationale: "Artikel resep masakan, bukan berita supply/demand/harga."

═══════════════════════════════════════════════
ATURAN KEPUTUSAN (Decision Rules — Kerangka Kausalitas Ekonomi Makro)
═══════════════════════════════════════════════

KATEGORI IRRELEVANT — Filter Kebisingan dan Distorsi:
  Labeli IRRELEVANT jika memenuhi MINIMAL SATU kondisi berikut:
  • Kebocoran Spasial: kejadian/harga/gangguan logistik di LUAR Prov. Sumatera Utara.
  • Distorsi Harga Intervensi: harga subsidi, diskon ritel modern, operasi pasar murah
    — angka ini tidak mencerminkan ekuilibrium pasar organik.
  • Retorika Politik / Regulasi Normatif: imbauan/rencana pejabat tanpa tindakan fisik
    atau data volume riil di lapangan.
  • Penyebutan Non-Kausal: komoditas disebut dalam konteks resep, restoran, pakan hewan,
    kurban individu.

KATEGORI PRICEREPORT — Evaluasi Retrospektif dan Reaktif:
  Gunakan untuk teks yang memotret kondisi SUDAH TERJADI tanpa variabel prediktif:
  • Laporan Harga Murni: hanya menyatakan harga naik/turun/stabil tanpa kausalitas fisik.
  • Rilis Statistik Resmi: publikasi IHK/NTP dari otoritas statistik (rekapitulasi historis).
  • Evaluasi Stok Pasif: pernyataan "aman"/"terkendali" tanpa injeksi pasokan baru masif.

KATEGORI SUPPLYSHOCK — Pergeseran Kurva Penawaran:
  Teks HARUS secara eksplisit memuat variabel fundamental yang mengubah volume komoditas:
  • Disrupsi Produksi & Logistik: gagal panen, hama, cuaca ekstrem, infrastruktur putus,
    pungutan liar logistik, penjarahan fasilitas penyimpanan.
  • Momentum Produksi Mayor: musim panen raya yang mengubah ekspektasi volume pasokan.
  • Injeksi Volume Intervensi: CBP/SPHP Bulog ribuan ton — fokus pada penambahan fisik
    barang di pasar, BUKAN pada harga jual subsidi.

KATEGORI DEMANDSHOCK — Pergeseran Kurva Permintaan (KETAT):
  Pembuktian kausalitas harus SANGAT ketat — hindari kerancuan dengan PRICEREPORT:
  • Bukti Penarikan Volume Tiba-Tiba: WAJIB ada bukti perilaku konsumen menguras stok.
    "Harga naik akibat Ramadhan" SAJA tidak cukup. Contoh valid: "stok distributor menipis
    karena aksi borong warga", "pembelian massal oleh entitas politik".
  • Kejutan Permintaan Struktural: program institusional besar (contoh: Makan Bergizi Gratis)
    yang menciptakan serapan agregat baru di luar pola konsumsi organik.
  • Siklus Kalender Eksplisit: hanya jika teks secara langsung menyatakan kelangkaan/kenaikan
    pesanan di tingkat produsen dipicu persiapan hari besar keagamaan.

TIE-BREAKER RULE:
  Jika teks memuat laporan harga (PRICEREPORT) TETAPI juga menyertakan informasi utama
  mengenai jembatan putus, panen raya, atau gangguan rantai pasok lainnya:
  → SUPPLYSHOCK wajib diprioritaskan.
  Variabel kausalitas selalu memiliki hierarki informasi lebih tinggi dibanding angka harga
  reaktif, karena menyediakan daya prediktif bagi model BSTS.

HARD RULES:
  • SUPPLYSHOCK & DEMANDSHOCK → is_forward_looking HARUS true
    (karena fokus pada sinyal prediktif, bukan laporan reaktif)
  • PRICEREPORT → is_forward_looking HARUS false
  • confidence < 0.5 → pertimbangkan IRRELEVANT sebagai fallback

═══════════════════════════════════════════════
SCHEMA OUTPUT (harus JSON valid, ikuti persis)
═══════════════════════════════════════════════

{{
  "reasoning_chain": {{
    "step1_commodities": "komoditas yang ditemukan...",
    "step2_geography": "relevansi geografis...",
    "step3_event_type": "supply/demand/price/other + alasan...",
    "step4_temporal": "forward/backward-looking + bukti...",
    "step5_sentiment": "arah sentimen + skor + justifikasi..."
  }},
  "label": "SUPPLYSHOCK | DEMANDSHOCK | PRICEREPORT | IRRELEVANT",
  "sentiment_score": float (-1.0 to 1.0),
  "confidence": float (0.0 to 1.0),
  "commodities": ["komoditas1", "komoditas2"],
  "supply_location": "lokasi pasokan" | null,
  "is_forward_looking": true | false,
  "rationale": "ringkasan 1-2 kalimat alasan klasifikasi"
}}

═══════════════════════════════════════════════
ARTIKEL YANG HARUS DIANALISIS:
═══════════════════════════════════════════════
Tanggal: {article_date}
Sumber: {article_source}
Judul: {article_title}

{article_text}"""


PROMPT_TYPEAB = """Kamu menganalisis artikel SUPPLYSHOCK atau DEMANDSHOCK untuk menentukan
apakah artikel ini bersifat PREDIKTIF (Tipe A) atau REAKTIF (Tipe B).

DEFINISI:
Tipe A (Prediktif/Forward-looking):
  - Artikel MENDAHULUI pergerakan harga
  - Berisi informasi tentang ancaman/gangguan yang BELUM berdampak pada harga
  - Sinyal dini yang bisa jadi leading indicator
  - Kata kunci khas: "dikhawatirkan", "berpotensi", "diprediksi", "waspada",
    "antisipasi", "jika berlanjut", "petani khawatir", "panen terancam"
  - Contoh: "Banjir landa Karo, petani cabai khawatir panen gagal"

Tipe B (Reaktif/Backward-looking):
  - Artikel MENGIKUTI pergerakan harga yang sudah terjadi
  - Menjelaskan MENGAPA harga sudah naik
  - Lagging reporter - tidak berguna sebagai early warning
  - Kata kunci khas: "akibat", "karena", "disebabkan", "setelah", "pasca",
    "harga naik karena pasokan berkurang"
  - Contoh: "Harga cabai naik 20% akibat banjir Karo bulan lalu"

TUGAS:
Tentukan apakah artikel ini Tipe A atau Tipe B.
Kembalikan HANYA JSON:
{{
  "type_ab": "A" | "B",
  "ab_confidence": float,
  "temporal_signal": "kalimat/frasa yang menunjukkan timing artikel ini"
}}

ARTIKEL:
{article_text}"""


# ── Functions ────────────────────────────────────────────────

def build_prompt(article: dict, prompt_template: str = "main") -> str:
    """
    Build a prompt by injecting article data into a template.
    
    Args:
        article: dict with keys article_date/published_date, source, title, full_text
        prompt_template: "main" or "typeab"
    """
    if prompt_template == "main":
        template = PROMPT_MAIN
        return template.format(
            article_date=article.get("published_date", article.get("article_date", "N/A")),
            article_source=article.get("source", "N/A"),
            article_title=article.get("title", "N/A"),
            article_text=article.get("full_text", ""),
        )
    elif prompt_template == "typeab":
        template = PROMPT_TYPEAB
        return template.format(
            article_text=article.get("full_text", ""),
        )
    else:
        raise ValueError(f"Unknown prompt template: {prompt_template}")


def validate_extraction_schema(parsed: dict) -> bool:
    """
    Validate that parsed LLM output matches the expected schema.
    Raises ValidationError if invalid.
    """
    try:
        validate(instance=parsed, schema=EXTRACTION_SCHEMA)
        return True
    except ValidationError as e:
        logger.warning(f"Schema validation failed: {e.message}")
        raise


def validate_typeab_schema(parsed: dict) -> bool:
    """Validate Type A/B output schema."""
    try:
        validate(instance=parsed, schema=TYPEAB_SCHEMA)
        return True
    except ValidationError as e:
        logger.warning(f"TypeAB schema validation failed: {e.message}")
        raise


def save_prompts_to_files() -> None:
    """Save prompt templates to text files in prompts/ directory."""
    prompts_dir = ROOT_DIR / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    # Main prompt (v2 with CoT)
    main_path = prompts_dir / "prompt_main_v2_cot.txt"
    with open(main_path, "w", encoding="utf-8") as f:
        f.write(PROMPT_MAIN)
    logger.info(f"Main prompt saved: {main_path}")

    # TypeAB prompt
    typeab_path = prompts_dir / "prompt_typeAB_v1.txt"
    with open(typeab_path, "w", encoding="utf-8") as f:
        f.write(PROMPT_TYPEAB)
    logger.info(f"TypeAB prompt saved: {typeab_path}")

    # Schema
    schema_path = prompts_dir / "schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump({
            "extraction_schema": EXTRACTION_SCHEMA,
            "typeab_schema": TYPEAB_SCHEMA,
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"Schema saved: {schema_path}")


if __name__ == "__main__":
    cfg, log = get_cfg_and_logger(skip_env=True)
    save_prompts_to_files()
    logger.info("M3 Prompt & Schema design complete.")
