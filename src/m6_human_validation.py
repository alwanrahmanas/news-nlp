"""
M6 — HUMAN VALIDATION & INTER-RATER RELIABILITY
==================================================
Input:  data/processed/extractions_clean.parquet
Output: data/validation/validation_report.json

Tugas:
  - Membuat stratified sample 200 artikel untuk validasi manusia
  - Mengekspor ke Excel untuk annotator
  - Menghitung Krippendorff's alpha (inter-rater reliability)
  - Menghitung GPT vs Human consensus metrics (macro-F1, confusion matrix)

Changelog vs v1:
  - [FIX] full_text dipotong 800 char (bukan 10.000) untuk UX annotator
  - [FIX] majority_vote: tie → "CONFLICT" bukan hasil arbitrer Counter
  - [FIX] human_label di-strip whitespace & upper-cased sebelum metrik dihitung
  - [FIX] Stratified sampling: minimum 20 artikel per stratum (bukan pure frac)
  - [FIX] Validasi kolom wajib sebelum export
  - [FIX] Counter import dipindah ke top-level (bukan di dalam apply loop)
  - [NEW] CONFLICT articles di-exclude dari F1 dengan logging eksplisit
  - [NEW] Sanity check label values setelah load annotation files
"""

import json
import logging
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from m0_setup import ROOT_DIR, get_cfg_and_logger

logger = logging.getLogger("nlp_pipeline.m6")

# Reproducibility
random.seed(42)
np.random.seed(42)

# ── Konstanta ────────────────────────────────────────────────

VALID_LABELS   = {"SUPPLYSHOCK", "DEMANDSHOCK", "PRICEREPORT", "IRRELEVANT"}
LABEL_MAP      = {"SUPPLYSHOCK": 0, "DEMANDSHOCK": 1, "PRICEREPORT": 2, "IRRELEVANT": 3}
ALPHA_THRESHOLD = 0.67
KAPPA_THRESHOLD = 0.61
F1_THRESHOLD    = 0.75
ANNOTATION_TEXT_LIMIT = 800   # FIX #1: 800 char untuk UX annotator (v1 salah 10.000)
MIN_PER_STRATUM = 20          # FIX #4: jaminan minimum sampel per label


# ── Step 1: Create Validation Sample ─────────────────────────

def create_validation_sample(df: pd.DataFrame, n: int = 200,
                             min_per_stratum: int = MIN_PER_STRATUM) -> pd.DataFrame:
    """
    Stratified sampling dengan jaminan minimum per stratum.

    Strategi dua fase:
      Phase 1 — Guaranteed floor: setiap label dapat min(min_per_stratum, len(group))
      Phase 2 — Proportional fill: sisa kuota dibagi proporsional ke pool yang tersisa

    Kenapa bukan pure frac?
      Dengan frac-based, DEMANDSHOCK (86 artikel) hanya dapat ~7 sampel —
      terlalu sedikit untuk estimasi F1 yang stabil (std error ±0.25).
      Minimum 20 memastikan setiap label punya estimasi yang lebih andal.
    """
    if len(df) == 0:
        logger.warning("Empty DataFrame, cannot create validation sample")
        return pd.DataFrame()

    if n >= len(df):
        logger.warning(f"Sample size ({n}) >= data size ({len(df)}), using all data")
        return df.copy()

    labels = df["label"].unique()
    n_labels = len(labels)

    # Sesuaikan min_per_stratum jika terlalu besar untuk target n
    if min_per_stratum * n_labels > n:
        adjusted = n // n_labels
        logger.warning(
            f"min_per_stratum={min_per_stratum} × {n_labels} labels = "
            f"{min_per_stratum * n_labels} > target {n}. "
            f"Reducing min_per_stratum to {adjusted}."
        )
        min_per_stratum = adjusted

    # Phase 1: guaranteed floor per stratum
    phase1_samples = []
    indices_used   = set()

    for label in labels:
        group = df[df["label"] == label]
        k     = min(min_per_stratum, len(group))
        s     = group.sample(n=k, random_state=42)
        phase1_samples.append(s)
        indices_used.update(s.index)

    phase1_df    = pd.concat(phase1_samples)
    remaining_n  = n - len(phase1_df)

    # Phase 2: proportional fill untuk sisa kuota
    if remaining_n > 0:
        pool = df[~df.index.isin(indices_used)]

        if len(pool) < remaining_n:
            logger.warning(
                f"Pool only has {len(pool)} articles for {remaining_n} remaining slots. "
                "Using all available."
            )
            extra = pool
        else:
            # Proporsional ke distribusi label di pool
            extra = (
                pool
                .groupby("label", group_keys=False)
                .apply(lambda x: x.sample(
                    frac=remaining_n / len(pool), random_state=42
                ))
            )
            # Koreksi ke exact remaining_n
            delta = len(extra) - remaining_n
            if delta > 0:
                extra = extra.sample(n=remaining_n, random_state=42)
            elif delta < 0:
                fill_pool = pool[~pool.index.isin(extra.index)]
                need      = min(-delta, len(fill_pool))
                extra = pd.concat([extra, fill_pool.sample(n=need, random_state=42)])

        phase1_df = pd.concat([phase1_df, extra])

    result = (
        phase1_df
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
        .head(n)  # hard cap — jaga-jaga rounding drift
    )

    logger.info(f"Validation sample created: {len(result)} articles")
    logger.info(f"Sample distribution:\n{result['label'].value_counts().to_string()}")

    return result


# ── Step 2: Export for Human Annotators ──────────────────────

def export_for_annotation(sample_df: pd.DataFrame) -> Path:
    """
    Export sample ke Excel:
      Sheet 1 'Annotation'       — artikel + kolom annotasi kosong
      Sheet 2 'Label Definitions' — cheat sheet pelabelan

    Perubahan dari v1:
      - full_text dipotong 800 char (bukan 10.000) — FIX #1
      - Validasi kolom wajib sebelum akses — FIX #5
    """
    # FIX #5: validasi kolom wajib
    required_cols = ["article_id", "published_date", "title", "label", "sentiment_score"]
    optional_cols = ["source", "full_text"]
    missing = [c for c in required_cols if c not in sample_df.columns]
    if missing:
        raise ValueError(
            f"Required columns missing from sample_df: {missing}. "
            "Check that m5_parse_qc.py ran successfully."
        )

    output_path = ROOT_DIR / "data" / "validation" / "sample_200_for_annotation.xlsx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Bangun annotation sheet
    export_cols = [c for c in required_cols + optional_cols if c in sample_df.columns]
    annotation_df = sample_df[export_cols].copy()

    # FIX #1: potong full_text ke 800 char untuk kenyamanan annotator
    if "full_text" in annotation_df.columns:
        annotation_df["full_text"] = (
            annotation_df["full_text"]
            .astype(str)
            .str[:ANNOTATION_TEXT_LIMIT]
        )

    # Rename kolom prediksi GPT agar annotator tidak bingung
    annotation_df = annotation_df.rename(columns={
        "label":           "gpt_label",
        "sentiment_score": "gpt_sentiment_score",
    })

    # Kolom annotasi kosong untuk diisi manusia
    annotation_df["human_label"]           = ""
    annotation_df["human_sentiment_score"] = ""
    annotation_df["human_notes"]           = ""

    # Cheat sheet (tidak berubah dari v1 — konten masih valid)
    cheat_sheet = pd.DataFrame({
        "Label": [
            "IRRELEVANT", "IRRELEVANT", "IRRELEVANT", "IRRELEVANT",
            "PRICEREPORT", "PRICEREPORT", "PRICEREPORT",
            "SUPPLYSHOCK", "SUPPLYSHOCK", "SUPPLYSHOCK",
            "DEMANDSHOCK", "DEMANDSHOCK", "DEMANDSHOCK",
            "TIE-BREAKER",
        ],
        "Sub_Kriteria": [
            "Kebocoran Spasial",
            "Distorsi Harga Intervensi",
            "Retorika Politik / Regulasi Normatif",
            "Penyebutan Non-Kausal",
            "Laporan Harga Murni",
            "Rilis Statistik Resmi",
            "Evaluasi Stok Pasif",
            "Disrupsi Produksi & Logistik",
            "Momentum Produksi Mayor",
            "Injeksi Volume Intervensi",
            "Bukti Penarikan Volume Tiba-Tiba",
            "Kejutan Permintaan Struktural",
            "Siklus Kalender Eksplisit",
            "SUPPLYSHOCK > PRICEREPORT",
        ],
        "Penjelasan": [
            "Kejadian/harga/gangguan logistik di LUAR Prov. Sumatera Utara.",
            "Harga subsidi, diskon ritel modern, operasi pasar murah — bukan ekuilibrium organik.",
            "Imbauan/rencana pejabat tanpa tindakan fisik atau data volume riil.",
            "Komoditas disebut tapi dalam konteks resep, restoran, pakan hewan, kurban individu.",
            "Hanya menyatakan harga naik/turun/stabil TANPA penjelasan kausalitas fisik.",
            "Publikasi IHK/NTP dari otoritas statistik — rekapitulasi historis.",
            "Pernyataan 'aman'/'terkendali' TANPA injeksi pasokan baru berskala masif.",
            "Gagal panen, hama, cuaca ekstrem, infrastruktur putus, pungutan liar logistik.",
            "Musim panen raya yang mengubah ekspektasi volume pasokan lokal secara drastis.",
            "Penyaluran CBP/SPHP Bulog ribuan ton — fokus pada penambahan fisik barang, BUKAN harga subsidi.",
            "WAJIB ada bukti perilaku konsumen menguras stok (borong warga, pembelian massal). "
            "Narasi 'harga naik akibat Ramadhan' saja TIDAK cukup.",
            "Program institusional besar (contoh: Makan Bergizi Gratis) menciptakan serapan agregat baru.",
            "Teks secara langsung menyebut kelangkaan/kenaikan pesanan di produsen dipicu hari besar keagamaan.",
            "Jika teks memuat laporan harga TETAPI juga menyebut jembatan putus/panen raya → label SUPPLYSHOCK. "
            "Variabel kausalitas > angka harga reaktif.",
        ],
        "Sentiment_Score_Guide": [
            "0.0 (netral)", "0.0", "0.0", "0.0",
            "-1.0 (naik drastis) s.d. +1.0 (turun drastis)",
            "-1.0 s.d. +1.0", "-1.0 s.d. +1.0",
            "-1.0 (parah) s.d. -0.3 (ringan)",
            "+0.3 s.d. +1.0 (surplus positif)",
            "-0.5 s.d. +0.5",
            "-0.8 (lonjakan besar) s.d. -0.3 (moderat)",
            "-0.8 s.d. -0.3",
            "-0.6 s.d. -0.3",
            "(ikuti label yang dipilih)",
        ],
    })

    # Tulis Excel
    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            annotation_df.to_excel(writer, sheet_name="Annotation", index=False)
            cheat_sheet.to_excel(writer, sheet_name="Label Definitions", index=False)
        logger.info(f"Annotation Excel saved: {output_path}")

    except ImportError:
        csv_path = output_path.with_suffix(".csv")
        annotation_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.warning(f"openpyxl not available. Fallback CSV saved: {csv_path}")
        output_path = csv_path

    # Juga simpan JSONL untuk akses programatik
    jsonl_path = ROOT_DIR / "data" / "validation" / "sample_200.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in sample_df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False, default=str) + "\n")
    logger.info(f"Sample JSONL saved: {jsonl_path}")

    return output_path


# ── Step 2b: Load & Clean Annotation Files ───────────────────

def _load_annotation_file(path: Path) -> pd.DataFrame:
    """
    Load satu annotation file, bersihkan human_label:
      - strip whitespace
      - uppercase
      - validasi hanya nilai dari VALID_LABELS
    """
    if path.suffix == ".xlsx":
        df = pd.read_excel(path, sheet_name="Annotation")
    else:
        df = pd.read_csv(path)

    if "human_label" not in df.columns:
        raise ValueError(f"Column 'human_label' not found in {path.name}")

    # FIX #3: strip whitespace & uppercase — Excel sering tambah spasi tersembunyi
    df["human_label"] = (
        df["human_label"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace("NAN", np.nan)
        .replace("", np.nan)
    )

    # Sanity check: nilai di luar VALID_LABELS
    invalid_vals = set(
        df["human_label"].dropna().unique()
    ) - VALID_LABELS
    if invalid_vals:
        logger.warning(
            f"{path.name}: unexpected label values found: {invalid_vals}. "
            "Rows with these values will be treated as NaN."
        )
        df.loc[~df["human_label"].isin(VALID_LABELS), "human_label"] = np.nan

    n_empty = df["human_label"].isna().sum()
    if n_empty > 0:
        logger.info(f"{path.name}: {n_empty} rows with empty/invalid human_label")

    return df


# ── Step 3: Compute Inter-rater Reliability ──────────────────

def compute_inter_rater_reliability(annotations_dir: Path = None) -> dict:
    """
    Hitung Krippendorff's alpha dan Cohen's kappa pairwise.

    Perubahan dari v1:
      - _load_annotation_file() membersihkan whitespace & validasi nilai — FIX #3
      - Rows dengan label NaN/invalid di-exclude sebelum hitung metrik
    """
    if annotations_dir is None:
        annotations_dir = ROOT_DIR / "data" / "validation" / "hasil labeling"

    rater_files = [
        annotations_dir / "[1] sample_200_for_annotation.xlsx",
        annotations_dir / "[2] sample_200_for_annotation.xlsx",
    ]
    existing_files = [f for f in rater_files if f.exists()]

    if len(existing_files) < 2:
        logger.warning(
            f"Need at least 2 rater annotation files. Found {len(existing_files)}. "
            "Skipping inter-rater reliability computation."
        )
        return {"status": "pending_annotations", "files_found": len(existing_files)}

    # Load semua rater files
    raters = []
    for f in existing_files:
        try:
            raters.append(_load_annotation_file(f))
        except (ValueError, Exception) as e:
            logger.error(f"Failed to load {f.name}: {e}")
            continue

    if len(raters) < 2:
        return {"status": "invalid_format", "message": "Could not load 2 valid rater files"}

    # Gabung by article_id
    merged = raters[0][["article_id", "human_label"]].rename(
        columns={"human_label": "rater1"}
    )
    for i, r in enumerate(raters[1:], start=2):
        merged = merged.merge(
            r[["article_id", "human_label"]].rename(columns={"human_label": f"rater{i}"}),
            on="article_id",
            how="inner",
        )

    rater_cols = [c for c in merged.columns if c.startswith("rater")]

    # Drop rows yang salah satu raternya kosong
    before = len(merged)
    merged = merged.dropna(subset=rater_cols)
    after  = len(merged)
    if before != after:
        logger.info(f"Dropped {before - after} rows with incomplete annotations")

    if len(merged) < 10:
        logger.error(
            f"Only {len(merged)} complete rows after dropping NaN. "
            "Inter-rater metrics not computed."
        )
        return {"status": "insufficient_complete_rows", "n_complete": len(merged)}

    # Encode ke numerik untuk Krippendorff
    for col in rater_cols:
        merged[col + "_num"] = merged[col].map(LABEL_MAP)

    # Krippendorff's alpha
    alpha = None
    try:
        import krippendorff
        reliability_data = merged[[c + "_num" for c in rater_cols]].values.T.astype(float)
        alpha = krippendorff.alpha(
            reliability_data=reliability_data,
            level_of_measurement="nominal",
        )
    except ImportError:
        logger.warning(
            "krippendorff package not installed. Run: pip install krippendorff. "
            "Skipping alpha calculation."
        )

    # Cohen's kappa pairwise
    from sklearn.metrics import cohen_kappa_score
    kappa_results = {}
    for i in range(len(rater_cols)):
        for j in range(i + 1, len(rater_cols)):
            key       = f"{rater_cols[i]}_vs_{rater_cols[j]}"
            valid_idx = merged[rater_cols[i]].notna() & merged[rater_cols[j]].notna()
            ri        = merged.loc[valid_idx, rater_cols[i]].astype(str).values
            rj        = merged.loc[valid_idx, rater_cols[j]].astype(str).values
            if len(ri) == 0:
                continue
            kappa_results[key] = round(cohen_kappa_score(ri, rj), 4)

    alpha_pass = (alpha >= ALPHA_THRESHOLD) if alpha is not None else None

    results = {
        "n_articles"          : len(merged),
        "n_raters"            : len(rater_cols),
        "krippendorff_alpha"  : round(float(alpha), 4) if alpha is not None else None,
        "cohen_kappa_pairwise": kappa_results,
        "alpha_threshold"     : ALPHA_THRESHOLD,
        "kappa_threshold"     : KAPPA_THRESHOLD,
        "alpha_pass"          : alpha_pass,
    }

    logger.info(f"Krippendorff's alpha  : {results['krippendorff_alpha']} "
                f"(threshold={ALPHA_THRESHOLD}, pass={alpha_pass})")
    for k, v in kappa_results.items():
        kappa_pass = v >= KAPPA_THRESHOLD
        logger.info(
            f"Cohen's kappa ({k}): {v} "
            f"(threshold={KAPPA_THRESHOLD}, pass={kappa_pass})"
        )

    return results


# ── Step 4: GPT vs Human Consensus ──────────────────────────

def _majority_vote(row: pd.Series, rater_cols: list) -> str:
    """
    Hitung consensus dari beberapa rater.

    FIX #2: tie (misalnya 2 rater tidak sepakat) → kembalikan "CONFLICT"
    bukan resolusi arbitrer dari Counter.most_common().
    "CONFLICT" akan di-exclude dari perhitungan F1.
    """
    votes = [row[c] for c in rater_cols if pd.notna(row[c]) and str(row[c]).strip()]
    if not votes:
        return "CONFLICT"  # tidak ada vote valid = conflict

    # Counter diimport di top-level — tidak di dalam apply
    count     = Counter(votes)
    top_two   = count.most_common(2)

    # Cek tie
    if len(top_two) > 1 and top_two[0][1] == top_two[1][1]:
        return "CONFLICT"  # tie → bukan majority, exclude dari F1

    return top_two[0][0]


def compute_gpt_vs_human(sample_df: pd.DataFrame,
                          annotations_dir: Path = None) -> dict:
    """
    Bandingkan prediksi GPT vs human consensus (majority vote).
    Target: macro-F1 >= 0.75.

    Perubahan dari v1:
      - _load_annotation_file() untuk pembersihan label — FIX #3
      - _majority_vote() dengan CONFLICT handling — FIX #2
      - CONFLICT articles di-exclude dari F1 calculation dengan logging
    """
    if annotations_dir is None:
        annotations_dir = ROOT_DIR / "data" / "validation" / "hasil labeling"

    rater_files = [
        annotations_dir / "[1] sample_200_for_annotation.xlsx",
        annotations_dir / "[2] sample_200_for_annotation.xlsx",
    ]
    existing_files = [f for f in rater_files if f.exists()]

    if len(existing_files) < 2:
        logger.warning("Need at least 2 rater files for consensus. Skipping.")
        return {"status": "pending_annotations"}

    # Load & bersihkan semua rater files
    raters = []
    for f in existing_files:
        try:
            raters.append(_load_annotation_file(f))
        except (ValueError, Exception) as e:
            logger.error(f"Failed to load {f.name}: {e}")

    if len(raters) < 2:
        return {"status": "invalid_format"}

    # Gabung semua label per artikel
    labels_matrix = pd.DataFrame({"article_id": raters[0]["article_id"]})
    for i, r in enumerate(raters):
        labels_matrix[f"rater{i+1}"] = r["human_label"].values

    rater_cols = [c for c in labels_matrix.columns if c.startswith("rater")]

    # FIX #2: majority vote dengan CONFLICT handling
    labels_matrix["human_consensus"] = labels_matrix.apply(
        lambda row: _majority_vote(row, rater_cols), axis=1
    )

    n_conflicts = (labels_matrix["human_consensus"] == "CONFLICT").sum()
    if n_conflicts > 0:
        logger.warning(
            f"{n_conflicts}/{len(labels_matrix)} articles have rater CONFLICT "
            f"({n_conflicts/len(labels_matrix)*100:.1f}%) — excluded from F1 calculation."
        )

    # Gabung dengan prediksi GPT
    merged = labels_matrix.merge(
        sample_df[["article_id", "label"]].rename(columns={"label": "gpt_label"}),
        on="article_id",
        how="inner",
    )

    # Exclude CONFLICT dari evaluasi
    eval_df = merged[merged["human_consensus"] != "CONFLICT"].copy()

    if len(eval_df) < 10:
        logger.error(
            f"Only {len(eval_df)} non-conflict articles for F1 computation. "
            "Results not reliable."
        )
        return {"status": "insufficient_data", "n_eval": len(eval_df)}

    # Hitung metrik
    from sklearn.metrics import (
        f1_score, confusion_matrix, classification_report
    )

    labels_list = ["SUPPLYSHOCK", "DEMANDSHOCK", "PRICEREPORT", "IRRELEVANT"]

    y_true = eval_df["human_consensus"].fillna("IRRELEVANT").astype(str).values
    y_pred = eval_df["gpt_label"].fillna("IRRELEVANT").astype(str).values

    macro_f1      = f1_score(y_true, y_pred, average="macro",
                              labels=labels_list, zero_division=0)
    per_class_f1  = f1_score(y_true, y_pred, average=None,
                              labels=labels_list, zero_division=0)
    cm            = confusion_matrix(y_true, y_pred, labels=labels_list)
    report        = classification_report(y_true, y_pred, labels=labels_list,
                                          output_dict=True, zero_division=0)

    results = {
        "n_articles_total"  : len(merged),
        "n_articles_eval"   : len(eval_df),
        "n_conflict_excluded": int(n_conflicts),
        "macro_f1"          : round(float(macro_f1), 4),
        "per_class_f1"      : {
            label: round(float(f1), 4)
            for label, f1 in zip(labels_list, per_class_f1)
        },
        "confusion_matrix"  : cm.tolist(),
        "confusion_labels"  : labels_list,
        "classification_report": report,
        "f1_threshold"      : F1_THRESHOLD,
        "f1_pass"           : bool(macro_f1 >= F1_THRESHOLD),
    }

    logger.info("=" * 50)
    logger.info("GPT vs HUMAN CONSENSUS")
    logger.info("=" * 50)
    logger.info(f"Evaluated articles   : {len(eval_df)} (excluded {n_conflicts} conflicts)")
    logger.info(f"Macro-F1             : {results['macro_f1']} "
                f"(threshold={F1_THRESHOLD}, pass={results['f1_pass']})")
    for label, f1 in results["per_class_f1"].items():
        logger.info(f"  {label:15}: F1={f1:.4f}")

    return results


# ── Main ─────────────────────────────────────────────────────

def run_human_validation(cfg: dict = None) -> None:
    """
    Run the full human validation pipeline.

    Phase 1: Buat sample & export ke Excel untuk annotator
             (skip jika sample_200.jsonl sudah ada)
    Phase 2: Hitung metrik setelah anotasi selesai
             (skip jika file anotasi belum ada)
    """
    if cfg is None:
        cfg, _ = get_cfg_and_logger()

    input_path = ROOT_DIR / "data" / "processed" / "extractions_clean.parquet"

    if not input_path.exists():
        logger.error(
            f"extractions_clean.parquet not found: {input_path}. "
            "Run m5_parse_qc.py first."
        )
        return

    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} articles from extractions_clean.parquet")

    # ── Phase 1: Sample & Export ──────────────────────────────
    jsonl_path = ROOT_DIR / "data" / "validation" / "sample_200.jsonl"

    if jsonl_path.exists():
        sample = pd.read_json(jsonl_path, lines=True)
        logger.info(f"Loaded existing validation sample ({len(sample)} articles)")
    else:
        sample = create_validation_sample(df, n=200, min_per_stratum=MIN_PER_STRATUM)
        if sample.empty:
            logger.error("Failed to create validation sample. Aborting.")
            return
        export_for_annotation(sample)
        logger.info(
            "\n" + "=" * 50 + "\n"
            "ANNOTATION REQUIRED\n"
            "=" * 50 + "\n"
            f"File: data/validation/sample_200_for_annotation.xlsx\n"
            "Instruksi:\n"
            "  1. Buka Sheet 'Annotation'\n"
            "  2. Isi kolom 'human_label' untuk setiap artikel\n"
            "     Valid values: SUPPLYSHOCK, DEMANDSHOCK, PRICEREPORT, IRRELEVANT\n"
            "  3. Isi kolom 'human_sentiment_score' (-1.0 sampai +1.0)\n"
            "  4. Simpan sebagai:\n"
            "     [1] sample_200_for_annotation.xlsx (Rater 1)\n"
            "     [2] sample_200_for_annotation.xlsx (Rater 2)\n"
            "  5. Letakkan di: data/validation/hasil labeling/\n"
            "  6. Jalankan ulang: python run_pipeline.py --only m6\n"
            + "=" * 50
        )

    # ── Phase 2: Compute Metrics ──────────────────────────────
    irr_results = compute_inter_rater_reliability()
    gpt_results = compute_gpt_vs_human(sample)

    # Compile validation report
    report = {
        "n_sample"    : len(sample),
        "inter_rater" : irr_results,
        "gpt_vs_human": gpt_results,
    }

    # Tentukan pass/fail
    alpha_pass = irr_results.get("alpha_pass")
    f1_pass    = gpt_results.get("f1_pass")

    if alpha_pass is not None and f1_pass is not None:
        report["pass"]  = bool(alpha_pass and f1_pass)
        report["notes"] = ""
    else:
        report["pass"]  = None
        report["notes"] = (
            "Waiting for human annotations. "
            "Place completed files in data/validation/hasil labeling/ "
            "and rerun: python run_pipeline.py --only m6"
        )

    # Simpan report
    report_path = ROOT_DIR / "data" / "validation" / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Validation report saved: {report_path}")

    # ── Rekomendasi jika gagal ────────────────────────────────
    if report.get("pass") is False:
        logger.warning("=" * 50)
        logger.warning("VALIDATION FAILED — ACTION REQUIRED:")
        if not irr_results.get("alpha_pass", True):
            logger.warning(
                f"  Krippendorff alpha ({irr_results.get('krippendorff_alpha')}) "
                f"< {ALPHA_THRESHOLD} — kesepakatan antar-rater tidak cukup."
            )
        if not gpt_results.get("f1_pass", True):
            logger.warning(
                f"  Macro-F1 ({gpt_results.get('macro_f1')}) "
                f"< {F1_THRESHOLD} — GPT terlalu jauh dari konsensus manusia."
            )
        logger.warning("Langkah perbaikan:")
        logger.warning("  1. Lakukan error analysis — ambil 20 artikel salah klasifikasi")
        logger.warning("  2. Revisi prompt (tambah contoh boundary cases)")
        logger.warning("  3. Ulangi validasi pada 100 artikel baru")
        logger.warning("  JANGAN lanjutkan ke M7 sebelum threshold terpenuhi.")
        logger.warning("=" * 50)

    elif report.get("pass") is True:
        logger.info("=" * 50)
        logger.info("✓ VALIDATION PASSED — Pipeline dapat dilanjutkan ke M7.")
        logger.info("=" * 50)


if __name__ == "__main__":
    cfg, log = get_cfg_and_logger(skip_env=True)
    run_human_validation(cfg)
