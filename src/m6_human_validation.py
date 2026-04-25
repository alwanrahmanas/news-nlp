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
"""

import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

from m0_setup import ROOT_DIR, get_cfg_and_logger

logger = logging.getLogger("nlp_pipeline.m6")

# Reproducibility
random.seed(42)
np.random.seed(42)


# ── Step 1: Create Validation Sample ─────────────────────────

def create_validation_sample(df: pd.DataFrame, n: int = 200) -> pd.DataFrame:
    """
    Stratified sampling: proportional to actual label distribution.
    Example: 20.9% SUPPLYSHOCK -> ~42 articles; 42.1% PRICEREPORT -> ~84 articles
    """
    if len(df) == 0:
        logger.warning("Empty DataFrame, cannot create validation sample")
        return pd.DataFrame()

    if n >= len(df):
        logger.warning(f"Sample size ({n}) >= data size ({len(df)}), using all data")
        return df.copy()

    frac = n / len(df)
    sample = df.groupby("label", group_keys=False).apply(
        lambda x: x.sample(frac=frac, random_state=42)
    ).reset_index(drop=True)

    # Adjust to exact n if groupby rounding is off
    if len(sample) < n:
        remaining = df[~df.index.isin(sample.index)]
        extra = remaining.sample(n=n - len(sample), random_state=42)
        sample = pd.concat([sample, extra]).reset_index(drop=True)
    elif len(sample) > n:
        sample = sample.sample(n=n, random_state=42).reset_index(drop=True)

    logger.info(f"Validation sample created: {len(sample)} articles")
    logger.info(f"Sample distribution:\n{sample['label'].value_counts().to_string()}")

    return sample


# ── Step 2: Export for Human Annotators ──────────────────────

def export_for_annotation(sample_df: pd.DataFrame) -> Path:
    """
    Export sample to Excel with:
    - Sheet 1: Articles with blank annotation columns
    - Sheet 2: Label definitions cheat sheet
    """
    output_path = ROOT_DIR / "data" / "validation" / "sample_200_for_annotation.xlsx"

    # Prepare annotation sheet
    annotation_df = sample_df[["article_id", "published_date", "source", "title"]].copy()

    # Truncate full_text to 800 chars for readability
    if "full_text" in sample_df.columns:
        annotation_df["full_text"] = sample_df["full_text"].str[:10000]

    # Add GPT prediction (hidden column for later comparison)
    annotation_df["gpt_label"] = sample_df["label"]
    annotation_df["gpt_sentiment_score"] = sample_df["sentiment_score"]

    # Add blank annotation columns
    annotation_df["human_label"] = ""
    annotation_df["human_sentiment_score"] = ""
    annotation_df["human_notes"] = ""

    # Cheat sheet
    cheat_sheet = pd.DataFrame({
        "Label": ["SUPPLYSHOCK", "DEMANDSHOCK", "PRICEREPORT", "IRRELEVANT"],
        "Definisi": [
            "Gangguan/ancaman PASOKAN yang bisa naikkan harga. Banjir, kekeringan, gagal panen, hambatan distribusi.",
            "Lonjakan PERMINTAAN tidak biasa. Ramadan, Lebaran, Natal, ekspor mendadak.",
            "Laporan harga REAKTIF (sudah naik/turun), tanpa info supply/demand ke depan.",
            "Tidak terkait volatile food supply/demand/price di Medan/Sumut."
        ],
        "Contoh": [
            "Banjir landa Karo, petani cabai khawatir panen gagal",
            "Menjelang Ramadan, permintaan daging ayam naik tajam",
            "Harga cabai hari ini naik 15% di Pasar Induk Medan",
            "Gubernur Sumut resmikan jembatan baru di Karo"
        ],
        "Sentiment_Score_Guide": [
            "-1.0 (parah) s.d. -0.3 (ringan). Negatif = ancaman kenaikan harga",
            "-0.8 (lonjakan besar) s.d. -0.3 (peningkatan moderat)",
            "-1.0 (naik drastis) s.d. +1.0 (turun drastis). Refleksi pergerakan aktual",
            "0.0 (netral)"
        ],
    })

    # Save to Excel
    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            annotation_df.to_excel(writer, sheet_name="Annotation", index=False)
            cheat_sheet.to_excel(writer, sheet_name="Label Definitions", index=False)

        logger.info(f"Annotation file saved: {output_path}")
    except ImportError:
        # Fallback to CSV if openpyxl not available
        csv_path = output_path.with_suffix(".csv")
        annotation_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.warning(f"openpyxl not available. Saved as CSV: {csv_path}")
        output_path = csv_path

    # Also save as JSONL for programmatic access
    jsonl_path = ROOT_DIR / "data" / "validation" / "sample_200.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in sample_df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False, default=str) + "\n")
    logger.info(f"Sample JSONL saved: {jsonl_path}")

    return output_path


# ── Step 3: Compute Inter-rater Reliability ──────────────────

def compute_inter_rater_reliability(annotations_dir: Path = None) -> dict:
    """
    Compute Krippendorff's alpha and Cohen's kappa pairwise.
    Expects: [1] sample_200_for_annotation.xlsx, [2] sample_200_for_annotation.xlsx
    in data/validation/hasil labeling/
    """
    if annotations_dir is None:
        annotations_dir = ROOT_DIR / "data" / "validation" / "hasil labeling"

    rater_files = [
        annotations_dir / "[1] sample_200_for_annotation.xlsx",
        annotations_dir / "[2] sample_200_for_annotation.xlsx",
    ]

    # Check if annotation files exist
    existing_files = [f for f in rater_files if f.exists()]
    if len(existing_files) < 2:
        logger.warning(
            f"Need at least 2 rater annotation files. Found {len(existing_files)}. "
            "Skipping inter-rater reliability computation."
        )
        return {"status": "pending_annotations", "files_found": len(existing_files)}

    # Load annotations
    raters = []
    for f in existing_files:
        if f.suffix == ".xlsx":
            df = pd.read_excel(f, sheet_name="Annotation")
        else:
            df = pd.read_csv(f)
            
        if "human_label" not in df.columns:
            logger.error(f"Column 'human_label' not found in {f}")
            continue
        raters.append(df)

    if len(raters) < 2:
        return {"status": "invalid_format", "message": "human_label column missing"}

    # Align by article_id
    merged = raters[0][["article_id", "human_label"]].rename(columns={"human_label": "rater1"})
    for i, r in enumerate(raters[1:], start=2):
        merged = merged.merge(
            r[["article_id", "human_label"]].rename(columns={"human_label": f"rater{i}"}),
            on="article_id",
            how="inner",
        )

    # Encode labels to numeric
    label_map = {"SUPPLYSHOCK": 0, "DEMANDSHOCK": 1, "PRICEREPORT": 2, "IRRELEVANT": 3}
    rater_cols = [c for c in merged.columns if c.startswith("rater")]

    for col in rater_cols:
        merged[col + "_num"] = merged[col].map(label_map)

    # Krippendorff's alpha
    try:
        import krippendorff
        reliability_data = merged[[c + "_num" for c in rater_cols]].values.T
        alpha = krippendorff.alpha(
            reliability_data=reliability_data,
            level_of_measurement="nominal",
        )
    except ImportError:
        logger.warning("krippendorff package not installed. Skipping alpha calculation.")
        alpha = None

    # Cohen's kappa pairwise
    from sklearn.metrics import cohen_kappa_score
    kappa_results = {}
    for i in range(len(rater_cols)):
        for j in range(i + 1, len(rater_cols)):
            key = f"{rater_cols[i]}_vs_{rater_cols[j]}"
            valid_idx = merged[rater_cols[i]].notna() & merged[rater_cols[j]].notna()
            labels_i = merged.loc[valid_idx, rater_cols[i]].astype(str).values
            labels_j = merged.loc[valid_idx, rater_cols[j]].astype(str).values
            if len(labels_i) == 0:
                continue
            kappa = cohen_kappa_score(labels_i, labels_j)
            kappa_results[key] = round(kappa, 4)

    results = {
        "n_articles": len(merged),
        "n_raters": len(rater_cols),
        "krippendorff_alpha": round(alpha, 4) if alpha is not None else None,
        "cohen_kappa_pairwise": kappa_results,
        "alpha_threshold": 0.67,
        "kappa_threshold": 0.61,
        "alpha_pass": alpha >= 0.67 if alpha is not None else None,
    }

    logger.info(f"Krippendorff's alpha: {results['krippendorff_alpha']}")
    for k, v in kappa_results.items():
        logger.info(f"Cohen's kappa ({k}): {v}")

    return results


# ── Step 4: GPT vs Human Consensus ──────────────────────────

def compute_gpt_vs_human(sample_df: pd.DataFrame, annotations_dir: Path = None) -> dict:
    """
    Compare GPT predictions against human consensus (majority vote).
    Target: macro-F1 >= 0.75
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

    # Load and align
    raters = []
    for f in existing_files:
        if f.suffix == ".xlsx":
            raters.append(pd.read_excel(f, sheet_name="Annotation"))
        else:
            raters.append(pd.read_csv(f))
    labels_matrix = pd.DataFrame({"article_id": raters[0]["article_id"]})
    for i, r in enumerate(raters):
        labels_matrix[f"rater{i+1}"] = r["human_label"]

    # Majority vote
    rater_cols = [c for c in labels_matrix.columns if c.startswith("rater")]

    def majority_vote(row):
        from collections import Counter
        votes = [row[c] for c in rater_cols if pd.notna(row[c])]
        if not votes:
            return "IRRELEVANT"
        counter = Counter(votes)
        return counter.most_common(1)[0][0]

    labels_matrix["human_consensus"] = labels_matrix.apply(majority_vote, axis=1)

    # Merge with GPT labels
    merged = labels_matrix.merge(
        sample_df[["article_id", "label"]].rename(columns={"label": "gpt_label"}),
        on="article_id",
        how="inner",
    )

    # Compute metrics
    from sklearn.metrics import f1_score, confusion_matrix, classification_report

    y_true = merged["human_consensus"].fillna("IRRELEVANT").astype(str).values
    y_pred = merged["gpt_label"].fillna("IRRELEVANT").astype(str).values

    labels_list = ["SUPPLYSHOCK", "DEMANDSHOCK", "PRICEREPORT", "IRRELEVANT"]

    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels_list, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=labels_list, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels_list)
    report = classification_report(y_true, y_pred, labels=labels_list, output_dict=True, zero_division=0)

    results = {
        "n_articles": len(merged),
        "macro_f1": round(float(macro_f1), 4),
        "per_class_f1": {
            label: round(float(f1), 4)
            for label, f1 in zip(labels_list, per_class_f1)
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "f1_threshold": 0.75,
        "f1_pass": macro_f1 >= 0.75,
    }

    logger.info(f"GPT vs Human macro-F1: {results['macro_f1']}")
    for label, f1 in results["per_class_f1"].items():
        logger.info(f"  {label}: F1={f1}")

    return results


# ── Main ─────────────────────────────────────────────────────

def run_human_validation(cfg: dict = None) -> None:
    """
    Run the full human validation pipeline.
    Phase 1: Create sample and export for annotation
    Phase 2: (After annotation) Compute metrics
    """
    if cfg is None:
        cfg, _ = get_cfg_and_logger()

    input_path = ROOT_DIR / "data" / "processed" / "extractions_clean.parquet"

    if not input_path.exists():
        logger.error(f"extractions_clean.parquet not found: {input_path}")
        logger.error("Run m5_parse_qc.py first.")
        return

    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} articles from extractions_clean.parquet")

    # Phase 1: Create sample and export (skip if already exists)
    jsonl_path = ROOT_DIR / "data" / "validation" / "sample_200.jsonl"
    if jsonl_path.exists():
        sample = pd.read_json(jsonl_path, lines=True)
        logger.info(f"Loaded existing validation sample from {jsonl_path}")
    else:
        sample = create_validation_sample(df, n=200)
        if not sample.empty:
            export_for_annotation(sample)

    # Phase 2: Compute metrics (if annotations exist)
    irr_results = compute_inter_rater_reliability()
    gpt_results = compute_gpt_vs_human(sample)

    # Compile validation report
    report = {
        "n_sample": len(sample),
        "inter_rater": irr_results,
        "gpt_vs_human": gpt_results,
    }

    # Determine overall pass/fail
    alpha_pass = irr_results.get("alpha_pass")
    f1_pass = gpt_results.get("f1_pass")
    if alpha_pass is not None and f1_pass is not None:
        report["pass"] = alpha_pass and f1_pass
    else:
        report["pass"] = None
        report["notes"] = "Waiting for human annotations to complete metrics."

    # Save report
    report_path = ROOT_DIR / "data" / "validation" / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Validation report saved: {report_path}")

    # Recommendations if failed
    if report.get("pass") is False:
        logger.warning("=" * 50)
        logger.warning("VALIDATION FAILED — ACTION REQUIRED:")
        if not irr_results.get("alpha_pass", True):
            logger.warning(f"  Krippendorff alpha ({irr_results.get('krippendorff_alpha')}) < 0.67")
        if not gpt_results.get("f1_pass", True):
            logger.warning(f"  Macro-F1 ({gpt_results.get('macro_f1')}) < 0.75")
        logger.warning("  1. Lakukan error analysis: ambil 20 artikel salah klasifikasi")
        logger.warning("  2. Revisi prompt (tambah contoh boundary cases)")
        logger.warning("  3. Ulangi validasi pada 100 artikel baru")
        logger.warning("  4. JANGAN lanjutkan ke M7 sebelum threshold terpenuhi")
        logger.warning("=" * 50)


if __name__ == "__main__":
    cfg, log = get_cfg_and_logger(skip_env=True)
    run_human_validation(cfg)
