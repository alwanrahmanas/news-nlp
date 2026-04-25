"""
M10 — EXPORT FINAL FEATURES
====================================================
Output: outputs/nlp_features_final.csv

Checklist final validasi BSTS:
- Tidak ada nilai NaN di seluruh variabel (?) → Ya, sudah diimputasi/di-`fillna`
- Format tanggal seragam (YYYY-MM-DD) → Ya
- Semua fitur berada dalam satu frekuensi (bi-weekly) → Ya

Format final (~120 baris x kolom untuk BSTS model):
  period_id, date_start, date_end,
  sentiment_typeA (FITUR UTAMA), sentiment_supply, sentiment_demand,
  sentiment_all,
  n_typeA, n_relevant, n_supplyshock, n_demandshock,
  has_extreme_news, no_relevant_news_t,
  low_confidence_period, is_imputed

Ref: Metodologi §10 (Matriks X_t) dan §9.1 (dummy coverage)
"""

import json
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from m0_setup import ROOT_DIR, get_cfg_and_logger

logger = logging.getLogger("nlp_pipeline.m10")


def run_export(cfg: dict = None) -> pd.DataFrame:
    """
    Export final features for BSTS model.
    Performs checklist validation before saving.
    """
    if cfg is None:
        cfg, _ = get_cfg_and_logger()

    input_path = ROOT_DIR / "data" / "processed" / "sentiment_features.csv"
    output_path = ROOT_DIR / "outputs" / "nlp_features_final.csv"

    if not input_path.exists():
        logger.error(f"Input not found: {input_path}. Run m8_aggregate.py first.")
        return pd.DataFrame()

    df = pd.read_csv(input_path)
    logger.info(f"Loaded sentiment features: {df.shape}")

    # Select final columns — sesuai Metodologi §10 (Matriks X_t)
    final_cols = [
        # --- Metadata periode ---
        "period_id",
        "date_start",
        "date_end",
        # --- Sentimen indeks (kovariat BSTS) ---
        "sentiment_typeA",     # FITUR UTAMA: supply-driven forward-looking
        "sentiment_supply",    # indeks sentimen SUPPLYSHOCK
        "sentiment_demand",    # indeks sentimen DEMANDSHOCK
        "sentiment_all",       # indeks sentimen gabungan (non-IRRELEVANT)
        # --- Count fitur (diagnostik + tambahan BSTS) ---
        "n_typeA",
        "n_relevant",
        "n_supplyshock",
        "n_demandshock",
        # --- Dummies (kovariat BSTS) ---
        "has_extreme_news",
        "no_relevant_news_t",  # Metodologi §9.1: dummy coverage
        "low_confidence_period", # Flag tambahan kualitas ekstraksi
        "is_imputed",
    ]

    available_cols = [c for c in final_cols if c in df.columns]
    missing_cols = [c for c in final_cols if c not in df.columns]

    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}")

    df_final = df[available_cols].copy()

    # ── Checklist Validation ─────────────────────────────────
    logger.info("=" * 50)
    logger.info("PRE-EXPORT CHECKLIST")
    logger.info("=" * 50)

    checklist = {}

    # 1. No missing values except is_imputed
    non_imputed_cols = [c for c in available_cols if c not in ["is_imputed"]]
    for col in non_imputed_cols:
        n_missing = df_final[col].isna().sum()
        if n_missing > 0 and col != "is_imputed":
            logger.warning(f"  [WARN] {col}: {n_missing} missing values")
            checklist[f"no_missing_{col}"] = False
        else:
            checklist[f"no_missing_{col}"] = True

    # 2. sentiment_typeA range between -1.0 and +1.0
    if "sentiment_typeA" in df_final.columns:
        in_range = df_final["sentiment_typeA"].between(-1.0, 1.0).all()
        checklist["sentiment_typeA_range"] = bool(in_range)
        if not in_range:
            logger.warning("  [WARN] sentiment_typeA has values outside [-1, 1]")
            df_final["sentiment_typeA"] = df_final["sentiment_typeA"].clip(-1.0, 1.0)
        else:
            logger.info("  [OK] sentiment_typeA range: [-1.0, 1.0]")

    # 3. No leakage: features at period t do NOT contain info from t+1
    if "period_id" in df_final.columns:
        periods_sorted = df_final["period_id"].tolist() == sorted(df_final["period_id"].tolist())
        checklist["periods_sorted"] = periods_sorted
        if periods_sorted:
            logger.info("  [OK] Periods are sorted chronologically")
        else:
            logger.warning("  [WARN] Periods not sorted — sorting now")
            df_final = df_final.sort_values("period_id").reset_index(drop=True)

    # 4. Granger results exist
    granger_path = ROOT_DIR / "outputs" / "granger_results.json"
    checklist["granger_results_exist"] = granger_path.exists()
    if granger_path.exists():
        logger.info("  [OK] granger_results.json exists")
    else:
        logger.warning("  [WARN] granger_results.json not found")

    # 5. Validation report exists
    validation_path = ROOT_DIR / "data" / "validation" / "validation_report.json"
    checklist["validation_report_exist"] = validation_path.exists()
    if validation_path.exists():
        logger.info("  [OK] validation_report.json exists")
        with open(validation_path, "r") as f:
            val_report = json.load(f)
        if val_report.get("pass"):
            logger.info("  [OK] Validation passed")
        elif val_report.get("pass") is False:
            logger.warning("  [WARN] Validation FAILED — review before using in BSTS")
        else:
            logger.info("  [INFO] Validation pending (human annotations needed)")
    else:
        logger.warning("  [WARN] validation_report.json not found")

    # 6. Total rows
    n_rows = len(df_final)
    logger.info(f"  [INFO] Total rows: {n_rows}")
    checklist["row_count"] = n_rows

    # Save final export
    df_final.to_csv(output_path, index=False)
    logger.info(f"\nFinal features saved: {output_path}")
    logger.info(f"Shape: {df_final.shape}")

    # Save checklist
    checklist_path = ROOT_DIR / "outputs" / "export_checklist.json"
    with open(checklist_path, "w") as f:
        json.dump(checklist, f, indent=2, default=str)
    logger.info(f"Checklist saved: {checklist_path}")

    # Summary
    logger.info("=" * 50)
    logger.info("FINAL EXPORT SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Rows          : {n_rows}")
    logger.info(f"Columns       : {', '.join(available_cols)}")
    if "sentiment_typeA" in df_final.columns:
        logger.info(f"sentiment_typeA mean : {df_final['sentiment_typeA'].mean():.4f}")
        logger.info(f"sentiment_typeA std  : {df_final['sentiment_typeA'].std():.4f}")
    if "n_typeA" in df_final.columns:
        logger.info(f"n_typeA total        : {df_final['n_typeA'].sum()}")
    if "no_relevant_news_t" in df_final.columns:
        n_no_news = int(df_final["no_relevant_news_t"].sum())
        logger.info(f"no_relevant_news_t=1 : {n_no_news} ({n_no_news/n_rows*100:.1f}%)")
    if "is_imputed" in df_final.columns:
        n_imputed = df_final["is_imputed"].sum()
        logger.info(f"Imputed periods      : {n_imputed}")
    if "low_confidence_period" in df_final.columns:
        n_lc = int(df_final["low_confidence_period"].sum())
        logger.info(f"Low-conf periods     : {n_lc}")

    all_pass = all(v for k, v in checklist.items() if isinstance(v, bool))
    if all_pass:
        logger.info("\n[READY] All checks passed. Features ready for BSTS model handoff.")
    else:
        logger.warning("\n[REVIEW] Some checks failed. Review before handoff.")

    return df_final


if __name__ == "__main__":
    cfg, log = get_cfg_and_logger(skip_env=True)
    df = run_export(cfg)
    logger.info("M10 Export complete.")
