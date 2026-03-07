"""
M5 — PARSING & QC EKSTRAKSI
=============================
Input:  data/clean/extraction_cache.jsonl + data/clean/corpus_clean.jsonl
Output: data/processed/extractions_clean.parquet

Langkah:
  1. Gabungkan metadata artikel + hasil ekstraksi
  2. Filter confidence rendah
  3. Filter status failed
  4. Validasi range nilai
  5. Statistik distribusi label
  6. Simpan ke parquet
"""

import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from m0_setup import ROOT_DIR, get_cfg_and_logger

logger = logging.getLogger("nlp_pipeline.m5")


def load_jsonl(filepath: Path) -> list:
    """Load all records from a JSONL file."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_parse_qc(cfg: dict = None) -> pd.DataFrame:
    """Run the full parsing and QC pipeline."""
    if cfg is None:
        cfg, _ = get_cfg_and_logger()

    corpus_path = ROOT_DIR / "data" / "clean" / "corpus_clean.jsonl"
    cache_path = ROOT_DIR / "data" / "clean" / "extraction_cache.jsonl"
    output_path = ROOT_DIR / "data" / "processed" / "extractions_clean.parquet"

    # Load data
    if not corpus_path.exists():
        logger.error(f"corpus_clean.jsonl not found: {corpus_path}")
        return pd.DataFrame()

    if not cache_path.exists():
        logger.error(f"extraction_cache.jsonl not found: {cache_path}")
        return pd.DataFrame()

    corpus = load_jsonl(corpus_path)
    cache = load_jsonl(cache_path)

    logger.info(f"Corpus articles: {len(corpus)}")
    logger.info(f"Extraction cache entries: {len(cache)}")

    # Step 1: Merge corpus metadata + extraction results
    corpus_df = pd.DataFrame(corpus)
    cache_df = pd.DataFrame(cache)

    # Flatten extraction dict into columns
    if "extraction" in cache_df.columns:
        extraction_expanded = pd.json_normalize(cache_df["extraction"])
        cache_flat = pd.concat([
            cache_df.drop(columns=["extraction"]),
            extraction_expanded
        ], axis=1)
    else:
        cache_flat = cache_df

    # Rename status column to avoid confusion
    if "status" in cache_flat.columns:
        cache_flat = cache_flat.rename(columns={"status": "extraction_status"})

    # Merge
    df = corpus_df.merge(
        cache_flat,
        on="article_id",
        how="inner",
        suffixes=("", "_ext"),
    )
    logger.info(f"After merge: {len(df)} articles")

    # Step 2: Filter confidence < threshold -> relabel to IRRELEVANT
    threshold = cfg["sentiment"]["confidence_threshold"]
    low_conf_mask = df["confidence"] < threshold
    n_low_conf = low_conf_mask.sum()
    df.loc[low_conf_mask, "label"] = "IRRELEVANT"
    logger.info(f"Articles relabeled to IRRELEVANT (confidence < {threshold}): {n_low_conf}")

    # Step 3: Filter failed extractions
    if "extraction_status" in df.columns:
        n_failed = (df["extraction_status"] != "success").sum()
        df = df[df["extraction_status"] == "success"]
        logger.info(f"Failed extractions removed: {n_failed}")

    # Step 4: Validate value ranges
    # Sentiment score
    if "sentiment_score" in df.columns:
        out_of_range = ~df["sentiment_score"].between(-1.0, 1.0)
        if out_of_range.any():
            logger.warning(f"Clamping {out_of_range.sum()} sentiment_score values to [-1, 1]")
            df["sentiment_score"] = df["sentiment_score"].clip(-1.0, 1.0)

    # Confidence
    if "confidence" in df.columns:
        out_of_range = ~df["confidence"].between(0.0, 1.0)
        if out_of_range.any():
            logger.warning(f"Clamping {out_of_range.sum()} confidence values to [0, 1]")
            df["confidence"] = df["confidence"].clip(0.0, 1.0)

    # Label validation
    valid_labels = cfg["labels"]["valid"]
    if "label" in df.columns:
        invalid_labels = ~df["label"].isin(valid_labels)
        if invalid_labels.any():
            logger.warning(f"Invalid labels found ({invalid_labels.sum()}), setting to IRRELEVANT:")
            logger.warning(f"  {df.loc[invalid_labels, 'label'].value_counts().to_dict()}")
            df.loc[invalid_labels, "label"] = "IRRELEVANT"

    # Step 5: Label distribution statistics
    logger.info("=" * 50)
    logger.info("LABEL DISTRIBUTION")
    logger.info("=" * 50)
    if "label" in df.columns:
        label_counts = df["label"].value_counts()
        total = len(df)
        for label in valid_labels:
            count = label_counts.get(label, 0)
            pct = (count / total * 100) if total > 0 else 0
            logger.info(f"  {label:15s}: {count:>6,} ({pct:.1f}%)")
        logger.info(f"  {'Total':15s}: {total:>6,}")

    # Sentiment score distribution
    if "sentiment_score" in df.columns:
        logger.info(f"\nSentiment score stats:")
        logger.info(f"  Mean  : {df['sentiment_score'].mean():.3f}")
        logger.info(f"  Std   : {df['sentiment_score'].std():.3f}")
        logger.info(f"  Min   : {df['sentiment_score'].min():.3f}")
        logger.info(f"  Max   : {df['sentiment_score'].max():.3f}")

    # Step 6: Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Select columns to save
    cols_to_keep = [
        "article_id", "url", "source", "title", "published_date",
        "full_text", "char_length", "keyword_matched",
        "label", "sentiment_score", "confidence", "commodities",
        "supply_location", "is_forward_looking", "rationale",
        "model", "tokens_prompt", "tokens_completion",
        "extraction_status",
    ]
    cols_available = [c for c in cols_to_keep if c in df.columns]
    df_out = df[cols_available].copy()

    # Convert published_date to datetime
    if "published_date" in df_out.columns:
        df_out["published_date"] = pd.to_datetime(df_out["published_date"], errors="coerce")

    df_out.to_parquet(output_path, index=False)
    logger.info(f"\nSaved to {output_path}: {len(df_out)} rows, {len(cols_available)} columns")

    return df_out


if __name__ == "__main__":
    cfg, log = get_cfg_and_logger(skip_env=True)
    df = run_parse_qc(cfg)
    if not df.empty:
        logger.info("M5 Parse & QC complete.")
    else:
        logger.warning("M5 produced empty output. Check input data.")
