"""
M8 — AGREGASI TIME-SERIES BI-MINGGUAN
========================================
Input:  data/processed/typeAB_labels.parquet
Output: data/processed/sentiment_features.csv

Features per biweekly period:
  - sentiment indices (weighted with exponential decay)
  - article counts
  - intensity measures
  - extreme news dummies
  - no_relevant_news_t: dummy coverage (Metodologi §9.1)
"""

import logging
from pathlib import Path
from datetime import timedelta

import pandas as pd
import numpy as np

from m0_setup import ROOT_DIR, get_cfg_and_logger

logger = logging.getLogger("nlp_pipeline.m8")


# ── Biweekly Period Assignment ───────────────────────────────

def assign_biweek_period(date: pd.Timestamp, anchor: pd.Timestamp) -> str:
    """
    Hitung periode bi-mingguan sejak anchor date.
    Returns: string "BW0001", "BW0002", ...
    1 tahun = 26 periode bi-mingguan.
    """
    if pd.isna(date):
        return None

    delta_days = (date - anchor).days
    if delta_days < 0:
        return None
    period_num = delta_days // 14 + 1
    return f"BW{period_num:04d}"


def get_biweek_date_range(period_id: str, anchor: pd.Timestamp) -> tuple:
    """Get start and end dates for a biweekly period."""
    period_num = int(period_id.replace("BW", ""))
    start = anchor + timedelta(days=(period_num - 1) * 14)
    end = start + timedelta(days=13)
    return start, end


# ── Weighted Sentiment Index ─────────────────────────────────

def weighted_sentiment_index(group: pd.DataFrame, lambda_decay: float = 0.3) -> float:
    """
    Compute confidence-weighted sentiment index with exponential decay.
    More recent articles in the period get higher weights.

    Formula: w_i = confidence_i * exp(-lambda * days_from_end_of_period)
    Index = sum(w_i * sentiment_i) / sum(w_i)
    """
    if len(group) == 0:
        return 0.0

    period_end = group["published_date"].max()
    days_from_end = (period_end - group["published_date"]).dt.days.astype(float)
    weights = group["confidence"] * np.exp(-lambda_decay * days_from_end)

    total_weight = weights.sum()
    if total_weight == 0:
        return 0.0

    return (weights * group["sentiment_score"]).sum() / total_weight


# ── Aggregation Functions ────────────────────────────────────

def compute_period_features(group: pd.DataFrame, lambda_decay: float = 0.3) -> dict:
    """Compute all features for a single biweekly period."""

    features = {}

    # --- SENTIMENT INDICES ---
    # All labels except IRRELEVANT
    relevant = group[group["label"] != "IRRELEVANT"]
    features["sentiment_all"] = weighted_sentiment_index(relevant, lambda_decay) if len(relevant) > 0 else np.nan

    # SUPPLYSHOCK only
    supply = group[group["label"] == "SUPPLYSHOCK"]
    features["sentiment_supply"] = weighted_sentiment_index(supply, lambda_decay) if len(supply) > 0 else np.nan

    # DEMANDSHOCK only
    demand = group[group["label"] == "DEMANDSHOCK"]
    features["sentiment_demand"] = weighted_sentiment_index(demand, lambda_decay) if len(demand) > 0 else np.nan

    # Type A only (key feature)
    type_a = group[group["type_ab"] == "A"]
    features["sentiment_typeA"] = weighted_sentiment_index(type_a, lambda_decay) if len(type_a) > 0 else np.nan

    # Type B only
    type_b = group[group["type_ab"] == "B"]
    features["sentiment_typeB"] = weighted_sentiment_index(type_b, lambda_decay) if len(type_b) > 0 else np.nan

    # --- COUNT FEATURES ---
    features["n_articles_total"] = len(group)
    features["n_relevant"] = len(relevant)   # non-IRRELEVANT articles
    features["n_supplyshock"] = len(supply)
    features["n_demandshock"] = len(demand)
    features["n_pricereport"] = len(group[group["label"] == "PRICEREPORT"])
    features["n_typeA"] = len(type_a)

    # --- INTENSITY FEATURES ---
    if len(relevant) > 0:
        features["max_neg_sentiment"] = relevant["sentiment_score"].min()
        neg_articles = relevant[relevant["sentiment_score"] < -0.3]
        features["prop_negative_articles"] = len(neg_articles) / len(relevant)
    else:
        features["max_neg_sentiment"] = 0.0
        features["prop_negative_articles"] = 0.0

    # --- DUMMIES ---
    extreme_mask = (group["label"] == "SUPPLYSHOCK") & (group["sentiment_score"] < -0.7)
    features["has_extreme_news"] = 1 if extreme_mask.any() else 0

    # no_relevant_news_t: dummy coverage (Metodologi §9.1, Langkah C)
    # Membedakan "sentimen netral karena tenang" vs "netral karena data hilang"
    # Spike-slab akan memutuskan apakah dummy ini informatif
    features["no_relevant_news_t"] = 1 if len(relevant) == 0 else 0

    return features


# ── Main Pipeline ────────────────────────────────────────────

def run_aggregation(cfg: dict = None) -> pd.DataFrame:
    """Run the full aggregation pipeline."""
    if cfg is None:
        cfg, _ = get_cfg_and_logger()

    input_path = ROOT_DIR / "data" / "processed" / "typeAB_labels.parquet"
    output_path = ROOT_DIR / "data" / "processed" / "sentiment_features.csv"

    if not input_path.exists():
        logger.error(f"Input not found: {input_path}. Run m7_typeAB.py first.")
        return pd.DataFrame()

    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} articles")

    # Ensure published_date is datetime
    df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")

    # Drop rows with no date
    no_date = df["published_date"].isna().sum()
    if no_date > 0:
        logger.warning(f"Dropping {no_date} articles with no published_date")
        df = df.dropna(subset=["published_date"])

    # Anchor date
    anchor = pd.Timestamp(cfg["aggregation"]["biweek_anchor"])
    lambda_decay = cfg["aggregation"]["decay_lambda"]
    neutral_value = cfg["aggregation"]["neutral_impute_value"]

    # Coverage quality thresholds (from config, with fallback defaults)
    agg_cfg = cfg["aggregation"]
    min_relevant  = agg_cfg.get("min_relevant_articles", 2)
    sparse_thresh = agg_cfg.get("sparse_threshold", 4)
    dense_thresh  = agg_cfg.get("dense_threshold", 15)
    analysis_start = pd.Timestamp(agg_cfg.get("analysis_period_start", "2024-01-01"))

    # Assign biweekly periods
    df["period_id"] = df["published_date"].apply(lambda d: assign_biweek_period(d, anchor))
    df = df.dropna(subset=["period_id"])

    logger.info(f"Period range: {df['period_id'].min()} to {df['period_id'].max()}")
    logger.info(f"Unique periods with data: {df['period_id'].nunique()}")

    # Generate complete period list (BW0001 to expected last)
    date_end = pd.Timestamp(cfg["corpus"]["date_end"])
    total_days = (date_end - anchor).days
    total_periods = total_days // 14 + 1

    all_periods = [f"BW{i:04d}" for i in range(1, total_periods + 1)]
    logger.info(f"Total expected periods: {len(all_periods)}")

    # Compute features per period
    period_features = []
    for period_id in all_periods:
        period_data = df[df["period_id"] == period_id]

        if len(period_data) > 0:
            features = compute_period_features(period_data, lambda_decay)
        else:
            # Empty period — all NaN (will be imputed later)
            features = {
                "sentiment_all": np.nan,
                "sentiment_supply": np.nan,
                "sentiment_demand": np.nan,
                "sentiment_typeA": np.nan,
                "sentiment_typeB": np.nan,
                "n_articles_total": 0,
                "n_relevant": 0,
                "n_supplyshock": 0,
                "n_demandshock": 0,
                "n_pricereport": 0,
                "n_typeA": 0,
                "max_neg_sentiment": 0.0,
                "prop_negative_articles": 0.0,
                "has_extreme_news": 0,
                "no_relevant_news_t": 1,  # kosong = pasti no relevant news
            }

        # ── Coverage Flag per Period ──────────────────────────────────
        # Dari total artikel, berapa yang relevan (non-IRRELEVANT)?
        # Periode dengan n_relevant < min_relevant adalah "blank spot" meskipun
        # ada artikel — karena semua artikelnya IRRELEVANT.
        n_tot = features.get("n_articles_total", 0)
        n_rel = features.get("n_relevant", 0)

        if n_tot < sparse_thresh:
            features["coverage_flag"] = "SPARSE"
        elif n_tot >= dense_thresh:
            features["coverage_flag"] = "DENSE"
        else:
            features["coverage_flag"] = "NORMAL"

        # low_confidence = periode kosong ATAU semua artikel irrelevant
        # Ini adalah periode yang tidak memiliki sinyal ekonomi nyata
        features["low_confidence_period"] = bool(n_rel < min_relevant)

        # Add period metadata
        start, end = get_biweek_date_range(period_id, anchor)
        features["period_id"] = period_id
        features["date_start"] = start.strftime("%Y-%m-%d")
        features["date_end"] = end.strftime("%Y-%m-%d")
        # is_historical = periode sebelum analysis_period_start (sparse bias tinggi)
        features["is_historical"] = bool(start < analysis_start)

        period_features.append(features)

    result_df = pd.DataFrame(period_features)

    # Imputation: fill NaN sentiment values with neutral
    # PENTING: Untuk low_confidence_period, impute ke neutral (0.0) bukan ke
    # nilai sebelumnya (forward-fill) karena kita tidak punya sinyal valid.
    # Forward-fill HANYA berlaku untuk periode NORMAL yang kebetulan missing.
    sentiment_cols = [
        "sentiment_all", "sentiment_supply", "sentiment_demand",
        "sentiment_typeA", "sentiment_typeB",
    ]
    for col in sentiment_cols:
        # Untuk low-confidence: langsung neutral
        result_df.loc[result_df["low_confidence_period"], col] = \
            result_df.loc[result_df["low_confidence_period"], col].fillna(neutral_value)
        # Untuk periode NORMAL yang NaN (e.g., tidak ada supply shock bulan itu):
        # forward-fill lalu backward-fill sebagai fallback
        result_df.loc[~result_df["low_confidence_period"], col] = \
            result_df.loc[~result_df["low_confidence_period"], col] \
            .ffill().bfill().fillna(neutral_value)

    # Mark imputed periods (benar-benar kosong = 0 artikel)
    result_df["is_imputed"] = result_df["n_articles_total"] == 0

    n_imputed     = result_df["is_imputed"].sum()
    n_low_conf    = result_df["low_confidence_period"].sum()
    n_historical  = result_df["is_historical"].sum()
    pct_imputed   = n_imputed / len(result_df) * 100
    pct_low_conf  = n_low_conf / len(result_df) * 100

    logger.info(f"Imputed periods (0 artikel)     : {n_imputed}/{len(result_df)} ({pct_imputed:.1f}%)")
    logger.info(f"Low-confidence periods          : {n_low_conf}/{len(result_df)} ({pct_low_conf:.1f}%)")
    logger.info(f"Historical periods (pre-{analysis_start.year}): {n_historical}")

    flag_counts = result_df["coverage_flag"].value_counts()
    for flag, cnt in flag_counts.items():
        logger.info(f"  coverage_flag={flag:6s}: {cnt} periods")

    if pct_imputed > 20:
        logger.warning(f"WARNING: {pct_imputed:.1f}% periods imputed (> 20%). Data coverage may be insufficient.")
    if pct_low_conf > 30:
        logger.warning(f"WARNING: {pct_low_conf:.1f}% periods are low-confidence. Pertimbangkan temporal split.")

    # Reorder columns
    col_order = [
        "period_id", "date_start", "date_end",
        "is_historical", "coverage_flag", "low_confidence_period",
        "sentiment_all", "sentiment_supply", "sentiment_demand",
        "sentiment_typeA", "sentiment_typeB",
        "n_articles_total", "n_relevant", "n_supplyshock", "n_demandshock",
        "n_pricereport", "n_typeA",
        "max_neg_sentiment", "prop_negative_articles",
        "has_extreme_news", "no_relevant_news_t",
        "is_imputed",
    ]
    result_df = result_df[[c for c in col_order if c in result_df.columns]]

    # Save
    result_df.to_csv(output_path, index=False)
    logger.info(f"Saved sentiment features: {output_path}")
    logger.info(f"Shape: {result_df.shape}")

    # Summary stats
    logger.info("=" * 50)
    logger.info("AGGREGATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total periods                    : {len(result_df)}")
    logger.info(f"Periods dengan data              : {(~result_df['is_imputed']).sum()}")
    logger.info(f"Periods imputed (kosong)         : {n_imputed}")
    logger.info(f"Periods low-confidence           : {n_low_conf}")
    logger.info(f"Periods historical (pre-cutoff)  : {n_historical}")
    logger.info(f"Periods analitik (post-cutoff)   : {len(result_df) - n_historical}")
    logger.info(f"Mean artikel/period              : {result_df['n_articles_total'].mean():.1f}")
    logger.info(f"Mean relevan/period              : {result_df['n_relevant'].mean():.1f}")
    logger.info(f"Mean sentiment_all               : {result_df['sentiment_all'].mean():.3f}")
    logger.info(f"Mean sentiment_typeA             : {result_df['sentiment_typeA'].mean():.3f}")
    n_no_news = result_df["no_relevant_news_t"].sum()
    logger.info(f"Periods no_relevant_news_t=1     : {n_no_news} ({n_no_news/len(result_df)*100:.1f}%)")

    # Rekomendasi split
    analytic_df = result_df[~result_df["is_historical"]]
    analytic_lc = analytic_df["low_confidence_period"].sum() if len(analytic_df) > 0 else 0
    logger.info("=" * 50)
    logger.info("REKOMENDASI TEMPORAL SPLIT")
    logger.info("=" * 50)
    logger.info(f"  Analysis period start  : {analysis_start.date()}")
    logger.info(f"  Analytic periods total : {len(analytic_df)}")
    logger.info(f"  Analytic low-conf      : {analytic_lc} ({analytic_lc/max(len(analytic_df),1)*100:.1f}%)")
    logger.info("  → Gunakan is_historical=False untuk training/evaluation")
    logger.info("  → Gunakan low_confidence_period=False untuk periode dengan sinyal kuat")

    return result_df


if __name__ == "__main__":
    cfg, log = get_cfg_and_logger(skip_env=True)
    df = run_aggregation(cfg)
    logger.info("M8 Aggregation complete.")
