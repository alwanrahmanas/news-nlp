"""
M7 — DEKOMPOSISI TIPE A vs TIPE B
====================================
Input:  data/processed/extractions_clean.parquet
Output: data/processed/typeAB_labels.parquet

Scope: Hanya artikel SUPPLYSHOCK atau DEMANDSHOCK
  - Tipe A (Prediktif/Forward-looking): mendahului pergerakan harga
  - Tipe B (Reaktif/Backward-looking): mengikuti pergerakan harga
"""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np

from m0_setup import ROOT_DIR, get_cfg_and_logger
from m3_prompts import build_prompt, validate_typeab_schema

logger = logging.getLogger("nlp_pipeline.m7")


# ── Rule-based Keywords ──────────────────────────────────────

FORWARD_KEYWORDS = [
    "dikhawatirkan", "berpotensi", "diprediksi", "diprakirakan",
    "waspada", "antisipasi", "jika berlanjut", "terancam",
    "petani khawatir", "panen terancam", "bisa naik", "akan naik",
    "peringatan", "potensi kenaikan", "prospek harga",
    "diperkirakan", "mengancam", "potensi", "bakal",
]

BACKWARD_KEYWORDS = [
    "akibat", "karena", "disebabkan", "setelah", "pasca",
    "dampak dari", "imbas", "efek dari", "sebagai akibat",
    "menyebabkan", "berdampak", "telah naik", "sudah naik",
    "kemarin", "pekan lalu", "bulan lalu",
]


# ── Step 1: Rule-based Pre-filter ────────────────────────────

def rule_based_typeAB(text: str) -> tuple:
    """
    Returns ('A', score), ('B', score), or (None, 0) for ambiguous cases.
    Score indicates strength of rule-based classification.
    """
    text_lower = text.lower()

    forward_matches = [kw for kw in FORWARD_KEYWORDS if kw in text_lower]
    backward_matches = [kw for kw in BACKWARD_KEYWORDS if kw in text_lower]

    n_forward = len(forward_matches)
    n_backward = len(backward_matches)

    if n_forward > 0 and n_backward == 0:
        confidence = min(0.5 + n_forward * 0.1, 0.95)
        return "A", confidence, ", ".join(forward_matches)
    elif n_backward > 0 and n_forward == 0:
        confidence = min(0.5 + n_backward * 0.1, 0.95)
        return "B", confidence, ", ".join(backward_matches)
    elif n_forward > n_backward and n_forward >= 2:
        confidence = min(0.5 + (n_forward - n_backward) * 0.05, 0.7)
        return "A", confidence, ", ".join(forward_matches)
    elif n_backward > n_forward and n_backward >= 2:
        confidence = min(0.5 + (n_backward - n_forward) * 0.05, 0.7)
        return "B", confidence, ", ".join(backward_matches)
    else:
        return None, 0.0, ""


# ── Step 2: LLM for Ambiguous Cases ─────────────────────────

def classify_typeAB_llm(article: dict, client, cfg: dict) -> dict:
    """
    Use GPT-4o-mini to classify ambiguous articles.
    """
    prompt = build_prompt(article, "typeab")

    for attempt in range(cfg["llm"]["max_retries"]):
        try:
            response = client.chat.completions.create(
                model=cfg["llm"]["model"],
                temperature=cfg["llm"]["temperature"],
                max_tokens=150,
                messages=[
                    {
                        "role": "system",
                        "content": "Kamu analis temporal berita pangan. Kembalikan HANYA JSON valid."
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            raw_output = response.choices[0].message.content.strip()

            # Parse JSON
            json_str = raw_output
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()

            parsed = json.loads(json_str)

            # Validate
            validate_typeab_schema(parsed)

            return {
                "type_ab": parsed["type_ab"],
                "ab_confidence": float(parsed["ab_confidence"]),
                "temporal_signal": parsed["temporal_signal"],
                "method": "llm",
            }

        except json.JSONDecodeError:
            if attempt < cfg["llm"]["max_retries"] - 1:
                time.sleep(cfg["llm"]["retry_delay_sec"])
        except Exception as e:
            logger.error(f"TypeAB LLM error: {e}")
            time.sleep(cfg["llm"]["retry_delay_sec"] * (attempt + 1))

    # Fallback
    return {
        "type_ab": "B",
        "ab_confidence": 0.3,
        "temporal_signal": "classification_failed",
        "method": "fallback",
    }


# ── Main Pipeline ────────────────────────────────────────────

def run_typeAB_classification(cfg: dict = None, use_llm: bool = True) -> pd.DataFrame:
    """
    Run Type A/B decomposition pipeline.
    """
    if cfg is None:
        cfg, _ = get_cfg_and_logger()

    input_path = ROOT_DIR / "data" / "processed" / "extractions_clean.parquet"
    output_path = ROOT_DIR / "data" / "processed" / "typeAB_labels.parquet"

    if not input_path.exists():
        logger.error(f"Input not found: {input_path}. Run m5_parse_qc.py first.")
        return pd.DataFrame()

    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} articles")

    # Initialize type_ab columns
    df["type_ab"] = "N/A"
    df["ab_confidence"] = 0.0
    df["temporal_signal"] = ""
    df["ab_method"] = ""

    # Filter to SUPPLYSHOCK/DEMANDSHOCK only
    mask_shock = df["label"].isin(["SUPPLYSHOCK", "DEMANDSHOCK"])
    shock_df = df[mask_shock].copy()
    logger.info(f"SUPPLYSHOCK + DEMANDSHOCK articles: {len(shock_df)}")

    if len(shock_df) == 0:
        logger.warning("No SUPPLYSHOCK/DEMANDSHOCK articles found. Saving with all N/A.")
        df.to_parquet(output_path, index=False)
        return df

    # Step 1: Rule-based classification
    rule_results = []
    for idx, row in shock_df.iterrows():
        text = row.get("full_text", "") + " " + row.get("title", "")
        type_ab, confidence, signal = rule_based_typeAB(text)
        rule_results.append({
            "idx": idx,
            "type_ab": type_ab,
            "ab_confidence": confidence,
            "temporal_signal": signal,
        })

    # Apply rule-based results
    n_rule_a = 0
    n_rule_b = 0
    n_ambiguous = 0

    ambiguous_indices = []

    for result in rule_results:
        idx = result["idx"]
        if result["type_ab"] is not None:
            df.at[idx, "type_ab"] = result["type_ab"]
            df.at[idx, "ab_confidence"] = result["ab_confidence"]
            df.at[idx, "temporal_signal"] = result["temporal_signal"]
            df.at[idx, "ab_method"] = "rule"
            if result["type_ab"] == "A":
                n_rule_a += 1
            else:
                n_rule_b += 1
        else:
            ambiguous_indices.append(idx)
            n_ambiguous += 1

    logger.info(f"Rule-based: A={n_rule_a}, B={n_rule_b}, Ambiguous={n_ambiguous}")

    # Step 2: LLM for ambiguous cases
    if use_llm and ambiguous_indices:
        logger.info(f"Running LLM classification for {len(ambiguous_indices)} ambiguous articles...")

        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            from tqdm import tqdm
            for idx in tqdm(ambiguous_indices, desc="TypeAB LLM"):
                row = df.loc[idx]
                article = {
                    "full_text": row.get("full_text", ""),
                    "title": row.get("title", ""),
                }
                result = classify_typeAB_llm(article, client, cfg)

                df.at[idx, "type_ab"] = result["type_ab"]
                df.at[idx, "ab_confidence"] = result["ab_confidence"]
                df.at[idx, "temporal_signal"] = result["temporal_signal"]
                df.at[idx, "ab_method"] = result["method"]

                time.sleep(0.5)  # Rate limit

        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            logger.info("Falling back to 'B' for all ambiguous cases")
            for idx in ambiguous_indices:
                df.at[idx, "type_ab"] = "B"
                df.at[idx, "ab_confidence"] = 0.3
                df.at[idx, "temporal_signal"] = "fallback_no_llm"
                df.at[idx, "ab_method"] = "fallback"
    elif ambiguous_indices:
        logger.info("LLM disabled. Setting ambiguous cases to 'B' with low confidence.")
        for idx in ambiguous_indices:
            df.at[idx, "type_ab"] = "B"
            df.at[idx, "ab_confidence"] = 0.3
            df.at[idx, "temporal_signal"] = "rule_ambiguous"
            df.at[idx, "ab_method"] = "fallback"

    # Log distribution
    shock_final = df[mask_shock]
    type_counts = shock_final["type_ab"].value_counts()
    total_shock = len(shock_final)

    logger.info("=" * 50)
    logger.info("TYPE A/B DISTRIBUTION")
    logger.info("=" * 50)
    for t in ["A", "B"]:
        count = type_counts.get(t, 0)
        pct = (count / total_shock * 100) if total_shock > 0 else 0
        logger.info(f"  Tipe {t}: {count:>6,} ({pct:.1f}%)")
    logger.info(f"  Total : {total_shock:>6,}")

    # Sanity check: Tipe A should be 30-40%
    pct_a = (type_counts.get("A", 0) / total_shock * 100) if total_shock > 0 else 0
    if pct_a < 20:
        logger.warning(
            f"Tipe A = {pct_a:.1f}% (< 20%). "
            "Consider reviewing prompt — possible over-labeling to B."
        )
    elif pct_a > 50:
        logger.warning(
            f"Tipe A = {pct_a:.1f}% (> 50%). "
            "Consider reviewing prompt — possible over-labeling to A."
        )

    # Save
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved to {output_path}: {len(df)} rows")

    return df


if __name__ == "__main__":
    cfg, log = get_cfg_and_logger(skip_env=True)
    df = run_typeAB_classification(cfg, use_llm=False)
    logger.info("M7 Type A/B decomposition complete.")
