"""
M7 — DEKOMPOSISI TIPE A vs TIPE B
====================================
Input:  data/processed/extractions_clean.parquet
Output: data/processed/typeAB_labels.parquet

Scope: Hanya artikel SUPPLYSHOCK atau DEMANDSHOCK
  - Tipe A (Prediktif/Forward-looking): mendahului pergerakan harga
  - Tipe B (Reaktif/Backward-looking): mengikuti pergerakan harga

Changelog vs v1:
  - [FIX] tqdm import dipindah ke top-level (bukan di dalam try/except)
  - [FIX] Fallback ke "UNKNOWN" bukan "B" untuk hindari bias sistematis
  - [FIX] Checkpoint system untuk resume LLM loop jika koneksi putus
  - [FIX] Keyword "karena" & "menyebabkan" dihapus dari BACKWARD_KEYWORDS
  - [FIX] client init dipisah dari LLM loop via try/except/else
  - [NEW] Per-article error handling agar satu artikel gagal tidak hentikan loop
  - [NEW] Post-run sanity check distribusi A/B lebih detail
"""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from tqdm import tqdm  # ← FIX #1: top-level import, bukan di dalam try

from m0_setup import ROOT_DIR, get_cfg_and_logger
from m3_prompts import build_prompt, validate_typeab_schema

logger = logging.getLogger("nlp_pipeline.m7")


# ── Konstanta ────────────────────────────────────────────────

CHECKPOINT_PATH = ROOT_DIR / "data" / "raw" / "checkpoints" / "m7_typeab_checkpoint.json"


# ── Rule-based Keywords ──────────────────────────────────────

# Forward-looking: mengantisipasi kejadian yang BELUM terjadi
FORWARD_KEYWORDS = [
    "dikhawatirkan", "berpotensi", "diprediksi", "diprakirakan",
    "waspada", "antisipasi", "jika berlanjut", "terancam",
    "petani khawatir", "panen terancam", "bisa naik", "akan naik",
    "peringatan", "potensi kenaikan", "prospek harga",
    "diperkirakan", "mengancam", "potensi", "bakal",
    "rawan gagal panen", "ancaman kekeringan", "berisiko",
    "cuaca ekstrem dikhawatirkan",
]

# Backward-looking: melaporkan kejadian yang SUDAH terjadi
# CATATAN: "karena" dan "menyebabkan" SENGAJA dihapus —
#   terlalu umum, muncul di kalimat forward-looking juga.
#   Contoh: "Petani khawatir karena curah hujan berpotensi merusak panen"
#   → ada "karena" tapi artikel ini jelas Tipe A.
BACKWARD_KEYWORDS = [
    "akibat",
    "disebabkan oleh",
    "setelah terjadi",
    "pasca",
    "dampak dari",
    "imbas dari",
    "efek dari",
    "sebagai akibat dari",
    "berdampak pada",
    "telah naik",
    "sudah naik",
    "telah melonjak",
    "sudah melonjak",
    "kemarin",
    "pekan lalu",
    "bulan lalu",
    "minggu lalu",
    "harga sudah",
    "harga telah",
]


# ── Step 1: Rule-based Pre-filter ────────────────────────────

def rule_based_typeAB(text: str) -> tuple:
    """
    Klasifikasi rule-based berdasarkan keyword matching.

    Returns:
        (type_ab, confidence, matched_keywords)
        type_ab: 'A', 'B', atau None (ambiguous → dikirim ke LLM)
    """
    text_lower = text.lower()

    forward_matches = [kw for kw in FORWARD_KEYWORDS if kw in text_lower]
    backward_matches = [kw for kw in BACKWARD_KEYWORDS if kw in text_lower]

    n_forward = len(forward_matches)
    n_backward = len(backward_matches)

    if n_forward > 0 and n_backward == 0:
        # Clear forward signal
        confidence = min(0.5 + n_forward * 0.1, 0.95)
        return "A", confidence, ", ".join(forward_matches)

    elif n_backward > 0 and n_forward == 0:
        # Clear backward signal
        confidence = min(0.5 + n_backward * 0.1, 0.95)
        return "B", confidence, ", ".join(backward_matches)

    elif n_forward > n_backward and n_forward >= 2:
        # Forward dominates
        confidence = min(0.5 + (n_forward - n_backward) * 0.05, 0.7)
        return "A", confidence, ", ".join(forward_matches)

    elif n_backward > n_forward and n_backward >= 2:
        # Backward dominates
        confidence = min(0.5 + (n_backward - n_forward) * 0.05, 0.7)
        return "B", confidence, ", ".join(backward_matches)

    else:
        # Ambiguous → kirim ke LLM
        return None, 0.0, ""


# ── Step 2: LLM for Ambiguous Cases ─────────────────────────

def classify_typeAB_llm(article: dict, client, cfg: dict) -> dict:
    """
    Klasifikasi LLM untuk artikel ambiguous.

    Fallback ke "UNKNOWN" (bukan "B") jika semua retry gagal.
    UNKNOWN akan di-exclude dari analisis di M8, bukan dimasukkan ke Tipe B.
    Ini penting untuk menghindari bias sistematis pada distribusi A/B.
    """
    prompt = build_prompt(article, "typeab")
    provider = cfg["llm"].get("provider", "openai")

    for attempt in range(cfg["llm"]["max_retries"]):
        try:
            if provider == "gemini":
                from google.genai import types as genai_types
                response = client.models.generate_content(
                    model=cfg["llm"]["model"],
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=(
                            "Kamu analis temporal berita pangan. "
                            "Kembalikan HANYA JSON valid."
                        ),
                        max_output_tokens=min(cfg["llm"]["max_completion_tokens"], 500),
                        response_mime_type="application/json",
                    ),
                )
                raw_output = (response.text or "").strip()
            else:
                response = client.chat.completions.create(
                    model=cfg["llm"]["model"],
                    max_completion_tokens=min(cfg["llm"]["max_completion_tokens"], 500),
                    response_format={"type": "json_object"},
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Kamu analis temporal berita pangan. "
                                "Kembalikan HANYA JSON valid."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                raw_output = response.choices[0].message.content.strip()

            # Strip markdown fences jika ada
            json_str = raw_output
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()

            parsed = json.loads(json_str)
            validate_typeab_schema(parsed)

            return {
                "type_ab": parsed["type_ab"],
                "ab_confidence": float(parsed["ab_confidence"]),
                "temporal_signal": parsed["temporal_signal"],
                "method": "llm",
            }

        except json.JSONDecodeError as e:
            logger.debug(f"JSON parse error (attempt {attempt + 1}): {e}")
            if attempt < cfg["llm"]["max_retries"] - 1:
                time.sleep(cfg["llm"]["retry_delay_sec"])

        except Exception as e:
            logger.warning(f"LLM error (attempt {attempt + 1}): {e}")
            time.sleep(cfg["llm"]["retry_delay_sec"] * (attempt + 1))

    # ── FIX #2: Fallback ke UNKNOWN, BUKAN ke "B" ──────────
    # Kenapa? Kalau fallback ke B, setiap artikel yang gagal parse
    # akan menggelembungkan Tipe B secara artifisial.
    # UNKNOWN → di-exclude di M8, tidak masuk ke distribusi A/B.
    logger.warning("All retries exhausted — marking as UNKNOWN (will be excluded)")
    return {
        "type_ab": "UNKNOWN",
        "ab_confidence": 0.0,
        "temporal_signal": "classification_failed",
        "method": "fallback",
    }


# ── Checkpoint Helpers ───────────────────────────────────────

def load_checkpoint() -> dict:
    """Load existing LLM checkpoint jika ada."""
    if CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH, encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Checkpoint loaded: {len(data)} articles already classified")
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Checkpoint corrupted, starting fresh: {e}")
            return {}
    return {}


def save_checkpoint(completed: dict) -> None:
    """Simpan progress checkpoint ke disk."""
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(completed, f, ensure_ascii=False)


def clear_checkpoint() -> None:
    """Hapus checkpoint setelah pipeline selesai."""
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        logger.info("Checkpoint cleared")


# ── Main Pipeline ────────────────────────────────────────────

def run_typeAB_classification(cfg: dict = None, use_llm: bool = True) -> pd.DataFrame:
    """
    Run Type A/B decomposition pipeline.

    Flow:
      1. Rule-based keyword matching → clear cases langsung diklasifikasi
      2. LLM → hanya untuk ambiguous cases (hemat ~75% API calls)
      3. Checkpoint system → resumable jika koneksi putus
      4. UNKNOWN handling → artikel gagal di-exclude, bukan dipaksa ke B
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

    # Inisialisasi kolom type_ab
    df["type_ab"] = "N/A"
    df["ab_confidence"] = 0.0
    df["temporal_signal"] = ""
    df["ab_method"] = ""

    # Filter ke SUPPLYSHOCK/DEMANDSHOCK
    mask_shock = df["label"].isin(["SUPPLYSHOCK", "DEMANDSHOCK"])
    shock_df = df[mask_shock].copy()
    logger.info(f"SUPPLYSHOCK + DEMANDSHOCK articles: {len(shock_df)}")

    if len(shock_df) == 0:
        logger.warning("No SUPPLYSHOCK/DEMANDSHOCK articles found. Saving with all N/A.")
        df.to_parquet(output_path, index=False)
        return df

    # ── Step 1: Rule-based Classification ────────────────────
    n_rule_a, n_rule_b, n_ambiguous = 0, 0, 0
    ambiguous_indices = []

    for idx, row in shock_df.iterrows():
        text = str(row.get("full_text", "")) + " " + str(row.get("title", ""))
        type_ab, confidence, signal = rule_based_typeAB(text)

        if type_ab is not None:
            df.at[idx, "type_ab"] = type_ab
            df.at[idx, "ab_confidence"] = confidence
            df.at[idx, "temporal_signal"] = signal
            df.at[idx, "ab_method"] = "rule"
            if type_ab == "A":
                n_rule_a += 1
            else:
                n_rule_b += 1
        else:
            ambiguous_indices.append(idx)
            n_ambiguous += 1

    logger.info(
        f"Rule-based results → A: {n_rule_a}, B: {n_rule_b}, "
        f"Ambiguous: {n_ambiguous}"
    )

    # ── Step 2: LLM for Ambiguous Cases ──────────────────────
    if ambiguous_indices:
        if use_llm:
            logger.info(
                f"LLM classification for {len(ambiguous_indices)} ambiguous articles..."
            )

            # ── FIX #1: client init DIPISAH dari loop ─────────────
            # Kalau client init gagal, loop tidak perlu jalan sama sekali.
            # tqdm sudah diimport di top-level (bukan di sini).
            client = None
            try:
                provider = cfg["llm"].get("provider", "openai")
                if provider == "gemini":
                    from google import genai
                    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
                else:
                    from openai import OpenAI
                    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                logger.info(f"LLM client initialized: {provider}")

            except Exception as e:
                logger.error(f"Client initialization failed: {e}")
                logger.info("Falling back: all ambiguous → UNKNOWN (will be excluded)")
                for idx in ambiguous_indices:
                    df.at[idx, "type_ab"] = "UNKNOWN"
                    df.at[idx, "ab_confidence"] = 0.0
                    df.at[idx, "temporal_signal"] = "client_init_failed"
                    df.at[idx, "ab_method"] = "fallback"
                client = None

            # ── FIX #3: Checkpoint + per-article error handling ───
            if client is not None:
                completed = load_checkpoint()
                n_resumed = 0

                for idx in tqdm(ambiguous_indices, desc="TypeAB LLM"):
                    str_idx = str(idx)

                    if str_idx in completed:
                        # Resume dari checkpoint
                        result = completed[str_idx]
                        n_resumed += 1
                    else:
                        row = df.loc[idx]
                        article = {
                            "full_text": str(row.get("full_text", "")),
                            "title": str(row.get("title", "")),
                        }
                        # Per-article try/except: satu artikel gagal
                        # tidak menghentikan seluruh loop
                        try:
                            result = classify_typeAB_llm(article, client, cfg)
                        except Exception as e:
                            logger.error(f"Unexpected error for idx={idx}: {e}")
                            result = {
                                "type_ab": "UNKNOWN",
                                "ab_confidence": 0.0,
                                "temporal_signal": "unexpected_error",
                                "method": "fallback",
                            }

                        # Simpan ke checkpoint setelah setiap artikel
                        completed[str_idx] = result
                        save_checkpoint(completed)

                    df.at[idx, "type_ab"] = result["type_ab"]
                    df.at[idx, "ab_confidence"] = result["ab_confidence"]
                    df.at[idx, "temporal_signal"] = result["temporal_signal"]
                    df.at[idx, "ab_method"] = result["method"]

                    time.sleep(0.5)  # rate limiting

                if n_resumed > 0:
                    logger.info(f"Resumed {n_resumed} articles from checkpoint")

                # Hapus checkpoint setelah selesai
                clear_checkpoint()

        else:
            # LLM disabled — gunakan UNKNOWN, bukan B
            logger.info(
                "LLM disabled. Ambiguous articles marked as UNKNOWN "
                "(excluded from A/B analysis)."
            )
            for idx in ambiguous_indices:
                df.at[idx, "type_ab"] = "UNKNOWN"
                df.at[idx, "ab_confidence"] = 0.0
                df.at[idx, "temporal_signal"] = "rule_ambiguous_llm_disabled"
                df.at[idx, "ab_method"] = "fallback"

    # ── FIX #2: Handle UNKNOWN secara eksplisit ──────────────
    # UNKNOWN bukan kelas yang valid untuk analisis sentimen —
    # set ke N/A agar M8 memperlakukannya sebagai non-shock.
    mask_unknown = (mask_shock) & (df["type_ab"] == "UNKNOWN")
    n_unknown = mask_unknown.sum()
    if n_unknown > 0:
        logger.warning(
            f"{n_unknown} articles classified as UNKNOWN — "
            f"set to N/A (excluded from Type A/B analysis). "
            f"Review prompt atau tambah max_retries jika angka ini > 5%."
        )
        df.loc[mask_unknown, "type_ab"] = "N/A"
        df.loc[mask_unknown, "ab_confidence"] = 0.0

    # ── Distribusi Final ──────────────────────────────────────
    shock_final = df[mask_shock]
    type_counts = shock_final["type_ab"].value_counts()
    total_shock = len(shock_final)
    total_classified = type_counts.get("A", 0) + type_counts.get("B", 0)

    logger.info("=" * 50)
    logger.info("TYPE A/B DISTRIBUTION")
    logger.info("=" * 50)
    for t in ["A", "B", "N/A"]:
        count = type_counts.get(t, 0)
        pct_of_total = (count / total_shock * 100) if total_shock > 0 else 0
        logger.info(f"  Tipe {t:>3}: {count:>6,} ({pct_of_total:.1f}% dari total shock)")

    logger.info(f"  Total shock articles : {total_shock:,}")
    logger.info(f"  Total classified A+B : {total_classified:,}")
    logger.info(f"  Rule-based           : {n_rule_a + n_rule_b:,}")
    logger.info(f"  LLM-classified       : {total_classified - (n_rule_a + n_rule_b):,}")
    logger.info(f"  UNKNOWN/excluded     : {n_unknown:,}")

    # Sanity check distribusi A/B
    pct_a = (type_counts.get("A", 0) / total_classified * 100) if total_classified > 0 else 0
    pct_na = (type_counts.get("N/A", 0) / total_shock * 100) if total_shock > 0 else 0

    if pct_a < 20:
        logger.warning(
            f"⚠ Tipe A = {pct_a:.1f}% dari classified (< 20%). "
            "Kemungkinan: (1) over-labeling ke B, (2) keyword BACKWARD terlalu agresif, "
            "atau (3) LLM fallback bias. Review prompt di M3."
        )
    elif pct_a > 50:
        logger.warning(
            f"⚠ Tipe A = {pct_a:.1f}% dari classified (> 50%). "
            "Kemungkinan over-labeling ke A. Review prompt di M3."
        )
    else:
        logger.info(f"✓ Tipe A = {pct_a:.1f}% — dalam rentang wajar (20–50%)")

    if pct_na > 5:
        logger.warning(
            f"⚠ N/A = {pct_na:.1f}% dari shock articles (> 5%). "
            "Pertimbangkan: tambah retry, perbaiki prompt, atau cek koneksi API."
        )

    # Detail per method
    method_counts = shock_final["ab_method"].value_counts()
    logger.info("Classification method breakdown:")
    for method, cnt in method_counts.items():
        logger.info(f"  {method:>12}: {cnt:,}")

    # ── Save ──────────────────────────────────────────────────
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved: {output_path} ({len(df)} rows)")

    return df


if __name__ == "__main__":
    cfg, log = get_cfg_and_logger(skip_env=True)
    df = run_typeAB_classification(cfg, use_llm=True)
    logger.info("M7 Type A/B decomposition complete.")
