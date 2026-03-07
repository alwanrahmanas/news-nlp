"""
M4 — EKSTRAKSI LLM (GPT-5-mini)
==================================
Input:  data/clean/corpus_clean.jsonl
Output: data/clean/extraction_cache.jsonl

Arsitektur: Cache-first untuk efisiensi biaya
- Setiap artikel dicek dulu di cache sebelum memanggil API
- Estimasi biaya sebelum jalankan
- Batch processing dengan progress tracking
"""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timezone

from openai import OpenAI
from tqdm import tqdm

from m0_setup import ROOT_DIR, get_cfg_and_logger
from m3_prompts import build_prompt, validate_extraction_schema, EXTRACTION_SCHEMA

logger = logging.getLogger("nlp_pipeline.m4")


# ── Cache Management ─────────────────────────────────────────

def load_cache(cfg: dict) -> dict:
    """Load extraction cache from JSONL file. Returns dict keyed by article_id."""
    cache_path = ROOT_DIR / "data" / "clean" / "extraction_cache.jsonl"
    cache = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        cache[record["article_id"]] = record
                    except (json.JSONDecodeError, KeyError):
                        continue
    return cache


def flush_cache(cache: dict, cfg: dict) -> None:
    """Save entire cache to JSONL file (overwrite)."""
    cache_path = ROOT_DIR / "data" / "clean" / "extraction_cache.jsonl"
    with open(cache_path, "w", encoding="utf-8") as f:
        for record in cache.values():
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_articles(filepath: Path) -> list:
    """Load articles from JSONL file."""
    articles = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                articles.append(json.loads(line))
    return articles


# ── Single Article Extraction ────────────────────────────────

def extract_single_article(article: dict, client: OpenAI, cfg: dict) -> dict:
    """
    Call GPT-5-mini for a single article.
    Returns: dict with extraction result + metadata (model, timestamp, tokens_used)
    """
    prompt = build_prompt(article, "main")

    for attempt in range(cfg["llm"]["max_retries"]):
        try:
            response = client.chat.completions.create(
                model=cfg["llm"]["model"],
                max_completion_tokens=cfg["llm"]["max_completion_tokens"],
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Kamu adalah analis senior ketahanan pangan Indonesia "
                            "yang menganalisis berita untuk model prediktif inflasi "
                            "pangan bergejolak di Medan, Sumatera Utara. "
                            "Berikan analisis Chain-of-Thought step-by-step, "
                            "lalu simpulkan dalam format JSON sesuai schema. "
                            "Kembalikan HANYA JSON valid tanpa teks tambahan."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            )
            raw_output = response.choices[0].message.content or ""
            raw_output = raw_output.strip()

            # Guard: empty response
            if not raw_output:
                finish_reason = response.choices[0].finish_reason
                logger.warning(
                    f"Empty response for {article['article_id']} "
                    f"(attempt {attempt + 1}/{cfg['llm']['max_retries']}, "
                    f"finish_reason={finish_reason!r})"
                )
                if attempt < cfg["llm"]["max_retries"] - 1:
                    time.sleep(cfg["llm"]["retry_delay_sec"])
                continue

            # Try to extract JSON from response (handle markdown code blocks)
            json_str = raw_output
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()

            parsed = json.loads(json_str)


            # Validate schema
            try:
                validate_extraction_schema(parsed)
            except Exception as e:
                logger.warning(f"Schema validation warning for {article['article_id']}: {e}")
                # Try to fix common issues
                if "reasoning_chain" not in parsed:
                    parsed["reasoning_chain"] = {
                        "step1_commodities": "auto-filled",
                        "step2_geography": "auto-filled",
                        "step3_event_type": "auto-filled",
                        "step4_temporal": "auto-filled",
                        "step5_sentiment": "auto-filled",
                    }
                if "label" not in parsed:
                    parsed["label"] = "IRRELEVANT"
                if "sentiment_score" not in parsed:
                    parsed["sentiment_score"] = 0.0
                if "confidence" not in parsed:
                    parsed["confidence"] = 0.0
                if "commodities" not in parsed:
                    parsed["commodities"] = []
                if "supply_location" not in parsed:
                    parsed["supply_location"] = None
                if "is_forward_looking" not in parsed:
                    parsed["is_forward_looking"] = False
                if "rationale" not in parsed:
                    parsed["rationale"] = "auto-filled"

            # Clamp values to valid range
            parsed["sentiment_score"] = max(-1.0, min(1.0, float(parsed["sentiment_score"])))
            parsed["confidence"] = max(0.0, min(1.0, float(parsed["confidence"])))

            return {
                "article_id": article["article_id"],
                "extraction": parsed,
                "model": cfg["llm"]["model"],
                "tokens_prompt": response.usage.prompt_tokens,
                "tokens_completion": response.usage.completion_tokens,
                "extracted_at": datetime.now(timezone.utc).isoformat(),
                "status": "success",
            }

        except json.JSONDecodeError as e:
            logger.warning(
                f"JSON decode error for {article['article_id']} "
                f"(attempt {attempt + 1}/{cfg['llm']['max_retries']}): {e}"
            )
            if attempt < cfg["llm"]["max_retries"] - 1:
                time.sleep(cfg["llm"]["retry_delay_sec"])

        except Exception as e:
            logger.error(
                f"Extraction error for {article['article_id']} "
                f"(attempt {attempt + 1}/{cfg['llm']['max_retries']}): {e}"
            )
            time.sleep(cfg["llm"]["retry_delay_sec"] * (attempt + 1))

    # Fallback after max retries
    logger.warning(f"All retries failed for {article['article_id']}")
    return {
        "article_id": article["article_id"],
        "extraction": {
            "reasoning_chain": {
                "step1_commodities": "extraction_failed",
                "step2_geography": "extraction_failed",
                "step3_event_type": "extraction_failed",
                "step4_temporal": "extraction_failed",
                "step5_sentiment": "extraction_failed",
            },
            "label": "IRRELEVANT",
            "sentiment_score": 0.0,
            "confidence": 0.0,
            "commodities": [],
            "supply_location": None,
            "is_forward_looking": False,
            "rationale": "extraction_failed",
        },
        "model": cfg["llm"]["model"],
        "tokens_prompt": 0,
        "tokens_completion": 0,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
    }


# ── Batch Processing ─────────────────────────────────────────

def run_extraction_pipeline(articles: list = None, cfg: dict = None,
                             auto_confirm: bool = False) -> None:
    """
    Loop all articles, skip cached ones, save every batch to extraction_cache.jsonl.
    """
    if cfg is None:
        cfg, _ = get_cfg_and_logger()

    if articles is None:
        input_path = ROOT_DIR / "data" / "clean" / "corpus_clean.jsonl"
        if not input_path.exists():
            logger.error(f"Input not found: {input_path}. Run m2_preprocess.py first.")
            return
        articles = load_articles(input_path)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    cache = load_cache(cfg)

    to_process = [a for a in articles if a["article_id"] not in cache]
    logger.info(f"Total: {len(articles)} | Cached: {len(cache)} | To process: {len(to_process)}")

    if not to_process:
        logger.info("All articles already cached. Nothing to do.")
        return

    # Estimate cost before running (GPT-5-mini pricing)
    estimated_tokens = len(to_process) * 1200  # ~1200 tokens per article (CoT prompt is larger)
    estimated_cost_input = (estimated_tokens / 1_000_000) * 0.25   # $0.25/1M input tokens
    estimated_cost_output = (len(to_process) * 300 / 1_000_000) * 2.00  # ~300 output tokens (incl. CoT), $2.00/1M
    total_cost = estimated_cost_input + estimated_cost_output
    logger.info(f"Estimated input tokens: {estimated_tokens:,}")
    logger.info(f"Estimated cost (GPT-5-mini): ${total_cost:.4f}")

    if not auto_confirm:
        confirm = input(f"Estimated cost (GPT-5-mini): ${total_cost:.4f}. Lanjutkan? (y/n): ")
        if confirm.lower() != "y":
            logger.info("Extraction cancelled by user.")
            return

    # Track stats
    total_tokens_prompt = 0
    total_tokens_completion = 0
    n_success = 0
    n_failed = 0

    batch = []
    for i, article in enumerate(tqdm(to_process, desc="Extracting")):
        result = extract_single_article(article, client, cfg)
        batch.append(result)
        cache[article["article_id"]] = result

        if result["status"] == "success":
            n_success += 1
            total_tokens_prompt += result.get("tokens_prompt", 0)
            total_tokens_completion += result.get("tokens_completion", 0)
        else:
            n_failed += 1

        # Flush cache every batch_size articles
        if len(batch) >= cfg["llm"]["batch_size"]:
            flush_cache(cache, cfg)
            batch = []
            time.sleep(1)  # Rate limit buffer

        # Log progress every 100 articles
        if (i + 1) % 100 == 0:
            logger.info(
                f"Progress: {i + 1}/{len(to_process)} | "
                f"Success: {n_success} | Failed: {n_failed} | "
                f"Tokens: {total_tokens_prompt + total_tokens_completion:,}"
            )

    # Final flush
    flush_cache(cache, cfg)

    # Save cost log
    cost_log = {
        "run_date": datetime.now(timezone.utc).isoformat(),
        "model": cfg["llm"]["model"],
        "articles_processed": len(to_process),
        "success": n_success,
        "failed": n_failed,
        "total_tokens_prompt": total_tokens_prompt,
        "total_tokens_completion": total_tokens_completion,
        "estimated_cost_usd": round(
            (total_tokens_prompt / 1_000_000 * 0.25)
            + (total_tokens_completion / 1_000_000 * 2.00),
            4,
        ),
    }

    cost_path = ROOT_DIR / "outputs" / "extraction_cost_log.json"
    with open(cost_path, "w", encoding="utf-8") as f:
        json.dump(cost_log, f, indent=2)

    logger.info("=" * 50)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Success: {n_success} | Failed: {n_failed}")
    logger.info(f"Total tokens: {total_tokens_prompt + total_tokens_completion:,}")
    logger.info(f"Cost log saved: {cost_path}")


if __name__ == "__main__":
    cfg, log = get_cfg_and_logger()
    run_extraction_pipeline(cfg=cfg)
