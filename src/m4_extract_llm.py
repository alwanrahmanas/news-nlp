# m4_extract_llm.py  ── Parallel version dengan asyncio
import os
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timezone

from openai import AsyncOpenAI
from google import genai
from google.genai import types as genai_types
from tqdm.asyncio import tqdm as atqdm

from m0_setup import ROOT_DIR, get_cfg_and_logger
from m3_prompts import build_prompt, validate_extraction_schema

logger = logging.getLogger("nlp_pipeline.m4")

# ── Cache (sama seperti sebelumnya) ─────────────────────────

def load_cache(cfg: dict) -> dict:
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
    cache_path = ROOT_DIR / "data" / "clean" / "extraction_cache.jsonl"
    with open(cache_path, "w", encoding="utf-8") as f:
        for record in cache.values():
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def load_articles(filepath: Path) -> list:
    articles = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                articles.append(json.loads(line))
    return articles


# ── Single Article: Async version ────────────────────────────

async def extract_single_article_async(
    article: dict,
    client,  # AsyncOpenAI or google.genai.Client
    cfg: dict,
    semaphore: asyncio.Semaphore,
    cache: dict,
    cache_lock: asyncio.Lock,
    flush_counter: list,  # mutable counter [n_since_flush]
) -> dict:
    """Async extraction dengan semaphore rate-limiting dan cache-flush otomatis."""
    async with semaphore:  # Batasi concurrent requests
        prompt = build_prompt(article, "main")

        system_instruction = (
            "Kamu adalah analis senior ketahanan pangan Indonesia "
            "yang menganalisis berita untuk model prediktif inflasi "
            "pangan bergejolak di Medan, Sumatera Utara. "
            "Berikan analisis Chain-of-Thought step-by-step, "
            "lalu simpulkan dalam format JSON sesuai schema. "
            "Kembalikan HANYA JSON valid tanpa teks tambahan."
        )

        for attempt in range(cfg["llm"]["max_retries"]):
            try:
                provider = cfg["llm"].get("provider", "openai")
                prompt_tokens = 0
                completion_tokens = 0
                
                if provider == "gemini":
                    response = await client.aio.models.generate_content(
                        model=cfg["llm"]["model"],
                        contents=prompt,
                        config=genai_types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            max_output_tokens=cfg["llm"]["max_completion_tokens"],
                            response_mime_type="application/json",
                        ),
                    )
                    raw_output = response.text.strip() if response.text else ""
                    
                    if hasattr(response, 'usage_metadata') and response.usage_metadata:
                        prompt_tokens = response.usage_metadata.prompt_token_count or 0
                        completion_tokens = response.usage_metadata.candidates_token_count or 0

                    if not raw_output:
                        logger.warning(f"Empty Gemini response {article['article_id']} (attempt {attempt+1})")
                        if attempt < cfg["llm"]["max_retries"] - 1:
                            await asyncio.sleep(cfg["llm"]["retry_delay_sec"])
                        continue
                else:
                    response = await client.chat.completions.create(
                        model=cfg["llm"]["model"],
                        max_completion_tokens=cfg["llm"]["max_completion_tokens"],
                        response_format={"type": "json_object"},
                        messages=[
                            {
                                "role": "system",
                                "content": system_instruction,
                            },
                            {"role": "user", "content": prompt},
                        ],
                    )
                    raw_output = (response.choices[0].message.content or "").strip()
                    prompt_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') else 0
                    completion_tokens = response.usage.completion_tokens if hasattr(response, 'usage') else 0
    
                    if not raw_output:
                        logger.warning(
                            f"Empty response for {article['article_id']} "
                            f"(attempt {attempt+1}, finish={response.choices[0].finish_reason!r})"
                        )
                        if attempt < cfg["llm"]["max_retries"] - 1:
                            await asyncio.sleep(cfg["llm"]["retry_delay_sec"])
                        continue

                json_str = raw_output
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0].strip()

                parsed = json.loads(json_str)
                if isinstance(parsed, list) and len(parsed) > 0:
                    parsed = parsed[0]
                elif isinstance(parsed, list):
                    parsed = {}

                # Schema validation + auto-fill (sama seperti sebelumnya)
                try:
                    validate_extraction_schema(parsed)
                except Exception as e:
                    logger.warning(f"Schema warning {article['article_id']}: {e}")
                    parsed.setdefault("reasoning_chain", {
                        "step1_commodities": "auto-filled",
                        "step2_geography": "auto-filled",
                        "step3_event_type": "auto-filled",
                        "step4_temporal": "auto-filled",
                        "step5_sentiment": "auto-filled",
                    })
                    for key, default in [
                        ("label", "IRRELEVANT"), ("rationale", "auto-filled"),
                        ("supply_location", None), ("commodities", []),
                        ("is_forward_looking", False),
                    ]:
                        parsed.setdefault(key, default)
                    parsed.setdefault("sentiment_score", 0.0)
                    parsed.setdefault("confidence", 0.0)

                parsed["sentiment_score"] = max(-1.0, min(1.0, float(parsed["sentiment_score"])))
                parsed["confidence"] = max(0.0, min(1.0, float(parsed["confidence"])))

                result = {
                    "article_id": article["article_id"],
                    "extraction": parsed,
                    "model": cfg["llm"]["model"],
                    "tokens_prompt": prompt_tokens,
                    "tokens_completion": completion_tokens,
                    "extracted_at": datetime.now(timezone.utc).isoformat(),
                    "status": "success",
                }

                # ── Thread-safe cache update + periodic flush ──
                async with cache_lock:
                    cache[article["article_id"]] = result
                    flush_counter[0] += 1
                    if flush_counter[0] >= cfg["llm"]["batch_size"]:
                        flush_cache(cache, cfg)
                        flush_counter[0] = 0

                return result

            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error {article['article_id']} attempt {attempt+1}: {e}")
                if attempt < cfg["llm"]["max_retries"] - 1:
                    await asyncio.sleep(cfg["llm"]["retry_delay_sec"])

            except Exception as e:
                logger.error(f"Error {article['article_id']} attempt {attempt+1}: {e}")
                await asyncio.sleep(cfg["llm"]["retry_delay_sec"] * (attempt + 1))

    # Fallback setelah max retries
    result = {
        "article_id": article["article_id"],
        "extraction": {
            "reasoning_chain": {k: "extraction_failed" for k in
                ["step1_commodities","step2_geography","step3_event_type",
                 "step4_temporal","step5_sentiment"]},
            "label": "IRRELEVANT", "sentiment_score": 0.0,
            "confidence": 0.0, "commodities": [],
            "supply_location": None, "is_forward_looking": False,
            "rationale": "extraction_failed",
        },
        "model": cfg["llm"]["model"],
        "tokens_prompt": 0, "tokens_completion": 0,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
    }
    async with cache_lock:
        cache[article["article_id"]] = result
    return result


# ── Main Async Pipeline ───────────────────────────────────────

async def run_extraction_pipeline_async(
    articles: list = None,
    cfg: dict = None,
    auto_confirm: bool = False,
) -> None:
    if cfg is None:
        cfg, _ = get_cfg_and_logger()

    if articles is None:
        input_path = ROOT_DIR / "data" / "clean" / "corpus_clean.jsonl"
        if not input_path.exists():
            logger.error(f"Input not found: {input_path}")
            return
        articles = load_articles(input_path)

    cache = load_cache(cfg)
    to_process = [a for a in articles if a["article_id"] not in cache]
    logger.info(f"Total: {len(articles)} | Cached: {len(cache)} | To process: {len(to_process)}")

    if not to_process:
        logger.info("All articles already cached.")
        return

    # Cost estimation
    estimated_tokens = len(to_process) * 1200
    provider = cfg["llm"].get("provider", "openai")
    
    if provider == "gemini":
        cost_input = (estimated_tokens / 1_000_000) * 0.075
        cost_output = (len(to_process) * 300 / 1_000_000) * 0.30
        total_cost = cost_input + cost_output
    else:
        cost_input  = (estimated_tokens / 1_000_000) * 0.150
        cost_output = (len(to_process) * 300 / 1_000_000) * 0.600
        total_cost = cost_input + cost_output
        
    logger.info(f"Estimated cost ({cfg['llm']['model']}): ${total_cost:.4f}")

    if not auto_confirm:
        confirm = input(f"Estimated cost: ${total_cost:.4f}. Lanjutkan? (y/n): ")
        if confirm.lower() != "y":
            logger.info("Cancelled.")
            return

    # ── Kunci: Semaphore kontrol concurrency ─────────────────
    # max_concurrent = 20 adalah titik aman untuk tier standar OpenAI/Gemini
    max_concurrent = cfg["llm"].get("max_concurrent", 20)
    semaphore = asyncio.Semaphore(max_concurrent)
    cache_lock = asyncio.Lock()
    flush_counter = [0]

    client = None
    if provider == "gemini":
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    else:
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    tasks = [
        extract_single_article_async(a, client, cfg, semaphore, cache, cache_lock, flush_counter)
        for a in to_process
    ]

    results = await atqdm.gather(*tasks, desc=f"Extracting (concurrency={max_concurrent})")

    # Final flush
    async with cache_lock:
        flush_cache(cache, cfg)

    # Stats
    n_success = sum(1 for r in results if r["status"] == "success")
    n_failed  = sum(1 for r in results if r["status"] == "failed")
    total_prompt     = sum(r.get("tokens_prompt", 0) for r in results)
    total_completion = sum(r.get("tokens_completion", 0) for r in results)

    rate_in = 0.075 if provider == "gemini" else 0.15
    rate_out = 0.30 if provider == "gemini" else 0.60
    
    cost_log = {
        "run_date": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": cfg["llm"]["model"],
        "articles_processed": len(to_process),
        "success": n_success, "failed": n_failed,
        "total_tokens_prompt": total_prompt,
        "total_tokens_completion": total_completion,
        "estimated_cost_usd": round(
            (total_prompt / 1_000_000 * rate_in) + (total_completion / 1_000_000 * rate_out), 4
        ),
        "max_concurrent": max_concurrent,
    }

    cost_path = ROOT_DIR / "outputs" / "extraction_cost_log.json"
    cost_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cost_path, "w", encoding="utf-8") as f:
        json.dump(cost_log, f, indent=2)

    logger.info(f"DONE | Success: {n_success} | Failed: {n_failed} | Cost log: {cost_path}")


# ── Entry point ───────────────────────────────────────────────

def run_extraction_pipeline(articles=None, cfg=None, auto_confirm=False):
    """Wrapper sync untuk kompatibilitas backward dengan kode lain."""
    asyncio.run(run_extraction_pipeline_async(articles, cfg, auto_confirm))


if __name__ == "__main__":
    cfg, log = get_cfg_and_logger()
    run_extraction_pipeline(cfg=cfg, auto_confirm=True)
