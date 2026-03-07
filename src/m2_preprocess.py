"""
M2 — PREPROCESSING & DEDUPLICATION
====================================
Input:  data/raw/corpus_raw.jsonl
Output: data/clean/corpus_clean.jsonl

Langkah:
  1. Filter panjang teks
  2. Deduplikasi (URL exact + MinHash similarity)
  3. Filter relevansi keyword (diperluas untuk coverage lebih baik)
  4. Normalisasi teks
  5. Truncate ke max_length
"""

import json
import re
import hashlib
import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from m0_setup import ROOT_DIR, get_cfg_and_logger

logger = logging.getLogger("nlp_pipeline.m2")


# ── Helpers ──────────────────────────────────────────────────

def load_jsonl(filepath: Path) -> list:
    """Load all records from a JSONL file."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def save_jsonl(records: list, filepath: Path) -> None:
    """Save records to a JSONL file (overwrite)."""
    with open(filepath, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ── Step 1: Filter panjang teks & kualitas (Gibberish) ───────────

def is_gibberish(text: str, min_alpha_ratio: float = 0.65) -> bool:
    """
    Deteksi teks hasil scraping yang rusak (enkripsi/obfuskasi JS/scrambled).
    Teks bahasa alami yang sehat didominasi huruf (a-z) dan spasi.
    Jika rasio huruf alfabetiknya terlalu rendah, maka dianggap gibberish.
    """
    if not text:
        return True
    
    alpha_count = sum(1 for c in text if c.isalpha())
    ratio = alpha_count / len(text)
    return ratio < min_alpha_ratio

def filter_by_length_and_quality(articles: list, min_len: int, max_len: int) -> list:
    """Filter articles by character length AND remove gibberish text."""
    filtered = []
    for a in articles:
        text_len = a.get("char_length", 0)
        if not (min_len <= text_len <= max_len):
            continue
            
        full_text = a.get("full_text", "")
        # Jika teks terlalu banyak simbol aneh (gibberish), drop!
        if is_gibberish(full_text):
            continue
            
        filtered.append(a)
        
    return filtered


# ── Step 2: Deduplikasi ──────────────────────────────────────

def get_shingles(text: str, k: int = 5) -> set:
    """Generate k-shingles (character-level) from text."""
    text = text.lower().strip()
    if len(text) < k:
        return {text}
    return {text[i:i+k] for i in range(len(text) - k + 1)}


def minhash_signature(shingles: set, num_perm: int = 128) -> list:
    """Generate MinHash signature for a set of shingles."""
    import random
    random.seed(42)

    # Generate random hash functions
    max_hash = 2**32 - 1
    hash_funcs = [
        (random.randint(1, max_hash), random.randint(0, max_hash))
        for _ in range(num_perm)
    ]

    signature = []
    for a, b in hash_funcs:
        min_val = max_hash + 1
        for s in shingles:
            h = (a * hash(s) + b) % max_hash
            if h < min_val:
                min_val = h
        signature.append(min_val)

    return signature


def jaccard_from_minhash(sig1: list, sig2: list) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if len(sig1) != len(sig2):
        return 0.0
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


def deduplicate_articles(articles: list, similarity_threshold: float = 0.85) -> list:
    """
    Deduplicate articles:
    1. First by exact URL match
    2. Then by MinHash similarity (keep longest version)
    """
    # Phase 1: URL dedup
    seen_urls = {}
    for a in articles:
        url = a["url"]
        if url not in seen_urls:
            seen_urls[url] = a
        else:
            # Keep longer version
            if a.get("char_length", 0) > seen_urls[url].get("char_length", 0):
                seen_urls[url] = a

    url_deduped = list(seen_urls.values())
    url_dropped = len(articles) - len(url_deduped)
    logger.info(f"URL dedup: {len(articles)} -> {len(url_deduped)} (dropped {url_dropped})")

    # Phase 2: MinHash similarity dedup
    if len(url_deduped) < 2:
        return url_deduped

    logger.info("Computing MinHash signatures for similarity dedup...")
    signatures = []
    for a in tqdm(url_deduped, desc="MinHash"):
        shingles = get_shingles(a.get("full_text", ""), k=5)
        sig = minhash_signature(shingles, num_perm=64)
        signatures.append(sig)

    # Find duplicates
    to_remove = set()
    for i in range(len(url_deduped)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(url_deduped)):
            if j in to_remove:
                continue
            sim = jaccard_from_minhash(signatures[i], signatures[j])
            if sim >= similarity_threshold:
                # Keep the longer one
                if url_deduped[i].get("char_length", 0) >= url_deduped[j].get("char_length", 0):
                    to_remove.add(j)
                else:
                    to_remove.add(i)
                    break  # i is removed, no need to check more

    minhash_deduped = [a for idx, a in enumerate(url_deduped) if idx not in to_remove]
    minhash_dropped = len(url_deduped) - len(minhash_deduped)
    logger.info(f"MinHash dedup: {len(url_deduped)} -> {len(minhash_deduped)} (dropped {minhash_dropped})")

    return minhash_deduped


# ── Step 3: Filter relevansi keyword ─────────────────────────

def filter_by_keyword_relevance(articles: list, cfg: dict) -> list:
    """
    Keep articles that contain at least 1 primary keyword
    AND at least one of the geo terms.

    If semantic_search is enabled in config, it also checks if the
    article title/content has a high semantic similarity with primary keywords.
    """
    primary_keywords = cfg["keywords"]["primary"]

    # Get geo_terms from config, or use expanded defaults
    geo_terms = cfg.get("keywords", {}).get("geo_terms", [
        "medan", "karo", "berastagi", "pasar induk",
        "sumatera utara", "sumatra utara", "sumut",
        "sumatera", "sumatra",
        "deli serdang", "langkat", "simalungun",
        "tanah karo", "binjai", "tebing tinggi",
        "pematangsiantar", "pematang siantar", "siantar",
        "toba", "tapanuli", "asahan", "labuhanbatu",
        "sibolga", "padangsidimpuan",
        "pasar sambu", "pasar aksara", "pasar helvetia",
    ])

    # Keyword yang secara inheren memilik geo-term
    geo_embedded_keywords = []
    for kw in primary_keywords:
        kw_lower = kw.lower()
        if any(gt in kw_lower for gt in geo_terms):
            geo_embedded_keywords.append(kw)

    semantic_cfg = cfg.get("semantic_search", {})
    use_semantic = semantic_cfg.get("enabled", False)
    sim_model = None
    kw_embeddings = None
    sim_threshold = semantic_cfg.get("similarity_threshold", 0.45)

    if use_semantic:
        try:
            from sentence_transformers import SentenceTransformer, util
            import torch
            logger.info(f"Loading Semantic Search model: {semantic_cfg.get('model_name')}")
            sim_model = SentenceTransformer(semantic_cfg.get('model_name', 'paraphrase-multilingual-MiniLM-L12-v2'))
            # Pre-compute embeddings for primary keywords
            kw_embeddings = sim_model.encode(primary_keywords, convert_to_tensor=True)
        except ImportError:
            logger.warning("sentence-transformers tidak terinstall. Fallback ke pure text matching.")
            use_semantic = False

    filtered = []
    logger.info("Mulai keyword & semantic relevance parsing...")
    for a in tqdm(articles, desc="Relevance Filter"):
        text = (a.get("full_text", "") + " " + a.get("title", "")).lower()

        # FIXED LOGIC: Instead of exact phrase match (which drops everything),
        # we check if ALL words in the query exist in the text.
        has_primary = False
        for kw in primary_keywords:
            words = kw.lower().split()
            if all(word in text for word in words):
                has_primary = True
                break
                
        # SEMANTIC SEARCH FALLBACK (kalau keyword exact match fail) 
        if not has_primary and use_semantic:
            # Kita uji context pendek: Titile + 500 chars awal untuk efisiensi
            context = (a.get("title", "") + ". " + a.get("full_text", "")[:500]).strip()
            if context:
                from sentence_transformers import util
                ctx_emb = sim_model.encode(context, convert_to_tensor=True)
                cos_scores = util.cos_sim(ctx_emb, kw_embeddings)[0]
                max_score = cos_scores.max().item()
                if max_score >= sim_threshold:
                    has_primary = True

        if not has_primary:
            continue

        # Check for geo term presence (either in text or inherently via matched keyword)
        matched_geo_kw = False
        for kw in geo_embedded_keywords:
            words = kw.lower().split()
            if all(w in text for w in words):
                matched_geo_kw = True
                break
                
        has_geo = matched_geo_kw or any(gt in text for gt in geo_terms)

        if has_geo:
            filtered.append(a)

    return filtered


# ── Step 4: Normalisasi teks ─────────────────────────────────

def normalize_text(text: str) -> str:
    """
    Normalize text:
    1. Remove leftover HTML tags
    2. Normalize whitespace
    3. Remove non-printable characters
    4. DO NOT do stemming/stopword removal (LLM needs original text)
    """
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove non-printable characters (keep \n and basic punctuation)
    text = re.sub(r"[^\x20-\x7E\u00C0-\u024F\u0100-\u017F\u0180-\u024F"
                  r"\u0250-\u02AF\u1E00-\u1EFF\u0300-\u036F"
                  r"\u00A0-\u00FF\n"
                  r"\u0020-\u007E"
                  r"\u00A1-\u00FF"
                  r"\u0100-\u024F]", " ", text)

    # Normalize whitespace (collapse multiple spaces, strip)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text


# ── Step 5: Truncate ke max_length ───────────────────────────

def truncate_at_sentence(text: str, max_chars: int) -> str:
    """
    Truncate text at the nearest sentence boundary before max_chars.
    """
    if len(text) <= max_chars:
        return text

    # Find sentence boundaries
    candidates = []
    for match in re.finditer(r'[.!?]\s+', text[:max_chars]):
        candidates.append(match.end())

    if candidates:
        cut_point = candidates[-1]
        return text[:cut_point].strip()
    else:
        # No sentence boundary found; cut at word boundary
        cut_point = text.rfind(" ", 0, max_chars)
        if cut_point > 0:
            return text[:cut_point].strip()
        return text[:max_chars].strip()


# ── Main Pipeline ────────────────────────────────────────────

def run_preprocessing(cfg: dict = None) -> None:
    """Run the full preprocessing pipeline."""
    if cfg is None:
        cfg, _ = get_cfg_and_logger()

    input_path = ROOT_DIR / "data" / "raw" / "corpus_raw.jsonl"
    output_path = ROOT_DIR / "data" / "clean" / "corpus_clean.jsonl"

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.error("Run m1_collect.py first to generate corpus_raw.jsonl")
        return

    # Load raw corpus
    articles = load_jsonl(input_path)
    total_raw = len(articles)
    logger.info(f"Total raw: {total_raw} artikel")

    # Step 1: Filter by length & quality
    min_len = cfg["corpus"]["min_length_chars"]
    max_len = cfg["corpus"]["max_length_chars"]
    articles = filter_by_length_and_quality(articles, min_len, max_len)
    after_length = len(articles)
    logger.info(f"After length/quality filter: {after_length} (dropped {total_raw - after_length})")

    # Step 2: Deduplication
    articles = deduplicate_articles(articles, similarity_threshold=0.85)
    after_dedup = len(articles)

    # Step 3: Keyword relevance filter (now uses cfg for geo_terms)
    articles = filter_by_keyword_relevance(articles, cfg)
    after_keyword = len(articles)
    logger.info(f"After keyword filter: {after_keyword} (dropped {after_dedup - after_keyword})")

    # Step 4: Normalize text
    for a in articles:
        a["full_text"] = normalize_text(a.get("full_text", ""))
        a["title"] = normalize_text(a.get("title", ""))
        a["char_length"] = len(a["full_text"])

    # Step 5: Truncate to max_length
    for a in articles:
        a["full_text"] = truncate_at_sentence(a["full_text"], max_len)
        a["char_length"] = len(a["full_text"])

    final_count = len(articles)

    # Save
    save_jsonl(articles, output_path)

    # Log summary
    logger.info("=" * 50)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total raw    : {total_raw} artikel")
    logger.info(f"After length : {after_length} (dropped {total_raw - after_length})")
    logger.info(f"After dedup  : {after_dedup} (dropped {after_length - after_dedup})")
    logger.info(f"After keyword: {after_keyword} (dropped {after_dedup - after_keyword})")
    logger.info(f"Final clean  : {final_count} artikel")

    # Date coverage
    if articles:
        dates = [a["published_date"] for a in articles if a.get("published_date")]
        if dates:
            logger.info(f"Coverage     : {min(dates)} - {max(dates)}")

            # Check for low-coverage months
            from collections import Counter
            month_counts = Counter(d[:7] for d in dates)
            low_months = [m for m, c in month_counts.items() if c < 5]
            if low_months:
                logger.warning(f"Low-coverage months (<5 articles): {sorted(low_months)}")

    logger.info(f"Output saved: {output_path}")


if __name__ == "__main__":
    cfg, log = get_cfg_and_logger(skip_env=True)
    run_preprocessing(cfg)
