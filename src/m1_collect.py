"""
M1 — KOLEKSI BERITA v3 (Direct Archive Scraping)
====================================================
Output: data/raw/corpus_raw.jsonl

Perbaikan dari v2:
  - Bug fix: date fallback tidak lagi menggunakan datetime.now()
  - Bug fix: httpx client digunakan secara konsisten (trafilatura sebagai extraction engine,
             client hanya untuk URL discovery)
  - Tambah: Waspada (waspada.id) sebagai sumber ke-7 (sesuai proposal)
  - Tambah: geo-filter wajib (Medan/Sumut/Karo) di fase ekstraksi
  - Tambah: date range filter di Google News RSS
  - Tambah: artikel ditolak disimpan ke corpus_rejected.jsonl untuk audit
  - Konfigurasi komoditas disesuaikan dengan PIHPS Pedagang Besar & proposal penelitian
  - Coverage threshold dipindah ke config.yaml
"""

import os
import json
import hashlib
import time
import re
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import quote_plus, urlparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from bs4 import BeautifulSoup
from tqdm import tqdm

try:
    import trafilatura
except ImportError:
    trafilatura = None

try:
    from googlenewsdecoder import gnewsdecoder
    _HAS_GNEWSDECODER = True
except ImportError:
    _HAS_GNEWSDECODER = False

from m0_setup import ROOT_DIR, get_cfg_and_logger

logger = logging.getLogger("nlp_pipeline.m1")


# ─────────────────────────────────────────────────────────────
# KONFIGURASI KOMODITAS
# Disesuaikan dengan:
#   (1) Komoditas Volatile Food di proposal penelitian (file proposal)
#   (2) Ketersediaan data di PIHPS Pedagang Besar Kota Medan
#   (3) Relevansi rantai pasok Kabupaten Karo → Medan
# ─────────────────────────────────────────────────────────────

# Komoditas inti VF yang tersedia di PIHPS Pedagang Besar Medan
# + relevan dengan rantai pasok Karo
COMMODITIES_VF_CORE = [
    # Hortikultura — rantai pasok langsung dari Karo (PIHPS: tersedia)
    "Cabai Merah",
    "Cabai Rawit",
    "Bawang Merah",
    "Bawang Putih",
    # Hortikultura — rantai pasok dari Karo (TIDAK ada di PIHPS PB,
    # tapi sangat relevan untuk sinyal CHIRPS; dimasukkan sebagai
    # primary scraping keyword agar berita guncangan pasokan tertangkap)
    "Tomat",
    "Kentang",
    # Protein hewani (PIHPS: tersedia)
    "Daging Ayam",
    "Telur Ayam",
    "Daging Sapi",
    # Beras — masuk VF Indonesia meski kadang ada intervensi Bulog
    # (PIHPS: tersedia sebagai Beras Kualitas Medium I)
    "Beras",
]




# Geo-terms wajib ada minimal 1 di teks/judul artikel agar lolos filter
# Diambil secara dinamis dari config untuk mempermudah parameterisasi.


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def make_article_id(url: str, date: str) -> str:
    raw = f"{url}|{date}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def save_jsonl(records: list, filepath: Path, mode: str = "a") -> None:
    with open(filepath, mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_jsonl(filepath: Path) -> list:
    if not filepath.exists():
        return []
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


def keyword_matches(text: str, keywords: list) -> list:
    text_lower = text.lower()
    return [kw for kw in keywords if kw.lower() in text_lower]


def is_geo_relevant(cfg: dict, text: str, title: str, min_hits: int = 1) -> bool:
    """
    Cek apakah artikel relevan secara geografis (Medan/Sumut/Karo).
    Artikel yang tidak lulus dikirim ke corpus_rejected.jsonl, bukan dibuang.
    """
    geo_terms = cfg.get("keywords", {}).get("geo_terms", [])
    combined = (text + " " + title).lower()
    hits = sum(1 for term in geo_terms if term in combined)
    return hits >= min_hits


def get_http_client(cfg: dict) -> httpx.Client:
    scraper_cfg = cfg.get("scrapers", {})
    headers = {
        "User-Agent": scraper_cfg.get(
            "user_agent",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "id-ID,id;q=0.9,en;q=0.5",
    }
    return httpx.Client(
        timeout=scraper_cfg.get("timeout_sec", 30),
        follow_redirects=True,
        headers=headers,
    )


class DomainRateLimiter:
    """
    Per-domain rate limiter — memastikan jeda minimum antar request
    ke domain yang SAMA, tapi request ke domain BERBEDA bisa paralel.
    Thread-safe.

    FIX v4: Lock ditahan selama seluruh siklus baca-sleep-tulis agar
    tidak ada race condition antar thread yang mengakses domain yang SAMA.
    Ini men-serialize request ke domain yang sama (yang memang diinginkan),
    sementara request ke domain berbeda tetap berjalan paralel.
    """
    def __init__(self, min_interval: float = 1.5):
        self._meta_lock = threading.Lock()          # hanya untuk buat per-domain lock
        self._domain_locks: dict[str, threading.Lock] = {}
        self._last_request: dict[str, float] = {}
        self._min_interval = min_interval

    def _get_domain_lock(self, domain: str) -> threading.Lock:
        """Lazily buat lock per domain (thread-safe)."""
        with self._meta_lock:
            if domain not in self._domain_locks:
                self._domain_locks[domain] = threading.Lock()
            return self._domain_locks[domain]

    def wait(self, domain: str) -> None:
        """
        Tunggu sampai aman untuk request ke domain ini.
        Lock ditahan per-domain — domain lain bebas jalan paralel.
        """
        domain_lock = self._get_domain_lock(domain)
        with domain_lock:
            # Baca, sleep, lalu update — semua di dalam lock
            now = time.monotonic()
            last = self._last_request.get(domain, 0.0)
            wait_time = self._min_interval - (now - last)

            if wait_time > 0:
                time.sleep(wait_time)

            self._last_request[domain] = time.monotonic()


class DomainConcurrencyLimiter:
    """
    Per-domain concurrency limiter — membatasi max N thread yang
    mengakses domain yang SAMA secara bersamaan. Thread-safe.

    Mencegah WinError 10060 (connection timeout) akibat terlalu banyak
    koneksi simultan ke server yang sama.
    """
    def __init__(self, max_concurrent: int = 2):
        self._lock = threading.Lock()
        self._semaphores: dict[str, threading.Semaphore] = {}
        self._max_concurrent = max_concurrent

    def acquire(self, domain: str) -> None:
        with self._lock:
            if domain not in self._semaphores:
                self._semaphores[domain] = threading.Semaphore(self._max_concurrent)
        self._semaphores[domain].acquire()

    def release(self, domain: str) -> None:
        if domain in self._semaphores:
            self._semaphores[domain].release()


# ─────────────────────────────────────────────────────────────
# Checkpoint System
# ─────────────────────────────────────────────────────────────

class CheckpointManager:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = checkpoint_dir / "scraping_checkpoint.json"
        # HARUS RLock (reentrant) karena add_urls() acquire lock lalu
        # memanggil save() yang juga acquire lock — Lock biasa = DEADLOCK!
        self._lock = threading.RLock()

    def load(self) -> dict:
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # Migrate list → set internally (backward compat)
            raw["completed_queries"] = set(raw.get("completed_queries", []))
            raw["collected_urls"]    = set(raw.get("collected_urls", []))
            if "url_metadata" not in raw:
                raw["url_metadata"] = {}
            return raw
        return {"completed_queries": set(), "collected_urls": set(), "url_metadata": {}}

    def save(self, state: dict) -> None:
        with self._lock:
            serializable = {
                "completed_queries": list(state["completed_queries"]),
                "collected_urls":    list(state["collected_urls"]),
                "url_metadata":      state.get("url_metadata", {}),
            }
            with open(self.checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)

    def mark_query_done(self, state: dict, source: str, keyword: str) -> None:
        key = f"{source}::{keyword}"
        state["completed_queries"].add(key)
        self.save(state)

    def is_query_done(self, state: dict, source: str, keyword: str) -> bool:
        return f"{source}::{keyword}" in state["completed_queries"]

    def add_urls(self, state: dict, entries: list) -> None:
        with self._lock:
            if "url_metadata" not in state:
                state["url_metadata"] = {}
            for entry in entries:
                url = entry["url"]
                if url not in state["collected_urls"]:
                    state["collected_urls"].add(url)
                    state["url_metadata"][url] = {
                        "title": entry.get("title", ""),
                        "source_keyword": entry.get("source_keyword", ""),
                        "source_name": entry.get("source_name", "unknown"),
                    }
            self.save(state)


# ─────────────────────────────────────────────────────────────
# URL Discovery: Per-Source Scrapers
# ─────────────────────────────────────────────────────────────

def fetch_with_retry(client: httpx.Client, url: str,
                     max_retries: int = 3, backoff: float = 2.0,
                     rate_limiter=None) -> httpx.Response | None:
    if rate_limiter:
        try:
            domain = urlparse(url).netloc.replace("www.", "")
            rate_limiter.wait(domain)
        except Exception:
            pass

    for attempt in range(max_retries):
        try:
            resp = client.get(url)
            if resp.status_code == 200:
                return resp
            elif resp.status_code == 429:
                wait = backoff ** (attempt + 2)
                logger.warning(f"Rate limited (429) dari {url[:60]}. Tunggu {wait:.0f}s...")
                time.sleep(wait)
            elif resp.status_code in (403, 404):
                logger.debug(f"HTTP {resp.status_code} — skip: {url[:80]}")
                return resp
            else:
                logger.debug(f"HTTP {resp.status_code} untuk {url[:80]}")
                return resp
        except Exception as e:
            logger.warning(f"Attempt {attempt+1}/{max_retries} gagal ({url[:60]}): {e}")
            if attempt < max_retries - 1:
                time.sleep(backoff ** (attempt + 1))
    return None


def _extract_links(soup: BeautifulSoup, selectors: list,
                   base_url: str, url_filter_fn) -> list:
    """Helper generik: ekstrak href dari soup dengan beberapa selector."""
    found = []
    seen = set()
    for selector in selectors:
        for link in soup.select(selector):
            href = link.get("href", "").strip()
            if not href:
                continue
            if href.startswith("/"):
                href = base_url.rstrip("/") + href
            if href in seen:
                continue
            if url_filter_fn(href):
                seen.add(href)
                found.append({
                    "url": href,
                    "title": link.get_text(strip=True),
                })
    return found


def discover_urls_antara(client: httpx.Client, keyword: str,
                         source_cfg: dict, max_pages: int, rate_limiter=None) -> list:
    """Antara News & Antara Sumut — arsip berita nasional/regional."""
    urls = []
    base_url = source_cfg.get("base_url", "https://www.antaranews.com")

    for page in range(1, max_pages + 1):
        search_url = source_cfg["search_url"].format(
            query=quote_plus(keyword), page=page
        )
        resp = fetch_with_retry(client, search_url, rate_limiter=rate_limiter)
        if resp is None or resp.status_code != 200:
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        links = _extract_links(
            soup,
            selectors=["a[href*='/berita/']", ".simple-listing a", ".latest a", "article a"],
            base_url=base_url,
            url_filter_fn=lambda h: "/berita/" in h,
        )
        if not links:
            break
        for l in links:
            l["source_keyword"] = keyword
        urls.extend(links)

    return urls


def discover_urls_detik(client: httpx.Client, keyword: str,
                        source_cfg: dict, max_pages: int, rate_limiter=None) -> list:
    """Detik.com & Detik Sumut."""
    urls = []

    for page in range(1, max_pages + 1):
        search_url = source_cfg["search_url"].format(
            query=quote_plus(keyword), page=page
        )
        resp = fetch_with_retry(client, search_url, rate_limiter=rate_limiter)
        if resp is None or resp.status_code != 200:
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        links = _extract_links(
            soup,
            selectors=["article a[href]", "h2 a[href]", "h3 a[href]"],
            base_url="https://www.detik.com",
            url_filter_fn=lambda h: "detik.com" in h and ("/d-" in h or "/berita/" in h),
        )
        if not links:
            break
        for l in links:
            l["source_keyword"] = keyword
        urls.extend(links)

    return urls


def discover_urls_kompas(client: httpx.Client, keyword: str,
                         source_cfg: dict, max_pages: int, rate_limiter=None) -> list:
    """Kompas.com."""
    urls = []

    for page in range(1, max_pages + 1):
        search_url = source_cfg["search_url"].format(
            query=quote_plus(keyword), page=page
        )
        resp = fetch_with_retry(client, search_url, rate_limiter=rate_limiter)
        if resp is None or resp.status_code != 200:
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        links = _extract_links(
            soup,
            selectors=["a.article__link", ".gsc-thumbnail-inside a",
                       "div.article-list a", "a[href*='/read/']"],
            base_url="https://www.kompas.com",
            url_filter_fn=lambda h: ("kompas.com" in h or "kompas.id" in h),
        )
        if not links:
            break
        for l in links:
            l["source_keyword"] = keyword
        urls.extend(links)

    return urls


def discover_urls_tribun(client: httpx.Client, keyword: str,
                         source_cfg: dict, max_pages: int, rate_limiter=None) -> list:
    """Tribunnews.com & TribunMedan.com."""
    urls = []

    for page in range(1, max_pages + 1):
        search_url = source_cfg["search_url"].format(
            query=quote_plus(keyword), page=page
        )
        resp = fetch_with_retry(client, search_url, rate_limiter=rate_limiter)
        if resp is None or resp.status_code != 200:
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        links = _extract_links(
            soup,
            selectors=["h3 a[href]", "li.art-list a[href]", ".f20 a[href]"],
            base_url="https://www.tribunnews.com",
            url_filter_fn=lambda h: ("tribunnews.com" in h or "tribunmedan.com" in h),
        )
        if not links:
            break
        for l in links:
            l["source_keyword"] = keyword
        urls.extend(links)

    return urls


def discover_urls_waspada(client: httpx.Client, keyword: str,
                          source_cfg: dict, max_pages: int, rate_limiter=None) -> list:
    """
    Waspada.id — media regional Sumatera Utara.
    Ditambahkan di v3 karena tercantum di proposal penelitian (bagian 3.2.3)
    tapi tidak ada di v1/v2.
    URL pattern: waspada.id/YYYY/MM/DD/slug/
    """
    urls = []

    for page in range(1, max_pages + 1):
        search_url = source_cfg["search_url"].format(
            query=quote_plus(keyword), page=page
        )
        resp = fetch_with_retry(client, search_url, rate_limiter=rate_limiter)
        if resp is None or resp.status_code != 200:
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        links = _extract_links(
            soup,
            selectors=[
                "h2.entry-title a", "h3.entry-title a",
                ".post-title a", "article a[href]",
                "a[href*='waspada.id']",
            ],
            base_url="https://www.waspada.id",
            url_filter_fn=lambda h: "waspada.id" in h,
        )
        if not links:
            break
        for l in links:
            l["source_keyword"] = keyword
        urls.extend(links)

    return urls


def discover_urls_google_rss(client: httpx.Client, keyword: str,
                              source_cfg: dict, max_pages: int,
                              date_start: str = None, date_end: str = None, rate_limiter=None) -> list:
    """
    Google News RSS — fallback untuk artikel terkini saja.
    CATATAN: RSS hanya menyediakan ~100 artikel terbaru. Tidak bisa
    digunakan untuk coverage historis 2021–2026. Dipakai sebagai
    safety net untuk berita minggu terakhir yang belum masuk arsip.

    Perbaikan v3: tambah date range filter agar artikel di luar
    rentang penelitian tidak masuk korpus.
    """
    urls = []
    encoded_kw = quote_plus(keyword)
    rss_url = (
        f"https://news.google.com/rss/search"
        f"?q={encoded_kw}&hl=id&gl=ID&ceid=ID:id"
    )

    resp = fetch_with_retry(client, rss_url, rate_limiter=rate_limiter)
    if resp is None or resp.status_code != 200:
        return urls

    dt_start = datetime.strptime(date_start, "%Y-%m-%d") if date_start else None
    dt_end   = datetime.strptime(date_end,   "%Y-%m-%d") if date_end   else None

    try:
        soup = BeautifulSoup(resp.text, "xml")
        for item in soup.find_all("item"):
            link_elem     = item.find("link")
            title_elem    = item.find("title")
            pub_date_elem = item.find("pubDate")

            if not link_elem:
                continue

            raw_url = link_elem.text.strip() if link_elem.text else ""
            decoded_url = raw_url
            if _HAS_GNEWSDECODER and "news.google.com" in raw_url:
                try:
                    result = gnewsdecoder(raw_url, interval=0.5)
                    if result.get("status") and result.get("decoded_url"):
                        decoded_url = result["decoded_url"]
                except Exception:
                    pass

            # Date range filter — BARU di v3
            pub_date_str = pub_date_elem.text.strip() if pub_date_elem and pub_date_elem.text else ""
            parsed_date  = parse_pub_date(pub_date_str)
            if parsed_date and (dt_start or dt_end):
                try:
                    article_dt = datetime.strptime(parsed_date, "%Y-%m-%d")
                    if dt_start and article_dt < dt_start:
                        continue
                    if dt_end and article_dt > dt_end:
                        continue
                except ValueError:
                    pass

            urls.append({
                "url": decoded_url,
                "title": title_elem.text.strip() if title_elem and title_elem.text else "",
                "pub_date": parsed_date,
                "source_keyword": keyword,
            })

    except Exception as e:
        logger.error(f"Error parsing Google RSS untuk '{keyword}': {e}")

    return urls


# Source name → discovery function
DISCOVERERS = {
    "antaranews":       discover_urls_antara,
    "antaranews_sumut": discover_urls_antara,
    "detik":            discover_urls_detik,
    "detik_sumut":      discover_urls_detik,
    "kompas":           discover_urls_kompas,
    "tribunnews":       discover_urls_tribun,
    "waspada":          discover_urls_waspada,   # BARU v3
    "google_news_rss":  discover_urls_google_rss,
}


# ─────────────────────────────────────────────────────────────
# Article Text Extraction
# ─────────────────────────────────────────────────────────────

def extract_article_text(url: str, html_cache_dir: Path) -> tuple[str, str, str]:
    """
    Ekstrak teks artikel menggunakan trafilatura.

    CATATAN ARSITEKTUR (v3): trafilatura.fetch_url() digunakan secara
    langsung karena ia menangani redirect kompleks, paywall detection,
    dan encoding secara lebih robust daripada httpx biasa untuk konten
    artikel. httpx client digunakan HANYA di fase URL discovery (search
    pages) di mana custom headers diperlukan. Ini adalah pembagian
    tanggung jawab yang disengaja, bukan bug.

    Returns: (full_text, source_domain, published_date_str)
             published_date_str adalah "" jika tidak ditemukan —
             TIDAK pernah di-default ke tanggal hari ini.
    """
    if trafilatura is None:
        logger.warning("trafilatura tidak terinstall — ekstraksi tidak bisa dilakukan")
        return "", "unknown", ""

    try:
        url_hash   = hashlib.md5(url.encode()).hexdigest()[:12]
        cache_file = html_cache_dir / f"{url_hash}.html"

        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                downloaded = f.read()
        else:
            downloaded = None
            for attempt in range(3):
                try:
                    downloaded = trafilatura.fetch_url(url)
                    if downloaded is not None:
                        break
                except Exception as e:
                    logger.debug(f"Download attempt {attempt+1}/3 gagal ({url[:60]}): {e}")
                    time.sleep(2)

            if downloaded is None:
                return "", "unknown", ""

            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(downloaded)
            except Exception:
                pass

        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
        ) or ""

        # Ekstrak tanggal dari metadata HTML
        pub_date = ""
        try:
            metadata = trafilatura.extract_metadata(downloaded)
            if metadata and metadata.date:
                pub_date = str(metadata.date)
        except Exception:
            pass

        source = "unknown"
        try:
            source = urlparse(url).netloc.replace("www.", "")
        except Exception:
            pass

        return text, source, pub_date

    except Exception as e:
        logger.error(f"Ekstraksi gagal ({url[:80]}): {e}")
        return "", "unknown", ""


def parse_pub_date(date_str: str) -> str:
    """Parse berbagai format tanggal ke YYYY-MM-DD. Return "" jika gagal."""
    if not date_str:
        return ""
    date_str = date_str.strip()

    for fmt in [
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d %B %Y",
        "%d %b %Y",
    ]:
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    match = re.search(r"(\d{4}-\d{2}-\d{2})", date_str)
    if match:
        return match.group(1)
    return ""


def extract_date_from_url(url: str) -> str:
    """
    Ekstrak tanggal dari pola URL.
    Kompas: /read/YYYY/MM/DD/
    Tribun & Waspada: /YYYY/MM/DD/
    """
    match = re.search(r"/read/(\d{4})/(\d{2})/(\d{2})/", url)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"

    match = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", url)
    if match:
        year = int(match.group(1))
        if 2018 <= year <= 2027:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"

    return ""


# ─────────────────────────────────────────────────────────────
# Phase 1: URL Discovery
# ─────────────────────────────────────────────────────────────

def _discover_single_query(
    source_key: str, src_cfg: dict, keyword: str,
    cfg: dict, max_pages: int, state: dict,
    ckpt_mgr: CheckpointManager,
    rate_limiter: DomainRateLimiter,
    concurrency_limiter: DomainConcurrencyLimiter,
) -> tuple[list, str, str, str]:
    """
    Worker: jalankan discovery untuk SATU (source, keyword) pair secara paralel.
    Setiap worker membuat httpx.Client SENDIRI (thread-safe, mencegah WinError 10060).

    Returns: (results, source_name, keyword, status_emoji)
    """
    discoverer  = DISCOVERERS.get(source_key)
    source_name = src_cfg.get("name", source_key)
    date_start  = cfg["corpus"]["date_start"]
    date_end    = cfg["corpus"]["date_end"]
    results     = []
    status      = "✓"

    if discoverer is None:
        logger.warning(f"[{source_name}] Tidak ada discoverer untuk '{source_key}'. Skip.")
        return results, source_name, keyword, "⚠ no_discoverer"

    if ckpt_mgr.is_query_done(state, source_key, keyword):
        return results, source_name, keyword, "↩ cached"

    # Tentukan domain untuk concurrency limiting
    base_domain = urlparse(src_cfg.get("search_url", "")).netloc or source_key
    concurrency_limiter.acquire(base_domain)
    try:
        # Setiap thread pakai client sendiri
        with get_http_client(cfg) as client:
            if source_key == "google_news_rss":
                found = discoverer(
                    client, keyword, src_cfg, max_pages,
                    date_start=date_start, date_end=date_end, rate_limiter=rate_limiter
                )
            else:
                found = discoverer(client, keyword, src_cfg, max_pages, rate_limiter=rate_limiter)

        for u in found:
            u["source_name"] = source_name

        results.extend(found)
        ckpt_mgr.add_urls(state, found)
        ckpt_mgr.mark_query_done(state, source_key, keyword)
        status = f"✓ {len(found)} URL"

    except Exception as e:
        err_str = str(e)
        if "WinError 10060" in err_str or "timed out" in err_str.lower() or "ConnectError" in err_str:
            logger.warning(f"[{source_name}] Timeout '{keyword}' — akan di-skip: {err_str[:80]}")
            status = "⏱ timeout"
        else:
            logger.error(f"[{source_name}] GAGAL '{keyword}': {err_str[:120]}")
            status = "✗ error"
    finally:
        concurrency_limiter.release(base_domain)

    return results, source_name, keyword, status


def discover_all_urls(cfg: dict) -> list:
    scraper_cfg  = cfg.get("scrapers", {})
    sources      = scraper_cfg.get("sources", {})
    max_pages    = scraper_cfg.get("max_pages_per_keyword", 30)
    rate_limit   = scraper_cfg.get("rate_limit_sec", 3)
    # Max concurrent threads per domain — mencegah WinError 10060
    max_concurrent_per_domain = scraper_cfg.get("max_concurrent_per_domain", 2)

    kw_cfg = cfg.get("keywords", {})
    all_keywords = kw_cfg.get("primary", []) + kw_cfg.get("secondary", [])

    checkpoint_dir = ROOT_DIR / "data" / "raw" / "checkpoints"
    ckpt_mgr = CheckpointManager(checkpoint_dir)
    state    = ckpt_mgr.load()

    # Load URL yang sudah ada dari checkpoint (untuk resume)
    all_urls     = []
    url_metadata = state.get("url_metadata", {})
    for url in state.get("collected_urls", set()):
        meta = url_metadata.get(url, {})
        all_urls.append({
            "url":            url,
            "title":          meta.get("title", ""),
            "source_keyword": meta.get("source_keyword", ""),
            "source_name":    meta.get("source_name", "unknown"),
        })

    sorted_sources = sorted(
        [(k, v) for k, v in sources.items() if v.get("enabled", True)],
        key=lambda x: x[1].get("priority", 99),
    )

    # ── Build task list (skip yang sudah selesai di checkpoint) ──
    tasks = []
    for source_key, src_cfg in sorted_sources:
        for keyword in all_keywords:
            if not ckpt_mgr.is_query_done(state, source_key, keyword):
                tasks.append((source_key, src_cfg, keyword))

    already_done = sum(
        1 for sk, scfg in sorted_sources for kw in all_keywords
        if ckpt_mgr.is_query_done(state, sk, kw)
    )
    total_queries = len(tasks) + already_done

    logger.info(
        f"Discovery Phase 1 — "
        f"{len(tasks)} queries tersisa dari total {total_queries} "
        f"({already_done} sudah selesai di checkpoint)."
    )

    if not tasks:
        logger.info("✓ Semua queries sudah selesai. Langsung ke Phase 2.")
    else:
        # ── Parallelisasi Per (Source × Keyword) dengan domain concurrency control ──
        # Setiap worker membuat httpx.Client sendiri (fix WinError 10060).
        # DomainRateLimiter  : jeda minimum antar request ke DOMAIN YANG SAMA.
        # DomainConcurrencyLimiter: max N thread simultan ke DOMAIN YANG SAMA.
        n_workers = min(16, len(tasks))
        rate_limiter       = DomainRateLimiter(min_interval=max(rate_limit, 1.5))
        concurrency_lim    = DomainConcurrencyLimiter(max_concurrent=max_concurrent_per_domain)

        # Statistik per sumber (untuk summary di akhir)
        domain_stats: dict[str, dict] = defaultdict(lambda: {"ok": 0, "timeout": 0, "error": 0, "urls": 0})

        url_count_lock = threading.Lock()
        total_urls_found = [0]  # mutable counter untuk diakses di closure

        pbar = tqdm(
            total=len(tasks),
            desc="Phase 1 · URL Discovery",
            unit="query",
            dynamic_ncols=True,
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} queries "
                "[{elapsed}<{remaining}, {rate_fmt}] "
                "{postfix}"
            ),
        )
        pbar.set_postfix(urls=0, done=already_done, refresh=False)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _discover_single_query,
                    source_key, src_cfg, keyword,
                    cfg, max_pages, state, ckpt_mgr,
                    rate_limiter, concurrency_lim,
                ): (src_cfg.get("name", source_key), keyword)
                for source_key, src_cfg, keyword in tasks
            }

            try:
                for fut in as_completed(futures, timeout=600):  # 10 menit max total
                    sname, kw = futures[fut]
                    try:
                        found, sname, kw, status = fut.result(timeout=300)  # 5 menit per query max
                        all_urls.extend(found)
                        n = len(found)

                        with url_count_lock:
                            total_urls_found[0] += n

                        # Update statistik
                        if "timeout" in status:
                            domain_stats[sname]["timeout"] += 1
                        elif "error" in status:
                            domain_stats[sname]["error"] += 1
                        else:
                            domain_stats[sname]["ok"] += 1
                            domain_stats[sname]["urls"] += n

                        # Log per query — hanya yang menghasilkan URL atau error
                        if n > 0:
                            logger.info(f"  [{sname}] '{kw}' → {n} URL  ({status})")
                        elif "error" in status or "timeout" in status:
                            logger.warning(f"  [{sname}] '{kw}' → {status}")

                    except TimeoutError:
                        domain_stats[sname]["timeout"] += 1
                        logger.warning(f"  [{sname}] '{kw}' — TIMEOUT (>5 menit), di-skip")
                    except Exception as e:
                        domain_stats[sname]["error"] += 1
                        logger.error(f"  [{sname}] '{kw}' — unexpected error: {e}")
                    finally:
                        pbar.update(1)
                        pbar.set_postfix(
                            urls=total_urls_found[0],
                            done=already_done + pbar.n,
                            refresh=True,
                        )
            except TimeoutError:
                logger.warning("⚠ Phase 1 global timeout (10 menit). Melanjutkan dengan URL yang sudah ditemukan.")

        pbar.close()

        # ── Summary per sumber ──
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1 SUMMARY — Discovery per Sumber")
        logger.info("=" * 60)
        for src, stats in sorted(domain_stats.items()):
            logger.info(
                f"  {src:<25} ok={stats['ok']:>3}  "
                f"timeout={stats['timeout']:>2}  "
                f"error={stats['error']:>2}  "
                f"urls={stats['urls']:>5}"
            )
        logger.info("=" * 60)

    seen = set()
    unique_urls = []
    for entry in all_urls:
        url = entry["url"]
        if url not in seen:
            seen.add(url)
            unique_urls.append(entry)

    logger.info(f"Total URL unik: {len(unique_urls)} (dari {len(all_urls)} raw)")
    return unique_urls


# ─────────────────────────────────────────────────────────────
# Phase 2: Article Extraction
# ─────────────────────────────────────────────────────────────

def _extract_single_article(entry: dict, html_cache_dir: Path,
                             rate_limiter: DomainRateLimiter) -> tuple[dict, str]:
    """
    Worker: ekstrak satu artikel. Dipanggil dari ThreadPoolExecutor.
    Returns: (result_dict, status) dimana status = 'accepted' | 'rejected' | 'skip'
    """
    url = entry["url"]

    # Rate-limit per domain agar server tidak kena flood
    try:
        domain = urlparse(url).netloc.replace("www.", "")
    except Exception:
        domain = "unknown"
    rate_limiter.wait(domain)

    full_text, source, pub_date = extract_article_text(url, html_cache_dir)

    if not full_text:
        return {
            "url": url, "source": source,
            "published_date": pub_date,
            "rejection_reason": "no_text",
            "title": entry.get("title", ""),
        }, "rejected"

    return {
        "url": url, "source": source, "pub_date": pub_date,
        "full_text": full_text, "title": entry.get("title", ""),
        "source_keyword": entry.get("source_keyword", ""),
        "source_name": entry.get("source_name", "unknown"),
    }, "extracted"


def extract_all_articles(url_entries: list, cfg: dict,
                         existing_urls: set) -> tuple[list, list]:
    """
    Ekstrak teks dari URL yang ditemukan — PARALEL per domain.

    Returns:
        articles_accepted : lolos date range + geo-filter → corpus_raw.jsonl
        articles_rejected : gagal geo-filter → corpus_rejected.jsonl (untuk audit)
    """
    html_cache_dir = ROOT_DIR / "data" / "raw" / "html_cache"
    html_cache_dir.mkdir(parents=True, exist_ok=True)

    scraper_cfg         = cfg.get("scrapers", {})
    rate_limit          = scraper_cfg.get("rate_limit_sec", 3)
    checkpoint_interval = scraper_cfg.get("checkpoint_interval", 50)

    date_start = datetime.strptime(cfg["corpus"]["date_start"], "%Y-%m-%d")
    date_end   = datetime.strptime(cfg["corpus"]["date_end"],   "%Y-%m-%d")

    kw_cfg = cfg.get("keywords", {})
    all_keywords = kw_cfg.get("primary", []) + kw_cfg.get("secondary", [])

    min_chars = cfg["corpus"].get("min_length_chars", 500)
    max_chars = cfg["corpus"].get("max_length_chars", 30000)

    # ── Layer skip tambahan: HTML cache ──
    # URL yang HTML-nya sudah pernah di-download (ada di html_cache/)
    # tapi tidak masuk corpus_raw.jsonl berarti pernah ditolak di run sebelumnya.
    # Kita skip URL ini juga untuk menghindari re-parsing yang sia-sia.
    cached_hashes = {f.stem for f in html_cache_dir.glob("*.html")}
    skip_from_explicit = 0
    skip_from_cache    = 0
    to_process = []
    for e in url_entries:
        url = e["url"]
        if url in existing_urls:
            skip_from_explicit += 1
            continue
        # Cek apakah HTML-nya sudah pernah di-download
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        if url_hash in cached_hashes:
            skip_from_cache += 1
            continue
        to_process.append(e)

    logger.info(
        f"URL untuk diekstrak: {len(to_process)} "
        f"(skip {skip_from_explicit} sudah di corpus, "
        f"{skip_from_cache} sudah pernah diproses [html_cache])"
    )

    articles_accepted = []
    articles_rejected = []
    rejected_saved_count = 0
    output_path   = ROOT_DIR / "data" / "raw" / "corpus_raw.jsonl"
    rejected_path = ROOT_DIR / "data" / "raw" / "corpus_rejected.jsonl"

    # ── Parallelisasi dengan per-domain rate limiting ──
    # Menggunakan DomainRateLimiter: request ke domain yang SAMA tetap
    # dijeda (rate_limit detik), tapi request ke domain BERBEDA berjalan
    # paralel. Ini menghormati server sambil mempercepat total throughput.
    rate_limiter = DomainRateLimiter(min_interval=max(rate_limit, 1.5))
    max_workers  = min(8, max(len(set(
        urlparse(e["url"]).netloc for e in to_process[:200]
    )), 1))
    logger.info(f"Extraction paralel: {max_workers} workers, "
                f"rate limit {rate_limit}s per domain")

    pbar = tqdm(total=len(to_process), desc="Extracting articles")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _extract_single_article, entry, html_cache_dir, rate_limiter
            ): entry
            for entry in to_process
        }

        for fut in as_completed(futures):
            pbar.update(1)
            try:
                result, status = fut.result()
            except Exception as e:
                entry = futures[fut]
                logger.debug(f"Extraction error ({entry['url'][:60]}): {e}")
                continue

            if status == "rejected":
                articles_rejected.append(result)
                continue

            # ── Post-processing (ringan, dilakukan di main thread) ──
            url       = result["url"]
            full_text = result["full_text"]
            source    = result["source"]
            pub_date  = result["pub_date"]
            title     = result["title"]

            # Length filter
            char_len = len(full_text)
            if char_len < min_chars or char_len > max_chars:
                articles_rejected.append({
                    "url": url, "source": source,
                    "published_date": pub_date,
                    "rejection_reason": f"length_out_of_range:{char_len}",
                    "title": title,
                })
                continue

            # Tentukan tanggal publikasi
            if not pub_date:
                pub_date = result.get("pub_date", "")
            if not pub_date:
                pub_date = extract_date_from_url(url)
            pub_date = parse_pub_date(pub_date)

            if not pub_date:
                articles_rejected.append({
                    "url": url, "source": source,
                    "rejection_reason": "date_unknown",
                    "title": title,
                })
                continue

            # Filter date range
            try:
                article_dt = datetime.strptime(pub_date, "%Y-%m-%d")
                if article_dt < date_start or article_dt > date_end:
                    continue
            except ValueError:
                continue

            # Geo-relevance filter
            if not is_geo_relevant(cfg, full_text, title, min_hits=1):
                articles_rejected.append({
                    "url": url, "source": source,
                    "published_date": pub_date,
                    "rejection_reason": "geo_irrelevant",
                    "title": title,
                })
                continue

            matched_kws = keyword_matches(full_text + " " + title, all_keywords)

            record = {
                "article_id":     make_article_id(url, pub_date),
                "url":            url,
                "source":         source,
                "title":          title,
                "published_date": pub_date,
                "crawled_date":   datetime.now().strftime("%Y-%m-%d"),
                "full_text":      full_text,
                "char_length":    len(full_text),
                "keyword_matched": matched_kws,
                "source_keyword": result.get("source_keyword", ""),
                "source_name":    result.get("source_name", "unknown"),
                "commodities_mentioned": [
                    c for c in COMMODITIES_VF_CORE
                    if c.lower() in (full_text + " " + title).lower()
                ],
            }
            articles_accepted.append(record)

            if len(articles_accepted) % checkpoint_interval == 0 and articles_accepted:
                save_jsonl(articles_accepted[-checkpoint_interval:], output_path)
                if len(articles_rejected) > rejected_saved_count:
                    save_jsonl(articles_rejected[rejected_saved_count:], rejected_path)
                    rejected_saved_count = len(articles_rejected)
                logger.info(f"  Checkpoint: {len(articles_accepted)} accepted, "
                            f"{len(articles_rejected)} rejected")

    pbar.close()
    return articles_accepted, articles_rejected


# ─────────────────────────────────────────────────────────────
# Coverage Analysis
# ─────────────────────────────────────────────────────────────

def analyze_coverage(articles: list, cfg: dict) -> dict:
    """
    Analisis coverage: per media, per bulan, per komoditas.
    Flag bulan dengan coverage di bawah threshold (dari config, default 20).
    """
    if not articles:
        return {"total": 0, "by_source": {}, "by_month": {}, "low_coverage_months": []}

    quality_cfg = cfg.get("quality", {})
    # Threshold dipindah ke config (v3) — tidak lagi hard-coded
    low_coverage_threshold = quality_cfg.get("low_coverage_threshold", 20)

    source_counts    = Counter(a["source"] for a in articles)
    month_counts     = Counter()
    commodity_counts = Counter()

    for a in articles:
        if a.get("published_date"):
            month_counts[a["published_date"][:7]] += 1
        for c in a.get("commodities_mentioned", []):
            commodity_counts[c] += 1

    low_coverage = [m for m, c in month_counts.items() if c < low_coverage_threshold]

    date_start = cfg["corpus"]["date_start"][:7]
    date_end   = cfg["corpus"]["date_end"][:7]
    all_months = set()
    current    = datetime.strptime(date_start + "-01", "%Y-%m-%d")
    end        = datetime.strptime(date_end   + "-01", "%Y-%m-%d")
    while current <= end:
        all_months.add(current.strftime("%Y-%m"))
        current = (current + timedelta(days=32)).replace(day=1)

    missing_months = sorted(all_months - set(month_counts.keys()))

    summary = {
        "total":                len(articles),
        "by_source":            dict(source_counts.most_common()),
        "by_month":             dict(sorted(month_counts.items())),
        "by_commodity":         dict(commodity_counts.most_common()),
        "low_coverage_months":  sorted(low_coverage),
        "missing_months":       missing_months,
        "coverage_pct":         round(
            len(set(month_counts.keys()) & all_months) / max(len(all_months), 1) * 100, 1
        ),
        "low_coverage_threshold_used": low_coverage_threshold,
    }

    dates = [a["published_date"] for a in articles if a.get("published_date")]
    if dates:
        summary["date_range"] = {"start": min(dates), "end": max(dates)}

    return summary


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def run_collection(cfg: dict = None) -> None:
    """
    Pipeline koleksi artikel lengkap.
      Phase 1: Discover URLs dari arsip situs berita
      Phase 2: Ekstrak teks artikel + geo-filter
      Phase 3: Coverage analysis
    """
    if cfg is None:
        cfg, _ = get_cfg_and_logger()

    output_path   = ROOT_DIR / "data" / "raw" / "corpus_raw.jsonl"
    rejected_path = ROOT_DIR / "data" / "raw" / "corpus_rejected.jsonl"

    # ── Load existing data untuk skip ──
    existing      = load_jsonl(output_path)
    existing_ids  = {a["article_id"] for a in existing}
    existing_urls = {a["url"] for a in existing}
    logger.info(f"Artikel existing di corpus_raw: {len(existing)} ({len(existing_urls)} unique URLs)")

    # BARU: juga load URL yang pernah ditolak — agar tidak di-proses ulang
    rejected_existing = load_jsonl(rejected_path)
    rejected_urls     = {r["url"] for r in rejected_existing if r.get("url")}
    logger.info(f"Artikel rejected sebelumnya   : {len(rejected_urls)} URLs (akan di-skip)")

    # Gabungkan: URL yang sudah pernah di-proses (accepted + rejected)
    all_processed_urls = existing_urls | rejected_urls
    logger.info(f"Total URL sudah diproses      : {len(all_processed_urls)}")

    # Phase 1
    logger.info("=" * 60)
    logger.info("PHASE 1: URL Discovery dari Arsip Situs Berita")
    logger.info("=" * 60)
    url_entries = discover_all_urls(cfg)

    # Phase 2
    logger.info("=" * 60)
    logger.info("PHASE 2: Ekstraksi Teks Artikel + Geo-Filter")
    logger.info("=" * 60)
    new_articles, rejected = extract_all_articles(url_entries, cfg, all_processed_urls)

    truly_new = [a for a in new_articles if a["article_id"] not in existing_ids]
    logger.info(f"Artikel baru: {len(truly_new)} (dari {len(new_articles)} diekstrak)")
    logger.info(f"Artikel ditolak (geo/date): {len(rejected)}")

    if truly_new:
        save_jsonl(truly_new, output_path)
        logger.info(f"Tersimpan {len(truly_new)} artikel baru → {output_path}")

    if rejected:
        save_jsonl(rejected, rejected_path)
        logger.info(f"Tersimpan {len(rejected)} artikel ditolak → {rejected_path}")

    # Phase 3
    all_combined = existing + truly_new
    summary      = analyze_coverage(all_combined, cfg)

    summary_path = ROOT_DIR / "data" / "raw" / "coverage_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("COLLECTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total artikel          : {summary['total']}")
    logger.info(f"Coverage               : {summary.get('coverage_pct', 0)}% bulan terkover")
    logger.info(f"Top sources            : {dict(list(summary['by_source'].items())[:5])}")
    logger.info(f"Top komoditas          : {dict(list(summary['by_commodity'].items())[:8])}")
    if summary.get("date_range"):
        logger.info(f"Date range             : {summary['date_range']['start']} — {summary['date_range']['end']}")
    if summary["low_coverage_months"]:
        logger.warning(
            f"Bulan low coverage (<{summary['low_coverage_threshold_used']} artikel): "
            f"{summary['low_coverage_months']}"
        )
    if summary.get("missing_months"):
        logger.warning(f"Bulan tidak ada data   : {summary['missing_months']}")
    logger.info(f"Coverage summary       → {summary_path}")


if __name__ == "__main__":
    cfg, log = get_cfg_and_logger(skip_env=True)
    run_collection(cfg)
