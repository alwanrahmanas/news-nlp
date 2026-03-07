"""
M0 — SETUP & STRUKTUR PROYEK
=============================
Tugas:
  1. Buat semua folder di struktur proyek jika belum ada
  2. Setup logger ke file logs/pipeline.log DAN stdout
  3. Load .env dan pastikan OPENAI_API_KEY ada
  4. Load config.yaml dan kembalikan sebagai dict global CFG
  5. Print ringkasan konfigurasi aktif ke log
"""

import os
import sys
import logging
import json
from pathlib import Path
from datetime import datetime

import yaml
from dotenv import load_dotenv

# ── Paths ────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent  # nlp_pipeline_v2/

REQUIRED_DIRS = [
    ROOT_DIR / "data" / "raw",
    ROOT_DIR / "data" / "raw" / "html_cache",
    ROOT_DIR / "data" / "raw" / "checkpoints",
    ROOT_DIR / "data" / "clean",
    ROOT_DIR / "data" / "processed",
    ROOT_DIR / "data" / "validation",
    ROOT_DIR / "data" / "external",
    ROOT_DIR / "prompts",
    ROOT_DIR / "outputs" / "granger_plots",
    ROOT_DIR / "logs",
    ROOT_DIR / "src",
]


def create_directories() -> None:
    """Buat semua folder yang diperlukan."""
    for d in REQUIRED_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def setup_logger(log_level: int = logging.INFO) -> logging.Logger:
    """
    Setup logger ke file logs/pipeline.log DAN stdout secara bersamaan.
    """
    log_dir = ROOT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("nlp_pipeline")
    logger.setLevel(log_level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    fh = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
    fh.setLevel(log_level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(log_level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


def load_env() -> None:
    """Load .env dan pastikan OPENAI_API_KEY ada."""
    env_path = ROOT_DIR / ".env"
    load_dotenv(dotenv_path=env_path)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-..."):
        raise EnvironmentError(
            f"OPENAI_API_KEY tidak ditemukan atau masih placeholder. "
            f"Pastikan file .env ada di {env_path} dan berisi API key yang valid."
        )


def load_config() -> dict:
    """Load config.yaml dan kembalikan sebagai dict."""
    config_path = ROOT_DIR / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml tidak ditemukan di {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg


def print_config_summary(cfg: dict, logger: logging.Logger) -> None:
    """Print ringkasan konfigurasi aktif ke log."""
    logger.info("=" * 60)
    logger.info("KONFIGURASI PIPELINE NLP/LLM v2 AKTIF")
    logger.info("=" * 60)
    logger.info(f"Project         : {cfg['project']['name']}")
    logger.info(f"Timezone        : {cfg['project']['timezone']}")
    logger.info(f"Corpus period   : {cfg['corpus']['date_start']} s.d. {cfg['corpus']['date_end']}")
    logger.info(f"LLM model       : {cfg['llm']['model']}")
    logger.info(f"LLM max_tokens  : {cfg['llm'].get('max_completion_tokens', 'N/A')}")
    logger.info(f"Batch size      : {cfg['llm']['batch_size']}")
    logger.info(f"Labels          : {cfg['labels']['valid']}")
    logger.info(f"Confidence thr  : {cfg['sentiment']['confidence_threshold']}")
    logger.info(f"Aggregation     : {cfg['aggregation']['frequency']}")
    logger.info(f"Decay lambda    : {cfg['aggregation']['decay_lambda']}")
    logger.info(f"Granger max lag : {cfg['granger']['max_lag_periods']}")
    logger.info(f"Granger alpha   : {cfg['granger']['significance_level']}")

    # Scraper sources
    sources = cfg.get("scrapers", {}).get("sources", {})
    enabled_sources = [v["name"] for v in sources.values() if v.get("enabled")]
    logger.info(f"Scraper sources : {len(enabled_sources)} ({', '.join(enabled_sources)})")

    # Keywords
    n_primary = len(cfg["keywords"]["primary"])
    n_secondary = len(cfg["keywords"]["secondary"])
    logger.info(f"Keywords        : {n_primary} primary, {n_secondary} secondary")
    logger.info("=" * 60)


def initialize_pipeline(skip_env_check: bool = False) -> tuple:
    """
    Master initialization function.
    Returns: (cfg: dict, logger: logging.Logger)
    """
    # 1. Create directories
    create_directories()

    # 2. Setup logger
    logger = setup_logger()
    logger.info(f"Pipeline v2 initialized at {datetime.now().isoformat()}")
    logger.info(f"Root directory: {ROOT_DIR}")

    # 3. Load .env
    if not skip_env_check:
        load_env()
        logger.info("OPENAI_API_KEY loaded successfully")
    else:
        load_dotenv(dotenv_path=ROOT_DIR / ".env")
        logger.info("Environment check skipped (skip_env_check=True)")

    # 4. Load config
    cfg = load_config()
    logger.info("config.yaml loaded successfully")

    # 5. Print summary
    print_config_summary(cfg, logger)

    return cfg, logger


# ── Convenience ──────────────────────────────────────────────
CFG = None
LOG = None


def get_cfg_and_logger(skip_env: bool = False):
    """Singleton-style getter untuk CFG dan LOG."""
    global CFG, LOG
    if CFG is None or LOG is None:
        CFG, LOG = initialize_pipeline(skip_env_check=skip_env)
    return CFG, LOG


if __name__ == "__main__":
    cfg, logger = initialize_pipeline(skip_env_check=True)
    logger.info("M0 Setup complete. All directories and configs ready.")
