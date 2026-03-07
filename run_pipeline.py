"""
run_pipeline.py — Master Pipeline Runner v2
==============================================
Jalankan seluruh pipeline NLP/LLM dari ujung ke ujung.
Versi 2: Menggunakan direct archive scraping untuk coverage 5 tahun.

Usage:
  python run_pipeline.py                     # Run all modules
  python run_pipeline.py --from m3           # Start from module M3
  python run_pipeline.py --only m0           # Run only M0
  python run_pipeline.py --only m0 m3        # Run only M0 and M3
  python run_pipeline.py --skip-env          # Skip environment check
  python run_pipeline.py --no-llm            # Skip LLM calls (rule-based only)
"""

import sys
import argparse
import time
from datetime import datetime
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description="NLP/LLM Pipeline Runner v2")
    parser.add_argument(
        "--from", dest="start_from", type=str, default=None,
        help="Start from a specific module (e.g., m3, m5)"
    )
    parser.add_argument(
        "--only", nargs="+", type=str, default=None,
        help="Run only specific modules (e.g., --only m0 m3)"
    )
    parser.add_argument(
        "--skip-env", action="store_true", default=False,
        help="Skip .env / API key check"
    )
    parser.add_argument(
        "--no-llm", action="store_true", default=False,
        help="Skip LLM API calls (use rule-based only)"
    )
    parser.add_argument(
        "--auto-confirm", action="store_true", default=False,
        help="Auto-confirm cost estimation prompts"
    )
    args = parser.parse_args()

    # Module definitions (order matters!)
    MODULES = {
        "m0": ("M0 Setup & Struktur Proyek", run_m0),
        "m1": ("M1 Koleksi Berita (Archive Scraping)", run_m1),
        "m2": ("M2 Preprocessing & Deduplication", run_m2),
        "m3": ("M3 Desain Prompt & Schema", run_m3),
        "m4": ("M4 Ekstraksi LLM (GPT-4o-mini)", run_m4),
        "m5": ("M5 Parsing & QC", run_m5),
        "m6": ("M6 Human Validation", run_m6),
        "m7": ("M7 Dekomposisi Tipe A/B", run_m7),
        "m8": ("M8 Agregasi Time-Series", run_m8),
        "m9": ("M9 Granger Causality Tests", run_m9),
        "m10": ("M10 Final Feature Export", run_m10),
    }

    # Determine which modules to run
    module_keys = list(MODULES.keys())

    if args.only:
        modules_to_run = [m.lower() for m in args.only if m.lower() in module_keys]
    elif args.start_from:
        start = args.start_from.lower()
        if start in module_keys:
            start_idx = module_keys.index(start)
            modules_to_run = module_keys[start_idx:]
        else:
            print(f"Unknown module: {start}. Available: {module_keys}")
            sys.exit(1)
    else:
        modules_to_run = module_keys

    print("=" * 60)
    print("NLP/LLM PIPELINE v2 — Inflasi Pangan Bergejolak Medan")
    print("(Archive Scraping Edition)")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Modules to run: {', '.join(modules_to_run)}")
    print("=" * 60)

    # Run modules
    for module_key in modules_to_run:
        name, func = MODULES[module_key]
        print(f"\n{'='*60}")
        print(f"[{module_key.upper()}] {name}")
        print(f"{'='*60}")

        start_time = time.time()
        try:
            func(args)
            elapsed = time.time() - start_time
            print(f"[{module_key.upper()}] Completed in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[{module_key.upper()}] FAILED after {elapsed:.1f}s: {e}")
            import traceback
            traceback.print_exc()

            if module_key in ["m4", "m7"]:  # LLM modules
                print("Note: LLM module failure. Check API key and network.")
            break

    print(f"\nPipeline finished at {datetime.now().isoformat()}")


# ── Module Runners ───────────────────────────────────────────

def run_m0(args):
    from m0_setup import initialize_pipeline
    cfg, logger = initialize_pipeline(skip_env_check=args.skip_env)
    logger.info("M0 complete.")


def run_m1(args):
    from m0_setup import get_cfg_and_logger
    from m1_collect import run_collection
    cfg, logger = get_cfg_and_logger(skip_env=args.skip_env)
    run_collection(cfg)


def run_m2(args):
    from m0_setup import get_cfg_and_logger
    from m2_preprocess import run_preprocessing
    cfg, logger = get_cfg_and_logger(skip_env=args.skip_env)
    run_preprocessing(cfg)


def run_m3(args):
    from m0_setup import get_cfg_and_logger
    from m3_prompts import save_prompts_to_files
    cfg, logger = get_cfg_and_logger(skip_env=args.skip_env)
    save_prompts_to_files()


def run_m4(args):
    if args.no_llm:
        print("Skipping M4 (--no-llm flag)")
        return
    from m0_setup import get_cfg_and_logger
    from m4_extract_llm import run_extraction_pipeline
    cfg, logger = get_cfg_and_logger()
    run_extraction_pipeline(cfg=cfg, auto_confirm=args.auto_confirm)


def run_m5(args):
    from m0_setup import get_cfg_and_logger
    from m5_parse_qc import run_parse_qc
    cfg, logger = get_cfg_and_logger(skip_env=args.skip_env)
    run_parse_qc(cfg)


def run_m6(args):
    from m0_setup import get_cfg_and_logger
    from m6_human_validation import run_human_validation
    cfg, logger = get_cfg_and_logger(skip_env=args.skip_env)
    run_human_validation(cfg)


def run_m7(args):
    from m0_setup import get_cfg_and_logger
    from m7_typeAB import run_typeAB_classification
    cfg, logger = get_cfg_and_logger(skip_env=args.skip_env)
    use_llm = not args.no_llm
    run_typeAB_classification(cfg, use_llm=use_llm)


def run_m8(args):
    from m0_setup import get_cfg_and_logger
    from m8_aggregate import run_aggregation
    cfg, logger = get_cfg_and_logger(skip_env=args.skip_env)
    run_aggregation(cfg)


def run_m9(args):
    from m0_setup import get_cfg_and_logger
    from m9_granger import run_granger_tests
    cfg, logger = get_cfg_and_logger(skip_env=args.skip_env)
    run_granger_tests(cfg)


def run_m10(args):
    from m0_setup import get_cfg_and_logger
    from m10_export import run_export
    cfg, logger = get_cfg_and_logger(skip_env=args.skip_env)
    run_export(cfg)


if __name__ == "__main__":
    main()
