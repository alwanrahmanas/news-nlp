# Quick test: hanya load data + section 2.5 coverage analysis
import json, sys, os
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

BASE = Path(__file__).resolve().parent
CORPUS_PATH = BASE / "data" / "clean" / "corpus_clean.jsonl"
EXTRACTION_PATH = BASE / "data" / "clean" / "extraction_cache.jsonl"

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

print("Loading...")
corpus = load_jsonl(CORPUS_PATH)
extractions = load_jsonl(EXTRACTION_PATH)
print(f"  corpus: {len(corpus)}, extractions: {len(extractions)}")

ext_by_id = {e["article_id"]: e for e in extractions}
merged = []
for c in corpus:
    aid = c["article_id"]
    ext = ext_by_id.get(aid, {})
    row = {**c}
    row["label"] = ext.get("extraction", {}).get("label", "UNKNOWN") if ext else "NO_EXTRACTION"
    merged.append(row)

print(f"  merged: {len(merged)}")

RELEVANT_LABELS = {"SUPPLYSHOCK", "DEMANDSHOCK", "PRICEREPORT"}
MIN_RELEVANT = 2

month_total = Counter()
month_relevant = Counter()
month_irrelevant = Counter()

for r in merged:
    try:
        m = datetime.strptime(r["published_date"], "%Y-%m-%d").strftime("%Y-%m")
    except Exception:
        continue
    month_total[m] += 1
    if r["label"] in RELEVANT_LABELS:
        month_relevant[m] += 1
    elif r["label"] == "IRRELEVANT":
        month_irrelevant[m] += 1

months_sorted = sorted(month_total.keys())

print(f"\nTotal bulan: {len(months_sorted)}")
print(f"{'Bulan':8s}  {'Total':>6s}  {'Relev':>6s}  {'Irrel':>6s}  {'%Rel':>5s}  Flag")
print("-" * 55)
for m in months_sorted:
    n_tot = month_total.get(m, 0)
    n_rel = month_relevant.get(m, 0)
    n_irr = month_irrelevant.get(m, 0)
    pct = (n_rel / n_tot * 100) if n_tot > 0 else 0.0
    flag = "LOW" if n_rel < MIN_RELEVANT else ""
    print(f"{m:8s}  {n_tot:>6d}  {n_rel:>6d}  {n_irr:>6d}  {pct:>4.0f}%  {flag}")

n_low = sum(1 for m in months_sorted if month_relevant.get(m, 0) < MIN_RELEVANT)
print(f"\nBulan low-confidence: {n_low}/{len(months_sorted)}")
print("Done!")
