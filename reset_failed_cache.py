"""
Hapus semua record 'failed' dari extraction_cache.jsonl
agar bisa di-retry pada run berikutnya.
"""
import json
from pathlib import Path

CACHE = Path(__file__).parent / "data" / "clean" / "extraction_cache.jsonl"

records = []
if CACHE.exists():
    with open(CACHE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                records.append(r)

before = len(records)
ok     = [r for r in records if r.get("status") == "success"]
failed = [r for r in records if r.get("status") != "success"]

print(f"Total cached : {before}")
print(f"  success    : {len(ok)}")
print(f"  failed/etc : {len(failed)}  <-- akan dihapus")

with open(CACHE, "w", encoding="utf-8") as f:
    for r in ok:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Cache setelah reset: {len(ok)} records (failed dihapus)")
