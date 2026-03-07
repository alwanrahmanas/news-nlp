"""
=============================================================================
  EDA — Eksplorasi & Analisis Data Clean (corpus + extraction)
  NLP Pipeline v2 — Inflasi Pangan Bergejolak Medan / Sumatera Utara
=============================================================================

Pendekatan:
  1. Statistik Deskriptif Corpus
  2. Distribusi Temporal (Timeline)
  3. Analisis Sumber Berita
  4. Analisis Keyword Matching
  5. Distribusi Label (SUPPLYSHOCK, DEMANDSHOCK, PRICEREPORT, IRRELEVANT)
  6. Analisis Sentimen
  7. Analisis Komoditas
  8. Analisis Supply Location
  9. Analisis Forward-Looking vs Reactive
  10. Cross-tabulation: Label x Komoditas
  11. Temporal Trends per Label
  12. LLM Token Usage & Cost
  13. Confidence Score Distribution
  14. Word Cloud (Top words in full_text)
  15. Ringkasan Temuan

Jalankan:
    python eda_clean_data.py
"""

import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# ── Paths ─────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
CORPUS_PATH = BASE / "data" / "clean" / "corpus_clean.jsonl"
EXTRACTION_PATH = BASE / "data" / "clean" / "extraction_cache.jsonl"
OUT_DIR = BASE / "outputs" / "eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load Data ─────────────────────────────────────────────────
def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

print("Loading data...")
corpus = load_jsonl(CORPUS_PATH)
extractions = load_jsonl(EXTRACTION_PATH)
print(f"  corpus_clean   : {len(corpus):,} records")
print(f"  extraction_cache: {len(extractions):,} records")

# Build lookup
ext_by_id = {e["article_id"]: e for e in extractions}

# Merge
merged = []
for c in corpus:
    aid = c["article_id"]
    ext = ext_by_id.get(aid, {})
    row = {**c}
    if ext:
        row["label"] = ext.get("extraction", {}).get("label", "UNKNOWN")
        row["sentiment_score"] = ext.get("extraction", {}).get("sentiment_score", None)
        row["confidence"] = ext.get("extraction", {}).get("confidence", None)
        row["commodities"] = ext.get("extraction", {}).get("commodities", [])
        row["supply_location"] = ext.get("extraction", {}).get("supply_location", None)
        row["is_forward_looking"] = ext.get("extraction", {}).get("is_forward_looking", None)
        row["rationale"] = ext.get("extraction", {}).get("rationale", "")
        row["tokens_prompt"] = ext.get("tokens_prompt", 0)
        row["tokens_completion"] = ext.get("tokens_completion", 0)
        row["status"] = ext.get("status", "")
    else:
        row["label"] = "NO_EXTRACTION"
        row["sentiment_score"] = None
        row["confidence"] = None
        row["commodities"] = []
        row["supply_location"] = None
        row["is_forward_looking"] = None
        row["rationale"] = ""
        row["tokens_prompt"] = 0
        row["tokens_completion"] = 0
        row["status"] = ""
    merged.append(row)

print(f"  merged         : {len(merged):,} records\n")


# ── Helper ────────────────────────────────────────────────────
def save_fig(fig, name):
    path = OUT_DIR / f"{name}.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [saved] {path.name}")

# Colour palette
LABEL_COLORS = {
    "SUPPLYSHOCK": "#e74c3c",
    "DEMANDSHOCK": "#3498db",
    "PRICEREPORT": "#2ecc71",
    "IRRELEVANT": "#95a5a6",
    "NO_EXTRACTION": "#bdc3c7",
    "UNKNOWN": "#7f8c8d",
}

# ═══════════════════════════════════════════════════════════════
#  1. STATISTIK DESKRIPTIF CORPUS
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("1. STATISTIK DESKRIPTIF CORPUS")
print("=" * 60)
char_lengths = [r["char_length"] for r in merged]
print(f"   Jumlah artikel       : {len(merged):,}")
print(f"   Panjang teks (chars) :")
print(f"     Min   : {min(char_lengths):,}")
print(f"     Max   : {max(char_lengths):,}")
print(f"     Mean  : {np.mean(char_lengths):,.0f}")
print(f"     Median: {np.median(char_lengths):,.0f}")
print(f"     Std   : {np.std(char_lengths):,.0f}")

dates = []
for r in merged:
    try:
        dates.append(datetime.strptime(r["published_date"], "%Y-%m-%d"))
    except:
        pass
if dates:
    print(f"   Rentang tanggal      : {min(dates).date()} – {max(dates).date()}")
    span_days = (max(dates) - min(dates)).days
    print(f"   Span                 : {span_days} hari (~{span_days/30:.1f} bulan)")

# Histogram char_length
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(char_lengths, bins=50, color="#3498db", edgecolor="white", alpha=0.85)
ax.set_xlabel("Panjang Teks (karakter)")
ax.set_ylabel("Jumlah Artikel")
ax.set_title("Distribusi Panjang Artikel")
ax.axvline(np.median(char_lengths), color="red", linestyle="--", label=f"Median={np.median(char_lengths):,.0f}")
ax.legend()
save_fig(fig, "01_char_length_distribution")


# ═══════════════════════════════════════════════════════════════
#  2. DISTRIBUSI TEMPORAL
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. DISTRIBUSI TEMPORAL")
print("=" * 60)

month_counter = Counter()
for d in dates:
    month_counter[d.strftime("%Y-%m")] += 1

months_sorted = sorted(month_counter.keys())
counts_sorted = [month_counter[m] for m in months_sorted]

print(f"   Bulan dengan artikel terbanyak: {max(month_counter, key=month_counter.get)} ({max(month_counter.values())} artikel)")
print(f"   Bulan dengan artikel tersedikit: {min(month_counter, key=month_counter.get)} ({min(month_counter.values())} artikel)")

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(range(len(months_sorted)), counts_sorted, color="#2ecc71", edgecolor="white")
ax.set_xticks(range(len(months_sorted)))
ax.set_xticklabels(months_sorted, rotation=45, ha="right", fontsize=7)
ax.set_xlabel("Bulan")
ax.set_ylabel("Jumlah Artikel")
ax.set_title("Distribusi Artikel per Bulan")
save_fig(fig, "02_temporal_distribution")


# ═══════════════════════════════════════════════════════════════
#  2.5 TEMPORAL COVERAGE ANALYSIS — Effective vs Nominal Coverage
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2.5 TEMPORAL COVERAGE ANALYSIS")
print("=" * 60)

# Hitung n_total dan n_relevant (non-IRRELEVANT) per bulan
RELEVANT_LABELS = {"SUPPLYSHOCK", "DEMANDSHOCK", "PRICEREPORT"}
MIN_RELEVANT_PER_MONTH = 2   # threshold: bulan dianggap "terwakili"

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

# Buat tabel per bulan (lengkap — termasuk bulan dengan 0 artikel)
all_months_full = months_sorted  # sudah sorted dari section 2
coverage_rows = []
for m in all_months_full:
    n_tot = month_total.get(m, 0)
    n_rel = month_relevant.get(m, 0)
    n_irr = month_irrelevant.get(m, 0)
    is_low_conf = n_rel < MIN_RELEVANT_PER_MONTH
    pct_rel = (n_rel / n_tot * 100) if n_tot > 0 else 0.0
    coverage_rows.append({
        "month": m,
        "n_total": n_tot,
        "n_relevant": n_rel,
        "n_irrelevant": n_irr,
        "pct_relevant": pct_rel,
        "low_confidence": is_low_conf,
    })

n_low_conf_months = sum(1 for r in coverage_rows if r["low_confidence"])
n_ok_months = len(coverage_rows) - n_low_conf_months

print(f"\n   Total bulan dalam corpus    : {len(coverage_rows)}")
print(f"   Bulan OK  (n_relevant ≥{MIN_RELEVANT_PER_MONTH})  : {n_ok_months}")
print(f"   Bulan LOW (n_relevant < {MIN_RELEVANT_PER_MONTH})  : {n_low_conf_months} ← perlu perhatian")
print(f"\n   Detail bulan LOW CONFIDENCE:")
print(f"   {'Bulan':8s}  {'N Total':>8s}  {'N Relev':>8s}  {'N Irrel':>8s}  {'% Relev':>8s}  Status")
print("   " + "-" * 60)
for row in coverage_rows:
    if row["low_confidence"]:
        status = "⚠ KOSONG SINYAL" if row["n_relevant"] == 0 else "⚠ TIPIS"
        print(f"   {row['month']:8s}  {row['n_total']:>8d}  {row['n_relevant']:>8d}  "
              f"{row['n_irrelevant']:>8d}  {row['pct_relevant']:>7.1f}%  {status}")

# ── Stacked bar chart: Relevant vs Irrelevant per bulan ──────
fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

x = range(len(all_months_full))
n_tot_arr  = [month_total.get(m, 0)     for m in all_months_full]
n_rel_arr  = [month_relevant.get(m, 0)  for m in all_months_full]
n_irr_arr  = [month_irrelevant.get(m, 0) for m in all_months_full]
low_mask   = [month_relevant.get(m, 0) < MIN_RELEVANT_PER_MONTH for m in all_months_full]

# Panel atas: Stacked bar RELEVANT vs IRRELEVANT
axes[0].bar(x, n_rel_arr, label="Relevan (SUPPLY/DEMAND/PRICE)", color="#27ae60", alpha=0.9)
axes[0].bar(x, n_irr_arr, bottom=n_rel_arr, label="IRRELEVANT", color="#95a5a6", alpha=0.7)

# Highlight bulan low-confidence dengan latar merah terang
for i, (lc, nt) in enumerate(zip(low_mask, n_tot_arr)):
    if lc:
        axes[0].axvspan(i - 0.5, i + 0.5, alpha=0.15, color="red", zorder=0)

axes[0].axhline(MIN_RELEVANT_PER_MONTH, color="red", linestyle="--", linewidth=1.2,
                label=f"Min. threshold = {MIN_RELEVANT_PER_MONTH} artikel relevan")
axes[0].set_ylabel("Jumlah Artikel")
axes[0].set_title("Effective Coverage per Bulan (merah = low-confidence period)")
axes[0].legend(fontsize=8)
axes[0].grid(axis="y", alpha=0.3)

# Panel bawah: % relevan per bulan
pct_rel_arr = [(n_rel_arr[i] / n_tot_arr[i] * 100) if n_tot_arr[i] > 0 else 0.0
               for i in range(len(all_months_full))]
bar_colors = ["#e74c3c" if lc else "#2ecc71" for lc in low_mask]
axes[1].bar(x, pct_rel_arr, color=bar_colors, alpha=0.85)
axes[1].axhline(50, color="grey", linestyle=":", linewidth=1, label="50%")
axes[1].set_ylabel("% Artikel Relevan")
axes[1].set_ylim(0, 110)
axes[1].set_xlabel("Bulan")
axes[1].set_title("Proporsi Artikel Relevan per Bulan")
axes[1].legend(fontsize=8)
axes[1].grid(axis="y", alpha=0.3)

# X-ticks: tampilkan setiap 3 bulan agar tidak crowded
step = max(1, len(all_months_full) // 20)
axes[1].set_xticks(range(0, len(all_months_full), step))
axes[1].set_xticklabels(
    [all_months_full[i] for i in range(0, len(all_months_full), step)],
    rotation=45, ha="right", fontsize=7
)
plt.tight_layout()
save_fig(fig, "02b_effective_coverage")

# Simpan daftar bulan bermasalah
low_conf_months_list = [r["month"] for r in coverage_rows if r["low_confidence"]]
print(f"\n   Daftar bulan low-confidence (total {len(low_conf_months_list)} bulan):")
print("   " + ", ".join(low_conf_months_list))


# ═══════════════════════════════════════════════════════════════
#  3. ANALISIS SUMBER BERITA
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. ANALISIS SUMBER BERITA")
print("=" * 60)

source_counter = Counter()
source_name_counter = Counter()
for r in merged:
    source_counter[r.get("source", "unknown")] += 1
    source_name_counter[r.get("source_name", "unknown")] += 1

print("   Berdasarkan domain (source):")
for src, cnt in source_counter.most_common():
    print(f"     {src:30s} : {cnt:4d} ({cnt/len(merged)*100:.1f}%)")

print("\n   Berdasarkan source_name:")
for src, cnt in source_name_counter.most_common():
    print(f"     {src:30s} : {cnt:4d} ({cnt/len(merged)*100:.1f}%)")

# Pie chart
fig, ax = plt.subplots(figsize=(8, 8))
labels_src = [f"{k}\n({v})" for k, v in source_name_counter.most_common()]
sizes = [v for _, v in source_name_counter.most_common()]
colors_pie = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
wedges, texts, autotexts = ax.pie(sizes, labels=labels_src, autopct='%1.1f%%',
                                   colors=colors_pie, startangle=140, pctdistance=0.85)
for t in texts:
    t.set_fontsize(8)
for t in autotexts:
    t.set_fontsize(7)
ax.set_title("Proporsi Sumber Berita")
save_fig(fig, "03_source_distribution")


# ═══════════════════════════════════════════════════════════════
#  4. ANALISIS KEYWORD MATCHING
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. ANALISIS KEYWORD MATCHING")
print("=" * 60)

keyword_counter = Counter()
source_kw_counter = Counter()
for r in merged:
    for kw in r.get("keyword_matched", []):
        keyword_counter[kw] += 1
    source_kw_counter[r.get("source_keyword", "unknown")] += 1

print("   Top matched keywords:")
for kw, cnt in keyword_counter.most_common(15):
    print(f"     {kw:30s} : {cnt:4d}")

print("\n   Source keywords (query used):")
for kw, cnt in source_kw_counter.most_common():
    print(f"     {kw:30s} : {cnt:4d}")

# Bar chart keyword
fig, ax = plt.subplots(figsize=(10, 5))
top_kw = keyword_counter.most_common(15)
ax.barh([k for k, _ in reversed(top_kw)], [v for _, v in reversed(top_kw)], color="#e67e22")
ax.set_xlabel("Jumlah Artikel")
ax.set_title("Top 15 Keyword yang Matched")
save_fig(fig, "04_keyword_distribution")


# ═══════════════════════════════════════════════════════════════
#  5. DISTRIBUSI LABEL
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. DISTRIBUSI LABEL KLASIFIKASI")
print("=" * 60)

label_counter = Counter()
for r in merged:
    label_counter[r["label"]] += 1

for lbl, cnt in label_counter.most_common():
    print(f"   {lbl:20s} : {cnt:4d} ({cnt/len(merged)*100:.1f}%)")

fig, ax = plt.subplots(figsize=(8, 5))
labels_lbl = [l for l, _ in label_counter.most_common()]
values_lbl = [v for _, v in label_counter.most_common()]
colors_lbl = [LABEL_COLORS.get(l, "#999999") for l in labels_lbl]
bars = ax.bar(labels_lbl, values_lbl, color=colors_lbl, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, values_lbl):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            str(val), ha="center", va="bottom", fontweight="bold")
ax.set_ylabel("Jumlah Artikel")
ax.set_title("Distribusi Label Klasifikasi LLM")
save_fig(fig, "05_label_distribution")


# ═══════════════════════════════════════════════════════════════
#  6. ANALISIS SENTIMEN
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. ANALISIS SENTIMEN")
print("=" * 60)

sentiments = [r["sentiment_score"] for r in merged if r["sentiment_score"] is not None]
print(f"   Jumlah dengan skor sentimen: {len(sentiments)}")
print(f"   Min  : {min(sentiments):.2f}")
print(f"   Max  : {max(sentiments):.2f}")
print(f"   Mean : {np.mean(sentiments):.3f}")
print(f"   Median: {np.median(sentiments):.3f}")

# Sentimen per label
print("\n   Sentimen rata-rata per label:")
for lbl in ["SUPPLYSHOCK", "DEMANDSHOCK", "PRICEREPORT", "IRRELEVANT"]:
    scores = [r["sentiment_score"] for r in merged if r["label"] == lbl and r["sentiment_score"] is not None]
    if scores:
        print(f"     {lbl:15s} : mean={np.mean(scores):.3f}, median={np.median(scores):.3f}, n={len(scores)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Histogram overall
axes[0].hist(sentiments, bins=30, color="#9b59b6", edgecolor="white", alpha=0.85)
axes[0].set_xlabel("Sentiment Score")
axes[0].set_ylabel("Jumlah Artikel")
axes[0].set_title("Distribusi Skor Sentimen (Semua)")
axes[0].axvline(0, color="grey", linestyle="--", alpha=0.5)

# Box plot per label
label_order = ["SUPPLYSHOCK", "DEMANDSHOCK", "PRICEREPORT", "IRRELEVANT"]
data_box = []
labels_box = []
for lbl in label_order:
    scores = [r["sentiment_score"] for r in merged if r["label"] == lbl and r["sentiment_score"] is not None]
    if scores:
        data_box.append(scores)
        labels_box.append(lbl)

bp = axes[1].boxplot(data_box, labels=labels_box, patch_artist=True)
for patch, lbl in zip(bp["boxes"], labels_box):
    patch.set_facecolor(LABEL_COLORS.get(lbl, "#999"))
    patch.set_alpha(0.7)
axes[1].set_ylabel("Sentiment Score")
axes[1].set_title("Sentimen per Label")
axes[1].tick_params(axis="x", rotation=15)
save_fig(fig, "06_sentiment_analysis")


# ═══════════════════════════════════════════════════════════════
#  7. ANALISIS KOMODITAS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("7. ANALISIS KOMODITAS")
print("=" * 60)

commodity_counter = Counter()
for r in merged:
    for c in r["commodities"]:
        commodity_counter[c.lower().strip()] += 1

print("   Top 20 komoditas yang disebut:")
for com, cnt in commodity_counter.most_common(20):
    print(f"     {com:30s} : {cnt:4d}")

# Horizontal bar
fig, ax = plt.subplots(figsize=(10, 7))
top_com = commodity_counter.most_common(20)
ax.barh([c for c, _ in reversed(top_com)], [v for _, v in reversed(top_com)], color="#1abc9c")
ax.set_xlabel("Jumlah Kemunculan")
ax.set_title("Top 20 Komoditas yang Disebut dalam Ekstraksi")
save_fig(fig, "07_commodity_distribution")

# Komoditas per label
print("\n   Komoditas per label (top 5 each):")
for lbl in ["SUPPLYSHOCK", "DEMANDSHOCK", "PRICEREPORT"]:
    cc = Counter()
    for r in merged:
        if r["label"] == lbl:
            for c in r["commodities"]:
                cc[c.lower().strip()] += 1
    top5 = cc.most_common(5)
    print(f"   {lbl}:")
    for c, v in top5:
        print(f"     {c:25s} : {v}")


# ═══════════════════════════════════════════════════════════════
#  8. ANALISIS SUPPLY LOCATION
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("8. ANALISIS SUPPLY LOCATION")
print("=" * 60)

loc_counter = Counter()
for r in merged:
    loc = r["supply_location"]
    if loc:
        loc_counter[loc] += 1
    else:
        loc_counter["(tidak terdeteksi)"] += 1

print("   Top supply locations:")
for loc, cnt in loc_counter.most_common(15):
    print(f"     {loc:40s} : {cnt:4d}")

fig, ax = plt.subplots(figsize=(10, 6))
top_loc = loc_counter.most_common(15)
ax.barh([l for l, _ in reversed(top_loc)], [v for _, v in reversed(top_loc)], color="#e74c3c", alpha=0.8)
ax.set_xlabel("Jumlah Artikel")
ax.set_title("Top 15 Supply Locations")
save_fig(fig, "08_supply_location")


# ═══════════════════════════════════════════════════════════════
#  9. FORWARD-LOOKING vs REACTIVE
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("9. FORWARD-LOOKING vs REACTIVE ARTICLES")
print("=" * 60)

fl_counter = Counter()
for r in merged:
    fl = r["is_forward_looking"]
    if fl is True:
        fl_counter["Forward-Looking"] += 1
    elif fl is False:
        fl_counter["Reactive/Historical"] += 1
    else:
        fl_counter["Unknown"] += 1

for k, v in fl_counter.most_common():
    print(f"   {k:25s} : {v:4d} ({v/len(merged)*100:.1f}%)")

# Forward-looking per label
print("\n   Forward-looking rate per label:")
for lbl in ["SUPPLYSHOCK", "DEMANDSHOCK", "PRICEREPORT", "IRRELEVANT"]:
    total = sum(1 for r in merged if r["label"] == lbl)
    fl = sum(1 for r in merged if r["label"] == lbl and r["is_forward_looking"] is True)
    if total > 0:
        print(f"     {lbl:15s} : {fl}/{total} = {fl/total*100:.1f}%")


# ═══════════════════════════════════════════════════════════════
#  10. CROSS-TABULATION: LABEL x TOP KOMODITAS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("10. CROSS-TABULATION: LABEL x KOMODITAS")
print("=" * 60)

top_commodities = [c for c, _ in commodity_counter.most_common(10)]
cross_tab = {}
for lbl in ["SUPPLYSHOCK", "DEMANDSHOCK", "PRICEREPORT", "IRRELEVANT"]:
    cross_tab[lbl] = Counter()
    for r in merged:
        if r["label"] == lbl:
            for c in r["commodities"]:
                cl = c.lower().strip()
                if cl in top_commodities:
                    cross_tab[lbl][cl] += 1

# Print table
header = f"{'Label':15s} | " + " | ".join(f"{c[:12]:>12s}" for c in top_commodities)
print("   " + header)
print("   " + "-" * len(header))
for lbl in ["SUPPLYSHOCK", "DEMANDSHOCK", "PRICEREPORT", "IRRELEVANT"]:
    vals = " | ".join(f"{cross_tab[lbl].get(c, 0):>12d}" for c in top_commodities)
    print(f"   {lbl:15s} | {vals}")

# Stacked bar chart
fig, ax = plt.subplots(figsize=(12, 6))
x_idx = np.arange(len(top_commodities))
width = 0.2
for i, lbl in enumerate(["SUPPLYSHOCK", "DEMANDSHOCK", "PRICEREPORT"]):
    vals = [cross_tab[lbl].get(c, 0) for c in top_commodities]
    ax.bar(x_idx + i * width, vals, width, label=lbl, color=LABEL_COLORS[lbl], alpha=0.85)
ax.set_xticks(x_idx + width)
ax.set_xticklabels(top_commodities, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Jumlah")
ax.set_title("Komoditas per Label (Top 10)")
ax.legend()
save_fig(fig, "10_label_x_commodity")


# ═══════════════════════════════════════════════════════════════
#  11. TEMPORAL TRENDS PER LABEL
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("11. TEMPORAL TRENDS PER LABEL")
print("=" * 60)

label_monthly = {}
for lbl in ["SUPPLYSHOCK", "DEMANDSHOCK", "PRICEREPORT", "IRRELEVANT"]:
    mc = Counter()
    for r in merged:
        if r["label"] == lbl:
            try:
                m = datetime.strptime(r["published_date"], "%Y-%m-%d").strftime("%Y-%m")
                mc[m] += 1
            except:
                pass
    label_monthly[lbl] = mc

fig, ax = plt.subplots(figsize=(14, 5))
for lbl in ["SUPPLYSHOCK", "DEMANDSHOCK", "PRICEREPORT"]:
    mc = label_monthly[lbl]
    x_months = sorted(mc.keys())
    y_vals = [mc[m] for m in x_months]
    ax.plot(x_months, y_vals, marker="o", label=lbl, color=LABEL_COLORS[lbl],
            markersize=4, linewidth=1.5, alpha=0.8)
ax.set_xticks(range(0, len(months_sorted), max(1, len(months_sorted)//15)))
ax.set_xticklabels([months_sorted[i] for i in range(0, len(months_sorted), max(1, len(months_sorted)//15))],
                   rotation=45, ha="right", fontsize=7)
ax.set_ylabel("Jumlah Artikel")
ax.set_title("Tren Bulanan per Label")
ax.legend()
ax.grid(alpha=0.3)
save_fig(fig, "11_temporal_trends_per_label")


# ═══════════════════════════════════════════════════════════════
#  12. LLM TOKEN USAGE & COST
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("12. LLM TOKEN USAGE")
print("=" * 60)

prompt_tokens = [r["tokens_prompt"] for r in merged if r["tokens_prompt"] > 0]
completion_tokens = [r["tokens_completion"] for r in merged if r["tokens_completion"] > 0]
total_prompt = sum(prompt_tokens)
total_completion = sum(completion_tokens)
total_tokens = total_prompt + total_completion

# gpt-4o-mini pricing: $0.15/1M input, $0.60/1M output
cost_input = total_prompt / 1_000_000 * 0.15
cost_output = total_completion / 1_000_000 * 0.60
total_cost = cost_input + cost_output

print(f"   Total prompt tokens    : {total_prompt:,}")
print(f"   Total completion tokens: {total_completion:,}")
print(f"   Total tokens           : {total_tokens:,}")
print(f"   Avg prompt tokens      : {np.mean(prompt_tokens):.0f}")
print(f"   Avg completion tokens  : {np.mean(completion_tokens):.0f}")
print(f"   Estimated cost (gpt-4o-mini):")
print(f"     Input  : ${cost_input:.4f}")
print(f"     Output : ${cost_output:.4f}")
print(f"     Total  : ${total_cost:.4f}")


# ═══════════════════════════════════════════════════════════════
#  13. CONFIDENCE SCORE DISTRIBUTION
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("13. CONFIDENCE SCORE DISTRIBUTION")
print("=" * 60)

confidences = [r["confidence"] for r in merged if r["confidence"] is not None]
print(f"   Min  : {min(confidences):.2f}")
print(f"   Max  : {max(confidences):.2f}")
print(f"   Mean : {np.mean(confidences):.3f}")
print(f"   Median: {np.median(confidences):.3f}")

# Confidence per label
print("\n   Confidence rata-rata per label:")
for lbl in ["SUPPLYSHOCK", "DEMANDSHOCK", "PRICEREPORT", "IRRELEVANT"]:
    confs = [r["confidence"] for r in merged if r["label"] == lbl and r["confidence"] is not None]
    if confs:
        print(f"     {lbl:15s} : mean={np.mean(confs):.3f}, n={len(confs)}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(confidences, bins=20, color="#f39c12", edgecolor="white", alpha=0.85)
ax.set_xlabel("Confidence Score")
ax.set_ylabel("Jumlah Artikel")
ax.set_title("Distribusi Confidence Score LLM")
ax.axvline(np.mean(confidences), color="red", linestyle="--", label=f"Mean={np.mean(confidences):.2f}")
ax.legend()
save_fig(fig, "13_confidence_distribution")


# ═══════════════════════════════════════════════════════════════
#  14. TOP WORDS / TERMS IN TEXT
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("14. TOP TERMS IN ARTICLES")
print("=" * 60)

# Simple word frequency (Indonesian stopwords basic filter)
stopwords_id = set("""
yang di dan ke dari ini itu dengan untuk pada tidak ada adalah juga
akan sudah kami kami kita saya mereka dia ia oleh dalam atas telah
bisa karena setelah menjadi secara atau sebagai namun serta hingga
juga masih bahwa dapat lebih kata ujar per lalu pun maka agar demi
saat belum maupun terhadap sehingga begitu antara tersebut melalui
seperti selain berdasarkan menurut kata ujar baca juga sebelumnya
pewarta editor copyright antara dilarang
""".split())

word_counter = Counter()
for r in merged:
    text = r.get("full_text", "").lower()
    words = text.split()
    for w in words:
        # basic cleaning
        w = w.strip(".,!?\"':;()[]{}…—–-/\\")
        if len(w) > 3 and w not in stopwords_id and not w.isdigit():
            word_counter[w] += 1

print("   Top 30 terms (excluding stopwords):")
for w, cnt in word_counter.most_common(30):
    print(f"     {w:25s} : {cnt:5d}")

fig, ax = plt.subplots(figsize=(10, 8))
top_words = word_counter.most_common(30)
ax.barh([w for w, _ in reversed(top_words)], [v for _, v in reversed(top_words)], color="#8e44ad", alpha=0.8)
ax.set_xlabel("Frekuensi")
ax.set_title("Top 30 Terms dalam Artikel")
save_fig(fig, "14_top_terms")


# ═══════════════════════════════════════════════════════════════
#  15. RINGKASAN TEMUAN
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("15. RINGKASAN TEMUAN")
print("=" * 60)

relevant = sum(1 for r in merged if r["label"] not in ["IRRELEVANT", "NO_EXTRACTION", "UNKNOWN"])
irrelevant = sum(1 for r in merged if r["label"] == "IRRELEVANT")
supply_shock = label_counter.get("SUPPLYSHOCK", 0)
demand_shock = label_counter.get("DEMANDSHOCK", 0)
price_report = label_counter.get("PRICEREPORT", 0)

print(f"""
   CORPUS OVERVIEW:
   • {len(merged)} artikel dari {len(source_name_counter)} sumber media
   • Periode: {min(dates).date()} s/d {max(dates).date()} ({span_days} hari)
   • Panjang rata-rata: {np.mean(char_lengths):,.0f} karakter

   KLASIFIKASI:
   • Relevan       : {relevant} ({relevant/len(merged)*100:.1f}%)
   • Irrelevant    : {irrelevant} ({irrelevant/len(merged)*100:.1f}%)
   • SUPPLYSHOCK   : {supply_shock} ({supply_shock/len(merged)*100:.1f}%)
   • DEMANDSHOCK   : {demand_shock} ({demand_shock/len(merged)*100:.1f}%)
   • PRICEREPORT   : {price_report} ({price_report/len(merged)*100:.1f}%)

   KOMODITAS UTAMA:
   • Top 5: {', '.join(c for c, _ in commodity_counter.most_common(5))}

   SENTIMEN:
   • Rata-rata skor sentimen: {np.mean(sentiments):.3f}
   • Mayoritas bernuansa {'positif' if np.mean(sentiments) > 0 else 'negatif'}

   FORWARD-LOOKING:
   • {fl_counter.get('Forward-Looking', 0)} artikel ({fl_counter.get('Forward-Looking', 0)/len(merged)*100:.1f}%) bersifat prediktif

   LLM USAGE:
   • Total tokens: {total_tokens:,}
   • Estimated cost: ${total_cost:.4f}
""")

# Save summary to file
summary_path = OUT_DIR / "eda_summary.txt"
with open(str(summary_path), "w", encoding="utf-8") as f:
    f.write(f"EDA Summary — NLP Pipeline v2\n")
    f.write(f"Generated: {datetime.now().isoformat()}\n\n")
    f.write(f"Total Articles: {len(merged)}\n")
    f.write(f"Date Range: {min(dates).date()} to {max(dates).date()}\n")
    f.write(f"Sources: {len(source_name_counter)}\n\n")
    f.write(f"Label Distribution:\n")
    for lbl, cnt in label_counter.most_common():
        f.write(f"  {lbl}: {cnt} ({cnt/len(merged)*100:.1f}%)\n")
    f.write(f"\nTop 10 Commodities:\n")
    for c, cnt in commodity_counter.most_common(10):
        f.write(f"  {c}: {cnt}\n")
    f.write(f"\nSentiment: mean={np.mean(sentiments):.3f}, median={np.median(sentiments):.3f}\n")
    f.write(f"Confidence: mean={np.mean(confidences):.3f}\n")
    f.write(f"Forward-Looking: {fl_counter.get('Forward-Looking', 0)}/{len(merged)}\n")
    f.write(f"Total LLM tokens: {total_tokens:,} (est. ${total_cost:.4f})\n")

print(f"\n   Summary saved to: {summary_path}")
print(f"   All charts saved to: {OUT_DIR}")
print("\n[DONE] EDA selesai!")
