"""
score_data.py — Idea Diversity for Data.xlsx (3 conditions × 4 items = 12 groups)

Data format:
  - Wide: one row per respondent, 4 idea columns, 1 condition column
  - Groups: "Condition × Item" (e.g. "Nhóm A × Camera")

Usage:
  python src/score_data.py
  python src/score_data.py --input data/Data.xlsx --no-sbert
"""

import sys, io, argparse, json
from pathlib import Path
from datetime import datetime

# Force UTF-8 stdout (Vietnamese text)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
from idea_diversity import get_embedder, average_pairwise_cosine_distance, TFIDFEmbedder

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Column config ────────────────────────────────────────────────────────────

CONDITION_COL = "Bạn được chia vào nhóm điều kiện nào"

IDEA_COLS = {
    "Camera":  'Hãy đề xuất một cách sử dụng mới, sáng tạo và khác biệt cho "Camera "',
    "Sensor":  'Hãy đề xuất một cách sử dụng mới, sáng tạo và khác biệt cho  "Sensor (cảm biến, bộ phận dùng để xác định chuyển động, nhiệt độ, vật cản,...)"',
    "Đèn":     "Hãy đề xuất một cách sử dụng mới, sáng tạo và khác biệt cho Đèn (Lights)",
    "Loa":     'Hãy đề xuất một cách sử dụng mới, sáng tạo và khác biệt cho "Loa (Speakers)"',
}

CONDITION_SHORT = {
    "Nhóm A - Question mode":   "A_Question",
    "Nhóm B - Suggestion mode": "B_Suggestion",
    "Nhóm C - Control":         "C_Control",
}

# ── Main ─────────────────────────────────────────────────────────────────────

def run(input_path: str, use_sbert: bool = True, output_dir: str = "results"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(input_path, engine="openpyxl")
    log.info(f"Loaded {len(df)} rows from {Path(input_path).name}")

    # Drop rows with no condition
    df = df[df[CONDITION_COL].notna()].copy()
    log.info(f"Rows with condition: {len(df)}")

    conditions = sorted(df[CONDITION_COL].unique())
    log.info(f"Conditions: {conditions}")

    # ── Build 12 groups ──────────────────────────────────────────────────────
    # groups dict: { "A_Question × Camera": [idea1, idea2, ...], ... }
    groups = {}
    for cond in conditions:
        cond_short = CONDITION_SHORT.get(cond, cond)
        subset = df[df[CONDITION_COL] == cond]
        for item_short, col in IDEA_COLS.items():
            label = f"{cond_short} × {item_short}"
            ideas = subset[col].dropna().astype(str).str.strip()
            ideas = [i for i in ideas if i]
            groups[label] = ideas

    log.info(f"Built {len(groups)} groups:")
    for name, ideas in groups.items():
        log.info(f"  {name}: {len(ideas)} ideas")

    # ── Embedder ─────────────────────────────────────────────────────────────
    embedder = get_embedder(prefer_sbert=use_sbert)

    # For TF-IDF: fit global vocabulary on ALL ideas
    if isinstance(embedder, TFIDFEmbedder):
        all_ideas = [idea for ideas in groups.values() for idea in ideas]
        embedder.fit(all_ideas)

    # ── Score each group ─────────────────────────────────────────────────────
    results = []
    for label, ideas in groups.items():
        log.info(f"  Scoring '{label}' ({len(ideas)} ideas)...")
        score, stats = average_pairwise_cosine_distance(ideas, embedder)
        results.append({
            "group":           label,
            "condition":       label.split(" × ")[0],
            "item":            label.split(" × ")[1],
            "n_ideas":         len(ideas),
            "diversity_score": score,
            "pairwise_min":    stats["min"],
            "pairwise_max":    stats["max"],
            "pairwise_std":    stats["std"],
            "n_pairs":         stats["n_pairs"],
            "method":          embedder.method_name,
        })
        log.info(f"    → {score:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "sbert" if "BERT" in embedder.method_name else "tfidf"
    stem = f"diversity_12groups_{tag}_{ts}"

    import csv
    csv_path = output_dir / f"{stem}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    json_path = output_dir / f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"run_at": ts, "method": embedder.method_name,
                   "input": input_path, "results": results},
                  f, ensure_ascii=False, indent=2)

    log.info(f"\nSaved: {csv_path}")

    # ── Print table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("IDEA DIVERSITY — 12 GROUPS (3 conditions × 4 items)")
    print("=" * 70)
    print(f"{'Group':<30} {'N':>4}  {'Diversity':>10}  {'Std':>7}")
    print("-" * 70)

    # Print grouped by condition
    for cond_short in ["A_Question", "B_Suggestion", "C_Control"]:
        cond_rows = [r for r in results if r["condition"] == cond_short]
        for r in cond_rows:
            print(f"  {r['group']:<28} {r['n_ideas']:>4}  {r['diversity_score']:>10.4f}  {r['pairwise_std']:>7.4f}")
        # Condition average
        avg = sum(r["diversity_score"] for r in cond_rows) / len(cond_rows)
        print(f"  {'→ avg ' + cond_short:<28} {'':>4}  {avg:>10.4f}")
        print()

    # Item averages
    print("Item averages across conditions:")
    for item in IDEA_COLS:
        item_rows = [r for r in results if r["item"] == item]
        avg = sum(r["diversity_score"] for r in item_rows) / len(item_rows)
        print(f"  {item:<10} {avg:.4f}")

    print("=" * 70)
    print(f"Method: {embedder.method_name}")
    print(f"Output: {csv_path}")
    print()

    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/Data.xlsx")
    p.add_argument("--no-sbert", action="store_true")
    p.add_argument("--output-dir", default="results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, use_sbert=not args.no_sbert, output_dir=args.output_dir)
