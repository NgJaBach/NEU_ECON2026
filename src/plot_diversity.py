#!/usr/bin/env python3
"""
Box plot: Idea Diversity pairwise distances by condition and item.
Each box = distribution of all N(N-1)/2 pairwise cosine distances within a group.
Mean marker (triangle) shows the group diversity score.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics.pairwise import cosine_distances

# Import from sibling scripts
sys.path.insert(0, str(Path(__file__).parent))
from score_data import CONDITION_COL, IDEA_COLS, CONDITION_SHORT
from idea_diversity import get_embedder

# ── Style config ──────────────────────────────────────────────────────────────

CONDITION_COLORS = {
    "A_Question":   "#FF9E4A",   # orange
    "B_Suggestion": "#5DBF5F",   # green
    "C_Control":    "#5B9BD5",   # blue
}
CONDITION_LABELS = {
    "A_Question":   "A-Question",
    "B_Suggestion": "B-Suggestion",
    "C_Control":    "C-Control",
}
ITEM_ORDER  = ["Camera", "Sensor", "Đèn", "Loa"]
ITEM_XLABELS = ["Vòng 1\n(Camera)", "Vòng 2\n(Sensor)", "Vòng 3\n(Đèn)", "Vòng 4\n(Loa)"]
COND_ORDER  = ["A_Question", "B_Suggestion", "C_Control"]


def compute_pairwise_distances(ideas: list[str], embedder) -> np.ndarray:
    """Return all N(N-1)/2 normalised pairwise cosine distances for a group."""
    ideas = [str(s).strip() for s in ideas if s and str(s).strip()]
    if len(ideas) < 2:
        return np.array([])
    vecs = embedder.embed(ideas)
    dist_mat = cosine_distances(vecs)
    n = len(ideas)
    idx = np.triu_indices(n, k=1)
    pairwise = dist_mat[idx] / 2.0          # normalise to [0, 1]
    return pairwise


def build_pairwise_df(input_path: str, use_sbert: bool = True) -> pd.DataFrame:
    """Load data, embed, and collect all pairwise distances into a long DataFrame."""
    df_raw = pd.read_excel(input_path, engine="openpyxl")
    df_raw = df_raw[df_raw[CONDITION_COL].notna()].copy()

    embedder = get_embedder(prefer_sbert=use_sbert)

    rows = []
    for cond_full, cond_short in CONDITION_SHORT.items():
        subset = df_raw[df_raw[CONDITION_COL] == cond_full]
        for item_short, col in IDEA_COLS.items():
            ideas = subset[col].dropna().astype(str).str.strip().tolist()
            ideas = [i for i in ideas if i]
            dists = compute_pairwise_distances(ideas, embedder)
            for d in dists:
                rows.append({
                    "condition": cond_short,
                    "item":      item_short,
                    "distance":  d,
                    "n_ideas":   len(ideas),
                })
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, output_path: str) -> None:
    """Draw grouped box plot matching the workload chart style."""
    fig, ax = plt.subplots(figsize=(12, 7))

    n_items = len(ITEM_ORDER)
    n_conds = len(COND_ORDER)
    group_width = 0.7            # total width occupied by one item's group of boxes
    box_width   = group_width / n_conds * 0.85

    # Offsets so boxes are side-by-side within each item slot
    offsets = np.linspace(-group_width / 2 + box_width / 2,
                           group_width / 2 - box_width / 2,
                           n_conds)

    for ci, cond in enumerate(COND_ORDER):
        color = CONDITION_COLORS[cond]
        for ii, item in enumerate(ITEM_ORDER):
            x_center = ii + offsets[ci]
            data = df[(df["condition"] == cond) & (df["item"] == item)]["distance"].values

            if len(data) == 0:
                continue

            bp = ax.boxplot(
                data,
                positions=[x_center],
                widths=box_width,
                patch_artist=True,
                showfliers=False,
                showmeans=True,
                meanprops=dict(marker="^", markerfacecolor="black",
                               markeredgecolor="black", markersize=6),
                medianprops=dict(color="black", linewidth=1.5),
                boxprops=dict(facecolor=color, alpha=0.85, linewidth=0),
                whiskerprops=dict(color="#555555", linewidth=1.2),
                capprops=dict(color="#555555", linewidth=1.2),
            )

            # Mean label
            mean_val = float(np.mean(data))
            ax.text(
                x_center, mean_val + 0.007,
                f"M = {mean_val:.2f}",
                ha="center", va="bottom",
                fontsize=7.5, color="black",
            )

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xticks(range(n_items))
    ax.set_xticklabels(ITEM_XLABELS, fontsize=11)
    ax.set_xlabel("Vòng", fontsize=12)
    ax.set_ylabel("Độ Đa Dạng Từng Cặp Ý Tưởng", fontsize=12)
    ax.set_title("Độ Đa Dạng Ý Tưởng theo Điều Kiện và Từng Vòng", fontsize=14, pad=14)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # Legend
    patches = [
        mpatches.Patch(facecolor=CONDITION_COLORS[c], label=CONDITION_LABELS[c])
        for c in COND_ORDER
    ]
    ax.legend(handles=patches, title="Điều kiện", loc="upper right", framealpha=0.9)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",      default="data/Data.xlsx")
    p.add_argument("--output",     default="results/figures/diversity_boxplot.png")
    p.add_argument("--no-sbert",   action="store_true")
    args = p.parse_args()

    print("Computing pairwise distances…")
    df = build_pairwise_df(args.input, use_sbert=not args.no_sbert)
    print(f"Total pairwise observations: {len(df):,}")
    plot(df, args.output)


if __name__ == "__main__":
    main()
