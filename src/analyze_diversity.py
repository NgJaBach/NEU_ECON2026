#!/usr/bin/env python3
"""
Statistical analysis of idea diversity scores (12 groups).
One-way ANOVA (condition, item), two-way additive ANOVA, Tukey HSD post-hoc.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import glob
import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import anova_lm
import statsmodels.formula.api as smf


def find_latest_csv(results_dir):
    pattern = os.path.join(results_dir, "diversity_12groups_sbert_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No diversity CSV found in {results_dir}")
    return files[-1]


def descriptive_stats(df, group_col, value_col="diversity_score"):
    return (
        df.groupby(group_col)[value_col]
        .agg(n="count", mean="mean", std="std", min="min", max="max")
        .round(6)
    )


def oneway_anova(df, group_col, value_col="diversity_score"):
    groups = [grp[value_col].values for _, grp in df.groupby(group_col)]
    f_stat, p_val = stats.f_oneway(*groups)
    kw_stat, kw_p = stats.kruskal(*groups)

    grand_mean = df[value_col].mean()
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = np.sum((df[value_col] - grand_mean) ** 2)
    eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0

    return {
        "f_statistic": round(float(f_stat), 4),
        "p_value": round(float(p_val), 4),
        "eta_squared": round(eta_sq, 4),
        "kruskal_H": round(float(kw_stat), 4),
        "kruskal_p": round(float(kw_p), 4),
    }


def tukey_result_to_list(tukey):
    rows = tukey.summary().data[1:]  # skip header row
    return [
        {
            "group1": str(r[0]),
            "group2": str(r[1]),
            "meandiff": round(float(r[2]), 6),
            "p_adj": round(float(r[3]), 4),
            "lower": round(float(r[4]), 6),
            "upper": round(float(r[5]), 6),
            "reject_H0": bool(r[6]),
        }
        for r in rows
    ]


def print_section(title):
    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(project_dir, "results")

    csv_path = find_latest_csv(results_dir)
    print(f"Source: {os.path.basename(csv_path)}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # ── 1. Descriptive stats ──────────────────────────────────────────────
    print_section("DESCRIPTIVE STATISTICS — BY CONDITION")
    desc_cond = descriptive_stats(df, "condition")
    print(desc_cond.to_string())

    print_section("DESCRIPTIVE STATISTICS — BY ITEM")
    desc_item = descriptive_stats(df, "item")
    print(desc_item.to_string())

    # ── 2. One-way ANOVA: condition ───────────────────────────────────────
    print_section("ONE-WAY ANOVA: CONDITION (A vs B vs C)")
    anova_cond = oneway_anova(df, "condition")
    for k, v in anova_cond.items():
        print(f"  {k:20s}: {v}")
    if anova_cond["p_value"] < 0.05:
        print("  → Significant (p < 0.05)")
    else:
        print("  → Not significant (p ≥ 0.05) — note: n=4 per group, low power")

    # ── 3. One-way ANOVA: item ────────────────────────────────────────────
    print_section("ONE-WAY ANOVA: ITEM (Camera / Sensor / Đèn / Loa)")
    anova_item = oneway_anova(df, "item")
    for k, v in anova_item.items():
        print(f"  {k:20s}: {v}")
    if anova_item["p_value"] < 0.05:
        print("  → Significant (p < 0.05)")
    else:
        print("  → Not significant (p ≥ 0.05) — note: n=3 per group, low power")

    # ── 4. Two-way ANOVA (additive, no interaction — df too small) ────────
    print_section("TWO-WAY ANOVA (additive): diversity ~ condition + item")
    # Map condition/item to simple tokens for formula
    df2 = df.copy()
    df2["cond"] = df2["condition"].str.replace(r"[^A-Za-z0-9]", "_", regex=True)
    df2["it"] = df2["item"].str.replace(r"[^A-Za-z0-9]", "_", regex=True)
    model = smf.ols("diversity_score ~ C(cond) + C(it)", data=df2).fit()
    anova2 = anova_lm(model, typ=2)
    print(anova2.round(4).to_string())
    print(f"\n  R² = {model.rsquared:.4f}   Adj-R² = {model.rsquared_adj:.4f}")

    # ── 5. Tukey HSD post-hoc: condition ─────────────────────────────────
    print_section("TUKEY HSD POST-HOC — CONDITION")
    tukey_cond = pairwise_tukeyhsd(df["diversity_score"], df["condition"])
    print(tukey_cond.summary())
    tukey_cond_list = tukey_result_to_list(tukey_cond)

    # ── 6. Tukey HSD post-hoc: item ───────────────────────────────────────
    print_section("TUKEY HSD POST-HOC — ITEM")
    tukey_item = pairwise_tukeyhsd(df["diversity_score"], df["item"])
    print(tukey_item.summary())
    tukey_item_list = tukey_result_to_list(tukey_item)

    # ── 7. Save results ───────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(results_dir, f"diversity_stats_{ts}.json")
    out_csv = os.path.join(results_dir, f"diversity_stats_{ts}.csv")

    result_obj = {
        "source_csv": os.path.basename(csv_path),
        "generated": ts,
        "note": "n=4 per condition, n=3 per item — limited statistical power for group-level tests",
        "descriptive_by_condition": desc_cond.reset_index().to_dict(orient="records"),
        "descriptive_by_item": desc_item.reset_index().to_dict(orient="records"),
        "anova_condition": anova_cond,
        "anova_item": anova_item,
        "twoway_anova": {
            "model": "diversity_score ~ C(condition) + C(item)",
            "r_squared": round(model.rsquared, 4),
            "adj_r_squared": round(model.rsquared_adj, 4),
            "anova_table": anova2.reset_index().rename(columns={"index": "term"}).round(4).to_dict(orient="records"),
        },
        "tukey_condition": tukey_cond_list,
        "tukey_item": tukey_item_list,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result_obj, f, ensure_ascii=False, indent=2)

    # Summary CSV (flat table for easy reading)
    summary_rows = []
    for rec in result_obj["descriptive_by_condition"]:
        summary_rows.append({"factor": "condition", **rec})
    for rec in result_obj["descriptive_by_item"]:
        summary_rows.append({"factor": "item", **rec})
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\nSaved: {os.path.basename(out_json)}")
    print(f"Saved: {os.path.basename(out_csv)}")


if __name__ == "__main__":
    main()
