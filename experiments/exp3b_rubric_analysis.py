"""
Analyze one or more FILLED-IN copies of the blind rubric produced by
experiment 3 (results/exp3_rubric_blind.csv).

Workflow:
  1. Run experiments/exp3_intervention_quality.py - it writes
     results/exp3_rubric_blind.csv (give copies of this to 1-5 raters,
     who do NOT see results/exp3_rubric_key.csv) and
     results/exp3_rubric_key.csv (keep this yourself).
  2. Each rater fills in the *_1_5 columns (1-5 scale) for message_a and
     message_b, not knowing which one is generic vs personalized, and
     saves their own copy, e.g. results/filled_rater1.csv.
  3. Run this script pointing at all filled copies plus the key:

     python experiments/exp3b_rubric_analysis.py \\
         --rubrics results/filled_rater1.csv results/filled_rater2.csv \\
         --key results/exp3_rubric_key.csv

This de-anonymizes each rater's A/B scores back into generic/personalized
using the key, reports per-rater and pooled paired comparisons, and (if 2+
raters are given) a simple inter-rater agreement check.
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

DIMENSIONS = ["relevance", "specificity", "appropriateness"]


def load_csv_as_dicts(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def deanonymize(rubric_rows, key_rows):
    """Return list of dicts with generic_<dim> / personalized_<dim> per id,
    using the key to know which of A/B was the personalized message."""
    key = {r["id"]: r["a_is_personalized"] in ("True", "true", "1") for r in key_rows}
    out = []
    for r in rubric_rows:
        a_is_personalized = key.get(r["id"])
        if a_is_personalized is None:
            continue  # row not in key, skip rather than crash
        row = {"id": r["id"]}
        for dim in DIMENSIONS:
            val_a = r.get(f"{dim}_a_1_5", "")
            val_b = r.get(f"{dim}_b_1_5", "")
            if val_a == "" or val_b == "":
                continue  # rater left this blank; skip this dimension for this row
            val_a, val_b = float(val_a), float(val_b)
            if a_is_personalized:
                row[f"personalized_{dim}"] = val_a
                row[f"generic_{dim}"] = val_b
            else:
                row[f"personalized_{dim}"] = val_b
                row[f"generic_{dim}"] = val_a
        out.append(row)
    return out


def analyze_rater(rows, rater_name):
    result = {"rater": rater_name, "n_rows_rated": len(rows)}
    for dim in DIMENSIONS:
        gen_key, per_key = f"generic_{dim}", f"personalized_{dim}"
        pairs = [(r[gen_key], r[per_key]) for r in rows if gen_key in r and per_key in r]
        if not pairs:
            result[dim] = {"n": 0, "note": "no rows rated on this dimension"}
            continue
        gen_vals = [p[0] for p in pairs]
        per_vals = [p[1] for p in pairs]
        t_stat, t_p = stats.ttest_rel(per_vals, gen_vals) if len(pairs) > 1 else (float("nan"),) * 2
        result[dim] = {
            "n": len(pairs),
            "mean_generic": float(np.mean(gen_vals)),
            "mean_personalized": float(np.mean(per_vals)),
            "paired_ttest_t": t_stat, "paired_ttest_p": t_p,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rubrics", nargs="+", required=True,
                         help="one or more filled-in copies of exp3_rubric_blind.csv, "
                              "one per rater")
    parser.add_argument("--key", default=os.path.join(RESULTS_DIR, "exp3_rubric_key.csv"))
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "exp3_rubric_analysis.json"))
    args = parser.parse_args()

    key_rows = load_csv_as_dicts(args.key)
    all_rater_results = []
    all_deanon_rows = []  # for the pooled analysis across raters

    for path in args.rubrics:
        rubric_rows = load_csv_as_dicts(path)
        deanon = deanonymize(rubric_rows, key_rows)
        all_deanon_rows.extend(deanon)
        rater_name = os.path.splitext(os.path.basename(path))[0]
        result = analyze_rater(deanon, rater_name)
        all_rater_results.append(result)
        print(f"\n=== {rater_name} (n={result['n_rows_rated']} rows) ===")
        for dim in DIMENSIONS:
            d = result[dim]
            if d.get("n", 0) == 0:
                print(f"  {dim}: no ratings")
                continue
            print(f"  {dim}: generic={d['mean_generic']:.2f}  "
                  f"personalized={d['mean_personalized']:.2f}  "
                  f"paired t-test p={d['paired_ttest_p']:.4f}  (n={d['n']})")

    pooled = analyze_rater(all_deanon_rows, "POOLED (all raters)")
    print(f"\n=== POOLED across {len(args.rubrics)} rater(s) ===")
    for dim in DIMENSIONS:
        d = pooled[dim]
        if d.get("n", 0) == 0:
            continue
        print(f"  {dim}: generic={d['mean_generic']:.2f}  "
              f"personalized={d['mean_personalized']:.2f}  "
              f"paired t-test p={d['paired_ttest_p']:.4f}  (n={d['n']})")

    agreement = None
    if len(args.rubrics) >= 2:
        # simple inter-rater agreement: correlate each rater's
        # (personalized - generic) delta, per dimension, across raters
        print("\n=== Inter-rater agreement (Pearson r of per-item deltas) ===")
        agreement = {}
        by_rater_deanon = []
        for path in args.rubrics:
            rubric_rows = load_csv_as_dicts(path)
            by_rater_deanon.append({r["id"]: r for r in deanonymize(rubric_rows, key_rows)})
        for dim in DIMENSIONS:
            deltas = []
            for rater_map in by_rater_deanon:
                d = {rid: r[f"personalized_{dim}"] - r[f"generic_{dim}"]
                     for rid, r in rater_map.items()
                     if f"personalized_{dim}" in r and f"generic_{dim}" in r}
                deltas.append(d)
            common_ids = set.intersection(*[set(d.keys()) for d in deltas]) if deltas else set()
            if len(common_ids) < 3:
                print(f"  {dim}: not enough common rated items to compute agreement")
                continue
            ids_sorted = sorted(common_ids)
            series = [[d[i] for i in ids_sorted] for d in deltas]
            corrs = []
            for i in range(len(series)):
                for j in range(i + 1, len(series)):
                    r_val, _ = stats.pearsonr(series[i], series[j])
                    corrs.append(r_val)
            mean_corr = float(np.mean(corrs)) if corrs else float("nan")
            agreement[dim] = {"mean_pairwise_pearson_r": mean_corr, "n_common_items": len(common_ids)}
            print(f"  {dim}: mean pairwise r={mean_corr:.2f} (n_common_items={len(common_ids)})")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"per_rater": all_rater_results, "pooled": pooled,
                   "inter_rater_agreement": agreement}, f, indent=2)
    print(f"\nSaved full analysis to {args.out}")


if __name__ == "__main__":
    main()
