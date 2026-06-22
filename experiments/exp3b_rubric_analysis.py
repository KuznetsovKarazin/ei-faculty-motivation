"""
Analyze one or more FILLED-IN copies of the blind multi-level rubric
produced by experiments/exp3_intervention_quality.py
(results/exp3_rubric_blind.csv).

Workflow:
  1. Run experiments/exp3_intervention_quality.py - it writes
     results/exp3_rubric_blind.csv (give copies to 1-5 raters, who do NOT
     see results/exp3_rubric_key.csv) and results/exp3_rubric_key.csv
     (keep this yourself - it maps each anonymous slot a/b/c/(d) back to
     the actual personalization level: generic / emotion_only / need_only
     / full_context).
  2. Each rater fills in the *_1_5 columns (1-5 scale) for every slot,
     across 4 dimensions (relevance, specificity, sdt_alignment,
     motivational_usefulness), not knowing which slot is which level, and
     saves their own copy, e.g. results/filled_rater1.csv.
  3. Run this script:

     python experiments/exp3b_rubric_analysis.py \\
         --rubrics results/filled_rater1.csv results/filled_rater2.csv \\
         --key results/exp3_rubric_key.csv

This de-anonymizes each rater's slot scores back into level names using
the key, reports per-rater and pooled means per level/dimension, the
staircase comparisons (generic -> emotion_only -> need_only ->
full_context), and (with 2+ raters) a simple inter-rater agreement check.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

DIMENSIONS = ["relevance", "specificity", "sdt_alignment", "motivational_usefulness"]


def load_csv_as_dicts(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def deanonymize(rubric_rows, key_rows):
    """Return {id: {level: {dimension: score}}} using the key to map each
    anonymous slot back to its actual personalization level."""
    key = {r["id"]: r for r in key_rows}
    slot_cols = [c for c in rubric_rows[0].keys() if c.startswith("slot_")] if False else None
    out = {}
    for r in rubric_rows:
        k = key.get(r["id"])
        if k is None:
            continue
        levels_here = {}
        for slot_key, level in k.items():
            if not slot_key.startswith("slot_"):
                continue
            slot = slot_key.split("_", 1)[1]  # "a", "b", ...
            dims = {}
            for dim in DIMENSIONS:
                val = r.get(f"{dim}_{slot}_1_5", "")
                if val != "":
                    dims[dim] = float(val)
            if dims:
                levels_here[level] = dims
        if levels_here:
            out[r["id"]] = levels_here
    return out


def analyze(deanon, rater_name):
    """deanon: {id: {level: {dim: score}}}. Returns per-level-per-dim means
    and staircase paired comparisons (generic -> emotion_only -> need_only
    -> full_context, using whichever of those levels are present)."""
    by_level_dim = defaultdict(lambda: defaultdict(list))
    for item in deanon.values():
        for level, dims in item.items():
            for dim, val in dims.items():
                by_level_dim[level][dim].append(val)

    levels_present = [lvl for lvl in ["generic", "emotion_only", "need_only", "full_context"]
                       if lvl in by_level_dim]

    result = {"rater": rater_name, "n_items": len(deanon), "levels": levels_present, "means": {}}
    for lvl in levels_present:
        result["means"][lvl] = {dim: float(np.mean(vals)) for dim, vals in by_level_dim[lvl].items()}

    # staircase: need paired (same-id) values per dimension
    staircase = {}
    for a, b in zip(levels_present[:-1], levels_present[1:]):
        staircase[f"{a}_vs_{b}"] = {}
        for dim in DIMENSIONS:
            pairs = [(item[a][dim], item[b][dim]) for item in deanon.values()
                     if a in item and b in item and dim in item[a] and dim in item[b]]
            if len(pairs) < 2:
                continue
            va, vb = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
            diff = vb - va
            t_stat, t_p = stats.ttest_rel(vb, va)
            sd = np.std(diff, ddof=1)
            d = float(np.mean(diff) / sd) if sd > 0 else 0.0
            value_added = "n.s." if t_p >= 0.05 else (
                "+" if abs(d) < 0.5 else "++" if abs(d) < 0.8 else "+++")
            staircase[f"{a}_vs_{b}"][dim] = {"n": len(pairs), "mean_diff": float(np.mean(diff)),
                                               "paired_ttest_p": float(t_p), "cohens_d": d,
                                               "value_added": value_added}
    result["staircase"] = staircase
    return result


def print_result(result):
    print(f"\n=== {result['rater']} (n={result['n_items']} items, "
          f"levels={result['levels']}) ===")
    for lvl in result["levels"]:
        means = result["means"].get(lvl, {})
        print(f"  {lvl:15s} " + "  ".join(f"{d}={means.get(d, float('nan')):.2f}" for d in DIMENSIONS))
    for pair, dims in result["staircase"].items():
        for dim, st in dims.items():
            print(f"  {pair:30s} [{dim}] diff={st['mean_diff']:+.2f} "
                  f"d={st['cohens_d']:+.2f} p={st['paired_ttest_p']:.4f} "
                  f"(n={st['n']})  value_added={st['value_added']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rubrics", nargs="+", required=True,
                         help="one or more filled-in copies of exp3_rubric_blind.csv")
    parser.add_argument("--key", default=os.path.join(RESULTS_DIR, "exp3_rubric_key.csv"))
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "exp3_rubric_analysis.json"))
    args = parser.parse_args()

    key_rows = load_csv_as_dicts(args.key)
    per_rater_results = []
    pooled_deanon = {}

    for path in args.rubrics:
        rubric_rows = load_csv_as_dicts(path)
        deanon = deanonymize(rubric_rows, key_rows)
        rater_name = os.path.splitext(os.path.basename(path))[0]
        result = analyze(deanon, rater_name)
        per_rater_results.append(result)
        print_result(result)
        for item_id, item in deanon.items():
            pooled_deanon.setdefault(f"{rater_name}::{item_id}", item)

    pooled = analyze(pooled_deanon, f"POOLED ({len(args.rubrics)} rater(s))")
    print_result(pooled)

    agreement = None
    if len(args.rubrics) >= 2:
        print("\n=== Inter-rater agreement (Pearson r on common items, per level/dim) ===")
        per_rater_deanon = []
        for path in args.rubrics:
            rubric_rows = load_csv_as_dicts(path)
            per_rater_deanon.append(deanonymize(rubric_rows, key_rows))
        common_ids = set.intersection(*[set(d.keys()) for d in per_rater_deanon])
        agreement = {}
        if len(common_ids) >= 3:
            levels_present = pooled["levels"]
            for lvl in levels_present:
                agreement[lvl] = {}
                for dim in DIMENSIONS:
                    series = []
                    ok = True
                    for d in per_rater_deanon:
                        vals = [d[i].get(lvl, {}).get(dim) for i in sorted(common_ids)]
                        if any(v is None for v in vals):
                            ok = False
                            break
                        series.append(vals)
                    if not ok or len(series) < 2:
                        continue
                    corrs = []
                    for i in range(len(series)):
                        for j in range(i + 1, len(series)):
                            r_val, _ = stats.pearsonr(series[i], series[j])
                            corrs.append(r_val)
                    mean_r = float(np.mean(corrs)) if corrs else float("nan")
                    agreement[lvl][dim] = mean_r
                    print(f"  {lvl:15s} [{dim}] mean pairwise r={mean_r:.2f}")
        else:
            print("  not enough common rated items to compute agreement")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"per_rater": per_rater_results, "pooled": pooled,
                   "inter_rater_agreement": agreement}, f, indent=2)
    print(f"\nSaved full analysis to {args.out}")


if __name__ == "__main__":
    main()
