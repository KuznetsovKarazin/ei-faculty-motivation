"""
Rigorous re-analysis of the human evaluation (Experiment 3) data:
1. Build a long-format table: rater x vignette x level x dimension x score.
2. Rater-level aggregated paired analysis (n=21 raters, not n=525 pseudo-independent ratings).
3. Crossed mixed-effects model (random intercepts for rater AND vignette) per dimension,
   fit three times (releveling the reference category) to get all three adjacent
   staircase-step contrasts with proper standard errors.
4. Effect sizes (Cohen's d on rater-level paired differences) for the new effect-size figure.
"""
import csv
import json
import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

BASE = "/home/claude/work/ei-faculty-motivation"
RATINGS_DIR = os.path.join(BASE, "results", "human_ratings")
KEY_PATH = os.path.join(BASE, "results", "exp3_rubric_key.csv")

DIMENSIONS = ["relevance", "specificity", "sdt_alignment", "motivational_usefulness"]
LEVELS = ["generic", "emotion_only", "need_only", "full_context"]
SLOTS = ["a", "b", "c", "d"]

# ---- 1. load rubric key: vignette_id -> {slot: level} ----
key = {}
with open(KEY_PATH, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        key[row["id"]] = {s: row[f"slot_{s}"] for s in SLOTS}

# ---- 2. load all rater files into long format ----
records = []
rater_files = sorted(f for f in os.listdir(RATINGS_DIR) if f.startswith("rater_"))
print(f"Found {len(rater_files)} rater files.")

for fname in rater_files:
    rater_id = fname.replace(".csv", "")
    path = os.path.join(RATINGS_DIR, fname)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = row["id"]
            slot_to_level = key[vid]
            for slot in SLOTS:
                level = slot_to_level[slot]
                for dim in DIMENSIONS:
                    col = f"{dim}_{slot}_1_5"
                    val = row.get(col, "")
                    if val not in (None, ""):
                        records.append({
                            "rater": rater_id,
                            "vignette": vid,
                            "level": level,
                            "dimension": dim,
                            "score": float(val),
                        })

df = pd.DataFrame.from_records(records)
print("Long-format table shape:", df.shape)
print(df.groupby(["dimension", "level"]).size())

df.to_csv(os.path.join(BASE, "results", "human_ratings_long.csv"), index=False)

# ---- 3. rater-level aggregation (n = 21) ----
rater_means = (
    df.groupby(["rater", "level", "dimension"])["score"]
    .mean()
    .reset_index()
    .pivot_table(index=["rater", "dimension"], columns="level", values="score")
    .reset_index()
)
rater_means = rater_means[["rater", "dimension"] + LEVELS]
rater_means.to_csv(os.path.join(BASE, "results", "rater_level_means.csv"), index=False)

staircase_pairs = [
    ("generic", "emotion_only"),
    ("emotion_only", "need_only"),
    ("need_only", "full_context"),
]

rater_level_results = {}
for dim in DIMENSIONS:
    sub = rater_means[rater_means["dimension"] == dim]
    dim_results = {}
    for lo, hi in staircase_pairs:
        diffs = sub[hi].values - sub[lo].values
        t, p = stats.ttest_rel(sub[hi].values, sub[lo].values)
        w_stat, w_p = stats.wilcoxon(sub[hi].values, sub[lo].values)
        d = diffs.mean() / diffs.std(ddof=1)
        dim_results[f"{lo}_vs_{hi}"] = {
            "n_raters": len(diffs),
            "mean_diff": float(diffs.mean()),
            "sd_diff": float(diffs.std(ddof=1)),
            "t": float(t),
            "paired_ttest_p": float(p),
            "wilcoxon_p": float(w_p),
            "cohens_d": float(d),
        }
    rater_level_results[dim] = dim_results

print(json.dumps(rater_level_results, indent=2))

# overall means per level (rater-level n=21 average of rater means)
rater_level_grand_means = (
    rater_means.groupby("dimension")[LEVELS].mean().to_dict(orient="index")
)
print("Grand means (mean of 21 rater-level means):")
print(json.dumps(rater_level_grand_means, indent=2))

with open(os.path.join(BASE, "results", "rater_level_staircase.json"), "w") as f:
    json.dump({
        "staircase": rater_level_results,
        "grand_means": rater_level_grand_means,
        "n_raters": int(rater_means["rater"].nunique()),
    }, f, indent=2)

# ---- 4. crossed mixed-effects model per dimension, releveled 3x ----
mixed_results = {}
df["level"] = pd.Categorical(df["level"], categories=LEVELS, ordered=True)

for dim in DIMENSIONS:
    sub = df[df["dimension"] == dim].copy()
    dim_out = {}
    for ref in ["generic", "emotion_only", "need_only"]:
        cats = [ref] + [l for l in LEVELS if l != ref]
        sub["level_releveled"] = pd.Categorical(sub["level"], categories=cats)
        model = smf.mixedlm(
            "score ~ C(level_releveled, Treatment(reference='%s'))" % ref,
            data=sub,
            groups=np.ones(len(sub)),
            vc_formula={
                "rater": "0 + C(rater)",
                "vignette": "0 + C(vignette)",
            },
        )
        try:
            result = model.fit(reml=True, method=["powell"], maxiter=500)
        except Exception as exc:
            dim_out[ref] = {"error": str(exc)}
            continue
        var_rater = float(result.vcomp[0]) if len(result.vcomp) > 0 else None
        var_vignette = float(result.vcomp[1]) if len(result.vcomp) > 1 else None
        coefs = {}
        for name, est, se, p in zip(
            result.params.index, result.params.values, result.bse.values, result.pvalues.values
        ):
            if "level_releveled" in name:
                lvl = name.split("T.")[-1].rstrip("]")
                coefs[lvl] = {"estimate": float(est), "se": float(se), "p": float(p)}
        dim_out[ref] = {
            "reference": ref,
            "coefs_vs_reference": coefs,
            "n_obs": int(result.nobs),
            "converged": bool(result.converged),
            "var_rater": var_rater,
            "var_vignette": var_vignette,
            "residual_scale": float(result.scale),
        }
    mixed_results[dim] = dim_out

with open(os.path.join(BASE, "results", "mixed_effects_results.json"), "w") as f:
    json.dump(mixed_results, f, indent=2)

print(json.dumps(mixed_results, indent=2))
print("DONE")
