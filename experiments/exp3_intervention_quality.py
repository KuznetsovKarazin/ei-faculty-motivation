"""
Experiment 3: intervention message quality, as a 4-level ablation.

Instead of just "generic vs personalized", this compares 4 levels so a
paper can show what each layer actually adds:

  1. generic        - same message regardless of anything detected
  2. emotion_only    - acknowledges the raw detected emotion, no theory
  3. need_only       - routes through the SDT need (what earlier versions
                       of this repo called simply "personalized")
  4. full_context    - SDT need + the specific vignette text + a broad
                       stressor category (always via LLM; skipped cleanly
                       for ALL vignettes if no ANTHROPIC_API_KEY, rather
                       than partially generating it)

Two conditions, same as before:
  - oracle:     emotion = gold label
  - end_to_end: emotion = llm_fewshot's prediction (from
                results/exp2_predictions.csv, if Exp.2 was run with
                --use_llm). Skipped with a clear message otherwise.

For both conditions and all available levels, scores messages against the
GOLD need description with TF-IDF cosine similarity (secondary, automatic,
structurally biased toward the more specific levels - see README 8.4) and
exports a BLIND multi-level rubric (results/exp3_rubric_blind.csv, key in
results/exp3_rubric_key.csv) on a stratified subsample
(--rubric_sample_size, default 25) across 4 dimensions: relevance,
specificity, sdt_alignment, motivational_usefulness. The rubric, not the
TF-IDF score, is the result a paper should lead with - analyze filled-in
copies with experiments/exp3b_rubric_analysis.py.

Usage:
    python experiments/exp3_intervention_quality.py
    python experiments/exp3_intervention_quality.py --mode llm
    python experiments/exp3_intervention_quality.py --skip_full_context
    python experiments/exp3_intervention_quality.py --rubric_sample_size 30
"""

import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

from src.intervention_generator import (GENERIC_MESSAGE,
                                          FullContextUnavailableError,
                                          generate_emotion_only,
                                          generate_full_context,
                                          generate_need_only)
from src.metrics import pairwise_tfidf_similarity
from src.plotting import plot_ablation_comparison
from src.sdt_mapping import get_need

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
VIGNETTES_PATH = os.path.join(REPO_ROOT, "data", "vignettes", "faculty_vignettes.csv")
NEED_DESC_PATH = os.path.join(REPO_ROOT, "data", "vignettes", "need_descriptions.csv")
EXP2_PREDICTIONS_PATH = os.path.join(RESULTS_DIR, "exp2_predictions.csv")

SEED = 42
ALL_LEVELS = ["generic", "emotion_only", "need_only", "full_context"]
RUBRIC_DIMENSIONS = ["relevance", "specificity", "sdt_alignment", "motivational_usefulness"]


def load_csv_as_dicts(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _generate_levels(emotion, need, vignette_text, stressor_category, mode,
                      try_full_context, rng):
    """Return {level_name: message_text} for the levels we are using."""
    out = {
        "generic": GENERIC_MESSAGE,
        "emotion_only": generate_emotion_only(emotion, mode=mode, rng=rng),
        "need_only": generate_need_only(emotion, need, mode=mode, rng=rng),
    }
    if try_full_context:
        out["full_context"] = generate_full_context(vignette_text, emotion, need,
                                                      stressor_category)
    return out


def run(mode="template", skip_full_context=False, rubric_sample_size=25):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    random.seed(SEED)
    rng = random.Random(SEED)

    vignettes = load_csv_as_dicts(VIGNETTES_PATH)
    need_desc = {r["need"]: r["description"] for r in load_csv_as_dicts(NEED_DESC_PATH)}

    predicted_emotion = {}
    if os.path.exists(EXP2_PREDICTIONS_PATH):
        pred_rows = load_csv_as_dicts(EXP2_PREDICTIONS_PATH)
        if pred_rows and "llm_fewshot_pred" in pred_rows[0]:
            predicted_emotion = {r["id"]: r["llm_fewshot_pred"] for r in pred_rows}
    have_e2e = len(predicted_emotion) > 0
    conditions = ["oracle"] + (["end_to_end"] if have_e2e else [])

    # Probe full_context once before looping over all 75 vignettes x conditions,
    # so a missing API key fails fast instead of after 1 wasted call.
    levels = list(ALL_LEVELS)
    if skip_full_context:
        levels.remove("full_context")
        print("[exp3] --skip_full_context set: running generic/emotion_only/need_only only.")
    else:
        try:
            generate_full_context(vignettes[0]["text"], vignettes[0]["expected_emotion"],
                                   get_need(vignettes[0]["expected_emotion"]),
                                   vignettes[0]["stressor_category"])
        except FullContextUnavailableError as exc:
            levels.remove("full_context")
            print(f"[exp3] full_context level skipped for all vignettes: {exc}")

    rows = []  # one row per (vignette, condition)
    for v in vignettes:
        gold_emotion = v["expected_emotion"]
        gold_need = get_need(gold_emotion)
        gold_desc = need_desc.get(gold_need, "")

        for cond in conditions:
            emotion = gold_emotion if cond == "oracle" else predicted_emotion.get(v["id"], gold_emotion)
            need = gold_need if cond == "oracle" else get_need(emotion)
            messages = _generate_levels(emotion, need, v["text"], v["stressor_category"],
                                         mode, "full_context" in levels, rng)
            row = {"id": v["id"], "condition": cond, "emotion_used": emotion,
                   "need_used": need, "gold_emotion": gold_emotion, "gold_need": gold_need,
                   "stressor_category": v["stressor_category"], "vignette_text": v["text"],
                   "need_description": gold_desc}
            for lvl in levels:
                row[f"message_{lvl}"] = messages[lvl]
            rows.append(row)

    # automatic TF-IDF scoring, per level, against the GOLD need description
    references = [r["need_description"] for r in rows]
    sims = {}
    for lvl in levels:
        sims[lvl] = pairwise_tfidf_similarity([r[f"message_{lvl}"] for r in rows], references)
        for r, s in zip(rows, sims[lvl]):
            r[f"sim_{lvl}"] = s

    out_csv = os.path.join(RESULTS_DIR, "exp3_outputs.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {"n_vignettes": len(vignettes), "conditions": conditions,
               "levels": levels, "mode": mode}
    for cond in conditions:
        cond_rows = [r for r in rows if r["condition"] == cond]
        cond_summary = {"n": len(cond_rows)}
        means, sems = {}, {}
        for lvl in levels:
            vals = [r[f"sim_{lvl}"] for r in cond_rows]
            means[lvl] = float(np.mean(vals))
            sems[lvl] = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        cond_summary["mean_similarity"] = means
        # staircase comparisons: each level vs the previous one
        staircase = {}
        for a, b in zip(levels[:-1], levels[1:]):
            va = [r[f"sim_{a}"] for r in cond_rows]
            vb = [r[f"sim_{b}"] for r in cond_rows]
            t_stat, t_p = stats.ttest_rel(vb, va)
            w_stat, w_p = stats.wilcoxon(vb, va)
            staircase[f"{a}_vs_{b}"] = {"paired_ttest_p": t_p, "wilcoxon_p": w_p,
                                          "mean_diff": means[b] - means[a]}
        cond_summary["staircase_comparisons"] = staircase
        summary[cond] = cond_summary

        plot_ablation_comparison(
            means, sems, os.path.join(FIGURES_DIR, f"exp3_ablation_{cond}.png"),
            title=f"Message relevance by personalization level ({cond})")

        print(f"\n=== {cond} (n={len(cond_rows)}) ===")
        for lvl in levels:
            print(f"  {lvl:15s} mean_sim={means[lvl]:.3f}")
        for pair, st in staircase.items():
            print(f"  {pair:30s} diff={st['mean_diff']:+.3f}  "
                  f"paired t-test p={st['paired_ttest_p']:.4f}")

    summary_path = os.path.join(RESULTS_DIR, "exp3_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _write_blind_rubric(rows, levels, conditions, rubric_sample_size, rng)

    print(f"\nSaved per-row outputs to {out_csv}")
    print(f"Saved summary statistics to {summary_path}")
    print(f"Saved figures to {FIGURES_DIR}")
    print(f"Saved blind rubric + key to {RESULTS_DIR}")
    if not have_e2e:
        print("\n[note] end_to_end condition skipped: results/exp2_predictions.csv has "
              "no 'llm_fewshot_pred' column yet. Run experiments/exp2_domain_shift.py "
              "--use_llm first to enable it.")
    return summary


def _write_blind_rubric(rows, levels, conditions, sample_size, rng):
    """Blind multi-level rubric on a stratified subsample of vignettes, for
    the most realistic available condition (end_to_end if present, else
    oracle). Message identity is randomized into slots A/B/C/(D); the key
    mapping slots back to levels is saved separately and must NOT be
    shown to raters.
    """
    cond = "end_to_end" if "end_to_end" in conditions else "oracle"
    cond_rows = [r for r in rows if r["condition"] == cond]

    # stratify the subsample by gold_emotion so all 5 emotions are represented
    by_emotion = {}
    for r in cond_rows:
        by_emotion.setdefault(r["gold_emotion"], []).append(r)
    n_per_emotion = max(1, sample_size // max(len(by_emotion), 1))
    subsample = []
    for emo, group in by_emotion.items():
        subsample.extend(rng.sample(group, min(n_per_emotion, len(group))))

    slot_letters = ["a", "b", "c", "d"][:len(levels)]
    rubric_rows, key_rows = [], []
    for r in subsample:
        order = list(levels)
        rng.shuffle(order)
        row = {"id": r["id"], "vignette_text": r["vignette_text"]}
        key_row = {"id": r["id"], "condition": cond}
        for slot, lvl in zip(slot_letters, order):
            row[f"message_{slot}"] = r[f"message_{lvl}"]
            key_row[f"slot_{slot}"] = lvl
            for dim in RUBRIC_DIMENSIONS:
                row[f"{dim}_{slot}_1_5"] = ""
        rubric_rows.append(row)
        key_rows.append(key_row)

    if not rubric_rows:
        return
    rubric_path = os.path.join(RESULTS_DIR, "exp3_rubric_blind.csv")
    with open(rubric_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rubric_rows[0].keys()))
        w.writeheader()
        w.writerows(rubric_rows)

    key_path = os.path.join(RESULTS_DIR, "exp3_rubric_key.csv")
    with open(key_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(key_rows[0].keys()))
        w.writeheader()
        w.writerows(key_rows)

    print(f"  rubric built on {len(rubric_rows)} vignettes (condition={cond}), "
          f"{len(levels)} levels/slots, {len(RUBRIC_DIMENSIONS)} dimensions each")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["template", "llm"], default="template",
                         help="generation mode for emotion_only/need_only "
                              "('llm' needs ANTHROPIC_API_KEY); full_context "
                              "always uses the LLM regardless of this flag")
    parser.add_argument("--skip_full_context", action="store_true",
                         help="don't attempt the full_context level at all "
                              "(saves API calls)")
    parser.add_argument("--rubric_sample_size", type=int, default=25,
                         help="number of vignettes (stratified by emotion) "
                              "included in the blind human-rating rubric")
    args = parser.parse_args()
    run(mode=args.mode, skip_full_context=args.skip_full_context,
        rubric_sample_size=args.rubric_sample_size)
