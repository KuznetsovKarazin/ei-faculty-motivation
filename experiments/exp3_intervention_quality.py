"""
Experiment 3: intervention message quality.

For each faculty vignette, generates two messages:
  (a) GENERIC_MESSAGE   — the same non-personalized message every time
  (b) personalized      — generated from the detected emotion's SDT need
                           (src/sdt_mapping.py + src/intervention_generator.py)

and scores how relevant each message is to the actual underlying need, using
TF-IDF cosine similarity against a short canonical description of that need
(data/vignettes/need_descriptions.csv). This is a fully automatic, offline
relevance proxy. It does NOT measure whether the message would actually
make a real person feel more motivated — that requires a human study with
real faculty (see README "Limitations").

Also exports a blank rubric (results/exp3_rubric_template.csv) so a small
panel of reviewers can manually score relevance / specificity /
appropriateness for both message types on a 1-5 scale, as a complementary,
human-judged comparison.

Usage:
    python experiments/exp3_intervention_quality.py
"""

import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy import stats

from src.intervention_generator import GENERIC_MESSAGE, generate_message
from src.metrics import pairwise_tfidf_similarity
from src.sdt_mapping import get_need

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
VIGNETTES_PATH = os.path.join(REPO_ROOT, "data", "vignettes", "faculty_vignettes.csv")
NEED_DESC_PATH = os.path.join(REPO_ROOT, "data", "vignettes", "need_descriptions.csv")

SEED = 42


def load_csv_as_dicts(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run(mode="template"):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    random.seed(SEED)

    vignettes = load_csv_as_dicts(VIGNETTES_PATH)
    need_desc = {r["need"]: r["description"] for r in load_csv_as_dicts(NEED_DESC_PATH)}

    rows = []
    for v in vignettes:
        emotion = v["expected_emotion"]
        need = get_need(emotion)
        personalized = generate_message(emotion, need, mode=mode)
        rows.append({
            "id": v["id"],
            "emotion": emotion,
            "need": need,
            "vignette_text": v["text"],
            "generic_message": GENERIC_MESSAGE,
            "personalized_message": personalized,
            "need_description": need_desc.get(need, ""),
        })

    references = [r["need_description"] for r in rows]
    sim_generic = pairwise_tfidf_similarity([r["generic_message"] for r in rows],
                                             references)
    sim_personalized = pairwise_tfidf_similarity([r["personalized_message"] for r in rows],
                                                  references)
    for r, sg, sp in zip(rows, sim_generic, sim_personalized):
        r["sim_generic"] = sg
        r["sim_personalized"] = sp

    out_csv = os.path.join(RESULTS_DIR, "exp3_outputs.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # blank rubric for optional human expert scoring (1-5 scale)
    rubric_csv = os.path.join(RESULTS_DIR, "exp3_rubric_template.csv")
    rubric_fields = ["id", "emotion", "need", "generic_message",
                      "personalized_message", "relevance_generic_1_5",
                      "relevance_personalized_1_5", "specificity_generic_1_5",
                      "specificity_personalized_1_5",
                      "appropriateness_generic_1_5",
                      "appropriateness_personalized_1_5"]
    with open(rubric_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rubric_fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "id": r["id"], "emotion": r["emotion"], "need": r["need"],
                "generic_message": r["generic_message"],
                "personalized_message": r["personalized_message"],
            })

    t_stat, t_p = stats.ttest_rel(sim_personalized, sim_generic)
    w_stat, w_p = stats.wilcoxon(sim_personalized, sim_generic)

    summary = {
        "n": len(rows),
        "mode": mode,
        "mean_sim_generic": sum(sim_generic) / len(sim_generic),
        "mean_sim_personalized": sum(sim_personalized) / len(sim_personalized),
        "paired_ttest": {"t": t_stat, "p": t_p},
        "wilcoxon": {"stat": w_stat, "p": w_p},
    }
    summary_path = os.path.join(RESULTS_DIR, "exp3_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"n = {summary['n']}")
    print(f"mean TF-IDF similarity to target need (generic)      = "
          f"{summary['mean_sim_generic']:.3f}")
    print(f"mean TF-IDF similarity to target need (personalized) = "
          f"{summary['mean_sim_personalized']:.3f}")
    print(f"paired t-test: t={t_stat:.3f}, p={t_p:.4f}")
    print(f"Wilcoxon signed-rank: stat={w_stat:.3f}, p={w_p:.4f}")
    print(f"\nSaved per-vignette outputs to {out_csv}")
    print(f"Saved blank human-rating rubric to {rubric_csv}")
    print(f"Saved summary statistics to {summary_path}")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["template", "llm"], default="template",
                         help="'llm' requires ANTHROPIC_API_KEY and internet access")
    args = parser.parse_args()
    run(mode=args.mode)
