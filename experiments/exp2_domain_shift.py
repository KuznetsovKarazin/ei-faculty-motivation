"""
Experiment 2: domain-shift test on faculty vignettes.

There is no public, labeled dataset of higher-education faculty emotions
(checked before writing this repo). So instead of pretending one exists,
this experiment uses a small set of original illustrative vignettes
(data/vignettes/faculty_vignettes.csv): 75 short, first-person statements
that operationalise stressor categories reported in the existing
qualitative literature on academic-staff burnout and motivation (workload,
recognition, autonomy/bureaucracy, isolation, evaluation anxiety, etc.),
each manually labeled with one of the 5 common emotion classes.

This gives a 3-point domain-shift ladder for each classifier:
  1) GoEmotions test (in-domain: Reddit comments)
  2) ISEAR (cross-dataset: self-reported situations, many countries)
  3) Faculty vignettes (near-domain: higher-ed/workplace statements)

IMPORTANT: these 75 vignettes are illustrative examples written by the
research team for this feasibility study, not data collected from real
faculty. Treat Experiment 2 as a domain-shift stress test of the emotion
classifiers, not as evidence about real faculty populations.

Usage:
    python experiments/exp2_domain_shift.py
    python experiments/exp2_domain_shift.py --quick
    python experiments/exp2_domain_shift.py --use_llm
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_goemotions
from src.label_mapping import COMMON_LABELS
from src.metrics import classification_metrics
from src.models import (LLMFewShotBaseline, LLMUnavailableError,
                         NRCLexiconBaseline, TfidfBaseline)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
VIGNETTES_PATH = os.path.join(REPO_ROOT, "data", "vignettes", "faculty_vignettes.csv")


def load_vignettes():
    rows = []
    with open(VIGNETTES_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def per_group_accuracy(rows, pred_key, group_key="expected_emotion"):
    """Simple accuracy breakdown grouped by a column (emotion or category)."""
    groups = {}
    for r in rows:
        g = r[group_key]
        groups.setdefault(g, [0, 0])
        groups[g][1] += 1
        if r[pred_key] == r["expected_emotion"]:
            groups[g][0] += 1
    return {g: correct / total for g, (correct, total) in groups.items()}


def run(quick=False, sample_size=None, use_llm=False, llm_examples_per_label=5,
        llm_workers=8):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading GoEmotions train split (to train the same classifiers as Exp.1) ...")
    train_df = load_goemotions("train")
    if quick and sample_size is None:
        sample_size = 4000
    if sample_size:
        train_df = train_df.sample(n=min(sample_size, len(train_df)),
                                    random_state=42).reset_index(drop=True)

    vignette_rows = load_vignettes()
    vignette_texts = [r["text"] for r in vignette_rows]
    vignette_labels = [r["expected_emotion"] for r in vignette_rows]
    print(f"Loaded {len(vignette_rows)} faculty vignettes")

    models = {
        "nrc_lexicon": NRCLexiconBaseline(),
        "tfidf_logreg": TfidfBaseline(),
    }

    all_metrics = {}
    for name, model in models.items():
        print(f"\n=== {name} ===")
        model.fit(train_df["text"], train_df["label"])
        preds = model.predict(vignette_texts, COMMON_LABELS)

        pred_key = f"{name}_pred"
        for r, p in zip(vignette_rows, preds):
            r[pred_key] = p

        metrics = classification_metrics(vignette_labels, preds, COMMON_LABELS)
        by_emotion = per_group_accuracy(vignette_rows, pred_key, "expected_emotion")
        by_category = per_group_accuracy(vignette_rows, pred_key, "context_tag")

        print(f"vignette accuracy={metrics['accuracy']:.3f} "
              f"macro_f1={metrics['macro_f1']:.3f}")
        print("accuracy by emotion:", {k: round(v, 2) for k, v in by_emotion.items()})

        all_metrics[name] = {
            "overall": metrics,
            "accuracy_by_emotion": by_emotion,
            "accuracy_by_context_tag": by_category,
        }

    if use_llm:
        print("\n=== llm_fewshot (optional) ===")
        try:
            lmodel = LLMFewShotBaseline(n_examples_per_label=llm_examples_per_label,
                                         max_workers=llm_workers)
            lmodel.fit(train_df["text"], train_df["label"])
            preds = lmodel.predict(vignette_texts, COMMON_LABELS)

            pred_key = "llm_fewshot_pred"
            for r, p in zip(vignette_rows, preds):
                r[pred_key] = p

            metrics = classification_metrics(vignette_labels, preds, COMMON_LABELS)
            by_emotion = per_group_accuracy(vignette_rows, pred_key, "expected_emotion")
            by_category = per_group_accuracy(vignette_rows, pred_key, "context_tag")
            print(f"vignette accuracy={metrics['accuracy']:.3f} "
                  f"macro_f1={metrics['macro_f1']:.3f}")
            print("accuracy by emotion:", {k: round(v, 2) for k, v in by_emotion.items()})
            all_metrics["llm_fewshot"] = {
                "overall": metrics,
                "accuracy_by_emotion": by_emotion,
                "accuracy_by_context_tag": by_category,
            }
        except LLMUnavailableError as exc:
            print(f"[skipped] llm_fewshot baseline unavailable: {exc}")
            all_metrics["llm_fewshot"] = {"skipped_reason": str(exc)}

    # save per-vignette predictions for inspection / appendix table
    pred_path = os.path.join(RESULTS_DIR, "exp2_predictions.csv")
    fieldnames = list(vignette_rows[0].keys())
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(vignette_rows)

    metrics_path = os.path.join(RESULTS_DIR, "exp2_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nSaved per-vignette predictions to {pred_path}")
    print(f"Saved metrics to {metrics_path}")
    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                         help="use a small training subsample for a fast smoke test")
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--use_llm", action="store_true",
                         help="also try the optional LLM few-shot baseline on all "
                              "75 vignettes (needs ANTHROPIC_API_KEY and internet)")
    parser.add_argument("--llm_examples_per_label", type=int, default=5)
    parser.add_argument("--llm_workers", type=int, default=8)
    args = parser.parse_args()
    run(quick=args.quick, sample_size=args.sample_size, use_llm=args.use_llm,
        llm_examples_per_label=args.llm_examples_per_label,
        llm_workers=args.llm_workers)
