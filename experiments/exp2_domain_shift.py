"""
Experiment 2: domain-shift test on faculty vignettes.

There is no public, labeled dataset of higher-education faculty emotions
(checked before writing this repo). So instead of pretending one exists,
this experiment uses a **researcher-authored scenario set / vignette-based
stress test** (data/vignettes/faculty_vignettes.csv): 75 short, first-person
statements that operationalise stressor categories reported in the existing
qualitative literature on academic-staff burnout and motivation (workload,
recognition, autonomy/bureaucracy, isolation, evaluation anxiety, etc.),
each manually labeled with one of the 5 common emotion classes. Do not
refer to this as a "faculty emotion dataset" in any write-up - it is a
constructed scenario set for stress-testing classifiers, not a sample of
real faculty data. See data/vignettes/vignette_validation_template.csv for
an optional expert face-validity check on the scenarios themselves.

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
                         NRCLexiconBaseline, TfidfBaseline, TransformerBaseline,
                         TransformerUnavailableError)
from src.plotting import plot_confusion_matrix

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
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
        llm_workers=8, use_transformer=False, transformer_model_name="distilbert-base-uncased",
        transformer_epochs=3, skip_transformer_5class=False):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

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
        plot_confusion_matrix(metrics["confusion_matrix"], COMMON_LABELS,
                               f"{name} - faculty vignettes (near-domain)",
                               os.path.join(FIGURES_DIR, f"{name}_vignettes_cm.png"))

    # Fair, label-space-matched TF-IDF: tfidf_logreg above is trained on all
    # 7 GoEmotions classes and can predict "neutral"/"surprise" on vignettes,
    # which are guaranteed wrong here and inflate its apparent error rate
    # (see out_of_label_space_rate in the saved metrics). This variant is
    # the same algorithm trained ONLY on the 5 classes the vignettes use, so
    # the comparison with nrc_lexicon/llm_fewshot (also restricted to 5
    # classes) is apples-to-apples.
    print("\n=== tfidf_logreg_5class (fair comparison) ===")
    train_5class = train_df[train_df["label"].isin(COMMON_LABELS)].reset_index(drop=True)
    tfidf5 = TfidfBaseline()
    tfidf5.fit(train_5class["text"], train_5class["label"])
    preds5 = tfidf5.predict(vignette_texts)
    pred_key = "tfidf_logreg_5class_pred"
    for r, p in zip(vignette_rows, preds5):
        r[pred_key] = p
    metrics5 = classification_metrics(vignette_labels, preds5, COMMON_LABELS)
    by_emotion5 = per_group_accuracy(vignette_rows, pred_key, "expected_emotion")
    by_category5 = per_group_accuracy(vignette_rows, pred_key, "context_tag")
    print(f"vignette accuracy={metrics5['accuracy']:.3f} macro_f1={metrics5['macro_f1']:.3f} "
          f"(n_train={len(train_5class)})")
    print("accuracy by emotion:", {k: round(v, 2) for k, v in by_emotion5.items()})
    all_metrics["tfidf_logreg_5class"] = {
        "overall": metrics5,
        "accuracy_by_emotion": by_emotion5,
        "accuracy_by_context_tag": by_category5,
    }
    plot_confusion_matrix(metrics5["confusion_matrix"], COMMON_LABELS,
                           "tfidf_logreg_5class - faculty vignettes (fair, label-matched)",
                           os.path.join(FIGURES_DIR, "tfidf_logreg_5class_vignettes_cm.png"))

    if use_transformer:
        print("\n=== transformer (optional) ===")
        try:
            tmodel = TransformerBaseline(model_name=transformer_model_name,
                                          num_epochs=2 if quick else transformer_epochs)
            # If Exp.1 was already run with --use_transformer on the same
            # train_df, this hits the on-disk cache instead of retraining -
            # see TransformerBaseline's cache_dir / _cache_key.
            tmodel.fit(train_df["text"], train_df["label"])
            preds_t = tmodel.predict(vignette_texts)
            pred_key = "transformer_pred"
            for r, p in zip(vignette_rows, preds_t):
                r[pred_key] = p
            metrics_t = classification_metrics(vignette_labels, preds_t, COMMON_LABELS)
            by_emotion_t = per_group_accuracy(vignette_rows, pred_key, "expected_emotion")
            by_category_t = per_group_accuracy(vignette_rows, pred_key, "context_tag")
            print(f"vignette accuracy={metrics_t['accuracy']:.3f} macro_f1={metrics_t['macro_f1']:.3f} "
                  f"out_of_label_space={metrics_t['out_of_label_space_rate']:.1%}")
            print("accuracy by emotion:", {k: round(v, 2) for k, v in by_emotion_t.items()})
            all_metrics["transformer"] = {
                "overall": metrics_t,
                "accuracy_by_emotion": by_emotion_t,
                "accuracy_by_context_tag": by_category_t,
            }
            plot_confusion_matrix(metrics_t["confusion_matrix"], COMMON_LABELS,
                                   "transformer - faculty vignettes (near-domain)",
                                   os.path.join(FIGURES_DIR, "transformer_vignettes_cm.png"))

            if not skip_transformer_5class:
                print("\n=== transformer_5class (fair comparison) ===")
                train_5class = train_df[train_df["label"].isin(COMMON_LABELS)].reset_index(drop=True)
                tmodel5 = TransformerBaseline(model_name=transformer_model_name,
                                               num_epochs=2 if quick else transformer_epochs)
                tmodel5.fit(train_5class["text"], train_5class["label"])
                preds5_t = tmodel5.predict(vignette_texts)
                pred_key5 = "transformer_5class_pred"
                for r, p in zip(vignette_rows, preds5_t):
                    r[pred_key5] = p
                metrics5_t = classification_metrics(vignette_labels, preds5_t, COMMON_LABELS)
                by_emotion5_t = per_group_accuracy(vignette_rows, pred_key5, "expected_emotion")
                by_category5_t = per_group_accuracy(vignette_rows, pred_key5, "context_tag")
                print(f"vignette accuracy={metrics5_t['accuracy']:.3f} macro_f1={metrics5_t['macro_f1']:.3f} "
                      f"(n_train={len(train_5class)})")
                all_metrics["transformer_5class"] = {
                    "overall": metrics5_t,
                    "accuracy_by_emotion": by_emotion5_t,
                    "accuracy_by_context_tag": by_category5_t,
                }
                plot_confusion_matrix(metrics5_t["confusion_matrix"], COMMON_LABELS,
                                      "transformer_5class - faculty vignettes (fair)",
                                      os.path.join(FIGURES_DIR, "transformer_5class_vignettes_cm.png"))
        except TransformerUnavailableError as exc:
            print(f"[skipped] transformer baseline unavailable: {exc}")
            if exc.__cause__:
                print(f"[skipped] underlying error (the real cause - read this, "
                      f"not just the message above): {type(exc.__cause__).__name__}: {exc.__cause__}")
            all_metrics["transformer"] = {"skipped_reason": str(exc)}

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
            plot_confusion_matrix(metrics["confusion_matrix"], COMMON_LABELS,
                                   "llm_fewshot - faculty vignettes (near-domain)",
                                   os.path.join(FIGURES_DIR, "llm_fewshot_vignettes_cm.png"))
        except LLMUnavailableError as exc:
            print(f"[skipped] llm_fewshot baseline unavailable: {exc}")
            all_metrics["llm_fewshot"] = {"skipped_reason": str(exc)}

    # save per-vignette predictions for inspection / appendix table
    pred_path = os.path.join(RESULTS_DIR, "exp2_predictions.csv")
    if os.path.exists(pred_path):
        with open(pred_path, encoding="utf-8") as f:
            previous_rows = {r["id"]: r for r in csv.DictReader(f)}
        current_cols = set(vignette_rows[0].keys())
        for row in vignette_rows:
            prev = previous_rows.get(row["id"])
            if not prev:
                continue
            for col, val in prev.items():
                if col not in current_cols and col not in row:
                    row[col] = val
    fieldnames = list(vignette_rows[0].keys())
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(vignette_rows)

    metrics_path = os.path.join(RESULTS_DIR, "exp2_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8") as f:
            previous = json.load(f)
        for key, val in previous.items():
            if key not in all_metrics:
                all_metrics[key] = val
                print(f"[exp2] kept existing '{key}' results from a previous "
                      f"run (not recomputed this time)")
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
    parser.add_argument("--use_transformer", action="store_true",
                         help="also try the transformer baseline on the vignettes "
                              "(needs internet access to Hugging Face Hub; reuses "
                              "the cached model from Exp.1 if it was trained there "
                              "with the same settings)")
    parser.add_argument("--transformer_model_name", default="distilbert-base-uncased")
    parser.add_argument("--transformer_epochs", type=int, default=3)
    parser.add_argument("--skip_transformer_5class", action="store_true")
    args = parser.parse_args()
    run(quick=args.quick, sample_size=args.sample_size, use_llm=args.use_llm,
        llm_examples_per_label=args.llm_examples_per_label,
        llm_workers=args.llm_workers, use_transformer=args.use_transformer,
        transformer_model_name=args.transformer_model_name,
        transformer_epochs=args.transformer_epochs,
        skip_transformer_5class=args.skip_transformer_5class)
