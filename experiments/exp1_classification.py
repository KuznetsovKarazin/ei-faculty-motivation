"""
Experiment 1: emotion classification accuracy and cross-dataset
generalization.

Trains each baseline on GoEmotions (train split) and evaluates it two ways:
  (a) in-domain: on the GoEmotions test split (7 classes: Ekman-6 + neutral)
  (b) cross-dataset: on ISEAR, restricted to the 5 labels shared with
      GoEmotions (anger, disgust, fear, joy, sadness). This tests whether a
      model trained on social-media text (Reddit) generalizes to a
      different text register, source, and (for ISEAR) culture.

Usage:
    python experiments/exp1_classification.py
    python experiments/exp1_classification.py --quick
    python experiments/exp1_classification.py --use_transformer
    python experiments/exp1_classification.py --use_llm --llm_sample_size 150 \\
        --llm_examples_per_label 5 --llm_workers 8
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_goemotions, load_isear_common_labels, stratified_sample
from src.label_mapping import COMMON_LABELS, EKMAN_PLUS_NEUTRAL
from src.metrics import classification_metrics
from src.models import (LLMFewShotBaseline, LLMUnavailableError,
                         NRCLexiconBaseline, TfidfBaseline, TransformerBaseline,
                         TransformerUnavailableError)

from src.plotting import plot_confusion_matrix

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def run(quick=False, use_transformer=False, sample_size=None, use_llm=False,
        llm_sample_size=150, llm_examples_per_label=5, llm_workers=8):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading GoEmotions ...")
    train_df = load_goemotions("train")
    test_df = load_goemotions("test")

    if quick and sample_size is None:
        sample_size = 4000
    if sample_size:
        train_df = train_df.sample(n=min(sample_size, len(train_df)),
                                    random_state=42).reset_index(drop=True)
        print(f"[quick mode] using {len(train_df)} training rows")

    print(f"GoEmotions train={len(train_df)} test={len(test_df)}")

    print("Loading ISEAR (common-label subset) ...")
    isear_df = load_isear_common_labels()
    print(f"ISEAR common-label rows={len(isear_df)}")

    models = {
        "nrc_lexicon": NRCLexiconBaseline(),
        "tfidf_logreg": TfidfBaseline(),
    }

    all_results = {}

    for name, model in models.items():
        print(f"\n=== {name} ===")
        model.fit(train_df["text"], train_df["label"])

        # (a) in-domain evaluation on GoEmotions test
        if name == "nrc_lexicon":
            preds_in = model.predict(test_df["text"], EKMAN_PLUS_NEUTRAL)
        else:
            preds_in = model.predict(test_df["text"])
        metrics_in = classification_metrics(test_df["label"], preds_in,
                                             EKMAN_PLUS_NEUTRAL)
        print(f"in-domain accuracy={metrics_in['accuracy']:.3f} "
              f"macro_f1={metrics_in['macro_f1']:.3f}")

        # (b) cross-dataset evaluation on ISEAR
        if name == "nrc_lexicon":
            preds_cross = model.predict(isear_df["text"], COMMON_LABELS)
        else:
            preds_cross = model.predict(isear_df["text"])
        metrics_cross = classification_metrics(isear_df["label"], preds_cross,
                                                 COMMON_LABELS)
        print(f"cross-dataset (ISEAR) accuracy={metrics_cross['accuracy']:.3f} "
              f"macro_f1={metrics_cross['macro_f1']:.3f} "
              f"out_of_label_space={metrics_cross['out_of_label_space_rate']:.1%}")

        all_results[name] = {"in_domain": metrics_in, "cross_dataset": metrics_cross}

        plot_confusion_matrix(metrics_in["confusion_matrix"], EKMAN_PLUS_NEUTRAL,
                               f"{name} - GoEmotions test (in-domain)",
                               os.path.join(FIGURES_DIR, f"{name}_in_domain_cm.png"))
        plot_confusion_matrix(metrics_cross["confusion_matrix"], COMMON_LABELS,
                               f"{name} - ISEAR (cross-dataset)",
                               os.path.join(FIGURES_DIR, f"{name}_cross_dataset_cm.png"))

    # --- Fair, label-space-matched comparison ---
    # tfidf_logreg above is trained on all 7 GoEmotions classes, so on
    # ISEAR (5 classes) it can predict "neutral"/"surprise", which are
    # guaranteed wrong there and make its cross-dataset score partly an
    # artifact of label-space mismatch rather than pure generalization
    # failure (see out_of_label_space_rate above - it's large for
    # tfidf_logreg). nrc_lexicon and llm_fewshot are explicitly restricted
    # to the eval set's labels, so they never have this advantage/penalty.
    # This variant levels the field: same algorithm, trained ONLY on the
    # 5 classes ISEAR actually uses, so it can never "waste" a prediction.
    print("\n=== tfidf_logreg_5class (fair comparison, ISEAR-only) ===")
    train_5class = train_df[train_df["label"].isin(COMMON_LABELS)].reset_index(drop=True)
    tfidf5 = TfidfBaseline()
    tfidf5.fit(train_5class["text"], train_5class["label"])
    preds_cross5 = tfidf5.predict(isear_df["text"])
    metrics_cross5 = classification_metrics(isear_df["label"], preds_cross5, COMMON_LABELS)
    print(f"cross-dataset (ISEAR) accuracy={metrics_cross5['accuracy']:.3f} "
          f"macro_f1={metrics_cross5['macro_f1']:.3f} "
          f"out_of_label_space={metrics_cross5['out_of_label_space_rate']:.1%} "
          f"(n_train={len(train_5class)})")
    all_results["tfidf_logreg_5class"] = {"cross_dataset": metrics_cross5,
                                           "note": "trained only on the 5 ISEAR-shared "
                                                    "classes; not evaluated in-domain "
                                                    "since GoEmotions test includes "
                                                    "surprise/neutral which this "
                                                    "variant can never predict"}
    plot_confusion_matrix(metrics_cross5["confusion_matrix"], COMMON_LABELS,
                           "tfidf_logreg_5class - ISEAR (fair, label-matched)",
                           os.path.join(FIGURES_DIR, "tfidf_logreg_5class_cross_dataset_cm.png"))

    if use_transformer:
        print("\n=== transformer (optional) ===")
        try:
            tmodel = TransformerBaseline(num_epochs=2 if quick else 3)
            tmodel.fit(train_df["text"], train_df["label"])
            preds_in = tmodel.predict(test_df["text"])
            metrics_in = classification_metrics(test_df["label"], preds_in,
                                                 EKMAN_PLUS_NEUTRAL)
            preds_cross = tmodel.predict(isear_df["text"])
            metrics_cross = classification_metrics(isear_df["label"], preds_cross,
                                                     COMMON_LABELS)
            all_results["transformer"] = {"in_domain": metrics_in,
                                           "cross_dataset": metrics_cross}
            print(f"in-domain accuracy={metrics_in['accuracy']:.3f}")
            print(f"cross-dataset accuracy={metrics_cross['accuracy']:.3f}")
        except TransformerUnavailableError as exc:
            print(f"[skipped] transformer baseline unavailable: {exc}")
            all_results["transformer"] = {"skipped_reason": str(exc)}

    if use_llm:
        print("\n=== llm_fewshot (optional) ===")
        try:
            lmodel = LLMFewShotBaseline(n_examples_per_label=llm_examples_per_label,
                                         max_workers=llm_workers)
            lmodel.fit(train_df["text"], train_df["label"])

            print(f"(LLM calls cost money/time: evaluating on up to "
                  f"{llm_sample_size} rows PER CLASS, not the full sets. "
                  "Rare classes use all available rows instead. "
                  "Increase --llm_sample_size for a larger, more reliable estimate.)")
            test_sub = stratified_sample(test_df, "label", llm_sample_size)
            isear_sub = stratified_sample(isear_df, "label", llm_sample_size)

            preds_in = lmodel.predict(test_sub["text"], EKMAN_PLUS_NEUTRAL)
            metrics_in = classification_metrics(test_sub["label"], preds_in,
                                                 EKMAN_PLUS_NEUTRAL)
            preds_cross = lmodel.predict(isear_sub["text"], COMMON_LABELS)
            metrics_cross = classification_metrics(isear_sub["label"], preds_cross,
                                                     COMMON_LABELS)
            all_results["llm_fewshot"] = {"in_domain": metrics_in,
                                           "cross_dataset": metrics_cross,
                                           "n_in_domain": len(test_sub),
                                           "n_cross_dataset": len(isear_sub)}
            print(f"in-domain accuracy={metrics_in['accuracy']:.3f} "
                  f"(n={len(test_sub)})")
            print(f"cross-dataset accuracy={metrics_cross['accuracy']:.3f} "
                  f"(n={len(isear_sub)})")
            plot_confusion_matrix(metrics_in["confusion_matrix"], EKMAN_PLUS_NEUTRAL,
                                   "llm_fewshot - GoEmotions test (in-domain)",
                                   os.path.join(FIGURES_DIR, "llm_fewshot_in_domain_cm.png"))
            plot_confusion_matrix(metrics_cross["confusion_matrix"], COMMON_LABELS,
                                   "llm_fewshot - ISEAR (cross-dataset)",
                                   os.path.join(FIGURES_DIR, "llm_fewshot_cross_dataset_cm.png"))
        except LLMUnavailableError as exc:
            print(f"[skipped] llm_fewshot baseline unavailable: {exc}")
            all_results["llm_fewshot"] = {"skipped_reason": str(exc)}

    out_path = os.path.join(RESULTS_DIR, "exp1_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved metrics to {out_path}")
    print(f"Saved confusion matrix plots to {FIGURES_DIR}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                         help="use a small training subsample for a fast smoke test")
    parser.add_argument("--sample_size", type=int, default=None,
                         help="explicit number of training rows to use")
    parser.add_argument("--use_transformer", action="store_true",
                         help="also try the optional transformer baseline "
                              "(needs internet access to Hugging Face Hub)")
    parser.add_argument("--use_llm", action="store_true",
                         help="also try the optional LLM few-shot baseline "
                              "(needs ANTHROPIC_API_KEY and internet access)")
    parser.add_argument("--llm_sample_size", type=int, default=150,
                         help="target rows PER CLASS to evaluate the LLM baseline "
                              "on (stratified sample; rare classes use all their "
                              "rows if fewer than this). Controls API cost. "
                              "Default: 150/class.")
    parser.add_argument("--llm_examples_per_label", type=int, default=5,
                         help="number of in-context few-shot examples per class "
                              "shown to the LLM (default: 5)")
    parser.add_argument("--llm_workers", type=int, default=8,
                         help="number of parallel API calls for the LLM baseline "
                              "(default: 8; lower this if you hit rate limits)")
    args = parser.parse_args()
    run(quick=args.quick, use_transformer=args.use_transformer,
        sample_size=args.sample_size, use_llm=args.use_llm,
        llm_sample_size=args.llm_sample_size,
        llm_examples_per_label=args.llm_examples_per_label,
        llm_workers=args.llm_workers)
