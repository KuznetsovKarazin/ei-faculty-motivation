"""
Runs Experiment 1, 2 and 3 in sequence and prints a short combined summary
at the end. This is the single entry point for a full pipeline run.

Usage:
    python run_all.py            # full run (recommended once you have time)
    python run_all.py --quick    # fast smoke test with a small data subsample
    python run_all.py --use_transformer   # also attempt the transformer
                                            # baseline in Experiment 1
                                            # (needs internet access to the
                                            # Hugging Face Hub)
"""

import argparse
import sys

sys.path.insert(0, ".")

from experiments import exp1_classification, exp2_domain_shift, exp3_intervention_quality


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                         help="small data subsample, fast smoke test")
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--use_transformer", action="store_true",
                         help="also try the transformer baseline in Exp.1 "
                              "(needs internet access to Hugging Face Hub)")
    parser.add_argument("--use_llm", action="store_true",
                         help="also try the LLM few-shot baseline in Exp.1/Exp.2 "
                              "(needs ANTHROPIC_API_KEY and internet access)")
    parser.add_argument("--llm_sample_size", type=int, default=150,
                         help="target rows PER CLASS for the LLM baseline in Exp.1 "
                              "(stratified; rare classes use all available rows)")
    parser.add_argument("--llm_examples_per_label", type=int, default=5,
                         help="few-shot examples per class shown to the LLM (default: 5)")
    parser.add_argument("--llm_workers", type=int, default=8,
                         help="parallel API calls for the LLM baseline (default: 8)")
    parser.add_argument("--llm_interventions", action="store_true",
                         help="use the Anthropic API for the emotion_only/need_only "
                              "Exp.3 levels too, not just full_context "
                              "(needs ANTHROPIC_API_KEY)")
    parser.add_argument("--skip_full_context", action="store_true",
                         help="skip the full_context level in Exp.3 (saves API calls)")
    parser.add_argument("--rubric_sample_size", type=int, default=25,
                         help="vignettes included in the blind Exp.3 rubric (default: 25)")
    args = parser.parse_args()

    print("\n##### EXPERIMENT 1: emotion classification + cross-dataset #####")
    exp1_results = exp1_classification.run(quick=args.quick,
                                            use_transformer=args.use_transformer,
                                            sample_size=args.sample_size,
                                            use_llm=args.use_llm,
                                            llm_sample_size=args.llm_sample_size,
                                            llm_examples_per_label=args.llm_examples_per_label,
                                            llm_workers=args.llm_workers)

    print("\n##### EXPERIMENT 2: domain shift on faculty vignettes #####")
    exp2_results = exp2_domain_shift.run(quick=args.quick,
                                          sample_size=args.sample_size,
                                          use_llm=args.use_llm,
                                          llm_examples_per_label=args.llm_examples_per_label,
                                          llm_workers=args.llm_workers)

    print("\n##### EXPERIMENT 3: intervention message quality (4-level ablation) #####")
    mode = "llm" if args.llm_interventions else "template"
    exp3_results = exp3_intervention_quality.run(
        mode=mode, skip_full_context=args.skip_full_context,
        rubric_sample_size=args.rubric_sample_size)

    print("\n##### SUMMARY #####")
    accuracies_for_plot = {
        "GoEmotions\n(in-domain)": {},
        "ISEAR\n(cross-dataset)": {},
        "Vignettes\n(near-domain)": {},
    }
    for model in ("nrc_lexicon", "tfidf_logreg", "tfidf_logreg_5class", "llm_fewshot"):
        if model not in exp1_results or "skipped_reason" in exp1_results.get(model, {}):
            continue
        m1 = exp1_results[model]
        cross = m1["cross_dataset"]["accuracy"]
        in_dom = m1["in_domain"]["accuracy"] if "in_domain" in m1 else None
        vign = exp2_results.get(model, {}).get("overall", {}).get("accuracy")
        in_dom_str = f"{in_dom:.3f}" if in_dom is not None else "n/a (not a fair in-domain comparison)"
        vign_str = f"{vign:.3f}" if vign is not None else "n/a"
        print(f"{model:20s}  GoEmotions(in-domain)={in_dom_str}  "
              f"ISEAR(cross-dataset)={cross:.3f}  "
              f"FacultyVignettes(near-domain)={vign_str}")
        if in_dom is not None:
            accuracies_for_plot["GoEmotions\n(in-domain)"][model] = in_dom
        accuracies_for_plot["ISEAR\n(cross-dataset)"][model] = cross
        if vign is not None:
            accuracies_for_plot["Vignettes\n(near-domain)"][model] = vign

    from src.plotting import plot_domain_shift_ladder
    ladder_path = "results/figures/domain_shift_ladder.png"
    plot_domain_shift_ladder(accuracies_for_plot, ladder_path)
    print(f"\nSaved domain-shift summary chart to {ladder_path}")

    for cond in exp3_results["conditions"]:
        means = exp3_results[cond]["mean_similarity"]
        print(f"\nintervention relevance ({cond}), TF-IDF sim to need: "
              + "  ".join(f"{lvl}={v:.3f}" for lvl, v in means.items()))
        for pair, st in exp3_results[cond]["staircase_comparisons"].items():
            print(f"  {pair}: diff={st['mean_diff']:+.3f}  p={st['paired_ttest_p']:.4f}")
    print("\nAll detailed results are saved under results/")


if __name__ == "__main__":
    main()
