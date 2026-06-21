"""
Experiment 3: intervention message quality.

Two conditions are compared:
  - "oracle"     — the emotion fed into the SDT-need mapping is the GOLD
                   label (data/vignettes/faculty_vignettes.csv). This is an
                   upper bound: "if emotion detection were perfect, would
                   personalization help?" It does NOT by itself demonstrate
                   the full pipeline works end-to-end.
  - "end_to_end" — the emotion fed in is the PREDICTED label from the best
                   available classifier (llm_fewshot's predictions saved by
                   Experiment 2, results/exp2_predictions.csv). This shows
                   how classification errors propagate into the final
                   message. Skipped with a clear message if Exp.2 has not
                   been run with --use_llm yet.

For both conditions, generates:
  (a) GENERIC_MESSAGE   — the same non-personalized message every time
  (b) personalized      — generated from the (oracle or predicted)
                           emotion's SDT need (src/sdt_mapping.py +
                           src/intervention_generator.py)

and scores relevance to the actual underlying need (always the GOLD need,
since that's the real target regardless of what was predicted) using
TF-IDF cosine similarity against a short canonical description of that
need (data/vignettes/need_descriptions.csv). This is a fully automatic,
offline relevance proxy, not a measure of real motivational impact.

Also exports a BLIND A/B rubric (results/exp3_rubric_blind.csv) so a small
panel of reviewers can manually score relevance / specificity /
appropriateness without knowing which message is generic vs personalized
(de-anonymization key kept separately in results/exp3_rubric_key.csv - do
NOT share that file with raters). Analyze filled-in rubrics with
experiments/exp3b_rubric_analysis.py.

Usage:
    python experiments/exp3_intervention_quality.py
    python experiments/exp3_intervention_quality.py --mode llm
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
from src.plotting import plot_paired_comparison
from src.sdt_mapping import get_need

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
VIGNETTES_PATH = os.path.join(REPO_ROOT, "data", "vignettes", "faculty_vignettes.csv")
NEED_DESC_PATH = os.path.join(REPO_ROOT, "data", "vignettes", "need_descriptions.csv")
EXP2_PREDICTIONS_PATH = os.path.join(RESULTS_DIR, "exp2_predictions.csv")

SEED = 42


def load_csv_as_dicts(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _paired_tests(a, b):
    t_stat, t_p = stats.ttest_rel(b, a)
    w_stat, w_p = stats.wilcoxon(b, a)
    return {"t": t_stat, "p": t_p}, {"stat": w_stat, "p": w_p}


def run(mode="template"):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    random.seed(SEED)

    vignettes = load_csv_as_dicts(VIGNETTES_PATH)
    need_desc = {r["need"]: r["description"] for r in load_csv_as_dicts(NEED_DESC_PATH)}

    # try to load the best classifier's predictions for the end-to-end condition
    predicted_emotion = {}
    if os.path.exists(EXP2_PREDICTIONS_PATH):
        pred_rows = load_csv_as_dicts(EXP2_PREDICTIONS_PATH)
        if pred_rows and "llm_fewshot_pred" in pred_rows[0]:
            predicted_emotion = {r["id"]: r["llm_fewshot_pred"] for r in pred_rows}
    have_e2e = len(predicted_emotion) > 0

    rows = []
    for v in vignettes:
        gold_emotion = v["expected_emotion"]
        gold_need = get_need(gold_emotion)
        gold_desc = need_desc.get(gold_need, "")
        personalized_oracle = generate_message(gold_emotion, gold_need, mode=mode)

        row = {
            "id": v["id"],
            "gold_emotion": gold_emotion,
            "gold_need": gold_need,
            "vignette_text": v["text"],
            "generic_message": GENERIC_MESSAGE,
            "personalized_message_oracle": personalized_oracle,
            "need_description": gold_desc,
        }

        if have_e2e:
            pred_emotion = predicted_emotion.get(v["id"], gold_emotion)
            pred_need = get_need(pred_emotion)
            row["predicted_emotion"] = pred_emotion
            row["predicted_need"] = pred_need
            row["need_match"] = (pred_need == gold_need)
            # NOTE: end-to-end message is generated from the PREDICTED need,
            # but scored against the GOLD need's description, since that is
            # the person's real underlying need regardless of what the
            # classifier thought.
            row["personalized_message_e2e"] = generate_message(
                pred_emotion, pred_need, mode=mode)

        rows.append(row)

    references = [r["need_description"] for r in rows]
    sim_generic = pairwise_tfidf_similarity([r["generic_message"] for r in rows], references)
    sim_oracle = pairwise_tfidf_similarity(
        [r["personalized_message_oracle"] for r in rows], references)
    for r, sg, so in zip(rows, sim_generic, sim_oracle):
        r["sim_generic"] = sg
        r["sim_personalized_oracle"] = so

    summary = {"n": len(rows), "mode": mode}

    ttest_go, wil_go = _paired_tests(sim_generic, sim_oracle)
    summary["mean_sim_generic"] = sum(sim_generic) / len(sim_generic)
    summary["mean_sim_personalized_oracle"] = sum(sim_oracle) / len(sim_oracle)
    summary["oracle_vs_generic"] = {"paired_ttest": ttest_go, "wilcoxon": wil_go}

    if have_e2e:
        sim_e2e = pairwise_tfidf_similarity(
            [r["personalized_message_e2e"] for r in rows], references)
        for r, se in zip(rows, sim_e2e):
            r["sim_personalized_e2e"] = se
        need_match_rate = sum(r["need_match"] for r in rows) / len(rows)

        ttest_ge, wil_ge = _paired_tests(sim_generic, sim_e2e)
        ttest_oe, wil_oe = _paired_tests(sim_e2e, sim_oracle)
        summary["mean_sim_personalized_e2e"] = sum(sim_e2e) / len(sim_e2e)
        summary["need_match_rate"] = need_match_rate
        summary["e2e_vs_generic"] = {"paired_ttest": ttest_ge, "wilcoxon": wil_ge}
        summary["oracle_vs_e2e_degradation"] = {"paired_ttest": ttest_oe, "wilcoxon": wil_oe}
    else:
        summary["e2e_skipped_reason"] = (
            "results/exp2_predictions.csv not found or has no 'llm_fewshot_pred' "
            "column. Run experiments/exp2_domain_shift.py --use_llm first to "
            "enable the end-to-end (predicted-emotion) comparison."
        )

    out_csv = os.path.join(RESULTS_DIR, "exp3_outputs.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    _write_blind_rubric(rows, have_e2e)

    summary_path = os.path.join(RESULTS_DIR, "exp3_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_paired_comparison(
        sim_generic, sim_oracle, "generic", "personalized\n(oracle emotion)",
        os.path.join(FIGURES_DIR, "exp3_generic_vs_oracle.png"),
        title="Message relevance: generic vs personalized (oracle emotion)")
    if have_e2e:
        plot_paired_comparison(
            sim_generic, sim_e2e, "generic", "personalized\n(predicted emotion)",
            os.path.join(FIGURES_DIR, "exp3_generic_vs_e2e.png"),
            title="Message relevance: generic vs personalized (end-to-end)")

    print(f"n = {summary['n']}")
    print(f"mean similarity: generic={summary['mean_sim_generic']:.3f}  "
          f"personalized(oracle)={summary['mean_sim_personalized_oracle']:.3f}")
    print(f"  oracle vs generic: paired t-test p={ttest_go['p']:.4f}, "
          f"Wilcoxon p={wil_go['p']:.4f}")
    if have_e2e:
        print(f"  personalized(end-to-end)={summary['mean_sim_personalized_e2e']:.3f}  "
              f"(need_match_rate={need_match_rate:.1%})")
        print(f"  end-to-end vs generic: paired t-test p={ttest_ge['p']:.4f}, "
              f"Wilcoxon p={wil_ge['p']:.4f}")
        print(f"  oracle vs end-to-end (degradation from classifier errors): "
              f"paired t-test p={ttest_oe['p']:.4f}")
    else:
        print(f"  [end-to-end mode skipped] {summary['e2e_skipped_reason']}")

    print(f"\nSaved per-vignette outputs to {out_csv}")
    print(f"Saved blind rubric + key to {RESULTS_DIR}")
    print(f"Saved summary statistics to {summary_path}")
    print(f"Saved figures to {FIGURES_DIR}")
    return summary


def _write_blind_rubric(rows, have_e2e):
    """Write a BLIND rubric (message identity randomized as A/B, no labels
    revealing which is generic/personalized) plus a separate key file that
    is NOT meant to be shared with raters. If end-to-end messages are
    available, the rubric uses those (the realistic, deployable messages);
    otherwise it falls back to the oracle messages.
    """
    rng = random.Random(SEED + 1)
    rubric_rows = []
    key_rows = []
    for r in rows:
        personalized = r.get("personalized_message_e2e", r["personalized_message_oracle"])
        a_is_personalized = rng.random() < 0.5
        message_a = personalized if a_is_personalized else r["generic_message"]
        message_b = r["generic_message"] if a_is_personalized else personalized
        rubric_rows.append({
            "id": r["id"], "message_a": message_a, "message_b": message_b,
            "relevance_a_1_5": "", "relevance_b_1_5": "",
            "specificity_a_1_5": "", "specificity_b_1_5": "",
            "appropriateness_a_1_5": "", "appropriateness_b_1_5": "",
        })
        key_rows.append({"id": r["id"], "a_is_personalized": a_is_personalized,
                          "message_source": "end_to_end" if have_e2e else "oracle"})

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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["template", "llm"], default="template",
                         help="'llm' requires ANTHROPIC_API_KEY and internet access")
    args = parser.parse_args()
    run(mode=args.mode)
