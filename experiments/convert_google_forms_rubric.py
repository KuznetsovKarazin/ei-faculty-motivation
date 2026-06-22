"""
Converts a Google Forms export (.xlsx) of the blind rubric into per-rater
CSVs that experiments/exp3b_rubric_analysis.py can read directly.

Expected Google Forms column naming (one row per respondent):
    "S<n> Message <A/B/C/D> - <Dimension>"
where <n> is 1-indexed and matches the ROW ORDER of
results/exp3_rubric_blind.csv / results/exp3_rubric_key.csv (i.e. the form
was built by going through that CSV top to bottom, one section per row).
<Dimension> must be one of: Relevance, Specificity, SDT Alignment,
Motivational Usefulness (case-sensitive, matches experiments/
exp3_intervention_quality.py's RUBRIC_DIMENSIONS).

Any other columns (e.g. "Role", "Years of teaching experience", free-text
comments) are ignored - only rating columns matter here.

Usage:
    python experiments/convert_google_forms_rubric.py \\
        responses.xlsx --key results/exp3_rubric_key.csv \\
        --blind results/exp3_rubric_blind.csv \\
        --out_dir results/human_ratings

Then analyze:
    python experiments/exp3b_rubric_analysis.py \\
        --rubrics results/human_ratings/rater_*.csv \\
        --key results/exp3_rubric_key.csv
"""

import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DIM_MAP = {
    "Relevance": "relevance",
    "Specificity": "specificity",
    "SDT Alignment": "sdt_alignment",
    "Motivational Usefulness": "motivational_usefulness",
}
COLUMN_PATTERN = re.compile(r"^S(\d+) Message ([A-D]) [—-] (.+)$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xlsx_path")
    parser.add_argument("--sheet", default="Form Responses 1")
    parser.add_argument("--key", default="results/exp3_rubric_key.csv")
    parser.add_argument("--blind", default="results/exp3_rubric_blind.csv")
    parser.add_argument("--out_dir", default="results/human_ratings")
    args = parser.parse_args()

    import pandas as pd
    df = pd.read_excel(args.xlsx_path, sheet_name=args.sheet)

    matched_cols = [c for c in df.columns if COLUMN_PATTERN.match(c)]
    unmatched_cols = [c for c in df.columns if not COLUMN_PATTERN.match(c)]
    if not matched_cols:
        raise SystemExit(
            "No columns matched the expected 'S<n> Message <A-D> - <Dimension>' "
            f"pattern. Found columns: {list(df.columns)[:10]} ... "
            "Check --sheet name and column naming.")
    print(f"Matched {len(matched_cols)} rating columns, "
          f"ignoring {len(unmatched_cols)} other columns: {unmatched_cols}")

    n_scenarios = max(int(COLUMN_PATTERN.match(c).group(1)) for c in matched_cols)
    with open(args.blind, encoding="utf-8") as f:
        blind_rows = list(csv.DictReader(f))
    if len(blind_rows) != n_scenarios:
        print(f"[warning] {args.blind} has {len(blind_rows)} rows but the form "
              f"references S1..S{n_scenarios}. Make sure --blind/--key match the "
              "exact run that generated this form, or scenario numbers will be "
              "silently misaligned.")

    fieldnames = list(blind_rows[0].keys())
    os.makedirs(args.out_dir, exist_ok=True)

    for resp_idx, row in df.iterrows():
        out_rows = []
        for s_idx, base in enumerate(blind_rows, start=1):
            out_row = dict(base)
            for slot in ["A", "B", "C", "D"]:
                for excel_dim, internal_dim in DIM_MAP.items():
                    col = f"S{s_idx} Message {slot} \u2014 {excel_dim}"
                    if col not in df.columns:
                        col = f"S{s_idx} Message {slot} - {excel_dim}"
                    out_row[f"{internal_dim}_{slot.lower()}_1_5"] = row.get(col, "")
            out_rows.append(out_row)
        out_path = os.path.join(args.out_dir, f"rater_{resp_idx + 1:02d}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(out_rows)

    print(f"Wrote {len(df)} per-rater CSVs to {args.out_dir}/")
    print("Next: python experiments/exp3b_rubric_analysis.py "
          f"--rubrics {args.out_dir}/rater_*.csv --key {args.key}")


if __name__ == "__main__":
    main()
