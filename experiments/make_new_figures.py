import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/home/claude/work/ei-faculty-motivation/results/figures"

# ---------------------------------------------------------------
# Figure: effect-size staircase (rater-level, n=21, primary analysis)
# ---------------------------------------------------------------
with open("/home/claude/work/ei-faculty-motivation/results/rater_level_staircase.json") as f:
    rater = json.load(f)

DIMS = ["relevance", "specificity", "sdt_alignment", "motivational_usefulness"]
DIM_LABELS = ["Relevance", "Specificity", "SDT alignment", "Motivational\nusefulness"]
STEPS = ["generic_vs_emotion_only", "emotion_only_vs_need_only", "need_only_vs_full_context"]
STEP_LABELS = ["generic →\nemotion_only", "emotion_only →\nneed_only", "need_only →\nfull_context"]

colors = ["#03045e", "#0077b6", "#00b4d8", "#90e0ef"]

d_vals = np.zeros((len(DIMS), len(STEPS)))
p_vals = np.zeros((len(DIMS), len(STEPS)))
for i, dim in enumerate(DIMS):
    for j, step in enumerate(STEPS):
        rec = rater["staircase"][dim][step]
        d_vals[i, j] = rec["cohens_d"]
        p_vals[i, j] = rec["paired_ttest_p"]

fig, ax = plt.subplots(figsize=(8, 5))
n_dims = len(DIMS)
width = 0.18
x = np.arange(len(STEPS))
for i, dim in enumerate(DIMS):
    offset = (i - (n_dims - 1) / 2) * width
    bars = ax.bar(x + offset, d_vals[i], width, label=DIM_LABELS[i], color=colors[i])
    for j, bar in enumerate(bars):
        h = bar.get_height()
        marker = "***" if p_vals[i, j] < 0.001 else ("**" if p_vals[i, j] < 0.01 else
                  ("*" if p_vals[i, j] < 0.05 else "n.s."))
        ax.annotate(marker, (bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7)

ax.axhline(0, color="black", linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels(STEP_LABELS)
ax.set_ylabel("Cohen's d (rater-level paired difference, n=21)")
ax.set_title("Effect size of each staircase step, by rating dimension")
ax.legend(loc="upper left", fontsize=8, ncol=2)
ax.set_ylim(-0.1, 0.85)
fig.tight_layout()
fig.savefig(f"{OUT}/human_rubric_effect_size_staircase.png", dpi=200)
print("Saved effect-size staircase figure.")

# ---------------------------------------------------------------
# Figure: stressor-category distribution of the 75 vignettes
# ---------------------------------------------------------------
import csv
from collections import Counter

counts = Counter()
with open("/home/claude/work/ei-faculty-motivation/data/vignettes/faculty_vignettes.csv", newline="") as f:
    for row in csv.DictReader(f):
        counts[row["stressor_category"]] += 1

items = sorted(counts.items(), key=lambda kv: kv[1])
labels = [k.replace("_", " ").capitalize() for k, v in items]
values = [v for k, v in items]

fig2, ax2 = plt.subplots(figsize=(8, 4.5))
bars = ax2.barh(labels, values, color="#2d6a4f")
for bar, v in zip(bars, values):
    ax2.annotate(str(v), (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                 xytext=(4, 0), textcoords="offset points", va="center", fontsize=9)
ax2.set_xlabel("Number of vignettes (of 75)")
ax2.set_title("Stressor-category distribution of the 75 vignettes", fontsize=13)
fig2.tight_layout()
fig2.savefig(f"{OUT}/vignette_stressor_distribution.png", dpi=200)
print("Saved stressor-category distribution figure.")
