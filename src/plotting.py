"""
Shared plotting helpers used by experiments/exp1, exp2, and run_all.

Kept dependency-light (matplotlib only) and saved as static PNGs under
results/figures/, so figures are reviewable without re-running anything.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, labels, title, out_path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_domain_shift_ladder(accuracies, out_path, title="Accuracy across domain shift",
                              model_order=None):
    """Grouped bar chart: one group per evaluation point (e.g. GoEmotions /
    ISEAR / Vignettes), one bar per model within each group.

    accuracies: dict like
        {"GoEmotions\n(in-domain)": {"nrc_lexicon": 0.38, "tfidf_logreg": 0.60, ...},
         "ISEAR\n(cross-dataset)": {...},
         "Vignettes\n(near-domain)": {...}}
    All inner dicts should share the same model keys for a clean chart;
    missing values are skipped for that model/group.

    model_order: optional explicit list controlling bar/legend order (e.g.
    grouped by method family for a publication figure). Defaults to
    discovery order across eval points if not given.
    """
    eval_points = list(accuracies.keys())
    if model_order:
        model_names = [m for m in model_order
                       if any(m in d for d in accuracies.values())]
    else:
        model_names = []
        for d in accuracies.values():
            for k in d:
                if k not in model_names:
                    model_names.append(k)

    n_models = len(model_names)
    x = np.arange(len(eval_points))
    width = 0.8 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, model in enumerate(model_names):
        vals = [accuracies[ep].get(model, np.nan) for ep in eval_points]
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=model)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(eval_points)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ablation_comparison(means, sems, out_path,
                              title="Message relevance by personalization level",
                              ylabel="TF-IDF similarity to target need"):
    """Bar chart with error bars comparing 4 (or more) ablation levels.

    means/sems: dicts {level_name: value}, same keys, in display order.
    """
    levels = list(means.keys())
    vals = [means[k] for k in levels]
    errs = [sems.get(k, 0) for k in levels]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(levels, vals, yerr=errs, capsize=4,
                   color=["#adb5bd", "#74c69d", "#52b788", "#2d6a4f"][:len(levels)])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(errs, default=0) + 0.002,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels, rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_human_rating_staircase(means_by_level, out_path,
                                 title="Human expert ratings by personalization level",
                                 dimensions=None, n_raters=None):
    """Grouped bar chart: one group per rating dimension, one bar per
    personalization level within each group. means_by_level:
    {level: {dimension: mean_1_to_5}}.
    """
    levels = list(means_by_level.keys())
    dims = dimensions or list(next(iter(means_by_level.values())).keys())
    x = np.arange(len(dims))
    width = 0.8 / max(len(levels), 1)
    colors = ["#adb5bd", "#74c69d", "#52b788", "#2d6a4f"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, lvl in enumerate(levels):
        vals = [means_by_level[lvl][d] for d in dims]
        offset = (i - (len(levels) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=lvl, color=colors[i % len(colors)])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n") for d in dims])
    ax.set_ylabel("Mean rating (1-5 scale)")
    ax.set_ylim(1, 5.6)
    subtitle = f" (n={n_raters} raters)" if n_raters else ""
    ax.set_title(title + subtitle, fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_paired_comparison(values_a, values_b, label_a, label_b, out_path,
                            title="Generic vs personalized message relevance",
                            ylabel="TF-IDF similarity to target need"):
    """Dumbbell/slope chart: one line per paired observation (e.g. per
    vignette), connecting its value_a -> value_b, plus the two group means.
    Makes a paired, significant difference visually obvious instead of
    just reporting a p-value.
    """
    n = len(values_a)
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    for i in range(n):
        color = "#2a9d8f" if values_b[i] >= values_a[i] else "#e76f51"
        ax.plot([0, 1], [values_a[i], values_b[i]], color=color, alpha=0.35,
                linewidth=1, zorder=1)
    ax.scatter([0] * n, values_a, color="#888888", s=18, zorder=2)
    ax.scatter([1] * n, values_b, color="#888888", s=18, zorder=2)

    mean_a, mean_b = float(np.mean(values_a)), float(np.mean(values_b))
    ax.plot([0, 1], [mean_a, mean_b], color="black", linewidth=2.5, zorder=3,
             marker="o", markersize=7)
    ax.annotate(f"mean={mean_a:.3f}", (0, mean_a), textcoords="offset points",
                xytext=(-10, 8), ha="right", fontsize=9, fontweight="bold")
    ax.annotate(f"mean={mean_b:.3f}", (1, mean_b), textcoords="offset points",
                xytext=(10, 8), ha="left", fontsize=9, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_xticklabels([label_a, label_b])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, wrap=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
