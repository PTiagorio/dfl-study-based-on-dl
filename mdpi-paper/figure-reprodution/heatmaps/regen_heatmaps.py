#!/usr/bin/env python3
"""
Regenerate the three threshold-sensitivity heatmaps at the scale used by the
rest of the figures in the manuscript.

Content and layout are identical to heatmap_threshold_script_final.py.
Only presentation changes:
  - in-cell font sizes compensated for the on-page reduction to 0.9\textwidth
  - FPR and FNR on separate lines so they fit the cell at the larger size
  - successful cells additionally marked with a darker outline, so the
    distinction does not rely on colour alone (greyscale / CVD safety)
"""

import re
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

CLEAN_FILE = "9_nodes_unpoisoned.txt"
POISONED_FILE = "9_nodes_4_poisoned_quantization.txt"
N_NODES = 9

F1_THRESHOLDS = [0.40, 0.50, 0.60, 0.70, 0.80]
MAJORITY_THRESHOLDS = [0.40, 0.50, 0.60]
MAJORITY_THRESHOLDS_PRINT = ["4/9", "5/9", "6/9"]

CLEAN_POISONED_MODELS = set()
ATTACKED_POISONED_MODELS = {0, 1, 2, 3}

# ---------------------------------------------------------------
# Scale compensation
#
# The figure is 6.8 in wide and is included at 0.9\textwidth.
# MDPI textwidth = 394.35522 pt = 5.4573 in -> 4.9116 in on the page.
# ---------------------------------------------------------------
TEXTWIDTH_IN = 394.35522 / 72.27
FIG_W, FIG_H = 6.8, 4.9
SCALE = FIG_W / (0.9 * TEXTWIDTH_IN)          # ~1.384

FS_LABEL = 8.0 * SCALE                         # renders at 8.0 pt
FS_TICK = 7.5 * SCALE                          # renders at 7.5 pt
FS_MAIN = 9.0 * SCALE                          # renders at 9.0 pt
FS_METRIC = 7.5 * SCALE                        # renders at 7.5 pt
DPI = 600


def parse_evaluation_file(filename, n_nodes=9):
    """matrix[i, j] = weighted F1 obtained when evaluator i evaluates model j."""
    matrix = np.full((n_nodes, n_nodes), np.nan)
    current_evaluator = None
    evaluator_pattern = re.compile(r"^MODEL\s+(\d+):")
    score_pattern = re.compile(
        r"Model\s+(\d+)\s+(?:passed|failed)\s+with\s+"
        r"Weighted Avg F1-Score of\s+([0-9]*\.?[0-9]+)"
    )
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            m = evaluator_pattern.match(line)
            if m:
                current_evaluator = int(m.group(1))
                continue
            s = score_pattern.search(line)
            if s and current_evaluator is not None:
                matrix[current_evaluator, int(s.group(1))] = float(s.group(2))
    if np.isnan(matrix).any():
        raise ValueError(f"Unparsed evaluations in {filename}")
    return matrix


def classify_models(f1_matrix, f1_threshold, majority_threshold,
                    poisoned_models=None, collusion=False):
    n_voters = f1_matrix.shape[0]
    poisoned_models = set(poisoned_models or set())
    votes_matrix = (f1_matrix >= f1_threshold).copy()
    if collusion:
        for evaluator in poisoned_models:
            for candidate in range(f1_matrix.shape[1]):
                votes_matrix[evaluator, candidate] = candidate in poisoned_models
    votes = np.sum(votes_matrix, axis=0)
    required = math.ceil(majority_threshold * n_voters)
    return votes >= required, votes, required


def calculate_metrics(accepted, poisoned_models):
    n = len(accepted)
    poisoned_models = set(poisoned_models)
    benign = set(range(n)) - poisoned_models
    fp = sum(1 for i in benign if not accepted[i])
    fn = sum(1 for i in poisoned_models if accepted[i])
    return {
        "correct": n - fp - fn,
        "fpr": fp / len(benign) if benign else np.nan,
        "fnr": fn / len(poisoned_models) if poisoned_models else np.nan,
        "fp": fp, "fn": fn,
    }


def run_sensitivity_analysis(f1_matrix, poisoned_models, collusion=False):
    shape = (len(F1_THRESHOLDS), len(MAJORITY_THRESHOLDS))
    correct = np.zeros(shape)
    fpr = np.zeros(shape)
    fnr = np.full(shape, np.nan)
    for i, f1 in enumerate(F1_THRESHOLDS):
        for j, mj in enumerate(MAJORITY_THRESHOLDS):
            acc, _, _ = classify_models(f1_matrix, f1, mj, poisoned_models, collusion)
            m = calculate_metrics(acc, poisoned_models)
            correct[i, j], fpr[i, j], fnr[i, j] = m["correct"], m["fpr"], m["fnr"]
    return correct, fpr, fnr


def plot_detection_heatmap(correct_matrix, fpr_matrix, fnr_matrix, output_basename):
    success = (correct_matrix == N_NODES).astype(int)
    cmap = ListedColormap(["#eeeeee", "#b8cec0"])

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.imshow(success, vmin=0, vmax=1, cmap=cmap, aspect="auto",
              interpolation="nearest")

    ax.set_xticks(range(len(MAJORITY_THRESHOLDS_PRINT)))
    ax.set_xticklabels(MAJORITY_THRESHOLDS_PRINT)
    ax.set_xlabel("Consensus Threshold", fontsize=FS_LABEL)
    ax.set_yticks(range(len(F1_THRESHOLDS)))
    ax.set_yticklabels([f"{int(t * 100)}%" for t in F1_THRESHOLDS])
    ax.set_ylabel("Weighted F1-score Threshold", fontsize=FS_LABEL)
    ax.tick_params(axis="both", labelsize=FS_TICK)

    ax.set_xticks(np.arange(-0.5, len(MAJORITY_THRESHOLDS_PRINT), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(F1_THRESHOLDS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5, zorder=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(len(F1_THRESHOLDS)):
        for j in range(len(MAJORITY_THRESHOLDS)):
            correct = int(correct_matrix[i, j])
            fpr, fnr = fpr_matrix[i, j], fnr_matrix[i, j]

            has_fnr = not np.isnan(fnr)
            y_main = i - 0.20 if has_fnr else i - 0.13
            ax.text(j, y_main, f"{correct}/{N_NODES}", ha="center", va="center",
                    fontsize=FS_MAIN, fontweight="bold", color="black", zorder=5)
            ax.text(j, y_main + 0.20, f"FPR {100 * fpr:.0f}%", ha="center",
                    va="center", fontsize=FS_METRIC, color="black", zorder=5)
            if has_fnr:
                ax.text(j, y_main + 0.40, f"FNR {100 * fnr:.0f}%", ha="center",
                        va="center", fontsize=FS_METRIC, color="black", zorder=5)

            # Redundant non-colour cue for the successful cells.
            if success[i, j]:
                ax.add_patch(plt.Rectangle(
                    (j - 0.47, i - 0.47), 0.94, 0.94, fill=False,
                    edgecolor="#4a6b58", linewidth=0.9, zorder=4))

    # Operating point used throughout the main experiments: 70% F1, 5/9 consensus.
    ax.add_patch(plt.Rectangle(
        (MAJORITY_THRESHOLDS.index(0.50) - 0.5, F1_THRESHOLDS.index(0.70) - 0.5),
        1, 1, fill=False, edgecolor="black", linewidth=2.2,
        linestyle="--", zorder=10, clip_on=False))

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.35")

    fig.tight_layout()
    plt.savefig(f"{output_basename}.png", dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


clean_matrix = parse_evaluation_file(CLEAN_FILE, N_NODES)
poisoned_matrix = parse_evaluation_file(POISONED_FILE, N_NODES)

plot_detection_heatmap(
    *run_sensitivity_analysis(clean_matrix, CLEAN_POISONED_MODELS, False),
    output_basename="threshold_sensitivity_unpoisoned")

plot_detection_heatmap(
    *run_sensitivity_analysis(poisoned_matrix, ATTACKED_POISONED_MODELS, False),
    output_basename="threshold_sensitivity_4_poisoned_non_colluding")

plot_detection_heatmap(
    *run_sensitivity_analysis(poisoned_matrix, ATTACKED_POISONED_MODELS, True),
    output_basename="threshold_sensitivity_4_poisoned_colluding")

print("done")
