"""
visualize_results.py
--------------------
Generate performance comparison charts from classifier run logs.

Inputs (default):
  - logs/compute_results.txt  (Naive Bayes output)
  - logs/nr_results.txt       (Neural network + NB comparison output)

Outputs:
  - logs/plots/accuracy_comparison.png
  - logs/plots/confusion_matrices.png
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


LABEL_TO_IDX = {"spam": 0, "ham": 1}
IDX_TO_LABEL = {0: "spam", 1: "ham"}


def read_text(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing log file: {path}")
    with open(path, encoding="utf-8") as f:
        return f.read()


def extract_nb_accuracies(text: str) -> Dict[str, float]:
    map_match = re.search(r"MAP accuracy\s*:\s*\d+/\d+\s*\((\d+)%\)", text)
    ml_match = re.search(r"ML\s+accuracy\s*:\s*\d+/\d+\s*\((\d+)%\)", text)

    if not map_match or not ml_match:
        raise ValueError("Could not parse MAP/ML accuracies from compute_results log.")

    return {
        "NB-MAP": float(map_match.group(1)),
        "NB-ML": float(ml_match.group(1)),
    }


def extract_nn_accuracy(text: str) -> float:
    nn_match = re.search(r"Single Neuron\s*:\s*\d+/\d+\s*\((\d+)%\)", text)
    if not nn_match:
        raise ValueError("Could not parse Single Neuron accuracy from nr_results log.")
    return float(nn_match.group(1))


def parse_comparison_rows(text: str) -> List[Tuple[str, str, str, str]]:
    """
    Parse rows from the Task 3 comparison table in nr_results.txt.

    Expected columns:
      No.  True   x_ham   NB-MAP   NB-ML   Neuron   Email
    """
    rows: List[Tuple[str, str, str, str]] = []
    pattern = re.compile(
        r"^\s*\d+\s+(spam|ham)\s+[0-9.]+\s+(spam|ham)\s+(spam|ham)\s+(spam|ham)\s+",
        re.IGNORECASE,
    )

    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            true_label = match.group(1).lower()
            nb_map = match.group(2).lower()
            nb_ml = match.group(3).lower()
            neuron = match.group(4).lower()
            rows.append((true_label, nb_map, nb_ml, neuron))

    if not rows:
        raise ValueError("No comparison rows were parsed from nr_results log.")

    return rows


def build_confusion(rows: List[Tuple[str, str, str, str]], model_idx: int) -> List[List[int]]:
    """
    model_idx: 1 -> NB-MAP, 2 -> NB-ML, 3 -> Neuron
    Returns confusion matrix as:
      [[true_spam_pred_spam, true_spam_pred_ham],
       [true_ham_pred_spam,  true_ham_pred_ham]]
    """
    cm = [[0, 0], [0, 0]]
    for row in rows:
        true_label = row[0]
        pred_label = row[model_idx]
        i = LABEL_TO_IDX[true_label]
        j = LABEL_TO_IDX[pred_label]
        cm[i][j] += 1
    return cm


def plot_accuracy(accuracies: Dict[str, float], output_path: str) -> None:
    names = list(accuracies.keys())
    values = [accuracies[k] for k in names]
    colors = ["#1f77b4", "#2ca02c", "#d62728"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, values, color=colors[: len(names)])
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.title("Classifier Accuracy Comparison")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _draw_confusion(ax, cm: List[List[int]], title: str) -> None:
    ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_title(title)
    ax.set_xticks([0, 1], ["pred spam", "pred ham"])
    ax.set_yticks([0, 1], ["true spam", "true ham"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", color="black")


def plot_confusions(rows: List[Tuple[str, str, str, str]], output_path: str) -> None:
    cm_map = build_confusion(rows, 1)
    cm_ml = build_confusion(rows, 2)
    cm_nn = build_confusion(rows, 3)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    _draw_confusion(axes[0], cm_map, "NB-MAP")
    _draw_confusion(axes[1], cm_ml, "NB-ML")
    _draw_confusion(axes[2], cm_nn, "Single Neuron")

    fig.suptitle("Confusion Matrices (Test Set)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize classifier comparison results")
    parser.add_argument("--nb-log", default="logs/compute_results.txt")
    parser.add_argument("--nn-log", default="logs/nr_results.txt")
    parser.add_argument("--out-dir", default="logs/plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    nb_text = read_text(args.nb_log)
    nn_text = read_text(args.nn_log)

    accuracies = extract_nb_accuracies(nb_text)
    accuracies["Single Neuron"] = extract_nn_accuracy(nn_text)

    rows = parse_comparison_rows(nn_text)

    acc_path = os.path.join(args.out_dir, "accuracy_comparison.png")
    conf_path = os.path.join(args.out_dir, "confusion_matrices.png")

    plot_accuracy(accuracies, acc_path)
    plot_confusions(rows, conf_path)

    print("Generated visualizations:")
    print(f"  - {acc_path}")
    print(f"  - {conf_path}")
    print("Accuracy summary:")
    for name, value in accuracies.items():
        print(f"  {name:<14}: {value:.1f}%")


if __name__ == "__main__":
    main()
