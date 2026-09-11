"""Render the accuracy-against-parameters figure used in the tutorial notebook.

The notebook embeds this chart as a static image rather than plotting it inline. Run this
script from anywhere to regenerate `accuracy_vs_parameters.png` after the shipped CSVs change:

    python examples/semg_gesture/img/make_accuracy_plot.py
"""

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXAMPLE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = Path(__file__).resolve().parent / "accuracy_vs_parameters.png"

# Trainable parameters of the TENNs-R model the notebook builds.
MODEL_PARAMETERS = 96_128

PALETTE = {
    "ours": "#0b6bcb",
    "other": "#9aa7b4",
    "grid": "#d8dde3",
}


def load_csv(path):
    """Read a small CSV into a list of dicts, keeping the column order of the header.

    Args:
        path: Path to the CSV file.

    Returns:
        One dict per data row, keyed by column name.
    """
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def bubble_area(params):
    """Marker area in points squared, proportional to the parameter count."""
    return params / 600.0


def label_offset(params):
    """Points below a bubble's centre for its label, clear of the bubble itself."""
    return -(np.sqrt(bubble_area(params) / np.pi) + 16)


def plot_accuracy_versus_parameters(published, ours_params, ours_accuracy):
    """Plot published DB2 accuracy against parameter count, with this model highlighted.

    Args:
        published: rows from the shipped literature table.
        ours_params: trainable parameters of the TENNs-R model.
        ours_accuracy: its int8 cohort accuracy in percent.

    Returns:
        The matplotlib figure, ready to save.
    """
    figure, axes = plt.subplots(figsize=(11, 6.4))
    # Labels sit under their bubble, except where a neighbouring bubble reaches into the space
    # below this one: MvCNN under MSDS-FusionNet, and TraHGR-Huge under NKDFF-CNN.
    label_left = {"TraHGR-large", "MSDS-FusionNet", "NKDFF-CNN"}
    for row in published:
        if not row["params"]:
            continue
        params, accuracy = float(row["params"]), float(row["accuracy"])
        marker = "*" if row["n_classes"] != "49" else ""
        axes.scatter(params, accuracy, s=bubble_area(params), color=PALETTE["other"],
                     alpha=0.75, edgecolor="white", linewidth=1.2, zorder=2)
        caption = f"{row['model']}{marker}\n{params / 1000:,.0f}k"
        if row["model"] in label_left:
            axes.annotate(caption, xy=(params, accuracy), xytext=(label_offset(params), 0),
                          textcoords="offset points", ha="right", va="center", fontsize=9,
                          color="#4a5560")
        else:
            axes.annotate(caption, xy=(params, accuracy), xytext=(0, label_offset(params)),
                          textcoords="offset points", ha="center", va="top", fontsize=9,
                          color="#4a5560")

    axes.scatter(ours_params, ours_accuracy, s=900, facecolor="none",
                 edgecolor=PALETTE["ours"], linewidth=1.4, alpha=0.45, zorder=3)
    axes.scatter(ours_params, ours_accuracy, s=bubble_area(ours_params),
                 color=PALETTE["ours"], edgecolor="white", linewidth=1.2, zorder=4)
    axes.annotate(f"TENNs-R int8\n{ours_params / 1000:,.0f}k parameters",
                  xy=(ours_params, ours_accuracy), xytext=(22, 24), textcoords="offset points",
                  fontsize=11, fontweight="bold", color=PALETTE["ours"])

    axes.set_xscale("log")
    axes.set_xlabel("trainable parameters (log scale)")
    axes.set_ylabel("subject-averaged accuracy, 49 gestures (%)")
    axes.set_title("NinaPro DB2, all 40 subjects, train repetitions 1/3/4/6, test 2/5")
    axes.text(0.99, 0.02, "bubble area scales with parameter count.  "
              "* scored over 50 classes rather than 49.",
              transform=axes.transAxes, ha="right", fontsize=8.5, color="#6b7580")
    axes.set_ylim(74.5, 93)
    axes.set_xlim(4e4, 1.2e7)
    figure.tight_layout()
    return figure


def cohort_int8_accuracy():
    """Return the int8 accuracy averaged over the 40 subjects in the shipped cohort table."""
    cohort = load_csv(EXAMPLE_DIR / "results" / "exp12_cohort.csv")
    return float(np.mean([float(row["accuracy_int8"]) for row in cohort]))


def main():
    """Render the figure from the shipped CSVs and write it next to this script."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#8b959f",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.6,
        "font.size": 10,
        "legend.frameon": False,
    })
    published = load_csv(EXAMPLE_DIR / "results" / "literature_db2.csv")
    figure = plot_accuracy_versus_parameters(published, MODEL_PARAMETERS, cohort_int8_accuracy())
    figure.savefig(OUTPUT_PATH, dpi=150)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
