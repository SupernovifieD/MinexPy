#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${ROOT_DIR}/example_outputs"
mkdir -p "${OUTPUT_DIR}"

echo "Running MinexPy statistical and visualization examples..."
echo "Output directory: ${OUTPUT_DIR}"

if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Error: no usable Python interpreter found in PATH." >&2
    exit 1
  fi
fi

MINEXPY_ROOT="${ROOT_DIR}" "${PYTHON_BIN}" - <<'PY'
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from minexpy.correlation import (
    pearson_correlation,
    spearman_correlation,
    kendall_correlation,
    distance_correlation,
    biweight_midcorrelation,
    partial_correlation,
    correlation_matrix,
)
from minexpy.statviz import (
    plot_histogram,
    plot_box_violin,
    plot_ecdf,
    plot_qq,
    plot_pp,
    plot_scatter,
)

root_dir = os.environ.get("MINEXPY_ROOT", os.getcwd())
output_dir = os.path.join(root_dir, "example_outputs")
os.makedirs(output_dir, exist_ok=True)

rng = np.random.default_rng(2026)

# Synthetic geochemical-style data
depth = np.linspace(10, 280, 240)
zn = rng.lognormal(mean=2.4, sigma=0.42, size=240)
cu = 0.35 * zn + 0.04 * depth + rng.normal(0.0, 1.4, size=240)
pb = 0.18 * zn + rng.normal(0.0, 1.0, size=240)

print("\n=== Pairwise Correlations ===")
print("Pearson:", pearson_correlation(zn, cu))
print("Spearman:", spearman_correlation(zn, cu))
print("Kendall:", kendall_correlation(zn, cu))
print("Distance correlation:", distance_correlation(zn, cu))
print("Biweight midcorrelation:", biweight_midcorrelation(zn, cu))
print("Partial correlation (control=depth):", partial_correlation(zn, cu, controls=depth))

df = pd.DataFrame({"Zn": zn, "Cu": cu, "Pb": pb, "Depth": depth})

print("\n=== Correlation Matrices ===")
print("\nPearson matrix:")
print(correlation_matrix(df[["Zn", "Cu", "Pb"]], method="pearson"))
print("\nSpearman matrix:")
print(correlation_matrix(df[["Zn", "Cu", "Pb"]], method="spearman"))
print("\nDistance-correlation matrix:")
print(correlation_matrix(df[["Zn", "Cu", "Pb"]], method="distance"))


def save_current_figure(filename: str) -> None:
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# Histogram (linear and log)
plot_histogram(
    zn,
    bins=30,
    scale="linear",
    xlabel="Zn concentration (ppm)",
    ylabel="Frequency",
    title="Histogram (Linear Scale)",
)
save_current_figure("histogram_linear.png")

plot_histogram(
    zn,
    bins=30,
    scale="log",
    xlabel="Zn concentration (ppm, log scale)",
    ylabel="Frequency",
    title="Histogram (Log Scale)",
)
save_current_figure("histogram_log.png")

# Box and violin
plot_box_violin(
    df[["Zn", "Cu", "Pb"]],
    kind="box",
    ylabel="Concentration (ppm)",
    title="Box Plot",
)
save_current_figure("box_plot.png")

plot_box_violin(
    df[["Zn", "Cu", "Pb"]],
    kind="violin",
    ylabel="Concentration (ppm)",
    title="Violin Plot",
)
save_current_figure("violin_plot.png")

# ECDF
plot_ecdf(
    {"Zn": zn, "Cu": cu, "Pb": pb},
    xlabel="Concentration (ppm)",
    ylabel="Empirical cumulative probability",
    title="ECDF",
)
save_current_figure("ecdf.png")

# Q-Q and P-P
plot_qq(
    zn,
    distribution="norm",
    xlabel="Theoretical normal quantiles",
    ylabel="Zn sample quantiles",
    title="Q-Q Plot (Zn vs Normal)",
)
save_current_figure("qq_plot.png")

plot_pp(
    zn,
    distribution="norm",
    xlabel="Theoretical cumulative probability",
    ylabel="Empirical cumulative probability",
    title="P-P Plot (Zn vs Normal)",
)
save_current_figure("pp_plot.png")

# Scatter
plot_scatter(
    zn,
    cu,
    add_trendline=True,
    xlabel="Zn concentration (ppm)",
    ylabel="Cu concentration (ppm)",
    title="Scatter Plot (Zn vs Cu)",
    label="Samples",
)
save_current_figure("scatter_plot.png")

print("\nAll examples completed successfully.")
PY

echo "Done."
