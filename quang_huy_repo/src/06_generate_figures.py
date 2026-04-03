from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import ensure_dir, load_config, parse_args, setup_logging


def main() -> None:
    args = parse_args("Step 6: Generate reviewer-check figures and summary visuals")
    cfg = load_config(args.config)
    root_dir = Path(cfg["paths"]["root_dir"])
    results_dir = ensure_dir(root_dir / cfg["paths"]["results_dir"])
    figures_dir = ensure_dir(results_dir / "figures")
    tables_dir = ensure_dir(results_dir / "tables")
    logs_dir = ensure_dir(root_dir / cfg["paths"]["logs_dir"])
    setup_logging(logs_dir / "06_generate_figures.log")

    pca_summary = pd.read_csv(tables_dir / "pca_summary.csv")
    coeff_df = pd.read_csv(tables_dir / "temporal_coefficients.csv")
    explained_ratio = pca_summary["explained_variance_ratio"].to_numpy()

    # Figure 1: Explained variance
    fig = plt.figure(figsize=(6, 4))
    x = np.arange(1, len(explained_ratio) + 1)
    plt.bar(x, explained_ratio)
    plt.plot(x, np.cumsum(explained_ratio), marker="o")
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance ratio")
    plt.title("Explained variance and cumulative contribution")
    plt.tight_layout()
    fig.savefig(figures_dir / "explained_variance.png", dpi=300)
    plt.close(fig)

    # Figure 2: Temporal coefficients
    fig = plt.figure(figsize=(7, 4))
    phase_labels = coeff_df["phase"].tolist()
    x = np.arange(len(phase_labels))
    for col in coeff_df.columns[1:]:
        plt.plot(x, coeff_df[col].to_numpy(), marker="o", label=col)
    plt.xticks(x, phase_labels, rotation=45)
    plt.xlabel("Respiratory phase")
    plt.ylabel("Temporal coefficient (a.u.)")
    plt.title("Temporal evolution of retained PCA components")
    plt.legend()
    plt.tight_layout()
    fig.savefig(figures_dir / "temporal_coefficients.png", dpi=300)
    plt.close(fig)

    # Figure 3: Calibrated amplitudes in mm
    fig = plt.figure(figsize=(6, 4))
    labels = [f"{c}\n({d})" for c, d in zip(pca_summary["component"], pca_summary["dominant_direction"])]
    plt.bar(labels, pca_summary["amplitude_mm"].to_numpy())
    plt.ylabel("Amplitude (mm)")
    plt.title("Calibrated motion amplitudes")
    plt.tight_layout()
    fig.savefig(figures_dir / "amplitude_barplot.png", dpi=300)
    plt.close(fig)

    logging.info("Generated figures in %s", figures_dir)


if __name__ == "__main__":
    main()
