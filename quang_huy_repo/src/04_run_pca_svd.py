from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from utils import (
    compute_r2,
    compute_rmse,
    ensure_dir,
    infer_directional_dominance,
    load_config,
    parse_args,
    setup_logging,
)


def main() -> None:
    args = parse_args("Step 4: Run PCA/SVD and compute motion amplitudes")
    cfg = load_config(args.config)
    root_dir = Path(cfg["paths"]["root_dir"])
    processed_dir = ensure_dir(root_dir / cfg["paths"]["processed_dir"])
    results_dir = ensure_dir(root_dir / cfg["paths"]["results_dir"])
    tables_dir = ensure_dir(results_dir / "tables")
    logs_dir = ensure_dir(root_dir / cfg["paths"]["logs_dir"])
    setup_logging(logs_dir / "04_run_pca_svd.log")

    centered_motion = np.load(processed_dir / "motion_matrix_centered.npy")
    phase_names = cfg["phases"]["names"]
    n_components = int(cfg["pca"]["n_components"])
    calibration_units_per_mm = float(cfg["pca"]["calibration_units_per_mm"])

    # SVD decomposition of centered motion matrix X = U S Vt
    u, s, vt = np.linalg.svd(centered_motion, full_matrices=False)
    explained_variance = (s**2) / max(centered_motion.shape[0] - 1, 1)
    explained_ratio = explained_variance / explained_variance.sum()

    spatial_components = vt[:n_components, :]
    temporal_coefficients = u[:, :n_components] * s[:n_components]

    # Low-rank reconstruction for validation metrics
    reconstruction = temporal_coefficients @ spatial_components
    rmse = compute_rmse(centered_motion, reconstruction)
    r2 = compute_r2(centered_motion, reconstruction)

    amplitudes_units = []
    amplitudes_mm = []
    dominant_directions = []
    for i in range(n_components):
        coeff = temporal_coefficients[:, i]
        amplitude_units = 0.5 * (np.max(coeff) - np.min(coeff))
        amplitude_mm = amplitude_units / calibration_units_per_mm
        amplitudes_units.append(float(amplitude_units))
        amplitudes_mm.append(float(amplitude_mm))
        dominant_directions.append(infer_directional_dominance(spatial_components[i]))

    component_names = [f"PC{i+1}" for i in range(n_components)]
    summary_df = pd.DataFrame(
        {
            "component": component_names,
            "explained_variance_ratio": explained_ratio[:n_components],
            "cumulative_explained_variance_ratio": np.cumsum(explained_ratio[:n_components]),
            "dominant_direction": dominant_directions,
            "amplitude_units": amplitudes_units,
            "amplitude_mm": amplitudes_mm,
        }
    )
    summary_df.to_csv(tables_dir / "pca_summary.csv", index=False)

    coeff_df = pd.DataFrame(temporal_coefficients[:, :n_components], columns=component_names)
    coeff_df.insert(0, "phase", phase_names)
    coeff_df.to_csv(tables_dir / "temporal_coefficients.csv", index=False)

    np.save(processed_dir / "singular_values.npy", s)
    np.save(processed_dir / "spatial_components.npy", spatial_components)
    np.save(processed_dir / "temporal_coefficients.npy", temporal_coefficients)
    np.save(processed_dir / "explained_variance_ratio.npy", explained_ratio)

    metrics_df = pd.DataFrame(
        [{"rmse_reconstruction": rmse, "r2_reconstruction": r2, "n_components": n_components}]
    )
    metrics_df.to_csv(tables_dir / "reconstruction_metrics.csv", index=False)

    logging.info("Saved PCA summary to %s", tables_dir / "pca_summary.csv")
    logging.info("Low-rank reconstruction RMSE=%.6f, R2=%.6f", rmse, r2)
    logging.info("PCA/SVD step complete")


if __name__ == "__main__":
    main()
