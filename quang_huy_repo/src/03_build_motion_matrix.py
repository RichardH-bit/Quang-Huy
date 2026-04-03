from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from utils import ensure_dir, load_config, parse_args, save_json, setup_logging


def reorder_dvf_components_to_lr_ap_si(dvf: np.ndarray) -> np.ndarray:
    """Convert SimpleITK x,y,z vector ordering to [LR, AP, SI] feature order."""
    if dvf.shape[-1] != 3:
        raise ValueError(f"DVF last dimension must be 3, got {dvf.shape}")
    # x -> LR, y -> AP, z -> SI already consistent in naming.
    return dvf.astype(np.float32)


def main() -> None:
    args = parse_args("Step 3: Construct ROI-focused motion matrix from DVFs")
    cfg = load_config(args.config)
    root_dir = Path(cfg["paths"]["root_dir"])
    processed_dir = ensure_dir(root_dir / cfg["paths"]["processed_dir"])
    logs_dir = ensure_dir(root_dir / cfg["paths"]["logs_dir"])
    setup_logging(logs_dir / "03_build_motion_matrix.log")

    dvf_dir = processed_dir / "dvfs"
    phase_names = cfg["phases"]["names"]
    reference_phase = cfg["phases"]["reference_phase"]
    mask_path = root_dir / cfg["paths"]["masks_dir"] / cfg["roi"]["mask_filename"]
    mask = np.load(mask_path).astype(bool)
    voxel_count = int(mask.sum())
    logging.info("Loaded ROI mask with %d voxels", voxel_count)

    motion_rows = []
    dvf_summary = {}
    for phase in phase_names:
        dvf = np.load(dvf_dir / f"dvf_{phase}_to_{reference_phase}.npy")
        dvf = reorder_dvf_components_to_lr_ap_si(dvf)
        roi_vectors = dvf[mask]  # shape: [n_voxels, 3]
        flattened = roi_vectors.reshape(-1)
        motion_rows.append(flattened)
        dvf_summary[phase] = {
            "mean_abs_lr": float(np.mean(np.abs(roi_vectors[:, 0]))),
            "mean_abs_ap": float(np.mean(np.abs(roi_vectors[:, 1]))),
            "mean_abs_si": float(np.mean(np.abs(roi_vectors[:, 2]))),
        }

    motion_matrix = np.vstack(motion_rows).astype(np.float32)
    centered_motion_matrix = motion_matrix - motion_matrix.mean(axis=0, keepdims=True)

    np.save(processed_dir / "motion_matrix.npy", motion_matrix)
    np.save(processed_dir / "motion_matrix_centered.npy", centered_motion_matrix)
    save_json(dvf_summary, processed_dir / "dvf_roi_summary.json")

    logging.info("Saved motion matrix with shape %s", motion_matrix.shape)
    logging.info("Motion matrix construction complete")


if __name__ == "__main__":
    main()
