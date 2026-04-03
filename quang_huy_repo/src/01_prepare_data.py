from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from utils import (
    clip_and_normalize,
    ensure_dir,
    generate_spherical_mask,
    generate_synthetic_phase_stack,
    load_config,
    load_phase_volumes,
    parse_args,
    save_json,
    setup_logging,
)


def main() -> None:
    args = parse_args("Step 1: Prepare standardized phase volumes for PCA/SVD motion analysis")
    cfg = load_config(args.config)

    root_dir = Path(cfg["paths"]["root_dir"])
    processed_dir = ensure_dir(root_dir / cfg["paths"]["processed_dir"])
    masks_dir = ensure_dir(root_dir / cfg["paths"]["masks_dir"])
    logs_dir = ensure_dir(root_dir / cfg["paths"]["logs_dir"])
    example_phase_dir = ensure_dir(root_dir / cfg["paths"]["example_patient_dir"] / "phases")

    setup_logging(logs_dir / "01_prepare_data.log")
    logging.info("Starting data preparation")

    phase_names = cfg["phases"]["names"]
    reference_phase = cfg["phases"]["reference_phase"]
    expected_shape = tuple(cfg["image"]["expected_shape"])
    clip_range = tuple(cfg["image"]["intensity_clip"])
    normalize = bool(cfg["image"]["normalize"])
    rng_seed = int(cfg.get("random_seed", 42))

    phase_files_exist = all((example_phase_dir / f"{phase}.npy").exists() for phase in phase_names)
    if phase_files_exist:
        logging.info("Found user-provided processed phase volumes. Loading from %s", example_phase_dir)
        volumes = load_phase_volumes(example_phase_dir, phase_names)
    else:
        logging.info("No phase volumes found. Generating synthetic demonstration dataset.")
        volumes = generate_synthetic_phase_stack(phase_names, expected_shape, seed=rng_seed)
        for phase, volume in volumes.items():
            np.save(example_phase_dir / f"{phase}.npy", volume)

    prepared_phase_dir = ensure_dir(processed_dir / "phases")
    for phase, volume in volumes.items():
        if volume.shape != expected_shape:
            raise ValueError(f"Unexpected shape for {phase}: {volume.shape}, expected {expected_shape}")
        prepared = clip_and_normalize(volume, clip_range, normalize)
        np.save(prepared_phase_dir / f"{phase}.npy", prepared)
        logging.info("Saved standardized phase %s with shape %s", phase, prepared.shape)

    mask_filename = cfg["roi"]["mask_filename"]
    mask_path = masks_dir / mask_filename
    if mask_path.exists():
        mask = np.load(mask_path)
        logging.info("Loaded existing ROI mask from %s", mask_path)
    else:
        radius_voxels = int(cfg["roi"]["default_radius_voxels"])
        mask = generate_spherical_mask(expected_shape, radius_voxels=radius_voxels)
        np.save(mask_path, mask)
        logging.info("Generated synthetic ROI mask and saved to %s", mask_path)

    metadata = {
        "phase_names": phase_names,
        "reference_phase": reference_phase,
        "shape": list(expected_shape),
        "spacing_mm": list(cfg["image"]["spacing_mm"]),
        "prepared_phase_dir": str(prepared_phase_dir),
        "mask_path": str(mask_path),
    }
    save_json(metadata, processed_dir / "prepared_metadata.json")
    logging.info("Preparation complete")


if __name__ == "__main__":
    main()
