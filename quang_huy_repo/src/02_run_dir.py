from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from utils import ensure_dir, load_config, parse_args, setup_logging


def numpy_to_sitk(volume: np.ndarray, spacing_mm: list[float]) -> sitk.Image:
    image = sitk.GetImageFromArray(volume.astype(np.float32))
    image.SetSpacing(tuple(float(v) for v in spacing_mm))
    return image


def sitk_vector_to_numpy(vector_image: sitk.Image) -> np.ndarray:
    array = sitk.GetArrayFromImage(vector_image)
    # SimpleITK outputs [z, y, x, components] with components typically in x,y,z order.
    return np.asarray(array, dtype=np.float32)


def run_demons_registration(fixed_np: np.ndarray, moving_np: np.ndarray, spacing_mm: list[float], iterations: int) -> np.ndarray:
    fixed = numpy_to_sitk(fixed_np, spacing_mm)
    moving = numpy_to_sitk(moving_np, spacing_mm)

    demons = sitk.DemonsRegistrationFilter()
    demons.SetNumberOfIterations(iterations)
    demons.SetStandardDeviations(1.0)
    displacement_field = demons.Execute(fixed, moving)
    return sitk_vector_to_numpy(displacement_field)


def identity_dvf(shape: tuple[int, int, int]) -> np.ndarray:
    return np.zeros((*shape, 3), dtype=np.float32)


def main() -> None:
    args = parse_args("Step 2: Run deformable registration and export displacement vector fields")
    cfg = load_config(args.config)
    root_dir = Path(cfg["paths"]["root_dir"])
    processed_dir = ensure_dir(root_dir / cfg["paths"]["processed_dir"])
    logs_dir = ensure_dir(root_dir / cfg["paths"]["logs_dir"])
    setup_logging(logs_dir / "02_run_dir.log")

    phase_dir = processed_dir / "phases"
    dvf_dir = ensure_dir(processed_dir / "dvfs")
    phase_names = cfg["phases"]["names"]
    reference_phase = cfg["phases"]["reference_phase"]
    spacing_mm = cfg["image"]["spacing_mm"]
    iterations = int(cfg["registration"]["iterations"])
    use_identity_fallback = bool(cfg["registration"]["use_identity_fallback_if_failed"])

    fixed_np = np.load(phase_dir / f"{reference_phase}.npy")
    shape = fixed_np.shape
    logging.info("Using %s as fixed reference phase", reference_phase)

    for phase in phase_names:
        moving_np = np.load(phase_dir / f"{phase}.npy")
        output_path = dvf_dir / f"dvf_{phase}_to_{reference_phase}.npy"
        if phase == reference_phase:
            np.save(output_path, identity_dvf(shape))
            logging.info("Saved identity DVF for reference phase %s", phase)
            continue

        try:
            dvf = run_demons_registration(fixed_np, moving_np, spacing_mm, iterations)
            np.save(output_path, dvf)
            logging.info("Saved DVF for %s -> %s with shape %s", phase, reference_phase, dvf.shape)
        except Exception as exc:  # pragma: no cover - practical fallback path
            if not use_identity_fallback:
                raise
            logging.warning(
                "Registration failed for %s due to %s. Saving identity DVF as fallback.", phase, exc
            )
            np.save(output_path, identity_dvf(shape))

    logging.info("DIR step complete")


if __name__ == "__main__":
    main()
