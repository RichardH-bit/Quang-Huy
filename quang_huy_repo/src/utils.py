from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import yaml


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    return parser.parse_args()


def load_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def setup_logging(log_path: str | Path) -> None:
    log_path = Path(log_path)
    ensure_dir(log_path.parent)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )


def clip_and_normalize(volume: np.ndarray, clip_range: Tuple[float, float], do_normalize: bool) -> np.ndarray:
    volume = np.clip(volume.astype(np.float32), clip_range[0], clip_range[1])
    if do_normalize:
        mean = float(volume.mean())
        std = float(volume.std())
        if std > 0:
            volume = (volume - mean) / std
    return volume.astype(np.float32)


def generate_synthetic_phase_stack(
    phase_names: Iterable[str],
    shape: Tuple[int, int, int],
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate a small synthetic 4DCT-like stack for demonstration.

    The lesion moves predominantly in the superior-inferior direction with smaller
    anterior-posterior and left-right components.
    """
    rng = np.random.default_rng(seed)
    z, y, x = np.indices(shape)
    cx, cy, cz = shape[2] / 2, shape[1] / 2, shape[0] / 2

    # Background + synthetic lung noise
    base = -800 + 60 * rng.standard_normal(shape)
    base += 30 * np.sin(z / max(shape[0], 1) * 2 * np.pi)

    data: Dict[str, np.ndarray] = {}
    n_phases = len(list(phase_names))
    for idx, phase in enumerate(phase_names):
        angle = 2 * np.pi * idx / n_phases
        shift_si = 4.5 * np.sin(angle)
        shift_ap = 2.2 * np.sin(angle + 0.3)
        shift_lr = 1.8 * np.sin(angle - 0.2)

        lesion = np.exp(
            -(
                ((x - (cx + shift_lr)) ** 2) / (2 * 4.0**2)
                + ((y - (cy + shift_ap)) ** 2) / (2 * 5.0**2)
                + ((z - (cz + shift_si)) ** 2) / (2 * 3.5**2)
            )
        )
        volume = base + 900 * lesion
        data[phase] = volume.astype(np.float32)
    return data


def generate_spherical_mask(shape: Tuple[int, int, int], radius_voxels: int) -> np.ndarray:
    z, y, x = np.indices(shape)
    cz, cy, cx = shape[0] / 2, shape[1] / 2, shape[2] / 2
    dist_sq = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
    mask = (dist_sq <= radius_voxels**2).astype(np.uint8)
    return mask


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_phase_volumes(phases_dir: str | Path, phase_names: Iterable[str]) -> Dict[str, np.ndarray]:
    phases_dir = Path(phases_dir)
    volumes: Dict[str, np.ndarray] = {}
    for phase in phase_names:
        phase_path = phases_dir / f"{phase}.npy"
        if not phase_path.exists():
            raise FileNotFoundError(f"Missing phase file: {phase_path}")
        volumes[phase] = np.load(phase_path)
    return volumes


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a.ravel(), b.ravel()) / denom)


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def infer_directional_dominance(component_vector: np.ndarray) -> str:
    """Infer dominant anatomical direction assuming component vector layout repeats [LR, AP, SI]."""
    lr_energy = float(np.linalg.norm(component_vector[0::3]))
    ap_energy = float(np.linalg.norm(component_vector[1::3]))
    si_energy = float(np.linalg.norm(component_vector[2::3]))
    energies = {"L-R": lr_energy, "A-P": ap_energy, "S-I": si_energy}
    return max(energies, key=energies.get)
