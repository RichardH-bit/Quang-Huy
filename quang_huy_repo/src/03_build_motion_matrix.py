# -----------------------------
# BƯỚC 3: XÂY DỰNG MA TRẬN CHUYỂN ĐỘNG
# Trích xuất DVF trong ROI → tạo ma trận PCA
# -----------------------------

import numpy as np
from pathlib import Path
import logging

from utils import load_config, setup_logging, save_json


def main():
    cfg = load_config("config.yaml")

    root = Path(cfg["paths"]["root_dir"])
    dvf_dir = root / cfg["paths"]["processed_dir"] / "dvf"
    mask = np.load(root / cfg["paths"]["masks_dir"] / cfg["roi"]["mask_filename"])

    setup_logging(root / "logs" / "03_motion_matrix.log")

    motion_vectors = []

    for file in sorted(dvf_dir.glob("*.npy")):
        dvf = np.load(file)

        # Lấy vector chuyển động trong ROI
        roi_vectors = dvf[mask > 0]

        motion_vectors.append(roi_vectors.flatten())

    # Tạo ma trận X (N x T)
    X = np.stack(motion_vectors, axis=1)

    np.save(root / "processed/motion_matrix.npy", X)

    logging.info(f"Ma trận chuyển động: {X.shape}")


if __name__ == "__main__":
    main()
