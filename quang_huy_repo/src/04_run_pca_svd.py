# -----------------------------
# BƯỚC 4: PCA / SVD
# Trích xuất PC1–PC3 và phương sai
# -----------------------------

import numpy as np
from pathlib import Path
import logging

from utils import setup_logging


def main():
    root = Path(".")

    setup_logging(root / "logs" / "04_pca.log")

    X = np.load("processed/motion_matrix.npy")

    # Center dữ liệu
    X_centered = X - np.mean(X, axis=1, keepdims=True)

    # SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # PC1–PC3
    PCs = U[:, :3]
    np.save("processed/PCs.npy", PCs)

    # Variance explained
    var = (S ** 2) / np.sum(S ** 2)
    np.save("processed/variance.npy", var)

    logging.info(f"Variance PC1–PC3: {var[:3]}")


if __name__ == "__main__":
    main()
