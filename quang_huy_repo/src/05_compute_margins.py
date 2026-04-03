# -----------------------------
# BƯỚC 5: TÍNH BIÊN PTV CÁ NHÂN HÓA
# Áp dụng mô hình kết hợp PCA + yếu tố lâm sàng (Fpos, FT)
# -----------------------------

import numpy as np
import logging
from utils import setup_logging, load_config


def get_Fpos(tumor_location: str) -> float:
    """
    Xác định hệ số vị trí khối u
    upper lobe: chuyển động nhỏ hơn
    mid/lower lobe: chuyển động lớn hơn
    """
    if tumor_location.lower() == "upper":
        return 1.0
    elif tumor_location.lower() in ["middle", "lower"]:
        return 1.2
    else:
        raise ValueError("Unknown tumor location")


def get_FT(T_stage: str) -> float:
    """
    Hệ số giai đoạn T
    T1/T2: di động nhiều hơn
    T3/T4: bị cố định hơn → giảm biên
    """
    if T_stage in ["T1", "T2"]:
        return 1.1
    elif T_stage in ["T3", "T4"]:
        return 0.9
    else:
        raise ValueError("Unknown T stage")


def main():
    setup_logging("logs/05_margin.log")
    logging.info("Bắt đầu tính biên PTV cá nhân hóa")

    cfg = load_config("config.yaml")

    # -----------------------------
    # Load PCA amplitudes (mm)
    # -----------------------------
    PCs = np.load("processed/PCs.npy")

    # Biên độ dao động trung bình theo từng trục
    A_mean = np.std(PCs, axis=0)  # [SI, AP, LR]

    # -----------------------------
    # Load thông tin lâm sàng từ config
    # -----------------------------
    tumor_location = cfg["clinical"]["tumor_location"]  # upper / middle / lower
    T_stage = cfg["clinical"]["T_stage"]                # T1–T4

    Fpos = get_Fpos(tumor_location)
    FT = get_FT(T_stage)

    logging.info(f"Fpos = {Fpos}, FT = {FT}")

    # -----------------------------
    # Tham số hệ thống
    # -----------------------------
    Sigma = cfg["margin"]["Sigma"]   # systematic error
    sigma = cfg["margin"]["sigma"]   # random error
    alpha = cfg["margin"]["alpha"]   # scaling factor

    # -----------------------------
    # Tính biên theo từng trục
    # -----------------------------
    margins = np.sqrt(
        (2.5 * Sigma) ** 2 +
        (0.7 * sigma) ** 2 +
        (alpha * Fpos * FT * A_mean) ** 2
    )

    # -----------------------------
    # Tính biên 3D tổng hợp
    # -----------------------------
    margin_3D = np.sqrt(np.sum(margins ** 2)) + 2  # +2 mm safety

    np.save("processed/margins.npy", margins)

    logging.info(f"Biên theo trục (SI, AP, LR): {margins}")
    logging.info(f"Biên 3D tổng hợp: {margin_3D:.2f} mm")

    print("\n===== KẾT QUẢ =====")
    print(f"Biên SI, AP, LR (mm): {margins}")
    print(f"Biên 3D (mm): {margin_3D:.2f}")


if __name__ == "__main__":
    main()
