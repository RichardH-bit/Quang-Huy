# -----------------------------
# BƯỚC 2: ĐĂNG KÝ BIẾN DẠNG (DIR)
# Tính toán trường biến dạng (DVF) giữa các pha hô hấp
# -----------------------------

import logging
from pathlib import Path
import numpy as np
import SimpleITK as sitk

from utils import load_config, ensure_dir, setup_logging, save_json


def main():
    # Đọc cấu hình
    cfg = load_config("config.yaml")

    root_dir = Path(cfg["paths"]["root_dir"])
    processed_dir = root_dir / cfg["paths"]["processed_dir"]
    dvf_dir = ensure_dir(processed_dir / "dvf")

    setup_logging(root_dir / "logs" / "02_run_dir.log")
    logging.info("Bắt đầu đăng ký biến dạng (DIR)")

    phase_names = cfg["phases"]["names"]
    ref_phase = cfg["phases"]["reference_phase"]

    # Load pha tham chiếu (T50)
    ref = np.load(processed_dir / "phases" / f"{ref_phase}.npy")
    ref_img = sitk.GetImageFromArray(ref)

    for phase in phase_names:
        if phase == ref_phase:
            continue

        logging.info(f"Đăng ký pha {phase} → {ref_phase}")

        moving = np.load(processed_dir / "phases" / f"{phase}.npy")
        moving_img = sitk.GetImageFromArray(moving)

        # Sử dụng Demons registration (đơn giản, dễ tái lập)
        demons = sitk.DemonsRegistrationFilter()
        dvf = demons.Execute(ref_img, moving_img)

        dvf_array = sitk.GetArrayFromImage(dvf)
        np.save(dvf_dir / f"{phase}_to_{ref_phase}.npy", dvf_array)

    logging.info("Hoàn thành DIR")


if __name__ == "__main__":
    main()
