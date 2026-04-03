from __future__ import annotations

# Thư viện logging dùng để ghi lại tiến trình chạy chương trình vào màn hình hoặc file log
import logging

# Path giúp thao tác đường dẫn file/thư mục rõ ràng và an toàn hơn giữa các hệ điều hành
from pathlib import Path

# NumPy dùng để xử lý mảng số liệu và lưu/đọc file .npy
import numpy as np

# Import các hàm tiện ích từ file utils.py
from utils import (
    clip_and_normalize,          # Cắt ngưỡng cường độ ảnh và chuẩn hóa dữ liệu
    ensure_dir,                  # Tạo thư mục nếu chưa tồn tại
    generate_spherical_mask,     # Tạo ROI mask hình cầu giả lập
    generate_synthetic_phase_stack,  # Sinh bộ dữ liệu 4DCT giả lập theo các pha hô hấp
    load_config,                 # Đọc file cấu hình config.yaml
    load_phase_volumes,          # Nạp các volume ảnh theo từng pha
    parse_args,                  # Đọc tham số truyền từ dòng lệnh
    save_json,                   # Lưu metadata ra file JSON
    setup_logging,               # Thiết lập file log
)


def main() -> None:
    # ------------------------------------------------------------------
    # BƯỚC 1: Đọc tham số đầu vào từ dòng lệnh
    # Ví dụ: python 01_prepare_data.py --config config.yaml
    # ------------------------------------------------------------------
    args = parse_args("Step 1: Prepare standardized phase volumes for PCA/SVD motion analysis")

    # ------------------------------------------------------------------
    # BƯỚC 2: Đọc file cấu hình chung của toàn bộ pipeline
    # File config chứa đường dẫn, tên pha, kích thước ảnh, spacing, ROI...
    # ------------------------------------------------------------------
    cfg = load_config(args.config)

    # ------------------------------------------------------------------
    # BƯỚC 3: Xác định các thư mục làm việc chính từ config
    # root_dir: thư mục gốc của project
    # processed_dir: nơi lưu dữ liệu đã chuẩn hóa
    # masks_dir: nơi lưu ROI mask
    # logs_dir: nơi lưu file log
    # example_phase_dir: nơi chứa ảnh theo từng pha của 1 bệnh nhân ví dụ
    # ------------------------------------------------------------------
    root_dir = Path(cfg["paths"]["root_dir"])
    processed_dir = ensure_dir(root_dir / cfg["paths"]["processed_dir"])
    masks_dir = ensure_dir(root_dir / cfg["paths"]["masks_dir"])
    logs_dir = ensure_dir(root_dir / cfg["paths"]["logs_dir"])
    example_phase_dir = ensure_dir(root_dir / cfg["paths"]["example_patient_dir"] / "phases")

    # ------------------------------------------------------------------
    # BƯỚC 4: Khởi tạo hệ thống logging để theo dõi toàn bộ tiến trình xử lý
    # File log sẽ được lưu trong thư mục logs
    # ------------------------------------------------------------------
    setup_logging(logs_dir / "01_prepare_data.log")
    logging.info("Starting data preparation")

    # ------------------------------------------------------------------
    # BƯỚC 5: Đọc các thông số cần thiết từ file config
    # phase_names: danh sách các pha hô hấp (ví dụ T00, T10, ..., T90)
    # reference_phase: pha tham chiếu (ví dụ T50)
    # expected_shape: kích thước ảnh mong đợi (ví dụ 128x128x64 hoặc 512x512x78)
    # clip_range: khoảng cắt cường độ HU
    # normalize: có chuẩn hóa cường độ hay không
    # rng_seed: seed để sinh dữ liệu giả lập có thể tái lập được
    # ------------------------------------------------------------------
    phase_names = cfg["phases"]["names"]
    reference_phase = cfg["phases"]["reference_phase"]
    expected_shape = tuple(cfg["image"]["expected_shape"])
    clip_range = tuple(cfg["image"]["intensity_clip"])
    normalize = bool(cfg["image"]["normalize"])
    rng_seed = int(cfg.get("random_seed", 42))

    # ------------------------------------------------------------------
    # BƯỚC 6: Kiểm tra xem đã có sẵn file ảnh .npy của tất cả các pha hay chưa
    # Nếu có sẵn -> nạp dữ liệu thật/đã xử lý
    # Nếu chưa có -> tự động sinh dữ liệu giả lập để reviewer vẫn chạy được pipeline
    # ------------------------------------------------------------------
    phase_files_exist = all((example_phase_dir / f"{phase}.npy").exists() for phase in phase_names)

    if phase_files_exist:
        logging.info("Found user-provided processed phase volumes. Loading from %s", example_phase_dir)

        # Nạp toàn bộ volume ảnh theo từng pha
        volumes = load_phase_volumes(example_phase_dir, phase_names)
    else:
        logging.info("No phase volumes found. Generating synthetic demonstration dataset.")

        # Sinh bộ dữ liệu 4DCT giả lập cho tất cả các pha
        volumes = generate_synthetic_phase_stack(phase_names, expected_shape, seed=rng_seed)

        # Lưu từng volume giả lập ra thư mục example_phase_dir
        for phase, volume in volumes.items():
            np.save(example_phase_dir / f"{phase}.npy", volume)

    # ------------------------------------------------------------------
    # BƯỚC 7: Tạo thư mục lưu các phase đã chuẩn hóa
    # ------------------------------------------------------------------
    prepared_phase_dir = ensure_dir(processed_dir / "phases")

    # ------------------------------------------------------------------
    # BƯỚC 8: Chuẩn hóa từng pha ảnh
    # - Kiểm tra shape có đúng với cấu hình mong đợi hay không
    # - Cắt ngưỡng cường độ ảnh
    # - Chuẩn hóa dữ liệu nếu bật normalize
    # - Lưu lại thành file .npy trong thư mục processed/phases
    # ------------------------------------------------------------------
    for phase, volume in volumes.items():
        if volume.shape != expected_shape:
            raise ValueError(f"Unexpected shape for {phase}: {volume.shape}, expected {expected_shape}")

        # Cắt ngưỡng và chuẩn hóa cường độ
        prepared = clip_and_normalize(volume, clip_range, normalize)

        # Lưu dữ liệu đã chuẩn hóa
        np.save(prepared_phase_dir / f"{phase}.npy", prepared)
        logging.info("Saved standardized phase %s with shape %s", phase, prepared.shape)

    # ------------------------------------------------------------------
    # BƯỚC 9: Chuẩn bị ROI mask
    # Nếu đã có mask sẵn -> nạp lại
    # Nếu chưa có -> tạo mask hình cầu giả lập và lưu ra file
    # ------------------------------------------------------------------
    mask_filename = cfg["roi"]["mask_filename"]
    mask_path = masks_dir / mask_filename

    if mask_path.exists():
        # Nạp ROI mask đã có sẵn
        mask = np.load(mask_path)
        logging.info("Loaded existing ROI mask from %s", mask_path)
    else:
        # Tạo ROI mask giả lập với bán kính lấy từ config
        radius_voxels = int(cfg["roi"]["default_radius_voxels"])
        mask = generate_spherical_mask(expected_shape, radius_voxels=radius_voxels)

        # Lưu ROI mask
        np.save(mask_path, mask)
        logging.info("Generated synthetic ROI mask and saved to %s", mask_path)

    # ------------------------------------------------------------------
    # BƯỚC 10: Lưu metadata của bước chuẩn bị dữ liệu
    # Metadata này giúp các bước sau biết:
    # - danh sách pha
    # - pha tham chiếu
    # - kích thước ảnh
    # - spacing voxel
    # - đường dẫn dữ liệu đã chuẩn hóa
    # - đường dẫn ROI mask
    # ------------------------------------------------------------------
    metadata = {
        "phase_names": phase_names,
        "reference_phase": reference_phase,
        "shape": list(expected_shape),
        "spacing_mm": list(cfg["image"]["spacing_mm"]),
        "prepared_phase_dir": str(prepared_phase_dir),
        "mask_path": str(mask_path),
    }

    # Lưu metadata ra file JSON
    save_json(metadata, processed_dir / "prepared_metadata.json")

    # ------------------------------------------------------------------
    # BƯỚC 11: Kết thúc bước chuẩn bị dữ liệu
    # ------------------------------------------------------------------
    logging.info("Preparation complete")


# Điểm bắt đầu của chương trình khi chạy file trực tiếp
if __name__ == "__main__":
    main()
