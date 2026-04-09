from __future__ import annotations

# ---------------------------------------------------------
# Import các thư viện cần thiết
# ---------------------------------------------------------

# logging: dùng để ghi lại tiến trình chạy chương trình
import logging

# Path: hỗ trợ quản lý đường dẫn file/thư mục rõ ràng hơn
from pathlib import Path

# matplotlib.pyplot: dùng để vẽ biểu đồ và lưu hình
import matplotlib.pyplot as plt

# numpy: hỗ trợ thao tác mảng số liệu
import numpy as np

# pandas: dùng để đọc file CSV và xử lý dữ liệu dạng bảng
import pandas as pd

# Import các hàm tiện ích từ file utils.py
from utils import ensure_dir, load_config, parse_args, setup_logging


def main() -> None:
    # ---------------------------------------------------------
    # BƯỚC 1: Đọc tham số dòng lệnh và file cấu hình
    # ---------------------------------------------------------
    # parse_args: nhận tham số --config từ dòng lệnh khi chạy chương trình
    # Ví dụ: python 06_generate_figures.py --config config.yaml
    args = parse_args("Step 6: Generate reviewer-check figures and summary visuals")

    # load_config: đọc nội dung file cấu hình YAML
    cfg = load_config(args.config)

    # ---------------------------------------------------------
    # BƯỚC 2: Khai báo các thư mục chính của project
    # ---------------------------------------------------------
    # root_dir: thư mục gốc của toàn bộ project
    root_dir = Path(cfg["paths"]["root_dir"])

    # results_dir: thư mục tổng lưu kết quả
    results_dir = ensure_dir(root_dir / cfg["paths"]["results_dir"])

    # figures_dir: thư mục riêng để lưu hình
    figures_dir = ensure_dir(results_dir / "figures")

    # tables_dir: thư mục lưu các bảng CSV đầu vào cho bước vẽ hình
    tables_dir = ensure_dir(results_dir / "tables")

    # logs_dir: thư mục lưu file log
    logs_dir = ensure_dir(root_dir / cfg["paths"]["logs_dir"])

    # ---------------------------------------------------------
    # BƯỚC 3: Thiết lập file log cho bước 6
    # ---------------------------------------------------------
    # Mọi thông báo trong quá trình chạy sẽ được ghi lại vào file này
    setup_logging(logs_dir / "06_generate_figures.log")

    # ---------------------------------------------------------
    # BƯỚC 4: Đọc dữ liệu đầu vào từ các bảng CSV
    # ---------------------------------------------------------
    # pca_summary.csv: bảng tóm tắt PCA
    # Bao gồm:
    # - component: tên thành phần chính (PC1, PC2, PC3...)
    # - explained_variance_ratio: tỷ lệ phương sai giải thích
    # - dominant_direction: hướng chi phối (SI, AP, LR)
    # - amplitude_mm: biên độ đã hiệu chuẩn sang đơn vị mm
    pca_summary = pd.read_csv(tables_dir / "pca_summary.csv")

    # temporal_coefficients.csv: bảng chứa hệ số thời gian theo pha hô hấp
    # Bao gồm:
    # - cột phase: tên pha (T00, T10, ..., T90)
    # - các cột còn lại: hệ số của từng thành phần PCA theo từng pha
    coeff_df = pd.read_csv(tables_dir / "temporal_coefficients.csv")

    # Trích riêng cột tỷ lệ phương sai giải thích và chuyển thành mảng NumPy
    explained_ratio = pca_summary["explained_variance_ratio"].to_numpy()

    # ---------------------------------------------------------
    # BƯỚC 5: TẠO HÌNH 1 - TỶ LỆ PHƯƠNG SAI GIẢI THÍCH
    # ---------------------------------------------------------
    # Hình này gồm:
    # - biểu đồ cột: tỷ lệ phương sai giải thích của từng PC
    # - đường tích lũy: tổng phương sai tích lũy của các PC
    fig = plt.figure(figsize=(6, 4))

    # Tạo trục x = 1, 2, 3, ... tương ứng với các thành phần chính
    x = np.arange(1, len(explained_ratio) + 1)

    # Vẽ biểu đồ cột cho explained variance ratio
    plt.bar(x, explained_ratio)

    # Vẽ đường tích lũy tổng phương sai
    plt.plot(x, np.cumsum(explained_ratio), marker="o")

    # Gắn nhãn trục x
    plt.xlabel("Principal component")

    # Gắn nhãn trục y
    plt.ylabel("Explained variance ratio")

    # Tiêu đề hình
    plt.title("Explained variance and cumulative contribution")

    # Tự động căn chỉnh bố cục để tránh đè chữ
    plt.tight_layout()

    # Lưu hình ra file PNG với độ phân giải 300 dpi
    fig.savefig(figures_dir / "explained_variance.png", dpi=300)

    # Đóng hình để giải phóng bộ nhớ
    plt.close(fig)

    # ---------------------------------------------------------
    # BƯỚC 6: TẠO HÌNH 2 - HỆ SỐ THỜI GIAN THEO PHA HÔ HẤP
    # ---------------------------------------------------------
    # Hình này thể hiện sự biến đổi của các hệ số PCA theo các pha hô hấp
    fig = plt.figure(figsize=(7, 4))

    # Lấy danh sách nhãn pha hô hấp từ cột "phase"
    phase_labels = coeff_df["phase"].tolist()

    # Tạo trục x theo số lượng pha
    x = np.arange(len(phase_labels))

    # Duyệt qua từng cột hệ số PCA (bỏ qua cột đầu tiên là "phase")
    for col in coeff_df.columns[1:]:
        # Vẽ đường biểu diễn hệ số theo từng pha
        plt.plot(x, coeff_df[col].to_numpy(), marker="o", label=col)

    # Gắn nhãn các pha lên trục x và xoay 45 độ cho dễ đọc
    plt.xticks(x, phase_labels, rotation=45)

    # Nhãn trục x
    plt.xlabel("Respiratory phase")

    # Nhãn trục y
    plt.ylabel("Temporal coefficient (a.u.)")

    # Tiêu đề hình
    plt.title("Temporal evolution of retained PCA components")

    # Hiển thị chú giải cho từng đường
    plt.legend()

    # Tự động căn chỉnh bố cục
    plt.tight_layout()

    # Lưu hình ra file PNG
    fig.savefig(figures_dir / "temporal_coefficients.png", dpi=300)

    # Đóng hình
    plt.close(fig)

    # ---------------------------------------------------------
    # BƯỚC 7: TẠO HÌNH 3 - BIÊN ĐỘ CHUYỂN ĐỘNG ĐÃ HIỆU CHUẨN (mm)
    # ---------------------------------------------------------
    # Hình này biểu diễn biên độ chuyển động của từng PC
    # Sau khi đã quy đổi từ đơn vị PCA sang đơn vị mm
    fig = plt.figure(figsize=(6, 4))

    # Tạo nhãn trục x theo dạng:
    # PC1
    # (SI)
    # PC2
    # (AP)
    # ...
    labels = [
        f"{c}\n({d})"
        for c, d in zip(pca_summary["component"], pca_summary["dominant_direction"])
    ]

    # Vẽ biểu đồ cột với trục x là tên PC + hướng chi phối,
    # trục y là biên độ chuyển động tính bằng mm
    plt.bar(labels, pca_summary["amplitude_mm"].to_numpy())

    # Nhãn trục y
    plt.ylabel("Amplitude (mm)")

    # Tiêu đề hình
    plt.title("Calibrated motion amplitudes")

    # Tự động căn chỉnh bố cục
    plt.tight_layout()

    # Lưu hình ra file PNG
    fig.savefig(figures_dir / "amplitude_barplot.png", dpi=300)

    # Đóng hình
    plt.close(fig)

    # ---------------------------------------------------------
    # BƯỚC 8: Ghi log hoàn tất
    # ---------------------------------------------------------
    # Sau khi tạo xong toàn bộ các hình, ghi lại vị trí thư mục đầu ra
    logging.info("Generated figures in %s", figures_dir)


# ---------------------------------------------------------
# Điểm bắt đầu chạy chương trình
# ---------------------------------------------------------
# Khi chạy trực tiếp file này, hàm main() sẽ được thực thi
if __name__ == "__main__":
    main()
