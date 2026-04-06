#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Full PCA respiratory-motion pipeline
-----------------------------------
Mục tiêu:
1) Đọc dữ liệu PCA của nhiều bệnh nhân từ nhiều file Excel (mỗi sheet = 1 bệnh nhân)
2) Quy đổi PCA units -> mm (mặc định: 1000 units = 1 mm)
3) Fit mô hình sinusoidal theo 10 phase cho từng bệnh nhân, từng trục PC1/PC2/PC3
4) Tính RMSE (mm) và R²
5) Xuất bảng kết quả
6) Vẽ 3 hình:
   - Figure 1: Mean motion (mean ± SD) của PC1, PC2, PC3 theo 10 phase
   - Figure 2: Representative patient actual vs sinus fit (3 panel: PC1, PC2, PC3)
   - Figure 3: Mean 3D respiratory trajectory + 3 projection panels

Tác giả: ChatGPT hỗ trợ anh Huy
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score


# =========================================================
# 0. CẤU HÌNH
# =========================================================
# Anh sửa danh sách file tại đây cho đúng máy của anh.
FILES = [
    r"E:\HR_share\simulated_pca_78slices_part1_01-10.xlsx",
    r"E:\HR_share\simulated_pca_78slices_part2_30patients_11-40.xlsx",
    r"E:\HR_share\simulated_pca_78slices_part2_30patients_41-61.xlsx",
]

# Thư mục output
OUTPUT_DIR = Path("/mnt/data/pca_sinus_outputs")

# Quy đổi đơn vị
PCA_UNITS_PER_MM = 1000.0   # 1000 PCA units = 1 mm

# Phase chuẩn kỳ vọng
EXPECTED_PHASES = 10

# Patient đại diện để vẽ figure 2
# None = tự chọn bệnh nhân đầu tiên có fit thành công cả 3 trục
REPRESENTATIVE_PATIENT = None

# Font/style
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["axes.linewidth"] = 1.0


# =========================================================
# 1. HÀM PHỤ
# =========================================================
def sinusoid(p, A, phi, C):
    """
    Mô hình sinusoidal theo 10 phase:
    y = A * sin(2*pi/10 * (p-1) + phi) + C
    """
    return A * np.sin(2 * np.pi / 10.0 * (p - 1) + phi) + C


def safe_patient_name(name: str) -> str:
    """Làm sạch tên file xuất."""
    keep = []
    for ch in str(name):
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def load_patient_frames(files: list[str]) -> dict[str, pd.DataFrame]:
    """
    Đọc toàn bộ sheet từ nhiều file Excel.
    Nếu trùng tên sheet giữa các file, sẽ tự thêm hậu tố __dupN để không ghi đè.
    """
    patient_frames: dict[str, pd.DataFrame] = {}
    duplicate_count: dict[str, int] = {}

    for file_path in files:
        file_path = str(file_path)
        if not Path(file_path).exists():
            print(f"[WARNING] Không tìm thấy file: {file_path}")
            continue

        print(f"[INFO] Đang đọc file: {file_path}")
        xls = pd.ExcelFile(file_path)

        for sheet in xls.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet)

            # Kiểm tra cột bắt buộc
            required_cols = {"Phase", "PC1", "PC2", "PC3"}
            missing = required_cols - set(df.columns)
            if missing:
                print(f"[WARNING] Sheet {sheet} thiếu cột: {missing}. Bỏ qua.")
                continue

            # Ép kiểu số
            for col in ["Phase", "PC1", "PC2", "PC3"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(subset=["Phase", "PC1", "PC2", "PC3"]).copy()

            # Quy đổi sang mm
            df["PC1_mm"] = df["PC1"] / PCA_UNITS_PER_MM
            df["PC2_mm"] = df["PC2"] / PCA_UNITS_PER_MM
            df["PC3_mm"] = df["PC3"] / PCA_UNITS_PER_MM

            patient_name = str(sheet)
            if patient_name in patient_frames:
                duplicate_count[patient_name] = duplicate_count.get(patient_name, 1) + 1
                patient_name = f"{patient_name}__dup{duplicate_count[str(sheet)]}"

            patient_frames[patient_name] = df

    return patient_frames


def compute_phase_means(patient_frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Tính mean theo phase cho từng bệnh nhân.
    """
    phase_mean_dict: dict[str, pd.DataFrame] = {}

    for patient, df in patient_frames.items():
        phase_mean = (
            df.groupby("Phase")[["PC1_mm", "PC2_mm", "PC3_mm"]]
            .mean()
            .sort_index()
            .reset_index()
        )

        if len(phase_mean) < EXPECTED_PHASES:
            print(f"[WARNING] {patient}: chỉ có {len(phase_mean)} phase sau khi group.")
        phase_mean_dict[patient] = phase_mean

    return phase_mean_dict


def fit_all_patients(phase_mean_dict: dict[str, pd.DataFrame]):
    """
    Fit sinus cho từng bệnh nhân, từng trục.
    Trả về:
      - results_df: bảng RMSE, R², tham số fit
      - fit_curves: dict lưu actual/predicted để vẽ
    """
    fit_rows = []
    fit_curves = {}

    for patient, phase_mean in phase_mean_dict.items():
        p = phase_mean["Phase"].to_numpy(dtype=float)

        fit_curves[patient] = {}

        for axis in ["PC1_mm", "PC2_mm", "PC3_mm"]:
            y = phase_mean[axis].to_numpy(dtype=float)

            # initial guess giúp curve_fit ổn định hơn
            amplitude_guess = max((np.max(y) - np.min(y)) / 2.0, 1e-6)
            phi_guess = 0.0
            c_guess = np.mean(y)

            try:
                popt, _ = curve_fit(
                    sinusoid,
                    p,
                    y,
                    p0=[amplitude_guess, phi_guess, c_guess],
                    maxfev=20000,
                )

                y_pred = sinusoid(p, *popt)
                rmse = math.sqrt(mean_squared_error(y, y_pred))
                r2 = r2_score(y, y_pred)

                fit_rows.append(
                    {
                        "Patient": patient,
                        "Axis": axis.replace("_mm", ""),
                        "RMSE_mm": rmse,
                        "R2": r2,
                        "Amplitude_mm": popt[0],
                        "PhaseShift_rad": popt[1],
                        "Offset_mm": popt[2],
                    }
                )

                fit_curves[patient][axis] = {
                    "phase": p,
                    "actual": y,
                    "predicted": y_pred,
                    "params": popt,
                }

            except Exception as e:
                print(f"[WARNING] Fit failed: patient={patient}, axis={axis}, error={e}")

    results_df = pd.DataFrame(fit_rows)
    return results_df, fit_curves


def choose_representative_patient(results_df: pd.DataFrame, fit_curves: dict, preferred: str | None = None) -> str:
    """
    Chọn 1 bệnh nhân đại diện:
    - nếu preferred có tồn tại và đủ 3 trục -> dùng
    - ngược lại chọn bệnh nhân đầu tiên đủ 3 trục
    """
    if preferred is not None and preferred in fit_curves and len(fit_curves[preferred]) == 3:
        return preferred

    counts = results_df.groupby("Patient")["Axis"].nunique().sort_index()
    ok_patients = counts[counts == 3].index.tolist()
    if not ok_patients:
        raise RuntimeError("Không có bệnh nhân nào fit thành công cả 3 trục.")
    return ok_patients[0]


def plot_figure1_mean_motion(all_phase_df: pd.DataFrame, output_dir: Path):
    """
    Figure 1: Mean ± SD theo phase cho PC1/PC2/PC3
    """
    summary = (
        all_phase_df.groupby("Phase")[["PC1_mm", "PC2_mm", "PC3_mm"]]
        .agg(["mean", "std"])
        .sort_index()
    )

    phase = summary.index.to_numpy()

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.errorbar(
        phase,
        summary[("PC1_mm", "mean")].to_numpy(),
        yerr=summary[("PC1_mm", "std")].to_numpy(),
        marker="o",
        linestyle="-",
        linewidth=1.5,
        markersize=4.5,
        capsize=3,
        label="PC1 (S–I)",
    )
    ax.errorbar(
        phase,
        summary[("PC2_mm", "mean")].to_numpy(),
        yerr=summary[("PC2_mm", "std")].to_numpy(),
        marker="s",
        linestyle="--",
        linewidth=1.5,
        markersize=4.5,
        capsize=3,
        label="PC2 (A–P)",
    )
    ax.errorbar(
        phase,
        summary[("PC3_mm", "mean")].to_numpy(),
        yerr=summary[("PC3_mm", "std")].to_numpy(),
        marker="^",
        linestyle=":",
        linewidth=1.8,
        markersize=5.0,
        capsize=3,
        label="PC3 (L–R)",
    )

    ax.set_xlabel("Respiratory phase")
    ax.set_ylabel("Amplitude (mm)")
    ax.set_title("Mean PCA-derived respiratory motion across patients")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()

    png_path = output_dir / "Figure1_mean_motion.png"
    tiff_path = output_dir / "Figure1_mean_motion.tiff"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(tiff_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    return summary, png_path, tiff_path


def plot_figure2_representative_patient(rep_patient: str, fit_curves: dict, results_df: pd.DataFrame, output_dir: Path):
    """
    Figure 2: Representative patient actual vs sinus fit (3 panel)
    """
    axis_map = {
        "PC1_mm": "PC1 (S–I)",
        "PC2_mm": "PC2 (A–P)",
        "PC3_mm": "PC3 (L–R)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), sharex=True)

    for ax, axis in zip(axes, ["PC1_mm", "PC2_mm", "PC3_mm"]):
        curve = fit_curves[rep_patient][axis]
        phase = curve["phase"]
        actual = curve["actual"]
        pred = curve["predicted"]

        metric_row = results_df[
            (results_df["Patient"] == rep_patient) &
            (results_df["Axis"] == axis.replace("_mm", ""))
        ].iloc[0]

        ax.plot(phase, actual, marker="o", linestyle="-", linewidth=1.7, markersize=4.5, label="Actual")
        ax.plot(phase, pred, linestyle="--", linewidth=1.7, label="Sin fit")

        ax.set_title(axis_map[axis])
        ax.set_xlabel("Phase")
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.text(
            0.03,
            0.96,
            f"RMSE = {metric_row['RMSE_mm']:.3f} mm\nR² = {metric_row['R2']:.4f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.7"),
        )

    axes[0].set_ylabel("Amplitude (mm)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(f"Representative patient: {rep_patient}", y=1.10, fontsize=12)
    fig.tight_layout()

    safe_name = safe_patient_name(rep_patient)
    png_path = output_dir / f"Figure2_rep_patient_{safe_name}.png"
    tiff_path = output_dir / f"Figure2_rep_patient_{safe_name}.tiff"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(tiff_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    return png_path, tiff_path


def plot_figure3_3d_trajectory(summary: pd.DataFrame, output_dir: Path):
    """
    Figure 3:
    - panel trái: quỹ đạo 3D
    - 3 panel phải: projection 2D
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    phases = summary.index.to_numpy()
    pc1_mean = summary[("PC1_mm", "mean")].to_numpy()  # S-I
    pc2_mean = summary[("PC2_mm", "mean")].to_numpy()  # A-P
    pc3_mean = summary[("PC3_mm", "mean")].to_numpy()  # L-R

    # center để nhìn quỹ đạo rõ hơn
    pc1_center = pc1_mean - np.mean(pc1_mean)
    pc2_center = pc2_mean - np.mean(pc2_mean)
    pc3_center = pc3_mean - np.mean(pc3_mean)

    fig = plt.figure(figsize=(13.0, 8.0))

    # ---- 3D panel
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.plot(pc3_center, pc2_center, pc1_center, marker="o", linewidth=1.8, markersize=4.5)

    for i, ph in enumerate(phases):
        ax3d.text(pc3_center[i], pc2_center[i], pc1_center[i], str(int(ph)), fontsize=8)

    ax3d.scatter(pc3_center[0], pc2_center[0], pc1_center[0], marker="s", s=50)
    ax3d.scatter(pc3_center[-1], pc2_center[-1], pc1_center[-1], marker="^", s=50)

    ax3d.set_xlabel("L–R (mm)")
    ax3d.set_ylabel("A–P (mm)")
    ax3d.set_zlabel("S–I (mm)")
    ax3d.set_title("3D mean trajectory")
    ax3d.view_init(elev=22, azim=-55)

    # ---- projection 1
    ax1 = fig.add_subplot(2, 2, 2)
    ax1.plot(pc2_center, pc1_center, marker="o", linewidth=1.7)
    for i, ph in enumerate(phases):
        ax1.text(pc2_center[i], pc1_center[i], str(int(ph)), fontsize=8)
    ax1.set_xlabel("A–P (mm)")
    ax1.set_ylabel("S–I (mm)")
    ax1.set_title("Projection: S–I vs A–P")
    ax1.grid(True, linewidth=0.4, alpha=0.5)

    # ---- projection 2
    ax2 = fig.add_subplot(2, 2, 3)
    ax2.plot(pc3_center, pc1_center, marker="o", linewidth=1.7)
    for i, ph in enumerate(phases):
        ax2.text(pc3_center[i], pc1_center[i], str(int(ph)), fontsize=8)
    ax2.set_xlabel("L–R (mm)")
    ax2.set_ylabel("S–I (mm)")
    ax2.set_title("Projection: S–I vs L–R")
    ax2.grid(True, linewidth=0.4, alpha=0.5)

    # ---- projection 3
    ax3 = fig.add_subplot(2, 2, 4)
    ax3.plot(pc3_center, pc2_center, marker="o", linewidth=1.7)
    for i, ph in enumerate(phases):
        ax3.text(pc3_center[i], pc2_center[i], str(int(ph)), fontsize=8)
    ax3.set_xlabel("L–R (mm)")
    ax3.set_ylabel("A–P (mm)")
    ax3.set_title("Projection: A–P vs L–R")
    ax3.grid(True, linewidth=0.4, alpha=0.5)

    fig.suptitle("Mean respiratory trajectory across patients", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    png_path = output_dir / "Figure3_mean_3D_trajectory.png"
    tiff_path = output_dir / "Figure3_mean_3D_trajectory.tiff"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(tiff_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    return png_path, tiff_path


def export_summary_tables(results_df: pd.DataFrame, output_dir: Path):
    """
    Xuất:
    - patient-level fit results
    - axis-level summary
    """
    results_path = output_dir / "sinus_fit_results_by_patient.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")

    axis_summary = (
        results_df.groupby("Axis")[["RMSE_mm", "R2"]]
        .agg(["mean", "std", "median"])
        .round(6)
    )
    axis_summary_path = output_dir / "sinus_fit_summary_by_axis.csv"
    axis_summary.to_csv(axis_summary_path, encoding="utf-8-sig")

    return axis_summary, results_path, axis_summary_path


# =========================================================
# 2. MAIN
# =========================================================
def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] ===== START PIPELINE =====")

    # 1) Load data
    patient_frames = load_patient_frames(FILES)
    if not patient_frames:
        raise RuntimeError("Không đọc được bệnh nhân nào. Hãy kiểm tra lại đường dẫn FILES.")

    print(f"[INFO] Tổng số patient sheets đọc được: {len(patient_frames)}")

    # 2) Phase mean cho từng bệnh nhân
    phase_mean_dict = compute_phase_means(patient_frames)

    # Gộp toàn bộ phase mean để phục vụ figure tổng hợp
    all_phase_df = pd.concat(
        [
            df.assign(Patient=patient)
            for patient, df in phase_mean_dict.items()
        ],
        ignore_index=True,
    )

    # 3) Fit sinus
    results_df, fit_curves = fit_all_patients(phase_mean_dict)
    if results_df.empty:
        raise RuntimeError("Không có kết quả fit nào thành công.")

    # 4) Export bảng
    axis_summary, results_path, axis_summary_path = export_summary_tables(results_df, OUTPUT_DIR)

    print("\n[INFO] Axis-level summary:")
    print(axis_summary)

    # 5) Figure 1
    summary, fig1_png, fig1_tiff = plot_figure1_mean_motion(all_phase_df, OUTPUT_DIR)

    # 6) Figure 2
    rep_patient = choose_representative_patient(results_df, fit_curves, REPRESENTATIVE_PATIENT)
    fig2_png, fig2_tiff = plot_figure2_representative_patient(rep_patient, fit_curves, results_df, OUTPUT_DIR)

    # 7) Figure 3
    fig3_png, fig3_tiff = plot_figure3_3d_trajectory(summary, OUTPUT_DIR)

    # 8) Ghi README ngắn
    readme_path = OUTPUT_DIR / "README_outputs.txt"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("Pipeline outputs\n")
        f.write("================\n")
        f.write(f"Total patients read: {len(patient_frames)}\n")
        f.write(f"Representative patient: {rep_patient}\n\n")
        f.write("Files generated:\n")
        f.write(f"- {results_path.name}\n")
        f.write(f"- {axis_summary_path.name}\n")
        f.write(f"- {fig1_png.name}\n")
        f.write(f"- {fig1_tiff.name}\n")
        f.write(f"- {fig2_png.name}\n")
        f.write(f"- {fig2_tiff.name}\n")
        f.write(f"- {fig3_png.name}\n")
        f.write(f"- {fig3_tiff.name}\n")

    print("\n[INFO] ===== DONE =====")
    print(f"[INFO] Output folder: {OUTPUT_DIR.resolve()}")
    print(f"[INFO] Representative patient: {rep_patient}")
    print(f"[INFO] Saved: {fig1_png.name}, {fig2_png.name}, {fig3_png.name}")


if __name__ == "__main__":
    main()
