import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. INPUT FILES
# =========================
files = [
    r"E:\HR_share\simulated_pca_78slices_part1_01-10.xlsx",
    r"E:\HR_share\simulated_pca_78slices_part2_30patients_11-40.xlsx",
    r"E:\HR_share\simulated_pca_78slices_part2_30patients_41-61.xlsx",
]

# =========================
# 2. LOAD ALL SHEETS
# =========================
patient_frames = {}

for f in files:
    xls = pd.ExcelFile(f, engine="openpyxl")
    for sheet in xls.sheet_names:
        df = pd.read_excel(f, sheet_name=sheet, engine="openpyxl")
        patient_frames[sheet] = df.copy()

# =========================
# 3. COMPUTE AMPLITUDES
# =========================
rows = []

for patient, df in sorted(patient_frames.items()):
    phase_means = df.groupby("Phase")[["PC1", "PC2", "PC3"]].mean().sort_index()

    A1 = (phase_means["PC1"].max() - phase_means["PC1"].min()) / 2.0
    A2 = (phase_means["PC2"].max() - phase_means["PC2"].min()) / 2.0
    A3 = (phase_means["PC3"].max() - phase_means["PC3"].min()) / 2.0

    rows.append({
        "Patient": patient,
        "A1_PC1": A1,
        "A2_PC2": A2,
        "A3_PC3": A3
    })

amp_df = pd.DataFrame(rows)

# =========================
# 4. FIGURE: SMALLER + CLEANER
# =========================
vals = [
    amp_df["A1_PC1"].values,
    amp_df["A2_PC2"].values,
    amp_df["A3_PC3"].values
]
labels = ["A1 (PC1)", "A2 (PC2)", "A3 (PC3)"]
x = np.array([1, 2, 3], dtype=float)

# ---- KÍCH THƯỚC HÌNH ----
fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=300)

# ---- BOXPLOT NHỎ HƠN ----
ax.boxplot(
    vals,
    positions=x,
    widths=0.32,          # nhỏ hơn
    patch_artist=False,
    showfliers=False
)

# ---- PAIRED DOTS + LINES NHẸ HƠN ----
rng = np.random.default_rng(42)
j1 = rng.normal(0, 0.018, len(amp_df))   # jitter nhỏ hơn
j2 = rng.normal(0, 0.018, len(amp_df))
j3 = rng.normal(0, 0.018, len(amp_df))

x1 = np.full(len(amp_df), 1.0) + j1
x2 = np.full(len(amp_df), 2.0) + j2
x3 = np.full(len(amp_df), 3.0) + j3

for i in range(len(amp_df)):
    ax.plot(
        [x1[i], x2[i], x3[i]],
        [amp_df.loc[i, "A1_PC1"], amp_df.loc[i, "A2_PC2"], amp_df.loc[i, "A3_PC3"]],
        linewidth=0.5,    # line mảnh hơn
        alpha=0.12        # nhạt hơn
    )

ax.plot(x1, amp_df["A1_PC1"], "o", markersize=2.4, alpha=0.45)
ax.plot(x2, amp_df["A2_PC2"], "o", markersize=2.4, alpha=0.45)
ax.plot(x3, amp_df["A3_PC3"], "o", markersize=2.4, alpha=0.45)

# ---- MEAN ± SD ----
means = [
    amp_df["A1_PC1"].mean(),
    amp_df["A2_PC2"].mean(),
    amp_df["A3_PC3"].mean()
]
sds = [
    amp_df["A1_PC1"].std(ddof=1),
    amp_df["A2_PC2"].std(ddof=1),
    amp_df["A3_PC3"].std(ddof=1)
]

ax.errorbar(
    x, means, yerr=sds,
    fmt="s", capsize=3, linewidth=1.0, markersize=4
)

for xi, m, sd in zip(x, means, sds):
    ax.text(
        xi, m + sd + 0.02 * max(means),
        f"{m:.0f}±{sd:.0f}",
        ha="center", va="bottom", fontsize=8
    )

# ---- FONT NHỎ GỌN HƠN ----
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("PCA amplitude (a.u.)", fontsize=10)
ax.set_xlabel("Principal motion component", fontsize=10)
ax.set_title("Patient-level PCA amplitudes (n = 61)", fontsize=10)

# ---- TỐI GIẢN KHUNG ----
ax.grid(True, axis="y", alpha=0.18)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout(pad=0.8)
plt.savefig(r"E:\HR_share\figure_boxplot_paired_dots_pca_amplitudes_small.png", bbox_inches="tight")
plt.savefig(r"E:\HR_share\figure_boxplot_paired_dots_pca_amplitudes_small.pdf", bbox_inches="tight")
plt.show()