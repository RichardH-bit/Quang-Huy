import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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
    xls = pd.ExcelFile(f)
    for sheet in xls.sheet_names:
        df = pd.read_excel(f, sheet_name=sheet)
        patient_frames[sheet] = df.copy()

print(f"Loaded {len(patient_frames)} patient sheets.")

# =========================
# 3. CHECK COLUMN NAMES
# =========================
# Expected columns: Phase, PC1, PC2, PC3
# Nếu file anh khác tên cột, sửa lại ở đây.
required_cols = ["Phase", "PC1", "PC2", "PC3"]

for name, df in patient_frames.items():
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Sheet {name} is missing columns: {missing}")

# =========================
# 4. COMPUTE PATIENT-LEVEL AMPLITUDES
# =========================
# Cách tính:
# - group theo Phase
# - lấy mean của PC1/PC2/PC3 trên các slice
# - A = (max - min)/2

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
# 5. SUMMARY STATISTICS
# =========================
summary = {
    "A1 (PC1)": (amp_df["A1_PC1"].mean(), amp_df["A1_PC1"].std(ddof=1)),
    "A2 (PC2)": (amp_df["A2_PC2"].mean(), amp_df["A2_PC2"].std(ddof=1)),
    "A3 (PC3)": (amp_df["A3_PC3"].mean(), amp_df["A3_PC3"].std(ddof=1)),
}

print("\nPatient-level amplitude summary (a.u.):")
for k, (m, s) in summary.items():
    print(f"{k}: {m:.2f} ± {s:.2f}")

# =========================
# 6. SAVE TABLES
# =========================
amp_df.to_csv("pca_patient_level_amplitudes_61patients.csv", index=False)

summary_df = pd.DataFrame({
    "Component": ["A1 (PC1)", "A2 (PC2)", "A3 (PC3)"],
    "Mean_a.u.": [
        amp_df["A1_PC1"].mean(),
        amp_df["A2_PC2"].mean(),
        amp_df["A3_PC3"].mean()
    ],
    "SD_a.u.": [
        amp_df["A1_PC1"].std(ddof=1),
        amp_df["A2_PC2"].std(ddof=1),
        amp_df["A3_PC3"].std(ddof=1)
    ],
    "Median_a.u.": [
        amp_df["A1_PC1"].median(),
        amp_df["A2_PC2"].median(),
        amp_df["A3_PC3"].median()
    ],
    "Q1_a.u.": [
        amp_df["A1_PC1"].quantile(0.25),
        amp_df["A2_PC2"].quantile(0.25),
        amp_df["A3_PC3"].quantile(0.25)
    ],
    "Q3_a.u.": [
        amp_df["A1_PC1"].quantile(0.75),
        amp_df["A2_PC2"].quantile(0.75),
        amp_df["A3_PC3"].quantile(0.75)
    ]
})

summary_df.to_csv("pca_amplitude_summary_61patients.csv", index=False)

# =========================
# 7. FIGURE: BOXPLOT + PAIRED DOTS
# =========================
vals = [
    amp_df["A1_PC1"].values,
    amp_df["A2_PC2"].values,
    amp_df["A3_PC3"].values
]
labels = ["A1 (PC1)", "A2 (PC2)", "A3 (PC3)"]
x = np.array([1, 2, 3], dtype=float)

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# Boxplot
ax.boxplot(
    vals,
    positions=x,
    widths=0.45,
    patch_artist=False,
    showfliers=False
)

# Paired dots + paired lines
rng = np.random.default_rng(42)
j1 = rng.normal(0, 0.03, len(amp_df))
j2 = rng.normal(0, 0.03, len(amp_df))
j3 = rng.normal(0, 0.03, len(amp_df))

x1 = np.full(len(amp_df), 1.0) + j1
x2 = np.full(len(amp_df), 2.0) + j2
x3 = np.full(len(amp_df), 3.0) + j3

for i in range(len(amp_df)):
    ax.plot(
        [x1[i], x2[i], x3[i]],
        [amp_df.loc[i, "A1_PC1"], amp_df.loc[i, "A2_PC2"], amp_df.loc[i, "A3_PC3"]],
        linewidth=0.8,
        alpha=0.25
    )

ax.plot(x1, amp_df["A1_PC1"], "o", markersize=4, alpha=0.6)
ax.plot(x2, amp_df["A2_PC2"], "o", markersize=4, alpha=0.6)
ax.plot(x3, amp_df["A3_PC3"], "o", markersize=4, alpha=0.6)

# Mean ± SD
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
    fmt="s", capsize=5, linewidth=1.2
)

for xi, m, sd in zip(x, means, sds):
    ax.text(
        xi, m + sd + 0.03 * max(means),
        f"{m:.0f} ± {sd:.0f}",
        ha="center", va="bottom", fontsize=10
    )

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("PCA amplitude (a.u.)", fontsize=12)
ax.set_xlabel("Principal motion component", fontsize=12)
ax.set_title("Patient-level PCA amplitudes across the cohort (n = 61)", fontsize=13)

ax.grid(True, axis="y", alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("figure_boxplot_paired_dots_pca_amplitudes.png", bbox_inches="tight")
plt.savefig("figure_boxplot_paired_dots_pca_amplitudes.pdf", bbox_inches="tight")
plt.show()