# PCA/SVD-based Respiratory Motion Modeling for Individualized PTV Margin Optimization

This repository provides a reviewer-friendly, stepwise implementation of the computational workflow described in the manuscript on PCA/SVD-based respiratory motion modeling for individualized PTV margin optimization in NSCLC radiotherapy.

## What this repository contains

- A 6-step Python pipeline aligned with the Methods section of the manuscript
- Example processed data format for reproducibility
- Outputs for motion amplitudes, explained variance, RMSE, R², and individualized margins
- Scripts to generate tables and figures for manuscript checking

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── example_patient/
│   │   ├── README.md
│   │   └── phases/                # optional: T00.npy ... T90.npy
│   ├── masks/
│   └── processed/
├── docs/
│   └── workflow.md
├── results/
│   ├── figures/
│   ├── logs/
│   └── tables/
└── src/
    ├── 01_prepare_data.py
    ├── 02_run_dir.py
    ├── 03_build_motion_matrix.py
    ├── 04_run_pca_svd.py
    ├── 05_compute_margins.py
    ├── 06_generate_figures.py
    └── utils.py
```

## Required input

The pipeline is designed for processed phase-resolved 3D volumes. Raw clinical 4DCT DICOM files are not included in this archive due to privacy restrictions.

Expected phase names:

- `T00`, `T10`, `T20`, `T30`, `T40`, `T50`, `T60`, `T70`, `T80`, `T90`

Reference phase:

- `T50` (end-exhalation baseline)

Optional tumor mask:

- 3D binary mask in NumPy format aligned with the T50 volume

## Installation

```bash
pip install -r requirements.txt
```

## Quick start

### Step 1. Prepare processed volumes
```bash
python src/01_prepare_data.py --config config.yaml
```

### Step 2. Perform deformable registration and export DVFs
```bash
python src/02_run_dir.py --config config.yaml
```

### Step 3. Build the motion matrix from ROI-focused DVFs
```bash
python src/03_build_motion_matrix.py --config config.yaml
```

### Step 4. Run PCA/SVD and compute motion amplitudes
```bash
python src/04_run_pca_svd.py --config config.yaml
```

### Step 5. Compute individualized margins
```bash
python src/05_compute_margins.py --config config.yaml
```

### Step 6. Generate reviewer-check figures and summary tables
```bash
python src/06_generate_figures.py --config config.yaml
```

## Outputs produced by the pipeline

- `data/processed/prepared_metadata.json`
- `data/processed/dvfs/*.npy`
- `data/processed/motion_matrix.npy`
- `results/tables/pca_summary.csv`
- `results/tables/margins.csv`
- `results/figures/explained_variance.png`
- `results/figures/temporal_coefficients.png`
- `results/figures/amplitude_barplot.png`

## Reproducibility notes

- The scripts are intentionally separated by processing stage so that reviewers can inspect each step independently.
- Each script writes logs and intermediate outputs.
- The DIR script includes a practical fallback for environments where full clinical registration backends are unavailable.

## Suggested manuscript statement

Code and representative processed datasets used in this study are publicly available at the project repository. The repository includes the complete computational workflow for preprocessing, deformable registration, ROI-based motion matrix construction, PCA/SVD analysis, amplitude calibration, and individualized margin calculation.

## Citation

If this code is used in research, please cite the corresponding manuscript and archived DOI release.
