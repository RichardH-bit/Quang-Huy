# PERSONALIZED 3DCT-BASED MOTION-ADAPTIVE MARGINS #

Physics-Informed PCA-Based Respiratory Motion Modeling from 4DCT
A Reproducible Framework for Quantifying Tumor Motion and Enabling Individualized Margin Design in Lung Radiotherapy

------------------------------
Overview

This repository provides a fully transparent and reproducible implementation of a physics-informed pipeline for extracting respiratory-induced tumor motion from 4DCT imaging using:

Deformable Image Registration (DIR)
Centroid-based motion tracking
Principal Component Analysis (PCA)

The workflow converts raw 4DCT image data into clinically interpretable motion amplitudes (mm) along three anatomical axes (S–I, A–P, L–R), supporting individualized PTV margin strategies.
--------------------------------------
Scientific Rationale

Respiratory motion remains a major source of geometric uncertainty in lung radiotherapy. While 4DCT enables motion visualization, it is:

Resource-intensive
Not always available in routine clinical workflows

This framework addresses the gap by:

Extracting dominant motion patterns using PCA
Quantifying motion amplitudes in physical units (mm)
Enabling data-driven approximation of motion envelopes

Importantly, this method does not aim to replace 4DCT, but to provide a pragmatic, physics-informed bridge toward individualized margin design.

------------------------------------------------------
Computational Workflow

The implemented pipeline strictly follows the methodology described in the manuscript:

Step 1 – 4DCT Acquisition
10 respiratory phases (T00–T90)
Reference phase: end-exhale (e.g., T50)

Step 2 – Deformable Image Registration (DIR)
B-spline registration between phases
Output: Deformation Vector Fields (DVFs)

Step 3 – Centroid Extraction
Tumor centroid computed per phase
All coordinates mapped to a common reference frame

Step 4 – Motion Matrix Construction
Displacement vectors across phases
Units: millimeters (mm)

Step 5 – Principal Component Analysis (PCA)
Input: motion matrix
Output:
pc_scores → temporal coefficients
pc_components → spatial modes
explained_variance_ratio

Step 6 – Motion Amplitude Extraction
Peak-to-peak displacement derived from PCA coefficients
Anatomical mapping:
PC1 → S–I
PC2 → A–P
PC3 → L–R

-------------------------------------------------------------------------------------------
Output Files
| File                      | Description                                         |
| ------------------------- | --------------------------------------------------- |
| `pca_results.csv`         | PCA components, variance explained, amplitudes (mm) |
| `fit_params_by_slice.csv` | Sinusoidal fit parameters (A, φ, C) per slice       |
| `motion_curves.png`       | Motion trajectories and PCA modes                   |

------------------------------------------------------------------------------------
Methodological Clarification (Critical for Reviewers)

PCA outputs are frequently misunderstood

PC1–PC3 are NOT motion amplitudes
They represent orthogonal motion modes

Clinically relevant motion (mm) is derived from:

PCA temporal coefficients (pc_scores)
Peak-to-peak displacement

---------------------------------------------------------------------------------------
Reproducibility & Transparency

This repository ensures:

Full traceability from input images → final motion metrics
Script-based deterministic processing
Independent validation by third parties

All key steps (DIR, PCA, amplitude extraction) are explicitly implemented and accessible.

----------------------------------------------------------------------------------------
Relation to Manuscript

The repository corresponds directly to the manuscript sections:
| Manuscript Section   | Repository Component    |
| -------------------- | ----------------------- |
| Data Acquisition     | 4DCT input handling     |
| Motion Modeling      | DIR + DVF computation   |
| PCA Analysis         | PCA scripts             |
| Amplitude Extraction | Post-processing modules |

-----------------------------------------------------------------------------------------
Clinical Relevance

The framework supports:

Individualized motion quantification
Reduction of isotropic margin overestimation
Improved normal tissue sparing (e.g., lung V20, MLD)

It is compatible with:
Monaco TPS (v5.11.03)
IMRT / VMAT planning workflows

---------------------------------------------------------------------------------------
Requirements
Python ≥ 3.9
NumPy
SciPy
scikit-learn
matplotlib
(Optional) SimpleITK / 3D Slicer

------------------------------------------------------------------------------------------
Usage
git clone https://github.com/RichardH-bit/Quang-Huy.git
cd Quang-Huy

python pca_motion_pipeline.py

----------------------------------------------------------------------------------------
Validation Notes
PCA typically captures >90% motion variance
Motion amplitudes validated against 4DCT-derived displacement
Optional sinusoidal fitting provides temporal consistency (RMSE, R²)

----------------------------------------------------------------------------------------
Limitations
Dependent on DIR accuracy
PCA assumes quasi-periodic motion
External surrogate signals are not included

----------------------------------------------------------------------------------------
Citation

If you use this repository, please cite:

Dang Q.H. et al.
Bridging respiratory motion modeling and clinical margin personalization in lung radiotherapy using PCA.

------------------------------------------------------------------------------------------
Contact

Dr. Dang Quang Huy
Department of Radiotherapy
Military Hospital 175, Vietnam

--------------------------------------------------------------------------------------------
Disclaimer

This code is intended for research use only and requires clinical validation before deployment.
