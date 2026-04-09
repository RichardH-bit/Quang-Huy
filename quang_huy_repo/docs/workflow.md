# Workflow summary

## Step 1
Prepare and standardize phase-resolved volumes. If no example volumes are present, the script can generate a synthetic demonstration dataset with realistic respiratory motion around a simulated lesion.

## Step 2
Run deformable image registration from each phase to the T50 reference phase and export voxel-wise displacement vector fields (DVFs).

## Step 3
Apply the tumor ROI mask to the DVFs and construct the motion matrix with shape `(T, N)` where `T=10` respiratory phases and `N` is the number of retained displacement components.

## Step 4
Run PCA using SVD, compute explained variance, temporal coefficients, and calibrated amplitudes in mm.

## Step 5
Use calibrated amplitudes and van Herk-style uncertainty terms to compute individualized directional margins.

## Step 6
Generate figures and summary tables suitable for reviewer verification.
