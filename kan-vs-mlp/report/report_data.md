# Report Data Points
# Generated from experiment results for report writing reference
# Date: 2026-04-18

## Section A: Experiment 1 - Regression Results

| Config Name | Model Type | Hidden Dim | Total Params | Test MSE (mean +/- std) | Test R2 (mean +/- std) | Convergence Epoch (mean +/- std) | Avg Time/Epoch (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| G3_K2 | kan | 64 | 4032 | 0.2607 +/- 0.0064 | 0.8069 +/- 0.0099 | 9 +/- 1 | 1.8 |
| G3_K3 | kan | 64 | 4608 | 0.2579 +/- 0.0116 | 0.8089 +/- 0.0134 | 9 +/- 2 | 3.7 |
| G5_K2 | kan | 64 | 5184 | 0.2513 +/- 0.0054 | 0.8139 +/- 0.0088 | 7 +/- 0 | 3.3 |
| G5_K3 | kan | 64 | 5760 | 0.2555 +/- 0.0054 | 0.8108 +/- 0.0091 | 7 +/- 1 | 4.2 |
| G10_K2 | kan | 64 | 8064 | 0.2531 +/- 0.0047 | 0.8127 +/- 0.0020 | 5 +/- 2 | 3.3 |
| G10_K3 | kan | 64 | 8640 | 0.2432 +/- 0.0101 | 0.8199 +/- 0.0117 | 5 +/- 0 | 4.5 |
| G20_K2 | kan | 64 | 13824 | 0.2665 +/- 0.0083 | 0.8026 +/- 0.0114 | 3 +/- 0 | 3.6 |
| G20_K3 | kan | 64 | 14400 | 0.2627 +/- 0.0071 | 0.8054 +/- 0.0092 | 3 +/- 0 | 4.5 |
| G10_K2_matched | mlp | 84 | 7981 | 0.2489 +/- 0.0096 | 0.8156 +/- 0.0121 | 18 +/- 3 | 1.0 |
| G10_K3_matched | mlp | 88 | 8713 | 0.2490 +/- 0.0069 | 0.8155 +/- 0.0101 | 17 +/- 5 | 1.0 |
| G20_K2_matched | mlp | 112 | 13777 | 0.2487 +/- 0.0139 | 0.8157 +/- 0.0153 | 16 +/- 5 | 1.1 |
| G20_K3_matched | mlp | 115 | 14491 | 0.2502 +/- 0.0061 | 0.8147 +/- 0.0095 | 14 +/- 2 | 1.1 |
| G3_K2_matched | mlp | 58 | 4003 | 0.2600 +/- 0.0080 | 0.8074 +/- 0.0106 | 21 +/- 1 | 0.5 |
| G3_K3_matched | mlp | 63 | 4663 | 0.2545 +/- 0.0039 | 0.8116 +/- 0.0050 | 19 +/- 3 | 1.0 |
| G5_K2_matched | mlp | 67 | 5227 | 0.2467 +/- 0.0087 | 0.8173 +/- 0.0113 | 21 +/- 4 | 1.1 |
| G5_K3_matched | mlp | 71 | 5823 | 0.2507 +/- 0.0071 | 0.8144 +/- 0.0096 | 19 +/- 2 | 1.0 |

- Best KAN config: `G10_K3` with R2 = 0.8199 +/- 0.0117, MSE = 0.2432 +/- 0.0101
- Best MLP config: `G5_K2_matched` with R2 = 0.8173 +/- 0.0113, MSE = 0.2467 +/- 0.0087
- R2 margin (best KAN - best MLP): 0.0026
- KAN training speed ratio vs best matched MLP: 4.0524x

## Section B: Experiment 2 - Classification Results

| Name | Head Type | Mode | Hidden Dim | Total Params | Head Params | Test Accuracy (mean +/- std) | Test Loss (mean +/- std) | Convergence Epoch (mean +/- std) | Avg Time/Epoch (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BSplineMLP_ModeA | bspline_mlp | A | 256 | 1689610 | 136714 | 0.8680 +/- 0.0040 | 0.4170 +/- 0.0028 | 13 +/- 2 | 17.6 |
| KAN_ModeA | kan | A | 256 | 2889216 | 1336320 | 0.8696 +/- 0.0063 | 0.4534 +/- 0.0123 | 21 +/- 1 | 18.5 |
| MLP_ModeA | mlp | A | 256 | 1686794 | 133898 | 0.8782 +/- 0.0061 | 0.4078 +/- 0.0057 | 19 +/- 2 | 16.1 |
| BSplineMLP_ModeB | bspline_mlp | B | 256 | 1689610 | 136714 | 0.8680 +/- 0.0040 | 0.4170 +/- 0.0028 | 13 +/- 2 | 19.6 |
| KAN_ModeB | kan | B | 28 | 1699056 | 146160 | 0.8569 +/- 0.0089 | 0.4759 +/- 0.0139 | 15 +/- 1 | 20.2 |
| MLP_ModeB | mlp | B | 256 | 1686794 | 133898 | 0.8782 +/- 0.0061 | 0.4078 +/- 0.0057 | 19 +/- 2 | 16.2 |

- Mode A winner: `MLP_ModeA` with accuracy = 0.8782 +/- 0.0061
- Mode B winner: `MLP_ModeB` with accuracy = 0.8782 +/- 0.0061
- Mode A: MLP vs KAN accuracy difference = 0.0086
- Mode A: MLP vs BSpline-MLP accuracy difference = 0.0102
- Mode B: MLP vs KAN accuracy difference = 0.0213
- Mode B: MLP vs BSpline-MLP accuracy difference = 0.0102
- Training speed (avg_time_per_epoch_mean):
  - BSplineMLP_ModeA: 17.6 s/epoch
  - KAN_ModeA: 18.5 s/epoch
  - MLP_ModeA: 16.1 s/epoch
  - BSplineMLP_ModeB: 19.6 s/epoch
  - KAN_ModeB: 20.2 s/epoch
  - MLP_ModeB: 16.2 s/epoch
- Device used: not explicitly recorded in the CSV artifacts; only raw timing values are available in the result files.
- Mode B epochs_trained ranges across seeds:
  - BSplineMLP_ModeB: 29-34 epochs
  - KAN_ModeB: 28-37 epochs
  - MLP_ModeB: 40-50 epochs
- Learning rates used:
  - MLP_ModeA: 0.001
  - KAN_ModeA: 0.0005
  - BSplineMLP_ModeA: 0.002
  - MLP_ModeB: 0.001
  - KAN_ModeB: 0.002
  - BSplineMLP_ModeB: 0.002

## Section C: Experiment 3 - Interpretability Results

| Rank | KAN Nonlinearity | Score | SHAP Importance | Score |
|---:|---|---:|---|---:|
| 1 | AveOccup | 0.0258 | Latitude | 0.7289 |
| 2 | Population | 0.0240 | Longitude | 0.6754 |
| 3 | Latitude | 0.0206 | MedInc | 0.3564 |
| 4 | HouseAge | 0.0167 | AveOccup | 0.2498 |
| 5 | Longitude | 0.0121 | AveRooms | 0.1391 |
| 6 | AveBedrms | 0.0098 | HouseAge | 0.0630 |
| 7 | MedInc | 0.0066 | AveBedrms | 0.0415 |
| 8 | AveRooms | 0.0055 | Population | 0.0370 |

- Spearman rho = -0.048, p = 0.911
- Largest rank disagreement: `Population` with a rank difference of 6
- MedInc ranks: KAN rank 7, SHAP rank 3

## Section D: Hardware and Training Details

- Total training time for Experiment 1 (sum of all run `total_training_time` values): 7593.3 s (126.6 min, 2.11 h)
- Total training time for Experiment 2 (sum of all run `total_training_time` values): 12571.9 s (209.5 min, 3.49 h)
- Device used: not stored explicitly in the experiment result files.
- Number of seeds used in Experiment 1: 3 ([42, 123, 456])
- Number of seeds used in Experiment 2: 3 ([42, 123, 456])
- Epochs configured for Experiment 1: 200
- Epochs configured for Experiment 2: 50
