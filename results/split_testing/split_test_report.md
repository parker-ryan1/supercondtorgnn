# Split Testing Comprehensive Report

**Generated**: 2025-10-30 16:49:15

## 1. Split Ratio Testing

| Split Config | Train | Val | Test | R² Score | MAE (K) | RMSE (K) |
|--------------|-------|-----|------|----------|---------|----------|
| 60/20/20 | 1033 | 344 | 345 | 0.9217 | 9.22 | 17.95 |
| 70/15/15 | 1205 | 258 | 259 | 0.9401 | 9.57 | 16.86 |
| 80/10/10 | 1377 | 172 | 173 | 0.9334 | 7.64 | 15.53 |
| 85/10/5 | 1463 | 172 | 87 | 0.8784 | 9.71 | 21.56 |
| 70/20/10 | 1205 | 344 | 173 | 0.9249 | 9.97 | 18.51 |

**Best Split**: 70/15/15 (R²=0.9401)

## 2. K-Fold Cross-Validation

**K**: 5

**Average R²**: 0.9196 ± 0.0038

**Average MAE**: 9.43 ± 0.64K

| Fold | R² Score | MAE (K) | RMSE (K) | N Test |
|------|----------|---------|----------|--------|
| 1 | 0.9228 | 10.38 | 18.54 | 345 |
| 2 | 0.9209 | 8.61 | 17.80 | 345 |
| 3 | 0.9221 | 9.96 | 18.69 | 344 |
| 4 | 0.9201 | 9.15 | 17.78 | 344 |
| 5 | 0.9122 | 9.07 | 19.02 | 344 |

## 3. Stratified K-Fold Cross-Validation

**K**: 5

**Average R²**: 0.9267 ± 0.0205

**Average MAE**: 8.85 ± 1.23K

## 4. Random Seed Stability Analysis

**Number of Seeds Tested**: 3

**Average R²**: 0.9307 ± 0.0023

**Average MAE**: 8.31 ± 0.47K

**Coefficient of Variation (R²)**: 0.25%

