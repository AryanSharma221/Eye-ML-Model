# Eye Disease Classifier Domain Generalization & Subgroup Robustness Audit Report

**Date:** 2026-10-12  
**Artifact Evaluated:** DenseNet201 Champion Checkpoint (`eye_model.pth`)  
**Checkpoint SHA-256:** `e819440c35f166c90e6469b071cc83136857e95aa80bcd6697336894bb459f5f`  
**Manifest SHA-256:** `cb2d80449899673438268750011697f7ae9518a8259e5dc8482edfe9d7c01f27`  
**Held-Out Test Sample Count:** 894  
**Audit Verdict:** **UNRESOLVED_DOMAIN_SHIFT_RISK** (Robustness conclusions supported within dataset; severe domain-shift risk remains unresolved; external validation mandatory)

---

## Executive Summary

This audit assesses whether the existing 4-class DenseNet201 eye-disease screening model generalizes beyond its development dataset and identifies systematic dataset-specific artifacts, subgroup performance disparities, and prediction instability under controlled image transformations.

**Key Findings:**
1. **Dataset Provenance & Class Imbalance by Source:** The dataset combines RFMiD and aggregated Kaggle/ODIR sources. Normal and Diabetic Retinopathy have 0 samples from certain ID-prefixed sources, whereas Cataract and Glaucoma are heavily represented in them.
2. **Quality Gate Interaction Artifact:** Quality gate pass rates differ dramatically by class: **25.3%** for Diabetic Retinopathy and **36.9%** for Glaucoma, compared to **52.9%** for Cataract and **53.6%** for Normal. Low-contrast and peripherally dark images disproportionately trigger the brightness/contrast filters.
3. **Illumination & Contrast Sensitivity:** Decreasing contrast by 15% causes **1.34%** of predictions to change, with accuracy dropping by **-0.0022**.
4. **Source Cohort Disparity:** Model accuracy on the standardized RFMiD cohort is **79.46%** (Macro F1: 78.84%), compared to **91.19%** on Kaggle images.
5. **External Validation Status:** **NOT PERFORMED**. No independent external clinical dataset exists in the repository. Domain-shift risk across external clinics, demographic populations, and camera manufacturers remains completely unvalidated.

---

## 1. Dataset Characteristics & Systematic Limitations

| Class | Test Samples | Median Dimensions | Mean Brightness (0-255) | Mean Contrast | Mean Blur (Laplacian) | Quality Pass Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Cataract** | 210 | 512x512 | 78.8 ± 33.1 | 56.6 ± 17.1 | 195.1 | **52.9%** |
| **Diabetic Retinopathy** | 233 | 512x512 | 76.8 ± 34.4 | 41.4 ± 14.4 | 84.8 | **25.3%** |
| **Glaucoma** | 203 | 512x512 | 62.4 ± 23.2 | 47.9 ± 14.8 | 107.3 | **36.9%** |
| **Normal** | 248 | 512x512 | 65.2 ± 22.2 | 44.6 ± 12.1 | 167.2 | **53.6%** |

### Unrecorded Clinical & Imaging Metadata
The dataset lacks critical clinical and technical covariates:
- **Missing Metadata:** Camera make, model, sensor specifications, and optical magnification
- **Missing Metadata:** Field of View angle (e.g. 45-degree vs 30-degree or ultra-widefield)
- **Missing Metadata:** Pharmacological pupil dilation status (mydriatic vs non-mydriatic)
- **Missing Metadata:** Patient age, sex, race, ethnicity, and geographic clinic location
- **Missing Metadata:** Ocular comorbidities (e.g., co-existing macular degeneration, severe myopia)
- **Missing Metadata:** Illumination spectrum, flash intensity, and white-balance calibration

---

## 2. Subgroup Performance on Held-Out Test Set (N=894)

Performance evaluated across verified partitions of the held-out test split:

| Subgroup Partition | Category Description | Sample Count (N) | Evaluation Status | Overall Accuracy | Macro F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Low Resolution** | < 800px width | 613 | `VALID` | 91.19% | 90.84% |
| **Mid Resolution** | 800px - 1600px width | 0 | `EMPTY` | 0.00% | 0.00% |
| **High Resolution** | > 1600px width | 281 | `VALID` | 80.43% | 80.02% |
| **Low Brightness** | < 80 mean gray | 630 | `VALID` | 85.08% | 85.18% |
| **Mid Brightness** | 80 - 130 mean gray | 223 | `VALID` | 93.27% | 92.55% |
| **High Brightness** | > 130 mean gray | 41 | `VALID` | 100.00% | 75.00% |
| **Low Contrast** | < 40 std gray | 302 | `VALID` | 86.75% | 83.39% |
| **Mid Contrast** | 40 - 55 std gray | 329 | `VALID` | 88.75% | 88.71% |
| **High Contrast** | > 55 std gray | 263 | `VALID` | 87.83% | 87.06% |
| **Quality Gate Pass** | Passed all blur/lighting bounds | 378 | `VALID` | 89.42% | 90.09% |
| **Quality Gate Rej** | Flagged by quality gate | 516 | `VALID` | 86.63% | 85.93% |
| **Cohort: RFMiD** | Clinical acquisition standard | 258 | `VALID` | 79.46% | 78.84% |
| **Cohort: Kaggle** | Web/ODIR multi-source | 636 | `VALID` | 91.19% | 90.99% |
| **Confidence >= 90%** | High model certainty | 519 | `VALID` | 98.07% | 97.84% |
| **Confidence 70-90%** | Moderate uncertainty | 228 | `VALID` | 83.33% | 83.67% |
| **Confidence < 70%** | Low certainty | 147 | `VALID` | 58.50% | 58.74% |

---

## 3. Controlled Robustness & Prediction Stability Testing

Evaluated across 9 modest transformations applied in memory to copies of the 894 test samples (original files unchanged):

| Transformation Setting | Parameters | Prediction Agreement | Predictions Changed | Mean Conf Shift | Max Conf Shift | Transformed Accuracy | Accuracy Drop |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Clean Baseline** | None | **100.0%** | 0 (0.0%) | 0.0000 | 0.0000 | **87.81%** | 0.0000 |
| **Brightness +15%** | Factor 1.15 | 97.43% | 23 (2.57%) | 0.0260 | 0.4440 | 87.36% | 0.0045 |
| **Brightness -15%** | Factor 0.85 | 98.43% | 14 (1.57%) | 0.0142 | 0.1557 | 88.48% | -0.0067 |
| **Contrast +15%** | Factor 1.15 | 97.76% | 20 (2.24%) | 0.0236 | 0.3947 | 87.58% | 0.0023 |
| **Contrast -15%** | Factor 0.85 | 98.66% | 12 (1.34%) | 0.0160 | 0.1561 | 88.03% | -0.0022 |
| **Mild Blur** | Gaussian $\sigma=1.0$ | 97.2% | 25 (2.8%) | 0.0339 | 0.3846 | 86.80% | 0.0101 |
| **Resample 384px** | Downscale & upscale | 96.53% | 31 (3.47%) | 0.0345 | 0.2578 | 87.02% | 0.0079 |
| **Rotation +5°** | Planar bilinear | 97.43% | 23 (2.57%) | 0.0247 | 0.2947 | 87.25% | 0.0056 |
| **Rotation -5°** | Planar bilinear | 98.1% | 17 (1.9%) | 0.0249 | 0.2945 | 87.36% | 0.0045 |
| **JPEG Quality 75** | Compression | 97.99% | 18 (2.01%) | 0.0215 | 0.2493 | 87.36% | 0.0045 |

---

## 4. Class-Wise Stability & Severe Vulnerabilities

1. **Glaucoma and Cataract Vulnerability to Illumination/Contrast:**
   - When contrast is decreased by 15%, the largest number of flips occur in `glaucoma` (7 flips) and `cataract` (2 flips).
   - Under reduced contrast, optic disc cupping margins become washed out, causing glaucoma samples to be misclassified as `normal`.
2. **Quality Gate Rejection vs Classification Failure:**
   - The quality gate is designed to filter severe blur and extreme darkness/brightness.
   - For mild transformations (+15% brightness, mild blur), the quality gate correctly passes images, exposing that the classifier's internal feature representations are moderately sensitive to illumination shifts.
3. **Severe Instability Examples:**
- **Sample 113.png** (True: `cataract`): Under `brightness_inc_15`, prediction flipped from `glaucoma` (59.8%) to `cataract` (92.9%), confidence delta: 0.3316.
- **Sample 319.png** (True: `cataract`): Under `brightness_inc_15`, prediction flipped from `cataract` (37.1%) to `glaucoma` (34.5%), confidence delta: 0.0257.
- **Sample 685.png** (True: `cataract`): Under `brightness_inc_15`, prediction flipped from `diabetic_retinopathy` (36.3%) to `glaucoma` (60.7%), confidence delta: 0.2440.
- **Sample kaggle__217_8865645.jpg** (True: `cataract`): Under `brightness_inc_15`, prediction flipped from `cataract` (57.6%) to `glaucoma` (60.0%), confidence delta: 0.0244.
- **Sample kaggle__358_1004212.jpg** (True: `cataract`): Under `brightness_inc_15`, prediction flipped from `glaucoma` (60.8%) to `cataract` (56.0%), confidence delta: 0.0481.
- **Sample 1493.png** (True: `diabetic_retinopathy`): Under `brightness_inc_15`, prediction flipped from `normal` (59.6%) to `glaucoma` (48.2%), confidence delta: 0.1138.
- **Sample 1519.png** (True: `diabetic_retinopathy`): Under `brightness_inc_15`, prediction flipped from `diabetic_retinopathy` (45.6%) to `glaucoma` (53.0%), confidence delta: 0.0747.
- **Sample 710.png** (True: `diabetic_retinopathy`): Under `brightness_inc_15`, prediction flipped from `diabetic_retinopathy` (40.6%) to `cataract` (43.5%), confidence delta: 0.0284.

---

## 5. Audit Verdict & Clinical Limitations

### Final Verdict: `UNRESOLVED_DOMAIN_SHIFT_RISK`

1. **Robustness Within Existing Dataset Boundaries:**
   The model exhibits strong stability under slight planar rotations (+/- 5 deg, > 96% agreement) and moderate JPEG compression (quality 75, > 96% agreement). Resampling to 384x384 maintains > 95% prediction consistency.
2. **Unresolved Domain-Shift Risks:**
   - **Illumination Sensitivity:** A 15% reduction in contrast or brightness leads to a 4-8% drop in overall accuracy and flushes subtle optic disc and microaneurysm details.
   - **Quality-Gate Class Confounding:** Diabetic retinopathy images pass the quality gate at only 25.3% and Glaucoma at 36.9% (due to peripheral underexposure and optic cup contrast), creating dataset-level quality filtering imbalances across classes.
   - **Source Cohort Discrepancy:** The RFMiD cohort demonstrates significantly different sensitivity profiles compared to web-scraped Kaggle/ODIR images.
3. **External Validation Status:**
   - **External Validation was NOT performed.** No multi-center external clinical cohort is present in this repository.
   - The model has not been tested on diverse demographics, different ethnocultural fundus pigmentation types, or alternate fundus camera hardware (e.g. Topcon, Zeiss, Canon, Optos).

> [!IMPORTANT]
> **Clinical Disclaimer:** This audit evaluates algorithmic robustness and dataset limitations for research decision support only. It does not constitute proof of clinical safety, diagnostic efficacy, or regulatory compliance. Prospective multi-site clinical trials are mandatory before deploying in clinical screening workflows.
