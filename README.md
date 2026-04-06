# MICS6 — ECDI2030 Modelling 
##  Documentation

**Project:** Modelling Early Childhood Development using the UNICEF ECDI2030 Framework  
**Data:** Nepal Multiple Indicator Cluster Survey Round 6 (MICS6), 2019–2021  
**Author:** Prachi  
**Date:** March 2026  

---

## 1. Project Overview

This project uses machine learning to model the distribution and predictors of early
childhood development in Nepal, using the UNICEF ECDI2030 framework as the primary
outcome measure. The work supports UNICEF's SDG Target 4.2 monitoring — tracking the
proportion of children aged 24–59 months who are developmentally on track across health,
learning, and psychosocial well-being.

### Research Questions
1. What proportion of Nepali children aged 24–59 months are developmentally on track?
2. Which household, caregiver, and child-level factors most strongly predict developmental outcomes?
3. Are there distinct developmental profiles (clusters) among children?
4. How do outcomes vary across provinces and wealth groups?

---

## 2. Data Source

### Nepal MICS6 (2019–2021)
- **Conducted by:** Central Bureau of Statistics, Nepal with UNICEF support
- **Target population:** Households with children under 18 and women aged 15–49
- **Sampling:** Multi-stage stratified cluster sampling
- **Download:** MICS portal (mics.unicef.org) — requires registered account

### Files Used

| File | Module | Rows | Columns |
|------|--------|------|---------|
| `ch.sav` | Children under 5 (primary analysis file) | 6,749 | 449 |
| `hh.sav` | Households | 12,800 | 471 |
| `wm.sav` | Women 15–49 (mother/caregiver data) | 15,019 | 414 |

### Merge Logic
```
ch.sav  ──(HH1, HH2)──▶  hh.sav   [household characteristics]
ch.sav  ──(HH1, HH2, UF4=WMLNA)──▶  wm.sav   [mother characteristics]
```

### Age Filter
ECDI2030 targets children aged **24–59 months**. Age in months is stored in `CAGE`.
After filtering: **4,178 children** (from 6,749 total under-5s).

---

## 3. ECDI2030 Framework

### Overview
The Early Childhood Development Index 2030 (ECDI2030) is UNICEF's official tool for
measuring SDG Indicator 4.2.1. It collects 10 binary items from mothers/caregivers.

### Domain Structure

| Domain | Variables | Items | On-track rule |
|--------|-----------|-------|---------------|
| **Literacy-Numeracy** | EC6, EC7, EC8 | Identifies ≥10 letters; reads ≥4 words; knows numbers 1–10 | Pass ALL 3 |
| **Physical** | EC9, EC10 | Picks up small object with 2 fingers; not too sick to play | Pass ALL 2 |
| **Learning** | EC11, EC12 | Follows simple directions; does something independently | Pass ALL 2 |
| **Socio-emotional** | EC13, EC14(R), EC15(R) | Gets along with peers; does NOT kick/bite/hit; not easily distracted | Pass ALL 3 |

> **EC14 and EC15 are reverse-coded**: value 1 = Yes (bad) → pass = 2 (No)

### Composite Score
A child is **"developmentally on track"** if on track in **≥ 3 of 4 domains**.

### Missing Data
Exactly **1,280 children** (30.6%) had missing ECDI data across all 10 items.
This uniform pattern indicates a systematic skip in data collection (likely children
for whom caregivers declined or were unavailable for the child questionnaire section).
After exclusion: **2,799 children** with complete ECDI2030 data used in ML analyses.

---

## 4. Key Results

### 4.1 ECDI2030 On-Track Rates

| Domain | % On Track | n |
|--------|-----------|---|
| Literacy-Numeracy | **26.9%** | 2,890 |
| Physical | **40.1%** | 2,881 |
| Learning | **75.5%** | 2,886 |
| Socio-emotional | **9.7%** | 2,828 |
| **Composite (≥3/4)** | **12.6%** | 2,799 |

**Notable finding:** The very low socio-emotional rate (9.7%) is largely driven by EC15
("child is easily distracted") — only ~23% of caregivers reported their child was NOT
easily distracted. This may reflect cultural response patterns or genuine prevalence.

### 4.2 Subgroup Comparisons

**By sex:**
| Sex | On-Track Rate |
|-----|--------------|
| Male | 11.8% |
| Female | 13.4% |

**By area:**
| Area | On-Track Rate |
|------|--------------|
| Urban | 13.8% |
| Rural | 11.0% |

**By wealth quintile:**
| Quintile | On-Track Rate |
|----------|--------------|
| 1 (Poorest) | 6.7% |
| 2 | 12.9% |
| 3 | 13.7% |
| 4 | 15.8% |
| 5 (Richest) | 17.9% |

**By province:**
| Province | On-Track Rate | n |
|----------|--------------|---|
| Bagmati | **19.3%** | 523 |
| Gandaki | 16.2% | 296 |
| Koshi | 12.0% | 417 |
| Lumbini | 12.8% | 399 |
| Madhesh | 9.7% | 487 |
| Sudurpashchim | 9.5% | 368 |
| Karnali | **6.5%** | 309 |

---

## 5. Pipeline Architecture

### File Structure
```
/Users/prachi/nepal_ecdi_project/
│
├── 01_data_loading.py       — loads ch/hh/wm, merges, age-filters, saves parquet
├── 02_ecdi_scoring.py       — computes ECDI domain scores + composite
├── 03_eda.py                — 7 exploratory figures
├── 04_supervised_ml.py      — ML models + SHAP (main analysis)
├── 05_clustering.py         — k-means developmental profiles
├── 06_geospatial.py         — province-level maps + Moran's I
├── run_all.py               — runs all 6 steps end-to-end
│
├── nepal_merged.parquet     — merged dataset (4,178 children, 83 cols)
├── nepal_scored.parquet     — with ECDI scores added
├── nepal_clustered.parquet  — with cluster labels added
├── X_imputed.parquet        — imputed feature matrix (ML input)
├── shap_importance.csv      — ranked SHAP feature importances
├── province_summary.csv     — province-level aggregates
│
└── figures/
    ├── 01_ecdi_domains.png        — domain on-track bar chart
    ├── 02_ecdi_wealth.png         — on-track by wealth quintile (line chart)
    ├── 03_ecdi_age.png            — on-track by age group (grouped bars)
    ├── 04_ecdi_province.png       — on-track by province (horizontal bar)
    ├── 05_ecdi_area_sex.png       — urban/rural × sex (grouped bar)
    ├── 06_stimulation.png         — mother stimulation count vs on-track
    ├── 07_correlation.png         — correlation heatmap (domains + predictors)
    ├── 08_roc_curve.png           — cross-validated ROC (both models)
    ├── 08b_confusion_matrix.png   — OOF confusion matrix (HGB)
    ├── 09_shap_bar.png            — SHAP mean |value| bar chart
    ├── 10_shap_beeswarm.png       — SHAP beeswarm (direction + magnitude)
    ├── 11_cluster_selection.png   — elbow + silhouette for optimal k
    ├── 12_cluster_profiles.png    — domain rates per cluster
    ├── 13_cluster_pca.png         — PCA scatter coloured by cluster
    ├── 14_province_heatmap.png    — province × domain heatmap
    ├── 15_province_bubble.png     — wealth vs ECDI bubble chart
    └── 16_province_inequality.png — between-province range chart
```

---

## 6. Supervised ML Analysis

### 6.1 Features (21 total)

| Feature | Description | Type |
|---------|-------------|------|
| `CAGE` | Child age in months | Continuous |
| `is_female` | Sex (1=female) | Binary |
| `is_urban` | Urban area (1=urban) | Binary |
| `windex5` | Household wealth quintile (1–5) | Ordinal |
| `HAZ2` | Height-for-age z-score (WHO) | Continuous |
| `WAZ2` | Weight-for-age z-score (WHO) | Continuous |
| `stim_mother` | # of 6 stimulation activities by mother | Count (0–6) |
| `stim_father` | # of 6 stimulation activities by father | Count (0–6) |
| `stim_other` | # done by other adult | Count (0–6) |
| `stim_none` | # activities done by no one | Count (0–6) |
| `ece_attending` | Currently attending ECE programme | Binary |
| `has_books` | Has children's books at home | Binary |
| `has_toys` | Has toys | Binary |
| `mother_higher_ed` | Mother has higher education (level ≥3) | Binary |
| `UCD5` | Caregiver believes physical punishment needed | Binary |
| `prov_*` | Province dummies (Bagmati = reference) | Binary ×6 |

**Stimulation activities (EC5 series):** reading books, telling stories, singing songs,
taking outside, playing with child, naming/counting things — each coded separately
for mother, father, other adult, or no one.

### 6.2 Class Imbalance Handling
With only 12.6% on-track (352/2,799), both models use `class_weight="balanced"`,
which inversely weights classes by frequency (~7× weight on on-track children).

### 6.3 Missing Data
Imputed with **median imputation** (SimpleImputer). Main missing sources:
- `HAZ2`/`WAZ2`: ~3% missing (children not measured)
- `ece_attending`: ~2% missing

### 6.4 Model Comparison (5-fold stratified CV)

| Model | ROC-AUC | F1 |
|-------|---------|-----|
| Logistic Regression | **0.704 ± 0.018** | **0.314 ± 0.017** |
| HistGradientBoosting | 0.663 ± 0.036 | 0.268 ± 0.043 |

> Logistic Regression outperforms HGB here — likely because with 21 features and
> a small on-track class (352 children), the simpler linear model generalises better.
> This is a common finding in small, structured survey datasets.

### 6.5 HistGradientBoosting — OOF Classification Report

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Not on track | 0.90 | 0.81 | 0.85 |
| On track | 0.21 | 0.36 | 0.27 |
| **Accuracy** | | | **0.75** |

### 6.6 SHAP Feature Importance (Top 10)

Ranked by mean absolute SHAP value (impact on on-track probability):

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 1 | Currently attending ECE | 0.079 |
| 2 | Child age (months) | 0.073 |
| 3 | Height-for-age z-score | 0.071 |
| 4 | Has children's books | 0.062 |
| 5 | Weight-for-age z-score | 0.051 |
| 6 | Wealth quintile | 0.031 |
| 7 | Province: Koshi (vs Bagmati) | 0.024 |
| 8 | Mother: higher education | 0.021 |
| 9 | Sex (female) | 0.019 |
| 10 | Province: Sudurpashchim (vs Bagmati) | 0.012 |

**Key interpretation:**
- **ECE attendance** is the strongest predictor — attending an early childhood education
  programme significantly increases the probability of being on track
- **Nutritional status** (HAZ2, WAZ2) is a major driver, reflecting the well-established
  link between undernutrition and cognitive/developmental delay
- **Learning materials at home** (books) matter independently of wealth
- **Caregiver stimulation** (mother/father activities) shows near-zero SHAP — this may
  reflect measurement issues (EC5 asks about "last 3 days") or multicollinearity with
  other SES variables

---

## 7. Unsupervised ML — Developmental Profiles

### 7.1 Method
K-Means clustering on the 4 ECDI domain scores (2,799 complete-case children).
Optimal k selected using silhouette score (k=8 identified as optimal).

### 7.2 Cluster Profiles

| Cluster Label | Literacy-Num | Physical | Learning | Socio-emotional | Size |
|---------------|-------------|---------|---------|----------------|------|
| High overall | 100% | 100% | 100% | ~5% | 262 |
| Learning only | 0% | 0% | 100% | 0% | 817 |
| Physical + Learning | 0% | 100% | 100% | ~9% | 581 |
| Literacy + Learning | 100% | 0% | 100% | 0% | 348 |
| Physical only | 0% | 100% | 0% | ~11% | 218 |
| Low overall | 0% | 0% | 0% | 0% | 290 |
| Mixed physical | 21% | 0% | 71% | 100% | 167 |
| Literacy partial | 100% | 43% | 0% | ~10% | 116 |

**Key finding:** The largest cluster (n=817) is "Learning only" — children on track in
learning (follows directions, acts independently) but delayed in literacy-numeracy,
physical, and socio-emotional domains. This is the most common developmental profile
in Nepal.

**"Low overall"** cluster (n=290): concentrated in Karnali and Sudurpashchim provinces,
poorest wealth quintile, lowest HAZ scores — the most vulnerable group.

---

## 8. Geospatial Analysis

### 8.1 Province-Level Summary

| Province | ECDI (%) | n | Wealth (mean Q) | Urban % | HAZ (mean) |
|----------|---------|---|----------------|---------|------------|
| Bagmati | **19.3** | 523 | 3.48 | 68.6% | -0.03 |
| Gandaki | 16.2 | 296 | 2.95 | 55.0% | -0.22 |
| Koshi | 12.0 | 417 | 2.56 | 49.8% | -0.17 |
| Lumbini | 12.8 | 399 | 2.74 | 49.8% | -1.08 |
| Madhesh | 9.7 | 487 | 3.05 | 58.2% | 2.36 |
| Karnali | **6.5** | 309 | 1.33 | 52.5% | 0.39 |
| Sudurpashchim | 9.5 | 368 | 2.06 | 52.2% | -1.04 |

### 8.2 Spatial Autocorrelation
**Moran's I = 0.098** (p = 0.169) — no statistically significant spatial
autocorrelation at province level. This is expected given only 7 provinces (low power).
District-level data would be needed for meaningful spatial analysis.

**Note:** A choropleth map requires the Nepal province shapefile from GADM
(gadm.org/country/NPL). Download `gadm41_NPL_1.shp` and update `06_geospatial.py`
with the file path to generate the map.

---

## 9. Technical Notes

### 9.1 Software
| Package | Purpose | Version |
|---------|---------|---------|
| `pyreadstat` | Load SPSS .sav files | — |
| `pandas` | Data manipulation | — |
| `numpy` | Numerical operations | — |
| `scikit-learn` | ML models, CV, imputation | — |
| `shap` | Model interpretability | — |
| `matplotlib` / `seaborn` | Visualisation | — |
| `geopandas` / `libpysal` / `esda` | Geospatial analysis | — |
| `scipy` | Hierarchical clustering | — |

### 9.2 Known Limitations

1. **~30.6% missing ECDI data** — systematic missingness; complete-case analysis
   used. Missing at random (MAR) assumption not verified.

2. **Proxy stimulation measure** — EC5 asks about stimulation in the "last 3 days"
   only, which is a noisy snapshot of habitual caregiver behaviour.

3. **Socio-emotional domain** — 9.7% on-track rate is very low and may partly
   reflect cultural differences in how EC14/EC15 items are interpreted.

4. **No causal identification** — SHAP gives association-based importance, not
   causal effects. DAG-based causal analysis is a recommended next step.

5. **Province-level spatial analysis** — only 7 units, insufficient for Moran's I.
   District-level analysis would require PSU-level data and appropriate shapefiles.

6. **HGB underperforms LR** — with small positive class (n=352) and tabular survey
   data, gradient boosting may overfit. Regularisation tuning recommended.

### 9.3 Next Steps

- [ ] Multiple imputation (mice) instead of single median imputation
- [ ] Download GADM shapefile for Nepal choropleth maps
- [ ] Causal DAG analysis using DoWhy
- [ ] Subgroup analysis: under-3 vs 3-5 year olds separately
- [ ] Random effects / multilevel model to account for cluster sampling design
- [ ] Extend to India using NFHS-5 proxy ECDI (pending DHS data approval)

---

## 10. How to Run

```bash
# Run entire pipeline from scratch
cd /Users/prachi/nepal_ecdi_project
python3 run_all.py

# Or run individual steps
python3 01_data_loading.py    # ~10 seconds
python3 02_ecdi_scoring.py    # ~5 seconds
python3 03_eda.py             # ~30 seconds
python3 04_supervised_ml.py   # ~3 minutes (SHAP computation)
python3 05_clustering.py      # ~30 seconds
python3 06_geospatial.py      # ~15 seconds
```

**Data path:** `/Users/prachi/Downloads/MICS_Datasets/Nepal MICS6 Datasets/Nepal MICS6 SPSS Datasets/`  
**Output path:** `/Users/prachi/nepal_ecdi_project/`

---


