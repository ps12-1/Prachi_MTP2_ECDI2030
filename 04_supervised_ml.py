"""
Nepal MICS6 — Supervised ML: Predicting ECDI2030 On-Track Status
=================================================================
Models:
  1. Logistic Regression (baseline, interpretable)
  2. HistGradientBoostingClassifier (sklearn gradient boosting)

Evaluation:
  - 5-fold stratified cross-validation (ROC-AUC, F1)
  - Cross-validated ROC curve (honest out-of-fold predictions)
  - Confusion matrix on OOF predictions

Interpretability:
  - SHAP values via PermutationExplainer (subsample=500)
  - Global importance bar chart
  - Beeswarm plot (direction + magnitude)

Note on class imbalance:
  ~12.6% of children are "on track" — class_weight="balanced" used in both
  models to prevent trivial all-zeros predictions.

Note on training classification report:
  The report at the end uses the full training set; CV metrics are
  the honest estimate of generalisation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, roc_auc_score,
                             roc_curve, ConfusionMatrixDisplay, confusion_matrix)
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
import shap

OUT = Path("/Users/prachi/nepal_ecdi_project/figures")
OUT.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.1)

# ── Load data ─────────────────────────────────────────────────────────────────

df = pd.read_parquet("/Users/prachi/nepal_ecdi_project/nepal_scored.parquet")
df = df[df["ecdi_composite"].notna()].copy()
print(f"Analysis sample: {len(df):,} children with complete ECDI2030")

# ── 1. Feature engineering ────────────────────────────────────────────────────

# Caregiver stimulation counts (0–6 activities)
stim_cols = {
    "stim_mother": ["EC5AA","EC5BA","EC5CA","EC5DA","EC5EA","EC5FA"],
    "stim_father": ["EC5AB","EC5BB","EC5CB","EC5DB","EC5EB","EC5FB"],
    "stim_other":  ["EC5AX","EC5BX","EC5CX","EC5DX","EC5EX","EC5FX"],
    "stim_none":   ["EC5AY","EC5BY","EC5CY","EC5DY","EC5EY","EC5FY"],
}
for name, cols in stim_cols.items():
    present = [c for c in cols if c in df.columns]
    df[name] = df[present].apply(lambda row: (row == 1).sum(), axis=1)

# ECE attendance (current)
if "UB8" in df.columns:
    df["ece_attending"] = (df["UB8"] == 1).astype(float)
elif "UB7" in df.columns:
    df["ece_attending"] = (df["UB7"] == 1).astype(float)
else:
    df["ece_attending"] = np.nan

# Learning materials
if "EC1" in df.columns:
    df["has_books"] = (df["EC1"] >= 1).astype(float)
if all(c in df.columns for c in ["EC2A","EC2B","EC2C"]):
    df["has_toys"] = df[["EC2A","EC2B","EC2C"]].apply(
        lambda row: int(any(row == 1)), axis=1).astype(float)

# Child / household characteristics
df["is_female"] = (df["sex"] == "Female").astype(float)
df["is_urban"]  = (df["area"] == "Urban").astype(float)

# Province one-hot (Bagmati is reference / dropped)
df = pd.get_dummies(df, columns=["province"], prefix="prov",
                    drop_first=True, dummy_na=False)

# Mother education: 0=none/pre-primary, 1=primary, 2=secondary, 3+=higher
if "melevel1" in df.columns:
    df["mother_higher_ed"] = (df["melevel1"] >= 3).astype(float)
elif "wm_melevel1" in df.columns:
    df["mother_higher_ed"] = (df["wm_melevel1"] >= 3).astype(float)

# ── 2. Feature matrix ─────────────────────────────────────────────────────────

base_features = [
    "CAGE",            # child age in months
    "is_female",       # sex
    "is_urban",        # urban/rural
    "windex5",         # household wealth quintile
    "HAZ2",            # height-for-age z-score (stunting)
    "WAZ2",            # weight-for-age z-score (underweight)
    "stim_mother",     # # of 6 stimulation activities done by mother
    "stim_father",     # # of 6 stimulation activities done by father
    "stim_other",      # # done by other adult
    "stim_none",       # # activities done by no one (deprivation count)
    "ece_attending",   # currently attending ECE programme
]
optional = ["has_books", "has_toys", "mother_higher_ed", "UCD5"]
prov_dummies = [c for c in df.columns if c.startswith("prov_")]

feature_cols = (base_features
                + [c for c in optional if c in df.columns]
                + prov_dummies)
feature_cols = [c for c in feature_cols if c in df.columns]

print(f"Features: {len(feature_cols)}")

# Cast to float (province dummies may be bool)
X = df[feature_cols].copy().astype(float)
y = df["ecdi_composite"].astype(int)
print(f"On-track rate: {y.mean()*100:.1f}%  |  "
      f"Class 0: {(y==0).sum():,}  Class 1: {(y==1).sum():,}")

# Imputed matrix (needed for SHAP; HGB handles NaN natively but SHAP doesn't)
imputer = SimpleImputer(strategy="median")
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

# ── 3. Cross-validation setup ─────────────────────────────────────────────────

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── 4. Logistic Regression baseline ──────────────────────────────────────────
# Pipeline handles its own imputation — pass raw X (with NaN), not X_imp

lr_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
    ("model",  LogisticRegression(max_iter=1000, random_state=42,
                                  C=1.0, class_weight="balanced")),
])

lr_auc = cross_val_score(lr_pipe, X, y, cv=cv, scoring="roc_auc")
lr_f1  = cross_val_score(lr_pipe, X, y, cv=cv, scoring="f1")
lr_oof = cross_val_predict(lr_pipe, X, y, cv=cv, method="predict_proba")[:, 1]

print(f"\nLogistic Regression (5-fold CV)")
print(f"  AUC: {lr_auc.mean():.3f} ± {lr_auc.std():.3f}")
print(f"  F1:  {lr_f1.mean():.3f} ± {lr_f1.std():.3f}")

# ── 5. HistGradientBoosting ───────────────────────────────────────────────────

hgb = HistGradientBoostingClassifier(
    max_iter=300,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    class_weight="balanced",
)

hgb_auc = cross_val_score(hgb, X_imp, y, cv=cv, scoring="roc_auc")
hgb_f1  = cross_val_score(hgb, X_imp, y, cv=cv, scoring="f1")
hgb_oof = cross_val_predict(hgb, X_imp, y, cv=cv, method="predict_proba")[:, 1]

print(f"\nHistGradientBoosting (5-fold CV)")
print(f"  AUC: {hgb_auc.mean():.3f} ± {hgb_auc.std():.3f}")
print(f"  F1:  {hgb_f1.mean():.3f} ± {hgb_f1.std():.3f}")

# ── 6. Cross-validated ROC curve ─────────────────────────────────────────────
# Uses out-of-fold (OOF) predictions — honest generalisation estimate

fig, ax = plt.subplots(figsize=(7, 6))
for oof, label, color in [
    (lr_oof,  f"Logistic Regression (AUC={roc_auc_score(y, lr_oof):.3f})",  "#C44E52"),
    (hgb_oof, f"HistGradientBoosting (AUC={roc_auc_score(y, hgb_oof):.3f})", "#4C72B0"),
]:
    fpr, tpr, _ = roc_curve(y, oof)
    ax.plot(fpr, tpr, lw=2, label=label, color=color)
ax.plot([0,1],[0,1], "k--", lw=1, label="Random classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Cross-validated ROC — ECDI2030 On-Track Prediction\n"
             "(out-of-fold predictions, 5-fold CV)", fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
fig.savefig(OUT / "08_roc_curve.png", dpi=150)
plt.close()
print("\nSaved 08_roc_curve.png")

# ── 7. Confusion matrix (OOF, HGB) ───────────────────────────────────────────

hgb_oof_pred = (hgb_oof >= 0.5).astype(int)
cm = confusion_matrix(y, hgb_oof_pred)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(cm, display_labels=["Not on track","On track"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix — HistGradientBoosting (OOF)", fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "08b_confusion_matrix.png", dpi=150)
plt.close()
print("Saved 08b_confusion_matrix.png")

print("\nClassification Report (OOF predictions — HistGradientBoosting):")
print(classification_report(y, hgb_oof_pred,
                             target_names=["Not on track","On track"]))

# ── 8. Fit final model on full data (for SHAP) ────────────────────────────────

hgb_final = HistGradientBoostingClassifier(
    max_iter=300, max_depth=5, learning_rate=0.05,
    random_state=42, class_weight="balanced",
)
hgb_final.fit(X_imp, y)

# ── 9. SHAP — subsample 500 rows for speed ────────────────────────────────────

print("\nComputing SHAP values (n=500 subsample)...")
rng = np.random.default_rng(42)
idx = rng.choice(len(X_imp), size=min(500, len(X_imp)), replace=False)
X_shap = X_imp.iloc[idx].reset_index(drop=True)

masker   = shap.maskers.Independent(X_shap, max_samples=50)
explainer = shap.PermutationExplainer(hgb_final.predict_proba, masker)
sv_raw   = explainer(X_shap)
sv       = sv_raw[..., 1]   # class-1 (on-track) SHAP values

# Human-readable feature names
feature_labels = {
    "CAGE":            "Child age (months)",
    "is_female":       "Sex (female=1)",
    "is_urban":        "Urban area",
    "windex5":         "Wealth quintile",
    "HAZ2":            "Height-for-age z-score",
    "WAZ2":            "Weight-for-age z-score",
    "stim_mother":     "Mother stimulation (# of 6 activities)",
    "stim_father":     "Father stimulation (# of 6 activities)",
    "stim_other":      "Other adult stimulation",
    "stim_none":       "Activities done by no one (deprivation)",
    "ece_attending":   "Currently attending ECE",
    "has_books":       "Has children's books",
    "has_toys":        "Has toys",
    "mother_higher_ed":"Mother: higher education",
    "UCD5":            "Belief: child needs physical punishment",
    "prov_Gandaki":    "Province: Gandaki (vs Bagmati)",
    "prov_Karnali":    "Province: Karnali (vs Bagmati)",
    "prov_Koshi":      "Province: Koshi (vs Bagmati)",
    "prov_Lumbini":    "Province: Lumbini (vs Bagmati)",
    "prov_Madhesh":    "Province: Madhesh (vs Bagmati)",
    "prov_Sudurpashchim": "Province: Sudurpashchim (vs Bagmati)",
}
display_names = [feature_labels.get(c, c) for c in feature_cols]
sv.feature_names = display_names

# SHAP bar chart
fig, ax = plt.subplots(figsize=(10, 7))
mean_shap = pd.Series(
    np.abs(sv.values).mean(axis=0),
    index=display_names,
    name="mean_shap",
).sort_values(ascending=True)
mean_shap.plot(kind="barh", ax=ax, color="#4C72B0")
ax.set_xlabel("Mean |SHAP value| (impact on on-track probability)")
ax.set_title("Feature Importance (SHAP) — ECDI2030 On-Track\n"
             "HistGradientBoosting, n=500 subsample", fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "09_shap_bar.png", dpi=150)
plt.close()
print("Saved 09_shap_bar.png")

# SHAP beeswarm
fig = plt.figure(figsize=(11, 8))
shap.plots.beeswarm(sv, max_display=15, show=False)
plt.title("SHAP Beeswarm — Direction & Magnitude of Predictors\n"
          "(red = high feature value, blue = low)", fontweight="bold", pad=15)
plt.tight_layout()
fig.savefig(OUT / "10_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 10_shap_beeswarm.png")

# ── 10. Save outputs ──────────────────────────────────────────────────────────

importance_df = (mean_shap
                 .sort_values(ascending=False)
                 .reset_index()
                 .rename(columns={"index": "feature", "mean_shap": "mean_abs_shap"}))
importance_df.to_csv("/Users/prachi/nepal_ecdi_project/shap_importance.csv", index=False)

X_imp.to_parquet("/Users/prachi/nepal_ecdi_project/X_imputed.parquet", index=False)

# ── 11. Summary table ─────────────────────────────────────────────────────────

print("\n" + "="*55)
print("MODEL COMPARISON — 5-FOLD CROSS-VALIDATED PERFORMANCE")
print("="*55)
print(f"{'Model':<30} {'ROC-AUC':>9} {'F1':>8}")
print("-"*55)
print(f"{'Logistic Regression':<30} "
      f"{lr_auc.mean():>7.3f}±{lr_auc.std():.3f}  "
      f"{lr_f1.mean():>6.3f}±{lr_f1.std():.3f}")
print(f"{'HistGradientBoosting':<30} "
      f"{hgb_auc.mean():>7.3f}±{hgb_auc.std():.3f}  "
      f"{hgb_f1.mean():>6.3f}±{hgb_f1.std():.3f}")
print("\nTop 5 SHAP predictors:")
print(importance_df.head(5).to_string(index=False))
print("\nSaved: shap_importance.csv, X_imputed.parquet")
