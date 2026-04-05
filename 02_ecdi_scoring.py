"""
Nepal MICS6 — ECDI2030 Scoring
================================
Computes official ECDI2030 domain scores and composite index.

ECDI2030 Domains (children aged 24-59 months):
  Literacy-Numeracy : EC6, EC7, EC8  → on track if ALL pass
  Physical          : EC9, EC10      → on track if ALL pass
  Learning          : EC11, EC12     → on track if ALL pass
  Socio-emotional   : EC13, EC14(R), EC15(R) → on track if ALL pass

EC14 and EC15 are REVERSE coded (1=yes is BAD):
  EC14: kicks/bites/hits → on track if EC14 == 2 (No)
  EC15: easily distracted → on track if EC15 == 2 (No)

Composite: child is "developmentally on track" if on track in >= 3 of 4 domains.

Reference: ECDI2030 Technical Manual (UNICEF, Sept 2023)
"""

import pandas as pd
import numpy as np

df = pd.read_parquet("/Users/prachi/nepal_ecdi_project/nepal_merged.parquet")
print(f"Loaded: {df.shape[0]:,} children aged 24-59 months")

# ── 1. ECDI item value distributions ─────────────────────────────────────────

ecdi_items = ["EC6","EC7","EC8","EC9","EC10","EC11","EC12","EC13","EC14","EC15"]
print("\nECDI item value counts (1=Yes, 2=No, 9=DK/Missing):")
for item in ecdi_items:
    if item in df.columns:
        print(f"  {item}: {df[item].value_counts(dropna=False).to_dict()}")

# ── 2. Recode items to binary (1 = pass) ──────────────────────────────────────

# Standard items: 1 = Yes = pass
for item in ["EC6","EC7","EC8","EC9","EC10","EC11","EC12","EC13"]:
    col = f"{item}_pass"
    df[col] = np.where(df[item] == 1, 1,
               np.where(df[item] == 2, 0, np.nan))

# Reverse items: 2 = No = pass (not kicking/biting, not distracted)
for item in ["EC14","EC15"]:
    col = f"{item}_pass"
    df[col] = np.where(df[item] == 2, 1,
               np.where(df[item] == 1, 0, np.nan))

# ── 3. Domain scores ──────────────────────────────────────────────────────────

# Literacy-Numeracy: on track if ALL 3 pass
df["ecdi_literacy"] = np.where(
    df[["EC6_pass","EC7_pass","EC8_pass"]].isnull().any(axis=1), np.nan,
    (df[["EC6_pass","EC7_pass","EC8_pass"]].sum(axis=1) == 3).astype(float)
)

# Physical: on track if ALL 2 pass
df["ecdi_physical"] = np.where(
    df[["EC9_pass","EC10_pass"]].isnull().any(axis=1), np.nan,
    (df[["EC9_pass","EC10_pass"]].sum(axis=1) == 2).astype(float)
)

# Learning: on track if ALL 2 pass
df["ecdi_learning"] = np.where(
    df[["EC11_pass","EC12_pass"]].isnull().any(axis=1), np.nan,
    (df[["EC11_pass","EC12_pass"]].sum(axis=1) == 2).astype(float)
)

# Socio-emotional: on track if ALL 3 pass (EC13 + reversed EC14 + reversed EC15)
df["ecdi_socioemotional"] = np.where(
    df[["EC13_pass","EC14_pass","EC15_pass"]].isnull().any(axis=1), np.nan,
    (df[["EC13_pass","EC14_pass","EC15_pass"]].sum(axis=1) == 3).astype(float)
)

# ── 4. Composite ECDI2030 score ───────────────────────────────────────────────

domains = ["ecdi_literacy","ecdi_physical","ecdi_learning","ecdi_socioemotional"]

# Count domains on track — skipna=False ensures NaN propagates correctly
# so domains_on_track is NaN whenever any domain is missing
df["domains_on_track"] = df[domains].sum(axis=1, skipna=False)
df["ecdi_composite"] = np.where(
    df["domains_on_track"].isnull(), np.nan,
    (df["domains_on_track"] >= 3).astype(float)
)

# ── 5. Stimulation index ──────────────────────────────────────────────────────

# Mother stimulation: any of reading, stories, singing, outside, playing, naming
# EC5XY = 'no one' flag — if 1, that activity done by no one
# We want: did mother do each activity (EC5XA=1 means mother did it)
stim_mother_cols = ["EC5AA","EC5BA","EC5CA","EC5DA","EC5EA","EC5FA"]
stim_mother_cols = [c for c in stim_mother_cols if c in df.columns]
df["stim_mother_count"] = df[stim_mother_cols].apply(
    lambda row: (row == 1).sum(), axis=1
)
df["stim_any_adult"] = np.where(
    df[["EC5AY","EC5BY","EC5CY","EC5DY","EC5EY","EC5FY"]].apply(
        lambda row: (row == 1).sum(), axis=1
    ) < 6, 1, 0   # at least one activity done by someone
)

# ── 6. Report ─────────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("ECDI2030 RESULTS — Nepal MICS6")
print("="*55)

# Use CAGE (age in months) throughout
total = df["ecdi_composite"].notna().sum()
on_track = df["ecdi_composite"].sum()
print(f"\nChildren with complete ECDI2030 data: {total:,.0f}")
print(f"Developmentally on track (>=3/4 domains): {on_track:,.0f} ({on_track/total*100:.1f}%)")

print("\nDomain-level on-track rates:")
domain_labels = {
    "ecdi_literacy":       "Literacy-Numeracy",
    "ecdi_physical":       "Physical",
    "ecdi_learning":       "Learning",
    "ecdi_socioemotional": "Socio-emotional",
}
for col, label in domain_labels.items():
    valid = df[col].notna().sum()
    rate = df[col].mean() * 100
    print(f"  {label:<22}: {rate:.1f}%  (n={valid:,})")

print("\nBy sex:")
print(df.groupby("sex")["ecdi_composite"].agg(["mean","count"]).round(3))

print("\nBy area (urban/rural):")
print(df.groupby("area")["ecdi_composite"].agg(["mean","count"]).round(3))

print("\nBy wealth quintile:")
print(df.groupby("windex5")["ecdi_composite"].agg(["mean","count"]).round(3))

print("\nBy province:")
print(df.groupby("province")["ecdi_composite"].agg(["mean","count"]).round(3))

# ── 7. Save ───────────────────────────────────────────────────────────────────
df.to_parquet("/Users/prachi/nepal_ecdi_project/nepal_scored.parquet", index=False)
print("\nSaved to nepal_scored.parquet")
