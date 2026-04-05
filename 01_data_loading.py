"""
Nepal MICS6 — Data Loading and Merging
=======================================
Loads ch.sav, hh.sav, wm.sav and merges into a single analysis dataframe.
"""

import pandas as pd
import pyreadstat
import numpy as np
from pathlib import Path

DATA_DIR = Path("/Users/prachi/Downloads/MICS_Datasets/Nepal MICS6 Datasets/Nepal MICS6 SPSS Datasets")

# ── 1. Load raw files ─────────────────────────────────────────────────────────

def load_sav(filename):
    path = DATA_DIR / filename
    df, meta = pyreadstat.read_sav(str(path), apply_value_formats=False)
    print(f"  {filename}: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df, meta

print("Loading MICS6 modules...")
ch, meta_ch = load_sav("ch.sav")   # Children under 5
hh, meta_hh = load_sav("hh.sav")   # Households
wm, meta_wm = load_sav("wm.sav")   # Women 15-49

# ── 2. Select relevant columns ────────────────────────────────────────────────

# Child module: identifiers + ECDI items + key child-level vars
ch_cols = (
    ["HH1", "HH2", "LN", "UF4",          # IDs
     "UB2",                                # age in years
     "CAGE",                               # age in months (computed)
     "HL4",                                # sex
     "HH6", "HH7", "HH7b", "HH7c",       # urban/rural, region, domain, province
     "chweight",                           # sampling weight
     # ECDI2030 items
     "EC6", "EC7", "EC8",                  # literacy-numeracy
     "EC9", "EC10",                        # physical
     "EC11", "EC12",                       # learning
     "EC13", "EC14", "EC15",              # socio-emotional
     # Caregiver stimulation (EC5 series)
     "EC5AA","EC5AB","EC5AX","EC5AY",     # read books: mother/father/other/no one
     "EC5BA","EC5BB","EC5BX","EC5BY",     # tell stories
     "EC5CA","EC5CB","EC5CX","EC5CY",     # sing songs
     "EC5DA","EC5DB","EC5DX","EC5DY",     # took outside
     "EC5EA","EC5EB","EC5EX","EC5EY",     # played with
     "EC5FA","EC5FB","EC5FX","EC5FY",     # named/counted
     # Learning materials
     "EC1", "EC2A", "EC2B", "EC2C",
     # Early childhood education
     "UB6", "UB7", "UB8",
     # Anthropometrics
     "HAZ2", "WAZ2", "WHZ2",
     # Discipline
     "UCD5",
     # Mother education (precomputed)
     "melevel1",
     # Wealth
     "windex5", "wscore",
    ]
)

# Keep only cols that exist in the file
ch_cols = [c for c in ch_cols if c in ch.columns]
ch_sub = ch[ch_cols].copy()

# Household module: key socioeconomic vars
hh_cols = [
    "HH1", "HH2",
    "HC1A","HC1B","HC2","HC3",    # water source
    "HC4","HC5","HC6","HC7A","HC7B",  # sanitation, floor, walls, roof
    "HC8","HC9A","HC9B","HC9C","HC9D","HC9E",  # assets
    "HH6", "HH7",
    "wscore", "windex5",
]
hh_cols = [c for c in hh_cols if c in hh.columns]
hh_sub = hh[hh_cols].copy()
# Rename to avoid collision with ch columns
hh_sub = hh_sub.rename(columns={c: f"hh_{c}" for c in hh_sub.columns
                                  if c not in ["HH1","HH2"]})

# Women module: mother characteristics
wm_cols = [
    "HH1", "HH2", "LN",
    "WM17",   # mother's age
    "melevel1" if "melevel1" in wm.columns else None,
]
wm_cols = [c for c in wm_cols if c is not None and c in wm.columns]
wm_sub = wm[wm_cols].copy()
wm_sub = wm_sub.rename(columns={"LN": "WMLNA"})
# Rename wm columns to avoid collision
rename_wm = {c: f"wm_{c}" for c in wm_sub.columns
             if c not in ["HH1","HH2","WMLNA"]}
wm_sub = wm_sub.rename(columns=rename_wm)

# ── 3. Merge ──────────────────────────────────────────────────────────────────

# Child has UF4 = mother's line number → matches wm LN
ch_sub = ch_sub.rename(columns={"UF4": "WMLNA"})

df = ch_sub.merge(hh_sub, on=["HH1","HH2"], how="left")
df = df.merge(wm_sub, on=["HH1","HH2","WMLNA"], how="left")

print(f"\nMerged dataset: {df.shape[0]:,} children × {df.shape[1]} variables")

# ── 4. Basic cleaning ─────────────────────────────────────────────────────────

# Keep only children aged 24-59 months (ECDI2030 target age range)
# CAGE is age in months; UB2 is age in years
df = df[df["CAGE"].between(24, 59)].copy()
print(f"After age filter (24-59 months): {df.shape[0]:,} children")

# Recode sex: 1=Male, 2=Female
df["sex"] = df["HL4"].map({1: "Male", 2: "Female"})

# Recode urban/rural
df["area"] = df["HH6"].map({1: "Urban", 2: "Rural"})

# Province name map (Nepal MICS6 uses HH7c)
province_map = {
    1: "Koshi", 2: "Madhesh", 3: "Bagmati",
    4: "Gandaki", 5: "Lumbini", 6: "Karnali", 7: "Sudurpashchim"
}
df["province"] = df["HH7c"].map(province_map)

print(f"\nFinal analysis dataset: {df.shape[0]:,} rows × {df.shape[1]} cols")
print(f"Provinces: {df['province'].value_counts().to_dict()}")
print(f"Sex: {df['sex'].value_counts().to_dict()}")
print(f"Area: {df['area'].value_counts().to_dict()}")

# ── 5. Save ───────────────────────────────────────────────────────────────────
df.to_parquet("/Users/prachi/nepal_ecdi_project/nepal_merged.parquet", index=False)
print("\nSaved to nepal_merged.parquet")
