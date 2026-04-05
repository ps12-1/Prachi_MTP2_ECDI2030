"""
Nepal MICS6 — Exploratory Data Analysis
=========================================
Produces plots and summary statistics for the ECDI2030 analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

OUT = Path("/Users/prachi/nepal_ecdi_project/figures")
OUT.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

df = pd.read_parquet("/Users/prachi/nepal_ecdi_project/nepal_scored.parquet")
print(f"Loaded: {df.shape[0]:,} children")

domains = ["ecdi_literacy","ecdi_physical","ecdi_learning","ecdi_socioemotional"]
domain_labels = ["Literacy-\nNumeracy", "Physical", "Learning", "Socio-\nemotional"]

# ── 1. ECDI domain bar chart ──────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))
rates = [df[d].mean()*100 for d in domains]
bars = ax.bar(domain_labels, rates, color=["#4C72B0","#55A868","#C44E52","#8172B2"],
              edgecolor="white", linewidth=1.5)
ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=12, fontweight="bold")
ax.axhline(df["ecdi_composite"].mean()*100, color="black", linestyle="--",
           linewidth=1.5, label=f"Composite on-track: {df['ecdi_composite'].mean()*100:.1f}%")
ax.set_ylim(0, 110)
ax.set_ylabel("% Children On Track")
ax.set_title("ECDI2030 Domain-Level On-Track Rates — Nepal MICS6", fontweight="bold")
ax.legend()
plt.tight_layout()
fig.savefig(OUT / "01_ecdi_domains.png", dpi=150)
plt.close()
print("Saved 01_ecdi_domains.png")

# ── 2. On-track by wealth quintile ───────────────────────────────────────────

wq = (df.groupby("windex5")[["ecdi_composite"] + domains]
        .mean() * 100).reset_index()
fig, ax = plt.subplots(figsize=(9, 5))
for i, (col, label) in enumerate(zip(["ecdi_composite"] + domains,
                                      ["Composite"] + domain_labels)):
    ax.plot(wq["windex5"], wq[col], marker="o", label=label.replace("\n"," "),
            linewidth=2)
ax.set_xlabel("Wealth Quintile (1=Poorest, 5=Richest)")
ax.set_ylabel("% On Track")
ax.set_title("ECDI2030 by Wealth Quintile — Nepal MICS6", fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
fig.savefig(OUT / "02_ecdi_wealth.png", dpi=150)
plt.close()
print("Saved 02_ecdi_wealth.png")

# ── 3. On-track by age group ──────────────────────────────────────────────────

df["age_group"] = pd.cut(df["CAGE"], bins=[23,29,35,41,47,53,59],
                          labels=["24-29","30-35","36-41","42-47","48-53","54-59"])
age_data = (df.groupby("age_group", observed=True)[["ecdi_composite"] + domains]
              .mean() * 100).reset_index()

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(age_data))
width = 0.18
colors = ["#2d6a4f","#4C72B0","#55A868","#C44E52","#8172B2"]
all_cols = ["ecdi_composite"] + domains
all_labels = ["Composite"] + domain_labels
for i, (col, label) in enumerate(zip(all_cols, all_labels)):
    ax.bar(x + i*width, age_data[col], width, label=label.replace("\n"," "),
           color=colors[i], alpha=0.85)
ax.set_xticks(x + width*2)
ax.set_xticklabels(age_data["age_group"])
ax.set_xlabel("Age Group (months)")
ax.set_ylabel("% On Track")
ax.set_title("ECDI2030 by Age Group — Nepal MICS6", fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(OUT / "03_ecdi_age.png", dpi=150)
plt.close()
print("Saved 03_ecdi_age.png")

# ── 4. Province-level composite rates ────────────────────────────────────────

prov = (df.groupby("province")["ecdi_composite"].mean() * 100).sort_values()
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(prov.index, prov.values,
               color=plt.cm.RdYlGn(prov.values / 100), edgecolor="white")
ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=11)
ax.set_xlabel("% Children Developmentally On Track")
ax.set_title("ECDI2030 Composite by Province — Nepal MICS6", fontweight="bold")
ax.axvline(df["ecdi_composite"].mean()*100, color="navy", linestyle="--",
           linewidth=1.5, label="National average")
ax.legend()
plt.tight_layout()
fig.savefig(OUT / "04_ecdi_province.png", dpi=150)
plt.close()
print("Saved 04_ecdi_province.png")

# ── 5. Urban vs Rural × Sex ───────────────────────────────────────────────────

grp = (df.groupby(["area","sex"])["ecdi_composite"].mean() * 100).reset_index()
fig, ax = plt.subplots(figsize=(7, 5))
sns.barplot(data=grp, x="area", y="ecdi_composite", hue="sex",
            palette=["#4C72B0","#C44E52"], ax=ax)
ax.set_ylabel("% On Track")
ax.set_xlabel("")
ax.set_title("ECDI2030 by Area and Sex — Nepal MICS6", fontweight="bold")
ax.set_ylim(0, 100)
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=11)
plt.tight_layout()
fig.savefig(OUT / "05_ecdi_area_sex.png", dpi=150)
plt.close()
print("Saved 05_ecdi_area_sex.png")

# ── 6. Caregiver stimulation vs ECDI ─────────────────────────────────────────

stim_groups = df.groupby("stim_mother_count")["ecdi_composite"].mean() * 100
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(stim_groups.index, stim_groups.values,
       color=sns.color_palette("Blues_d", len(stim_groups)))
ax.set_xlabel("Number of Stimulation Activities by Mother (out of 6)")
ax.set_ylabel("% Children On Track")
ax.set_title("Caregiver Stimulation vs ECDI2030 — Nepal MICS6", fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "06_stimulation.png", dpi=150)
plt.close()
print("Saved 06_stimulation.png")

# ── 7. Correlation heatmap ────────────────────────────────────────────────────

num_cols = domains + ["UB2","windex5","HAZ2","WAZ2","stim_mother_count"]
num_cols = [c for c in num_cols if c in df.columns]
corr = df[num_cols].corr()
labels = {
    "ecdi_literacy":"Literacy-Num", "ecdi_physical":"Physical",
    "ecdi_learning":"Learning", "ecdi_socioemotional":"Socio-emotional",
    "UB2":"Age (months)", "windex5":"Wealth quintile",
    "HAZ2":"Height-for-age z", "WAZ2":"Weight-for-age z",
    "stim_mother_count":"Mother stimulation"
}
corr.index = [labels.get(c, c) for c in corr.index]
corr.columns = [labels.get(c, c) for c in corr.columns]

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, ax=ax, linewidths=0.5)
ax.set_title("Correlation Matrix — ECDI Domains and Key Predictors", fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "07_correlation.png", dpi=150)
plt.close()
print("Saved 07_correlation.png")

# ── 8. Missing data summary ───────────────────────────────────────────────────

ecdi_items = [f"EC{i}" for i in range(6, 16)]
ecdi_items = [c for c in ecdi_items if c in df.columns]
missing = df[ecdi_items].isnull().mean() * 100
print("\nMissing data in ECDI items (%):")
print(missing.round(2).to_string())

print(f"\nComplete ECDI2030 composite (no missing): {df['ecdi_composite'].notna().sum():,} / {len(df):,}")
print("\nAll EDA figures saved to:", OUT)
