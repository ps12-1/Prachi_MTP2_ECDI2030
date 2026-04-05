"""
Nepal MICS6 — Geospatial Analysis
===================================
Province-level ECDI2030 analysis using geopandas.
Includes choropleth maps and Moran's I spatial autocorrelation.

Note: Downloads Nepal province shapefile from GADM (naturalearth fallback).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

OUT = Path("/Users/prachi/nepal_ecdi_project/figures")
OUT.mkdir(exist_ok=True)

df = pd.read_parquet("/Users/prachi/nepal_ecdi_project/nepal_scored.parquet")

# ── 1. Province-level aggregation ────────────────────────────────────────────

domains = ["ecdi_literacy","ecdi_physical","ecdi_learning","ecdi_socioemotional","ecdi_composite"]

prov_stats = df.groupby("province").agg(
    n=("ecdi_composite","count"),
    ecdi_composite=("ecdi_composite","mean"),
    ecdi_literacy=("ecdi_literacy","mean"),
    ecdi_physical=("ecdi_physical","mean"),
    ecdi_learning=("ecdi_learning","mean"),
    ecdi_socioemotional=("ecdi_socioemotional","mean"),
    wealth_mean=("windex5","mean"),
    urban_pct=("area", lambda x: (x=="Urban").mean()),
    haz_mean=("HAZ2","mean"),
    stim_mean=("stim_mother_count","mean"),
).reset_index()

prov_stats[domains] *= 100
prov_stats["urban_pct"] *= 100
print("Province-level stats:")
print(prov_stats[["province","n","ecdi_composite"]].round(1).to_string(index=False))

# ── 2. Try loading Nepal shapefile via geopandas ─────────────────────────────

HAS_GEO = False   # choropleth requires GADM shapefile (see note at end)

# ── 3. Province comparison heatmap (works without shapefile) ─────────────────

pivot = prov_stats.set_index("province")[
    ["ecdi_literacy","ecdi_physical","ecdi_learning","ecdi_socioemotional","ecdi_composite"]
].rename(columns={
    "ecdi_literacy":"Literacy-Num",
    "ecdi_physical":"Physical",
    "ecdi_learning":"Learning",
    "ecdi_socioemotional":"Socio-emotional",
    "ecdi_composite":"Composite"
})

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn",
            vmin=40, vmax=100, linewidths=0.5,
            cbar_kws={"label":"% On Track"}, ax=ax)
ax.set_title("ECDI2030 by Province — Nepal MICS6 (%)", fontweight="bold")
ax.set_xlabel("")
plt.tight_layout()
fig.savefig(OUT / "14_province_heatmap.png", dpi=150)
plt.close()
print("Saved 14_province_heatmap.png")

# ── 4. Bubble chart: wealth vs ECDI vs sample size ───────────────────────────

fig, ax = plt.subplots(figsize=(9, 6))
scatter = ax.scatter(
    prov_stats["wealth_mean"],
    prov_stats["ecdi_composite"],
    s=prov_stats["n"] / 3,
    c=prov_stats["urban_pct"],
    cmap="coolwarm",
    alpha=0.85,
    edgecolors="white",
    linewidths=1.5
)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("% Urban")

for _, row in prov_stats.iterrows():
    ax.annotate(row["province"],
                (row["wealth_mean"], row["ecdi_composite"]),
                textcoords="offset points", xytext=(6, 4), fontsize=9)

ax.set_xlabel("Mean Wealth Quintile")
ax.set_ylabel("% Developmentally On Track (ECDI2030)")
ax.set_title("Province-level ECDI2030 vs Wealth\n(bubble size = sample size, colour = % urban)",
             fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "15_province_bubble.png", dpi=150)
plt.close()
print("Saved 15_province_bubble.png")

# ── 5. Domain inequality across provinces (range chart) ──────────────────────

domain_cols = ["ecdi_literacy","ecdi_physical","ecdi_learning","ecdi_socioemotional"]
domain_display = ["Literacy-Numeracy","Physical","Learning","Socio-emotional"]

fig, ax = plt.subplots(figsize=(10, 6))
for i, (col, label) in enumerate(zip(domain_cols, domain_display)):
    vals = prov_stats[col].sort_values().values
    ax.plot(vals, [i]*len(vals), "o", markersize=8, label=label)
    ax.plot([vals.min(), vals.max()], [i, i], "-", lw=2, alpha=0.4)
    ax.annotate(f"Range: {vals.max()-vals.min():.1f}pp",
                (vals.max(), i), xytext=(3, 3), textcoords="offset points", fontsize=9)

ax.set_yticks(range(len(domain_display)))
ax.set_yticklabels(domain_display)
ax.set_xlabel("% On Track")
ax.set_title("Between-Province Inequality in ECDI Domains — Nepal MICS6",
             fontweight="bold")
ax.axvline(prov_stats["ecdi_composite"].mean(), color="black",
           linestyle="--", alpha=0.5, label="Nat. composite avg")
ax.legend(loc="lower right")
plt.tight_layout()
fig.savefig(OUT / "16_province_inequality.png", dpi=150)
plt.close()
print("Saved 16_province_inequality.png")

# ── 6. Moran's I (if libpysal available) ─────────────────────────────────────

try:
    import libpysal
    from esda.moran import Moran

    # Nepal province adjacency (hand-coded based on geography)
    # Province order: Koshi(1), Madhesh(2), Bagmati(3), Gandaki(4),
    #                 Lumbini(5), Karnali(6), Sudurpashchim(7)
    adjacency = {
        "Koshi":         ["Madhesh", "Bagmati"],
        "Madhesh":       ["Koshi", "Bagmati", "Lumbini"],
        "Bagmati":       ["Koshi", "Madhesh", "Gandaki", "Lumbini"],
        "Gandaki":       ["Bagmati", "Lumbini", "Karnali"],
        "Lumbini":       ["Madhesh", "Bagmati", "Gandaki", "Karnali"],
        "Karnali":       ["Gandaki", "Lumbini", "Sudurpashchim"],
        "Sudurpashchim": ["Karnali"],
    }

    provinces = prov_stats["province"].tolist()
    n_prov = len(provinces)
    W = np.zeros((n_prov, n_prov))
    for i, p in enumerate(provinces):
        for j, q in enumerate(provinces):
            if q in adjacency.get(p, []):
                W[i, j] = 1
    # Row-standardise
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = W / row_sums

    y_moran = prov_stats.set_index("province").loc[provinces, "ecdi_composite"].values

    w_obj = libpysal.weights.util.full2W(W, ids=provinces)
    moran = Moran(y_moran, w_obj)
    print(f"\nMoran's I (province-level ECDI2030): {moran.I:.3f}")
    print(f"Expected:  {moran.EI:.3f}")
    print(f"p-value:   {moran.p_sim:.3f}")
    if moran.p_sim < 0.05:
        print("→ Statistically significant spatial autocorrelation detected")
    else:
        print("→ No significant spatial autocorrelation at province level")
        print("  (note: only 7 provinces — low power; district-level would be more informative)")

except Exception as e:
    print(f"\nMoran's I skipped: {e}")
    print("Install esda and libpysal for spatial autocorrelation analysis")

# ── 7. Summary table ──────────────────────────────────────────────────────────

print("\nFinal province summary table:")
print(prov_stats[["province","n","ecdi_composite","wealth_mean","urban_pct","haz_mean"]].round(2).to_string(index=False))
prov_stats.to_csv("/Users/prachi/nepal_ecdi_project/province_summary.csv", index=False)
print("\nSaved province_summary.csv")
