"""
Nepal MICS6 — Unsupervised ML: Clustering Developmental Profiles
=================================================================
Clusters children by their ECDI domain scores to identify
distinct developmental profiles.

Methods:
  1. K-Means on 4 ECDI domain scores
  2. Hierarchical clustering (Ward linkage) for dendrogram
  3. Profile characterisation: who is in each cluster?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.impute import SimpleImputer

OUT = Path("/Users/prachi/nepal_ecdi_project/figures")
OUT.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.1)

df = pd.read_parquet("/Users/prachi/nepal_ecdi_project/nepal_scored.parquet")
domains = ["ecdi_literacy","ecdi_physical","ecdi_learning","ecdi_socioemotional"]
domain_labels = ["Literacy-\nNumeracy", "Physical", "Learning", "Socio-\nemotional"]

# Use only children with complete domain data
df_c = df[df[domains].notna().all(axis=1)].copy()
print(f"Complete domain data: {len(df_c):,} children")

X = df_c[domains].values

# ── 1. Elbow method + Silhouette scores ───────────────────────────────────────

inertias, sil_scores = [], []
K_range = range(2, 9)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X, labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(list(K_range), inertias, "o-", color="#4C72B0", linewidth=2)
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Inertia")
ax1.set_title("Elbow Method", fontweight="bold")

ax2.plot(list(K_range), sil_scores, "o-", color="#55A868", linewidth=2)
best_k = list(K_range)[np.argmax(sil_scores)]
ax2.axvline(best_k, color="red", linestyle="--", alpha=0.7, label=f"Best k={best_k}")
ax2.set_xlabel("Number of Clusters (k)")
ax2.set_ylabel("Silhouette Score")
ax2.set_title("Silhouette Score", fontweight="bold")
ax2.legend()

plt.suptitle("Optimal Number of Clusters — ECDI Domains", fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUT / "11_cluster_selection.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved 11_cluster_selection.png  (best k={best_k})")

# ── 2. Fit K-Means with optimal k ─────────────────────────────────────────────

K = best_k
km_final = KMeans(n_clusters=K, random_state=42, n_init=20)
df_c["cluster"] = km_final.fit_predict(X)

print(f"\nCluster sizes:")
print(df_c["cluster"].value_counts().sort_index())

# ── 3. Cluster profiles (radar/bar charts) ───────────────────────────────────

cluster_means = df_c.groupby("cluster")[domains].mean() * 100
print("\nCluster mean domain on-track rates (%):")
print(cluster_means.round(1))

# Assign descriptive names based on profiles
def name_cluster(row):
    high = [d for d, v in zip(domain_labels, row) if v >= 70]
    low  = [d for d, v in zip(domain_labels, row) if v < 40]
    if len(high) >= 3:
        return "High overall"
    elif len(low) >= 2:
        return "Low overall"
    elif row[0] < 40:
        return "Literacy-delayed"
    elif row[2] < 40:
        return "Learning-delayed"
    else:
        return f"Profile {int(row.name) + 1}"

cluster_means["label"] = cluster_means.apply(name_cluster, axis=1)
label_map = cluster_means["label"].to_dict()
df_c["cluster_label"] = df_c["cluster"].map(label_map)

# Bar chart: domain rates per cluster
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(domain_labels))
width = 0.8 / K
colors = sns.color_palette("Set2", K)

for i, (cidx, row) in enumerate(cluster_means.iterrows()):
    vals = row[domains].values * 100 if row[domains].values.max() <= 1 else row[domains].values
    n = (df_c["cluster"] == cidx).sum()
    label_name = row["label"]
    ax.bar(x + i*width, vals, width,
           label=f"{label_name} (n={n:,})", color=colors[i], alpha=0.85)

ax.set_xticks(x + width*(K-1)/2)
ax.set_xticklabels(domain_labels)
ax.set_ylabel("% On Track")
ax.set_title(f"ECDI Domain Profiles by Cluster (k={K}) — Nepal MICS6", fontweight="bold")
ax.legend(loc="lower right")
ax.set_ylim(0, 110)
plt.tight_layout()
fig.savefig(OUT / "12_cluster_profiles.png", dpi=150)
plt.close()
print("Saved 12_cluster_profiles.png")

# ── 4. PCA visualisation of clusters ─────────────────────────────────────────

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
df_c["pca1"] = X_pca[:, 0]
df_c["pca2"] = X_pca[:, 1]

fig, ax = plt.subplots(figsize=(9, 7))
for i, (cidx, row) in enumerate(cluster_means.iterrows()):
    mask = df_c["cluster"] == cidx
    ax.scatter(df_c.loc[mask,"pca1"], df_c.loc[mask,"pca2"],
               c=[colors[i]], alpha=0.4, s=15, label=row["label"])

# Plot cluster centroids
centroids_pca = pca.transform(km_final.cluster_centers_)
ax.scatter(centroids_pca[:,0], centroids_pca[:,1], c="black",
           marker="X", s=200, zorder=5, label="Centroids")

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.set_title("PCA — ECDI Developmental Profiles", fontweight="bold")
ax.legend(markerscale=2)
plt.tight_layout()
fig.savefig(OUT / "13_cluster_pca.png", dpi=150)
plt.close()
print("Saved 13_cluster_pca.png")

# ── 5. Cluster characteristics ────────────────────────────────────────────────

char_cols = ["windex5","is_urban" if "is_urban" in df_c.columns else "HH6",
             "CAGE","HAZ2","WAZ2","stim_mother_count"]
char_cols = [c for c in char_cols if c in df_c.columns]

print("\nCluster characteristics (means):")
print(df_c.groupby("cluster_label")[char_cols].mean().round(2))

print("\nCluster × Province:")
print(pd.crosstab(df_c["cluster_label"], df_c["province"],
                  normalize="index").round(2))

# ── 6. Save scored data with cluster labels ───────────────────────────────────

df_c.to_parquet("/Users/prachi/nepal_ecdi_project/nepal_clustered.parquet", index=False)
print("\nSaved nepal_clustered.parquet")
