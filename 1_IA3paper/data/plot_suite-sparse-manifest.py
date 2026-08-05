#!/usr/bin/env python3
"""
Plot the minimum nnz_max_capacity for each (matrix name, mapping type) pair.

Usage:
    python plot_nnz_capacity.py input.csv [output.png]
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt



csv_path = "ss-max-nnz-coo.csv"
out_path = "../figures/suite_sparse_manifest.pdf"

df = pd.read_csv(csv_path)

# Drop fully blank rows and rows missing the value we care about
df = df.dropna(subset=["name", "nnz_max_capacity", "mapping", "matrix_nrows"])
df["nnz_max_capacity"] = pd.to_numeric(df["nnz_max_capacity"], errors="coerce")
df = df.dropna(subset=["nnz_max_capacity"])

# Matrix name = first part of the name string (before the first '.')
df["matrix"] = df["name"].str.split(".").str[0]

 
# Min nnz_max_capacity per (matrix, mapping)
grouped = (
    df.groupby(["matrix", "mapping"])["nnz_max_capacity"]
    .min()
    .reset_index()
)

# Order matrices by first appearance in the original file
matrix_order = df["matrix"].drop_duplicates().tolist()
grouped["matrix"] = pd.Categorical(grouped["matrix"], categories=matrix_order, ordered=True)

# Add matrix_nrows back to grouped
grouped = grouped.merge(
    df[["matrix", "matrix_nrows"]].drop_duplicates(),
    on="matrix"
)
grouped = grouped.sort_values("matrix_nrows")

fig, ax = plt.subplots(figsize=(10, 6))

markers = ["o", "s", "^", "D", "v", "P", "X"]
for i, (mapping, sub) in enumerate(grouped.groupby("mapping")):
    ax.scatter(
        sub["matrix"],
        sub["nnz_max_capacity"],
        label=mapping,
        marker=markers[i % len(markers)],
        s=80,
    )

ax.set_xlabel("Matrix")
ax.set_ylabel("nnz_max_capacity (min)")
ax.set_title("Min nnz_max_capacity per matrix, by mapping type")
ax.set_yscale("log")
plt.xticks(rotation=45, ha="right")
ax.legend(title="mapping")
ax.grid(True, which="both", linestyle="--", alpha=0.4)

fig.tight_layout()
fig.savefig(out_path, dpi=150)
print(f"Saved plot to {out_path}")


