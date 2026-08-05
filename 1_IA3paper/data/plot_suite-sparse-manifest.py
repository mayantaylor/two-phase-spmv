import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import numpy as np
from pathlib import Path
try:
    from scipy import sparse as sp_sparse
except Exception:
    sp_sparse = None

# Input files
manifest_csv = "suite-sparse-manifest073026.csv"
results_csv = "suite-sparse-results080326-coo.csv"

# Read CSVs
manifest_df = pd.read_csv(manifest_csv)
results_df = pd.read_csv(results_csv)

# Required columns
manifest_required = {"matrix_file", "mapping", "enabled"}
manifest_missing = manifest_required - set(manifest_df.columns)

if manifest_missing:
    raise ValueError(f"Missing required columns in manifest CSV: {manifest_missing}")

# Copy NNZ_CAPACITY by row order
manifest_df["NNZ_CAPACITY"] = pd.to_numeric(manifest_df["NNZ_CAPACITY"], errors="coerce")

# Normalize columns
manifest_df["matrix_file"] = manifest_df["matrix_file"].astype(str).apply(os.path.basename)
manifest_df["mapping"] = manifest_df["mapping"].astype(str).str.strip()
manifest_df["enabled"] = manifest_df["enabled"].astype(str).str.strip()

# Normalize results CSV matrix_file for matching
results_df["matrix_file"] = results_df["name"].astype(str).str.split('.').str[0] + ".npz"
results_df["mapping"] = results_df["name"].astype(str).str.split('.').str[-1]


# Abbreviate labels
def abbreviate_matrix_file(name, max_len=20):
    name = str(name)
    if len(name) <= max_len:
        return name
    return name[:8] + "..." + name[-8:]

# Build x-axis from all matrices
all_matrix_files = list(manifest_df["matrix_file"].drop_duplicates())

# Determine row counts for each matrix. Prefer columns in the manifest if present,
# otherwise try to load the .npz from ../scripts/matrices/ and read its shape.
def _get_row_from_results(name):
    # name of column in results_df is 'matrix_nrows'
    matrix_name = name.replace(".npz", "")
    val = results_df.loc[results_df["name"].astype(str).str.split('.').str[0] == matrix_name, 'matrix_nrows']
    npes = results_df.loc[results_df["name"].astype(str).str.split('.').str[0] == matrix_name, 'nrows']
    if len(val) > 0:
        return val.values[0] / npes.values[0] if len(npes) > 0 else None
    return None

nrows_map = {}
for name in all_matrix_files:
    n = _get_row_from_results(name)
    nrows_map[name] = n

# Sort matrices by row count (missing counts go to the end)
all_matrix_files = sorted(all_matrix_files, key=lambda n: (nrows_map.get(n) is None, nrows_map.get(n, 0)))

# Labels include abbreviated name + row count when available
x_map = {name: i for i, name in enumerate(all_matrix_files)}

# Keep rows with valid capacity
df = manifest_df.dropna(subset=["NNZ_CAPACITY"]).copy()
x_labels = [name.replace('.npz', '') for name in all_matrix_files]

df["x"] = df["matrix_file"].map(x_map)

# Identify which datapoints have results
has_results = df.set_index(["matrix_file", "mapping"]).index.isin(
    results_df.set_index(["matrix_file", "mapping"]).index
)

df["has_results"] = has_results

# Colors by mapping
unique_mappings = list(df["mapping"].drop_duplicates())
cmap = plt.get_cmap("tab10")
mapping_color_map = {
    mapping: cmap(i % 10) for i, mapping in enumerate(unique_mappings)
}
df["color"] = df["mapping"].map(mapping_color_map)


# Add vertical divider between atmosmodl and delaunay_n23
split_matrix = "atmosmodl.npz"

if split_matrix in x_map:
    split_x = x_map[split_matrix] + 0.5
    plt.axvline(
        x=split_x,
        color="black",
        linewidth=1.5,
        linestyle="-",
        zorder=1
    )
    
# Plot
plt.figure(figsize=(10, 6))


# Base layer: all points
plt.scatter(
    df[~df["has_results"]]["x"],
    df[~df["has_results"]]["NNZ_CAPACITY"],
    c=df[~df["has_results"]]["color"],
    s=80,
    zorder=2,
    label="No results"
)

# Highlight layer: points with results
plt.scatter(
    df[df["has_results"]]["x"],
    df[df["has_results"]]["NNZ_CAPACITY"],
    c=df[df["has_results"]]["color"],
    s=150,
    marker="*",
    edgecolors="red",
    linewidths=2,
    zorder=3,
    label="Has results"
)

# Threshold lines
plt.axhline(y=7800, color="black", linestyle="--", linewidth=1, zorder=1)
plt.axvline(x=split_x, color="black", linewidth=1.5, zorder=1)


plt.xticks(range(len(all_matrix_files)), x_labels, rotation=45, ha="right", fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel("max nnz count", fontsize=20)
plt.yscale('log')
plt.grid(True, linestyle="--", alpha=0.3)
plt.xlim(-0.5, len(all_matrix_files) - 0.5)



# Legend
mapping_legend = [
    Line2D([0], [0], marker='o', color='w', label=mapping,
           markerfacecolor=mapping_color_map[mapping], markersize=10)
    for mapping in unique_mappings
]
mapping_legend.append(Line2D([0], [0], marker='*', color='w', label="Evaluated",
                             markerfacecolor='gray', markersize=15, markeredgecolor='red', markeredgewidth=2))


plt.legend(handles=mapping_legend, title="Mapping", fontsize=16)
plt.tight_layout()



plt.savefig("../figures/suite_sparse_manifest.pdf", format="pdf", dpi=300)
