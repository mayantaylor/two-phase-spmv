import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Input files
manifest_csv = "suite-sparse-manifest073026.csv"

# Read CSVs
manifest_df = pd.read_csv(manifest_csv)

# Required columns
manifest_required = {"matrix_file", "mapping", "enabled"}
capacity_required = {"NNZ_CAPACITY"}

manifest_missing = manifest_required - set(manifest_df.columns)

if manifest_missing:
    raise ValueError(f"Missing required columns in manifest CSV: {manifest_missing}")


# Copy NNZ_CAPACITY by row order
manifest_df["NNZ_CAPACITY"] = pd.to_numeric(manifest_df["NNZ_CAPACITY"], errors="coerce")

import os
# Normalize columns
manifest_df["matrix_file"] = manifest_df["matrix_file"].astype(str).apply(os.path.basename)
manifest_df["mapping"] = manifest_df["mapping"].astype(str).str.strip()
manifest_df["enabled"] = manifest_df["enabled"].astype(str).str.strip()

# Abbreviate labels
def abbreviate_matrix_file(name, max_len=20):
    name = str(name)
    if len(name) <= max_len:
        return name
    return name[:8] + "..." + name[-8:]

# Build x-axis from all matrices
all_matrix_files = list(manifest_df["matrix_file"].drop_duplicates())
x_map = {name: i for i, name in enumerate(all_matrix_files)}
x_labels = [abbreviate_matrix_file(name) for name in all_matrix_files]

# Keep rows with valid capacity
df = manifest_df.dropna(subset=["NNZ_CAPACITY"]).copy()
df["x"] = df["matrix_file"].map(x_map)

# Colors by mapping
unique_mappings = list(df["mapping"].drop_duplicates())
cmap = plt.get_cmap("tab10")
mapping_color_map = {
    mapping: cmap(i % 10) for i, mapping in enumerate(unique_mappings)
}
df["color"] = df["mapping"].map(mapping_color_map)

# Split enabled rows for overlay
enabled_df = df[df["enabled"] == "1"].copy()

# Plot
plt.figure(figsize=(14, 8))

plt.axhline(y=7800, color="black", linestyle="--", linewidth=1)

# Base layer: all points colored by mapping
plt.scatter(
    df["x"],
    df["NNZ_CAPACITY"],
    c=df["color"],
    s=80,
    zorder=2
)

plt.xticks(range(len(all_matrix_files)), x_labels, rotation=45, ha="right")
plt.xlabel("Matrix File")
plt.ylabel("NNZ_CAPACITY")
plt.yscale('log')
plt.title("Mappings by Matrix File and NNZ_CAPACITY")
plt.grid(True, linestyle="--", alpha=0.3)
plt.xlim(-0.5, len(all_matrix_files) - 0.5)

# Legend
mapping_legend = [
    Line2D([0], [0], marker='o', color='w', label=mapping,
           markerfacecolor=mapping_color_map[mapping], markersize=10)
    for mapping in unique_mappings
]



plt.tight_layout()
plt.savefig("../figures/suite-sparse-manifest073026.pdf", format="pdf", dpi=300)
plt.show()