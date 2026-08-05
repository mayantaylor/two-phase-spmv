import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Read data
# -------------------------
wse = pd.read_csv("matrix-space-wse.csv")
gpu = pd.read_csv("matrix-space-gpu.csv")

# -------------------------
# Parse WSE metadata
# -------------------------
wse["matrix_nnz_per_row"] = (
    wse["name"]
    .str.extract(r"_(\d+)nnzperrow_")
    .astype(int)
)

wse["timems"] = wse["max_time"] / 1000000


# -------------------------
# Parse GPU metadata
# -------------------------
gpu["matrix_nrows"] = gpu["rows"]
gpu["matrix_nnz"] = gpu["nnz"]
gpu["matrix_nnz_per_row"] = gpu["nnz"] / gpu["rows"]

# Keep only needed columns
gpu = gpu[[
    "matrix_nrows",
    "matrix_nnz_per_row",
    "gpu_coo_alg1_mean_ms"
]]

# -------------------------
# Best WSE configuration
# -------------------------
best = wse.loc[
    wse.groupby(["matrix_nrows", "matrix_nnz_per_row"])["timems"].idxmin()
].copy()

# -------------------------
# Merge GPU times
# -------------------------
best = best.merge(
    gpu,
    on=["matrix_nrows", "matrix_nnz_per_row"],
    how="left"
)

# -------------------------
# Compute speedup
# -------------------------
best["speedup"] = (
    best["gpu_coo_alg1_mean_ms"] / best["timems"]
    
)

# -------------------------
# Pivot into heatmap
# -------------------------
heatmap = best.pivot(
    index="matrix_nnz_per_row",
    columns="matrix_nrows",
    values="speedup",
)

heatmap = heatmap.sort_index().sort_index(axis=1)

# -------------------------
# Plot
# -------------------------
fig, ax = plt.subplots(figsize=(10, 6))
norm = plt.Normalize(vmin=0, vmax=50)
cmap = plt.get_cmap("viridis")

im = ax.imshow(
    heatmap.values,
    origin="lower",
    aspect="auto",
    cmap=cmap,
    vmin=0,
    vmax=50,
)

# Axis labels
ax.set_xticks(np.arange(len(heatmap.columns)))
ax.set_xticklabels(heatmap.columns)

ax.set_yticks(np.arange(len(heatmap.index)))
ax.set_yticklabels(heatmap.index)

ax.set_xlabel("Matrix Rows")
ax.set_ylabel("Matrix NNZ")
ax.set_title("Best WSE Speedup over GPU")

# Annotate


for i in range(heatmap.shape[0]):
    for j in range(heatmap.shape[1]):
        val = heatmap.iloc[i, j]
        if pd.notna(val):
            rgba = cmap(norm(val))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(
                j,
                i,
                f"{val:.2f}×",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if luminance < 0.5 else "black",
            )
cbar = fig.colorbar(im)
cbar.set_label("Speedup over GPU")

plt.tight_layout()
plt.savefig("../figures/matrix-space-speedup.pdf", format='pdf')
