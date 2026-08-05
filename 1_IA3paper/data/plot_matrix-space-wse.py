import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

# Read results
df = pd.read_csv("matrix-space-wse.csv")

# Use the nrows column as the number of PEs
df["num_pes"] = df["nrows"]

df["matrix_nnz"] = (
    df["name"]
    .str.extract(r"_(\d+)nnzperrow_")
    .astype(int)
)

# For each matrix, find the PE count giving the minimum runtime
best = df.loc[
    df.groupby(["matrix_nrows", "matrix_nnz"])["max_time"].idxmin()
].copy()

all_pes = [64, 128, 256, 384, 512]
tested = (
    df.groupby(["matrix_nrows", "matrix_nnz"])["num_pes"]
      .apply(lambda x: sorted(x.unique()))
      .to_dict()
)

all_pes = sorted(df["num_pes"].unique())

skipped = {}
for key, vals in tested.items():
    skipped[key] = sorted(set(all_pes) - set(vals))

# Pivot into heatmap form
heatmap = best.pivot(
    index="matrix_nnz",
    columns="matrix_nrows",
    values="num_pes",
)



heatmap = heatmap.sort_index().sort_index(axis=1)

# Unique PE counts
pes = np.sort(best["num_pes"].unique())

# Discrete colormap
cmap = plt.get_cmap("viridis", len(pes))
norm = colors.BoundaryNorm(
    np.arange(len(pes) + 1) - 0.5,
    cmap.N
)

# Convert PE values to integer indices
value_to_index = {v: i for i, v in enumerate(pes)}
indexed = heatmap.replace(value_to_index)

fig, ax = plt.subplots(figsize=(10, 6))

im = ax.imshow(
    indexed.values,
    origin="lower",
    aspect="auto",
    cmap=cmap,
    norm=norm,
)

# Axis labels
ax.set_xticks(np.arange(len(heatmap.columns)))
ax.set_xticklabels(heatmap.columns)

ax.set_yticks(np.arange(len(heatmap.index)))
ax.set_yticklabels(heatmap.index)

ax.set_xlabel("Matrix Rows")
ax.set_ylabel("NNZ per Row")
ax.set_title("Optimal PE Count (minimum max_time)")

for i, nnz in enumerate(heatmap.index):
    for j, rows in enumerate(heatmap.columns):
        val = heatmap.loc[nnz, rows]

        if pd.notna(val):
            missing = skipped[(rows, nnz)]

            if len(missing) == 0:
                label = f"{int(val)}"
            else:
                missing_str = ",".join(map(str, missing))
                label = f"{int(val)}\n(-{missing_str})"

            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=7,
                color="white",
            )
# Discrete colorbar
cbar = fig.colorbar(im, ticks=np.arange(len(pes)))
cbar.ax.set_yticklabels([str(int(x)) for x in pes])
cbar.set_label("Optimal # PEs")

plt.tight_layout()
plt.savefig("../figures/matrix-space-wse.pdf", format='pdf')
