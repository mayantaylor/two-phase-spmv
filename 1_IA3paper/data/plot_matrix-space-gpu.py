import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from style import figure, style_axes, save

# Load CSV
df = pd.read_csv("matrix-space-gpu.csv")

# Remove empty rows (if present)
df = df.dropna(subset=["rows", "nnz"])

# Compute NNZ per row
df["nnz_per_row"] = (df["nnz"] / df["rows"]).astype(int)

# Nice ordering
row_sizes = sorted(df["rows"].unique())
nnz_per_rows = sorted(df["nnz_per_row"].unique())

# ---------------------------------------------------------
# Row Scaling
# ---------------------------------------------------------
fig, ax = figure("rect")

for nnzpr in nnz_per_rows:
    subset = (
        df[df["nnz_per_row"] == nnzpr]
        .sort_values("rows")
    )

    plt.plot(
        subset["rows"],
        subset["gpu_coo_alg1_bw_GBs"],
        marker="o",
        linewidth=2,
        label=f"{nnzpr} nnz/row",
    )

plt.xscale("log", base=2)
plt.xticks(row_sizes)

plt.xlabel("N")
plt.ylabel("Bandwidth (GB/s)")
plt.ylim(top=4000)
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right")
style_axes(ax)
save(fig,"../figures/matrix-space-gpu.pdf")

