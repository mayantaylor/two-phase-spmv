import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
plt.figure(figsize=(6,4))

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
plt.xticks(row_sizes, [f"{int(r):,}" for r in row_sizes], rotation=45)

plt.xlabel("Matrix Rows")
plt.ylabel("Bandwidth (GB/s)")
plt.title("Row Scaling")
plt.grid(True, alpha=0.3)
plt.legend(title="Fixed NNZ/Row")
plt.tight_layout()
plt.savefig("../figures/matrix-space-gpu.pdf", format='pdf')


# ---------------------------------------------------------
# NNZ Scaling
# ---------------------------------------------------------
plt.figure(figsize=(6,4))

for rows in row_sizes:
    subset = (
        df[df["rows"] == rows]
        .sort_values("nnz_per_row")
    )

    plt.plot(
        subset["nnz_per_row"],
        subset["gpu_coo_alg1_bw_GBs"],
        marker="o",
        linewidth=2,
        label=f"{int(rows):,} rows",
    )

plt.xscale("log", base=2)
plt.xticks(nnz_per_rows, [str(x) for x in nnz_per_rows])

plt.xlabel("NNZ per Row")
plt.ylabel("Bandwidth (GB/s)")
plt.title("NNZ Scaling")
plt.grid(True, alpha=0.3)
plt.legend(title="Fixed Rows")
plt.tight_layout()
plt.savefig("../figures/matrix-space-gpu-scalennz.pdf")
