import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# Bytes moved per nonzero -- EDIT THIS to match your data format.
# Default assumes: 8-byte double value + 4-byte int32 column index.
# ---------------------------------------------------------
BYTES_PER_NNZ = 12

# Load CSV
df = pd.read_csv("matrix-space-wse.csv")

# Use the nrows column as the number of PEs
df["num_pes"] = df["nrows"]

# Parse nnz per row from the matrix name
df["matrix_nnz"] = (
    df["name"]
    .str.extract(r"_(\d+)nnzperrow_")
    .astype(int)
)

# Total nonzeros and total bytes moved
df["total_nnz"] = df["matrix_nrows"] * df["matrix_nnz"]
df["total_bytes"] = df["total_nnz"] * BYTES_PER_NNZ

# Throughput in GB/s
df["throughput_GBs"] = (df["total_bytes"] / df["max_time"])

# For each matrix (rows, nnz), take the best (max) throughput across all PE counts tested
best = (
    df.groupby(["matrix_nrows", "matrix_nnz"])["throughput_GBs"]
    .max()
    .reset_index()
)

# Nice ordering
row_sizes = sorted(best["matrix_nrows"].unique())
nnz_per_rows = sorted(best["matrix_nnz"].unique())

# ---------------------------------------------------------
# Row Scaling: x = matrix_nrows, one line per nnz/row
# ---------------------------------------------------------
plt.figure(figsize=(6, 4))

for nnzpr in nnz_per_rows:
    subset = (
        best[best["matrix_nnz"] == nnzpr]
        .sort_values("matrix_nrows")
    )

    plt.plot(
        subset["matrix_nrows"],
        subset["throughput_GBs"],
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
plt.savefig("../figures/matrix-space-wse-tp.pdf", format='pdf')
