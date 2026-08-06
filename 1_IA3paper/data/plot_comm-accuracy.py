import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from style import figure, style_axes, save

# Read results
df = pd.read_csv("matrix-space-wse.csv")

df["num_pes"] = df["nrows"]

df["matrix_nnz"] = (
    df["name"]
    .str.extract(r"_(\d+)nnzperrow_")
    .astype(int)
)

df["B"] = df["matrix_nrows"] / df["nrows"]

df["comm_model"] = (
    df["B"]
    + 12 * (df["nrows"] + 1)
    + 9 / 64 * df["nrows"]
    + df["B"]
    + 2 * df["nrows"]
    + 10
    + 9 / 128 * df["nrows"]
)

# --- Pick 3 representative cases ---
# Option A: auto-pick low/med/high nnz at a fixed matrix size
TARGET_NROWS = df["matrix_nrows"].median()  # or hardcode e.g. 4096
closest_nrows = df["matrix_nrows"].iloc[
    (df["matrix_nrows"] - TARGET_NROWS).abs().argsort()[:1]
].values[0]

candidates = df[df["matrix_nrows"] == closest_nrows]
nnz_options = sorted(candidates["matrix_nnz"].unique())

low, mid, high = nnz_options[0], nnz_options[len(nnz_options) // 2], nnz_options[-1]
cases = [(closest_nrows, low), (closest_nrows, mid), (closest_nrows, high)]

# Option B: just hardcode the exact cases you want, e.g.:
# cases = [(4096, 8), (4096, 32), (4096, 128)]

# --- Plot ---
fig, ax = figure("rect")

colors_list = plt.cm.viridis(np.linspace(0.15, 0.85, len(cases)))

for (nrows, nnz), color in zip(cases, colors_list):
    sub = df[
        (df["matrix_nrows"] == nrows) & (df["matrix_nnz"] == nnz)
    ].sort_values("num_pes")

    if sub.empty:
        continue

    label_base = f"nnz/row={nnz}"

    ax.plot(
        sub["num_pes"], sub["max_time"],
        marker="o", color=color, linestyle="-",
        label=f"{label_base}",
    )
        
ax.plot(
    sub["num_pes"], sub["comm_model"],
    marker="x", color='k', linestyle="--",
    label=f"Communication Model",
)

ax.set_xlabel("PE Grid Size")
ax.set_ylabel("Time (ms)")
ax.legend()
style_axes(ax)
save(fig,"../figures/comm-accuracy.pdf" )
