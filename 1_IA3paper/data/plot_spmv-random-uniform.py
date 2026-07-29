import argparse
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    csv_file = "spmv-random-uniform.csv"
    matrices = ['rand.001.65536', 'rand.65536', 'rand.0001.65536']
    output = "../figures/spmv-random-uniform.pdf"

    df = pd.read_csv(csv_file, skipinitialspace=True)

    # Fix malformed values if present
    if "max_time" in df.columns:
        df["max_time"] = (
            df["max_time"]
            .astype(str)
            .str.replace("61692.0.0", "61692.0", regex=False)
        )

    numeric_cols = [
        "nrows", "ncols", "NROWS", "XPERCOL", "NCOLS", "ROWSPERPE",
        "NNZ_CAPACITY", "SEED", "DENSITY", "max_time", "min_time",
        "max_chain_time", "min_chain_time"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["matrix"] = df["name"]

    # Choose x-axis quantity here
    df["num_pes"] = df["NROWS"]

    plt.figure(figsize=(8, 6))



    colormap = plt.colormaps['tab20']  # Good choice for discrete categories
    unique_colors = colormap(np.linspace(0, 1, len(matrices))) 


    for idx, matrix in enumerate(matrices):
        sub = df[df["matrix"] == matrix].copy()
        if sub.empty:
            continue

        c = unique_colors[idx]

        # Group repeated x-values
        grouped = (
            sub.groupby("num_pes")
            .agg(
                max_time_mean=("max_time", "mean"),
                max_time_std=("max_time", "std"),
                max_chain_time_mean=("max_chain_time", "mean"),
                max_chain_time_std=("max_chain_time", "std"),
                ROWSPERPE_mean=("ROWSPERPE", "first"),
                n=("num_pes", "size"),
            )
            .reset_index()
            .sort_values("num_pes")
        )

        # Replace NaN std (happens when only one point exists) with 0
        grouped["max_time_std"] = grouped["max_time_std"].fillna(0)
        grouped["max_chain_time_std"] = grouped["max_chain_time_std"].fillna(0)

        # If you want standard error instead of standard deviation, use:
        # grouped["max_time_err"] = grouped["max_time_std"] / np.sqrt(grouped["n"])
        # grouped["max_chain_time_err"] = grouped["max_chain_time_std"] / np.sqrt(grouped["n"])

        # Here using standard deviation as error bars
        grouped["max_time_err"] = grouped["max_time_std"]
        grouped["max_chain_time_err"] = grouped["max_chain_time_std"]

        plt.errorbar(
            grouped["num_pes"],
            grouped["max_time_mean"],
            yerr=grouped["max_time_err"],
            marker="o",
            linestyle="-",
            capsize=4,
            label=f"{matrix} total",
            color=c
        )

        plt.errorbar(
            grouped["num_pes"],
            grouped["max_chain_time_mean"],
            yerr=grouped["max_chain_time_err"],
            marker="o",
            linestyle=":",
            capsize=4,
            label=f"{matrix} max compute",
            color=c
        )

        formula_y = 4 * grouped["ROWSPERPE_mean"] + 14.6 * grouped["num_pes"]
        plt.plot(
            grouped["num_pes"],
            formula_y,
            marker="x",
            linestyle="--",
            label=f"{matrix} communication",
            color=c
        )

    plt.xlabel("Number of PEs")
    plt.ylabel("Time")
    plt.title("Max Total Time and Max Compute Time vs Number of PEs")
    plt.xticks(sorted(df["num_pes"].dropna().unique()))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=200, bbox_inches="tight", format='pdf')
    else:
        plt.show()


if __name__ == "__main__":
    main()