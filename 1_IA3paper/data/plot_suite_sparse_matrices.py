#!/usr/bin/env python3
"""
Plot matrices from a CSV file as a scatter plot:
  x = nrows (dimension)
  y = nnz  (number of nonzeros)

Usage:
    python plot_matrices.py [path/to/matrices.csv] [-o output.png]

The CSV is expected to have (at least) these columns:
    name, id, nrows, ncols, nnz, category
Blank/incomplete rows are skipped automatically.
"""
import sys

import pandas as pd
import matplotlib.pyplot as plt


def load_matrices(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Drop fully blank rows and rows missing the columns we need
    df = df.dropna(subset=["nrows", "nnz"])
    df["nrows"] = pd.to_numeric(df["nrows"], errors="coerce")
    df["nnz"] = pd.to_numeric(df["nnz"], errors="coerce")
    df = df.dropna(subset=["nrows", "nnz"])
    return df


def plot_matrices(df: pd.DataFrame, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 10))

    categories = df["category"].unique() if "category" in df.columns else [None]
    colors = plt.cm.tab10.colors

    for i, cat in enumerate(categories):
        subset = df[df["category"] == cat] if cat is not None else df
        ax.scatter(
            subset["nrows"],
            subset["nnz"]/subset["nrows"],
            label=cat if cat is not None else "matrices",
            color=colors[i % len(colors)],
            s=80,
            edgecolor="black",
            alpha=0.8,
        )
        # Label each point with its matrix name
        if "name" in subset.columns:
            for _, row in subset.iterrows():
                ax.annotate(
                    row["name"],
                    (row["nrows"], row["nnz"]/row["nrows"]),
                    textcoords="offset points",
                    xytext=(6, 4),
                    fontsize=16,
                )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dimension (nrows)", fontsize=16)
    ax.set_ylabel("Nonzeros per Row", fontsize=16)
    ax.set_title("Matrix nnz vs. dimension", fontsize=16)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=16)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, format='pdf')


def main():
    csv_path="suite_sparse_matrices.csv"
    output="../figures/suite_sparse_matrices.pdf"

    try:
        df = load_matrices(csv_path)
    except FileNotFoundError:
        print(f"Error: could not find '{csv_path}'", file=sys.stderr)
        sys.exit(1)

    if df.empty:
        print("No valid matrix rows found in the CSV.", file=sys.stderr)
        sys.exit(1)

    plot_matrices(df, output)


if __name__ == "__main__":
    main()
