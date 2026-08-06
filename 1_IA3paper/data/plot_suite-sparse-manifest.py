#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MAPPING_COLORS = {
    "blocked": "tab:blue",
    "cyclic": "tab:orange",
    "random": "tab:green",
}


def normalize_bool(series):
    truthy = {"1", "true", "yes", "y", "on"}
    return series.astype(str).str.strip().str.lower().isin(truthy)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot final manifest as scatter points by matrix and mapping type, "
            "using the minimum MAX_NNZ point for each mapping."
        )
    )
    parser.add_argument("manifest_csv", help="Path to final manifest CSV")

    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=(16, 6),
        metavar=("W", "H"),
        help="Figure size in inches (default: 16 6)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest_csv)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    required = {"matrix_file", "mapping", "MAX_NNZ", "matrix_nrows", "valid"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    df["MAX_NNZ"] = pd.to_numeric(df["MAX_NNZ"], errors="raise")
    df["matrix_nrows"] = pd.to_numeric(df["matrix_nrows"], errors="raise")
    df["_valid_bool"] = normalize_bool(df["valid"])
    
    df = df[df["NROWS"] == 512]

    # Derive a compact display name from matrix_file, e.g. /path/foo.npz -> foo
    df["matrix_name"] = (
        df["matrix_file"]
        .astype(str)
        .str.split("/")
        .str[-1]
        .str.replace(".npz", "", regex=False)
    )
    
    

    # For each matrix + mapping, keep only the row with minimum MAX_NNZ.
    # Tie-break by original row order for deterministic behavior.
    df = df.reset_index(drop=False).rename(columns={"index": "_row_index"})
    best_df = (
        df.sort_values(["matrix_name", "mapping", "MAX_NNZ", "_row_index"])
          .groupby(["matrix_name", "mapping"], as_index=False)
          .first()
    )

    # Matrix ordering on x-axis: sort by matrix_nrows, then name
    matrix_order_df = (
        best_df[["matrix_name", "matrix_nrows"]]
        .drop_duplicates()
        .sort_values(["matrix_nrows", "matrix_name"])
        .reset_index(drop=True)
    )
    matrix_order = matrix_order_df["matrix_name"].tolist()
    x_positions = {name: i for i, name in enumerate(matrix_order)}

    best_df["x"] = best_df["matrix_name"].map(x_positions)
    best_df["y"] = best_df["MAX_NNZ"]

    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    # Plot valid and invalid separately so they are visibly distinct.
    # Valid: filled circle
    # Invalid: x marker
    for mapping, color in MAPPING_COLORS.items():
        sub = best_df[best_df["mapping"] == mapping]
        if sub.empty:
            continue

        sub_valid = sub[sub["_valid_bool"]]
        sub_invalid = sub[~sub["_valid_bool"]]

        if not sub_valid.empty:
            ax.scatter(
                sub_valid["x"],
                sub_valid["y"],
                color=color,
                marker="o",
                s=70,
                edgecolors="black",
                linewidths=0.7,
                label=f"{mapping} valid",
                alpha=0.9,
            )

        if not sub_invalid.empty:
            ax.scatter(
                sub_invalid["x"],
                sub_invalid["y"],
                color=color,
                marker="x",
                s=70,
                linewidths=1.8,
                label=f"{mapping} invalid",
                alpha=0.9,
            )

    # Add a colored outline circle around the minimum valid MAX_NNZ point
    # for each matrix. Outline color matches the point's mapping color.
    for matrix_name, sub in best_df.groupby("matrix_name"):
        sub_valid = sub[sub["_valid_bool"]]
        if sub_valid.empty:
            continue

        min_idx = sub_valid["y"].idxmin()
        highlight = sub_valid.loc[[min_idx]]
        mapping = highlight["mapping"].iloc[0]
        color = MAPPING_COLORS[mapping]

        ax.scatter(
            highlight["x"],
            highlight["y"],
            s=220,
            facecolors="none",
            edgecolors=color,
            linewidths=2.2,
            marker="o",
            zorder=5,
        )

    ax.set_xlabel("Matrix")
    ax.set_ylabel("MAX_NNZ")
    ax.set_yscale("log")
    ax.set_title("Minimum MAX_NNZ per Matrix and Mapping Type")

    ax.set_xticks(range(len(matrix_order)))
    ax.set_xticklabels(matrix_order, rotation=90)

    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(ncol=3, fontsize=9, loc='upper left')
    plt.tight_layout()

    
    output_path = "../figures/suite_sparse_manifest.pdf"
    fig.savefig(output_path, dpi=200, bbox_inches="tight", format='pdf')
    print(f"Wrote plot to {output_path}")
    


if __name__ == "__main__":
    main()
