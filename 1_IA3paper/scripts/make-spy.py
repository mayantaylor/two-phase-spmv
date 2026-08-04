#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from scipy import sparse


def main():
    parser = argparse.ArgumentParser(
        description="Generate a spy plot for a sparse matrix stored in .npz format."
    )
    parser.add_argument(
        "matrix",
        help="Input sparse matrix (.npz) file"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI (default: 300)"
    )
    args = parser.parse_args()

    matrix_path = Path(args.matrix)

    # Load sparse matrix
    A = sparse.load_npz(matrix_path)

    # Output filename: <matrix_name>_spy.pdf
    output_file = f"../figures/{matrix_path.stem}_spy.pdf"

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.spy(A, markersize=0.5, rasterized=True)

    ax.set_aspect("equal")

    plt.tight_layout()
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(output_file, bbox_inches="tight", format='pdf')
    plt.close(fig)

    print(f"Saved spy plot to: {output_file}")


if __name__ == "__main__":
    main()
