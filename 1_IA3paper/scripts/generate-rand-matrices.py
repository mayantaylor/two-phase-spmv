#!/usr/bin/env python3

import argparse
import os
import numpy as np
import scipy.sparse as sp


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate sparse matrices and save them as .npz files."
    )

    parser.add_argument("outfile", type=str, help="Output .npz filename")
    parser.add_argument("m", type=int, help="Number of rows")
    parser.add_argument("n", type=int, help="Number of columns")
    parser.add_argument(
        "mode",
        choices=["random", "diagonal_spectrum"],
        help="Matrix generation mode",
    )

    parser.add_argument(
        "--nnz",
        type=int,
        default=None,
        help="Exact number of nonzeros to generate",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=None,
        help="Density in [0,1]. If --nnz is not provided, nnz is computed from density.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=4,
        help="Random seed",
    )

    parser.add_argument(
        "--vmin",
        type=float,
        default=0.0,
        help="Minimum value for generated nonzeros",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=10.0,
        help="Maximum value for generated nonzeros",
    )
    parser.add_argument(
        "--integer-values",
        action="store_true",
        help="Use integer values in [vmin, vmax)",
    )

    parser.add_argument(
        "--spread",
        type=float,
        default=0.0,
        help=(
            "Diagonal spectrum parameter in [0,1]. "
            "0 = as diagonal-concentrated as possible for the requested nnz, "
            "1 = fully uniform random."
        ),
    )

    parser.add_argument(
        "--generate-spectrum5",
        action="store_true",
        help=(
            "For diagonal_spectrum mode, generate 5 matrices spanning the full spectrum "
            "with spreads [0.0, 0.25, 0.5, 0.75, 1.0]. "
            "Files are written using the output filename stem."
        ),
    )

    return parser.parse_args()


def resolve_nnz(m, n, nnz, density):
    total = m * n

    if nnz is not None and density is not None:
        raise ValueError("Specify only one of --nnz or --density, not both")

    if nnz is None and density is None:
        raise ValueError("You must specify one of --nnz or --density")

    if density is not None:
        if not (0.0 <= density <= 1.0):
            raise ValueError("density must be in [0,1]")
        nnz = int(round(density * total))

    if nnz is None:
        raise ValueError("Failed to resolve nnz")

    if nnz < 0 or nnz > total:
        raise ValueError(f"nnz must satisfy 0 <= nnz <= m*n = {total}")

    return int(nnz)


def sample_values(rng, nnz, vmin, vmax, integer_values):
    if nnz == 0:
        return np.array([], dtype=np.float32)

    if integer_values:
        lo = int(np.floor(vmin))
        hi = int(np.ceil(vmax))
        if hi <= lo:
            raise ValueError("For integer values, need vmax > vmin")
        return rng.integers(lo, hi, size=nnz).astype(np.float32)

    if vmax < vmin:
        raise ValueError("Need vmax >= vmin")
    return rng.uniform(vmin, vmax, size=nnz).astype(np.float32)


def generate_random_matrix(m, n, nnz, rng, vmin, vmax, integer_values):
    if nnz == 0:
        return sp.coo_matrix((m, n), dtype=np.float32)

    flat = rng.choice(m * n, size=nnz, replace=False)
    rows = flat // n
    cols = flat % n
    data = sample_values(rng, nnz, vmin, vmax, integer_values)

    return sp.coo_matrix((data, (rows, cols)), shape=(m, n), dtype=np.float32)


def diagonal_center_col(i, m, n):
    return min((i * n) // m, n - 1)


def count_band_capacity(m, n, w):
    total = 0
    for i in range(m):
        j0 = diagonal_center_col(i, m, n)
        lo = max(0, j0 - w)
        hi = min(n, j0 + w + 1)
        total += (hi - lo)
    return total


def min_width_for_nnz(m, n, nnz):
    lo = 0
    hi = max(m, n)

    while lo < hi:
        mid = (lo + hi) // 2
        cap = count_band_capacity(m, n, mid)
        if cap >= nnz:
            hi = mid
        else:
            lo = mid + 1

    return lo


def enumerate_band_positions(m, n, w):
    row_chunks = []
    col_chunks = []

    for i in range(m):
        j0 = diagonal_center_col(i, m, n)
        lo = max(0, j0 - w)
        hi = min(n, j0 + w + 1)

        if hi <= lo:
            continue

        cols = np.arange(lo, hi, dtype=np.int64)
        rows = np.full(cols.shape, i, dtype=np.int64)

        row_chunks.append(rows)
        col_chunks.append(cols)

    if not row_chunks:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    return np.concatenate(row_chunks), np.concatenate(col_chunks)


def generate_diagonal_spectrum_matrix(m, n, nnz, spread, rng, vmin, vmax, integer_values):
    if not (0.0 <= spread < 1.0):
        raise ValueError("spread must be in [0,1)")

    if nnz == 0:
        return sp.coo_matrix((m, n), dtype=np.float32)
    
    w_min = min_width_for_nnz(m, n, nnz)
    w_full = max(m, n)
    w = int(round(w_min + spread * (w_full - w_min)))

    if w >= n:
        return generate_random_matrix(m, n, nnz, rng, vmin, vmax, integer_values)

    band_rows, band_cols = enumerate_band_positions(m, n, w)
    capacity = band_rows.size

    if capacity < nnz:
        raise RuntimeError(
            f"Band capacity {capacity} is smaller than requested nnz {nnz}"
        )

    idx = rng.choice(capacity, size=nnz, replace=False)
    rows = band_rows[idx]
    cols = band_cols[idx]
    data = sample_values(rng, nnz, vmin, vmax, integer_values)

    return sp.coo_matrix((data, (rows, cols)), shape=(m, n), dtype=np.float32)


def save_matrix(outfile, A):
    A = A.tocsr()
    sp.save_npz(outfile, A)
    return A


def print_summary(outfile, mode, A, m, n, nnz, spread=None):
    realized_density = A.nnz / (m * n)

    print(f"Saved matrix to {outfile}")
    print(f"Mode: {mode}")
    print(f"Shape: {A.shape}")
    print(f"NNZ: {A.nnz}")
    print(f"Density: {realized_density:.8f}")

    if spread is not None:
        w_min = min_width_for_nnz(m, n, nnz)
        w_full = max(m, n)
        w = int(round(w_min + spread * (w_full - w_min)))
        print(f"Spread: {spread:.6f}")
        print(f"Minimum feasible band half-width: {w_min}")
        print(f"Chosen band half-width: {w}")

    print("")


def split_outfile(outfile):
    stem, ext = os.path.splitext(outfile)
    if ext == "":
        ext = ".npz"
    return stem, ext


def main():
    args = parse_args()

    if args.m <= 0:
        raise ValueError("m must be > 0")
    if args.n <= 0:
        raise ValueError("n must be > 0")

    nnz = resolve_nnz(args.m, args.n, args.nnz, args.density)

    if args.generate_spectrum5 and args.mode != "diagonal_spectrum":
        raise ValueError("--generate-spectrum5 is only valid with mode=diagonal_spectrum")

    if args.mode == "random":
        rng = np.random.default_rng(args.seed)
        A = generate_random_matrix(
            args.m,
            args.n,
            nnz,
            rng,
            args.vmin,
            args.vmax,
            args.integer_values,
        )
        A = save_matrix(args.outfile, A)
        print_summary(args.outfile, args.mode, A, args.m, args.n, nnz)

    elif args.mode == "diagonal_spectrum":
        if args.generate_spectrum5:
            spreads = [0.0, 0.25, 0.5, 0.75, 1.0]
            stem, ext = split_outfile(args.outfile)

            for idx, spread in enumerate(spreads):
                rng = np.random.default_rng(args.seed + idx)
                outfile = f"{stem}_s{spread:.2f}{ext}"

                A = generate_diagonal_spectrum_matrix(
                    args.m,
                    args.n,
                    nnz,
                    spread,
                    rng,
                    args.vmin,
                    args.vmax,
                    args.integer_values,
                )
                A = save_matrix(outfile, A)
                print_summary(outfile, args.mode, A, args.m, args.n, nnz, spread=spread)
        else:
            rng = np.random.default_rng(args.seed)
            A = generate_diagonal_spectrum_matrix(
                args.m,
                args.n,
                nnz,
                args.spread,
                rng,
                args.vmin,
                args.vmax,
                args.integer_values,
            )
            A = save_matrix(args.outfile, A)
            print_summary(args.outfile, args.mode, A, args.m, args.n, nnz, spread=args.spread)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()