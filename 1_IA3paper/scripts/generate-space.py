#!/usr/bin/env python3
"""
Batch-runs generate-rand-matrices.py for a grid of (m=n, nnz-per-row) values.

Usage:
    python run_generate_matrices.py [--script PATH] [--outdir DIR] [--dry-run]

Assumes generate-rand-matrices.py has the CLI:
    generate-rand-matrices.py [-h] [--nnz NNZ] outfile m n {random}
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Matrix dimensions (m = n)
SIZES = [8192, 32768, 131072, 524288, 2097152]

# nnz per row/matrix
NNZ_VALUES = [16, 64, 256, 1024]

MODE = "random"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--script",
        default="generate-rand-matrices.py",
        help="Path to generate-rand-matrices.py (default: %(default)s)",
    )
    parser.add_argument(
        "--outdir",
        default="matrices",
        help="Directory to write generated matrix files into (default: %(default)s)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to invoke the script with (default: current interpreter)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands instead of running them",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    failures = []

    for m in SIZES:
        n = m
        for nnz in NNZ_VALUES:
            full_nnz = nnz * m
            outfile = outdir / f"matrix_{m}rows_{nnz}nnzperrow.npz"

            cmd = [
                args.python,
                args.script,
                "--nnz", str(full_nnz),
                str(outfile),
                str(m),
                str(n),
                MODE,
            ]

            print("Running:", " ".join(cmd))

            if args.dry_run:
                continue

            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"  -> FAILED (exit code {result.returncode}): {outfile}", file=sys.stderr)
                failures.append((m, n, nnz, result.returncode))

    if failures:
        print("\nSome runs failed:", file=sys.stderr)
        for m, n, nnz, code in failures:
            print(f"  m={m} n={n} nnzperrow={nnz} -> exit {code}", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nAll matrices generated successfully." if not args.dry_run else "\nDry run complete.")


if __name__ == "__main__":
    main()
