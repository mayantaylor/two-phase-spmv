"""
Each .npz file is loaded with scipy.sparse.load_npz().

Requirements:
    pip install cupy-cuda12x scipy numpy
    (adjust cupy-cuda12x to match your CUDA version)
"""

import numpy as np
import scipy.sparse as sp
import time
import os
import argparse

import csv


# ── Try to import CuPy (cuSPARSE backend) ────────────────────────────────────
try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
    HAVE_CUPY = True
except ImportError:
    HAVE_CUPY = False
    print("CuPy not found — will run CPU-only reference timings.\n")


import csv

        
def append_results_csv(csv_path: str, results: list):
    """
    Append benchmark results to a CSV file. Creates the file and header
    if it does not already exist or is empty.
    """
    header = [
        "matrix_path",
        "rows",
        "cols",
        "nnz",
        "cpu_mean_ms",
        "cpu_bw_GBs",
        "gpu_mean_ms",
        "gpu_bw_GBs",
    ]

    file_exists = os.path.exists(csv_path)
    write_header = (not file_exists) or os.path.getsize(csv_path) == 0

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow(header)

        for r in results:
            cpu = r["cpu"]
            gpu = r.get("gpu")

            writer.writerow([
                r["path"],
                r["shape"][0],
                r["shape"][1],
                r["nnz"],
                cpu["mean_ms"],
                cpu["bw_GBs"],
                gpu["mean_ms"] if gpu else "",
                gpu["bw_GBs"] if gpu else "",
            ])

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Manifest parsing and matrix loading
# ─────────────────────────────────────────────────────────────────────────────

def read_manifest(manifest_path: str) -> list[str]:
    """
    Read a manifest file listing .npz matrix paths, one per line.

    Blank lines and lines starting with '#' are skipped. Relative paths are
    resolved relative to the manifest's directory first, then relative to
    the current working directory (whichever exists).
    """
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))

    paths = []
    with open(manifest_path) as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            path = line
            if not os.path.isabs(path):
                candidate = os.path.join(manifest_dir, path)
                if os.path.exists(candidate):
                    path = candidate
                # else: leave as-is, resolved relative to CWD; existence is
                # checked later when the file is actually loaded.

            paths.append(path)

    if not paths:
        raise ValueError(f"Manifest '{manifest_path}' contains no matrix paths.")

    return paths


def load_and_run_matrices(paths: list[str], cpu_trials, gpu_trials, np_dtype: np.dtype) -> tuple[list[dict], dict | None]:

    results = []

    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Matrix file not found: '{path}'.\n"
                f"Check that the manifest path is correct and the file exists."
            )

        A = sp.load_npz(path)
        A = A.tocsr().astype(np_dtype)
        
        print("A has ", A.nnz, "nonzeros")
        
        cpu_stats = benchmark_cpu(A, n_trials=cpu_trials)
        gpu_stats = benchmark_cusparse(A, n_trials=gpu_trials) if HAVE_CUPY else None
        
        results.append({
            "path":  path,
            "shape": A.shape,
            "nnz":   A.nnz,
            "cpu":   cpu_stats,
            "gpu":   gpu_stats,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CPU reference SpMV
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_cpu(A_scipy: sp.csr_matrix, n_trials: int = 50) -> dict:
    n = A_scipy.shape[1]
    val_bytes = A_scipy.dtype.itemsize          # 4 (float32) or 8 (float64)
    x = np.random.rand(n).astype(A_scipy.dtype)

    # Warmup
    for _ in range(5):
        _ = A_scipy @ x

    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        y = A_scipy @ x          # noqa: F841
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    bytes_moved = (A_scipy.nnz * val_bytes       # values  (float32/64)
                   + A_scipy.nnz * 4             # col_ind (int32)
                   + (A_scipy.shape[0]+1)*4      # row_ptr (int32)
                   + n * val_bytes               # x vector
                   + A_scipy.shape[0] * val_bytes)  # y vector
    bw = bytes_moved / times.mean() / 1e9    # GB/s

    return {
        "mean_ms":   float(times.mean()      * 1e3),
        "bw_GBs":    float(bw),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  cuSPARSE SpMV via CuPy
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_cusparse(A_scipy: sp.csr_matrix, n_trials: int = 200) -> dict:
    """
    Uploads the matrix once, then times SpMV on-device with CUDA events
    for accurate GPU timing.
    """
    n = A_scipy.shape[1]
    val_bytes = A_scipy.dtype.itemsize          # 4 (float32) or 8 (float64)
    cp_dtype  = cp.dtype(A_scipy.dtype)

    # Upload to device
    A_gpu = cpsp.csr_matrix(A_scipy)
    x_gpu = cp.random.rand(n, dtype=cp_dtype)

    # Warmup — triggers JIT compilation / cuSPARSE handle init
    for _ in range(20):
        y_gpu = A_gpu @ x_gpu   # noqa: F841
    cp.cuda.Stream.null.synchronize()

    start_ev = cp.cuda.Event()
    stop_ev  = cp.cuda.Event()

    start_ev.record()
    for _ in range(n_trials):
        y_gpu = A_gpu @ x_gpu   # noqa: F841
    stop_ev.record()
    stop_ev.synchronize()
    mean_ms = cp.cuda.get_elapsed_time(start_ev, stop_ev) / n_trials

    bytes_moved = (A_scipy.nnz * val_bytes
                   + A_scipy.nnz * 4
                   + (A_scipy.shape[0]+1)*4
                   + n * val_bytes
                   + A_scipy.shape[0] * val_bytes)
    bw = bytes_moved / (mean_ms * 1e-3) / 1e9  # GB/s

    return {
        "mean_ms": float(mean_ms),
        "bw_GBs":  float(bw),
    }



# ─────────────────────────────────────────────────────────────────────────────
# 6.  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SpMV benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "manifest", metavar="MANIFEST_TXT",
        help="Path to a manifest.txt file listing one .npz matrix path per line",
    )
    
    parser.add_argument(
        "output_csv",
        help="Path to output.csv file"
    )
    parser.add_argument("--cpu-trials", type=int, default=20,
                        help="Number of CPU SpMV trials per matrix (default: 20)")
    parser.add_argument("--gpu-trials", type=int, default=50,
                        help="Number of GPU SpMV trials per matrix (default: 50)")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                        help="Floating-point precision for SpMV (default: float64)")

    args = parser.parse_args()
    np_dtype = np.dtype(args.dtype)

    all_paths = read_manifest(args.manifest)

 
    try:
        results = load_and_run_matrices(all_paths, args.cpu_trials, args.gpu_trials, np_dtype)
    except FileNotFoundError as exc:
        print(f"  ERROR: {exc}")

    append_results_csv(args.output_csv, results)
    print(f"Results appended to {args.output_csv}")


if __name__ == "__main__":
    main()
