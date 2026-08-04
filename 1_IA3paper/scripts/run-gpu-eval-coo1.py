"""
GPU-only SpMV benchmark, pinned to CUSPARSE_SPMV_COO_ALG1.

Each .npz file is loaded with scipy.sparse.load_npz(), converted to COO,
and timed on the GPU via a direct ctypes call into libcusparse (CuPy's
high-level `A @ x` dispatch does not expose the cuSPARSE algorithm enum,
so it can't be used to pin ALG1). No CPU reference and no CSR path are
run — GPU COO ALG1 only.

Requirements:
    pip install cupy-cuda12x scipy numpy
    (adjust cupy-cuda12x to match your CUDA version)
"""

import ctypes
import ctypes.util
import glob
import os

import numpy as np
import scipy.sparse as sp
import time
import argparse
import csv


# ── CuPy (cuSPARSE backend) — required, this script is GPU-only ─────────────
import cupy as cp


# ─────────────────────────────────────────────────────────────────────────────
# 0. Raw ctypes bindings to libcusparse, pinned to CUSPARSE_SPMV_COO_ALG1
# ─────────────────────────────────────────────────────────────────────────────

# cusparseStatus_t (0 == CUSPARSE_STATUS_SUCCESS)
_CUSPARSE_STATUS_SUCCESS = 0

# cusparseOperation_t
CUSPARSE_OPERATION_NON_TRANSPOSE = 0

# cusparseIndexType_t
CUSPARSE_INDEX_32I = 2

# cusparseIndexBase_t
CUSPARSE_INDEX_BASE_ZERO = 0

# cudaDataType (library_types.h) — stable across CUDA releases
CUDA_R_32F = 0
CUDA_R_64F = 1

# cusparseSpMVAlg_t (verified against current cuSPARSE docs)
CUSPARSE_SPMV_ALG_DEFAULT = 0
CUSPARSE_SPMV_COO_ALG1 = 1
CUSPARSE_SPMV_CSR_ALG1 = 2
CUSPARSE_SPMV_CSR_ALG2 = 3
CUSPARSE_SPMV_COO_ALG2 = 4


def _load_libcusparse():
    """Locate and load libcusparse.so, handling both system installs and
    pip-installed nvidia-cusparse-cuXXx wheels (which aren't on the
    ldconfig cache, so ctypes.util.find_library often can't see them)."""
    name = ctypes.util.find_library("cusparse")
    if name:
        return ctypes.CDLL(name)

    search_dirs = []
    try:
        import nvidia.cusparse
        search_dirs.append(os.path.join(os.path.dirname(nvidia.cusparse.__file__), "lib"))
    except ImportError:
        pass

    for d in search_dirs:
        for so in sorted(glob.glob(os.path.join(d, "libcusparse.so*"))):
            try:
                return ctypes.CDLL(so)
            except OSError:
                continue

    raise RuntimeError(
        "Could not locate libcusparse.so. Ensure the CUDA toolkit or the "
        "'nvidia-cusparse-cuXXx' pip package is installed, or add its lib "
        "directory to LD_LIBRARY_PATH."
    )


def _check(status, fn_name):
    if status != _CUSPARSE_STATUS_SUCCESS:
        raise RuntimeError(f"{fn_name} failed with cusparseStatus_t={status}")


class CusparseCooSpmvAlg1:
    """Thin ctypes wrapper around the generic cuSPARSE SpMV API, hard-pinned
    to CUSPARSE_SPMV_COO_ALG1. Mirrors the official C sample 1:1."""

    def __init__(self):
        self.lib = _load_libcusparse()
        lib = self.lib

        lib.cusparseCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        lib.cusparseCreate.restype = ctypes.c_int

        lib.cusparseDestroy.argtypes = [ctypes.c_void_p]
        lib.cusparseDestroy.restype = ctypes.c_int

        lib.cusparseSetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.cusparseSetStream.restype = ctypes.c_int

        lib.cusparseCreateCoo.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),  # spMatDescr*
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,  # rows, cols, nnz
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # rowInd, colInd, values
            ctypes.c_int, ctypes.c_int, ctypes.c_int,  # idxType, idxBase, valueType
        ]
        lib.cusparseCreateCoo.restype = ctypes.c_int

        lib.cusparseCreateDnVec.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int64, ctypes.c_void_p, ctypes.c_int,
        ]
        lib.cusparseCreateDnVec.restype = ctypes.c_int

        lib.cusparseSpMV_bufferSize.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_int,     # opA
            ctypes.c_void_p,  # alpha
            ctypes.c_void_p,  # matA
            ctypes.c_void_p,  # vecX
            ctypes.c_void_p,  # beta
            ctypes.c_void_p,  # vecY
            ctypes.c_int,     # computeType
            ctypes.c_int,     # alg
            ctypes.POINTER(ctypes.c_size_t),  # bufferSize*
        ]
        lib.cusparseSpMV_bufferSize.restype = ctypes.c_int

        lib.cusparseSpMV.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
        ]
        lib.cusparseSpMV.restype = ctypes.c_int

        lib.cusparseDestroySpMat.argtypes = [ctypes.c_void_p]
        lib.cusparseDestroySpMat.restype = ctypes.c_int
        lib.cusparseDestroyDnVec.argtypes = [ctypes.c_void_p]
        lib.cusparseDestroyDnVec.restype = ctypes.c_int

        handle = ctypes.c_void_p()
        _check(lib.cusparseCreate(ctypes.byref(handle)), "cusparseCreate")
        self.handle = handle
        # keep cuSPARSE on CuPy's default (null) stream so cp.cuda.Event
        # timing stays accurate
        _check(lib.cusparseSetStream(self.handle, ctypes.c_void_p(cp.cuda.Stream.null.ptr)),
               "cusparseSetStream")

    def close(self):
        _check(self.lib.cusparseDestroy(self.handle), "cusparseDestroy")

    def spmv_coo_alg1(self, rows, cols, nnz, row_ind, col_ind, values,
                       x, y, cuda_dtype, n_trials, alpha=1.0, beta=0.0):
        """row_ind, col_ind, values, x, y are cupy device arrays.
        cuda_dtype is CUDA_R_32F or CUDA_R_64F. Returns mean ms over n_trials."""
        lib = self.lib
        ctype = ctypes.c_float if cuda_dtype == CUDA_R_32F else ctypes.c_double
        alpha_c = ctype(alpha)
        beta_c = ctype(beta)

        matA = ctypes.c_void_p()
        _check(lib.cusparseCreateCoo(
            ctypes.byref(matA), rows, cols, nnz,
            ctypes.c_void_p(row_ind.data.ptr),
            ctypes.c_void_p(col_ind.data.ptr),
            ctypes.c_void_p(values.data.ptr),
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, cuda_dtype,
        ), "cusparseCreateCoo")

        vecX = ctypes.c_void_p()
        _check(lib.cusparseCreateDnVec(
            ctypes.byref(vecX), cols, ctypes.c_void_p(x.data.ptr), cuda_dtype,
        ), "cusparseCreateDnVec(x)")

        vecY = ctypes.c_void_p()
        _check(lib.cusparseCreateDnVec(
            ctypes.byref(vecY), rows, ctypes.c_void_p(y.data.ptr), cuda_dtype,
        ), "cusparseCreateDnVec(y)")

        buffer_size = ctypes.c_size_t(0)
        _check(lib.cusparseSpMV_bufferSize(
            self.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            ctypes.byref(alpha_c), matA, vecX, ctypes.byref(beta_c), vecY,
            cuda_dtype, CUSPARSE_SPMV_COO_ALG1, ctypes.byref(buffer_size),
        ), "cusparseSpMV_bufferSize")

        d_buffer = cp.cuda.alloc(buffer_size.value) if buffer_size.value > 0 else None
        buffer_ptr = ctypes.c_void_p(d_buffer.ptr if d_buffer is not None else 0)

        def run_once():
            _check(lib.cusparseSpMV(
                self.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                ctypes.byref(alpha_c), matA, vecX, ctypes.byref(beta_c), vecY,
                cuda_dtype, CUSPARSE_SPMV_COO_ALG1, buffer_ptr,
            ), "cusparseSpMV")

        print(f"    [coo_alg1] nnz={nnz}, buffer_size={buffer_size.value/1e6:.1f} MB, "
              f"starting warmup...", flush=True)

        # Warmup — timed individually so a slow ALG1 run is visible, not silent
        warmup_start = time.perf_counter()
        for i in range(20):
            t0 = time.perf_counter()
            run_once()
            cp.cuda.Stream.null.synchronize()
            if i == 0:
                print(f"    [coo_alg1] first call took {(time.perf_counter()-t0)*1e3:.1f} ms",
                      flush=True)
        print(f"    [coo_alg1] warmup done in {time.perf_counter()-warmup_start:.2f}s, "
              f"running {n_trials} timed trials...", flush=True)

        start_ev = cp.cuda.Event()
        stop_ev = cp.cuda.Event()
        start_ev.record()
        for _ in range(n_trials):
            run_once()
        stop_ev.record()
        stop_ev.synchronize()
        mean_ms = cp.cuda.get_elapsed_time(start_ev, stop_ev) / n_trials

        _check(lib.cusparseDestroySpMat(matA), "cusparseDestroySpMat")
        _check(lib.cusparseDestroyDnVec(vecX), "cusparseDestroyDnVec(x)")
        _check(lib.cusparseDestroyDnVec(vecY), "cusparseDestroyDnVec(y)")

        return mean_ms


_CUSPARSE_ALG1 = CusparseCooSpmvAlg1()


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
        "gpu_coo_alg1_mean_ms",
        "gpu_coo_alg1_bw_GBs",
    ]

    file_exists = os.path.exists(csv_path)
    write_header = (not file_exists) or os.path.getsize(csv_path) == 0

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow(header)

        for r in results:
            gpu = r["gpu"]

            writer.writerow([
                r["path"],
                r["shape"][0],
                r["shape"][1],
                r["nnz"],
                gpu["mean_ms"],
                gpu["bw_GBs"],
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


def load_and_run_matrices(paths: list[str], gpu_trials, np_dtype: np.dtype) -> list[dict]:

    results = []

    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Matrix file not found: '{path}'.\n"
                f"Check that the manifest path is correct and the file exists."
            )

        A = sp.load_npz(path)
        A_coo = A.tocoo().astype(np_dtype)           # for cuSPARSE COO ALG1

        gpu_stats = benchmark_cusparse_coo_alg1(A_coo, n_trials=gpu_trials)

        results.append({
            "path":  path,
            "shape": A_coo.shape,
            "nnz":   A_coo.nnz,
            "gpu":   gpu_stats,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2.  cuSPARSE SpMV — pinned to CUSPARSE_SPMV_COO_ALG1
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_cusparse_coo_alg1(A_coo: sp.coo_matrix, n_trials: int = 200) -> dict:
    """
    Uploads the matrix (COO) once, then times cusparseSpMV with
    CUSPARSE_SPMV_COO_ALG1 explicitly, using CUDA events for accurate
    GPU-side timing.
    """
    rows, cols = A_coo.shape
    nnz = A_coo.nnz
    val_bytes = A_coo.dtype.itemsize
    cuda_dtype = CUDA_R_32F if A_coo.dtype == np.float32 else CUDA_R_64F

    row_ind = cp.asarray(A_coo.row.astype(np.int32))
    col_ind = cp.asarray(A_coo.col.astype(np.int32))
    values = cp.asarray(A_coo.data)
    x = cp.random.rand(cols).astype(A_coo.dtype)
    y = cp.zeros(rows, dtype=A_coo.dtype)

    mean_ms = _CUSPARSE_ALG1.spmv_coo_alg1(
        rows, cols, nnz, row_ind, col_ind, values, x, y,
        cuda_dtype, n_trials,
    )

    bytes_moved = (nnz * val_bytes            # values
                   + nnz * 4                  # rowInd (int32)
                   + nnz * 4                  # colInd (int32)
                   + cols * val_bytes         # x vector
                   + rows * val_bytes)        # y vector
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
        description="SpMV benchmark (GPU pinned to CUSPARSE_SPMV_COO_ALG1)",
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
    parser.add_argument("--gpu-trials", type=int, default=50,
                        help="Number of GPU SpMV trials per matrix (default: 50)")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                        help="Floating-point precision for SpMV (default: float32)")

    args = parser.parse_args()
    np_dtype = np.dtype(args.dtype)

    all_paths = read_manifest(args.manifest)

    try:
        results = load_and_run_matrices(all_paths, args.gpu_trials, np_dtype)
    except FileNotFoundError as exc:
        print(f"  ERROR: {exc}")
        return

    append_results_csv(args.output_csv, results)
    print(f"Results appended to {args.output_csv}")

    _CUSPARSE_ALG1.close()


if __name__ == "__main__":
    main()
