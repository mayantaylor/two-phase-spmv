#!/usr/bin/env python3
"""
Download SuiteSparse Matrix Collection matrices listed in a CSV file.

CSV format expected (header + rows), e.g.:

    name,id,nrows,ncols,nnz,category,
    MAWI/mawi_201512020030,2802,68863315,68863315,143414960,Undirected Weighted Graph,
    LAW/indochina-2004,2451,7414866,7414866,194109311,Directed Graph,
    ...

Only the "name" column (in "Group/MatrixName" form) is actually needed to
build the download URL; the other columns are informational and are
ignored (but tolerated).

Usage:
    python download_suitesparse.py matrices.csv /path/to/output/dir
    python download_suitesparse.py matrices.csv /path/to/output/dir --format MM
    python download_suitesparse.py matrices.csv /path/to/output/dir --format npz
    python download_suitesparse.py matrices.csv /path/to/output/dir --jobs 4
"""

import argparse
import csv
import glob
import os
import shutil
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# Matrix Market (.tar.gz) is the default/most portable format. Other options
# supported by the SuiteSparse site: "MM" (Matrix Market), "RB" (Rutherford
# Boeing), "mat" (MATLAB .mat file). "npz" is a derived format: we download
# the Matrix Market .tar.gz, convert it with scipy, and save a scipy-sparse
# .npz file (SuiteSparse does not host .npz directly, so it must be built
# locally).
BASE_URL = "https://suitesparse-collection-website.herokuapp.com"

EXT_BY_FORMAT = {
    "MM": ".tar.gz",
    "RB": ".tar.gz",
    "mat": ".mat",
    "npz": ".npz",
}


def build_url(group: str, name: str, fmt: str) -> str:
    if fmt == "mat":
        return f"{BASE_URL}/mat/{group}/{name}.mat"
    if fmt == "npz":
        # npz is derived locally from the MM download; fetch MM under the hood.
        return f"{BASE_URL}/MM/{group}/{name}.tar.gz"
    # MM and RB share the same directory scheme, differing only in the
    # sub-path prefix.
    return f"{BASE_URL}/{fmt}/{group}/{name}.tar.gz"


def parse_csv(csv_path: str):
    """Yield (group, name) tuples parsed from the 'name' column of the CSV."""
    entries = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_name = (row.get("name") or "").strip()
            if not raw_name:
                continue  # skip blank/trailer rows
            if "/" not in raw_name:
                print(f"  ! skipping malformed name (no '/'): {raw_name!r}", file=sys.stderr)
                continue
            group, name = raw_name.split("/", 1)
            entries.append((group.strip(), name.strip()))
    return entries


def already_downloaded(out_path: str, min_size_bytes: int = 1024) -> bool:
    """
    Consider a file already downloaded if it exists and is non-trivially
    sized (guards against a previous truncated/failed download of 0 bytes).
    """
    return os.path.exists(out_path) and os.path.getsize(out_path) >= min_size_bytes


def _download_raw(url: str, dest_path: str, timeout: int):
    """Stream-download url to dest_path, raising on error/incomplete transfer."""
    with requests.get(url, stream=True, timeout=timeout) as resp:
        if resp.status_code == 404:
            raise FileNotFoundError(f"404 not found at {url}")
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        if total and downloaded != total:
            raise IOError(f"incomplete download: got {downloaded} of {total} bytes")


def _convert_mm_targz_to_npz(targz_path: str, name: str, out_path: str):
    """Extract a Matrix Market .tar.gz and save its matrix as a scipy .npz."""
    import scipy.io
    import scipy.sparse

    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(targz_path, "r:gz") as tar:
            tar.extractall(tmpdir)

        mtx_candidates = glob.glob(os.path.join(tmpdir, "**", "*.mtx"), recursive=True)
        if not mtx_candidates:
            raise FileNotFoundError(f"no .mtx file found inside {targz_path}")

        # Prefer a file matching the matrix name exactly if there are multiple
        # (some archives include an extra coordinate/auxiliary .mtx file).
        mtx_path = next(
            (p for p in mtx_candidates if os.path.splitext(os.path.basename(p))[0] == name),
            mtx_candidates[0],
        )

        matrix = scipy.io.mmread(mtx_path)
        matrix = scipy.sparse.csr_matrix(matrix)
        scipy.sparse.save_npz(out_path, matrix)


def download_one(group: str, name: str, out_dir: str, fmt: str, retries: int = 3, timeout: int = 60):
    ext = EXT_BY_FORMAT[fmt]
    filename = f"{name}{ext}"
    out_path = os.path.join(out_dir, filename)

    if already_downloaded(out_path):
        return name, "skipped (already downloaded)", None

    url = build_url(group, name, fmt)

    for attempt in range(1, retries + 1):
        tmp_raw_path = None
        try:
            if fmt == "npz":
                # Download the underlying MM tar.gz to a temp file, convert,
                # then clean up the intermediate archive.
                fd, tmp_raw_path = tempfile.mkstemp(suffix=".tar.gz")
                os.close(fd)
                _download_raw(url, tmp_raw_path, timeout)

                tmp_out_path = out_path + ".part"
                _convert_mm_targz_to_npz(tmp_raw_path, name, tmp_out_path)
                # scipy.sparse.save_npz appends .npz itself; account for that.
                if not os.path.exists(tmp_out_path) and os.path.exists(tmp_out_path + ".npz"):
                    tmp_out_path = tmp_out_path + ".npz"
                os.replace(tmp_out_path, out_path)
            else:
                tmp_path = out_path + ".part"
                _download_raw(url, tmp_path, timeout)
                os.replace(tmp_path, out_path)

            return name, "downloaded", None

        except Exception as e:
            for stray in (out_path + ".part", out_path + ".part.npz"):
                if os.path.exists(stray):
                    os.remove(stray)
            if tmp_raw_path and os.path.exists(tmp_raw_path):
                os.remove(tmp_raw_path)
            if isinstance(e, FileNotFoundError):
                return name, "failed", str(e)
            if attempt == retries:
                return name, "failed", str(e)
            wait = 2 ** attempt
            print(f"  ! {name}: attempt {attempt} failed ({e}); retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
        finally:
            if tmp_raw_path and os.path.exists(tmp_raw_path):
                os.remove(tmp_raw_path)


def main():
    parser = argparse.ArgumentParser(description="Download SuiteSparse matrices listed in a CSV.")
    parser.add_argument("csv_path", help="Path to input CSV file")
    parser.add_argument("output_dir", help="Directory to save downloaded matrix files into")
    parser.add_argument(
        "--format", choices=["MM", "RB", "mat", "npz"], default="npz",
        help=(
            "Matrix format to download (default: MM = Matrix Market .tar.gz). "
            "'npz' downloads Matrix Market under the hood and converts it "
            "locally to a scipy-sparse .npz file (requires scipy)."
        ),
    )
    parser.add_argument(
        "--jobs", type=int, default=4,
        help="Number of concurrent downloads (default: 4)",
    )
    args = parser.parse_args()

    if args.format == "npz":
        try:
            import scipy  # noqa: F401
        except ImportError:
            print(
                "The 'npz' format requires scipy. Install it with: pip install scipy",
                file=sys.stderr,
            )
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    entries = parse_csv(args.csv_path)
    if not entries:
        print("No matrix entries found in CSV.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(entries)} matrices in CSV. Output dir: {args.output_dir}")

    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(download_one, group, name, args.output_dir, args.format): name
            for group, name in entries
        }
        for future in as_completed(futures):
            name, status, err = future.result()
            results.append((name, status, err))
            if status == "downloaded":
                print(f"  \u2713 {name}: downloaded")
            elif status.startswith("skipped"):
                print(f"  - {name}: {status}")
            else:
                print(f"  \u2717 {name}: FAILED ({err})")

    downloaded = sum(1 for _, s, _ in results if s == "downloaded")
    skipped = sum(1 for _, s, _ in results if s.startswith("skipped"))
    failed = sum(1 for _, s, _ in results if s == "failed")

    print("\nSummary:")
    print(f"  downloaded: {downloaded}")
    print(f"  skipped (already present): {skipped}")
    print(f"  failed: {failed}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
