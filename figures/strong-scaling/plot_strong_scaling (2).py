"""
Plot strong-scaling SpMV results on the Cerebras WSE.

Reads the "strong-scaling" sheet of WSE3_Spmv_Results.xlsx, which contains
several stacked blocks (one per sparsity density), each with a header row
of the form:

    ('strong scaling  (<density> density, block decomp)', ...)

followed by a column-header row and then one data row per PE grid size
(pe_dim in {512, 256, 128, 64, 32}). Newer blocks report 3 seeds per
pe_dim (columns for max-per-pe / row-major-time / max-comp-time repeated
3x); the very first legacy block has only a single unlabeled run and is
skipped by default (see SKIP_LEGACY_BLOCK below).

Usage:
    python plot_strong_scaling.py [path_to_xlsx] [output_png]
"""

import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import openpyxl

SHEET_NAME = "strong-scaling"

# The first block in the sheet is an older, single-seed .01-density run
# that has since been superseded by a 3-seed .01-density block later in
# the sheet. Skip it to avoid plotting the same density twice.
SKIP_LEGACY_BLOCK = True


def parse_strong_scaling(path, sheet_name=SHEET_NAME):
    """Parse the strong-scaling sheet into:
        row_major:   {density: {pe_dim: [row_major_times]}}
        rows_per_pe: {density: {pe_dim: B}}   (B = rows/cols per PE)
        max_nnz:     {density: {pe_dim: [max_nnz_per_pe]}}  (one value per seed)
        comp_time:   {density: {pe_dim: [max_comp_times]}}  (one value per seed)
    """
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[sheet_name]
    rows = list(ws.iter_rows(values_only=True))

    row_major = defaultdict(lambda: defaultdict(list))
    rows_per_pe = defaultdict(dict)
    max_nnz = defaultdict(lambda: defaultdict(list))
    comp_time = defaultdict(lambda: defaultdict(list))

    i = 0
    first_block = True
    while i < len(rows):
        cell0 = rows[i][0]
        if isinstance(cell0, str) and cell0.strip().lower().startswith("strong scaling"):
            match = re.search(r"\(\s*([\d.]+)\s*density", cell0)
            density = float(match.group(1)) if match else None

            header_row = rows[i + 1]
            # Find every column labeled 'row-major' -- one per seed
            row_major_cols = [c for c, v in enumerate(header_row)
                               if isinstance(v, str) and v.strip().lower() == "row-major"]
            # Find every column labeled 'max per pe' -- one per seed
            max_nnz_cols = [c for c, v in enumerate(header_row)
                             if isinstance(v, str) and v.strip().lower() == "max per pe"]
            # Find every column labeled 'max comp time' -- one per seed
            comp_time_cols = [c for c, v in enumerate(header_row)
                               if isinstance(v, str) and v.strip().lower() == "max comp time"]

            is_legacy = (len(row_major_cols) <= 1)
            skip_this_block = is_legacy and first_block and SKIP_LEGACY_BLOCK
            first_block = False

            # Walk data rows until a blank row or a new block header
            j = i + 2
            while j < len(rows) and rows[j][0] is not None and not (
                isinstance(rows[j][0], str) and rows[j][0].strip().lower().startswith("strong scaling")
            ):
                pe_dim = rows[j][0]
                if pe_dim is not None and not skip_this_block:
                    b_val = rows[j][1]  # 'rows/cols per pe' column
                    if b_val is not None:
                        rows_per_pe[density][int(pe_dim)] = float(b_val)
                    for col in row_major_cols:
                        val = rows[j][col]
                        if val is not None:
                            row_major[density][int(pe_dim)].append(float(val))
                    for col in max_nnz_cols:
                        val = rows[j][col]
                        if val is not None:
                            max_nnz[density][int(pe_dim)].append(float(val))
                    for col in comp_time_cols:
                        val = rows[j][col]
                        if val is not None:
                            comp_time[density][int(pe_dim)].append(float(val))
                j += 1
            i = j
        else:
            i += 1

    return row_major, rows_per_pe, max_nnz, comp_time


def plot_strong_scaling(results, rows_per_pe, comp_time, out_path, skip_densities=None):
    skip_densities = set(skip_densities or [])
    fig, ax = plt.subplots(figsize=(7, 5))

    densities = [d for d in sorted(results.keys()) if d not in skip_densities]
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(densities)))

    comp_label_used = False
    for density, color in zip(densities, colors):
        pe_dims = sorted(results[density].keys())  # increasing: 32 -> 512
        means = [np.mean(results[density][d]) for d in pe_dims]
        stds = [np.std(results[density][d]) if len(results[density][d]) > 1 else 0
                for d in pe_dims]

        ax.errorbar(
            pe_dims, means, yerr=stds,
            marker="o", capsize=3, color=color,
            label=f"density = {density}",
        )

        # Highlight the minimum (best) point on this line
        min_idx = int(np.argmin(means))
        ax.scatter(
            [pe_dims[min_idx]], [means[min_idx]],
            s=180, facecolors="none", edgecolors=color,
            linewidths=2.2, zorder=5,
        )

        # Actual measured computation time (mean across seeds), per density
        comp_pe_dims = sorted(comp_time[density].keys())
        comp_vals = [np.mean(comp_time[density][d]) for d in comp_pe_dims]
        ax.plot(
            comp_pe_dims, comp_vals,
            linestyle=":", marker="^", markersize=6, color=color, linewidth=2,
            label="measured computation time" if not comp_label_used else None,
        )
        comp_label_used = True

    # Communication model: 4B + 14.6P, B = rows/cols per PE, P = pe grid dim.
    # B depends only on pe_dim (fixed problem size), not density, so this is
    # a single shared curve -- computed from whichever density has full data.
    ref_density = densities[0]
    model_pe_dims = sorted(rows_per_pe[ref_density].keys())
    model_vals = [4 * rows_per_pe[ref_density][d] + 14.6 * d for d in model_pe_dims]
    ax.plot(
        model_pe_dims, model_vals,
        linestyle=":", marker="s", markersize=6, color="black", linewidth=2,
        label="communication model (4B + 14.6P)",
    )

    ax.set_xscale("log", base=2)
    all_pe_dims = sorted(set().union(*[set(results[d].keys()) for d in densities]))
    ax.set_xticks(all_pe_dims)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    # x-axis increases left-to-right: 32 -> 512 (finer grid, more PEs)

    ax.set_xlabel("PE grid dimension (total PEs = dim$^2$)")
    ax.set_ylabel("Row-major execution time (cycles)")
    ax.set_title("Strong Scaling: SpMV on Cerebras WSE")
    ax.legend(title="Sparsity density")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    xlsx_path = sys.argv[1] if len(sys.argv) > 1 else "WSE3_Spmv_Results.xlsx"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "strong_scaling.png"

    row_major, rows_per_pe, max_nnz, comp_time = parse_strong_scaling(xlsx_path)

    print("Parsed densities and PE grid sizes found:")
    for density in sorted(row_major.keys()):
        print(f"  density={density}: pe_dims={sorted(row_major[density].keys(), reverse=True)}, "
              f"seeds per point={[len(v) for v in row_major[density].values()]}")

    # Skip density 0.01 in the plotted output
    plot_strong_scaling(row_major, rows_per_pe, comp_time, out_path, skip_densities=[0.01])
