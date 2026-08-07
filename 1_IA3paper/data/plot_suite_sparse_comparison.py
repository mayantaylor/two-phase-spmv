import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from style import figure, style_axes, save


df = pd.read_csv('ss-results/080326coo-gpu_eval.csv')

wse_results_df = pd.read_csv('ss-results/080626results_full-coprimes.csv')

# Group by matrix name and take the row with minimum max_time for each matrix
wse_results_df['matrix_name'] = wse_results_df['name'].astype(str).str.split('.').str[0]
wse_results_df = wse_results_df[wse_results_df['status'] != 'skipped']

# Group by matrix name and take the row with minimum max_time for each matrix
wse_results_df = wse_results_df.sort_values('max_time')
wse_results_df = wse_results_df.drop_duplicates(subset=['matrix_name'], keep='first')

# Extract the last part of the matrix path (filename)
df['matrix_name'] = df['matrix_path'].str.split('/').str[-1].str.split('.').str[0]
wse_results_df['matrix_name'] = wse_results_df['name'].astype(str).str.split('.').str[0]

df = df.groupby('matrix_name').agg({
    'gpu_mean_ms': 'mean',
    'rows': 'first',
    'nnz': 'first',
    'cols': 'first'
}).reset_index()

df = df.sort_values('rows')

# Merge with wse_results_df to include comparison data
df = df.merge(wse_results_df[['matrix_name', 'max_time', 'nnz_max_capacity', 'nrows', 'ncols']],
              on='matrix_name', how='left', suffixes=('', '_results'))
df = df.dropna(subset=['max_time'])

df['max_time'] = pd.to_numeric(df['max_time'], errors='coerce') / 1000000  # convert from cycles to ms

# Calculate throughput (nnz per ms)
df['throughput'] = df['nnz'] / df['gpu_mean_ms']

# Create figure with subplots
x = np.arange(len(df))
width = 0.35
font_size = 20

df['speedup'] = df['gpu_mean_ms'] / df['max_time']

df['nnz_capacity_ratio'] = df['nnz_max_capacity'] / (df['nnz'] / (df['nrows'] * df['ncols']))


fig, ax1 = figure("rect")
ax2 = ax1.twinx()

# Left axis: speedup bars rooted at 1
heights = df['speedup'] - 1
ax1.bar(
    x,
    heights,
    bottom=1,
    width=0.8,
    color='tab:blue',
    alpha=0.7,
    label='WSE Speedup',
)
ax1.axhline(1, color='red', linestyle='--')

ax1.set_ylabel("WSE Speedup", color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Left limits
y1min = 0
y1max = max(df['speedup'].max(), 1) + 0.5
ax1.set_ylim(y1min, y1max)

# Fractional location of y=1 on left axis
frac = (1 - y1min) / (y1max - y1min)

# Right axis: load imbalance ratio
ax2.plot(
    x,
    df['nnz_capacity_ratio'],
    color='k',
    marker='o',
    label='Load Imbalance Ratio'
)
ax2.set_ylabel("Load Imbalance Ratio", color='k')
ax2.tick_params(axis='y', labelcolor='k')

rmax = max(df['nnz_capacity_ratio'].max(), 1)

# Use the *actual* top of the right axis (including the +5 headroom) when
# solving for new_rmin. The previous version solved using rmax but then set
# the limit to rmax + 5, so the fraction used to place y=1 didn't match the
# fraction that resulted after the +5 was applied — that mismatch is what
# caused the 1-lines to be misaligned.
rtop = rmax + 5

# Compute new lower limit so that 1 sits at the same fraction as on ax1:
#   frac == (1 - new_rmin) / (rtop - new_rmin)
new_rmin = (frac * rtop - 1) / (frac - 1)

ax2.set_ylim(new_rmin, rtop)

# Shared x-axis
ax1.set_xticks(x)
ax1.set_xticklabels(df['matrix_name'], rotation=90, ha='right')

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

style_axes(ax1)
save(fig,'../figures/suite_sparse_gpu_comparison.pdf')

