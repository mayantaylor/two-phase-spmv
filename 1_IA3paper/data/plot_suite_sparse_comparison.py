import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Read the CSV file
if not os.path.exists('suite_sparse_gpu_eval.csv') or not os.path.exists('suite-sparse-results073026.csv'):
    print("Error: CSV files not found")
    exit()

df = pd.read_csv('suite_sparse_gpu_eval.csv')
results_df = pd.read_csv('suite-sparse-results073026.csv')

# Extract the last part of the matrix path (filename)
df['matrix_name'] = df['matrix_path'].str.split('/').str[-1]
results_df['matrix_name'] = results_df['name'].astype(str).str.split('.').str[0] + ".npz"

df = df.groupby('matrix_name').agg({
    'gpu_mean_ms': 'mean',
    'rows': 'first',
    'nnz': 'first',
    'cols': 'first'
}).reset_index()

df = df.sort_values('rows')

# Merge with results_df to include comparison data
df = df.merge(results_df[['matrix_name', 'max_time', 'nnz_max_capacity', 'nrows', 'ncols']], 
              on='matrix_name', how='left', suffixes=('', '_results'))

df['max_time'] = pd.to_numeric(df['max_time'], errors='coerce') / 1000000 # convert from cycles to ms

# Calculate throughput (nnz per ms)
df['throughput'] = df['nnz'] / df['gpu_mean_ms']

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

x = np.arange(len(df))
width = 0.35

df['speedup'] = df['gpu_mean_ms'] / df['max_time']

# Compute bar heights relative to 1
heights = df['speedup'] - 1
df['nnz_capacity_ratio'] = df['nnz_max_capacity'] / (df['nnz'] / (df['nrows'] * df['ncols']))

fig, ax1 = plt.subplots(figsize=(12, 5))
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
    label='WSE Speedup'
)
ax1.axhline(1, color='red', linestyle='--', linewidth=1)

ax1.set_ylabel("WSE Speedup", color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Example left limits
ax1.set_ylim(0, max(df['speedup'].max(), 1) + 0.5)

# Fractional location of y=1 on left axis
y1min, y1max = ax1.get_ylim()
frac = (1 - y1min) / (y1max - y1min)

# Right axis: whatever you're plotting (e.g., runtime)
ax2.plot(
    x,
    df['nnz_capacity_ratio'],
    color='tab:orange',
    marker='o',
    linewidth=2,
    label='NNZ Capacity Ratio'
)
ax2.set_ylabel("NNZ Capacity Ratio", color='tab:orange')

ax2.tick_params(axis='y', labelcolor='tab:orange')
rmin = min(df['nnz_capacity_ratio'].min(), 1)
rmax = max(df['nnz_capacity_ratio'].max(), 1)

# Compute new lower limit so that 1 sits at the same fraction
new_rmin = (frac * rmax - 1) / (frac - 1)

ax2.set_ylim(new_rmin, rmax)

# Shared x-axis
ax1.set_xlabel("Matrix")
ax1.set_xticks(x)
ax1.set_xticklabels(df['matrix_name'], rotation=45, ha='right')

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()

plt.tight_layout()
os.makedirs('../figures', exist_ok=True)
plt.savefig('../figures/suite_sparse_gpu_comparison.pdf', format='pdf')
plt.close()
