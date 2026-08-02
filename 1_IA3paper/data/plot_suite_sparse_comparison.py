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
df = df.merge(results_df[['matrix_name', 'max_time']], 
              on='matrix_name', how='left', suffixes=('', '_results'))

df['max_time'] = pd.to_numeric(df['max_time'], errors='coerce') / 1000000 # convert from cycles to ms

# Calculate throughput (nnz per ms)
df['throughput'] = df['nnz'] / df['gpu_mean_ms']

# Create figure with subplots
fig, ax1 = plt.subplots(figsize=(24, 12))

x = np.arange(len(df))
width = 0.35

df['speedup'] = df['gpu_mean_ms'] / df['max_time']

# Compute bar heights relative to 1
heights = df['speedup'] - 1

# Bars rooted at 1
ax1.bar(
    x,
    heights,
    width,
    bottom=1,
    label='WSE Speedup',
    color='blue',
    alpha=0.7
)

# Reference line at 1
ax1.axhline(y=1, color='red', linestyle='--', linewidth=1)

ax1.set_xlabel('Matrix')
ax1.set_ylabel('WSE Speedup')
ax1.set_title('GPU Runtimes Comparison (sorted by rows)')
ax1.set_xticks(x)
ax1.set_xticklabels(df['matrix_name'], rotation=45, ha='right')
ax1.legend()

plt.tight_layout()
os.makedirs('../figures', exist_ok=True)
plt.savefig('../figures/suite_sparse_gpu_comparison.pdf', format='pdf')
plt.close()
