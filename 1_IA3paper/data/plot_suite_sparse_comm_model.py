import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Read the CSV file
if not os.path.exists('suite_sparse_gpu_eval.csv') or not os.path.exists('suite-sparse-results073026.csv'):
    print("Error: CSV files not found")
    exit()

results_df = pd.read_csv('suite-sparse-results073026.csv')

# Extract the last part of the matrix path (filename)
results_df['matrix_name'] = results_df['name'].astype(str).str.split('.').str[0] + ".npz"
results_df['reduce_model'] = results_df['matrix_nrows'] / results_df['nrows'] + 12 * (512 + 1) + np.floor((results_df['ncols'] + 2) / 64) * 9
results_df['bcast_model'] = results_df['matrix_ncols'] / results_df['ncols'] + (512 + 2) + np.floor((results_df['nrows'] + 2) / 64) * 9

results_df.sort_values('matrix_nrows', inplace=True)

# Create figure with subplots
fig, ax1 = plt.subplots(figsize=(24, 12))

x = np.arange(len(results_df))
width = 0.35


fig, ax1 = plt.subplots(figsize=(12, 5))

ax1.bar(
    x - width/3,
    (results_df['reduce_model'] + results_df['bcast_model']) / 1000000,
    width=width,
    color='tab:blue',
    alpha=0.7,
    label='Communication model'
)

ax1.bar(
    x,
    results_df['max_time'] / 1000000,
    width=width,
    color='tab:orange',
    alpha=0.7,
    label='Total time'
)

ax1.set_ylabel("WSE Performance", color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax1.set_yscale('log')
# Shared x-axis
ax1.set_xlabel("Matrix")
ax1.set_xticks(x)
ax1.set_xticklabels(results_df['matrix_name'], rotation=45, ha='right')

# Combine legends from both axes

plt.tight_layout()

plt.tight_layout()
os.makedirs('../figures', exist_ok=True)
plt.savefig('../figures/suite_sparse_comm_model.pdf', format='pdf')
plt.close()
