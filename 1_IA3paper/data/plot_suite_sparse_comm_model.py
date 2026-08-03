import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

results_df = pd.read_csv('suite-sparse-results080326-coo.csv')

# Extract the last part of the matrix path (filename)
results_df['matrix_name'] = results_df['name'].astype(str).str.split('.').str[0]
results_df['reduce_model'] = results_df['matrix_nrows'] / results_df['nrows'] + 12 * (512 + 1) + np.floor((results_df['ncols'] + 2) / 64) * 9
results_df['bcast_model'] = results_df['matrix_ncols'] / results_df['ncols'] + (512 + 2) + np.floor((results_df['nrows'] + 2) / 64) * 9

results_df.sort_values('matrix_nrows', inplace=True)



results_df['nnz_max_capacity'] = pd.to_numeric(
    results_df['nnz_max_capacity'], errors='coerce'
)
# Create figure with subplots

x = np.arange(len(results_df))
width = 0.33
fs = 20

fig, ax1 = plt.subplots(figsize=(12, 8))
ax2 = ax1.twinx()


ax1.bar(
    x - width/2,
    (results_df['reduce_model'] + results_df['bcast_model']) / 1000000,
    width=width,
    color='tab:blue',
    bottom=(results_df['max_chain_time']) / 1000000,
    alpha=0.7,
    label='Comm model'
)

ax1.bar(
    x - width/2,
    results_df['max_chain_time'] / 1000000,
    width=width,
    color='tab:green',
    alpha=0.7,
    label='Measured compute'
)

ax1.bar(
    x + width/2,
    results_df['max_time'] / 1000000,
    width=width,
    color='tab:orange',
    alpha=0.7,
    label='Total time'
)

ax2.plot(
    x,
    results_df['nnz_max_capacity'],
    color='black',
    marker='o',
    linewidth=2,
    markersize=6,
    label='Max NNZ capacity',
)

ax2.set_ylabel("Max NNZ Count", fontsize=fs)
ax2.tick_params(axis='y', labelsize=16)
ax2.legend(loc='center right', fontsize=fs)

ax1.legend(loc='upper left', fontsize=fs)

ax1.set_ylabel("Runtime (ms)", fontsize=fs)
ax1.tick_params(axis='y', labelsize=16)

# Shared x-axis
ax1.set_xlabel("Matrix", fontsize=fs)
ax1.set_xticks(x)
ax1.set_xticklabels(results_df['matrix_name'], rotation=45, ha='right', fontsize=fs)

# Combine legends from both axes

plt.tight_layout()

plt.tight_layout()
os.makedirs('../figures', exist_ok=True)
plt.savefig('../figures/suite_sparse_comm_model.pdf', format='pdf')
plt.close()
