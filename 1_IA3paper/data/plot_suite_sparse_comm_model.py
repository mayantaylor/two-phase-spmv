import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

results_df = pd.read_csv('ss-results/080326coo-512cyclic.csv')
new_results_df = pd.read_csv('ss-results/080526coo-256_384.csv')
coprime_df = pd.read_csv('ss-results/080526coo-coprimes512_256.csv')

# Combine both dataframes
combined_df = pd.concat([results_df, new_results_df, coprime_df], ignore_index=True)

# Group by matrix name and take the row with minimum max_time for each matrix
combined_df['matrix_name'] = combined_df['name'].astype(str).str.split('.').str[0]
combined_df = combined_df.sort_values('max_time')
combined_df = combined_df.drop_duplicates(subset=['matrix_name'], keep='first')

# Calculate models
combined_df['reduce_model'] = combined_df['matrix_nrows'] / combined_df['nrows'] + 12 * (combined_df['nrows'] + 1) + np.floor((combined_df['ncols'] + 2) / 64) * 9
combined_df['bcast_model'] = combined_df['matrix_ncols'] / combined_df['ncols'] + (combined_df['nrows'] + 2) + np.floor((combined_df['nrows'] + 2) / 64) * 9

combined_df.sort_values('matrix_nrows', inplace=True)

combined_df['nnz_max_capacity'] = pd.to_numeric(
    combined_df['nnz_max_capacity'], errors='coerce'
)

# Create figure with subplots
x = np.arange(len(combined_df))
width = 0.33
fs = 20

fig, ax1 = plt.subplots(figsize=(12, 8))
ax2 = ax1.twinx()

ax1.bar(
    x - width/2,
    (combined_df['reduce_model'] + combined_df['bcast_model']) / 1000000,
    width=width,
    color='tab:blue',
    bottom=(combined_df['max_chain_time']) / 1000000,
    alpha=0.7,
    label='Comm model'
)

ax1.bar(
    x - width/2,
    combined_df['max_chain_time'] / 1000000,
    width=width,
    color='tab:green',
    alpha=0.7,
    label='Measured compute'
)

bars = ax1.bar(
    x + width/2,
    combined_df['max_time'] / 1000000,
    width=width,
    color='tab:orange',
    alpha=0.7,
    label='Total time'
)

# Add nrows labels on top of bars
for i, (bar, nrows) in enumerate(zip(bars, combined_df['nrows'])):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(nrows)}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax2.plot(
    x,
    combined_df['nnz_max_capacity'],
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
ax1.set_xticklabels(combined_df['matrix_name'], rotation=45, ha='right', fontsize=fs)

plt.tight_layout()

os.makedirs('../figures', exist_ok=True)
plt.savefig('../figures/suite_sparse_comm_model.pdf', format='pdf')
plt.close()
