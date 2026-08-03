import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


results_df = pd.read_csv('random-vary-nnz.csv')
df_coo = pd.read_csv('rand-scale-nnz-results080326-coo.csv')

# Extract the last part of the matrix path (filename)
results_df['matrix_name'] = results_df['name'].astype(str).str.split('.').str[0]
results_df['reduce_model'] = results_df['matrix_nrows'] + 12 * (512 + 1) + np.floor((results_df['ncols'] + 2) / 64) * 9
results_df['bcast_model'] = results_df['matrix_ncols'] / results_df['ncols'] + (512 + 2) + np.floor((results_df['nrows'] + 2) / 64) * 9

results_df.sort_values('matrix_nrows', inplace=True)

# Extract the last part of the matrix path (filename)
df_coo['matrix_name'] = df_coo['name'].astype(str).str.split('.').str[0]
df_coo['reduce_model'] = df_coo['matrix_nrows'] / df_coo['nrows'] + 12 * (512 + 1) + np.floor((df_coo['ncols'] + 2) / 64) * 9
df_coo['bcast_model'] = df_coo['matrix_ncols'] / df_coo['ncols'] + (512 + 2) + np.floor((df_coo['nrows'] + 2) / 64) * 9

df_coo.sort_values('matrix_nrows', inplace=True)


fs = 20
fig, ax1 = plt.subplots(figsize=(8, 5))


# Evaluate the model at the matrix sizes
comm_model = results_df['reduce_model'] + results_df['bcast_model']

# Plot over the categorical bar positions


ax1.plot(
    results_df['nnz_max_capacity'],
    results_df['max_time'],
    label='csr - total',
    marker='o',
    color='orange',
    linestyle='-',
)
ax1.plot(
    results_df['nnz_max_capacity'],
    results_df['max_chain_time'],
    label='csr - compute',
    linestyle='--',
    color='orange',

)


ax1.plot(
    df_coo['nnz_max_capacity'],
    df_coo['max_time'],
    label='coo - total',
    color='blue',
    marker='o',
    linestyle='-',
)

ax1.plot(
    df_coo['nnz_max_capacity'],
    df_coo['max_chain_time'],
    label='coo - compute',
    color='blue',
    linestyle='--',
)



ax1.legend(loc='upper left', fontsize=16)
ax1.set_ylim(0, 1.1 * results_df['max_time'].max())
ax1.set_ylabel("Runtime (cycles)", fontsize=fs)
ax1.tick_params(axis='y', labelsize=16)
ax1.tick_params(axis='x', labelsize=16)

# Shared x-axis
ax1.set_xlabel("Maz NNZ", fontsize=fs)

# Combine legends from both axes


plt.tight_layout()
plt.savefig('../figures/compare-coo-scalennz.pdf', format='pdf')
plt.close()
