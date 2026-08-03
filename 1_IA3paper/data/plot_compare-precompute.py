import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


results_df = pd.read_csv('random-vary-matrix-nrows.csv')
df_precompute = pd.read_csv('rand-scale-rows-results080326-precompute.csv')

# Extract the last part of the matrix path (filename)
results_df['matrix_name'] = results_df['name'].astype(str).str.split('.').str[0]
results_df['reduce_model'] = results_df['matrix_nrows'] / results_df['nrows'] + 12 * (512 + 1) + np.floor((results_df['ncols'] + 2) / 64) * 9
results_df['bcast_model'] = results_df['matrix_ncols'] / results_df['ncols'] + (512 + 2) + np.floor((results_df['nrows'] + 2) / 64) * 9

results_df.sort_values('matrix_nrows', inplace=True)

# Extract the last part of the matrix path (filename)
df_precompute['matrix_name'] = df_precompute['name'].astype(str).str.split('.').str[0]
df_precompute['reduce_model'] = df_precompute['matrix_nrows'] / df_precompute['nrows'] + 12 * (512 + 1) + np.floor((df_precompute['ncols'] + 2) / 64) * 9
df_precompute['bcast_model'] = df_precompute['matrix_ncols'] / df_precompute['ncols'] + (512 + 2) + np.floor((df_precompute['nrows'] + 2) / 64) * 9

df_precompute.sort_values('matrix_nrows', inplace=True)


fs = 20
fig, ax1 = plt.subplots(figsize=(8, 5))


# Evaluate the model at the matrix sizes
comm_model = results_df['reduce_model'] + results_df['bcast_model']

# Plot over the categorical bar positions

ax1.plot(
    results_df['matrix_nrows'] / results_df['nrows'],
    results_df['max_chain_time'],
    label='Measured compute',
    marker='o',
    linestyle='--',
    color='orange',

)

ax1.plot(
    results_df['matrix_nrows'] / results_df['nrows'],
    results_df['max_time'],
    label='Total time',
    marker='o',
    color='orange',
    linestyle='-',
)


ax1.plot(
    df_precompute['matrix_nrows'] / df_precompute['nrows'],
    df_precompute['max_chain_time'],
    label='Precompute - measured compute',
    marker='o',
    color='blue',
    linestyle='--',
)

ax1.plot(
    df_precompute['matrix_nrows'] / df_precompute['nrows'],
    df_precompute['max_time'],
    label='Precompute Total time',
    color='blue',
    marker='o',
    linestyle='-',
)

# Fit using the actual matrix sizes
x_fit = results_df['matrix_nrows'].to_numpy()
y_fit = results_df['max_chain_time'].to_numpy() 

slope, intercept = np.polyfit(x_fit, y_fit, 1)


# Evaluate the model at the matrix sizes
y_line = slope * x_fit + intercept


ax1.legend(loc='upper left', fontsize=fs)
ax1.set_ylim(0, 1.1 * max(results_df['max_time'].max(), comm_model.max()))
ax1.set_ylabel("Runtime (cycles)", fontsize=fs)
ax1.tick_params(axis='y', labelsize=16)
ax1.tick_params(axis='x', labelsize=16)

# Shared x-axis
ax1.set_xlabel("Matrix Rows per PE row", fontsize=fs)

# Combine legends from both axes


plt.tight_layout()
plt.savefig('../figures/compare-precompute.pdf', format='pdf')
plt.close()
