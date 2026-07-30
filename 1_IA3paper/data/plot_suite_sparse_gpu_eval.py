import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('suite_sparse_gpu_eval.csv')

# Extract the last part of the matrix path (filename)
df['matrix_name'] = df['matrix_path'].str.split('/').str[-1]

df = df.groupby('matrix_name').agg({
    'gpu_mean_ms': 'mean',
    'rows': 'first',
    'nnz': 'first',
    'cols': 'first'  # Add cols if not already present
}).reset_index()

df = df.sort_values('rows')

# Calculate throughput (nnz per ms)
df['throughput'] = df['nnz'] / df['gpu_mean_ms']

# Create figure with six subplots (3 columns × 2 rows)
fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(24, 12))

# First subplot: runtime sorted by rows
ax1.bar(df['matrix_name'], df['gpu_mean_ms'])
ax1.set_xlabel('Matrix')
ax1.set_ylabel('GPU Runtime (ms)')
ax1.set_title('GPU Runtimes for Each Matrix (sorted by rows)')
ax1.set_yscale('log')
ax1.set_xticks(df['matrix_name'])
ax1.set_xticklabels(df['matrix_name'], rotation=45, ha='right')

# Second subplot: runtime sorted by nnz/rows
df_sorted_density = df.copy()
df_sorted_density['nnz_per_row'] = df_sorted_density['nnz'] / df_sorted_density['rows']
df_sorted_density = df_sorted_density.sort_values('nnz_per_row')
ax2.bar(df_sorted_density['matrix_name'], df_sorted_density['gpu_mean_ms'])
ax2.set_xlabel('Matrix')
ax2.set_ylabel('GPU Runtime (ms)')
ax2.set_title('GPU Runtimes for Each Matrix (sorted by nnz/rows)')
ax2.set_yscale('log')
ax2.set_xticks(df_sorted_density['matrix_name'])
ax2.set_xticklabels(df_sorted_density['matrix_name'], rotation=45, ha='right')

# Third subplot: runtime sorted by density
df_sorted_sparsity = df.copy()
df_sorted_sparsity['density'] = df_sorted_sparsity['nnz'] / (df_sorted_sparsity['rows'] * df_sorted_sparsity['cols'])
df_sorted_sparsity = df_sorted_sparsity.sort_values('density')
ax3.bar(df_sorted_sparsity['matrix_name'], df_sorted_sparsity['gpu_mean_ms'])
ax3.set_xlabel('Matrix')
ax3.set_ylabel('GPU Runtime (ms)')
ax3.set_title('GPU Runtimes for Each Matrix (sorted by density)')
ax3.set_yscale('log')
ax3.set_xticks(df_sorted_sparsity['matrix_name'])
ax3.set_xticklabels(df_sorted_sparsity['matrix_name'], rotation=45, ha='right')


plt.tight_layout()
plt.savefig('../figures/suite_sparse_gpu_eval.pdf', format='pdf')
