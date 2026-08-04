import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('suite_sparse_gpu_eval.csv')
df_coo = pd.read_csv('suite_sparse_gpu_eval-coo.csv')

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
fig, ax1 = plt.subplots(figsize=(24, 12))

# First subplot: runtime sorted by rows



# Extract the last part of the matrix path (filename)
df_coo['matrix_name'] = df_coo['matrix_path'].str.split('/').str[-1]

df_coo = df_coo.groupby('matrix_name').agg({
    'gpu_mean_ms': 'mean',
    'rows': 'first',
    'nnz': 'first',
    'cols': 'first'  # Add cols if not already present
}).reset_index()

df_coo = df_coo.sort_values('rows')

# Create figure with six subplots (3 columns × 2 rows)

# First subplot: runtime sorted by rows
ax1.bar(df_coo['matrix_name'], df['gpu_mean_ms'] /  df_coo['gpu_mean_ms'] - 1, bottom=1, label="coo")


ax1.set_xlabel('Matrix')
ax1.set_ylabel('COO Speedup')
ax1.set_xticks(df_coo['matrix_name'])
ax1.set_xticklabels(df_coo['matrix_name'], rotation=45, ha='right')

ax1.legend()

plt.tight_layout()
plt.savefig('../figures/suite_sparse_gpu_eval.pdf', format='pdf')
