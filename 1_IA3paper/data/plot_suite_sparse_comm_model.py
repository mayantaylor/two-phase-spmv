import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from style import figure, style_axes, save



# Combine both dataframes
combined_df = pd.read_csv('ss-results/080626results_full-coprimes.csv')

# Group by matrix name and take the row with minimum max_time for each matrix
combined_df['matrix_name'] = combined_df['name'].astype(str).str.split('.').str[0]
combined_df = combined_df[combined_df['status'] != 'skipped']

# Extract the mapping (cyclic/blocked/random) from the trailing token of `name`
combined_df['mapping'] = combined_df['name'].astype(str).str.split('.').str[-1]

# `format` is blank for the (implicit) column-major format and 'row' for row-major;
# fill blanks so every row has an explicit label.
combined_df['format'] = combined_df['format'].fillna('coo')
combined_df.loc[combined_df['format'] == 'row', 'format'] = 'csr'

combined_df = combined_df.sort_values('max_time', ascending=True)
combined_df = combined_df.drop_duplicates(subset=['matrix_name'], keep='first')

# Calculate models
combined_df['reduce_model'] = combined_df['matrix_nrows'] / combined_df['nrows'] + 12 * (combined_df['nrows'] + 1) + np.floor((combined_df['ncols'] + 2) / 64) * 9
combined_df['bcast_model'] = combined_df['matrix_ncols'] / combined_df['ncols'] + (combined_df['nrows'] + 2) + np.floor((combined_df['nrows'] + 2) / 64) * 9

combined_df.sort_values('matrix_nrows', inplace=True)


# Create figure with subplots
x = np.arange(len(combined_df))
width = 0.33

fig, ax1 = figure("rect")
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

# Add nrows + format/mapping labels on top of the "Total time" bars
for bar, nrows, fmt, mapping in zip(bars, combined_df['nrows'], combined_df['format'], combined_df['mapping']):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(nrows)}\n{fmt}/{mapping}',
            ha='center', va='bottom', fontsize=4)

ax2.plot(
    x,
    combined_df['nnz_max_capacity'],
    color='black',
    marker='o',
    markersize=2,
    label='Max NNZ capacity',
)

ax2.set_ylabel("Max NNZ Count",)
ax2.tick_params(axis='y')
ax2.legend(loc='center right',)

ax1.legend(loc='upper left',)

ax1.set_ylabel("Runtime (ms)",)
ax1.tick_params(axis='y')

ax1.set_ylim(top=.03)
ax2.set_ylim(top=600)

# Shared x-axis
ax1.set_xticks(x)
ax1.set_xticklabels(combined_df['matrix_name'], rotation=45, ha='right',)
style_axes(ax1)
save(fig,'../figures/suite_sparse_comm_model.pdf')
