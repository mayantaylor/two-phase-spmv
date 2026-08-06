import pandas as pd

df = pd.read_csv("ss-results/matrix-stats.csv")

# Remove empty columns from CSV export
df = df.dropna(axis=1, how='all')

# Remove ncols column
df = df.drop(columns=["ncols", "id"])

df['nnz/nrows'] = df['nnz'] / df['nrows']

# Sort by number of rows (ascending)
df = df.sort_values(by="nrows", ascending=True)

latex_tabular = df.to_latex(
    index=False,
    escape=True
)

latex = f"""
\\begin{{table*}}[t]
\\centering
\\caption{{Suite Sparse matrices used in evaluation}}
\\label{{tab:suite_sparse_matrices}}
{latex_tabular}
\\end{{table*}}
"""

with open("../figures/suite_sparse_matrices.tex", "w") as f:
    f.write(latex)
