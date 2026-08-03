import pandas as pd
import matplotlib.pyplot as plt


def main():
    csv_file = "random-scale-npes.csv"
    output = "../figures/random-scale-npes.pdf"

    df = pd.read_csv(csv_file, skipinitialspace=True)

    df["matrix_nrows"] = pd.to_numeric(df["matrix_nrows"], errors="coerce")
    df["nrows"] = pd.to_numeric(df["nrows"], errors="coerce")
    df["max_time"] = pd.to_numeric(df["max_time"], errors="coerce")
    df["max_chain_time"] = pd.to_numeric(df["max_chain_time"], errors="coerce")
    
    df["B"] = df["matrix_nrows"] / df['nrows']
    df["comm_model"] = df["B"] + 12 * (df['nrows'] + 1) + 9/64 * df['nrows'] + df["B"] + 2 * df['nrows'] + 10 + 9/128 * df['nrows']

    categories = sorted(df["matrix_nrows"].dropna().unique())
    if len(categories) == 0:
        raise ValueError("No matrix_nrows categories found in the CSV.")

    plt.figure(figsize=(10, 8))
    colormap = plt.colormaps["tab10"]

    for idx, matrix_nrows in enumerate(categories):
        sub = df[df["matrix_nrows"] == matrix_nrows].copy()
        if sub.empty:
            continue

        sub = sub.sort_values(by="nrows")
        color = colormap(idx % 10)

        plt.plot(
            sub["nrows"],
            sub["max_time"],
            linestyle="-",
            color=color,
            label=f"{int(matrix_nrows)} max_time"
        )

        plt.plot(
            sub["nrows"],
            sub["max_chain_time"],
            linestyle="--",
            color=color,
            label=f"{int(matrix_nrows)} max_chain_time"
        )
        
        plt.plot(
                    sub["nrows"],
                    sub["comm_model"],
                    marker="*",
                    markersize=13,
                    linestyle="-.",
                    color=color,
                    label=f"{int(matrix_nrows)} comm model"
                )

    plt.xlabel("nrows")
    plt.ylabel("cycles")
    
    plt.yscale('log')
    plt.xscale('log')
    plt.title("rand-4types: max_time and max_chain_time vs nrows")
    plt.xticks(sorted(df["nrows"].dropna().unique()))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches="tight", format="pdf")


if __name__ == "__main__":
    main()
