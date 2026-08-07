"""
Common matplotlib style for paper figures.
"""

import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Font sizes
# ----------------------------------------------------------------------

FONT = {
    "title": 11,
    "label": 6,
    "ticks": 5,
    "legend": 5,
    "annotation": 4,
}

# ----------------------------------------------------------------------
# Figure sizes (inches)
# ----------------------------------------------------------------------

FIGURES = {
    "square": (3.2, 3.2),          # IEEE single-column square
    "rect": (3.2, 2.3),            # IEEE single-column landscape
    "wide-single": (2.3, 1),
    "wide": (7.0, 3.0),            # Double-column
    "tall": (3.4, 4.5),
}

# ----------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------

plt.rcParams.update({
    "font.size": FONT["label"],
    "axes.labelsize": FONT["label"],
    "axes.titlesize": FONT["title"],
    "xtick.labelsize": FONT["ticks"],
    "ytick.labelsize": FONT["ticks"],
    "legend.fontsize": FONT["legend"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def figure(shape="rect"):
    """
    Create a figure using one of the predefined shapes.
    """
    return plt.subplots(figsize=FIGURES[shape])


def style_axes(ax):
    """Apply common axis formatting."""
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(direction="out")
    return ax

def save(fig, filename):
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
