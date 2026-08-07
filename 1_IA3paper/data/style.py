"""
Common matplotlib style for paper figures.

Usage
-----
import plot_style as style

fig, ax = style.figure("rect")
...
style.style_axes(ax)
style.save(fig, "figure.pdf")
"""

import matplotlib.pyplot as plt
from cycler import cycler

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
    "wide-single": (2.3, 1.0),
    "wide": (7.0, 3.0),            # IEEE double-column
    "tall": (3.4, 4.5),
}

# ----------------------------------------------------------------------
# Global matplotlib style
# ----------------------------------------------------------------------

plt.rcParams.update({

    # ------------------------------------------------------------
    # Fonts
    # ------------------------------------------------------------
    "font.family": "serif",
    # Uncomment these if you want LaTeX-rendered text.
    # Requires a working LaTeX installation.
    #
    # "text.usetex": True,
    # "text.latex.preamble": r"""
    # \usepackage{amsmath}
    # \usepackage{bm}
    # """,

    # ------------------------------------------------------------
    # Font sizes
    # ------------------------------------------------------------
    "font.size": FONT["label"],
    "axes.labelsize": FONT["label"],
    "axes.titlesize": FONT["title"],
    "xtick.labelsize": FONT["ticks"],
    "ytick.labelsize": FONT["ticks"],
    "legend.fontsize": FONT["legend"],

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------
    "figure.constrained_layout.use": True,

    # ------------------------------------------------------------
    # Lines
    # ------------------------------------------------------------
    "lines.linewidth": 1,
    "lines.markersize": 4,
    "axes.linewidth": 0.6,

    # ------------------------------------------------------------
    # Tick appearance
    # ------------------------------------------------------------
    "xtick.direction": "in",
    "ytick.direction": "in",

    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,

    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,

    # ------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------
    "grid.color": "0.7",
    "grid.linewidth": 0.1,

    # ------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------
    "legend.frameon": False,
    

    # ------------------------------------------------------------
    # Color cycle
    # ------------------------------------------------------------
    "axes.prop_cycle": cycler(
        "color",
        [
            "tab:blue",
            "tab:red",
            "tab:green",
            "tab:orange",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
            "tab:olive",
            "tab:cyan",
        ],
    ),

    # ------------------------------------------------------------
    # PDF/EPS output
    # ------------------------------------------------------------
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "savefig.bbox": "tight",
    "savefig.pad_inches": 1 / 72,
})


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def figure(shape="rect"):
    """
    Create a figure using one of the predefined figure sizes.

    Parameters
    ----------
    shape : str
        One of:
            square
            rect
            wide-single
            wide
            tall
    """
    return plt.subplots(figsize=FIGURES[shape])


def style_axes(ax, grid="y"):
    """
    Apply consistent styling to an axis.

    Parameters
    ----------
    grid : {"x", "y", "both", None}
        Which gridlines to show.
    """

    if grid == "both":
        ax.grid(True)
    elif grid in ("x", "y"):
        ax.grid(axis=grid)

    ax.minorticks_on()

    return ax


def save(fig, filename, dpi=300):
    """
    Save and close a figure.
    """
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)
