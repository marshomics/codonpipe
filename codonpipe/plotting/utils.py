"""Shared plotting utilities for CodonPipe.

Provides constants and helpers used by all plotting modules so that DPI,
output formats, figure saving, and style application are defined once.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger("codonpipe")

# Publication output defaults
DPI = 300
FORMATS = ["png", "svg"]

# Shared publication style parameters
STYLE_PARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    # SVG settings: embed fonts so glyphs render correctly in Illustrator
    # without requiring the font to be installed on the editing machine.
    "svg.fonttype": "none",
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
}


def apply_style() -> None:
    """Apply publication-quality matplotlib style."""
    plt.rcParams.update(STYLE_PARAMS)
    sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.alpha": 0.3})


def save_fig(
    fig: plt.Figure,
    path: Path,
    formats: list[str] | None = None,
    dpi: int = DPI,
) -> None:
    """Save a figure in multiple formats and close it.

    Args:
        fig: Matplotlib figure.
        path: Base output path (extension is replaced per format).
        formats: List of file formats (default: PNG + SVG).
        dpi: Resolution for raster formats.
    """
    formats = formats or FORMATS
    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        fig.savefig(out, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white")
        logger.debug("Saved figure: %s", out)
    plt.close(fig)
