"""
2D visualization of engine fields.

Provides heatmaps and radial profiles for:
- f(x): update fraction (the primitive)
- τ(x): proper time
- ϕ(x): derived gravitational potential
- ρ(x): source density

All plots use matplotlib with sensible defaults for scientific visualization.
"""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.axes import Axes

if TYPE_CHECKING:
    from mplsim.core.lattice import Lattice
    from mplsim.core.source_map import SourceMap
    from mplsim.analysis.comparison import ComparisonResult


# Custom colormap: dark purple → blue → teal → warm white (blanc-cassé)
# Replaces viridis yellow with a softer cream/off-white endpoint
def _create_gravity_cmap():
    """Create a colormap from dark to warm white (blanc-cassé)."""
    from matplotlib.colors import LinearSegmentedColormap

    # Colors: dark purple → blue → teal → warm white
    colors = [
        (0.267, 0.004, 0.329),   # Dark purple (viridis start)
        (0.282, 0.141, 0.458),   # Purple
        (0.253, 0.265, 0.529),   # Blue-purple
        (0.192, 0.407, 0.556),   # Blue
        (0.127, 0.566, 0.550),   # Teal
        (0.206, 0.718, 0.472),   # Green-teal
        (0.565, 0.820, 0.376),   # Light green
        (0.993, 0.978, 0.925),   # Warm white / blanc-cassé
    ]
    return LinearSegmentedColormap.from_list("gravity", colors)


def _create_potential_cmap():
    """Create a colormap for potential wells: warm white (shallow) → deep red/black."""
    from matplotlib.colors import LinearSegmentedColormap

    # Colors: warm white → orange → red → dark (reversed for potential wells)
    colors = [
        (0.05, 0.02, 0.02),     # Near black (deep well)
        (0.329, 0.075, 0.098),  # Dark red
        (0.600, 0.157, 0.110),  # Red
        (0.855, 0.345, 0.114),  # Orange-red
        (0.969, 0.588, 0.275),  # Orange
        (0.988, 0.812, 0.498),  # Light orange
        (0.993, 0.978, 0.925),  # Warm white / blanc-cassé
    ]
    return LinearSegmentedColormap.from_list("potential", colors)


# Register custom colormaps
CMAP_GRAVITY = _create_gravity_cmap()
CMAP_POTENTIAL = _create_potential_cmap()

# Default colormaps
CMAP_F = CMAP_GRAVITY        # f field: low (dark) → high (warm white)
CMAP_PHI = CMAP_POTENTIAL    # ϕ field: deep (dark) → shallow (warm white)
CMAP_TAU = "plasma"          # proper time
CMAP_RHO = "YlOrRd"          # source density (yellow-orange-red)


def plot_field(
    field: np.ndarray,
    title: str = "",
    cmap=None,
    vmin: float | None = None,
    vmax: float | None = None,
    ax: Axes | None = None,
    colorbar: bool = True,
    figsize: tuple[float, float] = (8, 6),
) -> tuple[Figure, Axes]:
    """
    Plot a 2D field as a heatmap.

    Args:
        field: 2D array to plot
        title: Plot title
        cmap: Colormap name
        vmin, vmax: Color scale limits (auto if None)
        ax: Existing axes to plot on (creates new figure if None)
        colorbar: Whether to add a colorbar
        figsize: Figure size if creating new figure

    Returns:
        (fig, ax) tuple
    """
    if cmap is None:
        cmap = CMAP_GRAVITY

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(
        field,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )

    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    return fig, ax


def plot_f_field(
    lattice: "Lattice",
    title: str = "Update Fraction f(x)",
    ax: Axes | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    """Plot the f(x) field from a lattice."""
    return plot_field(
        lattice.f,
        title=title,
        cmap=CMAP_F,
        vmin=0,
        vmax=1,
        ax=ax,
        **kwargs,
    )


def plot_proper_time(
    lattice: "Lattice",
    title: str = "Proper Time τ(x)",
    ax: Axes | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    """Plot the proper time field from a lattice."""
    return plot_field(
        lattice.proper_time.astype(float),
        title=title,
        cmap=CMAP_TAU,
        ax=ax,
        **kwargs,
    )


def plot_source_map(
    source_map: "SourceMap",
    title: str = "Source Density ρ(x)",
    ax: Axes | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    """Plot a source map."""
    return plot_field(
        source_map.rates,
        title=title,
        cmap=CMAP_RHO,
        ax=ax,
        **kwargs,
    )


def plot_phi(
    phi: np.ndarray,
    title: str = "Gravitational Potential ϕ(x)",
    ax: Axes | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    """Plot a potential field."""
    return plot_field(
        phi,
        title=title,
        cmap=CMAP_PHI,
        ax=ax,
        **kwargs,
    )


def plot_f_and_phi(
    lattice: "Lattice",
    phi: np.ndarray,
    figsize: tuple[float, float] = (14, 5),
) -> Figure:
    """
    Plot f(x) and ϕ(x) side by side.

    Args:
        lattice: Lattice with f field
        phi: Derived potential field

    Returns:
        Figure with two subplots
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    plot_f_field(lattice, ax=axes[0])
    plot_phi(phi, ax=axes[1])

    fig.tight_layout()
    return fig


def plot_engine_summary(
    lattice: "Lattice",
    source_map: "SourceMap",
    phi: np.ndarray | None = None,
    figsize: tuple[float, float] = (16, 4),
) -> Figure:
    """
    Plot comprehensive summary: source, f, τ, and optionally ϕ.

    Args:
        lattice: Lattice with computed fields
        source_map: Source density
        phi: Optional derived potential

    Returns:
        Figure with 3-4 subplots
    """
    n_plots = 4 if phi is not None else 3
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    plot_source_map(source_map, ax=axes[0])
    plot_f_field(lattice, ax=axes[1])
    plot_proper_time(lattice, ax=axes[2])

    if phi is not None:
        plot_phi(phi, ax=axes[3])

    fig.tight_layout()
    return fig


def plot_radial_profile(
    radii: np.ndarray,
    values: np.ndarray,
    label: str = "",
    ax: Axes | None = None,
    figsize: tuple[float, float] = (8, 5),
    **plot_kwargs,
) -> tuple[Figure, Axes]:
    """
    Plot a 1D radial profile.

    Args:
        radii: Radius values
        values: Field values at each radius
        label: Line label
        ax: Existing axes (creates new if None)
        figsize: Figure size

    Returns:
        (fig, ax) tuple
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(radii, values, label=label, **plot_kwargs)
    ax.set_xlabel("Distance from center (r)")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)

    if label:
        ax.legend()

    return fig, ax


def plot_radial_profiles(
    profiles: list[tuple[np.ndarray, np.ndarray, str]],
    title: str = "Radial Profiles",
    figsize: tuple[float, float] = (8, 5),
) -> Figure:
    """
    Plot multiple radial profiles on the same axes.

    Args:
        profiles: List of (radii, values, label) tuples
        title: Plot title

    Returns:
        Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for radii, values, label in profiles:
        ax.plot(radii, values, label=label, linewidth=2)

    ax.set_xlabel("Distance from center (r)")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig


def plot_comparison(
    result: "ComparisonResult",
    title: str = "Engine vs Poisson Comparison",
    figsize: tuple[float, float] = (16, 5),
) -> Figure:
    """
    Plot ϕ_engine vs ϕ_poisson comparison.

    Args:
        result: ComparisonResult from compare_phi_engine_vs_poisson
        title: Overall title

    Returns:
        Figure with three subplots: engine, poisson, difference
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Engine ϕ
    im1 = axes[0].imshow(
        result.phi_engine_normalized,
        origin="lower",
        cmap=CMAP_PHI,
        vmin=0,
        vmax=1,
    )
    axes[0].set_title("ϕ_engine (from f)")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Poisson ϕ
    im2 = axes[1].imshow(
        result.phi_poisson_normalized,
        origin="lower",
        cmap=CMAP_PHI,
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("ϕ_poisson (theory)")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Difference
    diff = result.phi_engine_normalized - result.phi_poisson_normalized
    max_diff = max(abs(diff.min()), abs(diff.max()))
    im3 = axes[2].imshow(
        diff,
        origin="lower",
        cmap="RdBu_r",
        vmin=-max_diff,
        vmax=max_diff,
    )
    axes[2].set_title(f"Difference (corr={result.correlation:.3f})")
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_time_dilation_clocks(
    clock_data: list[dict],
    title: str = "Clock Tick Rates vs Distance",
    figsize: tuple[float, float] = (10, 6),
) -> Figure:
    """
    Plot clock tick rates as a function of distance from source.

    Args:
        clock_data: List of dicts with 'distance', 'tick_rate', 'f', 'label'
        title: Plot title

    Returns:
        Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    distances = [d["distance"] for d in clock_data]
    tick_rates = [d["tick_rate"] for d in clock_data]
    f_values = [d.get("f", None) for d in clock_data]

    # Plot tick rates
    ax.scatter(distances, tick_rates, s=100, c="blue", label="Clock tick rate", zorder=3)
    ax.plot(distances, tick_rates, "b--", alpha=0.5)

    # Plot f values if available
    if all(f is not None for f in f_values):
        ax.scatter(distances, f_values, s=100, c="red", marker="x", label="f(x)", zorder=3)
        ax.plot(distances, f_values, "r--", alpha=0.5)

    ax.set_xlabel("Distance from source")
    ax.set_ylabel("Rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add reference line at y=1
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5, label="_nolegend_")

    fig.tight_layout()
    return fig


def save_figure(fig: Figure, path: str | Path, dpi: int = 150, **kwargs) -> None:
    """Save figure to file."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", **kwargs)
