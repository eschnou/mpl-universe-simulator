"""
Trajectory visualization for test particles.

Plots particle paths over f(x) or ϕ(x) background fields,
showing how particles fall toward mass concentrations.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from mplsim.viz.fields import CMAP_GRAVITY

if TYPE_CHECKING:
    from mplsim.patterns.particle import GeodesicParticle


def plot_trajectory(
    particle: "GeodesicParticle",
    background: np.ndarray | None = None,
    title: str = "Particle Trajectory",
    cmap=None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (8, 8),
    show_start: bool = True,
    show_end: bool = True,
    line_color: str = "white",
    line_width: float = 2.0,
) -> tuple[Figure, Axes]:
    """
    Plot a single particle trajectory.

    Args:
        particle: GeodesicParticle with recorded trajectory
        background: Optional 2D field to show as background (e.g., f or ϕ)
        title: Plot title
        cmap: Colormap for background
        ax: Existing axes (creates new if None)
        show_start: Mark starting position
        show_end: Mark ending position
        line_color: Trajectory line color
        line_width: Trajectory line width

    Returns:
        (fig, ax) tuple
    """
    if cmap is None:
        cmap = CMAP_GRAVITY

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Plot background field
    if background is not None:
        im = ax.imshow(
            background,
            origin="lower",
            cmap=cmap,
            aspect="equal",
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Get trajectory
    x_traj, y_traj = particle.get_trajectory_arrays()

    if len(x_traj) > 0:
        # Plot trajectory line
        ax.plot(x_traj, y_traj, color=line_color, linewidth=line_width, zorder=2)

        # Mark start and end
        if show_start:
            ax.scatter(
                [x_traj[0]], [y_traj[0]],
                color="green", s=100, marker="o", zorder=3,
                label="Start", edgecolors="white", linewidths=1.5
            )
        if show_end:
            ax.scatter(
                [x_traj[-1]], [y_traj[-1]],
                color="red", s=100, marker="x", zorder=3,
                label="End", linewidths=2
            )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if show_start or show_end:
        ax.legend(loc="upper right")

    return fig, ax


def plot_trajectories(
    particles: Sequence["GeodesicParticle"],
    background: np.ndarray | None = None,
    title: str = "Particle Trajectories",
    cmap=None,
    figsize: tuple[float, float] = (10, 10),
    colors: Sequence[str] | None = None,
    show_start: bool = True,
    show_end: bool = True,
    source_position: tuple[int, int] | None = None,
) -> Figure:
    """
    Plot multiple particle trajectories on the same background.

    Args:
        particles: Sequence of GeodesicParticle objects
        background: Optional 2D field for background
        title: Plot title
        cmap: Colormap for background
        colors: Optional list of colors for each trajectory
        show_start: Mark starting positions
        show_end: Mark ending positions
        source_position: Optional (x, y) to mark source location

    Returns:
        Figure
    """
    if cmap is None:
        cmap = CMAP_GRAVITY

    fig, ax = plt.subplots(figsize=figsize)

    # Plot background
    if background is not None:
        im = ax.imshow(
            background,
            origin="lower",
            cmap=cmap,
            aspect="equal",
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="f(x)")

    # Default colors
    if colors is None:
        cmap_lines = plt.cm.get_cmap("tab10")
        colors = [cmap_lines(i % 10) for i in range(len(particles))]

    # Plot each trajectory
    for i, particle in enumerate(particles):
        x_traj, y_traj = particle.get_trajectory_arrays()
        color = colors[i] if i < len(colors) else "white"

        if len(x_traj) > 0:
            ax.plot(
                x_traj, y_traj,
                color=color, linewidth=2, zorder=2,
                label=particle.config.pattern_id
            )

            if show_start:
                ax.scatter(
                    [x_traj[0]], [y_traj[0]],
                    color=color, s=80, marker="o", zorder=3,
                    edgecolors="white", linewidths=1
                )
            if show_end:
                ax.scatter(
                    [x_traj[-1]], [y_traj[-1]],
                    color=color, s=80, marker="s", zorder=3,
                    edgecolors="white", linewidths=1
                )

    # Mark source if specified
    if source_position is not None:
        ax.scatter(
            [source_position[0]], [source_position[1]],
            color="yellow", s=200, marker="*", zorder=4,
            edgecolors="black", linewidths=1.5, label="Source"
        )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    return fig


def plot_free_fall_analysis(
    particles: Sequence["GeodesicParticle"],
    source_position: tuple[int, int],
    title: str = "Free Fall Analysis",
    figsize: tuple[float, float] = (12, 5),
) -> Figure:
    """
    Analyze free fall: distance to source over time.

    Args:
        particles: Particles that have been simulated
        source_position: (x, y) of the source/mass
        title: Plot title

    Returns:
        Figure with distance-vs-time plot
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    sx, sy = source_position

    # Left plot: distance to source over time
    ax1 = axes[0]
    for particle in particles:
        x_traj, y_traj = particle.get_trajectory_arrays()
        if len(x_traj) == 0:
            continue

        distances = np.sqrt((x_traj - sx)**2 + (y_traj - sy)**2)
        times = np.arange(len(distances))

        ax1.plot(times, distances, label=particle.config.pattern_id, linewidth=2)

    ax1.set_xlabel("Time (ticks)")
    ax1.set_ylabel("Distance to source")
    ax1.set_title("Distance vs Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right plot: speed over time
    ax2 = axes[1]
    for particle in particles:
        traj = np.array(particle.trajectory)
        if len(traj) < 2:
            continue

        vx, vy = traj[:, 2], traj[:, 3]
        speed = np.sqrt(vx**2 + vy**2)
        times = np.arange(len(speed))

        ax2.plot(times, speed, label=particle.config.pattern_id, linewidth=2)

    ax2.set_xlabel("Time (ticks)")
    ax2.set_ylabel("Speed")
    ax2.set_title("Speed vs Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    return fig
