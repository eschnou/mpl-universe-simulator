"""
Visualization utilities.

- Field heatmaps (f, ϕ_engine, ρ)
- Trajectory plots
- Radial profiles
- Comparison plots
"""

from mplsim.viz.fields import (
    plot_field,
    plot_f_field,
    plot_proper_time,
    plot_source_map,
    plot_phi,
    plot_f_and_phi,
    plot_engine_summary,
    plot_radial_profile,
    plot_radial_profiles,
    plot_comparison,
    plot_time_dilation_clocks,
    save_figure,
)

from mplsim.viz.trajectories import (
    plot_trajectory,
    plot_trajectories,
    plot_free_fall_analysis,
)

__all__ = [
    "plot_field",
    "plot_f_field",
    "plot_proper_time",
    "plot_source_map",
    "plot_phi",
    "plot_f_and_phi",
    "plot_engine_summary",
    "plot_radial_profile",
    "plot_radial_profiles",
    "plot_comparison",
    "plot_time_dilation_clocks",
    "save_figure",
    "plot_trajectory",
    "plot_trajectories",
    "plot_free_fall_analysis",
]
