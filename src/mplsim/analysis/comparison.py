"""
Compare engine-derived ϕ with Poisson-predicted ϕ.

This validates that our bandwidth-limited model reproduces Newtonian gravity:
- ϕ_engine: derived from f(x) which emerges from congestion
- ϕ_poisson: theoretical prediction from ∇²ϕ = ρ

If they correlate well, the engine is reproducing gravity correctly.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mplsim.core.lattice import Lattice
    from mplsim.core.source_map import SourceMap

from mplsim.analysis.phi_field import PhiFromF
from mplsim.analysis.poisson import PoissonSolver


@dataclass
class ComparisonResult:
    """Results of comparing ϕ_engine vs ϕ_poisson."""

    phi_engine: np.ndarray
    phi_poisson: np.ndarray
    correlation: float
    rmse: float
    max_error: float

    # Normalized versions for visual comparison
    phi_engine_normalized: np.ndarray
    phi_poisson_normalized: np.ndarray


def normalize_field(field: np.ndarray) -> np.ndarray:
    """Normalize field to [0, 1] range for comparison."""
    vmin, vmax = field.min(), field.max()
    if vmax - vmin < 1e-10:
        return np.zeros_like(field)
    return (field - vmin) / (vmax - vmin)


def compare_phi_engine_vs_poisson(
    lattice: "Lattice",
    source_map: "SourceMap",
    phi_mode: str = "linear",
    phi_alpha: float = 1.0,
    boundary: str = "periodic",
) -> ComparisonResult:
    """
    Compare engine-derived ϕ with Poisson-predicted ϕ.

    Args:
        lattice: Lattice with computed f field
        source_map: Source map used to drive the engine
        phi_mode: "linear" or "log" for PhiFromF
        phi_alpha: Scaling factor for PhiFromF
        boundary: Boundary condition for Poisson solver

    Returns:
        ComparisonResult with both fields and metrics
    """
    # Compute ϕ_engine from f
    phi_computer = PhiFromF(mode=phi_mode, alpha=phi_alpha)
    phi_engine = phi_computer.compute(lattice.f)

    # Compute ϕ_poisson from source rates
    # Note: we use source_map.rates as the mass density ρ
    poisson = PoissonSolver(boundary=boundary)
    phi_poisson = poisson.solve(source_map.rates)

    # Normalize both fields for fair comparison
    phi_engine_norm = normalize_field(phi_engine)
    phi_poisson_norm = normalize_field(phi_poisson)

    # Compute correlation
    corr_matrix = np.corrcoef(phi_engine_norm.flatten(), phi_poisson_norm.flatten())
    correlation = corr_matrix[0, 1]

    # Compute RMSE on normalized fields
    rmse = np.sqrt(np.mean((phi_engine_norm - phi_poisson_norm) ** 2))

    # Max error
    max_error = np.max(np.abs(phi_engine_norm - phi_poisson_norm))

    return ComparisonResult(
        phi_engine=phi_engine,
        phi_poisson=phi_poisson,
        correlation=correlation,
        rmse=rmse,
        max_error=max_error,
        phi_engine_normalized=phi_engine_norm,
        phi_poisson_normalized=phi_poisson_norm,
    )


def compute_radial_profile(
    field: np.ndarray,
    center: tuple[int, int],
    max_radius: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute radial average of a field around a center point.

    Args:
        field: 2D field, shape [ny, nx]
        center: (cx, cy) center point
        max_radius: Maximum radius to compute (default: to edge)

    Returns:
        (radii, values) - 1D arrays of radius and averaged field values
    """
    ny, nx = field.shape
    cx, cy = center

    if max_radius is None:
        max_radius = min(cx, cy, nx - cx - 1, ny - cy - 1)

    # Create distance array
    yy, xx = np.ogrid[:ny, :nx]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    # Bin by integer radius
    radii = np.arange(0, max_radius + 1)
    values = np.zeros(len(radii))

    for i, r in enumerate(radii):
        if r == 0:
            mask = dist < 0.5
        else:
            mask = (dist >= r - 0.5) & (dist < r + 0.5)

        if mask.sum() > 0:
            values[i] = field[mask].mean()

    return radii, values
