"""
Analysis layer: derived quantities for visualization and comparison.

IMPORTANT: This is NOT seen by the engine. One-way derivation only.

- PhiFromF: compute ϕ_engine from measured f(x)
- PoissonSolver: compute ϕ_Poisson from analyst-specified ρ (for comparison)
- compare_phi_engine_vs_poisson: validate engine against theory
- verify_poisson_emergence: verify that Poisson equation emerges from dynamics
"""

from mplsim.analysis.phi_field import PhiFromF, compute_phi_linear, compute_phi_log
from mplsim.analysis.poisson import PoissonSolver, solve_poisson
from mplsim.analysis.comparison import (
    ComparisonResult,
    compare_phi_engine_vs_poisson,
    compute_radial_profile,
)
from mplsim.analysis.emergence import (
    EmergenceResult,
    verify_poisson_emergence,
    compute_graph_laplacian_2d,
    fit_kappa,
    compute_screening_length,
    compute_screening_mass_squared,
    ScreenedEmergenceResult,
    verify_screened_poisson_emergence,
    solve_theoretical_lambda,
)

__all__ = [
    "PhiFromF",
    "compute_phi_linear",
    "compute_phi_log",
    "PoissonSolver",
    "solve_poisson",
    "ComparisonResult",
    "compare_phi_engine_vs_poisson",
    "compute_radial_profile",
    # Emergence validation
    "EmergenceResult",
    "verify_poisson_emergence",
    "compute_graph_laplacian_2d",
    "fit_kappa",
    "compute_screening_length",
    # Screened Poisson (full theory)
    "compute_screening_mass_squared",
    "ScreenedEmergenceResult",
    "verify_screened_poisson_emergence",
    "solve_theoretical_lambda",
]
