"""
Emergence validation tools for verifying Poisson equation emergence.

The key claim of the paper is that the Poisson equation emerges from
bandwidth + synchronization dynamics:

    (Lλ)(x) ≈ κ · ρ_act(x)

where:
- L is the discrete graph Laplacian
- λ(x) = 1 - f(x) is the slowness (time-sag)
- ρ_act(x) is the activity density (source rates)
- κ is an effective coupling constant

These tools verify this emergence by:
1. Computing the graph Laplacian of the λ field
2. Comparing (Lλ) with ρ_act
3. Measuring correlation and fitting κ
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from mplsim.core.lattice import Lattice
    from mplsim.core.source_map import SourceMap


@dataclass
class EmergenceResult:
    """Results of Poisson emergence verification."""

    # Fields
    lambda_field: np.ndarray      # Slowness λ = 1 - f
    laplacian_lambda: np.ndarray  # (Lλ)(x) = Laplacian of slowness
    rho_activity: np.ndarray      # Activity density ρ_act

    # Metrics
    correlation: float        # Correlation between (Lλ) and ρ_act
    r_squared: float          # R² of linear fit
    fitted_kappa: float       # Best-fit κ from (Lλ) ≈ κρ
    rmse: float               # Root mean square error of fit

    # For plotting
    laplacian_flat: np.ndarray  # Flattened for scatter plots
    rho_flat: np.ndarray        # Flattened for scatter plots


def compute_graph_laplacian_2d(field: np.ndarray) -> np.ndarray:
    """
    Compute the discrete graph Laplacian of a 2D field.

    Uses the standard graph Laplacian L = D - A on a periodic grid:
    (Lf)(x) = 4*f_center - (f_N + f_S + f_E + f_W)

    With this convention:
    - (Lf) > 0 where f has a local maximum
    - (Lf) < 0 where f has a local minimum

    Since λ = 1 - f is highest at mass sources, (Lλ) > 0 near sources,
    which should correlate positively with ρ_act.

    Args:
        field: 2D array [ny, nx]

    Returns:
        Laplacian field [ny, nx]
    """
    # Get neighbors via roll (periodic boundary)
    f_N = np.roll(field, -1, axis=0)  # North neighbor
    f_S = np.roll(field, 1, axis=0)   # South neighbor
    f_E = np.roll(field, -1, axis=1)  # East neighbor
    f_W = np.roll(field, 1, axis=1)   # West neighbor

    # Standard graph Laplacian: L = D - A
    # (Lf)(x) = degree * f(x) - sum of neighbors
    laplacian = 4 * field - (f_N + f_S + f_E + f_W)

    return laplacian


def verify_poisson_emergence(
    lattice: "Lattice",
    source_map: "SourceMap",
) -> EmergenceResult:
    """
    Verify that the Poisson equation emerges from engine dynamics.

    Checks whether (Lλ)(x) ≈ κ · ρ_act(x) holds for the steady-state f field.

    Args:
        lattice: Lattice with established f field (after running scheduler)
        source_map: Source map defining activity density

    Returns:
        EmergenceResult with fields, metrics, and fitted κ
    """
    # Compute slowness λ = 1 - f
    lambda_field = 1.0 - lattice.f

    # Compute graph Laplacian of λ
    laplacian_lambda = compute_graph_laplacian_2d(lambda_field)

    # Get activity density
    rho_activity = source_map.rates.copy()

    # Flatten for linear fit (excluding edges for cleaner analysis)
    margin = 5
    inner_slice = (slice(margin, -margin), slice(margin, -margin))

    L_inner = laplacian_lambda[inner_slice].flatten()
    rho_inner = rho_activity[inner_slice].flatten()

    # Linear fit: (Lλ) = κ * ρ + c
    # We expect c ≈ 0 for proper emergence
    slope, intercept, r_value, p_value, std_err = stats.linregress(rho_inner, L_inner)

    # Compute correlation
    correlation = np.corrcoef(L_inner, rho_inner)[0, 1]

    # Compute RMSE
    predicted = slope * rho_inner + intercept
    rmse = np.sqrt(np.mean((L_inner - predicted) ** 2))

    return EmergenceResult(
        lambda_field=lambda_field,
        laplacian_lambda=laplacian_lambda,
        rho_activity=rho_activity,
        correlation=correlation,
        r_squared=r_value ** 2,
        fitted_kappa=slope,
        rmse=rmse,
        laplacian_flat=L_inner,
        rho_flat=rho_inner,
    )


def fit_kappa(
    lambda_field: np.ndarray,
    rho_field: np.ndarray,
    margin: int = 5,
) -> tuple[float, float, float]:
    """
    Fit the effective coupling constant κ from (Lλ) = κρ.

    Args:
        lambda_field: Slowness field λ = 1 - f
        rho_field: Activity density field
        margin: Edge margin to exclude

    Returns:
        (kappa, r_squared, rmse) - Fitted κ, R², and RMSE
    """
    # Compute Laplacian
    laplacian = compute_graph_laplacian_2d(lambda_field)

    # Use inner region
    inner_slice = (slice(margin, -margin), slice(margin, -margin))
    L_inner = laplacian[inner_slice].flatten()
    rho_inner = rho_field[inner_slice].flatten()

    # Linear fit
    slope, intercept, r_value, _, _ = stats.linregress(rho_inner, L_inner)

    # RMSE
    predicted = slope * rho_inner + intercept
    rmse = np.sqrt(np.mean((L_inner - predicted) ** 2))

    return slope, r_value ** 2, rmse


def compute_screening_length(beta: float, d: int = 4) -> float:
    """
    Compute theoretical screening length from paper Eq. 14.

    ξ ~ sqrt(β / [d(1 - β)])

    Args:
        beta: Synchronization coupling strength
        d: Lattice degree (4 for von Neumann, 8 for Moore)

    Returns:
        Screening length ξ
    """
    if beta >= 1.0:
        return float('inf')  # No screening in strong sync limit
    return np.sqrt(beta / (d * (1 - beta)))


def compute_screening_mass_squared(beta: float, d: int = 4) -> float:
    """
    Compute the screening mass squared m² for the screened Poisson equation.

    The full emergence relationship is:
        (Lλ) + m²λ = κρ

    where m² = d(1-β)/β. When β→1, m²→0 and we get pure Poisson.

    Args:
        beta: Synchronization coupling strength
        d: Lattice degree (4 for von Neumann, 8 for Moore)

    Returns:
        Screening mass squared m²
    """
    if beta >= 1.0:
        return 0.0
    if beta <= 0.0:
        return float('inf')
    return d * (1 - beta) / beta


@dataclass
class ScreenedEmergenceResult:
    """Results of screened Poisson emergence verification."""

    # The screened Poisson relationship: (Lλ) + m²λ = κρ
    m_squared_theoretical: float  # From beta: m² = d(1-β)/β
    m_squared_best_fit: float     # From optimization

    # Correlations
    correlation_pure: float       # corr((Lλ), ρ)
    correlation_screened: float   # corr((Lλ) + m²λ, ρ) with theoretical m²
    correlation_best: float       # corr with best-fit m²

    # Fit quality
    r_squared: float
    fitted_kappa: float

    # Fields for analysis
    lambda_field: np.ndarray
    laplacian_lambda: np.ndarray
    rho_activity: np.ndarray


def solve_theoretical_lambda(
    source_map: "SourceMap",
    beta: float = 0.9,
    gamma: float = 0.5,
    iterations: int = 500,
    capacity: float = 5.0,
) -> np.ndarray:
    """
    Solve the theoretical λ field from the paper's equation.

    The paper claims λ satisfies:
        λ(x) = γ·a(x) + β·⟨λ⟩_x

    where ⟨λ⟩_x is the average of λ over neighbors.

    This is a fixed-point equation that we solve iteratively.

    Args:
        source_map: Source map with activity rates
        beta: Synchronization coupling strength (0-1)
        gamma: Bandwidth coupling strength
        iterations: Number of iterations for convergence
        capacity: Link capacity for normalizing activity rates

    Returns:
        Theoretical λ field that should emerge from dynamics
    """
    # Scale activity by capacity (like bandwidth saturation)
    rates = source_map.rates
    a = rates / capacity

    # Initialize λ = 0
    lambda_field = np.zeros_like(a)

    # Iteratively solve λ = γa + β⟨λ⟩
    for _ in range(iterations):
        # Compute average of neighbors (periodic boundary)
        avg_lambda = (
            np.roll(lambda_field, 1, axis=0) +   # N
            np.roll(lambda_field, -1, axis=0) +  # S
            np.roll(lambda_field, 1, axis=1) +   # E
            np.roll(lambda_field, -1, axis=1)    # W
        ) / 4.0

        # Update: λ = γa + β⟨λ⟩
        lambda_field = gamma * a + beta * avg_lambda

    return lambda_field


def verify_screened_poisson_emergence(
    lattice: "Lattice",
    source_map: "SourceMap",
    beta: float = 0.9,
    d: int = 4,
) -> ScreenedEmergenceResult:
    """
    Verify the SCREENED Poisson equation emerges from engine dynamics.

    The full relationship (for β < 1) is:
        (Lλ)(x) + m²·λ(x) = κ·ρ(x)

    where m² = d(1-β)/β is the screening mass.

    When β → 1: m² → 0 and we get pure Poisson (Lλ) = κρ
    When β < 1: Screening suppresses long-range propagation

    Args:
        lattice: Lattice with established f field
        source_map: Source map defining activity density
        beta: Synchronization coupling strength used in simulation
        d: Lattice degree (4 for von Neumann)

    Returns:
        ScreenedEmergenceResult with correlations and fit quality
    """
    # Compute fields
    lambda_field = 1.0 - lattice.f
    laplacian_lambda = compute_graph_laplacian_2d(lambda_field)
    rho_activity = source_map.rates.copy()

    # Flatten for analysis (excluding edges)
    margin = 5
    inner_slice = (slice(margin, -margin), slice(margin, -margin))

    L_inner = laplacian_lambda[inner_slice].flatten()
    lambda_inner = lambda_field[inner_slice].flatten()
    rho_inner = rho_activity[inner_slice].flatten()

    # Theoretical m² from beta
    m_sq_theoretical = compute_screening_mass_squared(beta, d)

    # Pure Poisson correlation
    corr_pure = np.corrcoef(L_inner, rho_inner)[0, 1]

    # Screened Poisson with theoretical m²
    LHS_theoretical = L_inner + m_sq_theoretical * lambda_inner
    corr_screened = np.corrcoef(LHS_theoretical, rho_inner)[0, 1]

    # Find best m² via optimization
    best_m_sq = 0.0
    best_corr = corr_pure

    for m_sq in np.linspace(0, 20, 200):
        LHS = L_inner + m_sq * lambda_inner
        corr = np.corrcoef(LHS, rho_inner)[0, 1]
        if corr > best_corr:
            best_m_sq = m_sq
            best_corr = corr

    # Fit κ using best m²
    LHS_best = L_inner + best_m_sq * lambda_inner
    slope, intercept, r_value, _, _ = stats.linregress(rho_inner, LHS_best)

    return ScreenedEmergenceResult(
        m_squared_theoretical=m_sq_theoretical,
        m_squared_best_fit=best_m_sq,
        correlation_pure=corr_pure,
        correlation_screened=corr_screened,
        correlation_best=best_corr,
        r_squared=r_value ** 2,
        fitted_kappa=slope,
        lambda_field=lambda_field,
        laplacian_lambda=laplacian_lambda,
        rho_activity=rho_activity,
    )
