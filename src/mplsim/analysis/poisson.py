"""
Poisson solver for theoretical comparison.

Solves ∇²ϕ = 4πGρ (or in 2D: ∇²ϕ = 2πGρ) to get the Newtonian
gravitational potential from a mass density distribution.

IMPORTANT: This is for COMPARISON ONLY. The engine does NOT use this.
We compare ϕ_engine (derived from f) with ϕ_poisson (from Newtonian theory)
to validate that our bandwidth-limited model reproduces gravity.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


@dataclass
class PoissonSolver:
    """
    Solve the 2D Poisson equation ∇²ϕ = ρ.

    Supports periodic and Dirichlet (absorbing) boundary conditions.
    """

    boundary: Literal["periodic", "absorbing"] = "absorbing"
    dx: float = 1.0  # Grid spacing

    def solve(self, rho: np.ndarray) -> np.ndarray:
        """
        Solve ∇²ϕ = ρ for the potential ϕ.

        Args:
            rho: Source density field, shape [ny, nx]

        Returns:
            ϕ field, shape [ny, nx]
        """
        ny, nx = rho.shape
        n = ny * nx

        # Build the Laplacian matrix
        L = self._build_laplacian(ny, nx)

        # Flatten rho to 1D
        rho_flat = rho.flatten()

        if self.boundary == "periodic":
            # For periodic BC, the system is singular (constant mode)
            # We fix ϕ at one point (corner) to remove the degeneracy
            # Or use the mean-subtracted version
            rho_flat = rho_flat - rho_flat.mean()

        # Solve L @ phi = rho
        phi_flat = spsolve(L, rho_flat)

        # Reshape to 2D
        phi = phi_flat.reshape(ny, nx)

        # Negate for gravitational convention: we want ϕ to be a "well"
        # (most negative at the source, increasing toward zero at infinity)
        phi = -phi

        # For periodic, center around zero
        if self.boundary == "periodic":
            phi = phi - phi.mean()

        return phi

    def _build_laplacian(self, ny: int, nx: int) -> sparse.csr_matrix:
        """
        Build the 2D discrete Laplacian matrix.

        Uses 5-point stencil: ∇²ϕ ≈ (ϕ_E + ϕ_W + ϕ_N + ϕ_S - 4ϕ_C) / dx²
        """
        n = ny * nx
        dx2 = self.dx ** 2

        # Diagonal: -4/dx² for each point
        diag = -4.0 / dx2 * np.ones(n)

        # Off-diagonals for E/W neighbors (offset ±1)
        off_1 = np.ones(n - 1) / dx2

        # Off-diagonals for N/S neighbors (offset ±nx)
        off_nx = np.ones(n - nx) / dx2

        # Handle boundary conditions in off-diagonals
        if self.boundary == "absorbing":
            # Zero out connections that cross boundaries
            # E/W: zero out at row boundaries (where i % nx == nx-1 or 0)
            for i in range(n - 1):
                if (i + 1) % nx == 0:  # Right edge
                    off_1[i] = 0

        # Build sparse matrix
        diagonals = [diag, off_1, off_1, off_nx, off_nx]
        offsets = [0, 1, -1, nx, -nx]

        L = sparse.diags(diagonals, offsets, shape=(n, n), format="csr")

        if self.boundary == "periodic":
            # Add wrap-around connections
            L = L.tolil()

            for i in range(n):
                y, x = divmod(i, nx)

                # Wrap E/W
                if x == 0:
                    L[i, i + nx - 1] = 1.0 / dx2  # West wraps
                if x == nx - 1:
                    L[i, i - nx + 1] = 1.0 / dx2  # East wraps

                # Wrap N/S
                if y == 0:
                    L[i, i + (ny - 1) * nx] = 1.0 / dx2  # North wraps
                if y == ny - 1:
                    L[i, i - (ny - 1) * nx] = 1.0 / dx2  # South wraps

            L = L.tocsr()

        return L


def solve_poisson(rho: np.ndarray, boundary: str = "absorbing") -> np.ndarray:
    """Convenience function to solve Poisson equation."""
    return PoissonSolver(boundary=boundary).solve(rho)
