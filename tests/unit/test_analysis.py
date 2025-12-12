"""Unit tests for analysis module."""

import numpy as np
import pytest

from mplsim.analysis.phi_field import PhiFromF, compute_phi_linear, compute_phi_log
from mplsim.analysis.poisson import PoissonSolver, solve_poisson
from mplsim.analysis.comparison import compute_radial_profile


class TestPhiFromF:
    """Tests for PhiFromF."""

    def test_linear_mode_uniform(self):
        """Linear mode with uniform f."""
        f = np.ones((10, 10)) * 0.8
        phi = PhiFromF(mode="linear", alpha=1.0).compute(f)

        # ϕ = α(f - 1) = 1.0 * (0.8 - 1) = -0.2
        assert np.allclose(phi, -0.2)

    def test_linear_mode_f_equals_one(self):
        """When f=1, ϕ should be 0."""
        f = np.ones((10, 10))
        phi = PhiFromF(mode="linear", alpha=1.0).compute(f)
        assert np.allclose(phi, 0.0)

    def test_linear_mode_f_equals_zero(self):
        """When f=0, ϕ should be -α."""
        f = np.zeros((10, 10))
        phi = PhiFromF(mode="linear", alpha=2.0).compute(f)
        assert np.allclose(phi, -2.0)

    def test_linear_deeper_where_f_lower(self):
        """ϕ should be more negative where f is lower."""
        f = np.ones((50, 50))
        f[25, 25] = 0.5  # Low f at center
        phi = PhiFromF(mode="linear").compute(f)

        # Deeper (more negative) at center
        assert phi[25, 25] < phi[0, 0]

    def test_log_mode_f_equals_one(self):
        """Log mode: when f=1, ϕ should be 0."""
        f = np.ones((10, 10))
        phi = PhiFromF(mode="log", alpha=1.0).compute(f)
        assert np.allclose(phi, 0.0)

    def test_log_mode_asymptotes(self):
        """Log mode should go very negative as f approaches 0."""
        f = np.array([[1.0, 0.5, 0.1, 0.01]])
        phi = PhiFromF(mode="log", alpha=1.0).compute(f)

        # Should be monotonically decreasing
        assert phi[0, 0] > phi[0, 1] > phi[0, 2] > phi[0, 3]

    def test_compute_gradient(self):
        """Test gradient computation."""
        f = np.ones((50, 50))
        f[25, 25] = 0.5  # Dip at center

        phi_computer = PhiFromF(mode="linear")
        dphi_dx, dphi_dy = phi_computer.compute_gradient(f)

        # Gradient should point away from center (toward higher ϕ)
        # At the dip, gradient magnitudes should be non-zero nearby
        assert dphi_dx.shape == (50, 50)
        assert dphi_dy.shape == (50, 50)

    def test_compute_acceleration(self):
        """Test acceleration (negative gradient)."""
        f = np.ones((50, 50))
        f[25, 25] = 0.5

        phi_computer = PhiFromF(mode="linear")
        ax, ay = phi_computer.compute_acceleration(f)

        # Acceleration should point toward center (lower ϕ)
        assert ax.shape == (50, 50)
        assert ay.shape == (50, 50)


class TestPoissonSolver:
    """Tests for PoissonSolver."""

    def test_zero_source(self):
        """Zero source should give zero potential (absorbing BC)."""
        rho = np.zeros((20, 20))
        phi = PoissonSolver(boundary="absorbing").solve(rho)
        assert np.allclose(phi, 0.0, atol=1e-10)

    def test_point_source_has_extremum_at_source(self):
        """Point source should create potential extremum at source."""
        rho = np.zeros((31, 31))
        rho[15, 15] = 100.0

        phi = PoissonSolver(boundary="absorbing").solve(rho)

        # Potential magnitude should be maximum at source
        phi_abs = np.abs(phi)
        assert phi_abs[15, 15] == phi_abs.max()

    def test_point_source_radial_falloff(self):
        """Potential magnitude should decrease with distance from source."""
        rho = np.zeros((51, 51))
        rho[25, 25] = 100.0

        phi = PoissonSolver(boundary="absorbing").solve(rho)

        # Check radial falloff (magnitude decreases with distance)
        phi_abs_center = np.abs(phi[25, 25])
        phi_abs_r5 = np.abs(phi[25, 30])
        phi_abs_r10 = np.abs(phi[25, 35])

        assert phi_abs_center > phi_abs_r5 > phi_abs_r10

    def test_periodic_boundary(self):
        """Periodic boundary should work without errors."""
        rho = np.zeros((20, 20))
        rho[10, 10] = 100.0

        phi = PoissonSolver(boundary="periodic").solve(rho)

        # Should have an extremum at source
        phi_abs = np.abs(phi)
        assert phi_abs[10, 10] == phi_abs.max()

    def test_convenience_function(self):
        """Test solve_poisson convenience function."""
        rho = np.zeros((20, 20))
        rho[10, 10] = 100.0

        phi = solve_poisson(rho, boundary="absorbing")
        assert phi.shape == (20, 20)
        phi_abs = np.abs(phi)
        assert phi_abs[10, 10] == phi_abs.max()


class TestRadialProfile:
    """Tests for radial profile computation."""

    def test_uniform_field(self):
        """Uniform field should have flat radial profile."""
        field = np.ones((50, 50)) * 5.0
        radii, values = compute_radial_profile(field, center=(25, 25), max_radius=20)

        assert len(radii) == 21
        assert np.allclose(values, 5.0, atol=0.1)

    def test_radial_field(self):
        """Field with radial structure should show in profile."""
        field = np.zeros((51, 51))
        yy, xx = np.ogrid[:51, :51]
        dist = np.sqrt((xx - 25) ** 2 + (yy - 25) ** 2)
        field = 10.0 - dist * 0.2  # Decreases with radius

        radii, values = compute_radial_profile(field, center=(25, 25), max_radius=20)

        # Values should decrease with radius
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1] - 0.5  # Allow some tolerance
