"""
Tests for SchrodingerKernel.

Verifies:
- Norm preservation (unitarity)
- Wave packet propagation with momentum
- Dispersion relation
- Coupling to f-field gravitational potential
"""

import numpy as np
import pytest

from mplsim.core import (
    Lattice,
    LatticeConfig,
    SchrodingerKernel,
    SchrodingerKernelConfig,
    create_schrodinger_kernel,
    init_gaussian_packet,
)


@pytest.fixture
def complex_lattice():
    """Create a complex-valued lattice."""
    config = LatticeConfig(nx=40, ny=40, n_channels=1, state_dtype="complex")
    return Lattice(config)


@pytest.fixture
def schrodinger_kernel():
    """Create a Schrödinger kernel with default settings."""
    kernel = create_schrodinger_kernel(dt=0.05, mass=1.0)
    kernel.set_directions(["up", "down", "left", "right"])
    return kernel


class TestSchrodingerKernelConfig:
    """Test configuration dataclass."""

    def test_default_config(self):
        cfg = SchrodingerKernelConfig()
        assert cfg.dt == 0.1
        assert cfg.mass == 1.0
        assert cfg.potential_strength == 0.0
        assert cfg.external_potential is None
        assert cfg.split_operator is True
        # FFT-based kernel doesn't need sync (global operation)
        assert cfg.sync_required is False

    def test_custom_config(self):
        cfg = SchrodingerKernelConfig(
            dt=0.05,
            mass=2.0,
            potential_strength=0.5,
        )
        assert cfg.dt == 0.05
        assert cfg.mass == 2.0
        assert cfg.potential_strength == 0.5


class TestSchrodingerKernelFactory:
    """Test factory function."""

    def test_create_kernel(self):
        kernel = create_schrodinger_kernel(dt=0.1, mass=1.0)
        assert isinstance(kernel, SchrodingerKernel)
        assert kernel.config.dt == 0.1
        assert kernel.config.mass == 1.0

    def test_create_with_potential(self):
        def V(x, y):
            return 0.5 * (x**2 + y**2)

        kernel = create_schrodinger_kernel(
            dt=0.05,
            mass=1.0,
            potential_strength=0.1,
            external_potential=V,
        )
        assert kernel.config.potential_strength == 0.1
        assert kernel.config.external_potential is not None


class TestSchrodingerKernelProperties:
    """Test kernel protocol properties."""

    def test_uses_state_activity(self, schrodinger_kernel):
        assert schrodinger_kernel.uses_state_activity is True

    def test_required_inputs_empty_for_fft(self, schrodinger_kernel):
        """FFT-based kernel doesn't need neighbor inputs - it's a global operation."""
        assert schrodinger_kernel.required_inputs == set()

    def test_set_directions_is_noop(self):
        """set_directions is kept for protocol compatibility but does nothing."""
        kernel = create_schrodinger_kernel(dt=0.1, mass=1.0)
        kernel.set_directions(["up", "down", "left", "right"])
        # Still returns empty set - FFT doesn't use neighbors
        assert kernel.required_inputs == set()


class TestNormPreservation:
    """Test that FFT-based evolution preserves norm exactly."""

    def test_single_point_preserves_norm(self, complex_lattice):
        """Single point state should preserve total norm after evolution."""
        lat = complex_lattice
        lat.state[20, 20, 0] = 1.0 + 0.5j

        initial_norm = np.sum(np.abs(lat.state) ** 2)

        kernel = create_schrodinger_kernel(dt=0.05, mass=1.0)
        kernel.evolve_lattice(lat)

        final_norm = np.sum(np.abs(lat.state) ** 2)

        # FFT split-step is exactly unitary
        np.testing.assert_almost_equal(final_norm, initial_norm, decimal=10)

    def test_uniform_state_preserves_norm(self, complex_lattice):
        """Uniform state should preserve norm exactly."""
        lat = complex_lattice
        lat.state[:, :, 0] = 1.0 / np.sqrt(lat.shape[0] * lat.shape[1])

        initial_norm = np.sum(np.abs(lat.state[:, :, 0]) ** 2)

        kernel = create_schrodinger_kernel(dt=0.05, mass=1.0)
        kernel.evolve_lattice(lat)

        final_norm = np.sum(np.abs(lat.state[:, :, 0]) ** 2)

        # FFT split-step is exactly unitary
        np.testing.assert_almost_equal(final_norm, initial_norm, decimal=10)

    def test_gaussian_packet_preserves_norm(self, complex_lattice):
        """Gaussian packet should preserve total norm exactly over many steps."""
        lat = complex_lattice
        init_gaussian_packet(lat, center=(20.0, 20.0), sigma=5.0, normalize=True)

        initial_norm = np.sum(np.abs(lat.state) ** 2)

        kernel = create_schrodinger_kernel(dt=0.1, mass=1.0)

        # Evolve many steps
        for _ in range(100):
            kernel.evolve_lattice(lat)

        final_norm = np.sum(np.abs(lat.state) ** 2)

        # FFT split-step preserves norm exactly
        np.testing.assert_almost_equal(final_norm, initial_norm, decimal=10)


class TestWavePacketPropagation:
    """Test that wave packets with momentum actually move."""

    def test_momentum_creates_movement(self, complex_lattice):
        """Wave packet with momentum should shift its center of mass.

        Note: center=(cx, cy) where cx is x-coordinate and cy is y-coordinate.
        With momentum=(kx, 0), packet should move in +x direction.
        """
        lat = complex_lattice

        # Initialize with rightward momentum - center at x=20, y=20
        kx = 0.5
        init_gaussian_packet(
            lat, center=(20.0, 20.0), sigma=4.0, momentum=(kx, 0.0), normalize=True
        )

        # Compute initial center of mass
        yy, xx = np.meshgrid(
            np.arange(lat.shape[0]), np.arange(lat.shape[1]), indexing="ij"
        )
        prob = np.abs(lat.state[:, :, 0]) ** 2
        initial_x_com = np.sum(xx * prob) / np.sum(prob)

        kernel = create_schrodinger_kernel(dt=0.1, mass=1.0)

        # Run evolution steps using evolve_lattice (FFT-based)
        for _ in range(50):
            kernel.evolve_lattice(lat)

        # Compute final center of mass
        prob = np.abs(lat.state[:, :, 0]) ** 2
        final_x_com = np.sum(xx * prob) / np.sum(prob)

        # With positive kx, packet should move in +x direction
        # Group velocity v_g = k/m for continuous Schrödinger
        displacement = final_x_com - initial_x_com
        assert displacement > 1.0  # Should move significantly rightward


class TestGravityCoupling:
    """Test coupling to f-field as gravitational potential."""

    def test_potential_from_f_field(self, complex_lattice):
        """Low f regions should create attractive potential."""
        lat = complex_lattice

        # Create f-field gradient (lower f on right side)
        for x in range(lat.shape[1]):
            lat.f_smooth[:, x] = 1.0 - 0.3 * (x / lat.shape[1])

        kernel = create_schrodinger_kernel(
            dt=0.05, mass=1.0, potential_strength=1.0
        )
        kernel.set_directions(["up", "down", "left", "right"])

        # Check potential increases to the right (lower f = higher V in our convention)
        V_left = kernel._get_potential(20, 5, lat)
        V_right = kernel._get_potential(20, 35, lat)

        # With V = strength * (1 - f), lower f means higher V
        assert V_right > V_left

    def test_external_potential(self, complex_lattice):
        """External potential function should be added."""

        def harmonic(x, y):
            cx, cy = 20, 20
            return 0.01 * ((x - cx) ** 2 + (y - cy) ** 2)

        kernel = create_schrodinger_kernel(
            dt=0.05, mass=1.0, external_potential=harmonic
        )
        kernel.set_directions(["up", "down", "left", "right"])

        V_center = kernel._get_potential(20, 20, complex_lattice)
        V_edge = kernel._get_potential(20, 30, complex_lattice)

        assert V_center == 0.0
        assert V_edge > 0.0


class TestDispersion:
    """Test wave packet dispersion properties."""

    def test_packet_spreads_over_time(self, complex_lattice):
        """Wave packet should spread (disperse) over time."""
        lat = complex_lattice
        init_gaussian_packet(lat, center=(20.0, 20.0), sigma=3.0, normalize=True)

        # Measure initial width
        prob = np.abs(lat.state[:, :, 0]) ** 2
        yy, xx = np.meshgrid(
            np.arange(lat.shape[0]), np.arange(lat.shape[1]), indexing="ij"
        )
        x_com = np.sum(xx * prob) / np.sum(prob)
        y_com = np.sum(yy * prob) / np.sum(prob)
        initial_width = np.sqrt(
            np.sum(((xx - x_com) ** 2 + (yy - y_com) ** 2) * prob) / np.sum(prob)
        )

        # Use smaller mass for faster dispersion
        kernel = create_schrodinger_kernel(dt=0.1, mass=0.5)

        # Evolve using FFT-based method
        for _ in range(100):
            kernel.evolve_lattice(lat)

        # Measure final width
        prob = np.abs(lat.state[:, :, 0]) ** 2
        x_com = np.sum(xx * prob) / np.sum(prob)
        y_com = np.sum(yy * prob) / np.sum(prob)
        final_width = np.sqrt(
            np.sum(((xx - x_com) ** 2 + (yy - y_com) ** 2) * prob) / np.sum(prob)
        )

        # Packet should have spread
        assert final_width > initial_width * 1.1

    def test_heavier_mass_disperses_slower(self):
        """Heavier mass should disperse more slowly."""
        # Light particle
        lat_light = Lattice(
            LatticeConfig(nx=40, ny=40, n_channels=1, state_dtype="complex")
        )
        init_gaussian_packet(lat_light, center=(20.0, 20.0), sigma=3.0, normalize=True)

        # Heavy particle (same initial state)
        lat_heavy = Lattice(
            LatticeConfig(nx=40, ny=40, n_channels=1, state_dtype="complex")
        )
        init_gaussian_packet(lat_heavy, center=(20.0, 20.0), sigma=3.0, normalize=True)

        kernel_light = create_schrodinger_kernel(dt=0.1, mass=0.5)
        kernel_heavy = create_schrodinger_kernel(dt=0.1, mass=2.0)

        # Evolve both using FFT-based method
        for _ in range(100):
            kernel_light.evolve_lattice(lat_light)
            kernel_heavy.evolve_lattice(lat_heavy)

        # Measure final widths
        yy, xx = np.meshgrid(np.arange(40), np.arange(40), indexing="ij")

        prob_light = np.abs(lat_light.state[:, :, 0]) ** 2
        x_com = np.sum(xx * prob_light) / np.sum(prob_light)
        y_com = np.sum(yy * prob_light) / np.sum(prob_light)
        width_light = np.sqrt(
            np.sum(((xx - x_com) ** 2 + (yy - y_com) ** 2) * prob_light)
            / np.sum(prob_light)
        )

        prob_heavy = np.abs(lat_heavy.state[:, :, 0]) ** 2
        x_com = np.sum(xx * prob_heavy) / np.sum(prob_heavy)
        y_com = np.sum(yy * prob_heavy) / np.sum(prob_heavy)
        width_heavy = np.sqrt(
            np.sum(((xx - x_com) ** 2 + (yy - y_com) ** 2) * prob_heavy)
            / np.sum(prob_heavy)
        )

        # Light particle should have spread more
        assert width_light > width_heavy


class TestComputeOutputBits:
    """Test activity computation for bandwidth scheduling."""

    def test_returns_placeholder(self, schrodinger_kernel, complex_lattice):
        from mplsim.core import SourceMap

        ny, nx = complex_lattice.shape
        source_map = SourceMap(ny=ny, nx=nx)
        result = schrodinger_kernel.compute_output_bits(20, 20, source_map)
        assert result == 1.0
