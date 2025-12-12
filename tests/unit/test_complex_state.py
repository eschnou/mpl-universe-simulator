"""Unit tests for complex state support in Lattice."""

import numpy as np
import pytest

from mplsim.core.lattice import Lattice, LatticeConfig
from mplsim.core.state_initializers import (
    init_gaussian_packet,
    init_plane_wave,
    init_two_packets,
    init_ring,
)


class TestComplexLatticeConfig:
    """Tests for complex state configuration."""

    def test_default_is_real(self):
        """Default state_dtype should be 'real'."""
        cfg = LatticeConfig(nx=10, ny=10)
        assert cfg.state_dtype == "real"

    def test_complex_config(self):
        """Can configure complex state."""
        cfg = LatticeConfig(nx=10, ny=10, state_dtype="complex")
        assert cfg.state_dtype == "complex"


class TestComplexLattice:
    """Tests for complex state in Lattice."""

    def test_real_state_dtype(self):
        """Real state should be float64."""
        cfg = LatticeConfig(nx=10, ny=10, state_dtype="real")
        lat = Lattice(cfg)

        assert lat.state.dtype == np.float64
        assert not lat.is_complex

    def test_complex_state_dtype(self):
        """Complex state should be complex128."""
        cfg = LatticeConfig(nx=10, ny=10, state_dtype="complex")
        lat = Lattice(cfg)

        assert lat.state.dtype == np.complex128
        assert lat.is_complex

    def test_activity_field_exists(self):
        """Activity field should exist and be real."""
        cfg = LatticeConfig(nx=10, ny=10, state_dtype="complex")
        lat = Lattice(cfg)

        assert hasattr(lat, "activity")
        assert lat.activity.dtype == np.float64
        assert lat.activity.shape == (10, 10)

    def test_complex_state_shape(self):
        """Complex state should have correct shape."""
        cfg = LatticeConfig(nx=20, ny=15, n_channels=2, state_dtype="complex")
        lat = Lattice(cfg)

        assert lat.state.shape == (15, 20, 2)


class TestActivityFromState:
    """Tests for computing activity from state norm."""

    def test_activity_zero_state(self):
        """Zero state should give zero activity."""
        cfg = LatticeConfig(nx=10, ny=10, state_dtype="complex")
        lat = Lattice(cfg)

        activity = lat.compute_activity_from_state()

        assert np.all(activity == 0.0)

    def test_activity_single_node(self):
        """Activity at single node should be |psi|^2."""
        cfg = LatticeConfig(nx=10, ny=10, n_channels=2, state_dtype="complex")
        lat = Lattice(cfg)

        # Set known state: |0.6 + 0.8i|^2 = 0.36 + 0.64 = 1.0
        lat.state[5, 5, 0] = 0.6 + 0.8j
        lat.state[5, 5, 1] = 0.5  # Real: |0.5|^2 = 0.25

        activity = lat.compute_activity_from_state()

        # Total activity at (5,5) = 1.0 + 0.25 = 1.25
        assert np.isclose(activity[5, 5], 1.25)
        assert np.isclose(lat.activity[5, 5], 1.25)  # Should update field

    def test_activity_from_real_state(self):
        """Activity computation should work for real state too."""
        cfg = LatticeConfig(nx=10, ny=10, n_channels=1, state_dtype="real")
        lat = Lattice(cfg)

        lat.state[3, 3, 0] = 2.0

        activity = lat.compute_activity_from_state()

        assert np.isclose(activity[3, 3], 4.0)  # 2^2 = 4

    def test_activity_sums_over_channels(self):
        """Activity should sum |psi|^2 over all channels."""
        cfg = LatticeConfig(nx=5, ny=5, n_channels=4, state_dtype="complex")
        lat = Lattice(cfg)

        # Set each channel to 0.5 at one node
        lat.state[2, 2, :] = 0.5 + 0j

        activity = lat.compute_activity_from_state()

        # Total: 4 * 0.25 = 1.0
        assert np.isclose(activity[2, 2], 1.0)


class TestGaussianPacketInit:
    """Tests for Gaussian wave packet initialization."""

    def test_requires_complex_state(self):
        """Should raise if lattice has real state."""
        cfg = LatticeConfig(nx=20, ny=20, state_dtype="real")
        lat = Lattice(cfg)

        with pytest.raises(AssertionError):
            init_gaussian_packet(lat, center=(10, 10), sigma=3)

    def test_gaussian_shape(self):
        """Gaussian should be peaked at center."""
        cfg = LatticeConfig(nx=50, ny=50, n_channels=1, state_dtype="complex")
        lat = Lattice(cfg)

        init_gaussian_packet(lat, center=(25, 25), sigma=5, normalize=False, amplitude=1.0)

        # Maximum should be at center
        activity = np.abs(lat.state[:, :, 0]) ** 2
        max_idx = np.unravel_index(np.argmax(activity), activity.shape)
        assert max_idx == (25, 25)

        # Should decay away from center
        assert activity[25, 25] > activity[25, 30]
        assert activity[25, 25] > activity[25, 20]

    def test_gaussian_normalization(self):
        """Normalized Gaussian should have total |psi|^2 = 1."""
        cfg = LatticeConfig(nx=100, ny=100, n_channels=1, state_dtype="complex")
        lat = Lattice(cfg)

        init_gaussian_packet(lat, center=(50, 50), sigma=10, normalize=True)

        total_norm = np.sum(np.abs(lat.state) ** 2)
        assert np.isclose(total_norm, 1.0, rtol=1e-6)

    def test_gaussian_momentum(self):
        """Momentum should add phase gradient."""
        cfg = LatticeConfig(nx=50, ny=50, n_channels=1, state_dtype="complex")
        lat = Lattice(cfg)

        init_gaussian_packet(lat, center=(25, 25), sigma=5, momentum=(0.5, 0))

        # Phase should increase along x
        psi = lat.state[:, :, 0]
        phase_at_center = np.angle(psi[25, 25])
        phase_at_right = np.angle(psi[25, 30])

        # With kx=0.5, phase difference over 5 steps ≈ 2.5 radians
        phase_diff = (phase_at_right - phase_at_center) % (2 * np.pi)
        assert phase_diff > 0.5  # Should have positive phase gradient


class TestPlaneWaveInit:
    """Tests for plane wave initialization."""

    def test_plane_wave_uniform_magnitude(self):
        """Plane wave should have uniform magnitude."""
        cfg = LatticeConfig(nx=20, ny=20, n_channels=1, state_dtype="complex")
        lat = Lattice(cfg)

        init_plane_wave(lat, momentum=(0.3, 0.2))

        magnitudes = np.abs(lat.state[:, :, 0])

        # All magnitudes should be equal
        assert np.allclose(magnitudes, magnitudes[0, 0])

    def test_plane_wave_normalization(self):
        """Plane wave amplitude should give correct total norm."""
        cfg = LatticeConfig(nx=20, ny=20, n_channels=1, state_dtype="complex")
        lat = Lattice(cfg)

        init_plane_wave(lat, momentum=(0.3, 0.2), amplitude=1.0)

        total_norm = np.sum(np.abs(lat.state) ** 2)
        assert np.isclose(total_norm, 1.0, rtol=1e-6)


class TestTwoPacketsInit:
    """Tests for two-packet superposition initialization."""

    def test_two_packets_creates_superposition(self):
        """Should create two separate peaks."""
        cfg = LatticeConfig(nx=100, ny=50, n_channels=1, state_dtype="complex")
        lat = Lattice(cfg)

        init_two_packets(
            lat,
            centers=((25, 25), (75, 25)),
            sigma=5,
        )

        activity = np.abs(lat.state[:, :, 0]) ** 2

        # Should have peaks near both centers
        left_region = activity[20:30, 20:30]
        right_region = activity[20:30, 70:80]

        assert left_region.max() > activity.mean() * 5
        assert right_region.max() > activity.mean() * 5

    def test_two_packets_normalized(self):
        """Two-packet superposition should be normalized."""
        cfg = LatticeConfig(nx=100, ny=100, n_channels=1, state_dtype="complex")
        lat = Lattice(cfg)

        init_two_packets(
            lat,
            centers=((30, 50), (70, 50)),
            sigma=8,
        )

        total_norm = np.sum(np.abs(lat.state) ** 2)
        assert np.isclose(total_norm, 1.0, rtol=1e-6)


class TestRingInit:
    """Tests for ring wave function initialization."""

    def test_ring_shape(self):
        """Ring should be peaked at specified radius."""
        cfg = LatticeConfig(nx=100, ny=100, n_channels=1, state_dtype="complex")
        lat = Lattice(cfg)

        init_ring(lat, center=(50, 50), radius=20, width=3, normalize=False)

        activity = np.abs(lat.state[:, :, 0]) ** 2

        # Center should be low
        assert activity[50, 50] < activity[50, 70]

        # Ring location should be high
        assert activity[50, 70] > activity[50, 80]  # Outside ring
        assert activity[50, 70] > activity[50, 50]  # Inside ring

    def test_ring_angular_momentum(self):
        """Angular momentum should create phase winding."""
        cfg = LatticeConfig(nx=100, ny=100, n_channels=1, state_dtype="complex")
        lat = Lattice(cfg)

        init_ring(
            lat,
            center=(50, 50),
            radius=20,
            width=3,
            angular_momentum=2,
        )

        # Check phase around the ring
        psi = lat.state[:, :, 0]

        # Phase at different angles around ring
        # At x=70, y=50 (theta=0): should have phase=0 (mod 2pi)
        # At x=50, y=70 (theta=pi/2): should have phase=pi (for m=2)
        phase_0 = np.angle(psi[50, 70])
        phase_90 = np.angle(psi[70, 50])

        # For m=2, going 90 degrees should change phase by pi
        # (allowing for phase wrapping)
        phase_diff = np.abs(phase_90 - phase_0)
        # Should be near pi or 3pi (with wrapping)
        assert np.isclose(phase_diff, np.pi, atol=0.3) or np.isclose(
            phase_diff, 3 * np.pi, atol=0.3
        )
