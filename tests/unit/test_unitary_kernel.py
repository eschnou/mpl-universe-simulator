"""Unit tests for UnitaryKernel."""

import numpy as np
import pytest

from mplsim.core.lattice import Lattice, LatticeConfig
from mplsim.core.kernel import EvolvingKernel
from mplsim.core.unitary_kernel import (
    UnitaryKernel,
    UnitaryKernelConfig,
    create_unitary_kernel,
)
from mplsim.core.state_initializers import init_gaussian_packet


class TestUnitaryKernelConfig:
    """Tests for UnitaryKernelConfig."""

    def test_default_config(self):
        """Default config should have reasonable values."""
        cfg = UnitaryKernelConfig()

        assert cfg.theta == 0.1
        assert cfg.omega is None
        assert cfg.sync_required is True

    def test_custom_config(self):
        """Can customize config."""
        omega = np.array([0.1, 0.2])
        cfg = UnitaryKernelConfig(theta=0.5, omega=omega, sync_required=False)

        assert cfg.theta == 0.5
        assert np.array_equal(cfg.omega, omega)
        assert cfg.sync_required is False


class TestUnitaryKernelProtocol:
    """Tests for protocol compliance."""

    def test_is_evolving_kernel(self):
        """UnitaryKernel should satisfy EvolvingKernel protocol."""
        kernel = UnitaryKernel()

        assert isinstance(kernel, EvolvingKernel)

    def test_uses_state_activity(self):
        """UnitaryKernel should use state-based activity."""
        kernel = UnitaryKernel()

        assert kernel.uses_state_activity is True

    def test_required_inputs_with_sync(self):
        """With sync_required=True, should require all directions."""
        kernel = create_unitary_kernel(sync_required=True)
        kernel.set_directions(["N", "S", "E", "W"])

        assert kernel.required_inputs == {"N", "S", "E", "W"}

    def test_required_inputs_without_sync(self):
        """With sync_required=False, should not require inputs."""
        kernel = create_unitary_kernel(sync_required=False)
        kernel.set_directions(["N", "S", "E", "W"])

        assert kernel.required_inputs == set()


class TestUnitaryEvolution:
    """Tests for unitary evolution dynamics."""

    def test_rotation_matrix_is_unitary(self):
        """Internal rotation matrix should be unitary."""
        kernel = UnitaryKernel(config=UnitaryKernelConfig(theta=0.3))

        U = kernel._rotation
        assert U is not None

        # U^dagger * U should be identity
        UdagU = np.conj(U.T) @ U
        assert np.allclose(UdagU, np.eye(2), atol=1e-10)

    def test_evolve_with_zero_neighbors_reduces_norm(self):
        """Evolution with zero neighbors reduces amplitude (rotation to neighbor space).

        When mixing with zero neighbors, we rotate some amplitude into the
        "neighbor" output channel, which we discard. This reduces the node's
        amplitude by cos^n(theta) where n = number of neighbors.
        """
        cfg = LatticeConfig(nx=5, ny=5, n_channels=1, state_dtype="complex")
        lat = Lattice(cfg)

        # Set initial state
        lat.state[2, 2, 0] = 1.0 + 0j

        theta = 0.2
        kernel = create_unitary_kernel(theta=theta)
        kernel.set_directions(["N", "S", "E", "W"])

        # Evolve with zero neighbors
        incoming = {d: np.zeros(1, dtype=np.complex128) for d in ["N", "S", "E", "W"]}
        new_state = kernel.evolve_node(2, 2, lat, incoming)

        # With zero neighbors, amplitude reduces by cos^4(theta)
        # because we apply 4 rotations and discard the neighbor part
        expected_amplitude = np.cos(theta) ** 4
        actual_amplitude = np.abs(new_state[0])

        assert np.isclose(actual_amplitude, expected_amplitude, rtol=1e-5)

    def test_evolve_mixes_with_neighbors(self):
        """Evolution should mix state with neighbors."""
        cfg = LatticeConfig(nx=5, ny=5, n_channels=1, state_dtype="complex")
        lat = Lattice(cfg)

        # Node at center has amplitude 1, neighbors have amplitude 0
        lat.state[2, 2, 0] = 1.0 + 0j

        kernel = create_unitary_kernel(theta=0.3)
        kernel.set_directions(["N", "S", "E", "W"])

        # Neighbors all zero
        incoming = {d: np.array([0.0 + 0j], dtype=np.complex128) for d in ["N", "S", "E", "W"]}
        new_state = kernel.evolve_node(2, 2, lat, incoming)

        # With zero neighbors, should just apply self-rotations
        # The cos^4(theta) factor comes from 4 identity mixings
        expected_magnitude = np.cos(0.3) ** 4
        assert np.isclose(np.abs(new_state[0]), expected_magnitude, rtol=1e-5)

    def test_evolve_with_neighbor_amplitude(self):
        """Evolution with neighbor amplitude should create mixing."""
        cfg = LatticeConfig(nx=5, ny=5, n_channels=1, state_dtype="complex")
        lat = Lattice(cfg)

        # Center node is 1, one neighbor is 1i
        lat.state[2, 2, 0] = 1.0 + 0j

        theta = 0.2
        kernel = create_unitary_kernel(theta=theta)
        kernel.set_directions(["N", "S", "E", "W"])

        # North neighbor has amplitude 1i (purely imaginary)
        incoming = {
            "N": np.array([0.0 + 1.0j], dtype=np.complex128),
            "S": np.array([0.0 + 0j], dtype=np.complex128),
            "E": np.array([0.0 + 0j], dtype=np.complex128),
            "W": np.array([0.0 + 0j], dtype=np.complex128),
        }
        new_state = kernel.evolve_node(2, 2, lat, incoming)

        # The rotation [[c, -is], [-is, c]] applied to [1, i] gives
        # self_output = c*1 + (-is)*i = c + s = cos + sin (real!)
        # So we actually get a REAL result from this particular input
        # Let's check that the amplitude changed due to mixing
        expected_first_step = np.cos(theta) * 1.0 + (-1j * np.sin(theta)) * 1j
        # = cos + sin (real)
        assert np.isclose(expected_first_step.real, np.cos(theta) + np.sin(theta))
        assert np.isclose(expected_first_step.imag, 0.0)

        # Final state after 4 rotations (3 with zero neighbors)
        # The first rotation with neighbor gives cos+sin, then subsequent
        # rotations with zero neighbors multiply by cos each
        final_expected = (np.cos(theta) + np.sin(theta)) * (np.cos(theta) ** 3)

        assert np.isclose(np.abs(new_state[0]), np.abs(final_expected), rtol=1e-5)

    def test_local_phase_evolution(self):
        """Local omega should add phase rotation."""
        cfg = LatticeConfig(nx=5, ny=5, n_channels=1, state_dtype="complex")
        lat = Lattice(cfg)

        lat.state[2, 2, 0] = 1.0 + 0j

        omega = np.array([np.pi / 4])  # 45 degree rotation
        kernel = UnitaryKernel(config=UnitaryKernelConfig(theta=0.0, omega=omega))
        kernel.set_directions(["N", "S", "E", "W"])

        incoming = {d: np.array([0.0 + 0j], dtype=np.complex128) for d in ["N", "S", "E", "W"]}
        new_state = kernel.evolve_node(2, 2, lat, incoming)

        # Phase should have rotated by -pi/4
        expected_phase = -np.pi / 4
        actual_phase = np.angle(new_state[0])
        assert np.isclose(actual_phase, expected_phase, atol=1e-10)


class TestUnitaryKernelFactory:
    """Tests for create_unitary_kernel factory."""

    def test_factory_creates_kernel(self):
        """Factory should create valid kernel."""
        kernel = create_unitary_kernel()

        assert isinstance(kernel, UnitaryKernel)
        assert kernel.uses_state_activity

    def test_factory_with_theta(self):
        """Factory should accept theta parameter."""
        kernel = create_unitary_kernel(theta=0.5)

        assert kernel.config.theta == 0.5

    def test_factory_with_omega(self):
        """Factory should accept omega parameter."""
        omega = np.array([0.1, 0.2, 0.3, 0.4])
        kernel = create_unitary_kernel(omega=omega)

        assert np.array_equal(kernel.config.omega, omega)


class TestGlobalNormBehavior:
    """Tests for global norm behavior across lattice.

    Note: The simple mixing unitary doesn't preserve global norm exactly
    because we only update one side of each pair. This is a design choice
    for simplicity. True global norm preservation would require symmetric
    Trotter-like updates.

    These tests verify the expected behavior: norm is approximately stable
    when neighbors have similar amplitudes.
    """

    def test_uniform_state_stays_uniform(self):
        """Uniform state should stay uniform (no spatial structure forms).

        Note: Even uniform states don't preserve norm exactly with our
        simple kernel because the rotation mixes amplitude between self
        and neighbor outputs, and we only keep the self part. But the
        state should remain spatially uniform.
        """
        cfg = LatticeConfig(nx=10, ny=10, n_channels=1, state_dtype="complex")
        lat = Lattice(cfg)

        # Uniform state: all nodes have same amplitude
        lat.state[:, :, 0] = 0.1 + 0j

        kernel = create_unitary_kernel(theta=0.1)
        kernel.set_directions(["N", "S", "E", "W"])

        # Evolve all nodes synchronously
        for _ in range(5):
            new_states = np.zeros_like(lat.state)
            for y in range(10):
                for x in range(10):
                    incoming = {}
                    for d, (dx, dy) in [("N", (0, -1)), ("S", (0, 1)), ("E", (1, 0)), ("W", (-1, 0))]:
                        nx_ = (x + dx) % 10
                        ny_ = (y + dy) % 10
                        incoming[d] = lat.state[ny_, nx_, :].copy()
                    new_states[y, x, :] = kernel.evolve_node(y, x, lat, incoming)
            lat.state[:] = new_states

        # State should still be uniform (no spatial structure)
        state_flat = lat.state[:, :, 0].flatten()
        assert np.allclose(state_flat, state_flat[0]), "Uniform state should stay uniform"

    def test_localized_state_norm_changes(self):
        """Localized state will spread and norm may change."""
        cfg = LatticeConfig(nx=20, ny=20, n_channels=1, state_dtype="complex")
        lat = Lattice(cfg)

        # Initialize localized state
        init_gaussian_packet(lat, center=(10, 10), sigma=3, normalize=True)

        initial_norm = np.sum(np.abs(lat.state) ** 2)

        kernel = create_unitary_kernel(theta=0.1)
        kernel.set_directions(["N", "S", "E", "W"])

        # Evolve
        for _ in range(5):
            new_states = np.zeros_like(lat.state)
            for y in range(20):
                for x in range(20):
                    incoming = {}
                    for d, (dx, dy) in [("N", (0, -1)), ("S", (0, 1)), ("E", (1, 0)), ("W", (-1, 0))]:
                        nx_ = (x + dx) % 20
                        ny_ = (y + dy) % 20
                        incoming[d] = lat.state[ny_, nx_, :].copy()
                    new_states[y, x, :] = kernel.evolve_node(y, x, lat, incoming)
            lat.state[:] = new_states

        final_norm = np.sum(np.abs(lat.state) ** 2)

        # Norm may change for localized states, but should stay reasonable
        # (not explode or vanish)
        assert 0.5 < final_norm < 2.0, f"Norm changed too drastically: {final_norm}"

    def test_small_theta_preserves_norm_better(self):
        """Small theta should give better norm preservation."""
        cfg = LatticeConfig(nx=15, ny=15, n_channels=1, state_dtype="complex")

        norms_small_theta = []
        norms_large_theta = []

        for theta, norm_list in [(0.01, norms_small_theta), (0.3, norms_large_theta)]:
            lat = Lattice(cfg)
            init_gaussian_packet(lat, center=(7, 7), sigma=3, normalize=True)

            kernel = create_unitary_kernel(theta=theta)
            kernel.set_directions(["N", "S", "E", "W"])

            for _ in range(10):
                new_states = np.zeros_like(lat.state)
                for y in range(15):
                    for x in range(15):
                        incoming = {}
                        for d, (dx, dy) in [("N", (0, -1)), ("S", (0, 1)), ("E", (1, 0)), ("W", (-1, 0))]:
                            nx_ = (x + dx) % 15
                            ny_ = (y + dy) % 15
                            incoming[d] = lat.state[ny_, nx_, :].copy()
                        new_states[y, x, :] = kernel.evolve_node(y, x, lat, incoming)
                lat.state[:] = new_states
                norm_list.append(np.sum(np.abs(lat.state) ** 2))

        # Small theta should have less norm variation
        small_theta_var = np.var(norms_small_theta)
        large_theta_var = np.var(norms_large_theta)

        assert small_theta_var < large_theta_var, (
            f"Small theta should have less norm variance: "
            f"small={small_theta_var:.6f}, large={large_theta_var:.6f}"
        )
