"""Unit tests for Lattice and LatticeConfig."""

import numpy as np
import pytest

from mplsim.core.lattice import Lattice, LatticeConfig


class TestLatticeConfig:
    """Tests for LatticeConfig."""

    def test_default_config(self):
        cfg = LatticeConfig(nx=50, ny=50)
        assert cfg.nx == 50
        assert cfg.ny == 50
        assert cfg.neighborhood == "von_neumann"
        assert cfg.boundary == "periodic"
        assert cfg.link_capacity == 10.0
        assert cfg.n_channels == 4

    def test_custom_config(self):
        cfg = LatticeConfig(
            nx=100,
            ny=80,
            neighborhood="moore",
            boundary="absorbing",
            link_capacity=20.0,
            n_channels=8,
        )
        assert cfg.nx == 100
        assert cfg.ny == 80
        assert cfg.neighborhood == "moore"
        assert cfg.link_capacity == 20.0


class TestLattice:
    """Tests for Lattice."""

    def test_creation(self):
        cfg = LatticeConfig(nx=50, ny=40)
        lat = Lattice(cfg)

        assert lat.state.shape == (40, 50, 4)
        assert lat.proper_time.shape == (40, 50)
        assert lat.f.shape == (40, 50)
        assert np.all(lat.f == 1.0)  # Initial f is 1

    def test_shape_property(self):
        cfg = LatticeConfig(nx=100, ny=80)
        lat = Lattice(cfg)
        assert lat.shape == (80, 100)

    def test_directions_von_neumann(self):
        cfg = LatticeConfig(nx=10, ny=10, neighborhood="von_neumann")
        lat = Lattice(cfg)
        assert set(lat.directions) == {"N", "S", "E", "W"}

    def test_directions_moore(self):
        cfg = LatticeConfig(nx=10, ny=10, neighborhood="moore")
        lat = Lattice(cfg)
        assert set(lat.directions) == {"N", "S", "E", "W", "NE", "NW", "SE", "SW"}


class TestNeighborLookup:
    """Tests for neighbor coordinate lookup."""

    def test_periodic_boundary_wrap_x(self):
        cfg = LatticeConfig(nx=50, ny=50, boundary="periodic")
        lat = Lattice(cfg)

        # East from right edge should wrap to left
        assert lat.get_neighbor_coords(49, 25, "E") == (0, 25)

        # West from left edge should wrap to right
        assert lat.get_neighbor_coords(0, 25, "W") == (49, 25)

    def test_periodic_boundary_wrap_y(self):
        cfg = LatticeConfig(nx=50, ny=50, boundary="periodic")
        lat = Lattice(cfg)

        # South from bottom edge should wrap to top
        assert lat.get_neighbor_coords(25, 49, "S") == (25, 0)

        # North from top edge should wrap to bottom
        assert lat.get_neighbor_coords(25, 0, "N") == (25, 49)

    def test_absorbing_boundary_returns_none(self):
        cfg = LatticeConfig(nx=50, ny=50, boundary="absorbing")
        lat = Lattice(cfg)

        # East from right edge should be None
        assert lat.get_neighbor_coords(49, 25, "E") is None

        # West from left edge should be None
        assert lat.get_neighbor_coords(0, 25, "W") is None

    def test_reflective_boundary_clamps(self):
        cfg = LatticeConfig(nx=50, ny=50, boundary="reflective")
        lat = Lattice(cfg)

        # East from right edge should stay at right edge
        assert lat.get_neighbor_coords(49, 25, "E") == (49, 25)

        # Corner case
        assert lat.get_neighbor_coords(0, 0, "W") == (0, 0)
        assert lat.get_neighbor_coords(0, 0, "N") == (0, 0)

    def test_interior_neighbors(self):
        cfg = LatticeConfig(nx=50, ny=50, boundary="periodic")
        lat = Lattice(cfg)

        neighbors = lat.get_all_neighbors(25, 25)
        assert neighbors["N"] == (25, 24)
        assert neighbors["S"] == (25, 26)
        assert neighbors["E"] == (26, 25)
        assert neighbors["W"] == (24, 25)


class TestIterNodes:
    """Tests for node iteration."""

    def test_iter_nodes_count(self):
        cfg = LatticeConfig(nx=10, ny=8)
        lat = Lattice(cfg)

        nodes = list(lat.iter_nodes())
        assert len(nodes) == 80

    def test_iter_nodes_coverage(self):
        cfg = LatticeConfig(nx=5, ny=4)
        lat = Lattice(cfg)

        nodes = set(lat.iter_nodes())
        expected = {(x, y) for x in range(5) for y in range(4)}
        assert nodes == expected


class TestUpdateStatistics:
    """Tests for f(x) update statistics."""

    def test_update_f_from_counts(self):
        cfg = LatticeConfig(nx=10, ny=10)
        lat = Lattice(cfg)

        # Simulate some updates
        lat.possible_updates.fill(100)
        lat.realized_updates.fill(80)
        lat.realized_updates[5, 5] = 50  # One slow node

        lat.update_f(smoothing_window=1)  # No smoothing for test

        # Most nodes should have f ≈ 0.8
        assert np.isclose(lat.f[0, 0], 0.8, atol=0.1)

        # Slow node should have f ≈ 0.5
        assert np.isclose(lat.f[5, 5], 0.5, atol=0.1)

    def test_reset_statistics(self):
        cfg = LatticeConfig(nx=10, ny=10)
        lat = Lattice(cfg)

        lat.possible_updates.fill(100)
        lat.realized_updates.fill(80)

        lat.reset_statistics()

        assert np.all(lat.possible_updates == 0)
        assert np.all(lat.realized_updates == 0)
