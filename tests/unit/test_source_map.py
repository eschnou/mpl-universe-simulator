"""Unit tests for SourceMap."""

import numpy as np
import pytest

from mplsim.core.source_map import SourceMap


class TestSourceMapCreation:
    """Tests for SourceMap creation."""

    def test_creation_with_background(self):
        sm = SourceMap(ny=50, nx=100, background_rate=0.5)
        assert sm.ny == 50
        assert sm.nx == 100
        assert sm.rates.shape == (50, 100)
        assert np.all(sm.rates == 0.5)

    def test_default_background(self):
        sm = SourceMap(ny=10, nx=10)
        assert np.all(sm.rates == 0.1)


class TestPointSource:
    """Tests for point sources."""

    def test_add_point_source(self):
        sm = SourceMap(ny=50, nx=50, background_rate=0.1)
        sm.add_point_source(x=25, y=25, rate=10.0)

        assert sm.rates[25, 25] == 10.0
        assert sm.rates[0, 0] == 0.1  # Background unchanged

    def test_point_source_out_of_bounds_ignored(self):
        sm = SourceMap(ny=50, nx=50, background_rate=0.1)
        sm.add_point_source(x=100, y=25, rate=10.0)  # Out of bounds

        # Should not raise, and no change
        assert sm.max_rate() == 0.1


class TestGaussianSource:
    """Tests for Gaussian sources."""

    def test_gaussian_peak_at_center(self):
        sm = SourceMap(ny=100, nx=100, background_rate=0.0)
        sm.add_gaussian_source(cx=50, cy=50, peak_rate=10.0, sigma=5.0)

        # Peak should be at center
        assert np.isclose(sm.rates[50, 50], 10.0, atol=0.01)

    def test_gaussian_falls_off_with_distance(self):
        sm = SourceMap(ny=100, nx=100, background_rate=0.0)
        sm.add_gaussian_source(cx=50, cy=50, peak_rate=10.0, sigma=5.0)

        # Rate should decrease with distance
        rate_at_center = sm.rates[50, 50]
        rate_at_5 = sm.rates[50, 55]  # 5 pixels away
        rate_at_15 = sm.rates[50, 65]  # 15 pixels away

        assert rate_at_center > rate_at_5 > rate_at_15

    def test_gaussian_adds_to_existing(self):
        sm = SourceMap(ny=100, nx=100, background_rate=1.0)
        sm.add_gaussian_source(cx=50, cy=50, peak_rate=10.0, sigma=5.0)

        # Center should be background + peak
        assert sm.rates[50, 50] > 10.0


class TestRingSource:
    """Tests for ring sources."""

    def test_ring_creates_annulus(self):
        sm = SourceMap(ny=100, nx=100, background_rate=0.1)
        sm.add_ring_source(cx=50, cy=50, radius=20.0, width=4.0, rate=5.0)

        # At the ring radius, should be high
        assert sm.rates[50, 70] == 5.0  # 20 pixels to the right

        # At center, should be background
        assert sm.rates[50, 50] == 0.1

        # Far outside, should be background
        assert sm.rates[50, 95] == 0.1


class TestDiskSource:
    """Tests for disk sources."""

    def test_uniform_disk(self):
        sm = SourceMap(ny=100, nx=100, background_rate=0.1)
        sm.add_uniform_disk(cx=50, cy=50, radius=10.0, rate=5.0)

        # Inside disk
        assert sm.rates[50, 50] == 5.0
        assert sm.rates[50, 55] == 5.0  # 5 pixels from center

        # Outside disk
        assert sm.rates[50, 70] == 0.1


class TestSourceMapOperations:
    """Tests for SourceMap arithmetic operations."""

    def test_addition(self):
        sm1 = SourceMap(ny=50, nx=50, background_rate=1.0)
        sm2 = SourceMap(ny=50, nx=50, background_rate=2.0)

        sm3 = sm1 + sm2

        assert np.all(sm3.rates == 3.0)
        # Originals unchanged
        assert np.all(sm1.rates == 1.0)
        assert np.all(sm2.rates == 2.0)

    def test_inplace_addition(self):
        sm1 = SourceMap(ny=50, nx=50, background_rate=1.0)
        sm2 = SourceMap(ny=50, nx=50, background_rate=2.0)

        sm1 += sm2

        assert np.all(sm1.rates == 3.0)

    def test_addition_dimension_mismatch_raises(self):
        sm1 = SourceMap(ny=50, nx=50, background_rate=1.0)
        sm2 = SourceMap(ny=60, nx=50, background_rate=2.0)

        with pytest.raises(ValueError):
            _ = sm1 + sm2

    def test_copy(self):
        sm1 = SourceMap(ny=50, nx=50, background_rate=1.0)
        sm1.add_point_source(25, 25, 10.0)

        sm2 = sm1.copy()

        # Should be equal
        assert np.array_equal(sm1.rates, sm2.rates)

        # But independent
        sm2.add_point_source(0, 0, 20.0)
        assert sm1.rates[0, 0] == 1.0
        assert sm2.rates[0, 0] == 20.0


class TestSourceMapQueries:
    """Tests for SourceMap query methods."""

    def test_total_activity(self):
        sm = SourceMap(ny=10, nx=10, background_rate=1.0)
        assert sm.total_activity() == 100.0

        sm.add_point_source(5, 5, 11.0)  # Replaces 1.0 with 11.0
        assert sm.total_activity() == 110.0

    def test_max_rate(self):
        sm = SourceMap(ny=50, nx=50, background_rate=0.1)
        assert sm.max_rate() == 0.1

        sm.add_point_source(25, 25, 10.0)
        assert sm.max_rate() == 10.0

    def test_get_rate(self):
        sm = SourceMap(ny=50, nx=50, background_rate=0.1)
        sm.add_point_source(25, 25, 10.0)

        assert sm.get_rate(25, 25) == 10.0
        assert sm.get_rate(0, 0) == 0.1
